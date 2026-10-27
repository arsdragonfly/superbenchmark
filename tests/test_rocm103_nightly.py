# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contracts for the four-image nightly refresh and vendor A0 asset validation."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / 'dockerfile/rocm10.3-nightly'


def load(name, path):
    """Import build-only helpers without requiring GPU packages."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load('refresh_builder', RECIPES / 'build.py')
cache = load('validate_translation_cache', ROOT / 'dockerfile/rocm10.2-serving/validate_translation_cache.py')
with patch.dict('sys.modules', {'validate_translation_cache': cache}):
    validator = load('refresh_validator', RECIPES / 'validate.py')


class NightlyRefreshTest(unittest.TestCase):
    """Keep source identities, build dependencies and qualifications consistent."""

    def args(self, images=None):
        """Use the documented default plan with no private build prerequisites."""
        return argparse.Namespace(images=images or list(builder.OUTPUTS), jobs=24, source_revision='a' * 40,
                                  triton_llvm_image=None, flydsl312_image=None, flydsl314_image=None)

    def test_full_build_plan(self):
        """All local FROM image dependencies must be produced earlier in the plan."""
        produced = set()
        for command in builder.plan(self.args()):
            if command[1] != 'buildx':
                self.assertIn(command[4], produced)
                continue
            self.assertTrue((ROOT / command[command.index('-f') + 1]).is_file())
            for item in command:
                if item.startswith(('BASE_IMAGE=', 'SDK_IMAGE=', 'APP_IMAGE=', 'FLYDSL_IMAGE=',
                                    'FLYDSL_WHEEL_IMAGE=', 'LLVM_IMAGE=')):
                    self.assertIn(item.split('=', 1)[1], produced)
            produced.add(command[command.index('-t') + 1])
        self.assertTrue(set(builder.OUTPUTS.values()) <= produced)

    def test_base_only_does_not_build_frameworks(self):
        """The independently reusable PyTorch base has no serving/native-stack dependencies."""
        plan = list(builder.plan(self.args(['base'])))
        self.assertEqual(len(plan), 3)
        self.assertFalse(any('common.dockerfile' in ' '.join(command) for command in plan))

    def test_serving_only_uses_python312(self):
        """Serving profiles keep the ABI required by their package set."""
        plan = list(builder.plan(self.args(['vllm'])))
        self.assertFalse(any('py314' in ' '.join(command) for command in plan))
        self.assertTrue(any('PYTHON_VERSION=3.12' in command for command in plan))
        self.assertTrue(any('TORCH_DEVICE_EXTRAS=device-gfx1250' in command for command in plan))

    def test_current_source_and_native_archive_pins(self):
        """Neither a calendar tag nor an SDK update alone qualifies the compiler stack."""
        pins = json.loads((RECIPES / 'expected-manifest.json').read_text())
        self.assertEqual(pins['rocm_package_version'], '10.3.0a20261026')
        self.assertEqual(pins['submodules']['rocm-systems'], '39fd0ea559a5970e42afd87c30f275744e00b8d1')
        base = (RECIPES / 'base.dockerfile').read_text()
        self.assertIn('a3c5afb73ecd9e70046a983a8dd18fd676a30438', base)
        self.assertIn('LLVM_SYSPATH=/opt/triton-llvm', base)
        self.assertNotIn('select_a0_logic', base)
        self.assertIn('vendor-gfx1250v0', base)
        self.assertIn('bf13fff67ffcc231bc004400ddb8740437e1d4037cc20aaf379ca6a59890d206',
                      (RECIPES / 'build.py').read_text())
        for role in ('base', 'vllm', 'sglang'):
            constraints = (RECIPES / f'constraints-{role}.txt').read_text()
            self.assertNotIn('rocm10.2', constraints)
            self.assertIn('triton==3.8.0+gita3c5afb7.rocm10.3.0a20261026', constraints)

    def test_copy_inputs_exist(self):
        """No source input lives only on a previous build VM or chat session."""
        for recipe in RECIPES.glob('*.dockerfile'):
            text = recipe.read_text().replace('\\\n', ' ')
            for line in text.splitlines():
                if line.startswith('COPY ') and '--from=' not in line:
                    for source in line.split()[1:-1]:
                        if not source.startswith('--'):
                            self.assertTrue((ROOT / source).is_file(), (recipe.name, source))

    def test_mitigations_preserved(self):
        """No refresh silently re-enables unqualified A0 compiler paths."""
        sdk = (RECIPES / 'sdk.dockerfile').read_text()
        for setting in ('TRITON_HIP_USE_ASYNC_COPY=0', 'TRITON_HIP_USE_EXPERT_SCHEDULING=0'):
            self.assertIn(setting, sdk)
        self.assertIn('DEVICE_EXTRAS=devel,libraries,device-all', sdk)
        self.assertIn('/usr/local/bin/sb-serve', (RECIPES / 'qualify.dockerfile').read_text())
        self.assertIn(builder.AITER, (RECIPES / 'common.dockerfile').read_text())

    def test_vendor_inventory_requires_masters_in_both_views(self):
        """Code objects without an A0 selection table cannot pass packaging validation."""
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'Missing vendor A0 master'):
                validator.inventory_assets(site)
            for library in ('hipblaslt', 'hipsparselt'):
                for view in ('_rocm_sdk_libraries', '_rocm_sdk_devel'):
                    path = site / view / f'lib/{library}/library/gfx1250v0'
                    path.mkdir(parents=True)
                    (path / 'TensileLibrary_lazy_gfx1250.dat.zlib').write_bytes(b'index')
                    (path / 'TensileLiteLibrary_lazy_gfx1250_Mapping.dat.zlib').write_bytes(b'mapping')
                    (path / 'kernel.co').write_bytes(b'code')
            self.assertEqual(len(validator.inventory_assets(site)), 6)
            (site / '_rocm_sdk_devel/lib/hipblaslt/library/gfx1250v0/kernel.co').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'Inconsistent vendor A0 asset'):
                validator.inventory_assets(site)

    def test_no_secret_inputs(self):
        """Keep credentials and host-specific paths out of public build definitions."""
        for recipe in RECIPES.glob('*.dockerfile'):
            self.assertNotRegex(recipe.read_text(), r'(?i)(--password|accessToken|/home/hpcuser|\.copilot/)')
        self.assertRegex(builder.SDK_VERSION, r'^10\.3\.0a\d{8}$')


if __name__ == '__main__':
    unittest.main()
