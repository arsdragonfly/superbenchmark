# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contracts for the self-contained, CPU-buildable vLLM/SGLang recipes."""

import argparse
import importlib.util
from pathlib import Path
import re
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / 'dockerfile/rocm10.2-serving'


def load(name):
    """Load standalone build helpers without installing GPU dependencies."""
    spec = importlib.util.spec_from_file_location(name, RECIPES / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load('build')
cache = load('validate_translation_cache')
with patch.dict('sys.modules', {'validate_translation_cache': cache}):
    validator = load('validate')


class ServingBuildTest(unittest.TestCase):
    """Ensure the default plan produces every referenced local image."""

    def options(self, **overrides):
        """Supply the documented defaults without invoking a container build."""
        return argparse.Namespace(
            **{'app': 'all', 'jobs': 24, 'llvm_jobs': 32,
               'triton_llvm_image': None, 'flydsl_wheel_image': None, **overrides})

    def test_complete_default_graph(self):
        """The compilers, Python SDK, common stack and both applications are built."""
        commands = list(builder.commands(self.options()))
        builds = [command for command in commands if command[1] == 'buildx']
        tags = [command[command.index('-t') + 1] for command in builds]
        self.assertEqual(tags[:5], [builder.LLVM, builder.BASE, builder.COMMON, builder.FIXED, builder.FLYDSL])
        self.assertEqual(len(builds), 9)
        self.assertEqual(len(commands), 11)
        self.assertIn('llvm-source', commands[0])
        produced = set()
        for command in builds:
            self.assertTrue((ROOT / command[command.index('-f') + 1]).is_file())
            self.assertEqual(command[-1], '.')
            for argument in command:
                if argument.startswith(('BASE_IMAGE=', 'APP_IMAGE=', 'LLVM_IMAGE=', 'FLYDSL_IMAGE=')):
                    self.assertIn(argument.split('=', 1)[1], produced)
            produced.add(command[command.index('-t') + 1])
        for command in (commands[7], commands[-1]):
            self.assertIn('--network=none', command)
            self.assertIn('/opt/base-provenance/validate.py', command)

    def test_optional_compilers_are_not_required(self):
        """Explicit compatible caches can save compiler time but never skip final checks."""
        commands = list(builder.commands(self.options(
            app='sglang', triton_llvm_image='local-llvm:test', flydsl_wheel_image='local-flydsl:test')))
        self.assertEqual(len(commands), 6)
        self.assertIn('LLVM_IMAGE=local-llvm:test', commands[2])
        self.assertIn('FLYDSL_IMAGE=local-flydsl:test', commands[4])
        self.assertEqual(commands[-1][-1], 'sglang')

    def test_failure_stops_build(self):
        """Never continue to application builds after a failed prerequisite."""
        with patch.object(builder.subprocess, 'run', side_effect=RuntimeError('failed')) as run:
            with patch('sys.argv', ['build.py']), self.assertRaises(RuntimeError):
                builder.main()
            self.assertEqual(run.call_count, 1)

    def test_positive_jobs(self):
        """Reject zero and negative worker counts."""
        self.assertEqual(builder.positive('8'), 8)
        for value in ('0', '-1'):
            with self.assertRaises(argparse.ArgumentTypeError):
                builder.positive(value)

    def test_every_copy_source_exists(self):
        """No build input lives only in a previous session or remote checkout."""
        for path in RECIPES.glob('*.dockerfile'):
            text = path.read_text().replace('\\\n', ' ')
            for line in text.splitlines():
                if line.startswith('COPY ') and '--from=' not in line:
                    for source in line.split()[1:-1]:
                        self.assertTrue((ROOT / source).is_file(), (path.name, source))
            self.assertNotIn('/home/hpcuser', text)
            self.assertNotIn('.copilot', text)
            self.assertNotIn('translator-fixtures', text)
        self.assertFalse(list(RECIPES.rglob('*.elf')))

    def test_pins_and_compiler_guards(self):
        """Preserve October 15 serving pins and both compiler repairs."""
        triton = (RECIPES / 'triton.dockerfile').read_text()
        flydsl = (RECIPES / 'flydsl.dockerfile').read_text()
        qualify = (RECIPES / 'qualify.dockerfile').read_text()
        self.assertIn('66b3f9ff575eb8db6adec658bbf0badd9a8d2e25', triton)
        self.assertIn('1f3511677662afc58bcf141dec812ccf05334fd8', flydsl)
        self.assertIn('e2a39f504fee836e4def9581bed817ecc327b9dc', flydsl)
        self.assertIn('5653cbd792aa071647e307b39e932d8fbd2e39bf9d7aa9e4115b3ebce508bf07', flydsl)
        self.assertIn('pin_fixed_wheel.py flydsl', qualify)
        self.assertIn(
            'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AMD_AITER=0.1.26.dev97+g4eb7dcac7',
            (RECIPES / 'common.dockerfile').read_text())
        for role in ('vllm', 'sglang'):
            recipe = (RECIPES / f'{role}.dockerfile').read_text()
            self.assertIn('--no-build-isolation --no-deps', recipe)
            self.assertIn('fixed-triton-installed.sha256', recipe)
            self.assertIn('constraints-app.txt', recipe)
        self.assertIn('NOT_TESTED', (RECIPES / 'validate.py').read_text())

    def test_dependency_manifests_are_portable_version_pins(self):
        """Do not publish direct local wheel paths or accidentally unpinned dependencies."""
        for path in RECIPES.glob('constraints-*.txt'):
            lines = path.read_text().splitlines()
            self.assertGreater(len(lines), 50)
            for line in lines:
                self.assertRegex(line, r'^[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+$')
            self.assertEqual(len(lines), len(set(re.split('==', line)[0].lower() for line in lines)))


class TranslationCacheTest(unittest.TestCase):
    """Exercise the actual descriptor-bit check with a source-generated ELF fixture."""

    def elf(self, flags=0x400):
        """Construct one ELF64 AMDGPU kernel descriptor and its symbol table."""
        data = bytearray(512)
        identity = b'\x7fELF\x02\x01' + bytes(10)
        struct.pack_into('<16sHHIQQQIHHHHHH', data, 0,
                         identity, 3, 224, 1, 0, 0, 64, 0x49, 64, 0, 0, 64, 4, 0)
        struct.pack_into('<IIQQQQIIQQ', data, 128, 0, 1, 0, 0x1000, 384, 64, 0, 0, 8, 0)
        struct.pack_into('<IIQQQQIIQQ', data, 192, 0, 3, 0, 0, 320, 10, 0, 0, 1, 0)
        struct.pack_into('<IIQQQQIIQQ', data, 256, 0, 2, 0, 0, 464, 24, 2, 0, 8, 24)
        data[320:330] = b'\0probe.kd\0'
        struct.pack_into('<IBBHQQ', data, 464, 1, 0, 0, 1, 0x1000, 64)
        struct.pack_into('<H', data, 440, flags)
        return data

    def test_valid_and_cleared_wave32(self):
        """An A0 target label alone does not prove the descriptor is correct."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'kernel.obj'
            path.write_bytes(self.elf())
            self.assertEqual(cache.check_cache(root), {'objects': 1, 'descriptors': 1, 'result': 'PASS'})
            path.write_bytes(self.elf(0))
            with self.assertRaisesRegex(ValueError, 'cleared'):
                cache.check_cache(root)

    def test_empty_or_descriptorless_cache_fails(self):
        """Neither an empty scan nor a descriptorless ELF qualifies a translator."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                cache.check_cache(root)
            data = self.elf()
            data[320:330] = b'\0ordinary\0'
            (root / 'kernel.obj').write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'No kernel descriptors'):
                cache.check_cache(root)

    def test_truncated_object_fails(self):
        """Malformed ELF tables cannot silently pass."""
        for data in (b'', self.elf()[:80], self.elf()[:480]):
            with self.assertRaises(ValueError):
                cache.kernel_flags(data)


class ServingInventoryTest(unittest.TestCase):
    """Require complete pinned inventories in final application images."""

    def test_cpu_validation_does_not_indirectly_import_aiter(self):
        """Apex imports AITER's GPU discovery when AITER is installed."""
        self.assertIn('apex', validator.cpu_imports('sdk'))
        for role in ('base', 'vllm', 'sglang'):
            self.assertNotIn('apex', validator.cpu_imports(role))
            self.assertNotIn('aiter', validator.cpu_imports(role))
        self.assertIn('vllm._C', validator.cpu_imports('vllm'))
        self.assertIn('sgl_kernel', validator.cpu_imports('sglang'))

    def test_final_inventory_cannot_silently_lose_or_add_packages(self):
        """Missing dependencies and unpinned additions fail even if pip check succeeds."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('requirements-base.lock', 'constraints-base.txt', 'constraints-app.txt'):
                (root / name).write_text('package==1.0\n')
            for installed in ([], [SimpleNamespace(metadata={'Name': 'extra'}, version='1.0')]):
                with patch.object(validator, 'ROOT', root), patch.object(
                    validator, 'distributions', return_value=installed
                ), self.assertRaisesRegex(RuntimeError, 'Package inventory changed'):
                    validator.check_packages('vllm')

    def test_dependency_version_drift_fails(self):
        """Pinned versions are verified, not merely written to constraints files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('requirements-base.lock', 'constraints-base.txt'):
                (root / name).write_text('package==1.0\n')
            installed = [SimpleNamespace(metadata={'Name': 'package'}, version='2.0')]
            with patch.object(validator, 'ROOT', root), patch.object(
                validator, 'distributions', return_value=installed
            ), self.assertRaisesRegex(RuntimeError, 'Package mismatch'):
                validator.check_packages('sdk')


if __name__ == '__main__':
    unittest.main()
