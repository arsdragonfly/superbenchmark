# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Source-level invariants for reproducible ROCm 10.2 nightly images."""

from pathlib import Path
import unittest


class Rocm102DockerfilesTest(unittest.TestCase):
    """Keep the nightly stack, A0 fixes and full overlay requirements connected."""

    @classmethod
    def setUpClass(cls):
        """Read the actual recipes and dependency lock."""
        root = Path(__file__).resolve().parents[1]
        cls.base = (root / 'dockerfile/rocm10.2-nightly-base.dockerfile').read_text()
        cls.overlay = (root / 'dockerfile/rocm10.2-nightly.dockerfile').read_text()
        cls.lock = (root / 'dockerfile/rocm10.2-nightly/requirements-base.lock').read_text()
        cls.ignore = (root / '.dockerignore').read_text()

    def test_base_is_source_reproducible(self):
        """Private compiler caches are optional, not missing build prerequisites."""
        self.assertIn('ARG SDK_IMAGE=sdk-source', self.base)
        self.assertIn('ARG TRITON_LLVM_IMAGE=llvm-source', self.base)
        self.assertIn('5f07f818b51b786b0a87b4e514882600ecba112f', self.base)
        self.assertIn('66b3f9ff575eb8db6adec658bbf0badd9a8d2e25', self.base)
        self.assertIn('8a29d2e82b93f5065c7fe5630248d3689f0c3f06', self.base)
        self.assertIn('triton_unsigned_offset.py --expect-fixed', self.base)
        self.assertIn('torch==2.13.0+rocm10.2.0a20261018', self.lock)
        self.assertIn('rocm-sdk-core==10.2.0a20261018', self.lock)

    def test_a0_assets_and_aot_cache_reach_both_loader_views(self):
        """Python's core loader and standalone devel loader need coherent assets."""
        self.assertIn('--architecture gfx1250 --gfx1250v0', self.base)
        self.assertIn('_rocm_sdk_libraries/lib/hipblaslt/library/gfx1250v0/', self.base)
        self.assertIn('_rocm_sdk_devel/lib/hipblaslt/library/gfx1250v0/', self.base)
        self.assertIn('rocjitsu-pretranslate.py', self.base)
        self.assertIn('--fail-on-skipped', self.base)
        self.assertIn('_rocm_sdk_core/share/rocjitsu/translations', self.base)

    def test_overlay_preserves_the_supplied_stack(self):
        """Keep networking tools, RVS, AITER, Megatron and the native clients."""
        for component in ('doca-ofed-userspace', 'openmpi', 'amdrocm10-rvs', 'TransferBench',
                          'ROCm/aiter.git', 'NVIDIA/Megatron-LM.git', 'rocblas-bench', 'hipblaslt-bench'):
            self.assertIn(component, self.overlay)
        self.assertIn('apt-get download "amdrocm10-rvs=${RVS_VERSION}"', self.overlay)
        self.assertIn('RVS_PACKAGE_SHA256=47b4bc75', self.overlay)
        self.assertNotIn("'URIs: https://nightly.repo.amd.com/rocm/core/packages", self.overlay)
        self.assertNotIn('[trusted=yes]', self.overlay)
        self.assertNotIn('rocm-sdk init', self.overlay)
        self.assertNotIn('ROCM_VERSION=10.1.0', self.overlay)

    def test_native_clients_prefer_the_packaged_runtime_layout(self):
        """Native RCCL must use the same library layout as qualified PyTorch."""
        runtime = (
            '/opt/venv/lib/python3.14/site-packages/_rocm_sdk_core/lib:'
            '/opt/venv/lib/python3.14/site-packages/_rocm_sdk_libraries/lib:'
        )
        self.assertIn(runtime + '${LD_LIBRARY_PATH}', self.base)
        self.assertIn(runtime + '${ROCM_PATH}/lib', self.overlay)
        self.assertIn('export LD_LIBRARY_PATH=%q:', self.overlay)

    def test_te_remains_basic_and_flydsl_remains_fixed(self):
        """Do not reintroduce unqualified fused attention or an old independent LLVM."""
        for setting in ('NVTE_FLASH_ATTN=0', 'NVTE_FUSED_ATTN=0', 'NVTE_FUSED_ATTN_CK=0',
                        'NVTE_FUSED_ATTN_AOTRITON=0', 'NVTE_ROCM_UNFUSED_FP32_BGRAD=1'):
            self.assertIn(setting, self.overlay)
        self.assertNotIn('NVTE_FUSED_ATTN_AOTRITON=1', self.overlay)
        self.assertIn('rocm_fp32_bgrad.patch', self.overlay)
        self.assertIn('FLYDSL_PACKAGE_VERSION_OVERRIDE=0.3.4.1', self.overlay)
        self.assertIn('e2a39f504fee836e4def9581bed817ecc327b9dc', self.overlay)
        self.assertIn('/opt/fixed-flydsl-wheels', self.overlay)

    def test_local_settings_are_not_baked_into_images(self):
        """Exclude unrelated editor settings and common credential files."""
        for pattern in ('.vscode/', '.env', '.env.*'):
            self.assertIn(pattern, self.ignore)
        self.assertNotIn('!/.git', self.ignore)


if __name__ == '__main__':
    unittest.main()
