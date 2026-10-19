# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU contract tests for the isolated MI455X smoke runner."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('rocm102_smoke_runner', ROOT / 'dockerfile/rocm10.2-nightly/run_smoke.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Rocm102SmokeRunnerTest(unittest.TestCase):
    """Check process isolation, real distributed launch and strict result handling."""

    @classmethod
    def setUpClass(cls):
        """Load the real smoke profile."""
        cls.config = ROOT / 'superbench/config/amd_mi455x_rocm102_smoke.yaml'
        cls.benchmarks = yaml.safe_load(cls.config.read_text())['superbench']['benchmarks']

    def test_distributed_launch_uses_loopback_and_sufficient_shm(self):
        """Do not repeat standalone-hostname or default-64MiB shared-memory failures."""
        ranks, invocations = runner.commands(
            'sha256:example', self.config, Path('/tmp/result'), 'model-benchmarks:gpt2-small-fp8_hybrid',
            self.benchmarks['model-benchmarks:gpt2-small-fp8_hybrid'], [44, 993], 900
        )
        self.assertEqual(ranks, [0, 1, 2, 3])
        self.assertEqual(len(invocations), 1)
        command = invocations[0][2]
        self.assertIn('--shm-size=8g', command)
        self.assertIn('--master_addr=127.0.0.1', command)
        self.assertIn('--rdzv_backend=static', command)
        self.assertNotIn('--standalone', command)
        self.assertIn('HIP_VISIBLE_DEVICES=0,1,2,3', command)
        self.assertIn('/opt/superbench/dockerfile/rocm10.2-nightly/exec_rocblas.py', command)

    def test_local_gpu_processes_are_independent(self):
        """Each native local invocation sees only its assigned physical GPU."""
        ranks, invocations = runner.commands(
            'sha256:example', self.config, Path('/tmp/result'), 'hipblaslt-gemm:fp8',
            self.benchmarks['hipblaslt-gemm:fp8'], [44, 993], 900
        )
        self.assertEqual(ranks, [0, 1, 2, 3])
        self.assertEqual(len(invocations), 4)
        for rank, _, command in invocations:
            self.assertIn(f'HIP_VISIBLE_DEVICES={rank}', command)
            self.assertIn(f'PROC_RANK={rank}', command)
            self.assertIn('sb', command)
            self.assertNotIn('--cap-add', command)

    def test_gpu_copy_only_gets_nice_capability(self):
        """Do not grant unnecessary capabilities to other benchmark families."""
        _, invocations = runner.commands(
            'sha256:example', self.config, Path('/tmp/result'), 'gpu-copy-bw:correctness',
            self.benchmarks['gpu-copy-bw:correctness'], [44, 993], 900
        )
        self.assertEqual(len(invocations), 1)
        self.assertIn('SYS_NICE', invocations[0][2])

    def write_result(self, root, rank, values, raw=None):
        """Create a minimal real-shaped per-rank result."""
        path = root / f'sb/benchmarks/case/rank{rank}/results.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps([{'result': values, 'raw_data': raw or {}}]))

    def test_all_rank_results_are_required(self):
        """A zero-exit launcher cannot stand in for missing worker results."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_result(root, 0, {'return_code': [0], 'step_time': [1.0]})
            with self.assertRaises(RuntimeError):
                runner.verify_results(root, [0, 1], 'model-benchmarks:test')

    def test_main_rank_metrics_are_sufficient(self):
        """Distributed peers may legitimately report only their success code."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_result(root, 0, {'return_code': [0], 'step_time': [1.0]})
            self.write_result(root, 1, {'return_code': [0]})
            self.assertEqual(len(runner.verify_results(root, [0, 1], 'model-benchmarks:test')), 2)

    def test_nonfinite_metrics_are_rejected(self):
        """Preserve smoke functionality without accepting invalid measurements."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_result(root, 0, {'return_code': [0], 'time': [float('nan')]})
            with self.assertRaises(RuntimeError):
                runner.verify_results(root, [0], 'matmul')

    def test_rccl_correctness_is_required(self):
        """Check the numerical verification summary, not only the process status."""
        for raw, passed in (
            ({'output': ['# Out of bounds values : 0 OK']}, True),
            ({'output': ['# Out of bounds values : 1 FAILED']}, False),
            ({}, False),
        ):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_result(root, 0, {'return_code': [0], 'bw': [1.0]}, raw)
                if passed:
                    self.assertEqual(len(runner.verify_results(root, [0], 'rccl-bw:allreduce')), 1)
                else:
                    with self.assertRaises(RuntimeError):
                        runner.verify_results(root, [0], 'rccl-bw:allreduce')


if __name__ == '__main__':
    unittest.main()
