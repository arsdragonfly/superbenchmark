# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contract tests for the short, non-destructive MI455X smoke preset."""

from pathlib import Path
import unittest

import yaml


class Rocm102SmokeConfigTest(unittest.TestCase):
    """Keep the full approved case set while minimizing configurable work."""

    @classmethod
    def setUpClass(cls):
        """Load the real YAML, including its shared parameter anchors."""
        path = Path(__file__).resolve().parents[1] / 'superbench/config/amd_mi455x_rocm102_smoke.yaml'
        cls.config = yaml.safe_load(path.read_text())['superbench']
        cls.benchmarks = cls.config['benchmarks']

    def test_exact_case_set(self):
        """Do not silently drop hard cases or add unsafe benchmark families."""
        models = [
            'lstm', 'gpt2-small', 'gpt2-large', 'bert-base', 'bert-large',
            'resnet50', 'resnet101', 'resnet152', 'densenet169', 'densenet201',
            'vgg11', 'vgg13', 'vgg16', 'vgg19',
        ]
        expected = {
            'kernel-launch', 'gpu-copy-bw:correctness', 'gpu-copy-bw:performance',
            'matmul', 'sharding-matmul', 'computation-communication-overlap',
            'dist-inference:eager', 'dist-inference:graph',
        }
        expected.update(f'gemm-flops:{p}' for p in ('fp32_xdlops', 'fp16_xdlops', 'bf16_xdlops', 'fp64', 'int8_xdlops'))
        expected.update(f'hipblaslt-gemm:{p}' for p in ('fp32', 'fp16', 'bf16', 'fp8'))
        expected.update(f'rccl-bw:{op}' for op in ('allreduce', 'allgather', 'reducescatter', 'broadcast', 'alltoall'))
        expected.update(f'model-benchmarks:{model}-{p}' for model in models for p in ('float32', 'float16'))
        expected.update(f'model-benchmarks:{model}-fp8_hybrid' for model in models[1:5])
        self.assertEqual(len(expected), 54)
        self.assertEqual(set(self.benchmarks), expected)

    def test_minimum_iterations(self):
        """Use one measured step and zero warmup, except graph initialization."""
        for name, benchmark in self.benchmarks.items():
            with self.subTest(benchmark=name):
                parameters = benchmark['parameters']
                self.assertTrue(benchmark['enable'])
                self.assertEqual(parameters['run_count'], 1)
                self.assertFalse(parameters.get('tolerant_fail', False))
                if name.startswith('gpu-copy-bw:'):
                    self.assertEqual(parameters['num_loops'], 1)
                    self.assertEqual(parameters['num_warm_up'], 0)
                    self.assertTrue(parameters['check_data'])
                elif name.startswith('rccl-bw:'):
                    self.assertEqual(parameters['iters'], 1)
                    self.assertEqual(parameters['warmup_iters'], 0)
                    self.assertEqual(parameters['check'], 1)
                else:
                    self.assertEqual(parameters['num_steps'], 1)
                    self.assertEqual(parameters['num_warmup'], int(name == 'dist-inference:graph'))

    def test_model_limits_and_data(self):
        """A one-step run must not wait for a 100-step stop check or an empty loader."""
        for name, benchmark in self.benchmarks.items():
            if not name.startswith('model-benchmarks:'):
                continue
            with self.subTest(benchmark=name):
                parameters = benchmark['parameters']
                self.assertEqual(parameters['duration'], 0)
                self.assertEqual(parameters['check_frequency'], 1)
                self.assertEqual(parameters['log_n_steps'], 1)
                self.assertGreaterEqual(parameters['sample_count'], parameters['batch_size'])
                self.assertEqual(parameters['num_workers'], 0)
                self.assertEqual(parameters['model_action'], ['train'])
                if 'image_size' in parameters:
                    self.assertEqual(parameters['image_size'], 32)
                    self.assertEqual(parameters['batch_size'], 2)
                elif 'num_hidden_layers' in parameters:
                    self.assertEqual(parameters['num_hidden_layers'], 1)
                    self.assertEqual(parameters['seq_len'], 16)

    def test_communication_and_graph_coverage(self):
        """Keep genuine sharding, overlap, graph, and four-rank coverage."""
        self.assertFalse(self.config['monitor']['enable'])
        self.assertEqual(
            self.benchmarks['sharding-matmul']['parameters']['mode'],
            ['allreduce', 'allgather', 'nosharding'],
        )
        self.assertEqual(
            self.benchmarks['computation-communication-overlap']['parameters']['kernel'], ['mul', 'matmul']
        )
        self.assertTrue(self.benchmarks['dist-inference:graph']['parameters']['use_cuda_graph'])
        for benchmark in self.benchmarks.values():
            mode = benchmark['modes'][0]
            if mode['name'] == 'torch.distributed':
                self.assertEqual(mode['proc_num'], 4)
                self.assertEqual(mode['node_num'], 1)
            else:
                self.assertFalse(mode['parallel'])

    def test_bert_fp8_pooler_has_minimum_aligned_batch(self):
        """Sequence length cannot supply the rows removed by BERT pooling."""
        for model in ('bert-base', 'bert-large'):
            parameters = self.benchmarks[f'model-benchmarks:{model}-fp8_hybrid']['parameters']
            self.assertEqual(parameters['batch_size'], 8)
            self.assertEqual(parameters['sample_count'], 8)
            self.assertEqual(parameters['num_steps'], 1)
            self.assertEqual(self.benchmarks[f'model-benchmarks:{model}-float16']['parameters']['batch_size'], 1)


if __name__ == '__main__':
    unittest.main()
