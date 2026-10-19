# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU tests for the Transformer Engine module conversion used by FP8 models."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from transformers.pytorch_utils import Conv1D

from superbench.benchmarks.model_benchmarks.pytorch_gpt2 import PytorchGPT2


class FakeTeLinear(torch.nn.Linear):
    """Use ordinary CPU linear algebra to test TE conversion semantics."""

    def __init__(self, in_features, out_features, bias, params_dtype):
        """Accept the arguments passed to Transformer Engine Linear."""
        super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)


class TransformerEngineConversionTest(unittest.TestCase):
    """Preserve GPT2 Conv1D orientation and independently process siblings."""

    def test_gpt2_conv1d_preserves_forward_and_weights(self):
        """GPT2 projections must become real TE linears, not remain FP16 Conv1D."""
        model = torch.nn.Sequential(Conv1D(32, 16))
        inputs = torch.randn(2, 3, 16)
        expected = model(inputs).detach()
        weight = model[0].weight.detach().clone()
        bias = model[0].bias.detach().clone()
        benchmark = PytorchGPT2('conversion')
        with mock.patch('superbench.benchmarks.model_benchmarks.pytorch_base.te',
                        SimpleNamespace(Linear=FakeTeLinear)):
            benchmark._to_te_model(model)
        self.assertIsInstance(model[0], FakeTeLinear)
        torch.testing.assert_close(model[0].weight, weight.t(), rtol=0, atol=0)
        torch.testing.assert_close(model[0].bias, bias, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), expected)

    def test_unaligned_layer_does_not_skip_following_siblings(self):
        """An unsupported classifier must not prevent conversion of later layers."""
        model = torch.nn.ModuleDict({
            'unaligned': torch.nn.Linear(16, 7),
            'aligned': torch.nn.Linear(16, 32),
        })
        original = model['unaligned']
        expected = model['aligned'].weight.detach().clone()
        benchmark = PytorchGPT2('conversion')
        with mock.patch('superbench.benchmarks.model_benchmarks.pytorch_base.te',
                        SimpleNamespace(Linear=FakeTeLinear)):
            benchmark._to_te_model(model)
        self.assertIs(model['unaligned'], original)
        self.assertIsInstance(model['aligned'], FakeTeLinear)
        torch.testing.assert_close(model['aligned'].weight, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
