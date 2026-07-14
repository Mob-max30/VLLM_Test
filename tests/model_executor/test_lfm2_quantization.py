# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn


@pytest.mark.parametrize(
    ("module_name", "layer_class_name", "mlp_class_name"),
    [
        (
            "vllm.model_executor.models.lfm2",
            "Lfm2ShortConvDecoderLayer",
            "Lfm2MLP",
        ),
        (
            "vllm.model_executor.models.lfm2_moe",
            "Lfm2MoeShortConvDecoderLayer",
            "Lfm2MoeMlp",
        ),
    ],
)
def test_lfm2_short_conv_layer_receives_quant_config(
    module_name: str,
    layer_class_name: str,
    mlp_class_name: str,
):
    module = importlib.import_module(module_name)
    layer_class = getattr(module, layer_class_name)
    quant_config = Mock()
    config = Mock(num_dense_layers=1)

    with (
        patch.object(module, "ShortConv") as short_conv,
        patch.object(module, mlp_class_name),
        patch.object(module, "RMSNorm"),
    ):
        layer_class(config=config, layer_idx=0, quant_config=quant_config)

    assert short_conv.call_args.kwargs["quant_config"] is quant_config


def test_short_conv_projection_linears_receive_quant_config():
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba import short_conv as short_conv_module

    quant_config = Mock()
    config = SimpleNamespace(conv_L_cache=3, conv_bias=False)

    def make_linear(*args, **kwargs):
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.empty(1, 1))
        layer.bias = None
        return layer

    with (
        patch.object(
            short_conv_module,
            "ColumnParallelLinear",
            side_effect=make_linear,
        ) as column_linear,
        patch.object(
            short_conv_module,
            "MergedColumnParallelLinear",
            side_effect=make_linear,
        ) as merged_linear,
        patch.object(
            short_conv_module,
            "RowParallelLinear",
            side_effect=make_linear,
        ) as row_linear,
        set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig())),
    ):
        short_conv_module.ShortConv(
            config=config,
            dim=64,
            layer_idx=0,
            quant_config=quant_config,
            prefix="model.layers.0.short_conv",
        )

    assert "quant_config" not in column_linear.call_args.kwargs
    assert merged_linear.call_args.kwargs["quant_config"] is quant_config
    assert row_linear.call_args.kwargs["quant_config"] is quant_config
