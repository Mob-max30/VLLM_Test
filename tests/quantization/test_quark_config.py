# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused unit coverage for Quark configuration parsing and dispatch."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.quark.quark import (
    QuarkConfig,
    QuarkKVCacheMethod,
)


class DummyLayer:
    pass


OCP_MX_WEIGHT = {
    "dtype": "fp4",
    "qscheme": "per_group",
    "group_size": 32,
    "scale_format": "e8m0",
    "is_dynamic": False,
}


def make_quark_config(**overrides) -> QuarkConfig:
    raw_config = {
        "global_quant_config": {"name": "global"},
        "layer_quant_config": {},
        "layer_type_quant_config": {},
        "exclude": [],
    }
    raw_config.update(overrides)
    return QuarkConfig(raw_config)


@pytest.mark.parametrize("dtype", ["fp4", "fp6_e3m2", "fp6_e2m3"])
def test_ocp_mx_weight_scheme_detection_accepts_supported_dtypes(dtype: str):
    config = make_quark_config()
    weight = {**OCP_MX_WEIGHT, "dtype": dtype}

    assert config._is_w_ocp_mx_a_x(weight, input_quant=None)


@pytest.mark.parametrize(
    "weight",
    [
        pytest.param(None, id="missing"),
        pytest.param([OCP_MX_WEIGHT], id="list"),
        pytest.param({**OCP_MX_WEIGHT, "qscheme": "per_tensor"}, id="qscheme"),
        pytest.param({**OCP_MX_WEIGHT, "group_size": 16}, id="group-size"),
        pytest.param({**OCP_MX_WEIGHT, "scale_format": "fp8"}, id="scale-format"),
        pytest.param({**OCP_MX_WEIGHT, "dtype": "int4"}, id="dtype"),
    ],
)
def test_ocp_mx_weight_scheme_detection_rejects_invalid_configs(weight):
    config = make_quark_config()

    assert not config._is_w_ocp_mx_a_x(weight, input_quant=None)


@pytest.mark.parametrize("field", ["output_tensors", "bias"])
def test_scheme_rejects_quantized_output_or_bias(field: str):
    config = make_quark_config()
    layer_config = {
        "weight": OCP_MX_WEIGHT,
        "input_tensors": None,
        field: {"dtype": "int8"},
    }

    with pytest.raises(NotImplementedError, match="output_tensors and bias"):
        config._get_scheme_from_config(layer_config)


def test_scheme_rejects_unknown_quantization():
    config = make_quark_config()
    layer_config = {
        "weight": {"dtype": "int4", "qscheme": "per_group", "group_size": 64},
        "input_tensors": None,
    }

    with pytest.raises(NotImplementedError, match="No quark compatible scheme"):
        config._get_scheme_from_config(layer_config)


def test_find_matched_config_precedence():
    layer = DummyLayer()
    exact = {"name": "exact"}
    wildcard = {"name": "wildcard"}
    layer_type = {"name": "layer-type"}
    global_config = {"name": "global"}
    config = make_quark_config(
        global_quant_config=global_config,
        layer_quant_config={
            "model.layers.0.self_attn.q_proj": exact,
            "*.mlp.down_proj": wildcard,
        },
        layer_type_quant_config={type(layer).__name__: layer_type},
    )

    assert (
        config._find_matched_config("model.layers.0.self_attn.q_proj", layer) is exact
    )
    assert (
        config._find_matched_config("model.layers.4.mlp.down_proj", layer) is wildcard
    )
    assert (
        config._find_matched_config("model.layers.4.mlp.gate_proj", layer) is layer_type
    )

    config.quant_config["layer_type_quant_config"] = {}
    assert (
        config._find_matched_config("model.layers.4.mlp.gate_proj", layer)
        is global_config
    )


def test_fused_config_accepts_identical_shard_configs():
    shared_config = {"weight": OCP_MX_WEIGHT, "input_tensors": None}
    config = make_quark_config(
        layer_quant_config={
            f"model.layers.0.self_attn.{name}": deepcopy(shared_config)
            for name in ("q_proj", "k_proj", "v_proj")
        }
    )
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    matched = config._find_matched_config(
        "model.layers.0.self_attn.qkv_proj", DummyLayer()
    )

    assert matched == shared_config


def test_fused_config_rejects_different_shard_configs():
    layer_configs = {
        f"model.layers.0.self_attn.{name}": {
            "weight": OCP_MX_WEIGHT,
            "input_tensors": None,
        }
        for name in ("q_proj", "k_proj", "v_proj")
    }
    layer_configs["model.layers.0.self_attn.v_proj"] = {
        "weight": {**OCP_MX_WEIGHT, "dtype": "fp6_e3m2"},
        "input_tensors": None,
    }
    config = make_quark_config(layer_quant_config=layer_configs)
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    with pytest.raises(ValueError, match="different quantization configuration"):
        config._find_matched_config("model.layers.0.self_attn.qkv_proj", DummyLayer())


def test_from_config_extracts_matching_kv_cache_config():
    kv_config = {"dtype": "fp8_e4m3", "qscheme": "per_tensor"}
    raw_config = {
        "export": {
            "kv_cache_group": ["*k_proj", "*v_proj"],
            "pack_method": "reorder",
        },
        "global_quant_config": {},
        "layer_type_quant_config": {},
        "layer_quant_config": {
            "*q_proj": {"output_tensors": deepcopy(kv_config)},
            "*k_proj": {"output_tensors": deepcopy(kv_config)},
            "*v_proj": {"output_tensors": deepcopy(kv_config)},
        },
        "exclude": [],
    }

    config = QuarkConfig.from_config(raw_config)

    assert config.kv_cache_config == kv_config
    assert config.kv_cache_group == ["*k_proj", "*v_proj"]
    assert all(
        layer_config["output_tensors"] is None
        for layer_config in raw_config["layer_quant_config"].values()
    )


@pytest.mark.parametrize(
    ("kv_config", "error"),
    [
        (
            {"dtype": "int8", "qscheme": "per_tensor"},
            "dtype=fp8_e4m3",
        ),
        (
            {"dtype": "fp8_e4m3", "qscheme": "per_channel"},
            "per-tensor scaling factor",
        ),
    ],
)
def test_kv_cache_validation_rejects_unsupported_config(kv_config, error: str):
    with pytest.raises(NotImplementedError, match=error):
        QuarkKVCacheMethod.validate_kv_cache_config(kv_config)


@pytest.mark.parametrize(
    "kv_config",
    [None, {"dtype": "fp8_e4m3", "qscheme": "per_tensor"}],
)
def test_kv_cache_validation_accepts_supported_config(kv_config):
    QuarkKVCacheMethod.validate_kv_cache_config(kv_config)


def test_cache_scale_mapper_maps_weight_stream_names():
    mapper = QuarkConfig.get_cache_scale_mapper()
    tensors = [torch.tensor(1.0), torch.tensor(2.0)]

    mapped = list(
        mapper.apply(
            [
                ("model.layers.0.self_attn.k_proj.output_scale", tensors[0]),
                ("model.layers.0.self_attn.prob_output_scale", tensors[1]),
            ]
        )
    )

    assert [name for name, _ in mapped] == [
        "model.layers.0.self_attn.attn.k_scale",
        "model.layers.0.self_attn.attn.prob_scale",
    ]
    assert [tensor for _, tensor in mapped] == tensors


@pytest.mark.parametrize(
    ("model_type", "quant_dtype", "expected"),
    [
        ("deepseek_v3", "fp4", True),
        ("deepseek_v32", "fp4", True),
        ("deepseek_v3", "fp6_e3m2", False),
        ("qwen2_moe", "fp4", False),
    ],
)
def test_maybe_update_config_enables_only_deepseek_fp4(
    model_type: str, quant_dtype: str, expected: bool
):
    config = make_quark_config()
    hf_config = SimpleNamespace(
        model_type=model_type,
        quantization_config={"global_quant_config": {"weight": {"dtype": quant_dtype}}},
    )

    config.maybe_update_config("unused", hf_config=hf_config)

    assert config.dynamic_mxfp4_quant is expected


def test_maybe_update_config_without_hf_config_is_a_noop():
    config = make_quark_config()

    config.maybe_update_config("unused", hf_config=None)

    assert config.dynamic_mxfp4_quant is False
