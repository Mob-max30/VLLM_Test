# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused unit coverage for Quark configuration parsing and dispatch."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization.quark import quark_moe as quark_moe_module
from vllm.model_executor.layers.quantization.quark.quark import (
    QuarkConfig,
    QuarkKVCacheMethod,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkMoEMethod,
    _enforce_quark_w4a4_rounding_contract,
)
from vllm.model_executor.layers.quantization.quark.schemes import (
    QuarkNVFP4,
    QuarkOCP_MX,
    QuarkW4A8_MXFP4_FP8,
    QuarkW8A8Fp8,
    QuarkW8A8Int8,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_Scheme,
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
OCP_MX_DYNAMIC_INPUT = {**OCP_MX_WEIGHT, "is_dynamic": True}

FP8_PER_TENSOR_WEIGHT = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_tensor",
    "is_dynamic": False,
}
FP8_DYNAMIC_PER_TENSOR_INPUT = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_tensor",
    "is_dynamic": True,
}
INT8_STATIC_PER_TENSOR = {
    "dtype": "int8",
    "qscheme": "per_tensor",
    "is_dynamic": False,
    "symmetric": True,
}
MXFP4_STATIC_FP8_INPUT = {
    "dtype": "fp8_e4m3",
    "qscheme": "per_tensor",
    "is_dynamic": False,
    "symmetric": True,
}
INT8_DYNAMIC_PER_TOKEN_INPUT = {
    "dtype": "int8",
    "qscheme": "per_channel",
    "is_dynamic": True,
    "symmetric": True,
}
NVFP4_WEIGHT = [
    {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 16,
        "is_dynamic": False,
    },
    {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
    },
]
NVFP4_INPUT = [
    {
        "dtype": "fp4",
        "qscheme": "per_group",
        "group_size": 16,
        "is_dynamic": True,
    },
    {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
    },
]
FP8_W4A8_WEIGHT = [
    {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
    },
    {
        "dtype": "int4",
        "qscheme": "per_channel",
        "is_dynamic": False,
        "symmetric": True,
        "ch_axis": 0,
    },
]


def make_quark_config(**overrides) -> QuarkConfig:
    raw_config = {
        "global_quant_config": {"name": "global"},
        "layer_quant_config": {},
        "layer_type_quant_config": {},
        "exclude": [],
    }
    raw_config.update(overrides)
    return QuarkConfig(raw_config)


@pytest.mark.parametrize(
    ("weight", "input_quant", "expected"),
    [
        pytest.param(
            FP8_PER_TENSOR_WEIGHT,
            FP8_DYNAMIC_PER_TENSOR_INPUT,
            True,
            id="dynamic-per-tensor",
        ),
        pytest.param(
            {**FP8_PER_TENSOR_WEIGHT, "qscheme": "per_channel"},
            {**FP8_DYNAMIC_PER_TENSOR_INPUT, "qscheme": "per_channel"},
            True,
            id="dynamic-per-token",
        ),
        pytest.param(
            FP8_PER_TENSOR_WEIGHT,
            {**FP8_DYNAMIC_PER_TENSOR_INPUT, "is_dynamic": False},
            True,
            id="static-per-tensor",
        ),
        pytest.param(
            {**FP8_PER_TENSOR_WEIGHT, "is_dynamic": True},
            FP8_DYNAMIC_PER_TENSOR_INPUT,
            False,
            id="dynamic-weight",
        ),
        pytest.param(
            FP8_PER_TENSOR_WEIGHT,
            {
                **FP8_DYNAMIC_PER_TENSOR_INPUT,
                "is_dynamic": False,
                "qscheme": "per_channel",
            },
            False,
            id="static-per-channel-input",
        ),
        pytest.param(None, FP8_DYNAMIC_PER_TENSOR_INPUT, False, id="missing-weight"),
        pytest.param(FP8_PER_TENSOR_WEIGHT, None, False, id="missing-input"),
    ],
)
def test_fp8_w8a8_scheme_detection(weight, input_quant, expected: bool):
    assert make_quark_config()._is_fp8_w8a8(weight, input_quant) is expected


@pytest.mark.parametrize(
    ("weight", "input_quant", "expected"),
    [
        pytest.param(
            INT8_STATIC_PER_TENSOR,
            INT8_STATIC_PER_TENSOR,
            True,
            id="per-tensor",
        ),
        pytest.param(
            {**INT8_STATIC_PER_TENSOR, "qscheme": "per_channel"},
            {**INT8_STATIC_PER_TENSOR, "symmetric": False},
            True,
            id="per-channel-weight-asymmetric-input",
        ),
        pytest.param(
            {**INT8_STATIC_PER_TENSOR, "symmetric": False},
            INT8_STATIC_PER_TENSOR,
            False,
            id="asymmetric-weight",
        ),
        pytest.param(
            INT8_STATIC_PER_TENSOR,
            {**INT8_STATIC_PER_TENSOR, "is_dynamic": True},
            False,
            id="dynamic-input",
        ),
        pytest.param(
            INT8_STATIC_PER_TENSOR,
            {**INT8_STATIC_PER_TENSOR, "qscheme": "per_channel"},
            False,
            id="per-channel-input",
        ),
        pytest.param(None, INT8_STATIC_PER_TENSOR, False, id="missing-weight"),
        pytest.param(INT8_STATIC_PER_TENSOR, None, False, id="missing-input"),
    ],
)
def test_static_int8_w8a8_scheme_detection(weight, input_quant, expected: bool):
    assert make_quark_config()._is_static_tensor_w8a8(weight, input_quant) is expected


@pytest.mark.parametrize(
    ("weight", "input_quant", "expected_type", "expected_attributes"),
    [
        pytest.param(NVFP4_WEIGHT, NVFP4_INPUT, QuarkNVFP4, {}, id="nvfp4"),
        pytest.param(
            FP8_PER_TENSOR_WEIGHT,
            FP8_DYNAMIC_PER_TENSOR_INPUT,
            QuarkW8A8Fp8,
            {"is_static_input_scheme": False},
            id="fp8-w8a8",
        ),
        pytest.param(
            INT8_STATIC_PER_TENSOR,
            INT8_STATIC_PER_TENSOR,
            QuarkW8A8Int8,
            {"qscheme": "per_tensor", "is_static_input_scheme": True},
            id="static-int8-w8a8",
        ),
        pytest.param(
            OCP_MX_WEIGHT,
            MXFP4_STATIC_FP8_INPUT,
            QuarkW4A8_MXFP4_FP8,
            {"weight_dtype": "mxfp4", "is_static_input_scheme": True},
            id="mxfp4-fp8-w4a8",
        ),
        pytest.param(
            INT8_STATIC_PER_TENSOR,
            INT8_DYNAMIC_PER_TOKEN_INPUT,
            QuarkW8A8Int8,
            {"qscheme": "per_tensor", "is_static_input_scheme": False},
            id="dynamic-token-int8-w8a8",
        ),
        pytest.param(
            OCP_MX_WEIGHT,
            OCP_MX_DYNAMIC_INPUT,
            QuarkOCP_MX,
            {"weight_dtype": "mxfp4", "input_dtype": "mxfp4"},
            id="ocp-mx",
        ),
    ],
)
def test_scheme_dispatch_covers_every_supported_branch(
    default_vllm_config,
    monkeypatch,
    weight,
    input_quant,
    expected_type,
    expected_attributes,
):
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    config = make_quark_config()
    monkeypatch.setattr(
        config, "_check_scheme_supported", lambda *_args, **_kwargs: True
    )

    scheme = config._get_scheme_from_config(
        {"weight": deepcopy(weight), "input_tensors": deepcopy(input_quant)}
    )

    assert type(scheme) is expected_type
    for name, expected in expected_attributes.items():
        assert getattr(scheme, name) == expected


@pytest.mark.parametrize(
    ("weight", "input_quant", "expected_factory"),
    [
        pytest.param(
            FP8_W4A8_WEIGHT,
            FP8_DYNAMIC_PER_TENSOR_INPUT,
            "QuarkW4A8Fp8MoEMethod",
            id="fp8-int4-w4a8",
        ),
        pytest.param(
            NVFP4_WEIGHT,
            NVFP4_INPUT,
            "QuarkNvfp4MoEMethod",
            id="nvfp4",
        ),
        pytest.param(
            FP8_PER_TENSOR_WEIGHT,
            FP8_DYNAMIC_PER_TENSOR_INPUT,
            "QuarkW8A8Fp8MoEMethod",
            id="fp8-w8a8",
        ),
        pytest.param(
            OCP_MX_WEIGHT,
            OCP_MX_DYNAMIC_INPUT,
            "QuarkOCP_MX_MoEMethod",
            id="ocp-mx",
        ),
        pytest.param(
            INT8_STATIC_PER_TENSOR,
            INT8_STATIC_PER_TENSOR,
            "QuarkW8A8Int8MoEMethod",
            id="static-int8",
        ),
        pytest.param(
            INT8_STATIC_PER_TENSOR,
            INT8_DYNAMIC_PER_TOKEN_INPUT,
            "QuarkW8A8Int8MoEMethod",
            id="dynamic-token-int8",
        ),
    ],
)
def test_moe_dispatch_covers_every_supported_branch(
    monkeypatch,
    weight,
    input_quant,
    expected_factory: str,
):
    config = make_quark_config()
    layer_config = {
        "weight": deepcopy(weight),
        "input_tensors": deepcopy(input_quant),
    }
    monkeypatch.setattr(config, "_find_matched_config", lambda *_args: layer_config)

    factory_names = [
        "QuarkW4A8Fp8MoEMethod",
        "QuarkNvfp4MoEMethod",
        "QuarkW8A8Fp8MoEMethod",
        "QuarkOCP_MX_MoEMethod",
        "QuarkW8A8Int8MoEMethod",
    ]
    sentinels = {name: object() for name in factory_names}
    calls: list[str] = []

    for name in factory_names:

        def factory(*_args, _name=name, **_kwargs):
            calls.append(_name)
            return sentinels[_name]

        monkeypatch.setattr(quark_moe_module, name, factory)

    result = QuarkMoEMethod.get_moe_method(
        config,
        module=SimpleNamespace(moe_config=object()),
        layer_name="model.layers.0.mlp.experts",
    )

    assert result is sentinels[expected_factory]
    assert calls == [expected_factory]


def _ocp_spec(dtype: str, *, dynamic: bool) -> dict:
    return {**OCP_MX_WEIGHT, "dtype": dtype, "is_dynamic": dynamic}


@pytest.mark.parametrize(
    ("weight_dtype", "input_dtype", "expected_scheme"),
    [
        ("fp4", None, OCP_MX_Scheme.w_mxfp4),
        ("fp4", "fp4", OCP_MX_Scheme.w_mxfp4_a_mxfp4),
        ("fp4", "fp6_e3m2", OCP_MX_Scheme.w_mxfp4_a_mxfp6_e3m2),
        ("fp4", "fp6_e2m3", OCP_MX_Scheme.w_mxfp4_a_mxfp6_e2m3),
        ("fp4", "fp8_e4m3", OCP_MX_Scheme.w_mxfp4_a_fp8),
        ("fp6_e3m2", None, OCP_MX_Scheme.w_mxfp6_e3m2),
        (
            "fp6_e3m2",
            "fp6_e3m2",
            OCP_MX_Scheme.w_mxfp6_e3m2_a_mxfp6_e3m2,
        ),
        ("fp6_e3m2", "fp8_e4m3", OCP_MX_Scheme.w_mxfp6_e3m2_a_fp8),
        ("fp6_e2m3", None, OCP_MX_Scheme.w_mxfp6_e2m3),
        (
            "fp6_e2m3",
            "fp6_e2m3",
            OCP_MX_Scheme.w_mxfp6_e2m3_a_mxfp6_e2m3,
        ),
        ("fp6_e2m3", "fp8_e4m3", OCP_MX_Scheme.w_mxfp6_e2m3_a_fp8),
    ],
)
def test_ocp_mx_constructor_maps_every_supported_dtype_pair(
    weight_dtype: str,
    input_dtype: str | None,
    expected_scheme: OCP_MX_Scheme,
):
    input_quant = None if input_dtype is None else _ocp_spec(input_dtype, dynamic=True)
    scheme = QuarkOCP_MX(
        _ocp_spec(weight_dtype, dynamic=False),
        input_quant,
    )

    assert scheme.ocp_mx_scheme is expected_scheme


@pytest.mark.parametrize(
    ("weight_dtype", "input_dtype"),
    [
        ("fp6_e3m2", "fp4"),
        ("fp6_e3m2", "fp6_e2m3"),
        ("fp6_e2m3", "fp4"),
        ("fp6_e2m3", "fp6_e3m2"),
    ],
)
def test_ocp_mx_mixed_dtype_pairs_without_enum_use_emulation(
    weight_dtype: str, input_dtype: str
):
    config = make_quark_config()
    layer_config = {
        "weight": _ocp_spec(weight_dtype, dynamic=False),
        "input_tensors": _ocp_spec(input_dtype, dynamic=True),
    }

    scheme = config._get_scheme_from_config(layer_config)

    assert isinstance(scheme, QuarkOCP_MX)
    assert scheme.ocp_mx_scheme is None
    assert scheme.emulate


def test_quark_w4a4_auto_uses_even_correct_emulation():
    backend = _enforce_quark_w4a4_rounding_contract(
        Mxfp4MoeBackend.AITER_MXFP4_MXFP4, "auto"
    )

    assert backend is Mxfp4MoeBackend.EMULATION


@pytest.mark.parametrize("requested_backend", ["aiter", "aiter_mxfp4_mxfp4"])
def test_quark_w4a4_explicit_aiter_rejects_roundup(requested_backend: str):
    with pytest.raises(ValueError, match="requires Even rounding"):
        _enforce_quark_w4a4_rounding_contract(
            Mxfp4MoeBackend.AITER_MXFP4_MXFP4, requested_backend
        )


@pytest.mark.parametrize(
    "backend",
    [Mxfp4MoeBackend.EMULATION, Mxfp4MoeBackend.TRITON],
)
def test_quark_w4a4_rounding_contract_preserves_other_backends(
    backend: Mxfp4MoeBackend,
):
    assert _enforce_quark_w4a4_rounding_contract(backend, "auto") is backend


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
