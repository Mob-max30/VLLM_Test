# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
from importlib.util import find_spec

import pytest
import torch
import torch.nn.functional as F
from packaging import version

from vllm._aiter_ops import is_aiter_found
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4,
    quant_dequant_mxfp4,
)
from vllm.platforms import current_platform

ROCM_AVAILABLE = current_platform.is_rocm()
GFX950_AVAILABLE = False
if ROCM_AVAILABLE:
    from vllm.platforms.rocm import on_gfx950

    GFX950_AVAILABLE = on_gfx950()

QUARK_MXFP4_AVAILABLE = find_spec("quark") is not None and (
    version.parse(importlib.metadata.version("amd-quark")) >= version.parse("0.12.0")
    if version.parse(torch.__version__.split("+")[0]) >= version.parse("2.11")
    else True
)

pytestmark = [
    pytest.mark.skipif(not ROCM_AVAILABLE, reason="ROCm is required"),
    pytest.mark.skipif(not GFX950_AVAILABLE, reason="GFX950 is required"),
    pytest.mark.skipif(not is_aiter_found(), reason="AITER is required"),
    pytest.mark.skipif(
        not QUARK_MXFP4_AVAILABLE,
        reason="A torch-compatible amd-quark installation is required",
    ),
]

WEIGHT_QUANT_SPEC = {
    "dtype": "fp4",
    "qscheme": "per_group",
    "group_size": 32,
    "scale_format": "e8m0",
    "is_dynamic": False,
}
INPUT_QUANT_SPEC = {**WEIGHT_QUANT_SPEC, "is_dynamic": True}


def _make_scheme(*, dynamic_weight: bool, out_dtype: torch.dtype) -> QuarkOCP_MX:
    scheme = QuarkOCP_MX(
        WEIGHT_QUANT_SPEC,
        INPUT_QUANT_SPEC,
        dynamic_mxfp4_quant=dynamic_weight,
    )
    scheme.out_dtype = out_dtype
    assert not scheme.emulate
    if scheme.rocm_use_aiter_fp4_asm_gemm:
        pytest.skip("This test covers the default non-ASM AITER MXFP4 GEMM path")
    return scheme


def _quark_pack_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from quark.torch.export.nn.modules.realquantizer import StaticScaledRealQuantizer
    from quark.torch.quantization.config.config import FP4PerGroupSpec

    qspec = FP4PerGroupSpec(
        ch_axis=-1,
        group_size=32,
        scale_format="e8m0",
        scale_calculation_mode="even",
        is_dynamic=False,
    ).to_quantization_spec()
    quantizer = StaticScaledRealQuantizer(
        qspec=qspec,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=weight.dtype,
        device=weight.device,
    )
    observer = qspec.observer_cls(qspec, device=weight.device)
    observer(weight)
    scale, _ = observer._calculate_qparams()
    quantizer.scale = scale
    packed_weight = quantizer.to_real_quantize_params(weight).to(weight.device)
    quantizer.maybe_convert_and_transpose_scale()
    row_scale = quantizer.scale.to(weight.device)
    return packed_weight, row_scale


def _make_layer(weight: torch.Tensor, scale: torch.Tensor | None = None):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    if scale is not None:
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    return layer


@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("weight_source", ["checkpoint", "dynamic"])
@torch.inference_mode()
def test_quark_ocp_mx_native_process_and_apply(m: int, weight_source: str):
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n = k = 256

    float_weight = torch.randn(n, k, dtype=dtype, device=device)
    dynamic_weight = weight_source == "dynamic"
    scheme = _make_scheme(dynamic_weight=dynamic_weight, out_dtype=dtype)

    if dynamic_weight:
        layer = _make_layer(float_weight)
        scheme.process_weights_after_loading(layer)
        row_scale = layer.weight_scale.T.contiguous()
    else:
        packed_weight, row_scale = _quark_pack_weight(float_weight)
        assert row_scale.shape == (n, k // 32)
        layer = _make_layer(packed_weight, row_scale)
        scheme.process_weights_after_loading(layer)

    assert layer.weight_scale.shape == (k // 32, n)
    x = torch.randn(m, k, dtype=dtype, device=device)
    bias = torch.randn(n, dtype=dtype, device=device)

    dq_weight = dequant_mxfp4(layer.weight, row_scale, dtype)
    expected = F.linear(quant_dequant_mxfp4(x), dq_weight, bias)
    actual = scheme.apply_weights(layer, x, bias)

    max_error = (actual.float() - expected.float()).abs().max().item()
    print(f"Quark OCP MX {weight_source=} M={m}: max absolute error={max_error:.6g}")
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@torch.inference_mode()
def test_gemm_with_dynamic_quant_fake_tensor_fp16_metadata():
    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.float16
    n = k = 256

    scheme = _make_scheme(dynamic_weight=False, out_dtype=dtype)
    packed_weight, row_scale = _quark_pack_weight(
        torch.randn(n, k, dtype=dtype, device=device)
    )
    layer = _make_layer(packed_weight, row_scale)
    scheme.process_weights_after_loading(layer)
    x = torch.randn(1, k, dtype=dtype, device=device)

    torch.library.opcheck(
        torch.ops.vllm.gemm_with_dynamic_quant,
        (x, layer.weight, layer.weight_scale, False, torch.float16),
        test_utils=("test_faketensor",),
    )
