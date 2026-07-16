# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
import types
from dataclasses import dataclass
from importlib.util import find_spec

import pytest
import torch
from packaging import version

from tests.kernels.moe.utils import check_accuracy
from tests.quantization.reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch
from vllm._aiter_ops import is_aiter_found, rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    quant_dequant_mxfp4,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

# MXFP4 via quark requires amd-quark >= 0.12 on torch >= 2.11.
# Earlier torch releases work with older quark versions. See
# https://github.com/amd/Quark/issues/34
# TODO: Remove once amd-quark>=0.12.0
QUARK_MXFP4_TORCH_COMPATIBLE = find_spec("quark") is not None and (
    version.parse(importlib.metadata.version("amd-quark")) >= version.parse("0.12.0")
    if version.parse(torch.__version__.split("+")[0]) >= version.parse("2.11")
    else True
)

TRTLLM_GEN_MXFP4_AVAILABLE = (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
)

TRTLLM_GEN_MXFP8_AVAILABLE = TRTLLM_GEN_MXFP4_AVAILABLE

HOPPER_MXFP4_BF16_AVAILABLE = (
    current_platform.is_cuda()
    and current_platform.is_device_capability(90)
    and has_flashinfer()
)

# ROCm platform and dependencies
ROCM_AVAILABLE = current_platform.is_rocm()
ROCM_TRITON_KERNELS_AVAILABLE = False
ROCM_AITER_AVAILABLE = is_aiter_found()
ROCM_GFX950 = False

if ROCM_AVAILABLE:
    from vllm.platforms.rocm import on_gfx950
    from vllm.utils.import_utils import has_triton_kernels

    ROCM_TRITON_KERNELS_AVAILABLE = has_triton_kernels()
    ROCM_GFX950 = on_gfx950()

    if ROCM_AITER_AVAILABLE:
        from aiter.ops.quant import per_1x32_f4_quant
        from aiter.ops.triton.moe.quant_moe import upcast_from_mxfp
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

if TRTLLM_GEN_MXFP4_AVAILABLE:
    from flashinfer import (
        fp4_quantize,
        mxfp8_quantize,
        reorder_rows_for_gated_act_gemm,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        trtllm_fp4_block_scale_moe,
        trtllm_fp8_block_scale_moe,
    )
    from flashinfer.fp4_quantization import nvfp4_block_scale_interleave

if TRTLLM_GEN_MXFP8_AVAILABLE:
    from flashinfer.fused_moe.core import (
        Fp8QuantizationType,
        get_w2_permute_indices_with_cache,
    )


@dataclass
class ModelCase:
    model_id: str
    tp: int


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


@pytest.mark.parametrize(
    "model_case",
    [
        ModelCase("fxmarty/qwen_1.5-moe-a2.7b-mxfp4", tp=2),
        ModelCase("fxmarty/deepseek_r1_3_layers_mxfp4", tp=8),
        ModelCase("mawong-amd/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4", tp=1),
        ModelCase("fxmarty/Llama-3.1-70B-Instruct-2-layers-mxfp6", tp=1),
        ModelCase("fxmarty/Llama-3.1-70B-Instruct-2-layers-mxfp6", tp=4),
    ],
)
@pytest.mark.skipif(
    not QUARK_MXFP4_TORCH_COMPATIBLE,
    reason="MXFP4 via quark requires amd-quark >= 0.12 on torch >= 2.11.",
)
def test_mxfp4_loading_and_execution_moe(vllm_runner, model_case: ModelCase):
    if torch.accelerator.device_count() < model_case.tp:
        pytest.skip(
            f"This test requires >={model_case.tp} gpus, got only "
            f"{torch.accelerator.device_count()}"
        )

    # `cudagraph_capture_sizes=[16]` to reduce load time.
    with vllm_runner(
        model_case.model_id,
        tensor_parallel_size=model_case.tp,
        load_format="dummy",
        compilation_config={"cudagraph_capture_sizes": [16]},
        gpu_memory_utilization=0.8,  # mxfp6 models use more scratch space
    ) as llm:
        # Disabled as check_model is broken: https://github.com/vllm-project/vllm/pull/18465#issuecomment-3329880562
        # def check_model(model):
        #     from vllm.model_executor.layers.quantization.quark.quark import (  # noqa: E501
        #         QuarkLinearMethod)
        #     from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import QuarkOCP_MX  # noqa: E501
        #     from vllm.model_executor.layers.quantization.quark.quark_moe import (  # noqa: E501
        #         QuarkOCP_MX_MoEMethod)

        #     layer = model.model.layers[0]

        #     qkv_proj = layer.self_attn.qkv_proj

        #     assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
        #     assert isinstance(qkv_proj.scheme, QuarkOCP_MX)

        #     assert isinstance(layer.mlp.experts.quant_method,
        #                       QuarkOCP_MX_MoEMethod)

        # if model_case.model_id == "fxmarty/qwen_1.5-moe-a2.7b-mxfp4":
        #     llm.apply_model(check_model)

        output = llm.generate_greedy("Today I am in the French Alps and", max_tokens=20)
        assert output


def swiglu(x, alpha: float = 1.702, beta: float = 1.0, limit: float | None = None):
    # Note we add an extra bias of 1 to the linear layer
    # Uses chunked layout: first half is gate, second half is up
    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + beta)


def swigluoai(x, alpha: float = 1.702, limit: float = 7.0):
    # OAI swiglu uses interleaved layout: gate/up alternating
    # See SwigluOAIAndMul in vllm/model_executor/layers/activation.py
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


fp4_lookup_table = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]


def mxfp4_dequantize(x, scale):
    assert x.dtype == torch.uint8
    x = x.view(torch.uint8).to(torch.int32)
    x_unpacked = torch.zeros(
        *x.shape[:-1], x.shape[-1] * 2, dtype=torch.int32, device=x.device
    )
    x_unpacked[..., 0::2].copy_(x & 0xF)
    x_unpacked[..., 1::2].copy_((x >> 4) & 0xF)

    x_float = torch.zeros(x_unpacked.shape, dtype=torch.float32, device=x.device)
    for i, val in enumerate(fp4_lookup_table):
        x_float[x_unpacked == i] = val

    scale = scale.view(torch.uint8).to(torch.int32)
    scale = (scale << 23).view(torch.float32)
    scale = scale.reshape(*x.shape[:-1], -1)
    scale = torch.stack([scale] * 32, dim=-1).reshape(*x_float.shape)

    return x_float * scale


def mxfp8_dequantize(x, scale):
    assert x.dtype == torch.float8_e4m3fn
    x_float = x.to(torch.float32)

    scale = scale.view(torch.uint8).to(torch.int32)
    scale = (scale << 23).view(torch.float32)
    scale = scale.reshape(*x.shape[:-1], -1)
    scale = torch.stack([scale] * 32, dim=-1).reshape(*x_float.shape)

    return x_float * scale


def aiter_roundup_mxfp4_quant_dequantize(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x_2d = x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous()
    # AITER's W4A4 MoE path uses its per-1x32 RoundUp quantizer internally.
    # TODO(aiter): Quark checkpoints declare Even activation scaling. Plumb
    # the scale-rounding mode through AITER fused MoE, then use Even here.
    x_quant, x_scale = per_1x32_f4_quant(x_2d)
    x_dequant = upcast_from_mxfp(
        x_quant.view(torch.uint8), x_scale.view(torch.uint8), torch.bfloat16, axis=-1
    )
    return x_dequant.reshape(shape).to(x.dtype)


def quark_even_mxfp4_quant_dequantize(x: torch.Tensor) -> torch.Tensor:
    """Apply Quark's declared Even MXFP4 QDQ at the BF16 kernel boundary."""
    return quant_dequant_mxfp4(x.to(torch.bfloat16), "even").to(x.dtype)


def quark_pack_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack checkpoint-layout Quark MXFP4 weights with Even E8M0 scales."""
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
    packed = quantizer.to_real_quantize_params(weight).to(weight.device)
    quantizer.maybe_convert_and_transpose_scale()
    return packed, quantizer.scale.to(weight.device)


def static_fp8_quant_dequantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Independent GFX950 static-E4M3 QDQ with saturating conversion."""
    flat_scale = scale.float().reshape(-1)
    assert flat_scale.numel() > 0
    assert torch.all(flat_scale == flat_scale[0]), (
        "The AITER W4A8 kernel accepts one shared input scale per GEMM"
    )
    scalar_scale = flat_scale[0]
    assert torch.isfinite(scalar_scale) and scalar_scale > 0
    quantized = torch.clamp(x.float() / scalar_scale, -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return (quantized.float() * scalar_scale).to(x.dtype)


@pytest.mark.skipif(not ROCM_AVAILABLE, reason="ROCm is required for this test")
@pytest.mark.skipif(not ROCM_AITER_AVAILABLE, reason="AITER is required")
@pytest.mark.parametrize("scale_value", [0.125, 2.0])
@torch.inference_mode()
def test_aiter_static_fp8_quantization_matches_saturating_reference(
    scale_value: float,
):
    """Static FP8 conversion is discrete, so packed bytes must match exactly."""
    from aiter.ops.triton.quant_moe import downcast_to_static_fp8

    normalized = torch.tensor(
        [
            -1000.0,
            -449.0,
            -448.0,
            -1.0625,
            -1.0,
            -0.0,
            0.0,
            1.0,
            1.0625,
            447.0,
            448.0,
            449.0,
            1000.0,
        ],
        dtype=torch.float32,
        device="cuda",
    )
    x = (normalized * scale_value).repeat(3, 1).to(torch.bfloat16).contiguous()
    scale = torch.tensor(scale_value, dtype=torch.float32, device="cuda")

    actual = downcast_to_static_fp8(x, scale)
    expected = torch.clamp(x.float() / scale, -448.0, 448.0).to(torch.float8_e4m3fn)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    actual_qdq = (actual.float() * scale).to(x.dtype)
    expected_qdq = static_fp8_quant_dequantize(x, scale)
    assert torch.equal(actual_qdq.view(torch.uint16), expected_qdq.view(torch.uint16))


def reference_moe(
    roouting_logits,
    topk,
    num_experts,
    hidden_states,
    w13,
    bias13,
    w2,
    bias2,
    alpha,
    beta,
    limit,
    act_type,
    activation: str = "swiglu",
    use_interleaved_layout: bool = False,
    input_scale1: torch.Tensor | None = None,
    input_scale2: torch.Tensor | None = None,
    expert_weights_override: torch.Tensor | None = None,
    expert_indices_override: torch.Tensor | None = None,
):
    """
    Reference MoE implementation for accuracy testing.

    Args:
        activation: One of "swiglu", "silu", "relu2". Controls the activation
            function used after the first MLP.
        use_interleaved_layout: If True, uses interleaved gate/up layout
            (gate=x[..., ::2], up=x[..., 1::2]) as used by SWIGLUOAI.
            If False, uses chunked layout (gate, up = chunk(x, 2)) as used
            by standard swiglu/silu.
    """
    if expert_weights_override is None:
        assert expert_indices_override is None
        experts = torch.topk(roouting_logits, k=topk, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices
    else:
        assert expert_indices_override is not None
        expert_weights = expert_weights_override
        expert_indices = expert_indices_override
        assert expert_weights.shape == expert_indices.shape
        assert expert_weights.shape[-1] == topk
    t = hidden_states.clone()
    if act_type == "mxfp4_roundup":
        t = aiter_roundup_mxfp4_quant_dequantize(t)
    elif act_type in ("mxfp4_even", "mxfp4_even_emulation"):
        t = quark_even_mxfp4_quant_dequantize(t)
    elif act_type == "fp8_static":
        assert input_scale1 is not None
        t = static_fp8_quant_dequantize(t, input_scale1)
    # MLP #1
    mlp1_weight = w13[expert_indices, ...]
    mlp1_bias = bias13[expert_indices, ...]
    t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias

    # Triton emulation materializes the first GEMM result in BF16 before
    # applying the activation. Native AITER fuses that boundary differently,
    # so the emulation contract uses a distinct reference mode.
    if act_type in ("mxfp4_even_emulation", "bf16_pipeline"):
        t = t.to(torch.bfloat16).to(torch.float32)

    # Apply activation
    if activation in ("swiglu", "silu"):
        if use_interleaved_layout:
            # SWIGLUOAI: interleaved gate/up layout
            t = swigluoai(t, alpha=alpha, limit=limit)
        else:
            # Standard swiglu/silu: chunked layout
            t = swiglu(t, alpha=alpha, beta=beta, limit=limit)
    elif activation == "relu2":
        # RELU2_NO_MUL: relu(x)^2
        t = torch.relu(t)
        t = t * t
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if act_type == "mxfp8":
        t_quantized, t_scale = mxfp8_quantize(
            t.to(torch.bfloat16), is_sf_swizzled_layout=False
        )
        t = mxfp8_dequantize(t_quantized, t_scale)
    elif act_type == "mxfp4_roundup":
        t = aiter_roundup_mxfp4_quant_dequantize(t)
    elif act_type in ("mxfp4_even", "mxfp4_even_emulation"):
        t = quark_even_mxfp4_quant_dequantize(t)
    elif act_type == "fp8_static":
        assert input_scale2 is not None
        t = static_fp8_quant_dequantize(t, input_scale2)
    elif act_type in ("bf16_intermediate", "bf16_pipeline"):
        t = t.to(torch.bfloat16).to(torch.float32)
    # MLP #2
    mlp2_weight = w2[expert_indices, ...]
    mlp2_bias = bias2[expert_indices, ...]
    t = torch.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
    # Weighted sum of experts
    if act_type in {
        "mxfp4_even_emulation",
        "bf16_pipeline",
        "bf16_intermediate",
        "fp8_static",
    }:
        # These modular paths multiply each routed expert result by its top-k
        # weight and store it to a BF16 workspace.  ``moe_sum`` then performs
        # the top-k reduction in FP32 and rounds once to BF16.  Modeling that
        # discrete boundary is stronger than admitting a percentage tolerance
        # around a reference which performs the operations in another order.
        t = (t * expert_weights.unsqueeze(-1)).to(torch.bfloat16)
        t = t.float().sum(dim=1)
    else:
        t = torch.einsum("bec,be->bc", t, expert_weights)
    assert t.shape == hidden_states.shape
    return t.to(torch.bfloat16)


def tg_mxfp4_moe(
    router_logits,
    topk,
    num_experts,
    intermediate_size,
    hidden_size,
    hidden_states,
    hidden_states_scale,
    w13_weight,
    w13_weight_scale,
    w13_bias,
    w2_weight,
    w2_weight_scale,
    w2_bias,
    act_type,
    alpha,
    beta,
    limit,
    transpose_optimized: bool = False,
) -> torch.Tensor:
    sf_block_size = 32
    assert (
        w13_weight.dim() == 3
        and w13_weight.shape[0] == num_experts
        and w13_weight.shape[1] == intermediate_size * 2
        and w13_weight.shape[2] == hidden_size // 2
    )
    assert (
        w13_weight_scale.dim() == 3
        and w13_weight_scale.shape[0] == num_experts
        and w13_weight_scale.shape[1] == intermediate_size * 2
        and w13_weight_scale.shape[2] == hidden_size // sf_block_size
    )
    assert (
        w2_weight.dim() == 3
        and w2_weight.shape[0] == num_experts
        and w2_weight.shape[1] == hidden_size
        and w2_weight.shape[2] == intermediate_size // 2
    )
    assert (
        w2_weight_scale.dim() == 3
        and w2_weight_scale.shape[1] == hidden_size
        and w2_weight_scale.shape[2] == intermediate_size // sf_block_size
    )
    assert (
        w13_bias.dim() == 2
        and w13_bias.shape[0] == num_experts
        and w13_bias.shape[1] == intermediate_size * 2
    )
    assert (
        w2_bias.dim() == 2
        and w2_bias.shape[0] == num_experts
        and w2_bias.shape[1] == hidden_size
    )

    # Swap w1 and w3 as the definition of
    # swiglu is different in the trtllm-gen
    w13_weight_scale_ = w13_weight_scale.clone()
    w13_weight_ = w13_weight.clone()
    w13_bias_ = w13_bias.clone()
    w13_weight[:, :intermediate_size, :].copy_(w13_weight_[:, intermediate_size:, :])
    w13_weight[:, intermediate_size:, :].copy_(w13_weight_[:, :intermediate_size, :])
    w13_weight_scale[:, :intermediate_size, :].copy_(
        w13_weight_scale_[:, intermediate_size:, :]
    )
    w13_weight_scale[:, intermediate_size:, :].copy_(
        w13_weight_scale_[:, :intermediate_size, :]
    )
    w13_bias[:, :intermediate_size].copy_(w13_bias_[:, intermediate_size:])
    w13_bias[:, intermediate_size:].copy_(w13_bias_[:, :intermediate_size])

    # Interleave the weights and scaling factors for activation
    w13_weight_interleaved = []
    w13_weight_scale_interleaved = []
    w13_bias_interleaved = []
    for i in range(num_experts):
        w13_weight_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_weight[i].clone())
        )
        w13_weight_scale_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_weight_scale[i].clone())
        )
        w13_bias_interleaved.append(
            reorder_rows_for_gated_act_gemm(w13_bias[i].clone().reshape(-1, 1))
        )
    w13_weight = torch.stack(w13_weight_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 2
    )
    w13_weight_scale = torch.stack(w13_weight_scale_interleaved).reshape(
        num_experts, 2 * intermediate_size, hidden_size // 32
    )
    w13_bias = torch.stack(w13_bias_interleaved).reshape(
        num_experts, 2 * intermediate_size
    )

    # Shuffle weights and scaling factors for transposed mma output
    gemm1_weights_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_weights_shuffled = []
    gemm2_scales_shuffled = []
    gemm1_bias_shuffled = []
    gemm2_bias_shuffled = []
    epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
    _cache_permute_indices: dict[torch.Size, torch.Tensor] = {}
    if transpose_optimized:
        for i in range(num_experts):
            # w13 weight shuffling
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_weight[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_shuffled.append(
                w13_weight[i]
                .view(torch.uint8)[permute_indices.to(w13_weight.device)]
                .contiguous()
            )
            # w13 scale shuffling
            permute_sf_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_weight_scale[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm1_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w13_weight_scale[i]
                    .view(torch.uint8)[permute_sf_indices.to(w13_weight_scale.device)]
                    .contiguous()
                )
            )
            # w13 bias shuffling
            permute_bias_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w13_bias[i].clone().reshape(-1, 1),
                epilogue_tile_m,
            )
            gemm1_bias_shuffled.append(
                w13_bias[i]
                .clone()
                .reshape(-1, 1)[permute_bias_indices.to(w13_bias.device)]
                .contiguous()
            )
            # w2 weight shuffling
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_weight[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_shuffled.append(
                w2_weight[i]
                .view(torch.uint8)[permute_indices.to(w2_weight.device)]
                .contiguous()
            )
            # w2 scale shuffling
            permute_sf_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_weight_scale[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w2_weight_scale[i]
                    .view(torch.uint8)[permute_sf_indices.to(w2_weight_scale.device)]
                    .contiguous()
                )
            )
            # w2 bias shuffling
            permute_indices = get_w2_permute_indices_with_cache(
                _cache_permute_indices,
                w2_bias[i].clone().reshape(-1, 1),
                epilogue_tile_m,
            )
            gemm2_bias_shuffled.append(
                w2_bias[i]
                .clone()
                .reshape(-1, 1)[permute_indices.to(w2_bias.device)]
                .contiguous()
            )

    else:
        for i in range(num_experts):
            gemm1_weights_shuffled.append(
                shuffle_matrix_a(w13_weight[i].view(torch.uint8), epilogue_tile_m)
            )
            gemm1_scales_shuffled.append(
                shuffle_matrix_sf_a(
                    w13_weight_scale[i].view(torch.uint8), epilogue_tile_m
                )
            )

            gemm2_weights_shuffled.append(
                shuffle_matrix_a(w2_weight[i].view(torch.uint8), epilogue_tile_m)
            )
            gemm2_scales_shuffled.append(
                shuffle_matrix_sf_a(
                    w2_weight_scale[i].view(torch.uint8), epilogue_tile_m
                )
            )
            gemm1_bias_shuffled.append(
                shuffle_matrix_a(w13_bias[i].reshape(-1, 1), epilogue_tile_m)
            )
            gemm2_bias_shuffled.append(
                shuffle_matrix_a(w2_bias[i].reshape(-1, 1), epilogue_tile_m)
            )

    w13_weight = torch.stack(gemm1_weights_shuffled)
    w13_weight_scale = (
        torch.stack(gemm1_scales_shuffled)
        .reshape(num_experts, 2 * intermediate_size, hidden_size // sf_block_size)
        .view(torch.float8_e4m3fn)
    )
    w13_bias = torch.stack(gemm1_bias_shuffled).reshape(num_experts, -1)

    w2_weight = torch.stack(gemm2_weights_shuffled)
    w2_weight_scale = (
        torch.stack(gemm2_scales_shuffled)
        .reshape(num_experts, hidden_size, intermediate_size // sf_block_size)
        .view(torch.float8_e4m3fn)
    )
    w2_bias = torch.stack(gemm2_bias_shuffled).reshape(num_experts, -1)

    tg_result = trtllm_fp4_block_scale_moe(
        routing_logits=router_logits.to(torch.bfloat16),
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=w13_weight,
        gemm1_weights_scale=w13_weight_scale,
        gemm1_bias=w13_bias,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
        gemm2_weights=w2_weight,
        gemm2_weights_scale=w2_weight_scale,
        gemm2_bias=w2_bias,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=num_experts,
        top_k=topk,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,  # renormalize
        do_finalize=True,
    )[0]
    return tg_result


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [32, 128])
@pytest.mark.parametrize("num_tokens", [1, 128, 1024])
@pytest.mark.parametrize("intermediate_size,hidden_size", [(3072, 3072)])
@pytest.mark.parametrize("alpha,beta,limit", [(1.0, 1.0, None), (1.702, 1.0, 7.0)])
@pytest.mark.parametrize("act_type", ["mxfp8", "bf16"])
@pytest.mark.parametrize("transpose_optimized", [False, True])
@pytest.mark.skipif(
    not TRTLLM_GEN_MXFP4_AVAILABLE,
    reason="nvidia gpu and compute capability sm100 is required for this test",
)
def test_trtllm_gen_mxfp4_fused_moe(
    topk: int,
    num_experts: int,
    num_tokens: int,
    intermediate_size: int,
    hidden_size: int,
    alpha: float,
    beta: float,
    limit: float | None,
    act_type: str,
    transpose_optimized: bool,
):
    seed = 42
    torch.manual_seed(seed)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device="cuda:0", dtype=torch.bfloat16
    )
    w13 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    bias13 = torch.randn(num_experts, intermediate_size * 2, device="cuda:0") * 10
    bias2 = torch.randn(num_experts, hidden_size, device="cuda:0") * 10
    router_logits = torch.rand(num_tokens, num_experts, dtype=torch.float32).cuda()

    w13, w13_scale = fp4_quantize(
        w13,
        torch.tensor(1.0, device="cuda:0"),
        32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=False,
    )
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, intermediate_size * 2, hidden_size // 32
    )
    w2, w2_scale = fp4_quantize(
        w2,
        torch.tensor(1.0, device="cuda:0"),
        32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=False,
    )
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // 32
    )
    if act_type == "mxfp8":
        hidden_states, hidden_states_scale = mxfp8_quantize(
            hidden_states, is_sf_swizzled_layout=False
        )
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            *hidden_states.shape[:-1], -1
        )
    else:
        hidden_states_scale = None

    # reference result
    ref_result = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    w13_ref = mxfp4_dequantize(w13.clone(), w13_scale.clone())
    w2_ref = mxfp4_dequantize(w2.clone(), w2_scale.clone())
    bias13_ref = bias13
    bias2_ref = bias2
    if act_type == "mxfp8":
        hidden_states_ref = mxfp8_dequantize(hidden_states, hidden_states_scale).to(
            torch.float32
        )
    else:
        hidden_states_ref = hidden_states.to(torch.float32)
    # Process tokens in chunks of 32 to reduce memory usage
    chunk_size = 32
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_tokens)
        chunk_result = reference_moe(
            router_logits[start_idx:end_idx].to(torch.float32),
            topk,
            num_experts,
            hidden_states_ref[start_idx:end_idx],
            w13_ref,
            bias13_ref,
            w2_ref,
            bias2_ref,
            alpha,
            beta,
            limit,
            act_type,
            activation="swiglu",
            use_interleaved_layout=False,
        )
        ref_result[start_idx:end_idx].copy_(chunk_result)

    # trtllm-gen result
    if alpha is not None:
        alpha = torch.full((num_experts,), alpha, device=hidden_states.device)
    if limit is not None:
        limit = torch.full((num_experts,), limit, device=hidden_states.device)
    if beta is not None:
        beta = torch.full((num_experts,), beta, device=hidden_states.device)
    tg_result = tg_mxfp4_moe(
        router_logits=router_logits,
        topk=topk,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        w13_weight=w13,
        w13_weight_scale=w13_scale,
        w13_bias=bias13,
        w2_weight=w2,
        w2_weight_scale=w2_scale,
        w2_bias=bias2,
        act_type=act_type,
        alpha=alpha,
        beta=beta,
        limit=limit,
        transpose_optimized=transpose_optimized,
    )
    # relatively loose check since the mxfp4 quantization is less accurate
    check_accuracy(ref_result, tg_result, atol=0, rtol=0.3, percent=0.8)


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("intermediate_size,hidden_size", [(3072, 3072)])
@pytest.mark.parametrize("alpha,beta,limit", [(1.0, 1.0, None), (1.702, 1.0, 7.0)])
@pytest.mark.skipif(
    not HOPPER_MXFP4_BF16_AVAILABLE,
    reason="nvidia gpu sm90 and flashinfer are required for this test",
)
def test_flashinfer_cutlass_mxfp4_fused_moe(
    topk: int,
    num_experts: int,
    num_tokens: int,
    intermediate_size: int,
    hidden_size: int,
    alpha: float,
    beta: float,
    limit: float | None,
):
    torch.manual_seed(42)
    device = "cuda:0"

    # Inputs
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    # Random MXFP4 weights and scales (uint8), contiguous [w1; w3]
    w13_q = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w13_scale = torch.randint(
        118,
        123,
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        device=device,
        dtype=torch.uint8,
    )

    w2_q = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        device=device,
        dtype=torch.uint8,
    )
    w2_scale = torch.randint(
        118,
        123,
        (num_experts, hidden_size, intermediate_size // 32),
        device=device,
        dtype=torch.uint8,
    )
    # Bias contiguous [b1; b3]
    bias13 = (
        torch.randn(
            num_experts, 2 * intermediate_size, device=device, dtype=torch.bfloat16
        )
        * 10
    )
    bias2 = (
        torch.randn(num_experts, hidden_size, device=device, dtype=torch.bfloat16) * 10
    )
    router_logits = torch.rand(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    w13_ref = mxfp4_dequantize(w13_q.clone(), w13_scale.clone()).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )
    w2_ref = mxfp4_dequantize(w2_q.clone(), w2_scale.clone()).reshape(
        num_experts, hidden_size, intermediate_size
    )
    ref = reference_moe(
        router_logits.to(torch.float32),
        topk,
        num_experts,
        hidden_states.to(torch.float32),
        w13_ref,
        bias13.to(torch.float32),
        w2_ref,
        bias2.to(torch.float32),
        alpha,
        beta,
        limit,
        "bf16",
        activation="swiglu",
        use_interleaved_layout=False,
    )

    from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

    # Swap halves to arrange as [w3; w1] (kernel expectation)
    w1_w, w3_w = torch.chunk(w13_q, 2, dim=1)
    w13_q_swapped = torch.cat([w3_w, w1_w], dim=1)

    # SM90 mixed-input GEMM expects weights/scales in an interleaved layout;
    # without it the FP4->BF16 LUT reads bytes from wrong positions for K>128.
    from flashinfer.fused_moe import (
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )

    w13_q_swapped = interleave_moe_weights_for_sm90_mixed_gemm(
        w13_q_swapped, quant_type="fp4"
    )
    w2_q = interleave_moe_weights_for_sm90_mixed_gemm(w2_q, quant_type="fp4")

    b1, b3 = torch.chunk(bias13.to(torch.float32), 2, dim=-1)
    w13_b = torch.cat([b3, b1], dim=-1).to(torch.bfloat16)

    w1_s, w3_s = torch.chunk(w13_scale, 2, dim=1)
    w13_s = torch.cat([w3_s, w1_s], dim=1)
    w13_s_inter = interleave_moe_scales_for_sm90_mixed_gemm(w13_s)
    w2_s_inter = interleave_moe_scales_for_sm90_mixed_gemm(w2_scale)

    routing_weights = torch.nn.functional.softmax(
        router_logits, dim=1, dtype=torch.float32
    )
    token_final_scales, token_selected_experts = torch.topk(
        routing_weights, topk, dim=-1
    )
    token_final_scales = token_final_scales / token_final_scales.sum(
        dim=-1, keepdim=True
    )
    token_selected_experts = token_selected_experts.to(torch.int).contiguous()

    out = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    if alpha is not None:
        alpha = torch.full((num_experts,), alpha, device=hidden_states.device)
    if beta is not None:
        beta = torch.full((num_experts,), beta, device=hidden_states.device)
    if limit is not None:
        limit = torch.full((num_experts,), limit, device=hidden_states.device)

    _ = flashinfer_cutlass_fused_moe(
        input=hidden_states,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        fc1_expert_weights=w13_q_swapped,
        fc2_expert_weights=w2_q,
        output_dtype=torch.bfloat16,
        output=out,
        quant_scales=[w13_s_inter.to(torch.uint8), w2_s_inter.to(torch.uint8)],
        fc1_expert_biases=w13_b,
        fc2_expert_biases=bias2.to(torch.bfloat16),
        swiglu_alpha=alpha,
        swiglu_beta=beta,
        swiglu_limit=limit,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        use_w4_group_scaling=True,
    )

    # Allow some mismatch due to MXFP4 quantization
    check_accuracy(ref, out, atol=0, rtol=0.3, percent=0.8)


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("intermediate_size,hidden_size", [(3072, 3072)])
@pytest.mark.parametrize("alpha,beta,limit", [(1.0, 1.0, None), (1.702, 1.0, 7.0)])
@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_flashinfer()
    ),
    reason="NVIDIA GPU sm100 and flashinfer are required for this test",
)
def test_flashinfer_cutlass_mxfp4_mxfp8_fused_moe(
    topk: int,
    num_experts: int,
    num_tokens: int,
    intermediate_size: int,
    hidden_size: int,
    alpha: float | None,
    beta: float | None,
    limit: float | None,
):
    torch.manual_seed(42)
    device = "cuda:0"

    # Inputs
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    # Float weights in w13 format [w1; w3]
    w13 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    # Bias contiguous [b1; b3]
    bias13 = (
        torch.randn(
            num_experts, 2 * intermediate_size, device=device, dtype=torch.bfloat16
        )
        * 10
    )
    bias2 = (
        torch.randn(num_experts, hidden_size, device=device, dtype=torch.bfloat16) * 10
    )
    router_logits = torch.rand(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    # Quantize weights to MXFP4 per expert (SM100 path)
    from flashinfer import mxfp4_quantize

    def quant_mxfp4_batches(a: torch.Tensor, e: int):
        qs, sfs = [], []
        for i in range(e):
            q, sf = mxfp4_quantize(a[i].cuda())
            qs.append(q)
            sfs.append(sf)
        return torch.stack(qs), torch.stack(sfs)

    def dequant_mxfp4_batches(mat_fp4: torch.Tensor, scale_tensor: torch.Tensor):
        num_batches = mat_fp4.size(0)
        scale_tensor = scale_tensor.view(num_batches, -1)
        from flashinfer import mxfp4_dequantize

        return torch.stack(
            [
                mxfp4_dequantize(mat_fp4[b, :, :], scale_tensor[b, :])
                for b in range(num_batches)
            ]
        )

    w13_q, w13_scale = quant_mxfp4_batches(w13, num_experts)
    w2_q, w2_scale = quant_mxfp4_batches(w2, num_experts)

    # Reference result using dequantized tensors and reference_moe
    w13_ref = (
        dequant_mxfp4_batches(
            w13_q.view(torch.uint8), w13_scale.view(torch.uint8).reshape(-1)
        )
        .to(torch.float32)
        .reshape(num_experts, 2 * intermediate_size, hidden_size)
        .to(device)
    )
    w2_ref = (
        dequant_mxfp4_batches(
            w2_q.view(torch.uint8), w2_scale.view(torch.uint8).reshape(-1)
        )
        .to(torch.float32)
        .reshape(num_experts, hidden_size, intermediate_size)
        .to(device)
    )

    # Quantize activations for SM100 path and dequantize for reference
    hidden_states_q, hidden_states_sf = mxfp8_quantize(hidden_states, True, 32)
    # Reference uses BF16 input but quantizes intermediate activation to MXFP8
    ref = reference_moe(
        router_logits.to(torch.float32),
        topk,
        num_experts,
        hidden_states.to(torch.float32),
        w13_ref,
        bias13.to(torch.float32),
        w2_ref,
        bias2.to(torch.float32),
        alpha,
        beta,
        limit,
        "mxfp8",
        activation="swiglu",
        use_interleaved_layout=False,
    )

    # Prepare inputs for FlashInfer CUTLASS fused MoE
    from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

    # Swap halves to arrange as [w3; w1] (kernel expectation)
    w1_w, w3_w = torch.chunk(w13_q, 2, dim=1)
    w13_q_swapped = torch.cat([w3_w, w1_w], dim=1)

    # Swap scales halves to match swapped weights
    s1, s3 = torch.chunk(w13_scale, 2, dim=1)
    w13_scale_swapped = torch.cat([s3, s1], dim=1)

    b1, b3 = torch.chunk(bias13.to(torch.float32), 2, dim=-1)
    w13_b = torch.cat([b3, b1], dim=-1).to(torch.bfloat16)

    # Build routing for kernel
    routing_weights = torch.nn.functional.softmax(
        router_logits, dim=1, dtype=torch.float32
    )
    token_final_scales, token_selected_experts = torch.topk(
        routing_weights, topk, dim=-1
    )
    token_final_scales = token_final_scales / token_final_scales.sum(
        dim=-1, keepdim=True
    )
    token_selected_experts = token_selected_experts.to(torch.int).contiguous()

    out = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    if alpha is not None:
        alpha_t = torch.full((num_experts,), alpha, device=hidden_states.device)
    else:
        alpha_t = None
    if beta is not None:
        beta_t = torch.full((num_experts,), beta, device=hidden_states.device)
    else:
        beta_t = None
    if limit is not None:
        limit_t = torch.full((num_experts,), limit, device=hidden_states.device)
    else:
        limit_t = None

    # Quant scales for SM100 MXFP8+MXFP4 path
    fake_input_scale = torch.ones(num_experts, device=device)
    quant_scales = [
        w13_scale_swapped.view(torch.int32),
        fake_input_scale,
        w2_scale.view(torch.int32),
        fake_input_scale,
    ]

    _ = flashinfer_cutlass_fused_moe(
        input=hidden_states_q,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        fc1_expert_weights=w13_q_swapped.contiguous().view(torch.long),
        fc2_expert_weights=w2_q.contiguous().view(torch.long),
        output_dtype=torch.bfloat16,
        output=out,
        quant_scales=quant_scales,
        fc1_expert_biases=w13_b,
        fc2_expert_biases=bias2.to(torch.bfloat16),
        swiglu_alpha=alpha_t,
        swiglu_beta=beta_t,
        swiglu_limit=limit_t,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        use_mxfp8_act_scaling=True,
        input_sf=hidden_states_sf,
    )

    # Allow some mismatch due to MXFP4 quantization
    check_accuracy(ref, out, atol=0, rtol=0.3, percent=0.8)


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("intermediate_size,hidden_size", [(3072, 3072)])
@pytest.mark.parametrize("is_gated", [True], ids=["gated"])
@pytest.mark.skipif(
    not TRTLLM_GEN_MXFP8_AVAILABLE,
    reason="nvidia gpu and compute capability sm100 is required for this test",
)
def test_trtllm_gen_mxfp8_block_scale_moe(
    topk: int,
    num_experts: int,
    num_tokens: int,
    intermediate_size: int,
    hidden_size: int,
    is_gated: bool,
):
    torch.manual_seed(42)
    device = "cuda:0"

    inter_size = intermediate_size * (2 if is_gated else 1)

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 20
    )
    w13 = (
        torch.randn(
            num_experts,
            inter_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 20
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 20
    )
    router_logits = torch.rand(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )
    router_logits_kernel = router_logits.to(torch.bfloat16)

    # Quantize weights to MXFP8 and normalize scales to [E, M, K//32].
    w13_q, w13_scale = mxfp8_quantize(w13, is_sf_swizzled_layout=False)
    w2_q, w2_scale = mxfp8_quantize(w2, is_sf_swizzled_layout=False)
    if w13_scale.ndim == 1:
        w13_scale = w13_scale.view(
            num_experts,
            inter_size,
            hidden_size // 32,
        )
    if w2_scale.ndim == 1:
        w2_scale = w2_scale.view(num_experts, hidden_size, intermediate_size // 32)

    # Quantize activations to MXFP8.
    hidden_states_q, hidden_states_scale = mxfp8_quantize(
        hidden_states, is_sf_swizzled_layout=False
    )
    if hidden_states_scale.ndim == 1:
        hidden_states_scale = hidden_states_scale.view(num_tokens, hidden_size // 32)

    # Reference output using dequantized tensors + MXFP8 intermediate quantization.
    w13_ref = mxfp8_dequantize(w13_q, w13_scale).to(torch.float32)
    w2_ref = mxfp8_dequantize(w2_q, w2_scale).to(torch.float32)
    hidden_states_ref = mxfp8_dequantize(hidden_states_q, hidden_states_scale).to(
        torch.float32
    )
    bias13 = torch.zeros(
        num_experts,
        intermediate_size * (2 if is_gated else 1),
        device=device,
    )
    bias2 = torch.zeros(num_experts, hidden_size, device=device)
    ref = reference_moe(
        router_logits_kernel.to(torch.float32),
        topk,
        num_experts,
        hidden_states_ref,
        w13_ref,
        bias13,
        w2_ref,
        bias2,
        alpha=1.0,
        beta=0.0,
        limit=None,
        act_type="mxfp8",
        activation="swiglu" if is_gated else "relu2",
        use_interleaved_layout=False,
    )

    # Shuffle weights/scales with the same indexed layout used by TRTLLM kernels.
    epilogue_tile_m = 128
    gemm1_weights_shuffled = []
    gemm1_scales_shuffled = []
    gemm2_weights_shuffled = []
    gemm2_scales_shuffled = []
    for i in range(num_experts):
        w13_rows = intermediate_size * (2 if is_gated else 1)
        w13_interleaved = w13_q[i].clone().reshape(w13_rows, -1)
        w13_scale_interleaved = w13_scale[i].clone().reshape(w13_rows, -1)
        if is_gated:
            w13_interleaved = reorder_rows_for_gated_act_gemm(w13_interleaved)
            w13_scale_interleaved = reorder_rows_for_gated_act_gemm(
                w13_scale_interleaved
            )
        gemm1_weights_shuffled.append(
            shuffle_matrix_a(w13_interleaved.view(torch.uint8), epilogue_tile_m)
            .contiguous()
            .view(w13_q.dtype)
        )
        gemm2_weights_shuffled.append(
            shuffle_matrix_a(w2_q[i].view(torch.uint8), epilogue_tile_m)
            .contiguous()
            .view(w2_q.dtype)
        )

        gemm1_scales_shuffled.append(
            shuffle_matrix_sf_a(
                w13_scale_interleaved.view(torch.uint8).reshape(w13_rows, -1),
                epilogue_tile_m,
            )
            .contiguous()
            .view(w13_scale.dtype)
        )
        gemm2_scales_shuffled.append(
            shuffle_matrix_sf_a(
                w2_scale[i].view(torch.uint8).reshape(hidden_size, -1), epilogue_tile_m
            )
            .contiguous()
            .view(w2_scale.dtype)
        )

    out = trtllm_fp8_block_scale_moe(
        routing_logits=router_logits_kernel,
        routing_bias=None,
        hidden_states=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=torch.stack(gemm1_weights_shuffled),
        gemm1_weights_scale=torch.stack(gemm1_scales_shuffled),
        gemm2_weights=torch.stack(gemm2_weights_shuffled),
        gemm2_weights_scale=torch.stack(gemm2_scales_shuffled),
        num_experts=num_experts,
        top_k=topk,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=1,  # renormalize routing
        use_shuffled_weight=True,
        weight_layout=0,  # MajorK
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
    )

    # Block-scale MXFP8 kernels are approximate; require majority close.
    check_accuracy(ref, out, atol=0.1, rtol=0.85, percent=0.8)


# -----------------------------------------------------------------------------
# ROCm Oracle-based kernel execution tests
# -----------------------------------------------------------------------------
def assert_bf16_bit_equal(reference: torch.Tensor, actual: torch.Tensor) -> None:
    """Require bit equality for the deliberately exact structured fixture."""
    assert reference.shape == actual.shape
    assert torch.isfinite(reference).all()
    assert torch.isfinite(actual).all()
    assert torch.equal(
        reference.contiguous().view(torch.uint16),
        actual.contiguous().view(torch.uint16),
    )


def _exact_moe_activation(
    stage1: torch.Tensor, activation_name: str
) -> torch.Tensor:
    """Evaluate a deliberately boundary-safe gated activation exactly.

    Every gate in this fixture is 8.  For SILU, the distance between
    ``8 * sigmoid(8) * up`` and ``8 * up`` is below half a BF16 ULP for the
    selected ``|up| <= 4`` values.  SWIGLUOAI clamps the gate to 7; with
    alpha=1.702 its sigmoid error is below half a BF16 ULP for the selected
    ``|up + 1| <= 6`` values.  The high-precision calculation below verifies
    those rounding claims before the kernel result is used as an oracle.
    """
    assert stage1.dtype == torch.bfloat16
    if activation_name == "SILU":
        gate, up = stage1.float().chunk(2, dim=-1)
        assert torch.all(gate == 8.0)
        assert torch.all(up.abs() <= 4.0)
        real = (gate.double() * torch.sigmoid(gate.double())) * up.double()
        expected = (8.0 * up).to(torch.bfloat16)
        actual = torch.empty_like(expected)
        torch.ops._C.silu_and_mul(actual, stage1)
    elif activation_name == "SWIGLUOAI":
        gate = stage1[..., 0::2].float()
        up = stage1[..., 1::2].float()
        assert torch.all(gate == 8.0)
        assert torch.all((up + 1.0).abs() <= 6.0)
        clamped_gate = torch.full_like(gate, 7.0, dtype=torch.float64)
        real = (up.double() + 1.0) * clamped_gate * torch.sigmoid(
            1.702 * clamped_gate
        )
        expected = (7.0 * (up + 1.0)).to(torch.bfloat16)
        actual = torch.empty_like(expected)
        torch.ops._C.swigluoai_and_mul(actual, stage1, 1.702, 7.0)
    else:
        raise AssertionError(f"Unsupported exact activation: {activation_name}")

    # This is the mathematical rounding proof for the selected fixture, and
    # the second assertion separately checks the production activation op.
    assert_bf16_bit_equal(real.to(torch.bfloat16), expected)
    assert_bf16_bit_equal(expected, actual)
    return expected


@pytest.mark.parametrize("activation_name", ["SILU", "SWIGLUOAI"])
@pytest.mark.skipif(not ROCM_AVAILABLE, reason="ROCm is required for this test")
@pytest.mark.skipif(
    not ROCM_TRITON_KERNELS_AVAILABLE,
    reason="triton_kernels is required for MXFP4 emulation",
)
@pytest.mark.skipif(
    not QUARK_MXFP4_TORCH_COMPATIBLE,
    reason="A compatible amd-quark installation is required",
)
@torch.inference_mode()
def test_rocm_mxfp4_moe_oracle_emulation_exact(
    activation_name: str, monkeypatch: pytest.MonkeyPatch
):
    """Bit-exact, nondegenerate oracle for Quark MXFP4 MoE emulation.

    The fixture uses one product per GEMM output, so no reduction-order
    tolerance is needed.  Eight tokens route one-to-one through all eight
    experts.  Inputs, weights, biases, and outputs span both signs and several
    magnitudes, and Quark's packed weights are decoded by the independent
    PyTorch reference implementation before the expected result is computed.
    """
    import vllm.distributed.parallel_state as ps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        backend_to_kernel_cls,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_kernel,
        make_mxfp4_moe_quant_config,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    num_tokens = num_experts = 8
    topk = 1
    hidden_size = intermediate_size = 256
    dtype = torch.bfloat16
    device = "cuda:0"

    init_workspace_manager(torch.accelerator.current_device_index())
    monkeypatch.setattr(ps, "_TP", types.SimpleNamespace(world_size=1))

    backend = Mxfp4MoeBackend.EMULATION
    experts_classes = backend_to_kernel_cls(backend)
    assert experts_classes is not None and len(experts_classes) == 1
    activation = MoEActivation[activation_name]
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=activation,
        in_dtype=dtype,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
    )

    # Each 32-value block has maximum 4 and contains only E2M1 values, so the
    # independent Quark-Even QDQ must preserve every input bit.
    input_palette = torch.tensor(
        [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        dtype=dtype,
        device=device,
    )
    token_axis = torch.arange(num_tokens, device=device).unsqueeze(1)
    hidden_axis = torch.arange(hidden_size, device=device).unsqueeze(0)
    hidden_states = input_palette[
        (5 * token_axis + 7 * hidden_axis) % input_palette.numel()
    ].contiguous()
    hidden_states_qdq = qdq_mxfp4_torch(hidden_states, "even")
    assert_bf16_bit_equal(hidden_states, hidden_states_qdq)

    w13 = torch.zeros(
        num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device
    )
    w13_bias = torch.empty(
        num_experts, 2 * intermediate_size, dtype=dtype, device=device
    )
    coefficient_palette = torch.tensor(
        [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],
        dtype=dtype,
        device=device,
    )
    silu_up_palette = torch.tensor(
        [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0], dtype=dtype, device=device
    )
    oai_up_palette = torch.tensor(
        [-5.0, -3.0, 0.0, 1.0, 3.0, 5.0], dtype=dtype, device=device
    )
    up_palette = silu_up_palette if activation_name == "SILU" else oai_up_palette
    dims = torch.arange(intermediate_size, device=device)

    # Set exactly one nonzero coefficient in every row.  Biases force the
    # pre-activation gate to 8 and the up branch to the boundary-safe palette.
    for expert in range(num_experts):
        gate_rows = dims if activation_name == "SILU" else 2 * dims
        up_rows = (
            intermediate_size + dims if activation_name == "SILU" else 2 * dims + 1
        )
        gate_sources = (3 * dims + 17 * expert) % hidden_size
        up_sources = (11 * dims + 19 * expert + 1) % hidden_size
        gate_coefficients = coefficient_palette[(dims + 3 * expert) % 8].clone()
        up_coefficients = coefficient_palette[(5 * dims + expert + 1) % 8].clone()

        gate_products = (
            hidden_states_qdq[expert, gate_sources] * gate_coefficients
        )
        gate_zero_bias = gate_products == 8.0
        gate_coefficients[gate_zero_bias] *= -1
        gate_products = (
            hidden_states_qdq[expert, gate_sources] * gate_coefficients
        )

        up_targets = up_palette[(dims + 2 * expert) % up_palette.numel()]
        up_products = hidden_states_qdq[expert, up_sources] * up_coefficients
        up_zero_bias = up_products == up_targets
        up_coefficients[up_zero_bias] *= -1
        up_products = hidden_states_qdq[expert, up_sources] * up_coefficients

        w13[expert, gate_rows, gate_sources] = gate_coefficients
        w13[expert, up_rows, up_sources] = up_coefficients
        w13_bias[expert, gate_rows] = 8.0 - gate_products
        w13_bias[expert, up_rows] = up_targets - up_products

    assert torch.all(w13_bias != 0)
    assert torch.any(w13_bias < 0) and torch.any(w13_bias > 0)
    assert torch.any(w13 < 0) and torch.any(w13 > 0)
    assert torch.unique(w13.abs()).numel() > 4

    w13_packed, w13_scale = quark_pack_mxfp4(w13)
    w13_decoded = dq_mxfp4_torch(
        w13_packed,
        w13_scale.view(torch.uint8).reshape(*w13_packed.shape[:-1], -1),
        torch.bfloat16,
    )
    assert_bf16_bit_equal(w13, w13_decoded)

    selected_w13 = w13_decoded[torch.arange(num_experts, device=device)]
    stage1 = (
        torch.einsum("tch,th->tc", selected_w13.float(), hidden_states_qdq.float())
        + w13_bias.float()
    ).to(dtype)
    if activation_name == "SILU":
        gate, up = stage1.chunk(2, dim=-1)
    else:
        gate, up = stage1[..., 0::2], stage1[..., 1::2]
    assert torch.all(gate == 8.0)
    expected_up = up_palette[
        (dims.unsqueeze(0) + 2 * torch.arange(num_experts, device=device).unsqueeze(1))
        % up_palette.numel()
    ]
    assert_bf16_bit_equal(expected_up, up)
    activated = _exact_moe_activation(stage1, activation_name)
    activated_qdq = qdq_mxfp4_torch(activated, "even")

    # Construct the second sparse GEMM after the activation QDQ is known.  Its
    # exact dyadic targets make every expert and output coordinate observable.
    w2 = torch.zeros(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
    )
    w2_bias = torch.empty(num_experts, hidden_size, dtype=dtype, device=device)
    output_palette = torch.tensor(
        [-64.0, -32.0, -16.0, -8.0, 8.0, 16.0, 32.0, 64.0],
        dtype=dtype,
        device=device,
    )
    for expert in range(num_experts):
        output_dims = torch.arange(hidden_size, device=device)
        sources = (13 * output_dims + 23 * expert) % intermediate_size
        coefficients = coefficient_palette[(7 * output_dims + expert + 2) % 8].clone()
        products = activated_qdq[expert, sources] * coefficients
        targets = output_palette[(output_dims + 3 * expert) % output_palette.numel()]
        zero_bias = products == targets
        coefficients[zero_bias] *= -1
        products = activated_qdq[expert, sources] * coefficients
        w2[expert, output_dims, sources] = coefficients
        w2_bias[expert] = targets - products

    assert torch.all(w2_bias != 0)
    assert torch.any(w2_bias < 0) and torch.any(w2_bias > 0)
    assert torch.any(w2 < 0) and torch.any(w2 > 0)
    assert torch.unique(w2.abs()).numel() > 4

    w2_packed, w2_scale = quark_pack_mxfp4(w2)
    w2_decoded = dq_mxfp4_torch(
        w2_packed,
        w2_scale.view(torch.uint8).reshape(*w2_packed.shape[:-1], -1),
        torch.bfloat16,
    )
    assert_bf16_bit_equal(w2, w2_decoded)

    selected_w2 = w2_decoded[torch.arange(num_experts, device=device)]
    reference = (
        torch.einsum("thk,tk->th", selected_w2.float(), activated_qdq.float())
        + w2_bias.float()
    ).to(dtype)
    expected_reference = output_palette[
        (
            torch.arange(hidden_size, device=device).unsqueeze(0)
            + 3 * torch.arange(num_experts, device=device).unsqueeze(1)
        )
        % output_palette.numel()
    ]
    assert_bf16_bit_equal(expected_reference, reference)
    assert torch.any(reference < 0) and torch.any(reference > 0)
    assert torch.unique(reference, dim=0).shape[0] == num_tokens

    class MockLayer:
        pass

    layer = MockLayer()
    layer.w13_weight = w13_packed
    layer.w2_weight = w2_packed
    layer.w13_weight_scale = w13_scale
    layer.w2_weight_scale = w2_scale
    layer.w13_input_scale = None
    layer.w2_input_scale = None
    w1, w2_converted, s1, s2, b1, b2 = (
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=backend,
            layer=layer,  # type: ignore[arg-type]
            w13_weight=w13_packed,
            w2_weight=w2_packed,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
        )
    )
    quant_config = make_mxfp4_moe_quant_config(
        mxfp4_backend=backend,
        w1_scale=s1,
        w2_scale=s2,
        w1_bias=b1,
        w2_bias=b2,
        gemm1_alpha=1.702 if activation_name == "SWIGLUOAI" else 1.0,
        gemm1_beta=1.0 if activation_name == "SWIGLUOAI" else 0.0,
        swiglu_limit=7.0 if activation_name == "SWIGLUOAI" else None,
    )
    assert quant_config is not None

    with set_current_vllm_config(VllmConfig()):
        kernel = make_mxfp4_moe_kernel(
            moe_quant_config=quant_config,
            moe_config=moe_config,
            mxfp4_backend=backend,
            experts_cls=experts_classes[0],
            routing_tables=None,
            layer=None,
        )
        assert not kernel.is_monolithic
        topk_ids = torch.arange(num_experts, device=device).view(-1, 1)
        topk_weights = torch.ones(
            num_tokens, topk, dtype=torch.float32, device=device
        )
        assert torch.equal(
            torch.unique(topk_ids), torch.arange(num_experts, device=device)
        )
        actual = kernel.apply(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2_converted,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=num_experts,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    assert_bf16_bit_equal(reference, actual)


ROCM_BACKEND_CONFIGS = {
    "TRITON": {
        "activation": "SWIGLUOAI",
        "requires_aiter": False,
        "requires_gfx950": False,
    },
    "TRITON_UNFUSED": {
        "activation": "SWIGLUOAI",
        "requires_aiter": False,
        "requires_gfx950": False,
    },
    "AITER_MXFP4_BF16": {
        "activation": "SWIGLUOAI",
        "requires_aiter": True,
        "requires_gfx950": True,
    },
    "AITER_MXFP4_FP8": {
        "activation": "SWIGLUOAI",
        "requires_aiter": True,
        "requires_gfx950": True,
    },
    "AITER_MXFP4_MXFP4": {
        "activation": "SILU",
        "requires_aiter": True,
        "requires_gfx950": True,
    },
}

ROCM_REFERENCE_MODES = [
    pytest.param("backend", id="backend-characterization"),
    pytest.param("quark_even", id="quark-even-contract"),
]


@pytest.mark.parametrize("backend_name", list(ROCM_BACKEND_CONFIGS.keys()))
@pytest.mark.parametrize("reference_mode", ROCM_REFERENCE_MODES)
@pytest.mark.parametrize(
    "routing_mode",
    [
        "uniform",
        "dyadic_0",
        "dyadic_1",
        "dyadic_2",
        "dyadic_3",
        "onehot_0",
        "onehot_1",
        "onehot_2",
        "onehot_3",
    ],
)
@pytest.mark.parametrize("topk", [4])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("num_tokens,hidden_size,intermediate_size", [(16, 256, 256)])
@pytest.mark.skipif(
    not ROCM_AVAILABLE,
    reason="ROCm is required for this test",
)
@torch.inference_mode()
def test_rocm_mxfp4_moe_oracle(
    backend_name: str,
    reference_mode: str,
    routing_mode: str,
    topk: int,
    num_experts: int,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Characterize ROCm MXFP4 MoE oracle backends and the Quark-Even contract.

    This test validates that the oracle functions work end-to-end:
    - select_mxfp4_moe_backend() selects a valid backend
    - convert_gpt_oss_weight_to_mxfp4_moe_kernel_format() converts weights without error
    - make_mxfp4_moe_quant_config() builds a valid quant config
    - make_mxfp4_moe_kernel() creates a kernel that runs without error
    - The kernel output is within accuracy tolerance of reference
    """
    config = ROCM_BACKEND_CONFIGS[backend_name]
    backend_enum_name = config.get("backend", backend_name)
    is_emulation = backend_enum_name == "EMULATION"

    if reference_mode == "quark_even" and backend_enum_name != "AITER_MXFP4_MXFP4":
        pytest.skip("The separate Quark-Even diagnostic applies to native W4A4 only")
    if reference_mode == "quark_even" and routing_mode != "uniform":
        pytest.skip(
            "One uniform-routing case is sufficient for the rounding diagnostic"
        )
    if routing_mode.startswith("dyadic_") and (
        backend_enum_name != "AITER_MXFP4_MXFP4" or reference_mode != "backend"
    ):
        pytest.skip("Explicit dyadic routing is a modular AITER W4A4 contract test")

    # Check platform requirements
    if not ROCM_TRITON_KERNELS_AVAILABLE:
        pytest.skip("triton_kernels required for quantization")
    if config["requires_aiter"] and not ROCM_AITER_AVAILABLE:
        pytest.skip(f"Backend {backend_name} requires AITER")
    if config["requires_gfx950"] and not ROCM_GFX950:
        pytest.skip(f"Backend {backend_name} requires GFX950")

    import vllm.distributed.parallel_state as ps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        backend_to_kernel_cls,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_kernel,
        make_mxfp4_moe_quant_config,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    # Initialize workspace manager (needed for modular kernels)
    init_workspace_manager(torch.accelerator.current_device_index())

    # Set up the TP Group to prevent failure on should_use_cdna4_mx_scale_swizzle check
    monkeypatch.setattr(ps, "_TP", types.SimpleNamespace(world_size=1))

    # AITER must be enabled or aiter_mxfp4_w4a8_moe asserts before dispatch.
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)

    # Map string to enum
    backend = Mxfp4MoeBackend[backend_enum_name]

    # Get experts class from oracle
    experts_cls_list = backend_to_kernel_cls(backend)
    if experts_cls_list is None or len(experts_cls_list) == 0:
        pytest.skip(f"Backend {backend_name} not available")

    # Use first experts class
    experts_cls = experts_cls_list[0]

    torch.manual_seed(42)
    dtype = torch.bfloat16
    device = "cuda:0"

    # Create MoE config with Renormalize routing (required by monolithic kernels)
    from vllm.model_executor.layers.fused_moe import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation[config["activation"]],
        in_dtype=dtype,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
    )

    # Use a sparse, exactly representable problem for every backend. Each row
    # has one product, eliminating reduction-order ambiguity while still
    # varying signs, magnitudes, source dimensions, and expert layouts.
    assert hidden_size == intermediate_size
    w13_float = torch.zeros(
        num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device
    )
    w2_float = torch.zeros(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device=device
    )
    diagonal = torch.arange(hidden_size, device=device)
    coefficient_palette = torch.tensor(
        [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],
        dtype=dtype,
        device=device,
    )
    for expert in range(num_experts):
        source = (diagonal + 17 * expert) % hidden_size
        gate_coefficient = coefficient_palette[(diagonal + expert) % 8]
        up_coefficient = coefficient_palette[(3 * diagonal + 5 * expert + 1) % 8]
        if config["activation"] == "SWIGLUOAI":
            gate_rows = 2 * diagonal
            up_rows = gate_rows + 1
            # A zero gate weight plus the exact bias below fixes the gate at 6.
            # BF16 sigmoid(1.702 * 6) rounds to one, so the activation becomes
            # exact dyadic arithmetic while the up path remains fully varied.
            gate_coefficient = torch.zeros_like(gate_coefficient)
        else:
            gate_rows = diagonal
            up_rows = intermediate_size + diagonal
        w13_float[expert, gate_rows, source] = gate_coefficient
        w13_float[expert, up_rows, source] = up_coefficient

        # Experts have disjoint output supports.  Therefore a multi-expert
        # top-k reduction has at most one nonzero addend per coordinate and is
        # bit-exact even for backends which use BF16 atomics in an unspecified
        # order.  The separate emulation oracle above supplies dense bias and
        # signed/magnitude coverage.
        supported = diagonal % num_experts == expert
        output_rows = diagonal[supported]
        output_source = (5 * output_rows + 11 * expert) % intermediate_size
        output_coefficient = coefficient_palette[
            (7 * output_rows + expert + 2) % 8
        ]
        w2_float[expert, output_rows, output_source] = output_coefficient

    if is_emulation:
        # Emulation consumes the actual Quark checkpoint byte/scale layout.
        # AITER's dynamic quantizer is not layout-equivalent and would make
        # this test exercise invalid inputs.
        w13_quant, w13_scale = quark_pack_mxfp4(w13_float)
        w2_quant, w2_scale = quark_pack_mxfp4(w2_float)
    else:
        # Native AITER conversions consume AITER's packed source layout.
        # w13: [E, 2*I, H] -> [E*2*I, H] -> [E, 2*I, H//2]
        w13_2d = w13_float.reshape(-1, hidden_size)
        w13_quant_2d, w13_scale_2d = dynamic_mxfp4_quant(w13_2d)
        w13_quant = w13_quant_2d.reshape(num_experts, 2 * intermediate_size, -1)
        w13_scale = w13_scale_2d.reshape(num_experts, 2 * intermediate_size, -1)

        w2_2d = w2_float.reshape(-1, intermediate_size)
        w2_quant_2d, w2_scale_2d = dynamic_mxfp4_quant(w2_2d)
        w2_quant = w2_quant_2d.reshape(num_experts, hidden_size, -1)
        w2_scale = w2_scale_2d.reshape(num_experts, hidden_size, -1)

    expert_axis = torch.arange(num_experts, device=device).unsqueeze(1)
    w13_axis = torch.arange(2 * intermediate_size, device=device).unsqueeze(0)
    w2_axis = torch.arange(hidden_size, device=device).unsqueeze(0)
    w13_bias = (((3 * expert_axis + w13_axis) % 7) - 3).to(dtype) * 0.25
    if config["activation"] == "SWIGLUOAI":
        w13_bias[:, 0::2] = 6.0
    w2_bias = torch.zeros(
        num_experts, hidden_size, dtype=dtype, device=device
    )
    w2_bias_values = (
        64.0 + (((5 * expert_axis + 2 * w2_axis) % 9) - 4).to(dtype) * 0.25
    )
    w2_support = w2_axis % num_experts == expert_axis
    w2_bias[w2_support] = w2_bias_values[w2_support]
    if backend_name == "AITER_MXFP4_MXFP4":
        # The W4A4 AITER kernel does not accept expert biases.
        w13_bias.zero_()
        w2_bias.zero_()

    # Create static input scales for W4A8 backend (AITER_MXFP4_FP8)
    w13_input_scale: torch.Tensor | None = None
    w2_input_scale: torch.Tensor | None = None
    if backend_name == "AITER_MXFP4_FP8":
        # Static FP8 scales: one scale per expert
        # Non-unit powers of two exercise the QDQ scaling contract exactly;
        # the second-stage scale also exercises saturating FP8 conversion.
        w13_input_scale = torch.full(
            (num_experts,), 0.125, dtype=torch.float32, device=device
        )
        w2_input_scale = torch.full(
            (num_experts,), 2.0, dtype=torch.float32, device=device
        )

    # Create mock layer for oracle functions
    class MockLayer:
        w13_weight: torch.Tensor
        w2_weight: torch.Tensor
        w13_weight_scale: torch.Tensor
        w2_weight_scale: torch.Tensor
        w13_input_scale: torch.Tensor | None
        w2_input_scale: torch.Tensor | None

    layer = MockLayer()
    layer.w13_weight = w13_quant
    layer.w2_weight = w2_quant
    layer.w13_weight_scale = w13_scale
    layer.w2_weight_scale = w2_scale
    layer.w13_input_scale = w13_input_scale
    layer.w2_input_scale = w2_input_scale

    # Conversion is in-place for several backends. Preserve checkpoint-layout
    # tensors for the independent dequantized reference.
    w13_quant_ref = w13_quant.clone()
    w2_quant_ref = w2_quant.clone()
    w13_scale_ref = w13_scale.clone()
    w2_scale_ref = w2_scale.clone()
    w13_bias_ref = w13_bias.clone()
    w2_bias_ref = w2_bias.clone()

    # Convert weights using oracle
    w13_conv, w2_conv, w13_scale_conv, w2_scale_conv, w13_bias_conv, w2_bias_conv = (
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=backend,
            layer=layer,  # type: ignore[arg-type]
            w13_weight=w13_quant,
            w2_weight=w2_quant,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
        )
    )

    # Build quant config using oracle
    quant_config = make_mxfp4_moe_quant_config(
        mxfp4_backend=backend,
        w1_scale=w13_scale_conv,
        w2_scale=w2_scale_conv,
        w1_bias=w13_bias_conv,
        w2_bias=w2_bias_conv,
        a1_scale=w13_input_scale,
        a2_scale=w2_input_scale,
    )

    # Select activation based on backend
    activation_name = str(config["activation"])
    activation = MoEActivation[activation_name]

    # Build kernel using oracle
    assert quant_config is not None, "Failed to create quant config"
    with set_current_vllm_config(VllmConfig()):
        kernel = make_mxfp4_moe_kernel(
            moe_quant_config=quant_config,
            moe_config=moe_config,
            mxfp4_backend=backend,
            experts_cls=experts_cls,
            routing_tables=None,
            layer=None,
        )

        # Every token has a different signed/magnitude pattern. Values are
        # exactly representable in BF16, MXFP4, and the configured static FP8
        # scales, so the test isolates kernel/layout semantics from input QDQ.
        input_palette = torch.tensor(
            [
                -6.0,
                -4.0,
                -3.0,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
            ],
            dtype=dtype,
            device=device,
        )
        token_axis = torch.arange(num_tokens, device=device).unsqueeze(1)
        hidden_axis = torch.arange(hidden_size, device=device).unsqueeze(0)
        x = input_palette[(5 * token_axis + 3 * hidden_axis) % len(input_palette)]
        x = x.contiguous()

        # Route each token to a rotating set of four experts. Equal logits give
        # the exactly representable probability 1/4, while the distinct expert
        # weights and biases make every route observable in the output.
        router_logits = torch.full(
            (num_tokens, num_experts),
            -1000.0,
            dtype=torch.float32,
            device=device,
        )
        route_offsets = torch.tensor([0, 1, 3, 5], device=device)
        route_ids = (
            torch.arange(num_tokens, device=device).unsqueeze(1) + route_offsets
        ) % num_experts
        router_logits.scatter_(1, route_ids, 0.0)
        if routing_mode.startswith("onehot_"):
            dominant_position = int(routing_mode.removeprefix("onehot_"))
            # -200 is above the non-routed -1000 logits but exp(-200)
            # underflows in FP32, yielding an exact one-hot softmax.
            router_logits.scatter_(1, route_ids, -200.0)
            router_logits.scatter_(
                1, route_ids[:, dominant_position : dominant_position + 1], 0.0
            )
        topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1, sorted=True)
        topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)
        if routing_mode == "uniform":
            expected_weights = torch.full_like(topk_weights, 0.25)
        elif routing_mode.startswith("dyadic_"):
            # All weights are nonzero and exactly representable.  Moving the
            # largest weight through every top-k slot exercises slot handling
            # without triggering AITER's separately diagnosed zero-route bug.
            largest_position = int(routing_mode.removeprefix("dyadic_"))
            base_weights = torch.tensor(
                [0.5, 0.25, 0.125, 0.125],
                dtype=torch.float32,
                device=device,
            )
            expected_weights = torch.roll(
                base_weights, shifts=largest_position
            ).expand(num_tokens, -1)
            topk_weights = expected_weights.clone()
            topk_ids = route_ids.clone()
        else:
            expected_weights = torch.nn.functional.one_hot(
                torch.zeros(num_tokens, dtype=torch.long, device=device),
                num_classes=topk,
            ).to(topk_weights.dtype)
        assert torch.equal(topk_weights, expected_weights)
        assert torch.equal(
            torch.sort(topk_ids, dim=1).values, torch.sort(route_ids, dim=1).values
        )

        zero_route_variant_out: torch.Tensor | None = None
        # Run kernel - use appropriate method based on impl type
        if kernel.is_monolithic:
            assert not routing_mode.startswith("dyadic_")
            # Monolithic impl uses router_logits
            out = kernel.apply_monolithic(
                hidden_states=x,
                w1=w13_conv,
                w2=w2_conv,
                router_logits=router_logits,
                activation=activation,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )
        else:
            # Modular impl uses topk_weights and topk_ids
            out = kernel.apply(
                hidden_states=x,
                w1=w13_conv,
                w2=w2_conv,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=num_experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )
            if (
                backend_enum_name == "AITER_MXFP4_MXFP4"
                and reference_mode == "backend"
                and routing_mode.startswith("onehot_")
            ):
                zero_mask = topk_weights == 0
                slot_offsets = torch.arange(topk, device=device).view(1, -1) + 1
                changed_ids = torch.where(
                    zero_mask,
                    (topk_ids + slot_offsets) % num_experts,
                    topk_ids,
                )
                assert torch.equal(changed_ids[~zero_mask], topk_ids[~zero_mask])
                assert not torch.equal(changed_ids[zero_mask], topk_ids[zero_mask])
                zero_route_variant_out = kernel.apply(
                    hidden_states=x,
                    w1=w13_conv,
                    w2=w2_conv,
                    topk_weights=topk_weights,
                    topk_ids=changed_ids,
                    activation=activation,
                    global_num_experts=num_experts,
                    expert_map=None,
                    apply_router_weight_on_input=False,
                )

    # Verify output is valid (no NaN/Inf) and has expected shape
    assert out.shape == (num_tokens, hidden_size), f"Unexpected shape: {out.shape}"
    assert not torch.any(torch.isnan(out)), "Output contains NaN"
    assert not torch.any(torch.isinf(out)), "Output contains Inf"
    if zero_route_variant_out is not None:
        assert zero_route_variant_out.shape == out.shape
        assert torch.isfinite(zero_route_variant_out).all()

    # Verify output has reasonable magnitude (not all zeros)
    assert out.abs().max() > 0.01, "Output is effectively zero"

    # Dequantize weights for reference computation
    if is_emulation:
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            dequant_mxfp4,
        )

        w13_dq = dequant_mxfp4(w13_quant_ref, w13_scale_ref, torch.bfloat16)
        w2_dq = dequant_mxfp4(w2_quant_ref, w2_scale_ref, torch.bfloat16)
    else:
        w13_dq = upcast_from_mxfp(
            w13_quant_ref.view(torch.uint8),
            w13_scale_ref,
            torch.bfloat16,
            axis=-1,
        )
        w2_dq = upcast_from_mxfp(
            w2_quant_ref.view(torch.uint8), w2_scale_ref, torch.bfloat16, axis=-1
        )

    # Determine activation type and layout
    # SWIGLUOAI uses interleaved layout (gate/up alternating)
    # SILU uses chunked layout (first half gate, second half up)
    use_interleaved = activation == MoEActivation.SWIGLUOAI
    if activation in [MoEActivation.SWIGLUOAI, MoEActivation.SILU]:
        act_name = "swiglu"
    else:
        act_name = "relu2"

    def make_reference(act_type: str):
        return reference_moe(
            router_logits,
            topk,
            num_experts,
            x.to(torch.float32),
            w13_dq.to(torch.float32),
            w13_bias_ref.to(torch.float32),
            w2_dq.to(torch.float32),
            w2_bias_ref.to(torch.float32),
            alpha=1.702 if activation == MoEActivation.SWIGLUOAI else 1.0,
            beta=1.0 if activation == MoEActivation.SWIGLUOAI else 0.0,
            limit=7.0 if activation == MoEActivation.SWIGLUOAI else None,
            act_type=act_type,
            activation=act_name,
            use_interleaved_layout=use_interleaved,
            input_scale1=w13_input_scale,
            input_scale2=w2_input_scale,
            expert_weights_override=topk_weights,
            expert_indices_override=topk_ids,
        )

    reference_act_type = (
        ("mxfp4_even" if reference_mode == "quark_even" else "mxfp4_roundup")
        if backend_enum_name == "AITER_MXFP4_MXFP4"
        else (
            "fp8_static"
            if backend_enum_name == "AITER_MXFP4_FP8"
            else (
                "mxfp4_even_emulation"
                if is_emulation
                else (
                    "bf16_intermediate"
                    if backend_name == "AITER_MXFP4_BF16"
                    else "bf16_pipeline"
                )
            )
        )
    )
    ref = make_reference(reference_act_type)
    assert isinstance(ref, torch.Tensor)

    # Compute and print accuracy statistics
    diff = (ref.float() - out.float()).abs()
    bit_mismatches = (
        ref.contiguous().view(torch.uint16) != out.contiguous().view(torch.uint16)
    ).sum()

    print(f"\n[{backend_name}] Accuracy statistics:")
    print(
        f"  Reference: min={ref.min():.4f}, max={ref.max():.4f}, mean={ref.mean():.4f}"
    )
    print(
        f"  Output:    min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}"
    )
    print(
        f"  Abs diff:  min={diff.min():.4f}, max={diff.max():.4f}, "
        f"mean={diff.mean():.4f}"
    )
    print(f"  BF16 bit mismatches: {bit_mismatches}/{ref.numel()}")

    if reference_mode == "quark_even":
        # Do not mark the whole test xfail: setup, conversion, execution,
        # shape, and finiteness failures above must remain hard failures. If
        # AITER gains Even mode, this branch passes normally. Until then, only
        # an output that satisfies the RoundUp oracle and violates the Even
        # oracle is classified as the known limitation.
        if bit_mismatches == 0:
            return
        roundup_ref = make_reference("mxfp4_roundup")
        assert isinstance(roundup_ref, torch.Tensor)
        assert_bf16_bit_equal(roundup_ref, out)
        assert not torch.equal(
            ref.contiguous().view(torch.uint16),
            roundup_ref.contiguous().view(torch.uint16),
        )
        pytest.xfail(
            "AITER fused W4A4 matches RoundUp, but exposes no Quark-Even "
            "activation-scale rounding mode"
        )

    if (
        zero_route_variant_out is not None
        and not torch.equal(
            out.contiguous().view(torch.uint16),
            zero_route_variant_out.contiguous().view(torch.uint16),
        )
    ):
        pytest.xfail(
            "AITER fused W4A4 lets exact-zero top-k slots affect the "
            "result; changing only their expert ids changes BF16 output"
        )
        # A deterministic but incorrect result is not the known zero-route
        # signature and must remain a hard failure.

    assert_bf16_bit_equal(ref, out)
