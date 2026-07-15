# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm GFX950 quantized-MoE initialization and dispatch coverage.

The full-model tests exercise public compressed-tensors and Quark MoE models;
the focused oracle tests verify MXFP4 backend selection. All tests require real
MI3xx hardware, with the native AITER dispatch cases further restricted to
GFX950.
"""

import importlib.metadata
import importlib.util
import os
from typing import Any

import huggingface_hub
import pytest
import torch
from packaging import version

from tests.utils import RemoteOpenAIServer
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.platforms import current_platform


def on_mi3xx() -> bool:
    if not current_platform.is_rocm():
        return False

    from vllm.platforms.rocm import on_mi3xx as rocm_on_mi3xx

    return rocm_on_mi3xx()


pytestmark = pytest.mark.skipif(not on_mi3xx(), reason="MI300/MI350 ROCm only")

_TORCH_RELEASE = version.parse(torch.__version__).release
QUARK_MXFP4_MIN_VERSION = "0.12.0" if _TORCH_RELEASE[:2] >= (2, 11) else "0.9.0"


def _has_quark_mxfp4_support() -> bool:
    if importlib.util.find_spec("quark") is None:
        return False
    try:
        quark_version = version.parse(importlib.metadata.version("amd-quark"))
        return quark_version >= version.parse(QUARK_MXFP4_MIN_VERSION)
    except importlib.metadata.PackageNotFoundError:
        return False


QUARK_MXFP4_AVAILABLE = _has_quark_mxfp4_support()
QUARK_AVAILABLE = importlib.util.find_spec("quark") is not None

HF_OVERRIDE_TEXT = {
    "num_layers": 4,
    "num_hidden_layers": 4,
}

ROCM_AVAILABLE = current_platform.is_rocm()
ROCM_GFX950 = False
ROCM_AITER_CAPABLE = False

if ROCM_AVAILABLE:
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_CAPABLE = is_aiter_found_and_supported()


NO_AITER_ENV = {
    "VLLM_ROCM_USE_AITER": "0",
    "VLLM_ROCM_USE_AITER_MOE": "0",
}
AITER_MOE_ENV = {
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MOE": "1",
}

REQUIRES_GFX950 = pytest.mark.skipif(not ROCM_GFX950, reason="Requires GFX950 (mi355x)")
REQUIRES_AITER = pytest.mark.skipif(
    not ROCM_AITER_CAPABLE, reason="Requires AITER support"
)
ROCM_ATTENTION_BACKENDS = [
    pytest.param("ROCM_ATTN", id="rocm_attn"),
    pytest.param(
        "ROCM_AITER_UNIFIED_ATTN",
        marks=[REQUIRES_GFX950, REQUIRES_AITER],
        id="rocm_aiter_unified_attn",
    ),
]


@pytest.fixture(autouse=True)
def aiter_moe_control(monkeypatch):
    """Start every test with deterministic AITER MoE gates.

    The in-process oracle reads cached gates, while model tests launch fresh
    subprocesses with one of the explicit environment dictionaries above.
    """

    def set_enabled(enabled: bool) -> None:
        if not ROCM_AVAILABLE:
            return
        value = "1" if enabled else "0"
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", value)
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", value)
        rocm_aiter_ops.refresh_env_variables()

    set_enabled(False)
    yield set_enabled

    if ROCM_AVAILABLE:
        monkeypatch.undo()
        rocm_aiter_ops.refresh_env_variables()


def _has_huggingface_access(repo_id: str) -> bool:
    try:
        huggingface_hub.list_repo_refs(repo_id)
        return True
    except (
        huggingface_hub.errors.HfHubHTTPError,
        huggingface_hub.errors.RepositoryNotFoundError,
    ):
        return False


def _require_repo_access(repo_id: str) -> None:
    if not _has_huggingface_access(repo_id):
        message = f"Read access to huggingface.co/{repo_id} is required."
        if os.getenv("CI") or os.getenv("BUILDKITE"):
            pytest.fail(message)
        pytest.skip(message)


def _can_initialize(
    model: str,
    *,
    hf_overrides: dict[str, Any] | None = None,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    server_args = [
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "256",
        "--max-num-seqs",
        "1",
        "--load-format",
        "dummy",
        "--enforce-eager",
        "--disable-uvicorn-access-log",
        *(extra_args or []),
    ]

    with RemoteOpenAIServer(
        model,
        server_args,
        env_dict=env,
        max_wait_seconds=1500,
        override_hf_configs=hf_overrides,
    ) as server:
        completion = server.get_client().completions.create(
            model=model,
            prompt=["Hello, World!"],
            temperature=0,
            max_tokens=2,
        )
        print(completion)
        assert len(completion.choices) == 1
        assert completion.choices[0].finish_reason is not None
        assert completion.usage is not None
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.completion_tokens > 0


def _make_w4a4_moe_config(moe_backend: str = "auto") -> FusedMoEConfig:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


@pytest.fixture
def mxfp4_oracle_config():
    """Stub the config the oracle reads (``model_config.quantization_config``)
    so backend dispatch resolves without a real model / user override."""
    from unittest.mock import patch

    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.get_current_vllm_config"
    ) as mock_get_config:
        mock_get_config.return_value.model_config.quantization_config = None
        yield


@REQUIRES_GFX950
@REQUIRES_AITER
def test_w4a4_dispatches_to_aiter(mxfp4_oracle_config, aiter_moe_control):
    """With AITER enabled + GFX950, W4A4 selects AITER_MXFP4_MXFP4."""
    aiter_moe_control(True)
    config = _make_w4a4_moe_config()
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    assert experts_cls is not None


@REQUIRES_GFX950
def test_w4a4_falls_back_to_emulation_without_aiter(mxfp4_oracle_config):
    """Without AITER and no --moe-backend, W4A4 selects emulation."""
    config = _make_w4a4_moe_config()
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.EMULATION
    assert experts_cls is not None


@REQUIRES_GFX950
def test_w4a4_dispatches_to_emulation_with_moe_backend(mxfp4_oracle_config):
    """With --moe-backend emulation, W4A4 selects EMULATION."""
    config = _make_w4a4_moe_config(moe_backend="emulation")
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.EMULATION
    assert experts_cls is not None


@pytest.mark.parametrize("attention_backend", ROCM_ATTENTION_BACKENDS)
def test_nm_qwen15_w4a16_moe_initializes_across_rocm_attention_backends(
    attention_backend: str,
):
    """Initialize public Qwen W4A16 MoE with both ROCm attention backends."""
    repo_id = "nm-testing/Qwen1.5-MoE-A2.7B-Chat-quantized.w4a16"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=["--attention-backend", attention_backend],
        # Explicit attention backend selection does not require the AITER
        # master switch. Keep AITER MoE disabled so this matrix isolates the
        # attention backend instead of changing two subsystems at once.
        env=NO_AITER_ENV,
    )


def test_nm_mixtral_w4a16_moe_initializes():
    """Initialize a second public compressed-tensors MoE family on ROCm."""
    repo_id = "nm-testing/Mixtral-8x7B-Instruct-v0.1-W4A16-quantized"
    _require_repo_access(repo_id)
    _can_initialize(repo_id, hf_overrides=HF_OVERRIDE_TEXT, env=NO_AITER_ENV)


@pytest.mark.skipif(
    not QUARK_AVAILABLE,
    reason="quark package is required for ROCm Quark MoE tests",
)
def test_tiny_quark_int8_moe_initializes():
    """Initialize a small public Quark INT8 MoE model on MI3xx."""
    repo_id = "nameistoken/tiny-qwen3-moe-w8a8-int8-quark"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
        env=NO_AITER_ENV,
    )


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=(
        f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is required for ROCm MXFP4 MoE tests"
    ),
)
@pytest.mark.parametrize(
    "moe_backend",
    [
        pytest.param("aiter", marks=[REQUIRES_GFX950, REQUIRES_AITER], id="aiter"),
        pytest.param("triton", id="triton"),
    ],
)
def test_gptoss_rocm_quark_mxfp4_bf16_moe_backends_initialize(
    moe_backend: str,
):
    """Initialize GPT-OSS Quark MXFP4/BF16 with each ROCm MoE backend."""
    repo_id = "amd/gpt-oss-20b-w-mxfp4-a-bf16"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=[
            "--attention-backend",
            "ROCM_AITER_UNIFIED_ATTN",
            "--moe-backend",
            moe_backend,
            "--tokenizer",
            "openai/gpt-oss-20b",
            "--tensor-parallel-size",
            "1",
        ],
        env=AITER_MOE_ENV if moe_backend == "aiter" else NO_AITER_ENV,
    )


@pytest.mark.skipif(
    not current_platform.supports_fp8(),
    reason="FP8 not supported on this hardware",
)
@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=(
        f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is required for ROCm MXFP4 MoE tests"
    ),
)
@REQUIRES_GFX950
@REQUIRES_AITER
def test_gptoss_rocm_quark_mxfp4_fp8_moe_initializes():
    """Initialize GPT-OSS Quark MXFP4/FP8 with ROCm AITER enabled."""
    repo_id = "amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=[
            "--attention-backend",
            "ROCM_AITER_UNIFIED_ATTN",
            "--moe-backend",
            "aiter",
            "--tokenizer",
            "openai/gpt-oss-20b",
            "--tensor-parallel-size",
            "1",
        ],
        env=AITER_MOE_ENV,
    )


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=(
        f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is required for ROCm MXFP4 MoE tests"
    ),
)
@pytest.mark.parametrize(
    "moe_backend",
    [
        pytest.param(None, id="auto"),
        pytest.param("aiter", marks=[REQUIRES_GFX950, REQUIRES_AITER], id="aiter"),
        pytest.param("emulation", id="emulation"),
    ],
)
def test_deepseek_rocm_quark_mxfp4_uint8_moe_backends_initialize(
    moe_backend: str | None,
):
    """Initialize DeepSeek Quark MXFP4/UINT8 across ROCm MoE backends."""
    repo_id = "amd/DeepSeek-R1-WMXFP4-AMXFP4-Scale-UINT8-MoE-Quant"
    _require_repo_access(repo_id)
    _can_initialize(
        repo_id,
        hf_overrides=HF_OVERRIDE_TEXT,
        extra_args=[
            "--tensor-parallel-size",
            "1",
            *([] if moe_backend is None else ["--moe-backend", moe_backend]),
        ],
        env=AITER_MOE_ENV if moe_backend == "aiter" else NO_AITER_ENV,
    )
