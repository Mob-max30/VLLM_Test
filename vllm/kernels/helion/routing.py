# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph-aware routing for compiled Helion kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

logger = init_logger(__name__)

# These ops are emitted only by vLLM's post-grad fusion passes. The remaining
# registered Helion kernels either have eager call sites or an incompatible
# schema and need a different routing path.
_FUSION_OP_NAMES = (
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_block_quant",
    "silu_and_mul_per_block_quant",
    "fused_qk_norm_rope",
)

# TODO: Enable after the known B200 numerical mismatch is resolved.
_UNSUPPORTED_PLATFORMS = {
    "fused_qk_norm_rope": frozenset({"nvidia_b200"}),
}


def _schema_tail(op: torch._ops.OpOverload) -> str:
    schema = str(op._schema)
    return schema[schema.index("(") :]


def _mutation_signature(op: torch._ops.OpOverload) -> tuple[tuple[str, bool], ...]:
    return tuple(
        (arg.name, bool(arg.alias_info and arg.alias_info.is_write))
        for arg in op._schema.arguments
    )


def _make_routed_impl(
    native_op: torch._ops.OpOverload,
    helion_op: torch._ops.OpOverload,
) -> Callable[..., Any]:
    schema_args = list(helion_op._schema.arguments)
    names = [arg.name for arg in schema_args]
    defaults = {
        arg.name: arg.default_value for arg in schema_args if arg.has_default_value()
    }

    def impl(*args: object, **kwargs: object) -> Any:
        values = list(args)
        for name in names[len(args) :]:
            values.append(kwargs[name] if name in kwargs else defaults[name])
        if torch.cuda.is_current_stream_capturing():
            return helion_op(*values)
        return native_op(*values)

    return impl


def build_compiled_helion_op_map() -> dict[
    torch._ops.OpOverload, torch._ops.OpOverload
]:
    """Return native-to-routed mappings for compatible fusion-only ops."""
    from vllm.kernels.helion.ops import import_all_kernels
    from vllm.kernels.helion.register import _HOP_AVAILABLE, vllm_helion_lib
    from vllm.kernels.helion.utils import get_canonical_gpu_name

    if _HOP_AVAILABLE:
        return {}

    import_all_kernels()
    platform = get_canonical_gpu_name()
    routed: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}

    for name in _FUSION_OP_NAMES:
        if platform in _UNSUPPORTED_PLATFORMS.get(name, ()):
            logger.warning(
                "Skipping compiled Helion routing for '%s' on '%s' because "
                "correctness is not established",
                name,
                platform,
            )
            continue

        native_packet = getattr(torch.ops._C, name, None)
        helion_packet = getattr(torch.ops.vllm_helion, name, None)
        if native_packet is None or helion_packet is None:
            continue

        native_op = native_packet.default
        helion_op = helion_packet.default
        if _mutation_signature(native_op) != _mutation_signature(helion_op):
            logger.warning(
                "Skipping compiled Helion routing for '%s': incompatible schemas "
                "(native=%s, helion=%s)",
                name,
                native_op._schema,
                helion_op._schema,
            )
            continue

        routed_name = f"routed_{name}"
        if not hasattr(torch.ops.vllm_helion, routed_name):
            vllm_helion_lib.define(routed_name + _schema_tail(helion_op))
            vllm_helion_lib.impl(
                routed_name,
                _make_routed_impl(native_op, helion_op),
                "CUDA",
            )
            vllm_helion_lib._register_fake(routed_name, lambda *args, **kwargs: None)

        routed[native_op] = getattr(torch.ops.vllm_helion, routed_name).default

    return routed


def register_routed_helion_ops() -> None:
    """Eagerly define the routed Helion ops (idempotent, self-gating).

    ``build_compiled_helion_op_map`` defines the ``vllm_helion.routed_*`` ops as
    a side effect, but it is only reached when ``HelionFusionRoutingPass`` runs
    at compile time. On a torch.compile cache hit the pass never runs, yet the
    cached graph still references the routed ops, so they must already exist in
    the process. Call this once during engine init (before the first compiled
    forward) so the ops resolve regardless of compile-cache state.
    """
    if not (envs.VLLM_USE_HELION_KERNELS and has_helion()):
        return
    build_compiled_helion_op_map()
