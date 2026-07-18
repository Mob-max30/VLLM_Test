# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch

from vllm.kernels.helion.ops import import_all_kernels
from vllm.kernels.helion.register import get_registered_kernels
from vllm.kernels.helion.routing import build_compiled_helion_op_map
from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip("Helion is not installed", allow_module_level=True)


@pytest.mark.parametrize(
    "name",
    [
        "rms_norm_dynamic_per_token_quant",
        "rms_norm_per_block_quant",
        "silu_and_mul_per_block_quant",
        "fused_qk_norm_rope",
    ],
)
def test_compiled_route_uses_native_then_captures_helion(name: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import_all_kernels()
    op_map = build_compiled_helion_op_map()
    native_op = getattr(torch.ops._C, name).default
    if native_op not in op_map:
        pytest.skip(f"{name} is not supported on this platform")

    args = list(next(iter(get_registered_kernels()[name].get_inputs().values())))
    if name == "silu_and_mul_per_block_quant":
        # This is the path emitted by ActivationQuantFusionPass.
        args[4] = None

    expected_args = copy.deepcopy(args)
    fallback_args = copy.deepcopy(args)
    captured_args = copy.deepcopy(args)
    routed_op = op_map[native_op]

    native_op(*expected_args)
    routed_op(*fallback_args)
    for index, schema_arg in enumerate(native_op._schema.arguments):
        if schema_arg.alias_info and schema_arg.alias_info.is_write:
            torch.testing.assert_close(fallback_args[index], expected_args[index])

    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routed_op(*captured_args)
    graph.replay()
    torch.accelerator.synchronize()

    for index, schema_arg in enumerate(native_op._schema.arguments):
        if schema_arg.alias_info and schema_arg.alias_info.is_write:
            torch.testing.assert_close(
                captured_args[index].float(),
                expected_args[index].float(),
                rtol=0.1,
                atol=0.1,
            )
