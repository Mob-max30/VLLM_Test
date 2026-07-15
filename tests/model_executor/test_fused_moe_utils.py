# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket


@pytest.mark.parametrize(
    ("max_num_tokens", "dp_size", "expected"),
    [
        (1024, 1, 8192),
        (4096, 2, 8192),
        (8192, 2, 16384),
    ],
)
def test_fi_moe_largest_bucket_scales_data_parallel_capacity(
    max_num_tokens: int, dp_size: int, expected: int
):
    config = SimpleNamespace(max_num_tokens=max_num_tokens, dp_size=dp_size)

    assert fi_moe_largest_bucket(config) == expected
