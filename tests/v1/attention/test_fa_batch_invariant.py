# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backends.mla.flashattn_mla import FlashAttnMLABackend
from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
    FlashAttnMLASparseBackend,
)


def test_flashattention_mla_backends_reject_batch_invariance(monkeypatch):
    monkeypatch.setattr("vllm.envs.VLLM_BATCH_INVARIANT", True)

    assert not FlashAttnMLABackend.supports_batch_invariance()
    assert not FlashAttnMLASparseBackend.supports_batch_invariance()
