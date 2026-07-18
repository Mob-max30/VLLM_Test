# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType
from typing import Any, cast


def test_flashattention_mla_backends_reject_batch_invariance(monkeypatch):
    fake_flash_attn = ModuleType("vllm.vllm_flash_attn")
    fake_flash_attn_any = cast(Any, fake_flash_attn)
    fake_flash_attn_any.flash_attn_varlen_func = object()
    fake_flash_attn_any.get_scheduler_metadata = object()
    fake_flash_attn_interface = ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    fake_flash_attn_interface_any = cast(Any, fake_flash_attn_interface)
    fake_flash_attn_interface_any.flash_attn_varlen_func = object()
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", fake_flash_attn)
    monkeypatch.setitem(
        sys.modules,
        "vllm.vllm_flash_attn.flash_attn_interface",
        fake_flash_attn_interface,
    )
    monkeypatch.setattr("vllm.envs.VLLM_BATCH_INVARIANT", True)

    from vllm.v1.attention.backends.mla.flashattn_mla import (
        FlashAttnMLABackend,
    )
    from vllm.v1.attention.backends.mla.flashattn_mla_sparse import (
        FlashAttnMLASparseBackend,
    )

    assert not FlashAttnMLABackend.supports_batch_invariance()
    assert not FlashAttnMLASparseBackend.supports_batch_invariance()
