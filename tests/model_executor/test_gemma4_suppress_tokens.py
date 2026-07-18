# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.model_executor.models.gemma4_unified import (
    Gemma4UnifiedForConditionalGeneration,
)


class _FakeLanguageModel(nn.Module):
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.clone()


def test_gemma4_unified_suppress_tokens_masking():
    model = Gemma4UnifiedForConditionalGeneration.__new__(
        Gemma4UnifiedForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.language_model = _FakeLanguageModel()
    model.register_buffer(
        "_suppress_token_ids",
        torch.tensor([1, 3], dtype=torch.long),
        persistent=False,
    )

    logits = model.compute_logits(torch.zeros(2, 5))

    assert torch.isneginf(logits[:, [1, 3]]).all()
    assert torch.equal(logits[:, [0, 2, 4]], torch.zeros(2, 3))
