# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.cudagraph_utils import (
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


class MultiModuleMTPSpeculator(DraftModelSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.is_continued_prefill = UvaBackedTensor(self.max_num_reqs, dtype=torch.bool)

        self.supports_mm_inputs = MULTIMODAL_REGISTRY.supports_multimodal_inputs(
            self.draft_model_config
        )
        # HACK: the Inkling MTP draft has no MM processor of its own (its draft
        # config is flattened text-only), but it consumes the target's merged
        # embeddings at draft prefill — treat it as MM-capable whenever the
        # target is.
        if (
            not self.supports_mm_inputs
            and self.draft_model_config.hf_config.model_type == "inkling_mtp"
        ):
            self.supports_mm_inputs = MULTIMODAL_REGISTRY.supports_multimodal_inputs(
                vllm_config.model_config
            )
        self.inputs_embeds: torch.Tensor | None = None
        if self.supports_mm_inputs:
            self.inputs_embeds = torch.zeros(
                self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
            )

        self.cached_draft_input_ids = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps - 1,
            dtype=torch.int64,
            device=self.device,
        )
        self.cached_draft_input_embeds: torch.Tensor | None = None
        if self.supports_mm_inputs:
            self.cached_draft_input_embeds = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps - 1,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
        self.cached_target_hidden_states = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps - 1,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

        self.cudagraph_manager: SpeculatorCudaGraphManager | None = None

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        self.cudagraph_manager = SpeculatorCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.num_speculative_steps + 1,
        )

    def capture(self) -> None:
        logger.info("Capturing model for multi-module MTP speculator...")
        # Reset indices to zeros to prevent stale values from prior
        # dummy runs to cause out-of-bounds indexing during capture.
        self.last_token_indices.zero_()
        assert self.cudagraph_manager is not None
        if self.cudagraph_manager.use_breakable_cg:
            self.cudagraph_manager.init_breakable_cg_runner(self.model)
        self.cudagraph_manager.capture(
            self._generate_drafts,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing multi-module MTP CUDA graphs",
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        new_req_ids: set[str] | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_speculative_steps, self.max_model_len
        )

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        (
            query_start_loc_np,
            max_query_len,
            num_rejected,
            mm_inputs,
            num_tokens,
            num_tokens_padded,
        ) = self._preprocess_chunked_prefills(
            num_reqs, input_batch, num_rejected, mm_inputs, new_req_ids
        )

        self._prepare_inputs(
            last_hidden_states,
            input_batch,
            num_tokens,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            mm_inputs,
        )

        # When all requests are decoding (no true prefills), each has
        # num_speculative_steps + 1 tokens, enabling FULL graph replay. Widening
        # only happens with prefills present, which is never the uniform case.
        uniform_token_count = get_uniform_token_count(
            num_reqs,
            num_tokens,
            max_query_len,
        )
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.cudagraph_manager,
            num_reqs,
            num_tokens_padded,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        # Rebuild the slot mappings and attention metadata.
        skip_attn = dummy_run and skip_attn_for_dummy_run
        if not skip_attn:
            # Build the slot mappings and attention metadata.
            slot_mappings_tensor = self.block_tables.compute_slot_mappings(
                self.idx_mapping[:num_reqs],
                self.input_buffers.query_start_loc,
                self.input_buffers.positions,
                batch_desc.num_tokens,
            )
            # Apply padding values to slots not corresponding to real draft
            # tokens to prevent stale value writes.
            pad_trailing_draft_slots(
                slot_mappings_tensor,
                self.input_buffers.query_start_loc,
                self.last_token_indices[:num_reqs],
                num_reqs,
            )
            slot_mappings = build_slot_mappings_by_layer(
                slot_mappings_tensor, self.kv_cache_config
            )
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=batch_desc.num_reqs or num_reqs,
                num_tokens_padded=batch_desc.num_tokens,
                query_start_loc_np=query_start_loc_np,
            )

        self._prepare_eplb_forward(num_tokens)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.cudagraph_manager is not None
            self.cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_drafts(
                num_reqs,
                batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )
        return self.draft_tokens[:num_reqs]

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        spec_module_idx: int,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            model_inputs = dict(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=(
                    self.inputs_embeds[:num_tokens]
                    if self.inputs_embeds is not None
                    else None
                ),
                spec_step_idx=spec_module_idx,
            )
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                # PIECEWISE cudagraph (compiled PW or breakable), chosen inside
                # run_pw_graph.
                assert self.cudagraph_manager is not None
                ret_hidden_states = self.cudagraph_manager.run_pw_graph(
                    self.model, model_inputs
                )
            else:
                # Eager (NONE): call the raw model directly.
                ret_hidden_states = self.model(**model_inputs)
        # Some MTP models declare a single-tensor contract but return
        # (logits_hidden, feedback_hidden) for final-norm correctness.
        if isinstance(ret_hidden_states, tuple):
            last_hidden_states, hidden_states = ret_hidden_states
        else:
            last_hidden_states = ret_hidden_states
            hidden_states = ret_hidden_states
        return last_hidden_states, hidden_states

    def _preprocess_chunked_prefills(
        self,
        num_reqs: int,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        new_req_ids: set[str] | None = None,
    ) -> tuple[
        np.ndarray,
        int,
        torch.Tensor,
        tuple[list[torch.Tensor], torch.Tensor] | None,
        int,
        int,
    ]:
        """Expands query/sequence lengths for requests performing chunked
        prefills.

        During a continued prefill, the speculator's drafted tokens
        from the last step were all discarded, but not treated as
        "rejected" tokens. However, those discarded token's KVs remain
        in the last N - 1 MTP module's caches. The trailing tokens from
        the last chunked prefill must be re-prefilled to correct the
        stale KVs. This requires overriding num_rejected and expanding
        the query/sequence lengths and is_mm_embed mask (for MM models).
        """

        # Determine which requests are continued prefills.
        if new_req_ids is not None:
            is_new_req_np = np.fromiter(
                (rid in new_req_ids for rid in input_batch.req_ids),
                dtype=bool,
                count=num_reqs,
            )
            is_continued_prefill_np = input_batch.is_prefilling_np & ~is_new_req_np
        else:
            is_continued_prefill_np = np.zeros(num_reqs, dtype=bool)

        if not is_continued_prefill_np.any():
            # No expanding needed.
            self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
                input_batch.query_start_loc[: num_reqs + 1]
            )
            self.input_buffers.seq_lens[:num_reqs].copy_(
                input_batch.seq_lens[:num_reqs]
            )
            return (
                input_batch.query_start_loc_np,
                input_batch.num_scheduled_tokens.max(),
                num_rejected,
                mm_inputs,
                input_batch.num_tokens,
                input_batch.num_tokens_after_padding,
            )

        # Expand the CPU query lengths (needed for building attention metadata).
        expansion_np = np.minimum(
            self.num_speculative_steps, input_batch.num_computed_tokens_np + 1
        )
        expansion_np = np.where(is_continued_prefill_np, expansion_np, 0)
        query_lens_np = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
        query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(
            query_lens_np + expansion_np, out=query_start_loc_np[1 : num_reqs + 1]
        )
        num_tokens = int(query_start_loc_np[num_reqs])
        max_query_len = int(query_lens_np.max())

        # Expand the is_mm_embed mask (for MM models).
        if mm_inputs is not None:
            mm_embeds, is_mm_embed = mm_inputs
            expanded_is_mm_embed = torch.from_numpy(
                self._expanded_is_mm_embed(
                    is_mm_embed.numpy(),
                    input_batch.query_start_loc_np,
                    query_start_loc_np,
                    num_reqs,
                    num_tokens,
                )
            )
            mm_inputs = (mm_embeds, expanded_is_mm_embed)

        # Expand the GPU query start positions and sequence lengths.
        self.is_continued_prefill.np[:num_reqs] = is_continued_prefill_np
        is_continued_prefill_gpu = self.is_continued_prefill.copy_to_uva(num_reqs)
        num_rejected = num_rejected.clone()
        draft_query_lens = input_batch.query_start_loc.new_empty(num_reqs)
        expand_attention_metadata(
            num_reqs,
            is_continued_prefill_gpu,
            input_batch.query_start_loc,
            input_batch.seq_lens,
            num_rejected,
            self.input_buffers.query_start_loc,
            draft_query_lens,
            self.input_buffers.seq_lens,
            self.num_speculative_steps,
        )
        torch.cumsum(
            draft_query_lens,
            dim=0,
            out=self.input_buffers.query_start_loc[1 : num_reqs + 1],
        )

        # Update the padded token count for CUDA graph dispatch.
        num_tokens_padded = input_batch.num_tokens_after_padding + (
            num_tokens - input_batch.num_tokens
        )
        return (
            query_start_loc_np,
            max_query_len,
            num_rejected,
            mm_inputs,
            num_tokens,
            num_tokens_padded,
        )

    def _prepare_inputs(
        self,
        target_hidden_states: torch.Tensor,
        input_batch: InputBatch,
        num_tokens: int,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        prepare_input_buffers(
            num_reqs,
            input_batch,
            self.cached_draft_input_ids,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.input_buffers,
            self.last_token_indices,
            self.max_num_reqs,
            self.num_speculative_steps,
        )

        # For MM models, compute the input embeddings with the MM embeddings
        # merged in.
        if self.inputs_embeds is not None:
            mm_embeds, is_mm_embed = mm_inputs or (None, None)
            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_buffers.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

        prepare_input_hidden_states_and_embeddings(
            num_reqs,
            self.hidden_states,
            target_hidden_states,
            self.cached_target_hidden_states,
            self.inputs_embeds,
            self.cached_draft_input_embeds,
            input_batch,
            self.input_buffers,
            num_rejected,
            self.num_speculative_steps,
        )

    def _expanded_is_mm_embed(
        self,
        is_mm_embed: np.ndarray,
        target_query_start_loc_np: np.ndarray,
        draft_query_start_loc_np: np.ndarray,
        num_reqs: int,
        num_tokens: int,
    ) -> np.ndarray:
        expanded_is_mm_embed = np.zeros(num_tokens, dtype=bool)
        target_query_lens = (
            target_query_start_loc_np[1 : num_reqs + 1]
            - target_query_start_loc_np[:num_reqs]
        )
        draft_query_lens = (
            draft_query_start_loc_np[1 : num_reqs + 1]
            - draft_query_start_loc_np[:num_reqs]
        )
        offsets = np.maximum(draft_query_lens - target_query_lens - 1, 0)
        for i in range(num_reqs):
            tqs = target_query_start_loc_np[i]
            dqs = draft_query_start_loc_np[i] + offsets[i]
            qlen = target_query_lens[i]
            expanded_is_mm_embed[dqs : dqs + qlen] = is_mm_embed[tqs : tqs + qlen]
        return expanded_is_mm_embed

    def _generate_drafts(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_token_indices = self.last_token_indices[:num_reqs]
        sample_positions = self.input_buffers.positions[last_token_indices]
        idx_mapping = self.idx_mapping[:num_reqs]

        # Cache the trailing token's ids, hidden states (and embeddings for
        # MM models), which are needed if the trailing tokens are re-prefilled
        # during the next decode step.
        cache_inputs(
            self.input_buffers,
            self.inputs_embeds,
            self.hidden_states,
            self.cached_draft_input_ids,
            self.cached_draft_input_embeds,
            self.cached_target_hidden_states,
            last_token_indices,
            idx_mapping,
            num_reqs,
            self.num_speculative_steps,
            use_input_embeds=self.inputs_embeds is not None,
        )

        for step in range(self.num_speculative_steps):
            # Update the current draft step.
            self.current_draft_step.fill_(step)

            # Run the model forward pass.
            last_hidden_states, hidden_states = self._run_model(
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                spec_module_idx=step,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
            )

            # Sample draft tokens for the current step.
            sample_hidden_states = last_hidden_states[last_token_indices]
            draft_tokens = self.sample_draft(
                sample_hidden_states,
                sample_positions,
                idx_mapping,
                self.temperature,
                self.seeds,
                self.current_draft_step,
                self.draft_logits,
            )

            self.draft_tokens[:num_reqs, step] = draft_tokens
            if step < self.num_speculative_steps - 1:
                self.hidden_states[:num_tokens] = hidden_states
                # Shift the draft inputs left by one and append the freshly
                # sampled token id/embeddings.
                draft_embeds = (
                    self.model.embed_input_ids(draft_tokens)
                    if self.inputs_embeds is not None
                    else None
                )
                update_draft_inputs(
                    draft_tokens,
                    draft_embeds,
                    self.input_buffers,
                    self.inputs_embeds,
                    last_token_indices,
                    idx_mapping,
                    num_reqs,
                )
                sample_positions += 1


@triton.jit
def _expand_attention_metadata_kernel(
    needs_expanding_ptr,
    target_query_start_loc_ptr,
    target_seq_lens_ptr,
    num_rejected_ptr,
    draft_query_start_loc_ptr,
    draft_query_lens_ptr,
    draft_seq_lens_ptr,
    num_speculative_steps,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:
        tl.store(draft_query_start_loc_ptr, 0)

    query_start = tl.load(target_query_start_loc_ptr + req_idx)
    query_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(target_seq_lens_ptr + req_idx)

    needs_expanding = tl.load(needs_expanding_ptr + req_idx)
    if needs_expanding:
        num_computed = seq_len - query_len
        num_padding = min(num_speculative_steps, num_computed + 1)
        tl.store(num_rejected_ptr + req_idx, num_padding)
        query_len += num_padding
        seq_len += num_padding - 1

    tl.store(draft_query_lens_ptr + req_idx, query_len)
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)


def expand_attention_metadata(
    num_reqs: int,
    # [num_reqs] bool
    needs_expanding: torch.Tensor,
    # [num_reqs + 1]
    target_query_start_loc: torch.Tensor,
    # [num_reqs]
    target_seq_lens: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    draft_query_start_loc: torch.Tensor,
    # [max_num_reqs]
    draft_query_lens: torch.Tensor,
    # [max_num_reqs]
    draft_seq_lens: torch.Tensor,
    num_speculative_steps: int,
) -> None:
    _expand_attention_metadata_kernel[(num_reqs,)](
        needs_expanding,
        target_query_start_loc,
        target_seq_lens,
        num_rejected,
        draft_query_start_loc,
        draft_query_lens,
        draft_seq_lens,
        num_speculative_steps,
    )


@triton.jit
def _prepare_input_buffers_kernel(
    last_token_indices_ptr,
    draft_input_ids_ptr,
    draft_positions_ptr,
    draft_seq_lens_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    cached_draft_input_ids_ptr,
    cached_draft_input_ids_stride0,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    target_query_start_loc_ptr,
    draft_query_start_loc_ptr,
    max_num_reqs,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    target_query_start = tl.load(target_query_start_loc_ptr + req_idx)
    draft_query_start = tl.load(draft_query_start_loc_ptr + req_idx)
    draft_query_end = tl.load(draft_query_start_loc_ptr + req_idx + 1)
    query_len = draft_query_end - draft_query_start
    seq_len = tl.load(draft_seq_lens_ptr + req_idx)

    # Get the number of rejected tokens, and the number of trailing tokens from
    # the last decode step that need to be re-prefilled to update the stale
    # KV cache slots in the MTP modules.
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    num_reprefills = max(0, num_rejected - 1)
    # Adjust the query and sequence lengths to account for rejected/re-prefilled
    # tokens.
    query_len -= num_rejected
    seq_len -= num_reprefills

    # Get the next draft input token.
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        next_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefill. Seed with the next prefill token.
        next_token = tl.load(next_prefill_tokens_ptr + req_state_idx)

    # Copy the target's input ids (read at the target offset) shifted left by 1,
    # and right by the number of re-prefills, into the draft buffer.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(
            target_input_ids_ptr + target_query_start + block, mask=mask
        )
        tl.store(
            draft_input_ids_ptr + draft_query_start - 1 + num_reprefills + block,
            input_ids,
            mask=mask,
        )
    last_token_index = draft_query_start + query_len - 1 + num_reprefills
    tl.store(last_token_indices_ptr + req_idx, last_token_index)
    tl.store(draft_input_ids_ptr + last_token_index, next_token)

    # Copy positions, shifted over by the number of tokens to be re-prefilled.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(
            target_positions_ptr + target_query_start + block, mask=mask
        )
        tl.store(
            draft_positions_ptr + draft_query_start + num_reprefills + block,
            target_pos,
            mask=mask,
        )

    # Fill the re-prefill gap with the cached token ids from the previous
    # decode step. These tokens sit immediately before the query's first
    # token, so their positions are contiguous and derived here.
    first_position = tl.load(target_positions_ptr + target_query_start)
    for i in range(num_reprefills):
        cache_read_slot = num_speculative_steps - 1 - num_reprefills + i
        cached_token_id = tl.load(
            cached_draft_input_ids_ptr
            + req_state_idx * cached_draft_input_ids_stride0
            + cache_read_slot
        )
        tl.store(draft_input_ids_ptr + draft_query_start + i, cached_token_id)
        tl.store(
            draft_positions_ptr + draft_query_start + i,
            first_position - num_reprefills + i,
        )

    # Write the updated sequence lengths in place.
    tl.store(draft_seq_lens_ptr + req_idx, seq_len)

    if req_idx == (num_reqs - 1):
        # Pad query_start_loc for CUDA graphs.
        for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs + 1
            tl.store(draft_query_start_loc_ptr + block, draft_query_end, mask=mask)
        # Pad seq_lens for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(draft_seq_lens_ptr + block, 0, mask=mask)
        # Pad last_token_indices for CUDA graphs.
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_input_buffers(
    num_reqs: int,
    input_batch: InputBatch,
    # [max_num_reqs, num_speculative_steps - 1]
    cached_draft_input_ids: torch.Tensor | None,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [max_num_reqs]
    last_sampled: torch.Tensor,
    # [max_num_reqs]
    next_prefill_tokens: torch.Tensor,
    input_buffers: InputBuffers,
    # [max_num_reqs]
    last_token_indices: torch.Tensor,
    max_num_reqs: int,
    num_speculative_steps: int,
) -> None:
    _prepare_input_buffers_kernel[(num_reqs,)](
        last_token_indices,
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.seq_lens,
        input_batch.input_ids,
        input_batch.positions,
        cached_draft_input_ids,
        cached_draft_input_ids.stride(0) if cached_draft_input_ids is not None else 0,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        input_batch.query_start_loc,
        input_buffers.query_start_loc,
        max_num_reqs,
        num_speculative_steps,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _prepare_input_hidden_states_and_embeddings_kernel(
    draft_input_hidden_states_ptr,
    draft_input_hidden_states_stride0,
    target_hidden_states_ptr,
    target_hidden_states_stride0,
    cached_target_hidden_states_ptr,
    cached_target_hidden_states_stride0,
    cached_target_hidden_states_stride1,
    input_embeds_ptr,
    input_embeds_stride0,
    cached_draft_input_embeds_ptr,
    cached_draft_input_embeds_stride0,
    cached_draft_input_embeds_stride1,
    idx_mapping_ptr,
    num_rejected_ptr,
    target_query_start_loc_ptr,
    draft_query_start_loc_ptr,
    num_speculative_steps,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    USE_INPUT_EMBEDS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < hidden_size

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    target_query_start = tl.load(target_query_start_loc_ptr + req_idx)
    draft_query_start = tl.load(draft_query_start_loc_ptr + req_idx)
    draft_query_end = tl.load(draft_query_start_loc_ptr + req_idx + 1)
    query_len = draft_query_end - draft_query_start
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    num_reprefills = max(0, num_rejected - 1)
    query_len -= num_rejected

    # Fill the re-prefill gap with the cached hidden states (and embeddings for MM
    # models) from the previous decode step, mirroring the token ids and positions
    # inserted by _prepare_input_buffers_kernel.
    for i in range(num_reprefills):
        cache_read_slot = num_speculative_steps - 1 - num_reprefills + i
        cached_hidden_state = tl.load(
            cached_target_hidden_states_ptr
            + req_state_idx * cached_target_hidden_states_stride0
            + cache_read_slot * cached_target_hidden_states_stride1
            + block,
            mask=mask,
        )
        tl.store(
            draft_input_hidden_states_ptr
            + (draft_query_start + i) * draft_input_hidden_states_stride0
            + block,
            cached_hidden_state,
            mask=mask,
        )
        if USE_INPUT_EMBEDS:
            cached_embed = tl.load(
                cached_draft_input_embeds_ptr
                + req_state_idx * cached_draft_input_embeds_stride0
                + cache_read_slot * cached_draft_input_embeds_stride1
                + block,
                mask=mask,
            )
            tl.store(
                input_embeds_ptr
                + (draft_query_start + i) * input_embeds_stride0
                + block,
                cached_embed,
                mask=mask,
            )

    # Copy the output target hidden states (read at the target offset) as inputs
    # to the first MTP module, written at the draft offset.
    for i in range(query_len):
        hidden_state = tl.load(
            target_hidden_states_ptr
            + (target_query_start + i) * target_hidden_states_stride0
            + block,
            mask=mask,
        )
        tl.store(
            draft_input_hidden_states_ptr
            + (draft_query_start + num_reprefills + i)
            * draft_input_hidden_states_stride0
            + block,
            hidden_state,
            mask=mask,
        )


def prepare_input_hidden_states_and_embeddings(
    num_reqs: int,
    # [num_tokens, hidden_size]
    hidden_states: torch.Tensor,
    # [num_tokens, hidden_size]
    target_hidden_states: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_target_hidden_states: torch.Tensor | None,
    # [num_tokens, hidden_size]
    input_embeds: torch.Tensor | None,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_draft_input_embeds: torch.Tensor | None,
    input_batch: InputBatch,
    input_buffers: InputBuffers,
    # [num_reqs]
    num_rejected: torch.Tensor,
    num_speculative_steps: int,
) -> None:
    use_input_embeds = input_embeds is not None
    hidden_size = target_hidden_states.shape[-1]
    hidden_block_size = 1024
    num_dim_blocks = triton.cdiv(hidden_size, hidden_block_size)
    _prepare_input_hidden_states_and_embeddings_kernel[(num_reqs, num_dim_blocks)](
        hidden_states,
        hidden_states.stride(0),
        target_hidden_states,
        target_hidden_states.stride(0),
        cached_target_hidden_states,
        cached_target_hidden_states.stride(0)
        if cached_target_hidden_states is not None
        else 0,
        cached_target_hidden_states.stride(1)
        if cached_target_hidden_states is not None
        else 0,
        input_embeds,
        input_embeds.stride(0) if input_embeds is not None else 0,
        cached_draft_input_embeds,
        cached_draft_input_embeds.stride(0)
        if cached_draft_input_embeds is not None
        else 0,
        cached_draft_input_embeds.stride(1)
        if cached_draft_input_embeds is not None
        else 0,
        input_batch.idx_mapping,
        num_rejected,
        input_batch.query_start_loc,
        input_buffers.query_start_loc,
        num_speculative_steps,
        hidden_size,
        BLOCK_SIZE=hidden_block_size,
        USE_INPUT_EMBEDS=use_input_embeds,
    )


@triton.jit
def _pad_trailing_draft_slots_kernel(
    slot_mappings_ptr,
    slot_mappings_stride0,
    query_start_loc_ptr,
    last_token_indices_ptr,
    PAD_ID,
    BLOCK_SIZE: tl.constexpr,
):
    group_idx = tl.program_id(0)
    req_idx = tl.program_id(1)
    # Slots computed from stale token positions in the range
    # [last_token_index + 1, query_end) can result in writes to blocks.
    # Pad these slot values so that attention kernels ignore them.
    start = tl.load(last_token_indices_ptr + req_idx) + 1
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    base = slot_mappings_ptr + group_idx * slot_mappings_stride0
    for i in range(start, end, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < end
        tl.store(base + offs, PAD_ID, mask=mask)


def pad_trailing_draft_slots(
    # [num_groups, num_tokens_padded]
    slot_mappings: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
    # [num_reqs]
    last_token_indices: torch.Tensor,
    num_reqs: int,
) -> None:
    num_groups = slot_mappings.shape[0]
    _pad_trailing_draft_slots_kernel[(num_groups, num_reqs)](
        slot_mappings,
        slot_mappings.stride(0),
        query_start_loc,
        last_token_indices,
        PAD_SLOT_ID,
        BLOCK_SIZE=256,
    )


@triton.jit
def _cache_inputs_kernel(
    draft_input_ids_ptr,
    draft_input_embeds_ptr,
    draft_input_embeds_stride0,
    draft_input_hidden_states_ptr,
    draft_input_hidden_states_stride0,
    cached_draft_input_ids_ptr,
    cached_draft_input_ids_stride0,
    cached_draft_input_embeds_ptr,
    cached_draft_input_embeds_stride0,
    cached_draft_input_embeds_stride1,
    cached_target_hidden_states_ptr,
    cached_target_hidden_states_stride0,
    cached_target_hidden_states_stride1,
    idx_mapping_ptr,
    last_token_indices_ptr,
    query_start_loc_ptr,
    num_speculative_steps,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    USE_INPUT_EMBEDS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < hidden_size

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        # Skip cudagraph padded requests.
        return

    query_start = tl.load(query_start_loc_ptr + req_idx)
    last_token_index = tl.load(last_token_indices_ptr + req_idx)

    # Snapshot the last num_speculative_steps - 1 input draft token ids/hidden
    # states (and embeddings for MM models), indexed by request state. These
    # may be needed to re-prefill the tokens during the next decode step.
    cache_window_size = num_speculative_steps - 1
    window_start = last_token_index - cache_window_size + 1
    for i in range(max(window_start, query_start), last_token_index + 1):
        cache_write_slot = i - window_start
        if block_idx == 0:
            input_id = tl.load(draft_input_ids_ptr + i)
            tl.store(
                cached_draft_input_ids_ptr
                + req_state_idx * cached_draft_input_ids_stride0
                + cache_write_slot,
                input_id,
            )
        if USE_INPUT_EMBEDS:
            input_embeds = tl.load(
                draft_input_embeds_ptr + i * draft_input_embeds_stride0 + block,
                mask=mask,
            )
            tl.store(
                cached_draft_input_embeds_ptr
                + req_state_idx * cached_draft_input_embeds_stride0
                + cache_write_slot * cached_draft_input_embeds_stride1
                + block,
                input_embeds,
                mask=mask,
            )
        hidden_state = tl.load(
            draft_input_hidden_states_ptr
            + i * draft_input_hidden_states_stride0
            + block,
            mask=mask,
        )
        tl.store(
            cached_target_hidden_states_ptr
            + req_state_idx * cached_target_hidden_states_stride0
            + cache_write_slot * cached_target_hidden_states_stride1
            + block,
            hidden_state,
            mask=mask,
        )


def cache_inputs(
    input_buffers: InputBuffers,
    # [num_tokens, hidden_size]
    draft_input_embeds: torch.Tensor | None,
    # [num_tokens, hidden_size]
    draft_input_hidden_states: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1]
    cached_draft_input_ids: torch.Tensor,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_draft_input_embeds: torch.Tensor | None,
    # [max_num_reqs, num_speculative_steps - 1, hidden_size]
    cached_target_hidden_states: torch.Tensor,
    # [num_reqs]
    last_token_indices: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    num_reqs: int,
    num_speculative_steps: int,
    use_input_embeds: bool,
) -> None:
    hidden_size = draft_input_hidden_states.shape[-1]
    hidden_block_size = 1024
    _cache_inputs_kernel[(num_reqs, triton.cdiv(hidden_size, hidden_block_size))](
        input_buffers.input_ids,
        draft_input_embeds,
        draft_input_embeds.stride(0) if draft_input_embeds is not None else 0,
        draft_input_hidden_states,
        draft_input_hidden_states.stride(0),
        cached_draft_input_ids,
        cached_draft_input_ids.stride(0),
        cached_draft_input_embeds,
        cached_draft_input_embeds.stride(0)
        if cached_draft_input_embeds is not None
        else 0,
        cached_draft_input_embeds.stride(1)
        if cached_draft_input_embeds is not None
        else 0,
        cached_target_hidden_states,
        cached_target_hidden_states.stride(0),
        cached_target_hidden_states.stride(1),
        idx_mapping,
        last_token_indices,
        input_buffers.query_start_loc,
        num_speculative_steps,
        hidden_size,
        BLOCK_SIZE=hidden_block_size,
        USE_INPUT_EMBEDS=use_input_embeds,
    )


@triton.jit
def _shift_input_ids_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    last_token_indices_ptr,
    draft_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        # Skip cudagraph padded requests.
        return

    query_start = tl.load(query_start_loc_ptr + req_idx)
    # Use the post-rejection last token index so the shift and insertion align
    # with the position the draft token was sampled from.
    last_token_index = tl.load(last_token_indices_ptr + req_idx)
    query_len = last_token_index - query_start + 1

    # Shift input token ids to the left by one position and
    # insert the last sampled draft token.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(input_ids_ptr + query_start + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block - 1, input_ids, mask=mask)
    draft_token = tl.load(draft_tokens_ptr + req_idx)
    tl.store(input_ids_ptr + last_token_index, draft_token)


@triton.jit
def _shift_input_embeds_kernel(
    input_embeds_ptr,
    input_embeds_stride0,
    draft_embeds_ptr,
    draft_embeds_stride0,
    idx_mapping_ptr,
    query_start_loc_ptr,
    last_token_indices_ptr,
    hidden_size,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    if req_state_idx < 0:
        # Skip cudagraph padded requests.
        return

    block_idx = tl.program_id(1)
    dim_block = block_idx * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    dim_mask = dim_block < hidden_size

    query_start = tl.load(query_start_loc_ptr + req_idx)
    last_token_index = tl.load(last_token_indices_ptr + req_idx)
    query_len = last_token_index - query_start + 1

    # Shift input token embeddings to the left by one position and
    # insert the last sampled draft token's embeddings.
    for i in range(1, query_len, BLOCK_SIZE_Q):
        query_block = i + tl.arange(0, BLOCK_SIZE_Q)
        query_mask = query_block < query_len
        mask = query_mask[:, None] & dim_mask[None, :]
        input_embed = tl.load(
            input_embeds_ptr
            + (query_start + query_block)[:, None] * input_embeds_stride0
            + dim_block[None, :],
            mask=mask,
        )
        tl.store(
            input_embeds_ptr
            + (query_start + query_block - 1)[:, None] * input_embeds_stride0
            + dim_block[None, :],
            input_embed,
            mask=mask,
        )
    draft_embed = tl.load(
        draft_embeds_ptr + req_idx * draft_embeds_stride0 + dim_block,
        mask=dim_mask,
    )
    tl.store(
        input_embeds_ptr + last_token_index * input_embeds_stride0 + dim_block,
        draft_embed,
        mask=dim_mask,
    )


def update_draft_inputs(
    draft_tokens: torch.Tensor,
    draft_embeds: torch.Tensor | None,
    input_buffers: InputBuffers,
    input_embeds: torch.Tensor | None,
    last_token_indices: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_reqs: int,
) -> None:
    _shift_input_ids_kernel[(num_reqs,)](
        input_buffers.input_ids,
        idx_mapping,
        input_buffers.query_start_loc,
        last_token_indices,
        draft_tokens,
        BLOCK_SIZE=1024,
    )
    if input_embeds is not None:
        assert draft_embeds is not None
        hidden_size = input_embeds.shape[-1]
        hidden_block_size = 256
        _shift_input_embeds_kernel[
            (num_reqs, triton.cdiv(hidden_size, hidden_block_size))
        ](
            input_embeds,
            input_embeds.stride(0),
            draft_embeds,
            draft_embeds.stride(0),
            idx_mapping,
            input_buffers.query_start_loc,
            last_token_indices,
            hidden_size,
            BLOCK_SIZE_Q=16,
            BLOCK_SIZE_H=hidden_block_size,
        )
