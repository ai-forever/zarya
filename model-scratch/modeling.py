import inspect
import os
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np  # noqa: F401
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from torch.distributions.binomial import Binomial
from torch.nn.functional import cross_entropy
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import RepetitionPenaltyLogitsProcessor
from transformers.generation.configuration_utils import GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import ALL_CACHE_NAMES, GenerateOutput, GenerationMixin
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, can_return_tuple, logging

try:
    from transformers.utils.generic import merge_with_config_defaults
except ImportError:
    from transformers.utils.generic import check_model_inputs as merge_with_config_defaults

try:
    from transformers.distributed.fsdp import is_fsdp_managed_module
except ImportError:
    from transformers.integrations.fsdp import is_fsdp_managed_module

try:
    from transformers.distributed.utils import _get_torch_distributed_world_size
except ImportError:
    from transformers.pytorch_utils import _torch_distributed_available

    def _is_torch_distributed_initialized() -> bool:
        if not _torch_distributed_available:
            return False
        return torch.distributed.is_initialized()

    def _get_torch_distributed_world_size() -> int:
        if not _is_torch_distributed_initialized():
            return 1
        return torch.distributed.get_world_size()


try:
    from transformers.generation.utils import GENERATION_MODES_MAPPING
except ImportError:
    GENERATION_MODES_MAPPING = {
        GenerationMode.SAMPLE: "_sample",
        GenerationMode.GREEDY_SEARCH: "_sample",
        GenerationMode.BEAM_SEARCH: "_beam_search",
        GenerationMode.BEAM_SAMPLE: "_beam_search",
        GenerationMode.ASSISTED_GENERATION: "_assisted_decoding",
        # Deprecated methods
        GenerationMode.DOLA_GENERATION: "transformers-community/dola",
        GenerationMode.CONTRASTIVE_SEARCH: "transformers-community/contrastive-search",
        GenerationMode.GROUP_BEAM_SEARCH: "transformers-community/group-beam-search",
        GenerationMode.CONSTRAINED_BEAM_SEARCH: "transformers-community/constrained-beam-search",
    }

from .configuration import ZaryaConfig
from .generation_utils import ZaryaGenerationConfig

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ZaryaGenerationOutput(ModelOutput):
    """
    Output class for Zarya generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences, including the prompt if `input_ids` was provided to the `generate` method.
        scores (`None`):
            Unused. Kept in the interface for BC.
        logits (`None`):
            Unused. Kept in the interface for BC.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Unused. Kept in the interface for BC.
        hidden_states (`None`):
            Unused. Kept in the interface for BC.
        past_key_values (`Cache`):
            The cache used for generation. It can be passed to subsequent calls to `generate` to speed up generation,
            in multi-turn sessions.
        tokens_per_forward (`torch.LongTensor` of shape (`batch_size`)):
            The number of tokens per forward in this `generate` call, for each member in the batch. This is often
            used as a secondary evaluation metric for text diffusion models.
    """

    sequences: torch.LongTensor
    scores: Optional[tuple[torch.FloatTensor]] = None  # Unused for now, kept in the interface for BC with AR generation
    logits: Optional[tuple[torch.FloatTensor]] = None  # Unused for now, kept in the interface for BC with AR generation
    attentions: Optional[tuple[tuple[torch.FloatTensor]]] = (
        None  # Unused for now, kept in the interface for BC with AR generation
    )
    hidden_states: Optional[tuple[tuple[torch.FloatTensor]]] = (
        None  # Unused for now, kept in the interface for BC with AR generation
    )
    past_key_values: Optional[Cache] = None
    tokens_per_forward: Optional[int] = None


class DiffusionDynamicCache(DynamicCache):
    def __init__(self, num_hidden_layers: Optional[int] = None):
        super().__init__(num_hidden_layers)

    def full_update(
        self,
        new_kv: tuple,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        for layer_idx, (key_states, value_states) in enumerate(new_kv):
            self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def select_partial(
        self,
        indices: torch.Tensor,
    ):
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].keys = self.layers[layer_idx].keys[..., indices, :]
            self.layers[layer_idx].values = self.layers[layer_idx].values[..., indices, :]

    def batch_select_minibatch(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].keys = self.layers[layer_idx].keys[:indices, ...]
            self.layers[layer_idx].values = self.layers[layer_idx].values[:indices, ...]


@dataclass
class TextDiffusionLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    loss_seq: Optional[torch.FloatTensor] = None
    loss_dif: Optional[torch.FloatTensor] = None
    acc_seq: Optional[torch.FloatTensor] = None
    acc_dif: Optional[torch.FloatTensor] = None


def _apply_repetition_penalty(logits: torch.FloatTensor, cur_x: torch.LongTensor, repetition_penalty: float):
    """Apply repetition penalty to logits using RepetitionPenaltyLogitsProcessor."""
    if repetition_penalty == 1.0:
        return logits
    rep_penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
    processed_scores = torch.empty_like(logits)
    for idx in torch.arange(logits.shape[1], device=logits.device):
        processed_scores[:, idx, :] = rep_penalty_proc(cur_x, logits[:, idx, :])
    return processed_scores


def _shift_logits_for_ar(logits: torch.FloatTensor):
    """Shift logits right by one position for autoregressive prediction.

    In autoregressive generation, we predict token t+1 from tokens 1..t,
    so we shift logits right by one position.
    """
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def _compute_token_probabilities(logits: torch.FloatTensor, tokens: torch.LongTensor):
    """Softmax over logits, then gather probability of `tokens`."""
    probs = F.softmax(logits, dim=-1)
    return torch.gather(probs, dim=-1, index=torch.unsqueeze(tokens, -1)).squeeze(-1)


def _remove_accepted_slots(slots_x: torch.LongTensor, slots_pos_ids: torch.LongTensor, indices_to_remove: set):
    """Remove accepted slot indices from slot tensors and return the filtered pair."""
    keep_mask = torch.ones(slots_x.shape[1], dtype=torch.bool, device=slots_x.device)
    keep_mask[list(indices_to_remove)] = False
    return slots_x[:, keep_mask, :], slots_pos_ids[:, keep_mask, :]


def _verify_and_update_probs(
    model: Callable,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: DiffusionDynamicCache,
    cur_x: torch.Tensor,
    temperature: int,
    repetition_penalty: float,
):
    """
    One forward pass, return AR-shifted token probabilities and model outputs.

    This is the core verification pattern:
      forward → AR-shift → temperature scaling → rep penalty → softmax + gather

    """
    outputs = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    # Get autoregressive logits for verification
    logits = outputs.logits
    # Shift logits for AR (like in generation) to predict tokens in supplied positions
    logits = _shift_logits_for_ar(logits)
    if 0 < temperature < 1:
        logits = logits / temperature
    logits = _apply_repetition_penalty(logits, cur_x, repetition_penalty)
    return _compute_token_probabilities(logits, input_ids), outputs


def _sort_indices_only(
    indices: torch.Tensor,
    shuffle: bool,
    mask_token_id: int,
    masked: Optional[torch.Tensor] = None,
    masked_unshuffle: Optional[torch.Tensor] = None,
    keep_masks_unshuffled: bool = False,
):
    if masked is None:
        masked = indices == mask_token_id
    if shuffle:
        offsets = torch.rand(indices.shape).to(device=indices.device) * 0.9
        if keep_masks_unshuffled:
            if masked_unshuffle is None:
                masked_unshuffle = masked
            # induce left-to-right order within masked tokens
            # only for sequential part
            offsets[masked_unshuffle] = torch.linspace(1, 2, torch.sum(masked_unshuffle)).to(device=indices.device)
    else:
        offsets = torch.linspace(0, 0.9, indices.shape[1]).to(device=indices.device)
    sort_idx = (masked + offsets).argsort(descending=False)
    return sort_idx


def _init_generation_state(prompt: torch.Tensor, gen_length: int, mask_id: int, batch_size=1, **kwargs):
    """Prepare mask tokens, position IDs, and initial context.

    Handles both cases: prompt with/without mask tokens.

    Returns:
        (gen_x, gen_pos_ids, cur_x, prompt_pos_ids)
    """
    attention_mask = kwargs.get("attention_mask")
    position_ids = kwargs.get("position_ids")
    device = prompt.device

    masked = prompt == mask_id
    if attention_mask is not None:
        masked = torch.logical_and(masked, attention_mask)
    else:
        attention_mask = torch.ones_like(prompt, device=device)

    prompt_len = prompt.shape[1]
    masked_tokens_count = masked.sum(dim=-1)

    if masked_tokens_count.gt(0).any():
        leftover = gen_length - masked_tokens_count.max()
        # Prompt contains mask tokens — extract them into gen_x, reorder prompt.
        gen_x = torch.full((batch_size, masked_tokens_count.max()), mask_id, dtype=torch.long, device=device)
        # Extract positions of mask tokens in prompt
        # These are the positions that need to be generated
        gen_pos_ids = position_ids[0][masked[0]].unsqueeze(0).to(device)

        # Extract non-mask tokens and their positions
        # Reorder prompt: non-mask tokens first, then mask tokens will be processed
        non_mask_ids = prompt[0][~masked[0]]
        non_mask_pos = position_ids[0][~masked[0]]
        non_mask_attn = attention_mask[0][~masked[0]]
        # modify original prompt (reorder)
        prompt = non_mask_ids.unsqueeze(0).to(device)
        position_ids = non_mask_pos.unsqueeze(0).to(device)
        attention_mask = non_mask_attn.unsqueeze(0).to(device)

        if leftover > 0:
            extra_x = torch.full((batch_size, leftover), mask_id, dtype=torch.long, device=device)
            gen_x = torch.cat((gen_x, extra_x), dim=1)
            extra_pos = torch.arange(
                position_ids.max() + 1,
                position_ids.max() + 1 + leftover,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            gen_pos_ids = torch.cat((gen_pos_ids, extra_pos), dim=1)
    else:
        # ======================================================
        # USUAL GENERATION: Prefix completion (no masks in prompt)
        # ======================================================
        # Initialize generated sequence with mask tokens
        # Mask tokens will be filled in by the model during generation
        gen_x = torch.full((batch_size, gen_length), mask_id, dtype=torch.long, device=device)
        gen_pos_ids = torch.arange(prompt_len, prompt_len + gen_length, dtype=torch.long, device=device).unsqueeze(0)

    # Current context: prompt tokens and their positions
    # These are the known, verified tokens that form the context
    cur_x = prompt.clone()

    return gen_x, gen_pos_ids, cur_x, position_ids, attention_mask


def _build_blocks(gen_length: int, serial_num_blocks: int, slot_size: int, skip_len: int = 0):
    """Build the block schedule: divide gen_length into subblocks with slot-aligned boundaries."""
    num_blocks = max(serial_num_blocks, 1)
    gen_length = gen_length - skip_len
    block_length = gen_length // num_blocks  # Length of each serial block
    if block_length == 0:
        block_length = gen_length
        serial_num_blocks = 1
    slot_size = min(slot_size, block_length)
    aligned_len = (block_length // slot_size) * slot_size  # block_real_len

    subblocks = []
    for serial_num_block in range(serial_num_blocks):
        full_block_start = skip_len + serial_num_block * block_length
        full_block_end = full_block_start + aligned_len
        subblocks.append(
            {
                "start": full_block_start,
                "end": full_block_end,
                "slot_size": slot_size,
            }
        )
        if (maybe_slot_size := block_length % slot_size) > 0:
            if (serial_num_block == serial_num_blocks - 1) and (full_block_end + maybe_slot_size < gen_length):
                maybe_slot_size = gen_length - full_block_end
            while maybe_slot_size >= slot_size:
                subblocks.append(
                    {
                        "start": full_block_end,
                        "end": full_block_end + slot_size * (maybe_slot_size // slot_size),
                        "slot_size": slot_size,
                    }
                )
                full_block_end = full_block_end + slot_size * (maybe_slot_size // slot_size)
                maybe_slot_size = maybe_slot_size - slot_size * (maybe_slot_size // slot_size)
                if maybe_slot_size == 0:
                    break
            else:
                subblocks.append(
                    {
                        "start": full_block_end,
                        "end": full_block_end + maybe_slot_size,
                        "slot_size": maybe_slot_size,
                    }
                )
    return subblocks, slot_size, serial_num_blocks


def add_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
):
    """
    Apply Gumbel noise to logits for sampling from categorical distributions.

    The Gumbel-Max trick provides a way to sample from a categorical distribution
    parameterized by logits. This implementation follows the approach from
    arXiv:2409.02908 for MDM (Masked Diffusion Model).

    Key properties:
    - Temperature == 0 makes no sampling
    - Using float64 precision improves numerical stability for Gumbel sampling

    Args:
        logits: Raw model outputs of shape (batch_size, seq_len, vocab_size)
        temperature: Sampling temperature

    Returns:
        Noised logits with the same shape as input
    """
    if temperature == 0:
        return logits
    try:
        logits = logits.to(torch.float64)
    except TypeError:
        logits = logits.to(torch.float32)  # in case framework cannot work with float64
    # Sample uniform random values in (0, 1)
    noise = torch.rand_like(logits, dtype=logits.dtype)
    # Apply Gumbel noise: -log(-log(u)) where u ~ Uniform(0,1)
    # Simplified form: (-log(noise)) ** temperature
    gumbel_noise = (-torch.log(noise)) ** temperature
    # Convert back to probability space and normalize by Gumbel noise
    return logits.exp() / gumbel_noise


def suppress_token(
    logits: torch.FloatTensor, original_positions: torch.Tensor, position_limitation: torch.Tensor, token_id: int
):
    positions_to_suppress = original_positions.le(position_limitation)
    if positions_to_suppress.any():
        new_logit_values = logits.min(dim=-1).values
        logits[:, :, token_id][positions_to_suppress] = new_logit_values[positions_to_suppress]

    return logits


def _accept_verified_prefix(
    chosen_slots: torch.Tensor,
    chosen_pos: torch.Tensor,
    chosen_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    cur_x: torch.Tensor,
    cur_pos: torch.Tensor,
    cur_attn: torch.Tensor,
    past_key_values: DiffusionDynamicCache,
    flat_predicted: torch.Tensor,
    flat_predicted_pos: torch.Tensor,
    slot_size: int,
    total_slots: int,
    token_threshold: float,
    eos_token_id: int,
    mask_id: int,
    device: torch.device,
    stopping_criteria: StoppingCriteriaList,
    logits_processor: LogitsProcessorList,
):
    """Attempt to accept full slots based on verification probabilities.

    Returns updated state dict or None if no tokens could be accepted.

    """

    # =====================================================================
    # TOKEN-LEVEL ACCEPTANCE: Determine which tokens to keep
    # =====================================================================
    # Token-level acceptance based on confidence threshold
    # Only tokens with probability above threshold are accepted
    prob_mask = chosen_probs > token_threshold
    # CRITICAL: Always accept first token in each slot
    # The first token is used as reference for slot confidence, so it must be accepted
    prob_mask[:, 0] = True  # always accept first token of each slot
    # Always accept extra tokens at this stage
    # Cumulative product creates mask: zeroes after first zero seen
    # This implements early stopping: once a token is rejected, all subsequent
    # tokens in that block are also rejected
    # Example: [1, 1, 0, 1] -> [1, 1, 0, 0] - tokens 3+ are rejected if token 2 is rejected

    # Determine how many tokens can be accepted across all slots
    flat_acceptance = torch.cumprod(prob_mask.int().reshape(1, -1), dim=-1)
    prefix_len = torch.sum(flat_acceptance, dim=-1)
    flat_chosen = chosen_slots.reshape(1, -1)
    # Extract confidently accepted tokens
    confident_prefix_tokens = flat_chosen[:, :prefix_len]
    prefix_slot_tag = False  # Flag for prefix slots that were accepted
    sum_TPF_add = 0.0
    forward_count_add = 0
    eos_flag = False

    if prefix_len == 0:
        return None  # almost impossible because first tokens accepted

    # =====================================================================
    # HANDLE ACCEPTED TOKENS: Update context and KV cache
    # =====================================================================
    # Check if EOS token is in the accepted prefix
    is_eos_in_prefix = confident_prefix_tokens.squeeze(0) == eos_token_id
    eos_found_flag = torch.any(is_eos_in_prefix)

    is_early_stopping = stopping_criteria(
        torch.hstack(
            (
                cur_x.expand((chosen_slots.shape[0], -1)),
                torch.where(torch.cumprod(prob_mask.int(), dim=-1).bool(), chosen_slots, mask_id),
            )
        ),
        None,
        new_token_length=chosen_slots.shape[1],
    )
    early_stopping_flag = torch.any(is_early_stopping)

    remain_indices = []
    indices_to_remove = set()
    eos_slot_pos = len(topk_indices)
    early_stopping_slot_pos = len(topk_indices)
    if eos_found_flag:
        # =====================================================================
        # EOS HANDLING: Stop generation when end-of-sequence is found
        # =====================================================================
        # Find position of first EOS token in accepted prefix
        # torch.argmax returns the first (leftmost) index where condition is true
        first_eos_pos_tensor = torch.argmax(is_eos_in_prefix.int())

        # Calculate slot and position within slot for EOS
        eos_slot_pos = first_eos_pos_tensor // slot_size + 1
        eos_token_pos = first_eos_pos_tensor - (first_eos_pos_tensor // slot_size) * slot_size

    if early_stopping_flag:
        early_stopping_slot_pos = torch.argmax(is_early_stopping.int()).item() + 1

    if eos_found_flag or early_stopping_flag:
        remain_slot_pos = min(early_stopping_slot_pos, eos_slot_pos)
        eos_slot = topk_indices[remain_slot_pos - 1].item()
        # Keep slots up to and including the EOS slot
        remain_indices.extend(topk_indices[:remain_slot_pos].tolist())
        topk_indices = torch.tensor([], device=device)
        eos_flag = True

        # Mark all slots after EOS for removal
        indices_after_eos = list(range(eos_slot, total_slots))
        indices_to_remove.update(indices_after_eos)

    elif (prefix_len // slot_size) > 0:
        # =====================================================================
        # FULL SLOT ACCEPTANCE: All tokens in slot accepted
        # =====================================================================
        # Fully filled slot count:
        # Integer division tells us how many complete slots were accepted
        num_prefix_slots = prefix_len // slot_size
        remain_indices.extend(topk_indices[:num_prefix_slots].tolist())

        # Remove accepted slots from further processing
        topk_indices = topk_indices[num_prefix_slots:]

    if len(remain_indices) > 0:
        indices_to_remove.update(remain_indices)

        # =====================================================================
        # EXTRACT TOKEN INDICES: Get positions of accepted tokens
        # =====================================================================
        token_indices = []

        for i_idx, b_idx in enumerate(remain_indices):
            start_index = b_idx * slot_size

            current_block_len = slot_size
            # If EOS exists and this is the last slot, then adjust the length.
            if eos_found_flag and i_idx == len(remain_indices) - 1:
                current_block_len = eos_token_pos + 1

            end_index = start_index + current_block_len
            block_range = torch.arange(start_index, end_index, dtype=torch.long, device=device)

            token_indices.append(block_range)

        full_token_indices = torch.cat(token_indices)

        # =====================================================================
        # UPDATE CONTEXT: Append accepted tokens to current context
        # =====================================================================
        # Append accepted tokens to current context
        # These tokens are now verified and will not change
        cur_x = torch.cat((cur_x, flat_predicted[:, full_token_indices]), dim=1)
        cur_pos = torch.cat((cur_pos, flat_predicted_pos[:, full_token_indices]), dim=1)
        cur_attn = torch.cat((cur_pos, torch.ones_like(flat_predicted_pos[:, full_token_indices])), dim=1)

        # Update KV cache (crop to current context size)
        # The KV cache now contains all tokens up to cur_x
        past_key_values.crop(cur_x.shape[1])

        # Verify KV cache is properly synchronized
        assert cur_x.shape[-1] == past_key_values.layers[0].keys.shape[-2]

        prefix_slot_tag = True

        # Update TPF (tokens per forward) metric
        # This measures generation efficiency: more tokens per forward = better
        # The division by 2 accounts for the draft+verification forward passes
        sum_TPF_add = slot_size * len(remain_indices) / 2
        forward_count_add = 1

    return {
        "cur_x": cur_x,
        "cur_pos": cur_pos,
        "cur_attn": cur_attn,
        "past_key_values": past_key_values,
        "sum_TPF_add": sum_TPF_add,
        "forward_count_add": forward_count_add,
        "eos_found": eos_flag,
        "topk_indices": topk_indices,
        "prefix_slot_tag": prefix_slot_tag,
        "indices_to_remove": indices_to_remove,
    }


def _speculative_refinement(
    current_slots: torch.Tensor,
    chosen_pos: torch.Tensor,
    chosen_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    cur_x: torch.Tensor,
    cur_attn: torch.Tensor,
    past_key_values: DiffusionDynamicCache,
    slot_size: int,
    counts_slot: int,
    token_threshold: float,
    eos_token_id: int,
    mask_id: int,
    repetition_penalty: float,
    model: Callable,
    device: torch.device,
    batch_size: int,
    position_limitation: torch.Tensor,
    stopping_criteria: StoppingCriteriaList,
    logits_processor: LogitsProcessorList,
):
    """Iteratively refine tokens not accepted in the first verification pass.

    Returns dict with keys:
        kept_tokens, kept_pos_ids, past_key_values,
        sum_TPF_add, forward_count_add, eos_found, first_eos_slot_idx,
        accepted_indices (set), all accepted
    """
    # Prepare state
    # Token-level acceptance based on confidence threshold
    # Only tokens with probability above threshold are accepted
    prob_mask = chosen_probs > token_threshold
    # The first token is used as reference for slot confidence, so it must be accepted
    prob_mask[:, 0] = True  # always accept first token of each slot
    # Cumulative product creates mask: zeroes after first zero seen
    # This implements early stopping: once a token is rejected, all subsequent
    # tokens in that block are also rejected
    # Example: [1, 1, 0, 1] -> [1, 1, 0, 0] - tokens 3+ are rejected if token 2 is rejected
    acceptance_mask = torch.cumprod(prob_mask.int(), dim=-1)

    # Clone slots for iterative refinement
    # These are the slots that were not accepted in the prefix phase
    accepted_prefix_len = 0
    eos_found = False
    first_eos_slot_idx = -1

    # Expand KV cache for parallel slot processing
    if past_key_values is not None and counts_slot > 1:
        # Repeat KV cache for each slot to enable parallel processing
        # This allows us to verify multiple slots simultaneously
        past_key_values.batch_repeat_interleave(counts_slot)
        cur_attn = cur_attn.expand((counts_slot, -1))

    # =====================================================================
    # ITERATIVE REFINEMENT: Speculative decoding with verification
    # =====================================================================
    # Each iteration verifies and potentially accepts more tokens
    # This implements the "speculative" aspect - predicting multiple tokens
    # then verifying them efficiently using KV cache
    for loop_iter in range(slot_size):  # noqa: B007
        if acceptance_mask.all():
            loop_iter = loop_iter - 1  # iteration not started so we don't count it inside TPF
            break  # All tokens accepted, exit loop

        # =====================================================================
        # DRAFT PHASE: Model predicts masked tokens
        # =====================================================================
        # Prepare masked input for model
        remaining = accepted_prefix_len
        input_tokens = current_slots[:, remaining:]
        input_pos = chosen_pos[:, remaining:]

        cur_tags = acceptance_mask[:, remaining:]
        # --- Draft: mask unverified tokens and predict ---
        # Mask unverified tokens with mask_id (they will be predicted by model)
        # Only tokens with acceptance_mask == 0 need prediction (those not yet accepted)
        masked_input = torch.where(cur_tags.bool(), input_tokens, mask_id)

        # Prediction phase: model predicts masked tokens
        # NOTE: use_cache=False is critical here because:
        # 1. We're processing only a subset of tokens (from accepted_prefix_len onwards)
        # 2. The KV cache already contains the full context up to accepted_prefix_len
        # 3. If we used cache here, it would append to the cache incorrectly
        # 4. We need fresh logits for the draft tokens without affecting the main cache
        # 5. The verification phase (next) will properly update the cache with use_cache=True
        draft_outputs = model(
            input_ids=masked_input,
            position_ids=input_pos,
            attention_mask=torch.hstack((cur_attn, torch.ones_like(masked_input))),
            past_key_values=past_key_values,
            use_cache=False,
        )
        past_key_values.crop(-draft_outputs.logits.shape[1])
        draft_logits = suppress_token(draft_outputs.logits, input_pos, position_limitation, eos_token_id)
        proposed = torch.argmax(draft_logits, dim=-1)

        # Update tokens with draft predictions where still masked
        input_tokens = torch.where(cur_tags.bool(), input_tokens, proposed)
        current_slots[:, remaining:] = input_tokens

        # =====================================================================
        # VERIFICATION PHASE: Compute true probabilities for proposed tokens
        # =====================================================================
        # After draft phase, we have predictions for all remaining tokens
        # Now we verify these predictions with a single forward pass
        verify_probs, verify_outputs = _verify_and_update_probs(
            model,
            input_tokens,
            input_pos,
            torch.hstack((cur_attn, torch.ones_like(input_tokens))),
            past_key_values,
            cur_x,
            temperature=1,
            repetition_penalty=repetition_penalty,
        )

        # Update acceptance mask based on token_threshold
        new_prob_mask = verify_probs > token_threshold

        # Keep at least one token per slot (first token must always be accepted)
        # This ensures we make progress even if draft is poor
        keep_first = F.pad(acceptance_mask[:, remaining:], (1, 0), value=1)[:, :-1]
        new_prob_mask[keep_first.bool()] = True

        # Update cumulative tags (early stopping after first rejection)
        # Once a token is rejected, all subsequent tokens in that block are also rejected
        new_tags = torch.cumprod(new_prob_mask.int(), dim=-1)
        acceptance_mask[:, remaining:] = new_tags

        # Check for EOS token in newly verified region
        newly_verified = acceptance_mask[:, remaining:].bool()
        eos_in_new = (current_slots[:, remaining:] == eos_token_id) & newly_verified

        early_stopping_in_new = stopping_criteria(
            torch.hstack(
                (
                    cur_x.expand((current_slots.shape[0], -1)),
                    torch.where(newly_verified, current_slots[:, remaining:], mask_id),
                )
            ),
            None,
            new_token_length=current_slots[:, remaining:].shape[1],
        )
        first_eos_slot_idx = current_slots.shape[0]
        first_early_stopping_idx = current_slots.shape[0]
        if eos_in_new.any():
            first_eos_slot_idx = torch.where(torch.any(eos_in_new, dim=1))[0][0].item()
        if early_stopping_in_new.any():
            first_early_stopping_idx = torch.where(early_stopping_in_new)[0][0].item()

        if eos_in_new.any() or early_stopping_in_new.any():
            eos_found = True
            # Find first slot that contains EOS in newly verified region
            first_eos_slot_idx = min(first_early_stopping_idx, first_eos_slot_idx)

            # Truncate at EOS token to stop generation
            current_slots = current_slots[: first_eos_slot_idx + 1]
            acceptance_mask = acceptance_mask[: first_eos_slot_idx + 1]
            acceptance_mask[first_eos_slot_idx] = 1
            chosen_pos = chosen_pos[: first_eos_slot_idx + 1]
            topk_indices = topk_indices[: first_eos_slot_idx + 1]
            # Crop KV cache to exclude rejected slots
            if verify_outputs.past_key_values is not None:
                verify_outputs.past_key_values.batch_select_minibatch(first_eos_slot_idx + 1)

        # Advance accepted prefix length based on newly verified tokens
        cur_tags = acceptance_mask[:, remaining:]
        len_per_block = cur_tags.sum(dim=1)
        newly_accepted_len = len_per_block.min().item()
        if newly_accepted_len > 0:
            # Only advance if there are still unverified tokens
            add_len = newly_accepted_len if acceptance_mask.all() else newly_accepted_len - 1
            accepted_prefix_len += add_len

            # Update KV cache with new context
            past_key_values = verify_outputs.past_key_values
            if past_key_values is not None:
                # make new length: cur_x.shape[1] + accepted_prefix_len
                # past_key_values.crop(-(slot_size - accepted_prefix_len))
                past_key_values.crop(cur_x.shape[1] + accepted_prefix_len)

        if eos_found:
            break

    # =====================================================================
    # UPDATE METRICS: Track generation efficiency
    # =====================================================================
    # Update TPF (tokens per forward) metric for efficiency measurement
    # Higher TPF = more efficient generation (more tokens per model forward pass)
    # Formula: (tokens processed) / (forward passes * 2 + 2) approximates efficiency
    # The factor of 2 accounts for draft + verification forward passes
    # TPF accounting
    sum_TPF_add = (slot_size * counts_slot) / (loop_iter * 2 + 2)
    forward_count_add = 1

    # Extract AR KV cache for the last slot (most recent tokens)
    # This preserves context for subsequent iterations
    ar_kv_cache = tuple((lp[0][:, :, -slot_size:, :], lp[1][:, :, -slot_size:, :]) for lp in past_key_values)
    past_key_values.crop(cur_x.shape[1])
    past_key_values.batch_select_indices(torch.tensor([0]).to(device))

    # Handle EOS in speculative slots
    eos_mask = current_slots == eos_token_id  # (k*cur_slot_size)
    # Create mask: keep all tokens up to and including first EOS
    # cumsum - mask gives us all positions before the first EOS
    # This ensures generation stops at the first EOS token
    keep_mask = (torch.cumsum(eos_mask.flatten().int(), dim=-1) - eos_mask.flatten().int()) == 0
    kept_tokens = current_slots.flatten()[keep_mask].reshape(batch_size, -1)
    kept_pos_ids = chosen_pos.flatten()[keep_mask].reshape(batch_size, -1)

    # Update KV cache with kept tokens
    # This ensures the cache contains only the accepted tokens
    if kept_tokens.numel() > 0:
        new_past = []
        for key, val in ar_kv_cache:
            num_heads = key.shape[1]
            head_dim = key.shape[3]
            flat_k = key.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)
            flat_v = val.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)
            new_past.append((flat_k[:, :, keep_mask, :], flat_v[:, :, keep_mask, :]))
        past_key_values.full_update(tuple(new_past))

    return {
        "kept_tokens": kept_tokens,
        "kept_pos_ids": kept_pos_ids,
        "past_key_values": past_key_values,
        "sum_TPF_add": sum_TPF_add,
        "forward_count_add": forward_count_add,
        "eos_found": eos_found,
        "first_eos_slot_idx": first_eos_slot_idx,
        "topk_indices": topk_indices,
    }


class Linear(torch.nn.Module):
    def __init__(self, alpha_0=1, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.alpha_0 = alpha_0

    def forward(self, t):
        t = (1 - self.eps) * t
        alpha_t = self.alpha_0 * (1 - t)
        dalpha_t = -self.alpha_0 * (1 - self.eps)
        return dalpha_t, alpha_t


# Copied from https://github.com/jdeschena/sdtt/blob/bbc54d5b3c5fcffd79602cff17ed34dde1f3eff6/src/sdtt/core/sampling/utils.py#L10
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf"), dim=-1):
    """Filter a distribution of logits using top-k/top-p (nucleus) filtering.
    Adapted from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
      logits (Tensor): Tensor of logits
      top_k (int, optional): Number of top values to keep.
          Deactivated if k is 0. Defaults to 0.
      top_p (float, optional): Cumulative mass to retain.
          Deactivated if p = 0. Defaults to 0.0.
      filter_value (float, optional): Fill value to replace
          the entries removed by top-k/top-p filtering.
          Defaults to -float('Inf').
      dim (int, optional): Dimension of the filtering. Defaults to -1.

    Returns:
        logits: Tensor whose axis `dim` was filtered.
    """
    if dim != -1:
        logits = torch.transpose(logits, dim, -1)

    assert top_k < logits.size(dim)
    if top_k > 0:
        # Remove all tokens with a probability less than
        # the last token of the top-k
        values, _ = torch.topk(logits, k=top_k, dim=-1)
        to_remove_mask = logits < torch.min(values, dim=-1, keepdim=True)[0]  # min returns a tuple (values, indices)
        logits[to_remove_mask] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cum_probs > top_p
        # Ensures at least one token is kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask_to_remove = torch.empty_like(sorted_indices_to_remove)
        mask_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[mask_to_remove] = filter_value

    if dim != -1:
        logits = torch.transpose(logits, dim, -1)

    return logits


def get_reverse_indices(indices):
    """
    indices: LongTensor of shape [B, N] representing permutations
    returns: LongTensor of shape [B, N] representing the inverse permutations
    """
    B, N = indices.shape
    reverse_indices = torch.empty_like(indices)
    arange = torch.arange(N, device=indices.device).unsqueeze(0).expand(B, -1)
    reverse_indices.scatter_(1, indices, arange)
    return reverse_indices


class Zarya(PreTrainedModel, GenerationMixin):
    """HF-compatible model."""

    config: ZaryaConfig
    config_class = ZaryaConfig
    base_model_prefix = "backbone"
    prefix = "backbone"
    _skip_keys_device_placement = ["past_key_values"]
    # Flash Attention support
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    # SDPA support
    _supports_sdpa = True
    # Flex Attention support
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    supports_gradient_checkpointing = True
    _can_compile_fullgraph = True
    # This flag signal that the model can be used as an efficient backend in TGI and vLLM
    # In practice, it means that they support attention (mask) interface functions, fully pass the kwargs
    # through all modules up to the Attention layer, can slice logits with Tensor, and have a default TP plan
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": GradientCheckpointingLayer,
        "attentions": nn.Module,
    }

    def __init__(self, config: ZaryaConfig):
        super().__init__(config)
        self.config: ZaryaConfig = config
        self.generation_config._from_model_config = False
        try:
            self.generation_config = ZaryaGenerationConfig.from_pretrained(
                self.name_or_path, **self.generation_config.to_dict()
            )
        except OSError:
            self.generation_config = ZaryaGenerationConfig.from_model_config(config)
        self.backbone: PreTrainedModel = getattr(transformers.models, self.config.backbone_class)(config)
        self.alpha_0 = config.alpha_0
        self.noise = Linear(self.alpha_0, config.noise_eps)
        self.vocab_size = config.vocab_size
        self.neg_infinity = -torch.inf
        self.time_conditioning = config.time_conditioning
        self.sampling_eps = config.sampling_eps
        self.T = config.T
        self.noise_sigma_max = -torch.log1p(-(1 - self.sampling_eps) * torch.tensor(1.0))  # hack for mdlm imitation
        # for generation_config.T=0 use auto_unmask -> num_steps will be 4 times less than masked tokens count:
        self.auto_unmask = 4
        # Current submodel should register its tied weights
        try:  # noqa: SIM105
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=True)  # False
        except AttributeError:
            pass

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.backbone.set_input_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        _model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )
        # NOTE(Lin): we need to override the generation config
        # because the generation config loaded in `from_pretrained`
        # does not include all the attributes of ZaryaGenerationConfig
        output_loading_info = kwargs.pop("output_loading_info", False)
        if output_loading_info:
            _model, loading_info = _model
        proxies = kwargs.pop("proxies", None)
        subfolder = kwargs.pop("subfolder", "")
        from_auto_class = kwargs.pop("_from_auto", False)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        _model._can_record_outputs = _model.backbone._can_record_outputs
        _model.generation_config._from_model_config = False
        _model.generation_config = ZaryaGenerationConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            proxies=proxies,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **{**kwargs, **vars(_model.generation_config)},
        )
        if output_loading_info:
            return _model, loading_info
        return _model

    def _tokens_unmasked_per_step(
        self,
        num_steps: int,
        remaining_tokens: Union[int, torch.Tensor],
        diffusion_phase_only: bool = False,
        ignore_noise_schedule: bool = False,
    ) -> tuple[list, list]:
        dt = 1 / num_steps
        if ignore_noise_schedule:
            base = int(remaining_tokens * self.alpha_0) // num_steps
            remainder = int(remaining_tokens * self.alpha_0) % num_steps
            num_transfer_tokens = torch.zeros(1, num_steps, device=self.device, dtype=torch.int64) + base
            num_transfer_tokens[0, :remainder] += 1
            num_tokens_to_unmask = num_transfer_tokens[num_transfer_tokens.nonzero(as_tuple=True)].tolist()
            timestep_of_unmask = [t.item() for t in torch.linspace(start=1, end=dt, steps=num_steps)][
                : len(num_tokens_to_unmask)
            ]
        else:
            num_tokens_to_unmask = []
            timestep_of_unmask = []
            for t in torch.linspace(start=1, end=dt, steps=num_steps, device=self.device):
                _, alpha_t = self.noise(t)
                _, alpha_s = self.noise(t - dt)
                probs = self.generation_config.unmask_probs_coef * (alpha_s - alpha_t) / (1 - alpha_t)
                distribution = Binomial(total_count=remaining_tokens, probs=probs)
                n_unmask = distribution.sample()

                if n_unmask != 0 and remaining_tokens > n_unmask:
                    n_unmask = n_unmask.int()
                    num_tokens_to_unmask.append(n_unmask.item())
                    timestep_of_unmask.append(t.item())
                    remaining_tokens -= n_unmask
            if (remaining_tokens != 0 and self.alpha_0 == 1) or diffusion_phase_only:
                num_tokens_to_unmask.append(remaining_tokens.item())
                timestep_of_unmask.append(t.item())
        return num_tokens_to_unmask, timestep_of_unmask

    def q_xt(self, x: torch.LongTensor, p_mask: torch.FloatTensor):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          p_mask: float torch.Tensor with shape (batch_size, 1).
        """
        special_tokens = torch.tensor(
            [self.config.pad_token_id, self.config.bos_token_id, self.config.eos_token_id],
            dtype=x.dtype,
            device=x.device,
        )
        if self.config.grouped_noise:
            move_indices = torch.full(x.shape, fill_value=False, dtype=torch.bool, device=x.device)
            num_tokens = torch.isin(
                x,
                special_tokens,
                invert=True,
            ).sum(-1)
            num_tokens_to_mask = (p_mask.squeeze(-1) * num_tokens).ceil().to(dtype=num_tokens.dtype, device=x.device)

            span_length = torch.minimum(
                num_tokens_to_mask,
                torch.full_like(num_tokens_to_mask, self.config.max_span_length),  # "expanded" constant
            )
            # PrefixCompl + FIS
            eos_indices = torch.nonzero(x == self.config.eos_token_id)
            if len(eos_indices) == 0:
                # no eos
                for row_idx, span in enumerate(span_length):
                    if span == 0:
                        continue  # skip iteration when span is 0
                    move_indices[row_idx, -span:] = torch.where(
                        torch.isin(x[row_idx, -span:], special_tokens, invert=True),
                        True,
                        move_indices[row_idx, -span:],
                    )
            else:
                for eos_row, eos_col in eos_indices:
                    if span_length[eos_row] == 0:
                        continue  # skip iteration when span is 0
                    end_idx = eos_col
                    start_idx = eos_col - span_length[eos_row]
                    move_indices[eos_row, start_idx:end_idx] = torch.where(
                        torch.isin(x[eos_row, start_idx:end_idx], special_tokens, invert=True),
                        True,
                        move_indices[eos_row, start_idx:end_idx],
                    )

            left_to_mask = (num_tokens_to_mask - move_indices.sum(-1)).clamp(min=0)
            span_length = torch.minimum(
                left_to_mask,
                torch.full_like(num_tokens_to_mask, self.config.max_span_length),  # "expanded" constant
            )

            # FIP
            bos_indices = torch.nonzero(x == self.config.bos_token_id)
            if len(bos_indices) == 0:
                # no bos
                for row_idx, span in enumerate(span_length):
                    if span == 0:
                        continue  # skip iteration when span is 0
                    move_indices[row_idx, :span] = torch.where(
                        torch.isin(x[row_idx, :span], special_tokens, invert=True),
                        True,
                        move_indices[row_idx, :span],
                    )
            else:
                for bos_row, bos_col in bos_indices:
                    if span_length[bos_row] == 0:
                        continue  # skip iteration when span is 0
                    start_idx = bos_col + 1
                    end_idx = start_idx + span_length[bos_row]
                    move_indices[bos_row, start_idx:end_idx] = torch.where(
                        torch.isin(x[bos_row, start_idx:end_idx], special_tokens, invert=True),
                        True,
                        move_indices[bos_row, start_idx:end_idx],
                    )
            # FIM
            left_to_mask = (num_tokens_to_mask - move_indices.sum(-1)).clamp(min=0)
            span_length = torch.minimum(
                left_to_mask,
                torch.full_like(num_tokens_to_mask, self.config.max_span_length),  # "expanded" constant
            )
            if span_length.min() <= 0:
                spans_left = torch.floor_divide(
                    left_to_mask, span_length.float().masked_fill(span_length <= 0, torch.inf)
                ).to(dtype=span_length.dtype)
            else:
                spans_left = torch.floor_divide(left_to_mask, span_length)
            clean_tokens = (num_tokens - num_tokens_to_mask).clamp(min=0)
            buffer_min = torch.floor_divide(clean_tokens, spans_left + 1)
            for row_idx, span in enumerate(span_length):  # filling with spans of mask
                if spans_left[row_idx] == 0:
                    continue  # skip iteration when no spans to fill
                filler = ([False] * buffer_min[row_idx] + [True] * span) * spans_left[row_idx]
                total_fill_len = (move_indices[row_idx].logical_not()).sum()
                move_indices_filler = torch.tensor(
                    filler + [False] * (total_fill_len - len(filler)),  # also fills "buffer_min" to the right
                    dtype=move_indices.dtype,
                    device=move_indices.device,
                )
                move_indices[row_idx, move_indices[row_idx].logical_not()] = torch.where(
                    torch.isin(x[row_idx, move_indices[row_idx].logical_not()], special_tokens, invert=True),
                    move_indices_filler,
                    False,
                )

            # ordinary noise
            left_to_mask = (num_tokens_to_mask - move_indices.sum(-1)).clamp(min=0)
            for row_idx, tokens_to_mask in enumerate(left_to_mask):
                if tokens_to_mask == 0:
                    continue  # skip iteration when no tokens_to_mask
                # 1. Create a 1D tensor of random permutations of indices
                indices = torch.randperm(
                    len(
                        move_indices[
                            row_idx,
                            move_indices[row_idx].logical_not()
                            & torch.isin(x[row_idx, :], special_tokens, invert=True),
                        ]
                    )
                )

                # 2. Select the first 'tokens_to_mask' indices
                selected_indices = indices[:tokens_to_mask].sort()[0]
                selected_indices_bigsize = torch.arange(len(move_indices[row_idx]), device=move_indices.device)[
                    move_indices[row_idx].logical_not() & torch.isin(x[row_idx, :], special_tokens, invert=True)
                ][selected_indices]
                move_indices[row_idx, selected_indices_bigsize] = torch.full_like(
                    move_indices[row_idx, : len(selected_indices)], True
                )
        else:
            move_indices = torch.rand(*x.shape, device=x.device) < p_mask
        xt = torch.where(move_indices, self.config.mask_token_id, x)
        return xt

    @staticmethod
    def scale_to_bounds(x: torch.Tensor, new_min: torch.Tensor | float, new_max: float) -> torch.Tensor:
        # Calculate min and max of the input tensor
        min_val = x.min()
        max_val = x.max()

        # Check if the input range is zero to avoid division by zero
        if max_val - min_val == 0:
            # If all values are the same, they remain the same within the new range if possible
            # Or you can choose to handle this case differently, e.g., return a tensor of new_min
            return torch.full_like(x, (new_min + new_max) / 2.0)

        # Normalize to [0, 1]: (x - min) / (max - min)
        normalized_x = (x - min_val) / (max_val - min_val)

        # Scale to [new_min, new_max]: normalized_x * (new_max - new_min) + new_min
        scaled_x = normalized_x * (new_max - new_min) + new_min

        return scaled_x

    def _sample_t(self, n: int):
        if self.config.sample_t_override > 0:
            t = torch.full((n,), fill_value=self.config.sample_t_override, device=self.device)
        else:
            _eps_t = torch.rand(n, device=self.device)
            if self.config.ordered_sampling:
                offset = torch.arange(n, device=self.device) / n
                _eps_t = (_eps_t / n + offset) % 1
            t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
            if (0 < self.config.sample_t_upper < 1) and t.max() > self.config.sample_t_upper:
                lower_bound = t.min()  # Lower bound (inclusive)
                upper_bound = self.config.sample_t_upper  # Upper bound (exclusive)

                t = self.scale_to_bounds(t, lower_bound, upper_bound)
        return t

    def _process_logits(self, logits: torch.Tensor):
        if logits.isnan().any():
            logger.warning(f"logits have nans: {logits.detach().data}")
            logits = torch.nan_to_num(logits)
        return logits

    @can_return_tuple
    @merge_with_config_defaults
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        timesteps: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast, TextDiffusionLMOutputWithPast]:
        masked_indices: Optional[torch.Tensor] = kwargs.pop("masked_indices", None)
        p_mask: Optional[torch.Tensor] = kwargs.pop("p_mask", None)
        answer_lengths: Optional[torch.Tensor] = kwargs.pop("answer_lengths", None)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        use_cache = (
            use_cache
            if use_cache is not None
            else (getattr(self.config, "use_cache", False) if not self.training else False)
        )

        if use_cache and past_key_values is None:
            past_key_values = DiffusionDynamicCache()

        batch_size = input_ids.shape[0]
        loss = None
        sequential_loss_per_token = None
        diffusion_loss_per_token = None
        acc_seq = None
        acc_dif = None
        do_sequential = self.config.diffusion_loss_proportion != 1
        do_diffusion = self.config.diffusion_loss_proportion != 0
        ### Slotted training
        if self.config.slotted_training:
            model_output: BaseModelOutputWithPast = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if self.config.extra_processing:
                logits = self._process_logits(model_output[0])
            else:
                logits = model_output[0]
            if not torch.isfinite(logits).all():
                logger.warning(f"logits has nans or infs: {logits.detach().data}")

            if labels is not None:
                com_logits = logits.float()
                # Flatten the tokens
                com_logits = com_logits.view(-1, self.config.vocab_size)
                labels = labels.view(-1)  # labels already shift
                masked_indices = masked_indices.view(-1)
                p_mask = p_mask.view(-1)
                answer_lengths = answer_lengths.view(-1)
                labels = labels.to(com_logits.device)

                if do_sequential:
                    # AR loss
                    AR_indices = masked_indices.logical_not()
                    if labels[AR_indices].max() == -100:
                        # if every target is ignored, then consider loss to be 0
                        AR_loss = torch.tensor(0.0).to(com_logits.device)
                    else:
                        AR_loss = cross_entropy(
                            com_logits[AR_indices],
                            labels[AR_indices],
                            ignore_index=-100,
                            reduction="mean",
                        )
                        # calc acc_seq
                        target = labels[AR_indices]
                        prediction = com_logits[AR_indices].detach().argmax(-1)
                        acc_seq = (
                            torch.where(target != -100, target.eq(prediction), False).sum() / (target != -100).sum()
                        )

                    sequential_loss_per_token = AR_loss

                else:
                    sequential_loss_per_token = torch.tensor([0.0]).to(input_ids.device)

                if do_diffusion:
                    # #########
                    # MDM loss
                    MDM_token_loss = (
                        cross_entropy(
                            com_logits[masked_indices],
                            labels[masked_indices],
                            ignore_index=-100,
                            reduction="none",
                        )
                        / p_mask[masked_indices]
                    )
                    # calc acc_dif
                    target = labels[masked_indices]
                    prediction = com_logits[masked_indices].detach().argmax(-1)
                    acc_dif = torch.where(target != -100, target.eq(prediction), False).sum() / (target != -100).sum()
                    MDM_loss = torch.sum(MDM_token_loss / answer_lengths[masked_indices]) / batch_size
                    diffusion_loss_per_token = MDM_loss

                else:
                    diffusion_loss_per_token = torch.tensor([0.0]).to(input_ids.device)

                loss = (
                    self.config.diffusion_loss_proportion * diffusion_loss_per_token
                    + (1 - self.config.diffusion_loss_proportion) * sequential_loss_per_token
                )
            logits_predicted = logits
        else:
            if isinstance(timesteps, int):
                self.T = timesteps
                timesteps = None

            ##### create noisy input (one for all steps)
            if timesteps is None:
                timesteps = self._sample_t(batch_size)
            assert timesteps.shape[0] == batch_size
            if self.T > 0:
                timesteps = (timesteps * self.T).to(torch.int)
                timesteps = timesteps / self.T
                # timesteps \in {1/T, 2/T, ..., 1}
                timesteps += 1 / self.T

            if self.config.simple_masking:
                p_mask = timesteps.unsqueeze(-1)
            else:
                dalpha_t, alpha_t = self.noise(timesteps)
                alpha_t = alpha_t.unsqueeze(-1)
                p_mask = 1 - alpha_t
            assert p_mask.ndim == 2

            noisy_input = self.q_xt(input_ids, p_mask=p_mask)  # noisy sample
            if self.config.noise_sorting:
                # sort inputs and targets before passing to the model
                sort_idx = _sort_indices_only(
                    noisy_input, shuffle=self.config.diffusion_shuffle, mask_token_id=self.config.mask_token_id
                )
                noisy_input_sorted = torch.gather(noisy_input, dim=1, index=sort_idx)
                input_sorted = torch.gather(input_ids, dim=1, index=sort_idx)
                attention_mask_sorted = None
                if attention_mask is not None:
                    attention_mask_sorted = torch.gather(attention_mask, dim=1, index=sort_idx)
                sort_idx_reversed = get_reverse_indices(sort_idx)
                if cache_position is None:
                    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                    cache_position: torch.Tensor = torch.arange(
                        past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
                    )

                if position_ids is None:
                    position_ids = cache_position.unsqueeze(0)
                    if batch_size != position_ids.shape[0]:
                        position_ids = position_ids.expand(batch_size, -1)
                position_ids = torch.gather(position_ids, dim=1, index=sort_idx)
                x0 = input_sorted
                x_noisy = noisy_input_sorted
            else:
                x0 = input_ids
                x_noisy = noisy_input
                sort_idx = None
                sort_idx_reversed = None

            ##### end create noisy input
            logits_output = []

            if do_sequential:
                #### sequential AR-like, no sorting, no masking
                attention_mask_sequential = attention_mask_sorted if self.config.noise_sorting else attention_mask
                model_output = self.backbone.forward(
                    x0,  # clean input
                    attention_mask=attention_mask_sequential,
                    position_ids=position_ids,
                )
                if self.config.extra_processing:
                    logits = self._process_logits(model_output[0])
                else:
                    logits = model_output[0]
                if not torch.isfinite(logits).all():
                    logger.warning(f"logits has nans or infs: {logits.detach().data}")
                dont_learn = (
                    x_noisy != self.config.mask_token_id
                )  # clean, not masked tokens - we don't want to learn on them
                if attention_mask is not None:
                    dont_learn = torch.logical_or(dont_learn, torch.logical_not(attention_mask_sequential))
                target = torch.where(dont_learn, -100, x0)

                loss = self.loss_function(
                    logits=logits,
                    labels=target,
                    vocab_size=self.vocab_size,
                    num_items_in_batch=kwargs.get("num_items_in_batch"),
                    add_loss_path=self.config.add_loss_path,
                )
                # calc acc_seq
                prediction = logits.detach().argmax(-1)
                acc_seq = torch.where(target != -100, target.eq(prediction), False).sum() / (target != -100).sum()
                # output sorted back and detached logits to properly work with metrics
                if self.config.noise_sorting:
                    logits_sorted_back = torch.gather(
                        logits.detach(),
                        dim=1,
                        index=sort_idx_reversed.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
                    ).contiguous()
                else:
                    logits_sorted_back = logits.detach()
                # scale detached logits output
                logits_output.append(
                    logits_sorted_back
                    - torch.logsumexp(logits_sorted_back, dim=-1)
                    .to(logits_sorted_back.dtype)
                    .unsqueeze(-1)
                    .expand(-1, -1, logits_sorted_back.shape[-1])
                )
                if self.config.unnormalized_loss:
                    num_recons = logits.shape[0]
                elif attention_mask is not None:
                    num_recons = attention_mask_sequential.sum()
                else:
                    num_recons = logits.shape[1]
                sequential_loss = loss.sum()
                sequential_loss_per_token = sequential_loss / num_recons

                #### END sequential AR-like, no sorting, no masking

            else:
                sequential_loss_per_token = torch.tensor([0.0]).to(input_ids.device)

            if do_diffusion:
                if self.config.noise_sorting:
                    valid_tokens_diffusion = attention_mask_sorted
                else:
                    valid_tokens_diffusion = attention_mask
                    sort_idx = None
                    sort_idx_reversed = None

                model_output: BaseModelOutputWithPast = self.backbone.forward(
                    x_noisy,
                    attention_mask=valid_tokens_diffusion,
                    position_ids=position_ids,
                )

                if self.config.extra_processing:
                    logits = self._process_logits(model_output[0])
                else:
                    logits = model_output[0]
                if not torch.isfinite(logits).all():
                    logger.warning(f"logits has nans or infs: {logits.detach().data}")
                # -100 we don't want to learn, x0 we want to learn
                dont_learn = (
                    x_noisy != self.config.mask_token_id
                )  # clean, not masked tokens - we don't want to learn on them
                if attention_mask is not None:
                    dont_learn = torch.logical_or(dont_learn, torch.logical_not(valid_tokens_diffusion))
                target = torch.where(dont_learn, -100, x0)
                if self.config.simple_masking:
                    loss_scale = 1 / p_mask
                else:
                    loss_scale = -dalpha_t / p_mask  # p_mask == 1 - alpha_t
                loss = self.loss_function(
                    logits=logits,
                    labels=target,
                    loss_scale=loss_scale,
                    vocab_size=self.vocab_size,
                    num_items_in_batch=kwargs.get("num_items_in_batch"),
                    add_loss_path=self.config.add_loss_path,
                )
                # calc acc_dif
                prediction = logits.detach().argmax(-1)
                acc_dif = torch.where(target != -100, target.eq(prediction), False).sum() / (target != -100).sum()
                # output sorted back and detached logits to properly work with metrics
                if self.config.noise_sorting:
                    logits_sorted_back = torch.gather(
                        logits.detach(),
                        dim=1,
                        index=sort_idx_reversed.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
                    ).contiguous()
                else:
                    logits_sorted_back = logits.detach()
                # scale detached logits output
                logits_output.append(
                    logits_sorted_back
                    - torch.logsumexp(logits_sorted_back, dim=-1)
                    .to(logits_sorted_back.dtype)
                    .unsqueeze(-1)
                    .expand(-1, -1, logits_sorted_back.shape[-1])
                )

                if self.config.scale_by_batch:
                    num_diffusion = valid_tokens_diffusion.sum()
                elif self.config.unnormalized_loss:
                    num_diffusion = logits.shape[0]
                else:
                    num_diffusion = torch.logical_not(dont_learn).sum()
                diffusion_loss = loss.sum()
                diffusion_loss_per_token = diffusion_loss / num_diffusion
            else:
                diffusion_loss_per_token = torch.tensor([0.0]).to(input_ids.device)

            loss = (
                self.config.diffusion_loss_proportion * diffusion_loss_per_token
                + (1 - self.config.diffusion_loss_proportion) * sequential_loss_per_token
            )
            logits_predicted = (
                logits_output[0].add(logits_output[1]).contiguous() if len(logits_output) > 1 else logits_output[0]
            )

        return TextDiffusionLMOutputWithPast(
            loss=loss,
            logits=logits_predicted,
            past_key_values=model_output.past_key_values,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
            loss_seq=sequential_loss_per_token,
            loss_dif=diffusion_loss_per_token,
            acc_seq=acc_seq,
            acc_dif=acc_dif,
        )

    def loss_function(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        loss_scale: Optional[Union[torch.Tensor, int, float]] = None,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        add_loss_path: bool = False,
    ) -> torch.Tensor:
        batch_size = logits.shape[0]

        # Flatten the tokens
        logits = logits.view(-1, vocab_size)

        # Flatten the tokens
        labels = labels.view(-1)
        # Enable model parallelism
        labels = labels.to(logits.device)

        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()

        loss = cross_entropy(
            logits, labels, ignore_index=ignore_index, reduction="none"
        )  # "subs_parameterization" happens inside, sort of
        if add_loss_path:
            loss = loss * (1 + (-loss).detach().exp())
        if loss_scale is not None:
            loss = loss_scale * loss.view(batch_size, -1)
        else:
            loss = loss.view(batch_size, -1)

        return loss

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """
        Tries to infer position ids given attention mask and past kv cache length. All instances when
        `position_ids=None` should call this method.
        """
        # `input_ids` may be present in the model kwargs, instead of being the main input (e.g. multimodal model)
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        seq_length = inputs_tensor.shape[1]

        if (attention_mask := model_kwargs.get("attention_mask")) is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            # We need this as otherwise padding tokens appear as -1 in position
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        else:
            past_length = 0
            if (cache := model_kwargs.get("past_key_values")) is not None:
                past_length = cache.get_seq_length()

            position_ids = torch.arange(seq_length + past_length, dtype=torch.long, device=inputs_tensor.device)
            position_ids = position_ids.unsqueeze(0)
        return position_ids

    def _get_deprecated_gen_repo(
        self,
        generation_mode: GenerationMode,
        trust_remote_code: bool,
        custom_generate: str | None = None,
    ) -> str | None:
        """
        Returns the Hub repo for a deprecated generation mode, if any.
        """
        if custom_generate is not None or "/" not in (repo := GENERATION_MODES_MAPPING[generation_mode]):
            return None

        logger.warning_once(
            f"{generation_mode.name.replace('_', ' ').title()} was moved to a `custom_generate` repo: https://hf.co/{repo}. "
            f"To prevent loss of backward compatibility, add `custom_generate='{repo}'` "
            "to your `generate` call before v4.62.0."
        )
        if not trust_remote_code:
            raise ValueError(
                f"{generation_mode.name.replace('_', ' ').title()} requires `trust_remote_code=True` in your `generate` call, "
                f"since it loads https://hf.co/{repo}."
            )
        return repo

    def _extract_generation_mode_kwargs(
        self,
        custom_generate,
        kwargs,
        synced_gpus,
        assistant_model,
        streamer,
    ) -> dict[str, Any]:
        """
        Extracts and returns the generation mode related keyword arguments from the provided kwargs.
        """
        generation_mode_kwargs = {
            "tokenizer": kwargs.pop("tokenizer", None),
            "assistant_tokenizer": kwargs.pop("assistant_tokenizer", None),
            "assistant_model": assistant_model,
            "streamer": streamer,
        }
        world_size = _get_torch_distributed_world_size()
        generation_mode_kwargs["synced_gpus"] = (
            (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and world_size > 1
            if synced_gpus is None
            else synced_gpus
        )
        generation_mode_kwargs = {k: v for k, v in generation_mode_kwargs.items() if v is not None}
        # Custom_generate callables can have their own set of arguments
        # To extract them, we compare the signature with the standard _sample method
        if isinstance(custom_generate, Callable):
            usual_mode_kwargs = inspect.signature(GenerationMixin._sample).parameters.keys()
            custom_generate_kwargs = inspect.signature(custom_generate).parameters.keys()
            new_custom_keys = custom_generate_kwargs - usual_mode_kwargs
            generation_mode_kwargs = {k: kwargs.pop(k) for k in new_custom_keys if k in kwargs}
        return generation_mode_kwargs

    @torch.no_grad()
    def generate_samples(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        sequential_phase_only=False,
        diffusion_phase_only=False,
        **model_kwargs,
    ):
        """
        Generate samples from the model.
        """
        assert not (sequential_phase_only and diffusion_phase_only), (
            "diffusion_phase_only and sequential_phase_only can't be both True"
        )
        num_steps = self.generation_config.T
        ignore_noise_schedule = self.generation_config.ignore_noise_schedule
        batch_size = input_ids.shape[0]

        local_num_tokens = input_ids.shape[1]
        masked = input_ids == self.config.mask_token_id
        if attention_mask is not None:
            masked = torch.logical_and(masked, attention_mask)
        masked_tokens_count = masked.sum(dim=1)
        if num_steps <= 0:
            num_steps = torch.ceil(masked_tokens_count.max() / self.auto_unmask).int().item()
        if ignore_noise_schedule and masked_tokens_count.max() < num_steps:
            num_steps = masked_tokens_count.max().item()
        unmasked_tokens = local_num_tokens - masked_tokens_count  # known tokens, clean tokens

        unmask_k_tokens, unmask_timesteps = self._tokens_unmasked_per_step(
            num_steps,
            masked_tokens_count.max(),
            diffusion_phase_only=diffusion_phase_only,
            ignore_noise_schedule=ignore_noise_schedule,
        )
        num_diffusion_tokens = sum(unmask_k_tokens)
        num_sequential_tokens = masked_tokens_count.max() - num_diffusion_tokens

        if sequential_phase_only:
            shuffling = self.config.sequential_shuffle
            keep_mask_unshuffled = True
        else:
            shuffling = self.config.diffusion_shuffle
            keep_mask_unshuffled = False

        sort_idx = _sort_indices_only(
            input_ids,
            mask_token_id=self.config.mask_token_id,
            masked=masked,
            shuffle=shuffling,
            keep_masks_unshuffled=keep_mask_unshuffled,
        )
        # for tokens to be generated by sequential, don't shuffle (set order from left to right)
        sort_idx[:, (unmasked_tokens.min() + num_diffusion_tokens) :] = (
            sort_idx[:, (unmasked_tokens.min() + num_diffusion_tokens) :].sort().values
        )
        x = torch.gather(input_ids, dim=1, index=sort_idx)
        if sort_idx is not None:  # attention mask should be sorted accordingly
            attention_mask = torch.gather(attention_mask, dim=1, index=sort_idx)
            position_ids = torch.arange(start=0, end=x.shape[1]).to(device=x.device).to(dtype=torch.long).unsqueeze(0)
            if (batch_size := input_ids.shape[0]) != position_ids.shape[0]:
                position_ids = position_ids.expand(batch_size, -1)
            position_ids = torch.gather(position_ids, dim=1, index=sort_idx)

        if sequential_phase_only:
            unmask_k_tokens = [1] * masked_tokens_count.max()
        else:
            unmask_k_tokens = unmask_k_tokens + [1] * num_sequential_tokens

        assert sum(unmask_k_tokens) + unmasked_tokens.min() == input_ids.shape[1]

        kv_cache = self.generation_config.use_cache
        if kv_cache:
            past_key_values = past_key_values
            if past_key_values is None:
                past_key_values = DiffusionDynamicCache()
        else:
            past_key_values = None

        for i, k in enumerate(unmask_k_tokens):
            curr_k_start = unmasked_tokens.min()
            if uneven_slices := (
                batch_size > 1 and unmasked_tokens.unique().shape[0] > 1
            ):  # batch, may have different slices to fill
                real_unmasked = torch.where(
                    (k_difference := curr_k_start + k - unmasked_tokens).greater(0), k_difference, 0
                )
                indices_to_fill = [
                    (idx, torch.arange(unmasked_tokens[idx], unmasked_tokens[idx] + add_to_input))
                    for idx, add_to_input in enumerate(real_unmasked)
                    if add_to_input > 0
                ]
            else:
                indices_to_fill = slice(curr_k_start, curr_k_start + k)
            if i == 0:
                last_k_start = 0
            else:
                last_k_start = curr_k_start - unmask_k_tokens[i - 1]

            curr_k_end = curr_k_start + k
            if kv_cache:
                # expect x to be sorted
                current_input = x[:, last_k_start:curr_k_end]
                current_attention_mask = attention_mask[:, :curr_k_end]
                current_position_ids = position_ids[:, last_k_start:curr_k_end]
            else:
                current_input = x[:, :curr_k_end]
                current_attention_mask = attention_mask[:, :curr_k_end]
                current_position_ids = position_ids[:, :curr_k_end]

            output = self.backbone(
                input_ids=current_input,
                attention_mask=current_attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                labels=None,
                use_cache=kv_cache,
                cache_position=None,
            )
            logits = output.logits
            if kv_cache:
                past_key_values = output.past_key_values
                # need to store in cache only what was unmasked before this step
                past_key_values.crop(curr_k_start)
            if self.generation_config.use_float64:
                logits = logits.to(torch.float64)
            if 0 < self.generation_config.temperature < 1:
                logits[:, :, self.config.mask_token_id] = self.neg_infinity
                logits = logits / self.generation_config.temperature
            if self.generation_config.top_p < 1:
                logits[:, :, self.config.mask_token_id] = self.neg_infinity
                # top_k_top_p_filtering takes in logits (normalized or
                # unnormalized) and returns logits (unnormalized)
                logits = top_k_top_p_filtering(logits, top_p=self.generation_config.top_p)
            # logits is unnormalized, but that's okay
            # with the gumbel max trick because normalized and
            # unnormalized logits differ by a constant, i.e.,
            # the log normalizing constant, which doesn't
            # affect the argmax operation
            # generate noise on the fly to avoid memory issues
            u = torch.rand(
                (batch_size, k, logits.shape[2]),
                device=logits.device,
                dtype=logits.dtype,
            )
            noise = -torch.log(-torch.log(u))
            if kv_cache:
                y = (logits[:, (curr_k_start - last_k_start) :, :] + noise).argmax(-1)
            else:
                # doesn't matter if part was clean already - later we pick only specifics
                y = (logits[:, slice(curr_k_start, curr_k_start + k), :] + noise).argmax(-1)

            if uneven_slices:  # batch, may have different slices to fill
                for idx, coord in indices_to_fill:
                    x[idx, coord] = y[idx, -real_unmasked[idx] :]
                unmasked_tokens += real_unmasked
            else:
                x[:, indices_to_fill] = y
                unmasked_tokens += k

        if sort_idx is not None:
            sort_idx_reversed = get_reverse_indices(sort_idx)
            x = torch.gather(x, dim=1, index=sort_idx_reversed)
        return x

    @staticmethod
    @torch.no_grad()
    def generate_slotted(
        self: "PreTrainedModel",
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: ZaryaGenerationConfig,
        streamer: Optional = None,
        **model_kwargs,
    ) -> Union[ZaryaGenerationOutput, torch.LongTensor]:
        """
        Generate text using speculative decoding approach.

        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        return_dict_in_generate = generation_config.return_dict_in_generate

        batch_size = input_ids.shape[0]

        # `pad_token_id` is created on `inputs_tensor.device` in `_prepare_special_tokens`.
        # `inputs_tensor` and `input_ids` can live on different devices, so we need to
        # realign `pad_token_id` with `input_ids` to avoid cross-device ops below.
        if pad_token_id is not None:
            pad_token_id = pad_token_id.to(input_ids.device)

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )
        repetition_penalty: float = generation_config.repetition_penalty
        gen_length: int = generation_config.max_new_tokens or generation_config.max_length - input_ids.shape[1]
        temperature: float = generation_config.temperature
        mask_id: int = self.config.mask_token_id
        eos_token_id: int = self.config.eos_token_id
        slot_size: int = generation_config.slot_size
        serial_num_blocks: int = generation_config.serial_num_blocks
        slot_threshold: float = generation_config.slot_threshold
        token_threshold: float = generation_config.token_threshold

        # ======================================================
        # INITIALIZATION PHASE
        # ======================================================
        sum_TPF = 0.0
        forward_count = 0
        device = input_ids.device

        # --- Initialize generation state ---
        gen_x, gen_pos_ids, cur_x, prompt_pos_ids, cur_attn = _init_generation_state(
            input_ids, gen_length, mask_id, batch_size, **model_kwargs
        )
        gen_attn = torch.ones_like(gen_x, device=device)
        # Current context: prompt tokens positions (maybe after reorder)
        cur_pos = prompt_pos_ids.clone()
        # Flag to indicate if EOS token was generated
        eos_flag = False
        # Find pos where to suppress early EOS (for FIP and FIM tasks)
        max_existing_pos = prompt_pos_ids.max(dim=-1, keepdim=True).values

        inner_length = (gen_pos_ids < max_existing_pos).sum(dim=-1).item()
        inner_blocks = []
        if inner_length > 0:
            inner_blocks, _, _ = _build_blocks(inner_length, serial_num_blocks, slot_size)
        blocks, slot_size, serial_num_blocks = _build_blocks(gen_length, serial_num_blocks, slot_size, inner_length)
        blocks = inner_blocks + blocks

        # ======================================================
        # KV CACHE INITIALIZATION
        # ======================================================
        # KV cache stores key-value pairs from attention layers
        # This allows efficient generation by avoiding recomputation of past tokens
        past_key_values = None  # KV cache for autoregressive generation

        # ======================================================================
        # MAIN LOOP: iterate over blocks
        # ======================================================================
        for block in blocks:
            block_start = block["start"]
            block_end = block["end"]
            cur_slot_size = block["slot_size"]

            cur_gen_x = gen_x[:, block_start:block_end]
            cur_gen_pos_ids = gen_pos_ids[:, block_start:block_end]
            cur_gen_attn = gen_attn[:, block_start:block_end]

            # Reshape into slots: (batch, num_slots, slot_size)
            num_slots = cur_gen_x.numel() // cur_slot_size
            # Each slot contains cur_slot_size consecutive tokens
            slots_x = cur_gen_x.reshape(batch_size, num_slots, cur_slot_size)
            slots_pos = cur_gen_pos_ids.reshape(batch_size, num_slots, cur_slot_size)
            slots_attn = cur_gen_attn.reshape(batch_size, num_slots, cur_slot_size)

            # ==================================================================
            # SLOT LOOP: process slots until all are accepted
            # ==================================================================
            # Iteratively generate and verify slots within the current block
            # Continue until all slots in current block are processed
            while slots_x.numel() > 0:
                # Ensure proper shape for slot processing
                slots_x = slots_x.reshape(batch_size, -1, cur_slot_size)
                slots_pos = slots_pos.reshape(batch_size, -1, cur_slot_size)
                slots_attn = slots_attn.reshape(batch_size, -1, cur_slot_size)

                # Flatten for model input: (batch_size, num_slots * slot_size)
                flat_x = slots_x.reshape(batch_size, -1)
                flat_pos = slots_pos.reshape(batch_size, -1)
                flat_attn = slots_attn.reshape(batch_size, -1)

                # Replace tokens at prompt positions with actual prompt tokens
                # This should prevent overwriting prompt content with mask tokens
                prompt_overlap = torch.isin(flat_pos, prompt_pos_ids)
                if prompt_overlap.any():
                    flat_x[prompt_overlap] = input_ids[torch.isin(prompt_pos_ids, flat_pos)]

                # =====================================================================
                # MDM (Masked Diffusion Model) FORWARD PASS
                # =====================================================================
                # First iteration: concatenate prompt with generated slots
                # Subsequent iterations: only process generated slots (using KV cache)
                if past_key_values is None:
                    # First iteration: concatenate prompt with generated slots
                    # This builds the complete input sequence for the first forward pass
                    input_ids = torch.cat((cur_x, flat_x), dim=1)
                    input_pos = torch.cat((cur_pos, flat_pos), dim=1)
                    input_attn = torch.cat((cur_attn, flat_attn), dim=1)
                else:
                    # Subsequent iterations: only process generated slots
                    # KV cache already contains prompt context, so we only compute for new slots
                    input_ids = flat_x
                    input_pos = flat_pos
                    input_attn = flat_attn

                outputs = model_forward(
                    input_ids=input_ids,
                    position_ids=input_pos,
                    attention_mask=input_attn,
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=flat_x.shape[1],  # Calculate logits for the generated portion only
                )

                gen_logits = suppress_token(
                    outputs.logits, input_pos[:, -flat_x.shape[1] :], max_existing_pos, eos_token_id
                )

                # Update KV cache (cut to current context) and verify sync
                past_key_values = outputs.past_key_values
                past_key_values.crop(-flat_x.shape[1])
                # Verify KV cache is properly synchronized with current context
                assert cur_x.shape[-1] == past_key_values.layers[0].keys.shape[-2]

                # ==============================================================
                # 1. DRAFT GENERATION: use MDM logits to greedily predict tokens
                # ==============================================================
                # Apply Gumbel noise for sampling (if temperature > 0)
                logits_noised = add_gumbel_noise(gen_logits, temperature=temperature)
                logits_noised = _apply_repetition_penalty(logits_noised, cur_x, repetition_penalty)

                # Get most likely tokens (argmax)
                x0_gen = torch.argmax(logits_noised, dim=-1)  # (batch_size, num_slots * slot_size)
                # Reshape to block structure: (batch_size, num_slots, slot_size)
                x0_gen_slots = x0_gen.view(batch_size, -1, cur_slot_size)

                # =====================================================================
                # CONFIDENCE ESTIMATION
                # =====================================================================
                # Calculate confidence scores for generated tokens (probability)
                x0_p = _compute_token_probabilities(gen_logits, x0_gen)

                # Reshape to block structure
                x0_p_slots = x0_p.view(batch_size, -1, cur_slot_size)
                # The first token's probability represents the slot's overall confidence
                # Using only the first token as slot confidence is a simplification
                # that assumes the first token is representative of the slot's quality
                slot_conf = x0_p_slots[:, :, 0]  # (bsz, num_slots)  # first token = slot confidence

                # =====================================================================
                # BLOCK SELECTION: Identify confident slots
                # =====================================================================
                # Identify confident slots based on slot_threshold
                # Only slots with confidence above threshold are considered for acceptance
                # Select confident slots
                is_confident = slot_conf > slot_threshold
                counts_slot = is_confident.sum(dim=1).item()
                topk_indices = is_confident[0].nonzero(as_tuple=True)[0]

                # CRITICAL SAFETY MECHANISM:
                # If no slots are confident enough, select the most confident one
                # This ensures we always have at least one block to process
                # Without this, generation could stall entirely
                if counts_slot <= 0:
                    counts_slot = 1
                    _, topk_indices = torch.topk(slot_conf.squeeze(0), k=1)

                # Choose slot (sort indices for consistent processing order)
                topk_indices, _ = torch.sort(topk_indices)

                # Extract chosen slots for further processing
                chosen_slots = x0_gen_slots[0, topk_indices, :]
                chosen_pos = slots_pos[0, topk_indices, :]
                chosen_probs_draft = x0_p_slots[0, topk_indices, :]

                # ==============================================================
                # 2. VERIFY: single AR forward pass over chosen slots
                # ==============================================================
                # Use KV cache to efficiently verify the draft tokens
                # This is the key efficiency gain: verify multiple slots with one forward pass
                verify_probs, _ = _verify_and_update_probs(
                    model_forward,
                    chosen_slots.reshape(1, -1),
                    chosen_pos.reshape(1, -1),
                    torch.hstack((cur_attn, torch.ones_like(chosen_slots.reshape(1, -1)))),
                    past_key_values,
                    cur_x,
                    temperature,
                    repetition_penalty,
                )
                # Update slot probabilities with AR verification
                # Keep first token probability from draft (already computed),
                # update rest from verification to ensure consistency
                chosen_probs = chosen_slots.new_zeros(chosen_slots.shape, dtype=torch.float)
                # Keep draft probability for first token (more reliable as it's from the
                # full-context MDM pass). Update the rest with AR verification.
                chosen_probs[:, 0] = chosen_probs_draft[:, 0]
                chosen_probs[:, 1:] = verify_probs.reshape(-1, cur_slot_size)[:, 1:]

                # ==============================================================
                # 3. Phase A: Try to accept complete slots
                # ==============================================================

                result = _accept_verified_prefix(
                    chosen_slots,
                    chosen_pos,
                    chosen_probs,
                    topk_indices,
                    cur_x,
                    cur_pos,
                    cur_attn,
                    outputs.past_key_values,
                    x0_gen,
                    flat_pos,
                    cur_slot_size,
                    slots_x.shape[1],
                    token_threshold,
                    eos_token_id,
                    mask_id,
                    device,
                    stopping_criteria,
                    logits_processor,
                )
                if result is not None:  # prefix_slot_tag analog (except len(remain_indices)>0)
                    sum_TPF += result["sum_TPF_add"]
                    forward_count += result["forward_count_add"]
                    eos_flag = result["eos_found"]
                    cur_x = result["cur_x"]
                    cur_pos = result["cur_pos"]
                    cur_attn = result["cur_attn"]
                    indices_to_remove = result["indices_to_remove"]
                    past_key_values = result["past_key_values"]
                    topk_indices = result["topk_indices"]
                    prefix_slot_tag = result["prefix_slot_tag"]

                if prefix_slot_tag:
                    # =====================================================================
                    # UPDATE MASKS: Remove accepted slots from future processing
                    # =====================================================================
                    slots_x, slots_pos = _remove_accepted_slots(slots_x, slots_pos, indices_to_remove)

                    continue  # Reiterate with remaining slots

                else:
                    # No slots were accepted in prefix phase, update KV cache for next iteration
                    past_key_values = outputs.past_key_values
                    past_key_values.crop(-chosen_slots.reshape(1, -1).shape[1])
                    assert cur_x.shape[-1] == past_key_values.layers[0].keys.shape[-2]

                # ==============================================================
                # 4. Phase B: Speculative refinement for remaining tokens
                # ==============================================================
                refine_result = _speculative_refinement(
                    chosen_slots,
                    chosen_pos,
                    chosen_probs,
                    topk_indices,
                    cur_x,
                    cur_attn,
                    past_key_values,
                    cur_slot_size,
                    counts_slot,
                    token_threshold,
                    eos_token_id,
                    mask_id,
                    repetition_penalty,
                    model_forward,
                    device,
                    batch_size,
                    max_existing_pos,
                    stopping_criteria,
                    logits_processor,
                )

                sum_TPF += refine_result["sum_TPF_add"]
                forward_count += refine_result["forward_count_add"]

                kept_tokens = refine_result["kept_tokens"]
                kept_pos_ids = refine_result["kept_pos_ids"]
                past_key_values = refine_result["past_key_values"]

                # =====================================================================
                # APPEND ACCEPTED TOKENS: To current context
                # =====================================================================
                # Append accepted tokens to current context
                # These tokens are now verified and will form the basis for next iteration
                cur_x = torch.cat((cur_x, kept_tokens), dim=1)
                cur_pos = torch.cat((cur_pos, kept_pos_ids), dim=1)
                cur_attn = torch.cat((cur_attn, torch.ones_like(kept_tokens)), dim=1)

                # Verify KV cache is properly synchronized
                assert cur_x.shape[-1] == past_key_values.layers[0].keys.shape[-2]

                eos_in_loop = refine_result["eos_found"]
                first_eos_slot_idx = refine_result["first_eos_slot_idx"]
                accepted_indices = set(refine_result["topk_indices"].tolist())

                if eos_in_loop:
                    accepted_indices.update(range(first_eos_slot_idx, slots_x.shape[1]))
                    eos_flag = True

                # =====================================================================
                # REMOVE ACCEPTED Slots: From the mask for next iteration
                # =====================================================================
                slots_x, slots_pos = _remove_accepted_slots(slots_x, slots_pos, accepted_indices)

            if eos_flag:
                break

        # =====================================================================
        # FINALIZE: Reorder tokens by position and compute efficiency metric
        # =====================================================================
        # Reorder tokens by position (they might be out of order due to masking)
        _, reorder_idx = torch.sort(cur_pos, dim=-1)
        x = torch.gather(cur_x, dim=-1, index=reorder_idx)

        # Compute average tokens per forward pass (efficiency metric)
        # TPF (Tokens Per Forward) measures generation efficiency
        # Higher TPF = more efficient generation (more tokens per model forward pass)
        # A good speculative decoding implementation should have TPF > 1
        TPF = sum_TPF / max(forward_count, 1)

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            cache = None
            if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
                cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
                cache = model_kwargs[cache_key]

            return ZaryaGenerationOutput(
                sequences=x,
                past_key_values=cache,
                tokens_per_forward=TPF,
            )
        else:
            return x

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[ZaryaGenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        custom_generate: Optional[Union[str, Callable]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], list[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://huggingface.co/papers/2010.00904).
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            custom_generate (`str` or `Callable`, *optional*):
                One of the following:
                - `str` (Hugging Face Hub repository name): runs the custom `generate` function defined at
                  `custom_generate/generate.py` in that repository instead of the standard `generate` method. The
                  repository fully replaces the generation logic, and the return type may differ.
                - `str` (local repository path): same as above but from a local path. Local directories also
                  require `trust_remote_code=True` because the local `custom_generate/generate.py` is executed.
                - `Callable`: `generate` will perform the usual input preparation steps, then call the provided callable to
                  run the decoding loop.
                For more information, see [the docs](../../generation_strategies#custom-generation-methods).
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        synced_gpus = None
        assistant_model = None
        streamer = None
        # 0.a. If requested, load an arbitrary generation recipe from the Hub and run it instead
        trust_remote_code = kwargs.pop("trust_remote_code", None)

        if custom_generate is not None and isinstance(custom_generate, str):
            # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
            # they receive the same inputs as `generate`, with `model` instead of `self` and excluding the arguments to
            # trigger the custom generation. They can access to methods from `GenerationMixin` through `model`.
            global_keys_to_exclude = {
                "self",
                "kwargs",
                "global_keys_to_exclude",
                "trust_remote_code",
                "custom_generate",
            }
            generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
            generate_arguments.update(kwargs)

            custom_generate_function = self.load_custom_generate(
                custom_generate, trust_remote_code=trust_remote_code, **kwargs
            )
            return custom_generate_function(model=self, **generate_arguments)

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_mode_kwargs = self._extract_generation_mode_kwargs(
            custom_generate, kwargs, synced_gpus, assistant_model, streamer
        )

        # Check length values before updating the config with defaults. We'll use it later to define the final min/max length (# 6)
        has_default_max_length = (
            kwargs.get("max_length") is None
            and (generation_config is None or generation_config.max_length is None)
            and self.generation_config.max_length is None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None
            and (generation_config is None or generation_config.min_length is None)
            and self.generation_config.min_length is None
        )

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)

        generation_mode = generation_config.get_generation_mode(assistant_model)
        deprecated_mode_repo = self._get_deprecated_gen_repo(generation_mode, trust_remote_code, custom_generate)

        if isinstance(custom_generate, Callable):
            decoding_method = custom_generate

        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if generation_config._pad_token_tensor is not None and batch_size > 1 and len(inputs_tensor.shape) == 2:
                # When an attention mask is provided, use it to detect right-padding (more reliable than
                # checking token ids, which can produce false positives when pad_token_id == eos_token_id
                # or pad_token_id == bos_token_id, as is the case for Qwen3 and other models).
                attention_mask = model_kwargs.get("attention_mask", None)
                if attention_mask is not None and attention_mask.shape == inputs_tensor.shape:
                    # Right-padding means there are zeros (masked positions) at the end of some sequences
                    has_right_padding = torch.any(attention_mask[:, -1] == 0).item()
                else:
                    # Fallback: check if the last token is a pad token (original heuristic)
                    has_right_padding = torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
                if has_right_padding:
                    logger.warning(
                        "A decoder-only architecture is being used, but right-padding was detected! For correct "
                        "generation results, please set `padding_side='left'` when initializing the tokenizer."
                    )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and not self.config.is_encoder_decoder and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        kwargs_has_position_ids = model_kwargs.get("position_ids", None) is not None
        accepts_position_ids = "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        if not kwargs_has_position_ids and accepts_position_ids and not self.config.is_encoder_decoder:
            model_kwargs["position_ids"] = self._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # Expand inputs depending on the generation mode
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, generation_mode_kwargs.get("tokenizer"))

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[1]
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        try:  # transformers 4.56
            self._prepare_cache_for_generation(
                generation_config, model_kwargs, assistant_model, batch_size, max_cache_length
            )
        except TypeError:  # transformers 4.55
            self._prepare_cache_for_generation(
                generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. Prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        self.generation_config_default = deepcopy(self.generation_config)
        self.generation_config = generation_config

        if self.generation_config.slotted_generation:
            samples = []
            tpfs = []
            for idx in range(input_ids.shape[0]):
                model_kwargs_inner = {
                    key: (value[idx].unsqueeze(0) if key in {"attention_mask", "position_ids"} else value)
                    for key, value in model_kwargs.items()
                }
                result = self.generate_slotted(
                    self,
                    input_ids=input_ids[idx].unsqueeze(0),
                    logits_processor=prepared_logits_processor,
                    stopping_criteria=prepared_stopping_criteria,
                    generation_config=generation_config,
                    **model_kwargs_inner,
                )
                if generation_config.return_dict_in_generate:
                    sample = result.sequences
                    tpf = result.tokens_per_forward
                    tpfs.append(tpf)
                else:
                    sample = result
                samples.append(sample.squeeze(0))

            samples = pad_sequence(
                samples, batch_first=True, padding_value=self.generation_config.pad_token_id or self.config.pad_token_id
            )
            if generation_config.return_dict_in_generate:
                result.sequences = samples
                result.tokens_per_forward = torch.tensor(tpfs).mean().item()
            else:
                result = samples

        else:
            result = self.generate_samples(
                torch.cat(
                    (
                        input_ids,
                        self.config.mask_token_id
                        * torch.ones(
                            (input_ids.shape[0], self.generation_config.max_length - input_ids.shape[1]),
                            # self.config.mask_token_id,
                            device=input_ids.device,
                            dtype=input_ids.dtype,
                        ),
                    ),
                    dim=1,
                ),
                torch.cat(
                    (
                        model_kwargs["attention_mask"],
                        torch.ones(
                            model_kwargs["attention_mask"].shape[0],
                            self.generation_config.max_length - input_ids.shape[1],
                            device=model_kwargs["attention_mask"].device,
                            dtype=model_kwargs["attention_mask"].dtype,
                        ),
                    ),
                    dim=1,
                ),
                sequential_phase_only=self.generation_config.sequential_phase_only,
                diffusion_phase_only=self.generation_config.diffusion_phase_only,
            )

        return result


# Register the model so that it is available for transformer pipelines, auto-loading, etc.
ZaryaConfig.register_for_auto_class()
Zarya.register_for_auto_class("AutoModel")
Zarya.register_for_auto_class("AutoModelForCausalLM")
Zarya.register_for_auto_class("AutoModelForMaskedLM")
AutoConfig.register(ZaryaConfig.model_type, ZaryaConfig)
AutoModel.register(ZaryaConfig, Zarya)
