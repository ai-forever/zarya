import gzip
import json
import logging
import math
import os
import random
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets
import evaluate
import torch
import transformers
from clearml.backend_interface.task.repo import ScriptInfo
from datasets import load_dataset
from torch import Tensor
from torch.nn import Module
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from torch.utils import swap_tensors  # needs torch >= 2.3.0
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanMetric, MetricCollection
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import ClearMLCallback
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import ExportableState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalLoopOutput,
    EvalPrediction,
    SaveStrategy,
    get_last_checkpoint,
    speed_metrics,
)
from transformers.utils.versions import require_version

try:
    from transformers.trainer_utils import rotate_checkpoints

    rotate_checkpoints_old = False
except ImportError:
    rotate_checkpoints_old = True

logger = logging.getLogger(__name__)
LOG2 = torch.log(torch.tensor(2.0))
TRAIN_SHUFFLE_SEED = 31415926
IGNORE_INDEX = -100


class NLL(MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
          Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class NFEs(MeanMetric):
    pass


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor


def copy_parameters(target: Module, source: Module, use_buffers: bool = False):
    """Update target model parameters with source model parameters"""
    target_param = chain(target.parameters(), target.buffers()) if use_buffers else target.parameters()
    source_param = chain(source.parameters(), source.buffers()) if use_buffers else source.parameters()

    for p_target, p_source in zip(target_param, source_param):
        p_target.detach().copy_(p_source.detach().to(p_target.device))


def swap_parameters(target: Module, source: Module, use_buffers: bool = False):
    """Update target model parameters with source model parameters"""
    target_param = chain(target.parameters(), target.buffers()) if use_buffers else target.parameters()
    source_param = chain(source.parameters(), source.buffers()) if use_buffers else source.parameters()

    for p_target, p_source in zip(target_param, source_param):
        swap_tensors(p_target, p_source)


class TextDiffusionCallback(TrainerCallback):
    "A collection of callbacks"

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        self._trainer.train_metrics.reset()
        self._trainer.train_loss_seq.reset()
        self._trainer.train_loss_dif.reset()
        self._trainer.train_acc_seq.reset()
        self._trainer.train_acc_dif.reset()
        self._trainer.train_slot_size.reset()

    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        """
        Event called at the end of an epoch.
        """
        if self._trainer.save_epochs:
            # Save model checkpoint at the end of an epoch, ignoring save_total_limit
            checkpoint_folder = f"epoch-{int(state.epoch):03d}"
            run_dir = self._trainer._get_output_dir(trial=None)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self._trainer.save_model(output_dir, _internal_call=True)
            state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

    def on_train_begin(self, args: TrainingArguments, state, control, **kwargs):
        """
        Event called at the beginning of training.
        """
        if self._trainer.use_ema:
            # Set up EMA model wrapper (copies model actually)
            self._trainer.ema_model = AveragedModel(
                self._trainer.model_wrapped, multi_avg_fn=get_ema_multi_avg_fn(self._trainer.ema_decay)
            )

    def on_train_end(self, args: TrainingArguments, state, control, **kwargs):
        """
        Event called at the end of training.
        """
        # pass
        if self._trainer.ema_model is not None:
            model = kwargs["model"]
            update_bn(kwargs["train_dataloader"], self._trainer.ema_model)
            copy_parameters(model, self._trainer.ema_model.module)

    def on_optimizer_step(self, args: TrainingArguments, state, control, **kwargs):
        """
        Event called after the optimizer step but before gradients are zeroed out. Useful for monitoring gradients.
        """
        if self._trainer.ema_model is not None:
            model = kwargs["model"]
            self._trainer.ema_model.update_parameters(model)

    def on_evaluate(self, args: TrainingArguments, state, control, **kwargs):
        """
        Event called after an evaluation phase.
        """
        if self._trainer.ema_model is not None:
            model = kwargs["model"]
            # swapping back after evaluation
            swap_parameters(model, self._trainer.ema_model.module)


def scale_to_bounds(x: torch.Tensor, new_min: Union[torch.Tensor, float], new_max: float) -> torch.Tensor:
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


def sample_t(
    n: int,
    device: torch.device,
    base_p: Optional[Tensor] = None,
    sample_t_override: float = 0.0,
    offset_sampling: bool = False,
    sampling_eps: float = 1e-3,
    sample_t_upper: float = 1.0,
):
    if sample_t_override > 0:
        t = torch.full((n,), fill_value=sample_t_override, device=device)
    else:
        _eps_t = torch.rand(n, device=device) if base_p is None else base_p
        if offset_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - sampling_eps) * _eps_t + sampling_eps
        if (0 < sample_t_upper < 1) and t.max() > sample_t_upper:
            lower_bound = t.min()  # Lower bound (inclusive)
            upper_bound = sample_t_upper  # Upper bound (exclusive)

            t = scale_to_bounds(t, lower_bound, upper_bound)

    return t


def forward_process(
    input_ids: Tensor,
    attention_mask: Tensor,
    prompt_lengths: Tensor,
    mask_token_id: int,
    eps: float = 1e-3,
    slot_size: Optional[int] = None,
    ignore_index: int = -100,
    sequential_shuffle: bool = False,
    masked_shuffle: bool = False,
    slot_p_mask_variable: bool = False,
):
    """
    Transforms a batch of (prompt, answer) sequences into a hybrid training batch
    that combines two tasks within a single forward pass:

    1. **Sequential (AR-like) task**: For a randomly selected subset of "slots" (chunks
       of the answer), the tokens are kept visible. The model must predict each token
       auto-regressively (given previous tokens in the slot, but the label for the last
       token in each slot is set to ignore_index to avoid cross-slot leakage).
       These slots are also shuffled to break positional bias.

    2. **Masked Diffusion (MDM) task**: For the remaining slots, all tokens are replaced
       with mask_token_id. The model must reconstruct the original tokens, akin to a
       masked language modeling / diffusion objective.

    The proportion of slots assigned to the diffusion task is controlled by `p_mask`,
    which is sampled uniformly in [eps, 1] per sample to provide curriculum-like variety.

    Args:
        input_ids: Raw token IDs for the batch, shape (batch_size, seq_len).
        attention_mask: Binary mask indicating real tokens vs. padding, shape (batch_size, seq_len).
        prompt_lengths: Length of the prompt prefix (before the answer) for each sample, shape (batch_size, 1).
        mask_token_id: The special [MASK] token ID used for the diffusion masked slots.
        eps: Minimum mask probability floor (p_mask is sampled in [eps, 1]).
        slot_size: Size of each answer chunk (slot). If None, a random slot size is chosen per sample.
        ignore_index: The label value to ignore in the loss function (typically -100).

    Returns:
        A tuple of:
            - pro_input_ids:   Reordered input IDs with shuffled-visible + [MASK]-replaced slots.
            - pro_labels:      Target labels for loss computation (ignore_index for prompt + padding).
            - pro_masked_indices: Boolean mask indicating which positions belong to the diffusion task.
            - pro_p_masks:     The mask probability per sample, broadcast to (batch_size, seq_len).
            - pro_answer_lengths: Length of the original answer, broadcast to (batch_size, seq_len).
            - pro_position_ids: Position IDs that preserve the original token order within the sequence.
    """
    slot_size_randomization = False
    if slot_size is None:
        # When no slot_size is specified, randomize it per sample for regularization
        slot_size_randomization = True
        slot_size_set = [4, 8, 16, 32]

    device = input_ids.device

    batch_size, seq_length = input_ids.shape

    # Convert tensors to Python lists for per-sample processing
    input_ids = input_ids.tolist()
    prompt_lengths = prompt_lengths.squeeze(1).tolist()
    total_lengths = attention_mask.sum(dim=1).tolist()
    attention_mask = attention_mask.tolist()

    # Accumulators for the transformed batch
    pro_input_ids = []
    pro_attention_mask = []
    pro_labels = []
    pro_masked_indices = []
    pro_p_masks = []
    pro_answer_lengths = []
    pro_position_ids = []

    for batch_sample_idx in range(batch_size):
        if slot_size_randomization:
            # Pick a random slot size if slot_size_randomization is enabled
            slot_size = random.choice(slot_size_set)
        prompt_len = prompt_lengths[batch_sample_idx]
        total_len = int(total_lengths[batch_sample_idx])
        pad_len = seq_length - total_len

        # Split the sample into the prompt prefix and the answer suffix
        input_id = input_ids[batch_sample_idx][:prompt_len]
        end_id = input_ids[batch_sample_idx][prompt_len:total_len]
        len_input_id = len(input_id)
        answer_length = len(end_id)
        # Prompt tokens are never supervised (labels set to ignore_index)
        input_label = [ignore_index] * len_input_id
        input_attn = attention_mask[batch_sample_idx][:prompt_len]
        end_attn = attention_mask[batch_sample_idx][prompt_len:total_len]

        # Chunk the answer into slots of size `slot_size`
        answer_slots = [end_id[idx : idx + slot_size] for idx in range(0, answer_length, slot_size)]
        answer_slots_attn = [end_attn[idx : idx + slot_size] for idx in range(0, answer_length, slot_size)]

        num_answer_slots = len(answer_slots)
        if num_answer_slots == 0:
            # No answer tokens -- skip this sample
            continue

        # Build position IDs that reflect the original token ordering
        input_position_id = list(range(len_input_id))
        end_position_id = list(range(len_input_id, len_input_id + answer_length))
        answer_position_slots = [
            end_position_id[idx : idx + slot_size] for idx in range(0, len(end_position_id), slot_size)
        ]

        # Sample a mask probability p_mask uniformly in [eps, 1]
        # This determines what fraction of slots will be treated as the diffusion task
        t = random.random()
        p_mask = (1 - eps) * t + eps
        slot_mask = [random.random() < p_mask for _ in range(num_answer_slots)]

        # Split slots into unmasked (sequential/AR task) and masked (diffusion task)
        unmasked_indices = [idx for idx, masked in enumerate(slot_mask) if not masked]
        masked_indices = [idx for idx, masked in enumerate(slot_mask) if masked]

        # Shuffle the unmasked slots to break positional correlation with the original order
        if sequential_shuffle:
            random.shuffle(unmasked_indices)

        # Shuffle the masked slots
        if masked_shuffle:
            random.shuffle(masked_indices)

        # Build the reordered answer (end) part
        final_end_id = []
        final_end_attn = []
        final_answer_label = []
        final_masked_indices = []
        final_position_id = []
        final_p_masks = []

        # ---- Sequential (AR) task: unmasked slots ----
        # These tokens are kept visible. The label for each token is the *next* token
        # within the same slot (shifted left by 1). The last token of each slot gets
        # ignore_index to prevent it from predicting across the slot boundary.
        # The slots themselves are shuffled relative to the original order.
        for slot_idx in unmasked_indices:
            slot_content = answer_slots[slot_idx]
            final_end_id.extend(slot_content)
            final_end_attn.extend(answer_slots_attn[slot_idx])

            # AR shift: predict next token, last token in slot has no target
            ar_label = slot_content[1:] + [ignore_index]
            final_answer_label.extend(ar_label)

            final_masked_indices.extend([False] * (len(slot_content)))
            final_position_id.extend(answer_position_slots[slot_idx])
            final_p_masks.extend([p_mask] * len(slot_content))  # may be any float [eps, 1] as not used in AR loss

        # ---- Diffusion (MDM) task: masked slots ----
        # These slots are entirely replaced with [MASK]. The label is the original tokens,
        # and masked_indices=True tells the loss function to apply the diffusion loss.
        # Slots remain in their original relative order.
        for slot_idx in masked_indices:
            slot_content = answer_slots[slot_idx]
            final_end_id.extend([mask_token_id] * len(slot_content))
            final_end_attn.extend(answer_slots_attn[slot_idx])

            final_answer_label.extend(slot_content)

            final_masked_indices.extend([True] * len(slot_content))
            final_position_id.extend(answer_position_slots[slot_idx])
            slot_p_mask = [p_mask] * len(slot_content)
            if slot_p_mask_variable:
                slot_p_mask = sample_t(
                    len(slot_p_mask),
                    device=device,
                    base_p=torch.tensor(slot_p_mask, device=device),
                    offset_sampling=slot_p_mask_variable,
                    sampling_eps=eps,
                    sample_t_upper=p_mask,
                ).tolist()
            final_p_masks.extend(slot_p_mask)  # may be any float [eps, 1] as not used in AR loss

        # Assemble the full sequence: prompt + reordered answer + original padding
        final_input = input_id + final_end_id + input_ids[batch_sample_idx][total_len:]
        final_attn = input_attn + final_end_attn + attention_mask[batch_sample_idx][total_len:]
        final_label = input_label + final_answer_label + [ignore_index] * pad_len
        final_masked_indices = [False] * len_input_id + final_masked_indices + [False] * pad_len
        final_position_id = input_position_id + final_position_id + list(range(total_len, seq_length))
        final_p_masks = [p_mask] * len_input_id + final_p_masks + [p_mask] * pad_len

        # Sanity checks
        assert len(final_input) == len(final_label), f"{len(final_input)}, {len(final_label)}"
        assert len(final_input) == len(final_masked_indices), f"{len(final_input)}, {len(final_masked_indices)}"
        assert len(final_input) == len(final_position_id), f"{len(final_input)}, {len(final_position_id)}"

        pro_input_ids.append(torch.tensor(final_input))
        pro_attention_mask.append(torch.tensor(final_attn))
        pro_labels.append(torch.tensor(final_label))
        pro_masked_indices.append(torch.tensor(final_masked_indices))
        pro_p_masks.append(torch.tensor(final_p_masks))
        pro_answer_lengths.append(torch.tensor(answer_length))
        pro_position_ids.append(torch.tensor(final_position_id))

    # Broadcast scalar per-sample values back to full sequence length
    pro_answer_lengths = torch.stack(pro_answer_lengths).view(-1, 1).repeat(1, seq_length)

    return (
        torch.stack(pro_input_ids).to(device),
        torch.stack(pro_labels).to(device),
        torch.stack(pro_masked_indices).to(device),
        torch.stack(pro_p_masks).to(device),
        pro_answer_lengths.to(device),
        torch.stack(pro_position_ids).to(device),
        torch.stack(pro_attention_mask).to(device),
    )


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


class TextDiffusionTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args: Optional[TrainingArguments] = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        use_ema: bool = False,
        ema_decay: float = 1.0,
        save_epochs: bool = False,
        slotted_training: bool = False,
        slot_size_set: Optional[list[int]] = None,
        slot_step_borders: Optional[list[float]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.ema_model = None
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.save_epochs = save_epochs
        metric = MetricCollection(
            {
                "nll": NLL(),
                "bpd": BPD(),
                "ppl": Perplexity(),
            }
        )
        self.train_metrics = metric.clone(prefix="train/").to(self.args.device)
        self.valid_metrics = metric.clone(prefix="val/").to(self.args.device)
        self.train_loss_seq = MeanMetric().to(self.args.device)
        self.val_loss_seq = MeanMetric().to(self.args.device)
        self.train_loss_dif = MeanMetric().to(self.args.device)
        self.val_loss_dif = MeanMetric().to(self.args.device)
        self.train_acc_seq = MeanMetric().to(self.args.device)
        self.val_acc_seq = MeanMetric().to(self.args.device)
        self.train_acc_dif = MeanMetric().to(self.args.device)
        self.val_acc_dif = MeanMetric().to(self.args.device)
        self.train_slot_size = MeanMetric().to(self.args.device)
        self.val_slot_size = MeanMetric().to(self.args.device)
        self.model_accepts_loss_kwargs = False
        self.slotted_training = slotted_training
        self.slot_size_set: list = slot_size_set if slot_size_set is not None else [2, 4, 8, 16, 32]
        self.slot_step_borders: Optional[list[float]] = slot_step_borders
        if self.slot_step_borders is not None:
            self.slot_borders_calculated: Optional[torch.Tensor] = torch.tensor(
                self.slot_step_borders, device=self.args.device
            )
        else:
            self.slot_borders_calculated = None
        if self.slot_step_borders is not None and len(self.slot_step_borders) != len(self.slot_size_set):
            raise ValueError(
                f"slot_step_borders (len={len(self.slot_step_borders)}) must have the same length as "
                f"slot_size_set (len={len(self.slot_size_set)})"
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.slotted_training:
            if self.slot_step_borders is None:
                # Distribute slot_size_set uniformly across global_step range
                if self.slot_borders_calculated is None:
                    start, stop, num_parts = 0, self.state.max_steps, len(self.slot_size_set)
                    self.slot_borders_calculated = torch.linspace(start, stop, num_parts + 1)[1:].floor().int()
                cur_slot_idx = (
                    torch.cumprod(self.state.global_step > self.slot_borders_calculated, -1).argmin().item()
                    if self.state.global_step <= self.slot_borders_calculated.max()
                    else self.slot_borders_calculated.argmax().item()
                )
            else:
                # Use epoch-based borders provided as float values in self.slot_step_borders
                cur_slot_idx = (
                    torch.cumprod(self.state.epoch > self.slot_borders_calculated, -1).argmin().item()
                    if self.state.epoch <= self.slot_borders_calculated.max()
                    else self.slot_borders_calculated.argmax().item()
                )
            cur_slot_size = self.slot_size_set[cur_slot_idx]
            if inputs.get("prompt_lengths") is None:
                inputs["prompt_lengths"] = torch.zeros(
                    size=(inputs["input_ids"].shape[0], 1),
                    dtype=inputs["input_ids"].dtype,
                    device=inputs["input_ids"].device,
                )
            input_ids, labels, masked_indices, p_mask, answer_lengths, position_ids, attention_mask = forward_process(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["prompt_lengths"],
                self.processing_class.mask_token_id,
                slot_size=cur_slot_size,
                sequential_shuffle=self.model.config.sequential_shuffle,
                masked_shuffle=self.model.config.noise_sorting and self.model.config.diffusion_shuffle,
                slot_p_mask_variable=self.model.config.ordered_sampling,
            )
            sort_idx_reversed = get_reverse_indices(position_ids)
            inputs["input_ids"] = input_ids
            inputs["labels"] = labels
            inputs["masked_indices"] = masked_indices
            inputs["p_mask"] = p_mask
            inputs["answer_lengths"] = answer_lengths
            inputs["position_ids"] = position_ids
            inputs["attention_mask"] = attention_mask
        else:
            cur_slot_size = 0
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        if not return_outputs:
            self.train_metrics.update(loss)  # outputs
            if outputs.get("loss_seq") is not None:
                self.train_loss_seq.update(outputs.loss_seq)
                self.train_loss_dif.update(outputs.loss_dif)
            if outputs.get("acc_seq") is not None:
                self.train_acc_seq.update(outputs.acc_seq)
            if outputs.get("acc_dif") is not None:
                self.train_acc_dif.update(outputs.acc_dif)
            self.train_slot_size.update(cur_slot_size)
        elif return_outputs:
            if self.slotted_training:
                logits_sorted_back = torch.gather(
                    outputs.logits.detach(),
                    dim=1,
                    index=sort_idx_reversed.unsqueeze(-1).expand(-1, -1, outputs.logits.shape[-1]),
                ).contiguous()
                outputs["logits"] = logits_sorted_back
            self.valid_metrics.update(loss)
            if outputs.get("loss_seq") is not None:
                self.val_loss_seq.update(outputs.loss_seq)
                self.val_loss_dif.update(outputs.loss_dif)
            if outputs.get("acc_seq") is not None:
                self.val_acc_seq.update(outputs.acc_seq)
            if outputs.get("acc_dif") is not None:
                self.val_acc_dif.update(outputs.acc_dif)
            self.val_slot_size.update(cur_slot_size)

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        # we use as callback before evaluation
        if metric_key_prefix == "eval":
            self.valid_metrics.reset()
            self.val_loss_seq.reset()
            self.val_loss_dif.reset()
            self.val_acc_seq.reset()
            self.val_acc_dif.reset()
            self.val_slot_size.reset()

        if self.ema_model is not None:
            # Update bn statistics for the ema_model at the end
            update_bn(dataloader, self.ema_model)
            # by swapping we save memory in possible storage for restorable weights
            swap_parameters(self.model_wrapped, self.ema_model.module)  # need to swap back after evaluation

        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        if self.ema_model is not None:
            # Update bn statistics for the ema_model at the end
            # by swapping we save memory in possible storage for restorable weights
            swap_parameters(self.model_wrapped, self.ema_model.module)  # need to swap back after save

        super()._save(output_dir, state_dict)
        if self.ema_model is not None:
            # swap back after save
            swap_parameters(self.model_wrapped, self.ema_model.module)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                logs.update(speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen))

        if self.control.should_log and self.state.global_step >= self._globalstep_last_logged:
            train_metrics = self.train_metrics.compute()
            train_loss_seq = self.train_loss_seq.compute()
            train_loss_dif = self.train_loss_dif.compute()
            train_slot_size = self.train_slot_size.compute()
            acc_seq = (
                self.train_acc_seq.compute()
                if self.train_acc_seq.update_called
                else torch.tensor(0, device=self.train_acc_seq.device)
            )
            acc_dif = (
                self.train_acc_dif.compute()
                if self.train_acc_dif.update_called
                else torch.tensor(0, device=self.train_acc_dif.device)
            )
            logs = {
                **logs,
                **{k: v.item() for k, v in train_metrics.items()},
                "loss_seq": round(train_loss_seq.item(), 4),
                "loss_dif": round(train_loss_dif.item(), 4),
                "acc_seq": round(acc_seq.item(), 4),
                "acc_dif": round(acc_dif.item(), 4),
                "slot_size": train_slot_size.item(),
            }
            self.train_acc_seq.reset()
            self.train_acc_dif.reset()
            self.train_slot_size.reset()

        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _save_checkpoint(self, model: torch.nn.Module, trial) -> None:
        """Save model checkpoint, optimizer, scheduler, scaler, RNG states, and trainer state."""

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        if self.is_world_process_zero():
            self.save_model(output_dir, _internal_call=True)

        if (
            self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH, SaveStrategy.BEST]
            and self.state.best_global_step
        ):
            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if (not self.args.save_only_model) and self.is_world_process_zero():
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save and self.is_world_process_zero():
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save and self.is_world_process_zero():
            # we use mtime as default, filesystems without mtime support will be detected in `sort_checkpoints`
            if rotate_checkpoints_old:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
            else:
                rotate_checkpoints(
                    output_dir=run_dir,
                    save_total_limit=self.args.save_total_limit,
                    best_model_checkpoint=self.state.best_model_checkpoint,
                    use_mtime=True,
                )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    scratch_source: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a path to scratch templates storage"},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    pad_to_multiple_of: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad the embedding layer to a multiple depending on the device. ",
                "For NVIDIA GPUs, this will be a multiple of 8, for TPUs a multiple of 128.",
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa", metadata={"help": ("The attention implementation to use. ")}
    )
    ema_decay: float = field(default=0.9999, metadata={"help": "EMA decay rate."})
    enable_ema: bool = field(
        default=False,
        metadata={"help": "Enable Exponential Moving Average (EMA) of model weights during training"},
    )
    save_epochs: bool = field(
        default=False,
        metadata={"help": "Save model checkpoint at the end of an epoch, ignoring save_total_limit"},
    )
    sequential_shuffle: Optional[bool] = field(
        default=None,
        metadata={"help": "Shuffle tokens or token groups supposed for AR loss component"},
    )
    masked_shuffle: Optional[bool] = field(
        default=None,
        metadata={"help": "Shuffle tokens or token groups supposed for diffusion loss component"},
    )
    ordered_sampling: Optional[bool] = field(
        default=None,
        metadata={"help": "p_mask increase depending on token order from left to right"},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    train_split: Optional[str] = field(default=None, metadata={"help": "split setup string to train load."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    validation_split: Optional[str] = field(default=None, metadata={"help": "split setup string to validation load."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    do_group_texts: bool = field(
        default=True,
        metadata={"help": "Concate all samples and split the result into subsequences of max_seq_length tokens"},
    )
    do_shuffle: bool = field(
        default=False,
        metadata={"help": "Do shuffle datasets"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    custom_set: Optional[str] = field(default=None, metadata={"help": "custom_setup set to parse in code"})
    slotted_training: bool = field(
        default=False,
        metadata={"help": "Train with grouping by slots"},
    )
    slot_size_set: None | str | list[int] = field(
        default=None,
        metadata={
            "help": (
                "list of slot sizes (e.g., '4 8 16 32'). "
                "Overrides the default slot_size_set in TextDiffusionTrainer, "
                "which when None, defaults to [2, 4, 8, 16, 32]."
            )
        },
    )
    slot_step_borders: None | str | list[float] = field(
        default=None,
        metadata={
            "help": (
                "list of epoch thresholds for slot size scheduling "
                "(e.g., '[0.5  1.0  2.0]'). "
                "Must have the same length as slot_size_set. "
                "When None, slot sizes are distributed uniformly across all training steps."
            )
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = get_filename_ext(self.train_file)
                if extension not in ["csv", "json", "txt", "jsonl", "jsonl.gz"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = get_filename_ext(self.validation_file)
                if extension not in ["csv", "json", "txt", "jsonl", "jsonl.gz"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")
        if self.slot_size_set is not None and isinstance(self.slot_size_set, str):
            self.slot_size_set = [int(x) for x in self.slot_size_set.split()]
        if self.slot_step_borders is not None and isinstance(self.slot_step_borders, str):
            self.slot_step_borders = [float(x) for x in self.slot_step_borders.split()]


class CustomLoggingCallback(ClearMLCallback):
    _model_args = None
    _data_args = None

    def __init__(
        self,
        model_args: Optional[ModelArguments] = None,
        data_args: Optional[DataTrainingArguments] = None,
        args_file: Optional[str] = None,
    ):
        super().__init__()
        self._model_args = model_args
        self._data_args = data_args
        self._args_file = args_file

    def setup(self, args, state, model, tokenizer, **kwargs):
        super().setup(args, state, model, tokenizer, **kwargs)
        if state.is_world_process_zero and self._clearml_task is not None:
            # Report number of trainable parameters
            trainable_params = get_model_param_count(model, trainable_only=True)
            self._clearml_task.set_configuration_object(
                name="Num_trainable",
                config_text=f"{trainable_params:,} params ({trainable_params / 2**20:.2f}M, "
                f"{trainable_params / 2**30:.1f}B)",
                description="Number of trainable parameters",
            )
            # Saving processed model and data arguments
            if self._model_args is not None:
                self._copy_training_args_as_hparams(self._model_args, "ModelArguments")
            else:
                self._model_args = {}
            if self._data_args is not None:
                self._copy_training_args_as_hparams(self._data_args, "DataTrainingArguments")
            else:
                self._data_args = {}
            # Saving model architecture
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model_architecture = repr(model.module)
            else:
                model_architecture = repr(model)
            self._clearml_task.set_configuration_object(
                name="Model architecture",
                config_text=model_architecture,
                description="Model architecture printed from model class",
            )
            # Saving arguments original
            if self._args_file is not None and isinstance(self._args_file, (str, os.PathLike)):
                # add and upload local file artifact
                self._clearml_task.upload_artifact(
                    "arguments in file",
                    artifact_object=self._args_file,
                )
            elif isinstance(self._args_file, dict):
                # save non-default arguments from commandline
                self._clearml_task.upload_artifact(  # sorted for now
                    "parsed_args", artifact_object=self._args_file, auto_pickle=False, extension_name=".json"
                )
            # Get tags from an environment variable (set up like CLEARML_TASK_TAGS="tag_a,tag_b")
            env_tags_str = os.environ.get("CLEARML_TASK_TAGS")
            if env_tags_str:
                tags = [tag.strip() for tag in env_tags_str.split(",")]
                self._clearml_task.add_tags(tags)
            # Get git info from scratch source dir
            result, _ = ScriptInfo.get(
                filepaths=[self._model_args.scratch_source, self._model_args.model_name_or_path],
                create_requirements=False,
            )
            if result.script is not None:
                self._clearml_task.set_user_properties(**result.script)


def get_filename_ext(filename: str) -> str:
    if filename.endswith(".jsonl.gz"):
        return "jsonl.gz"
    elif filename.endswith(".jsonl.zstd"):
        return "jsonl.zstd"
    else:
        return filename.split(".")[-1]


def find_json_files(data_dir):
    data_path = Path(data_dir)
    jsonl_files = []
    for ext in ["jsonl.gz", "jsonl", "json", "jsonl.zst", "parquet"]:
        jsonl_files.extend(data_path.glob(f"**/*.{ext}"))
    print(f"Found {len(jsonl_files)} data files in {data_dir}")
    return [str(f) for f in jsonl_files]


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_dir, cache_dir, num_workers, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        logging.warning("Loading data...")  # noqa: LOG015

        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None
        assert tokenizer.pad_token_id is not None
        assert tokenizer.mask_token_id is not None

        data_paths = find_json_files(data_dir)
        list_data_dict = datasets.load_dataset(
            data_dir, data_files=data_paths, cache_dir=cache_dir, num_proc=num_workers
        )["train"]

        input_ids = []
        prompt_lengths = []
        attention_mask = []
        count = 0

        for idx, example in enumerate(list_data_dict):  # noqa: B007
            prompt = example["query"]
            target = example["response"]
            if target:
                messages = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                )

                input_id = inputs

                end_id = tokenizer.encode(text=target) + [tokenizer.eos_token_id]

                if len(input_id) + len(end_id) > tokenizer.model_max_length:
                    continue

                prompt_length = len(input_id)
                input_id.extend(end_id)

                input_ids.append(torch.tensor(input_id))
                prompt_lengths.append(torch.tensor(prompt_length))
                attention_mask.append(torch.tensor([1] * len(input_id)))

                count += 1
                if count % 1000 == 0:
                    print(f"Count reached: {count}")

        print(f"Number of items in the dataset: {count}")

        self.input_ids = input_ids
        self.labels = input_ids.copy()
        self.prompt_lengths = prompt_lengths
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(  # noqa: C408
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            prompt_lengths=self.prompt_lengths[i],
            attention_mask=self.attention_mask[i],
        )


class TokenizedDatasetBuilder:
    def __init__(self, tokenizer, sequence_length: int, logger):
        assert sequence_length > 0
        assert tokenizer.bos_token is not None
        assert tokenizer.eos_token is not None
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.logger = logger

    def read_jsonl_gz_files(self, data_dir: str) -> list[str]:
        """Get all JSON-compatible files from directory"""
        data_path = Path(data_dir)
        jsonl_files = []
        for ext in ["jsonl.gz", "jsonl", "json", "jsonl.zst", "parquet"]:
            jsonl_files.extend(data_path.glob(f"**/*.{ext}"))
        self.logger.info("Found %d data files in %s", len(jsonl_files), data_dir)
        return [str(f) for f in jsonl_files]

    def text_generator(self, file_paths: list[str]) -> Iterator[str]:
        """Generator that yields text from all jsonl.gz files"""
        for file_path in file_paths:
            try:
                if file_path.endswith("jsonl.gz"):
                    with gzip.open(file_path, "rt", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                sample = json.loads(line)
                                yield sample
                elif file_path.endswith("json"):
                    with open(file_path) as f:
                        for sample in json.load(f):
                            yield sample
                elif file_path.endswith("jsonl"):
                    with open(file_path) as f:
                        for line in f:
                            sample = json.load(line)
                            yield sample
                else:
                    raise NotImplementedError()

            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Error reading {file_path}: {e}")
                continue

    def tokenize_and_chunk(self, examples: dict[str, list]) -> dict[str, list]:
        """Tokenize texts and split into fixed-length sequences"""

        res = {"input_ids": [], "attention_mask": [], "prompt_lengths": [], "labels": []}
        for prompt, target in zip(examples["query"], examples["response"]):
            if target:
                messages = [{"role": "user", "content": prompt}]
                inputs = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, return_dict=False
                )

                input_id = inputs

                end_id = self.tokenizer.encode(text=target) + [self.tokenizer.eos_token_id]

                if len(input_id) + len(end_id) > self.tokenizer.model_max_length:
                    continue

                prompt_length = len(input_id)
                input_id.extend(end_id)

                res["input_ids"].append(input_id)
                res["labels"].append(list(input_id))
                res["prompt_lengths"].append([prompt_length])
                res["attention_mask"].append([1] * len(input_id))

        return res

    def build_dataset_from_files(
        self,
        data_dir: str,
        output_dir: str,
        cache_dir: str,
        num_workers: Optional[int] = None,
        batch_size: int = 1000,
    ) -> datasets.Dataset:
        """Main method to build tokenized dataset from jsonl.gz files"""

        # Get all data files
        file_paths = self.read_jsonl_gz_files(data_dir)
        if not file_paths:
            raise ValueError(f"No jsonl.gz files found in {data_dir}")
        else:
            self.logger.info("There are %d data files in %s", len(file_paths), data_dir)

        dataset = datasets.load_dataset(data_dir, data_files=file_paths, cache_dir=cache_dir, num_proc=num_workers)[
            "train"
        ]
        self.logger.info(f"Initial dataset size: {len(dataset)} samples")

        # Tokenize and chunk with multiprocessing
        self.logger.info(f"Tokenizing and chunking with {num_workers} processes...")

        dataset = dataset.map(
            self.tokenize_and_chunk,
            batched=True,
            batch_size=batch_size,
            num_proc=num_workers,
            remove_columns=dataset.column_names,
            desc="Tokenizing and chunking",
        ).shuffle()

        self.logger.info(f"Final dataset size: {len(dataset)} sequences of {self.sequence_length} tokens")

        # Save the dataset
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(output_dir, num_proc=num_workers)
            self.logger.info(f"Dataset saved to {output_dir}")

        return dataset


def tokenize_datafiles(src_data_dir, tokenized_data_dir, cache_dir, tokenizer, max_seq_len, num_workers, logger):
    # Path(tokenized_data_dir).mkdir(exist_ok=True, parents=True)

    # Build dataset only on master process
    logger.info("Dataset building: src_data_dir=%s tokenized_data_dir=%s", src_data_dir, tokenized_data_dir)

    # Basic version with dataset.map multiprocessing
    builder = TokenizedDatasetBuilder(tokenizer, sequence_length=max_seq_len, logger=logger)
    dataset = builder.build_dataset_from_files(
        data_dir=src_data_dir,
        output_dir=tokenized_data_dir,
        cache_dir=cache_dir,
        num_workers=num_workers,
        batch_size=1000,
    )
    logger.info("Tokenization completed.")

    return dataset


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels, prompt_lengths, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "prompt_lengths", "attention_mask")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(x) for x in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(x) for x in labels], batch_first=True, padding_value=IGNORE_INDEX
        )

        # prompt_lengths = torch.stack(prompt_lengths).view(-1, 1)
        prompt_lengths = torch.LongTensor(prompt_lengths).view(-1, 1)

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(x) for x in attention_mask], batch_first=True, padding_value=0
        )

        return dict(  # noqa: C408
            input_ids=input_ids,
            labels=labels,
            prompt_lengths=prompt_lengths,
            attention_mask=attention_mask,
        )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) >= 2 and sys.argv[0].endswith(".py") and sys.argv[-1].endswith(".json"):
        # If we pass >= 2 arguments to commandline and only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.

        # Getting json file name to put into clearml later, but before parsing args
        args_file = os.path.abspath(sys.argv[-1])

        model_args, data_args, training_args = parser.parse_json_file(json_file=args_file)
    elif len(sys.argv) >= 2 and sys.argv[0].endswith(".py") and sys.argv[-1].endswith((".yaml", ".yml")):
        # Getting yaml file name to put into clearml later, but before parsing args
        args_file = os.path.abspath(sys.argv[-1])

        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=args_file)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        # storing non-default arguments
        args_file = {key: value for key, value in vars(parser.parse_args()).items() if value != parser.get_default(key)}

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --resume_from_checkpoint to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--resume_from_checkpoint` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer_name = model_args.tokenizer_name
    elif model_args.model_name_or_path:
        tokenizer_name = model_args.model_name_or_path
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_args.scratch_source, **config_kwargs)
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

        # Overriding config params with tokenizer parameters
        if tokenizer.bos_token_id is not None:
            config.bos_token_id = tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.mask_token_id is not None:
            config.mask_token_id = tokenizer.mask_token_id
        logger.info(
            "Setting config: pad_token_id=%d bos_token_id=%d eos_token_id=%d mask_token_id=%d",
            config.pad_token_id,
            config.bos_token_id,
            config.eos_token_id,
            config.mask_token_id,
        )
        config.vocab_size = len(tokenizer)
        logger.info("Setting config: vocab_size=%d", config.vocab_size)

        logger.info(f"Model config: {config}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    if model_args.model_name_or_path:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            dtype=torch_dtype,
            attn_implementation=model_args.attn_implementation,
        )
    else:
        logger.info("Training new model from scratch")
        try:
            model = AutoModel.from_config(
                config,
                trust_remote_code=model_args.trust_remote_code,
                dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
        except ValueError:
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=model_args.trust_remote_code,
                dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Model loaded from config - Total size={n_params / 2**20:.2f}M params ({n_params / 2**30:.1f}B)")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        # setup pad_to_multiple_of https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    if (tokenizer.mask_token is None or tokenizer.mask_token_id is None) and config.mask_token_id is not None:
        tokenizer.mask_token_id = config.mask_token_id
        tokenizer.mask_token = tokenizer.convert_ids_to_tokens(config.mask_token_id)

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        max_seq_length = data_args.max_seq_length
        tokenizer.model_max_length = max_seq_length

    if model_args.sequential_shuffle is not None:
        model.config.sequential_shuffle = model_args.sequential_shuffle

    if model_args.masked_shuffle is not None:
        model.config.diffusion_shuffle = model_args.masked_shuffle
        model.config.noise_sorting = True

    if model_args.ordered_sampling is not None:
        model.config.ordered_sampling = model_args.ordered_sampling

    if data_args.slotted_training is not None:
        model.config.slotted_training = data_args.slotted_training

    if data_args.custom_set == "ultrafineweb2048" and data_args.dataset_name is not None:
        tokenized_dir = os.path.abspath(data_args.dataset_name)
        tokenized_train_dirs = []
        tokenized_test_dirs = []
        tokenized_train_dirs.append(os.path.join(tokenized_dir, "train-1-of-4"))
        tokenized_train_dirs.append(os.path.join(tokenized_dir, "train-2-of-4"))
        tokenized_train_dirs.append(os.path.join(tokenized_dir, "train-3-of-4"))
        tokenized_train_dirs.append(os.path.join(tokenized_dir, "train-4-of-4"))
        tokenized_test_dirs.append(os.path.join(tokenized_dir, "test"))

        logger.info("Loading train_dataset from %s", tokenized_train_dirs)
        train_dataset = datasets.concatenate_datasets(
            [datasets.load_from_disk(ds_dir) for ds_dir in tokenized_train_dirs]
        )
        logger.info("Loading eval_dataset from %s", tokenized_test_dirs)
        eval_dataset = datasets.concatenate_datasets(
            [datasets.load_from_disk(ds_dir) for ds_dir in tokenized_test_dirs]
        )

        logger.info(f"Shuffle train set with seed {TRAIN_SHUFFLE_SEED}")
        train_dataset = train_dataset.shuffle(seed=TRAIN_SHUFFLE_SEED)

        logger.info("Training dataset: %d samples", len(train_dataset))
        logger.info("Testing dataset:  %d samples", len(eval_dataset))
        tokenized_datasets = {"train": train_dataset, "validation": eval_dataset}
    elif (
        model_args.cache_dir
        and os.path.exists(model_args.cache_dir)
        and os.path.exists(os.path.join(model_args.cache_dir, "train_dataset.prepared"))
    ):
        fp = os.path.join(model_args.cache_dir, "train_dataset.prepared")
        logger.info("Loading train_dataset from %s", fp)
        train_dataset = datasets.load_from_disk(fp)

        fp = os.path.join(model_args.cache_dir, "eval_dataset.prepared")
        logger.info("Loading eval_dataset from %s", fp)
        eval_dataset = datasets.load_from_disk(fp)

        tokenized_datasets = {"train": train_dataset, "validation": eval_dataset}
    elif (
        data_args.custom_set == "sft"
        and model_args.cache_dir
        and os.path.exists(tokenized_dir := os.path.join(model_args.cache_dir, "tokenized"))
    ):
        tokenized_train_dir = os.path.abspath(os.path.join(tokenized_dir, "train"))
        tokenized_test_dir = os.path.abspath(os.path.join(tokenized_dir, "test"))

        logger.info("Loading tokenized dataset from %s...", tokenized_test_dir)
        eval_dataset = datasets.load_from_disk(tokenized_test_dir)
        logger.info("Loading tokenized dataset from %s...", tokenized_train_dir)
        train_dataset = datasets.load_from_disk(tokenized_train_dir)
        tokenized_datasets = {"train": train_dataset, "validation": eval_dataset}
    elif (
        data_args.custom_set == "sft"
        and model_args.cache_dir
        and not os.path.exists(tokenized_dir := os.path.join(model_args.cache_dir, "tokenized"))
    ):
        tokenized_train_dir = os.path.abspath(os.path.join(tokenized_dir, "train"))
        tokenized_test_dir = os.path.abspath(os.path.join(tokenized_dir, "test"))
        train_data_dir = os.path.abspath(os.path.dirname(data_args.train_file))
        test_data_dir = os.path.abspath(os.path.dirname(data_args.validation_file))

        logger.info("Start tokenizing the test split...")
        test_dataset = tokenize_datafiles(
            test_data_dir,
            tokenized_test_dir,
            model_args.cache_dir,
            tokenizer,
            max_seq_length,
            data_args.preprocessing_num_workers,
            logger,
        )
        logger.info("Testing dataset:  %d samples", len(test_dataset))

        logger.info("Start tokenizing the training split...")
        train_dataset = tokenize_datafiles(
            train_data_dir,
            tokenized_train_dir,
            model_args.cache_dir,
            tokenizer,
            max_seq_length,
            data_args.preprocessing_num_workers,
            logger,
        )
        logger.info("Training dataset: %d samples", len(train_dataset))

        detokenized_dir = os.path.abspath(os.path.join(tokenized_dir, "detokenized"))
        Path(detokenized_dir).mkdir(exist_ok=True)

        with open(os.path.join(detokenized_dir, "detokenized_train_samples.txt"), "w") as f:
            for i, sample1 in enumerate(train_dataset.take(10), start=1):
                f.write("\n\n" + "=" * 20 + f">>> TRAIN SAMPLE #{i} <<<" + "=" * 80 + "\n\n")
                f.write(tokenizer.decode(sample1["input_ids"]) + "\n")

        with open(os.path.join(detokenized_dir, "detokenized_test_samples.txt"), "w") as f:
            for i, sample1 in enumerate(test_dataset.take(10), start=1):
                f.write("\n\n" + "=" * 20 + f">>> TEST SAMPLE #{i} <<<" + "=" * 80 + "\n\n")
                f.write(tokenizer.decode(sample1["input_ids"]) + "\n")

        with open(os.path.join(detokenized_dir, "splits_info.txt"), "w") as f:
            f.write(f"max_seq_len={max_seq_length}  \n")
            f.write(f"tokenizer: {tokenizer_name}  \n")
            f.write(f"tokenizer.bos_token_id={tokenizer.bos_token_id}  \n")
            f.write(f"tokenizer.eos_token_id={tokenizer.eos_token_id}  \n")
            f.write(f"tokenizer.mask_token_id={tokenizer.mask_token_id}  \n")
            f.write(f"tokenizer.pad_token_id={tokenizer.pad_token_id} \n\n")

            f.write(
                f"Train split totals:  {len(train_dataset):,} samples   {len(train_dataset) * max_seq_length:,} tokens\n"
            )
            f.write(
                f"Test split totals:   {len(test_dataset):,} samples   {len(test_dataset) * max_seq_length:,} tokens\n"
            )

        logger.info("Tokenization completed, look at the report files in %s", detokenized_dir)

        logger.info("Loading tokenized dataset from %s...", tokenized_test_dir)
        eval_dataset = datasets.load_from_disk(tokenized_test_dir)
        logger.info("Loading tokenized dataset from %s...", tokenized_train_dir)
        train_dataset = datasets.load_from_disk(tokenized_train_dir)
        tokenized_datasets = {"train": train_dataset, "validation": eval_dataset}
    else:
        # raise RuntimeError()

        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            if "validation" not in raw_datasets:
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
                extension = get_filename_ext(data_args.train_file)

            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
                extension = get_filename_ext(data_args.validation_file)

            if extension == "txt":
                extension = "text"
            elif extension == "jsonl" or extension == "jsonl.gz" or extension == "jsonl.zstd":
                extension = "json"
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                num_proc=data_args.preprocessing_num_workers,
            )

            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets:
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                )

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)
        text_column_name = "text" if "text" in column_names else column_names[0]

        if data_args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if data_args.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples[text_column_name] = [
                    line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            with training_args.main_process_first(desc="dataset map tokenization"):
                if not data_args.streaming:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=[text_column_name],
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on dataset line_by_line",
                    )
                else:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        remove_columns=[text_column_name],
                    )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            with training_args.main_process_first(desc="dataset map tokenization"):
                if not data_args.streaming:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on every text in dataset",
                    )
                else:
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        remove_columns=column_names,
                    )

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
                total_length = len(concatenated_examples[list(examples)[0]])  # noqa: RUF015
                # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
                # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
                total_length = (total_length // max_seq_length) * max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/process#map

            with training_args.main_process_first(desc="grouping texts together"):
                if not data_args.streaming:
                    tokenized_datasets = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc=f"Grouping texts in chunks of {max_seq_length}",
                    )
                else:
                    tokenized_datasets = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                    )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric_evaluate = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    def compute_metrics(eval_preds: EvalPrediction):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        me = metric_evaluate.compute(predictions=preds, references=labels)
        metrics = trainer.valid_metrics.compute()
        loss_seq = trainer.val_loss_seq.compute()
        loss_dif = trainer.val_loss_dif.compute()
        slot_size = trainer.val_slot_size.compute()
        acc_seq = (
            trainer.val_acc_seq.compute()
            if trainer.val_acc_seq.update_called
            else torch.tensor(0, device=trainer.val_acc_seq.device)
        )
        acc_dif = (
            trainer.val_acc_dif.compute()
            if trainer.val_acc_dif.update_called
            else torch.tensor(0, device=trainer.val_acc_dif.device)
        )
        full_metrics = {
            **{k: v.item() for k, v in metrics.items()},
            "loss_seq": round(loss_seq.item(), 4),
            "loss_dif": round(loss_dif.item(), 4),
            "acc_seq": round(acc_seq.item(), 4),
            "acc_dif": round(acc_dif.item(), 4),
            "slot_size": slot_size.item(),
            **me,
        }

        return full_metrics

    # Data collator
    if data_args.custom_set == "sft":
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        # This one will take care of randomly masking the tokens.
        pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )

    # Parse optional slot scheduling parameters from DataTrainingArguments
    slot_size_set = None
    if data_args.slot_size_set is not None:
        # slot_size_set = json.loads(data_args.slot_size_set)
        slot_size_set = data_args.slot_size_set
        logger.info("Using slot_size_set from config: %s", slot_size_set)
    slot_step_borders = None
    if data_args.slot_step_borders is not None:
        # slot_step_borders = json.loads(data_args.slot_step_borders)
        slot_step_borders = data_args.slot_step_borders
        logger.info("Using slot_step_borders from config: %s", slot_step_borders)

    # Initialize our Trainer
    trainer = TextDiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        use_ema=model_args.enable_ema,
        ema_decay=model_args.ema_decay,
        save_epochs=model_args.save_epochs,
        slotted_training=data_args.slotted_training,
        slot_size_set=slot_size_set,
        slot_step_borders=slot_step_borders,
    )
    clearml_callback = trainer.pop_callback(ClearMLCallback)
    if clearml_callback is not None:
        # Here we know that clearml was turned on, and callback added, now we change it to our custom one
        clearml_callback = CustomLoggingCallback(model_args=model_args, data_args=data_args, args_file=args_file)
        trainer.add_callback(clearml_callback)

    trainer.add_callback(TextDiffusionCallback(trainer))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
