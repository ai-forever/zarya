# Zarya

**Zarya** is a [family of hybrid language models (0.6B / 1.7B / 4B)](https://hf.co/collections/ai-forever/zarya) that jointly optimizes an autoregressive (AR) objective and a masked-diffusion (MDM) objective within a single architecture.
The architecture can be built on top of any autoregressive model, but in this repo it uses the `Qwen3` backbone from Hugging Face Transformers.

Naming explanation: Naming explanation: Zarya (pronounced as [zɐˈrʲa] ([IPA notation](https://en.wiktionary.org/wiki/Appendix:Russian_pronunciation)), literally "Dawn" in English) is a figure from Slavic folklore — a female personification of dawn who may be considered a goddess.
In various traditions, she can manifest as a single being or as two or three sisters simultaneously.

This is a research prototype.
Training is driven by a single script, `train.py`, which uses a custom subclass of `Trainer` (class `TextDiffusionTrainer`).
It structures training data into variable-size slots and trains the model simultaneously on visible tokens (AR task) and on tokens replaced with `<|mdm_mask|>` (diffusion task).

---

**Hugging Face model collection**: https://hf.co/collections/ai-forever/zarya

---

## Key features

- **Hybrid training**: the final loss is `λ · loss_dif + (1 − λ) · loss_seq`, where `λ = diffusion_loss_proportion` (default 0.5).
The model simultaneously learns to predict the next token autoregressively and to reconstruct masked slots.
- **Slotted training**: answers are split into fixed-size slots; some slots remain visible (AR task), while others are fully masked (MDM task).
The mask ratio `p_mask` is sampled in `[eps, 1]` per example.
- **Slot-size curriculum**: the `slot_size_set` + `slot_step_borders` hyperparameters gradually increase the slot size during training (from small slots early on to larger ones later).
- **Flexible diffusion modes**: `noise_sorting` (positions ordered by noise level), `grouped_noise` (Prefix Completion, Fill-In-the-Prefix, Fill-In-the-Middle and random spans up to `max_span_length`), `ordered_sampling` (monotonically increasing `p_mask` left to right).
- **Full KV cache reuse** with causal attention mask by reordering tokens at inference time.
- **Two inference modes**, both reachable through a single `model.generate(...)` call:
  - MDM sampling with first-hitting denoising (masked-diffusion inference), and
  - slotted speculative parallel decoding with inter-slot diffusion-based selection and intra-slot autoregressive infilling, while reordering newly generated slots ahead of the remaining masks after each iteration, with the `tokens_per_forward` efficiency metric.
- **Training-inference decoupling**: unlike prior work where the training configuration dictates the inference mode, Zarya allows any trained model to be deployed in either MDM sampling or slotted speculative decoding mode via a single inference flag, offering higher flexibility.
Both modes fully use the KV cache with causal attention masks.
- **Hugging Face ecosystem**: the model is registered with `AutoConfig` / `AutoModel` and is fully compatible with the HF `Trainer`.
- **Logging**: ClearML via a custom `CustomLoggingCallback` callback, with extended training metrics (loss/accuracy tracked separately for the AR and MDM branches).

---

## The model family

Model sizes at training stage can be set through `config_overrides` option of `train.py`.
Examples provided in the corresponding configs under `train-config/`.

| Variant    | hidden_size | num_hidden_layers | num_attention_heads | intermediate_size | ordered_sampling | Config                         |
|------------|-------------|-------------------|---------------------|-------------------|------------------|--------------------------------|
| Zarya-0.6B | 1024        | 28                | 16                  | 3072              | `false`          | `train-config/Zarya-0.6B.json` |
| Zarya-1.7B | 2048        | 28                | 16                  | 6144              | `true`           | `train-config/Zarya-1.7B.json` |
| Zarya-4B   | 2560        | 36                | 32                  | 9728              | `true`           | `train-config/Zarya-4B.json`   |

---

## Architecture

The model backbone is `Qwen3ForCausalLM` (specified by `ZaryaConfig.backbone_class`).
A wrapper class `Zarya` is defined on top of it in `model-scratch/modeling.py`.

Because the Qwen3 backbone retains causal attention, masked positions cannot attend to future masked positions.
Thus, the diffusion objective used by Zarya is a causal masked-reconstruction objective rather than the fully bidirectional masked-token objective commonly used in masked diffusion language models.
The reordering of visible and masked positions ensures that all masked positions can attend to the visible prefix while preserving the causal attention pattern and KV-cache compatibility.

The config `ZaryaConfig` (`model_type="zarya"`, `model-scratch/configuration.py`) adds diffusion- and training-related hyperparameters:

- `alpha_0` (default 0.25) and `noise_eps` (1e-3) — parameters of the linear noise schedule;
- `diffusion_loss_proportion` (0.5) — share of the diffusion loss;
- `sequential_attn_mode` / `diffusion_attn_mode` — attention modes for the AR and MDM regimes;
- `sequential_shuffle` / `diffusion_shuffle` — shuffling of slots/tokens in the AR and MDM regimes;
- `sampling_eps`, `sample_t_override`, `sample_t_upper` — parameters controlling the sampling of the noise level `t`;
- `time_conditioning`, `simple_masking`, `add_loss_path`, `grouped_noise`, `max_span_length`, `scale_by_batch`, `unnormalized_loss` — different options for research of noise application and loss calculation;
- `slotted_training`, `ordered_sampling`, `noise_sorting` — control over the training mode.

---

## Training pipeline

Training is performed by `train.py`, in which the class `TextDiffusionTrainer` extends the standard HF `Trainer`.

### Hybrid batch (slotted training)

When `slotted_training` is set to True, the `forward_process()` function (`train.py`) transforms a batch of `(prompt, answer)` into a hybrid training batch:

1. **Slot partitioning**: the answer is split into slots of size `slot_size` (from `slot_size_set`).
2. **Mask sampling**: for each example in the batch, a mask probability `p_mask` is sampled uniformly in `[eps, 1]` — this determines what fraction of slots will be masked (treated as the diffusion task).
3. **Slot assignment**:
   - **AR slots**: tokens remain visible; the model predicts the next token within each slot  (yielding `loss_seq`).
     When `sequential_shuffle` is enabled, the slots are shuffled.
   - **MDM slots**: all tokens are replaced with the special `<|mdm_mask|>` token; the model reconstructs the original tokens, yielding `loss_dif`.
     Each token inside a slot is assigned a `p_mask` value used in per-token normalization as `1 / p_mask` weighting.

The parameters `slot_size_set` (e.g. `[2, 4, 8, 16, 32, 64]`) and `slot_step_borders` (e.g. `[0.06, 0.2, 0.4, 0.6, 0.8, 1.0]`) determine which slot size is active at each fraction of training steps.
If `slot_step_borders` is not set, slot sizes are distributed uniformly across all steps.

The final loss is a linear combination: `loss = diffusion_loss_proportion * loss_dif + (1 - diffusion_loss_proportion) * loss_seq`.
This design enables the model to simultaneously learn next-token prediction (AR) and masked-token reconstruction (MDM) on the same input.

Optional flags include:
- `ordered_sampling`: inside each slot, modify per-token `p_mask` so that it increases left-to-right. As `loss_dif` is scaled with the value of `1 / p_mask`, this makes the model learn that correct prediction of the slot beginning tokens is more important.

#### Slot-Size Curriculum

Slot sizes evolve during training via `slot_step_borders` (epoch or step thresholds). For example, with
```
slot_size_set = [2, 4, 8, 16, 32, 64],
slot_step_borders = [0.06, 0.2, 0.4, 0.6, 0.8, 1.0],
```
the slot size gradually increases from 2 to 64 over the course of training.

### Non-slotted mode (`slotted_training=false`)

When `slotted_training` is set to **false**, the input batch is **not** split into slots and the `forward_process()` preprocessing is skipped entirely.
Instead, the model runs a pure masked-diffusion (MDM) objective over the raw `(prompt, answer)` sequence, and the whole sequence is treated as one diffusion target.
The hybrid AR + MDM combination is achieved by running two separate forward passes over the *same* input (rather than over visible/masked slots), each producing its own loss term.

In this mode the `forward()` method of `Zarya` (`model-scratch/modeling.py`) proceeds as follows:

1. **Noise level sampling.** A noise level `t` per sample is drawn uniformly in `[sampling_eps, 1]` (see `_sample_t`).
   - `sample_t_override` > 0 fixes `t` to a constant value for all samples (useful for debugging);
   - `sample_t_upper` < 1 rescales the sampled `t` into `[t_min, sample_t_upper]`, restricting the maximum amount of masking;
   - `ordered_sampling` offsets `t` per token position so that `t` (and hence `p_mask`) increases left to right, easing the AR→MDM transition;
   - `sampling_eps` sets the minimum floor of `t`.

2. **Noisy input construction.** `q_xt` replaces tokens with `<|mdm_mask|>` based on the mask probability `p_mask`.
   - By default `p_mask` is sampled independently per token (`simple_masking=false`, `grouped_noise=false`).
     The per-token mask probability is `p_mask = 1 − alpha_t`.;
   - `simple_masking=true` sets `p_mask = t` directly instead of using the schedule below;
   - `grouped_noise=true` masks contiguous spans of up to `max_span_length` tokens to form Prefix Completion, Fill-in-the-Prefix, Fill-in-the-Middle and random-span patterns.

3. **Noise schedule.** Unless `simple_masking` is used, `p_mask = 1 − alpha_t` is derived from the linear schedule `alpha_t = alpha_0 · (1 − t)`, parameterized by `alpha_0` and `noise_eps` (`Linear`).
   The derivative `dalpha_t` drives the **subs-parameterization** loss scale below.

4. **Two forward passes.**
   - The **sequential (AR) phase** feeds the clean sequence `x_0` and predicts only the tokens that were masked in `x_noisy` (the rest are set to `ignore_index`). Its loss is `loss_seq`.
   - The **diffusion (MDM) phase** feeds the noisy sequence `x_t` and reconstructs the masked tokens. Its loss is `loss_dif`.

5. **Loss weighting.** The final loss is `loss = diffusion_loss_proportion * loss_dif + (1 - diffusion_loss_proportion) * loss_seq`, calculated the same way as in the slotted training regime.
   - The MDM loss is scaled by `loss_scale = -dalpha_t / p_mask` (the subs-parameterization weight, in the spirit of SEDD/MDLM/Eso-LMs), or by `1 / p_mask` when `simple_masking` is on.
   - `add_loss_path=true` multiplies each token loss by the confidence term `1 + exp(-loss)`.
   - `scale_by_batch=true` normalizes the MDM loss by the total number of valid (attended) tokens of the batch instead of the number of masked tokens.
   - `unnormalized_loss=true` scales the losses by the batch-size factor rather than by the token count.

6. **Optional permutation.** With `noise_sorting=true`, tokens are reordered by their mask/unmask state (`diffusion_shuffle` controls shuffling of the masked slots) before both forward passes, and logits are permuted back afterwards.

Note: in the non-slotted mode the `slot_size_set` / `slot_step_borders` and the `forward_process` slot curriculum are ignored, and no `slot_size` metric is reported during training.

## Additional training modes

- **`noise_sorting`** — input tokens or slots are sorted by noise applied, and the result is un-permuted after the forward pass.
- **`grouped_noise`** — instead of token-wise masking, grouped noise is used: Prefix Completion, Fill-in-the-Prefix, Fill-in-the-Middle and random spans (up to `max_span_length` tokens).
- **`ordered_sampling`** — `p_mask` increases left to right within slots, which smooths the transition between the AR and MDM branches.
- **`time_conditioning`, `simple_masking`, `add_loss_path`** — additional research flags in the config (see `model-scratch/configuration.py`).

---

## Generation

Generation supports two modes. **Both modes are available through a single `model.generate(...)` call.**
The `generate()` method routes between the two modes automatically based on `generation_config.slotted_generation` (set to `true` in `model-scratch/generation_config.json`):

- **`slotted_generation=true`** → `model.generate(...)` runs slotted speculative decoding (`generate_slotted`).
- **`slotted_generation=false`** → `model.generate(...)` runs MDM masked-diffusion sampling (`generate_samples`).

Critically, the training and inference regimes are fully decoupled: a model trained with `slotted_training=False` can still be deployed with `slotted_generation=True`, and vice versa.

### Mode A: MDM sampling (`slotted_generation=false`)

When `slotted_generation` is set to **false**, `model.generate(...)` runs `generate_samples` (`model-scratch/modeling.py`): an iterative masked-diffusion denoising process using the first-hitting sampler.
The generation starts by appending `<|mdm_mask|>` tokens after the prompt (up to `max_length`) and iteratively reveals masks until all are filled.

The decoding loop works as follows:

1. **Mask budget planning.** `_tokens_unmasked_per_step()` determines how many masks to reveal per step.
    `alpha_0` is the expected fraction of masked tokens generated with the diffusion process.
   - If the number of discretization steps is set to `T > 0`: exactly `T` diffusion steps are used.
     A Binomial distribution is used to calculate the number of masked tokens to denoise through the diffusion process, and the tokens left after that are denoised sequentially.
   - If the number of discretization steps is set to `T = 0` (ignoring noise calculations): `T` steps are auto-calculated as 1/4 of masked tokens.
   - `ignore_noise_schedule` selects the unmasking plan for the diffusion phase at inference time: default, when the number of unmasked tokens is determined with a Binomial distribution (`false`), or when the number of unmasked tokens is even across unmasking steps;
   - `unmask_probs_coef` scales the per-step unmask probability (multiplies `(alpha_s - alpha_t) / (1 - alpha_t)`);
   - the `Linear` schedule in the model config (`alpha_0`, `noise_eps`) defines `alpha_t` and hence the per-step fraction of tokens to reveal.

2. **Reordering.** The input sequence is reordered so that masked tokens are always after unmasked ones.

3. **Per-step sampling from the categorical distribution.** At each step, the model receives the progressively filled sequence (with KV cache reuse) and yields logits for masked positions.
    Gumbel noise is added for categorical sampling, and standard sampling parameters (`temperature`, `top_p`, `repetition_penalty`) are honored.
   - `use_float64=true` casts logits to `float64` for numerically stable Gumbel sampling;
   - `diffusion_phase_only=true` reveals only the tokens attributed to the diffusion schedule, while `sequential_phase_only=true` reveals the remaining masks one token at a time in left-to-right order (they cannot both be enabled).

5. **Restoration.** After all steps, the sequence is restored to the original token order.

The KV cache can be reused because the sequence is reordered so that tokens whose values are fixed at a given denoising step precede the remaining masked positions.
Under causal attention, the cached prefix states therefore remain unchanged when masked positions are progressively filled.
This property would not hold for a bidirectional masked-diffusion attention pattern, where changing any previously masked token could affect the representations of other masked positions.

Relevant parameters:

- `ZaryaGenerationConfig` (`model-scratch/generation_config.json`): `T`, `ignore_noise_schedule`, `unmask_probs_coef`, `use_float64`, `sequential_phase_only`, `diffusion_phase_only` (plus standard sampling options such as `temperature`, `top_p`, `repetition_penalty`).
- Model `ZaryaConfig` (`model-scratch/config.json`): `alpha_0`, `noise_eps` (the linear schedule), `sequential_shuffle`, `diffusion_shuffle` (ordering).

### Mode B: Slotted speculative decoding (`slotted_generation=true`)

When `slotted_generation` is set to **true**, inference code achieves parallelization by elevating decoding units from tokens to slots, fully reusing the KV cache to avoid recomputation.
1. **Reordering:** before the first forward pass, the input sequence is reordered so that masked tokens are always after unmasked.
2. **Block construction:** `max_new_tokens` count of masked tokens is divided into `serial_num_blocks` blocks of length `block_size = max_new_tokens / serial_num_blocks`.
    Within each block, tokens are grouped into slots of size `slot_size`.
    If `max_new_tokens` is small, `serial_num_blocks` is forced to 1 to prevent zero-length blocks.
3. **Draft phase:** an MDM forward pass drafts tokens for all slots in the current block in parallel.
    The confidence of each slot is estimated as the probability of its first token.
4. **Sampling from the categorical distribution:** if `temperature` is positive, apply Gumbel noise to logits for sampling from categorical distributions.
5. **Slot selection:** slots with confidence exceeding `slot_threshold` are accepted for further processing immediately.
    If no slots are confident enough, select the most confident one, so we always have at least one slot to process further.
6. **Verification phase:** selected slots undergo an AR verification forward pass.
    Tokens with probability exceeding `token_threshold` are accepted; those below are iteratively refined in a speculative loop.
7. **KV cache update:** the cache is updated incrementally with accepted tokens, avoiding recomputation for verified prefixes.
8. **Restoration:** after all steps, the sequence is restored to the original token order.

The `tokens_per_forward` metric reports the average number of accepted tokens per forward pass (values > 1 indicate speedup).

Important `ZaryaGenerationConfig` parameters: `T`, `ignore_noise_schedule`, `use_float64`, `sequential_phase_only`, `diffusion_phase_only`, `unmask_probs_coef`, `slot_size`, `serial_num_blocks`, `slot_threshold`, `token_threshold`.

---

## Example: training

Training is launched with `train.py`. The `--config_overrides` argument (a string of the form `hidden_size=1024,num_hidden_layers=28,...`) is applied on top of the base config from `model-scratch/config.json` and is used to set the model size. Three ready-made example training scripts are provided under `train-config/`:

- `train-config/Zarya-0.6B.json`
- `train-config/Zarya-1.7B.json`
- `train-config/Zarya-4B.json`

```bash
conda activate zarya
python train.py train-config/Zarya-0.6B.json
```

The same setup can be expressed with CLI flags:

```bash
python train.py \
    --scratch_source ./model-scratch \
    --tokenizer_name ./model-scratch \
    --trust_remote_code \
    --config_overrides "ordered_sampling=false,hidden_size=1024,intermediate_size=3072,num_hidden_layers=28,num_attention_heads=16" \
    --custom_set sft \
    --train_file ./localsets/train/Conversations.train.jsonl \
    --validation_file ./localsets/test/Conversations.test.jsonl \
    --cache_dir ./output-cache \
    --output_dir ./output \
    --max_seq_length 2048 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear --warmup_steps 2000 \
    --eval_strategy steps --eval_steps 250 \
    --save_strategy steps --save_steps 2000 \
    --logging_steps 10 \
    --bf16 --bf16_full_eval \
    --gradient_checkpointing \
    --slot_size_set '[2,4,8,16,32,64]' \
    --slot_step_borders '[0.06,0.2,0.4,0.6,0.8,1.0]'
```

Notes:

- `train.py` accepts configuration via JSON file, YAML file, or individual CLI flags (`HfArgumentParser` with `ModelArguments`, `DataTrainingArguments`, `TrainingArguments`).
- `--scratch_source ./model-scratch` points to the "from-scratch" model template directory.
- YAML configs are supported too: `python train.py train-config/custom.yaml`.

---

## Example: inference

Loading the model and tokenizer requires `trust_remote_code=True`.

```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("./output", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained("./output", trust_remote_code=True)

prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# Both modes go through model.generate(...).
# With generation_config.slotted_generation=true -> slotted speculative decoding:
out = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    slot_size=16,
    serial_num_blocks=4,
    slot_threshold=0.9,
    token_threshold=0.3,
)
# Setting generation_config.slotted_generation=false -> MDM sampling instead:
# out = model.generate(input_ids, max_new_tokens=256)

print(tokenizer.decode(out[0]))
```

---

## Repository layout

```
.
├── train.py                 # Single entry point: TextDiffusionTrainer,
│                            #   forward_process, callbacks, metrics, ClearML
├── requirements.yaml        # conda env requirements
├── ruff.toml                # ruff linter config
├── model-scratch/           # model definition "from scratch"
│   ├── configuration.py     # ZaryaConfig
│   ├── modeling.py          # Zarya model class
│   ├── generation_utils.py  # ZaryaGenerationConfig
│   ├── config.json          # base architecture config defaults
│   ├── generation_config.json # generation config defaults
│   ├── tokenizer.json / merges.txt / vocab.json / tokenizer_config.json # tokenizer setup
│   └── chat_template.jinja  # chat template from tokenizer
└── train-config/            # example training configs
    ├── Zarya-0.6B.json
    ├── Zarya-1.7B.json
    ├── Zarya-4B.json
```

---

## Installation

The environment is described by `requirements.yaml`:

```bash
git clone https://github.com/ai-forever/zarya.git
cd zarya
conda env create -f requirements.yml
conda activate zarya
```

When training on a GPU, make sure the CUDA driver version is compatible with the installed PyTorch.

---

## Tokenizer

A Qwen-style tokenizer (`Qwen2Tokenizer`, `model-scratch/tokenizer_config.json`) with `vocab_size=151936` and `model_max_length=131072`.

| Role | Token             | ID     |
|------|-------------------|--------|
| BOS  | `<\|im_start\|>`  | 151644 |
| EOS  | `<\|im_end\|>`    | 151645 |
| PAD  | `<\|endoftext\|>` | 151643 |
| MASK | `<\|mdm_mask\|>`  | 151669 |

---

## Metrics and logging

In addition to the standard HF `Trainer` metrics (`eval_loss`, accuracy), `train.py` tracks:

- `bpd` — bits per dimension;
- `perplexity` — perplexity (`exp(eval_loss)`);
- `loss_seq` / `loss_dif` — losses for the AR and MDM branches;
- `acc_seq` / `acc_dif` — accuracy for the AR and MDM branches;
- `slot_size` — the current slot size (to monitor the curriculum).

Logging is done via ClearML using the custom `CustomLoggingCallback` callback (structured arguments, architecture, number of trainable parameters).
If ClearML is not configured, the callback is not activated (the example configs use `report_to="none"`).

---
LM-eval Benchmarking

LM-eval benchmarking with `lm-eval` package is supported.
Example of run:

```bash
lm_eval run \
--tasks=gsm8k,ifeval,mbpp,mbpp_instruct,mbpp_plus,mbpp_plus_instruct,hellaswag \
--model=hf --confirm_run_unsafe_code \
--log_samples \
--apply_chat_template \
--output_path=./reports/lm-eval_results \
--model_args=pretrained=ai-forever/Zarya-0.6B,backend=causal,dtype=bfloat16,attn_implementation=sdpa,trust_remote_code=True \
--gen_kwargs slotted_generation=true,slot_size=16,serial_num_blocks=4,slot_threshold=0.9,token_threshold=0.4
```

---

## Data

### SFT (custom_set = "sft")

JSONL with `query` / `response` fields:

```json
{"query": "User question text", "response": "Model answer text"}
```

Data is pre-tokenized (including via `TokenizedDatasetBuilder` / `tokenize_datafiles()`) and cached under `--cache_dir`.

### Pretraining

Classic auto-regressive training on text files is supported: `line_by_line` and `group_texts` modes (concatenation and slicing into `max_seq_length` sequences), or pre-sliced tokenized datasets (e.g. `custom_set="ultrafineweb2048"`).

---

## Limitations

- This is a **research prototype**; inference performance and stability depend on the choice of `slot_size`, `serial_num_blocks`, and the `slot_threshold` / `token_threshold` thresholds.
- The code relies on private Hugging Face Transformers APIs; when upgrading versions, watch for compatibility (tested on Transformers 5.12.1 and PyTorch 2.9.0).

---

## License

This project is distributed under the **MIT** license.
The terms of use are provided in the [LICENSE](./LICENSE) file.

However, this software includes or imports other projects with their own separate licenses.
Please check the respective source files or directories for the license terms of those external components.

---

## Citation

If you find our work helpful, please consider citing (citation will be updated after peer-reviewed publication):

```bibtex
@misc{sinev-etal-2026-Zarya,
  author        = {Sinev, Leonid and Koziev, Ilya and Leshchuk, Vladislav},
  title         = {Zarya: A Hybrid Autoregressive--Masked Diffusion Language Model with Flexible Training and Dual-Mode Inference},
  year          = {2026},
  archiveprefix = {arXiv},
  eprint        = {2609.19868},
  primaryclass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.19868},
}
```
