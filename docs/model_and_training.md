# Model and Training Guide

This repo trains a STILL-style neural KV cache compactor for a frozen
decoder-only Hugging Face language model. The target setup follows Baseten's
article configuration: Qwen3-4B, 8192 source/context tokens, 1024 compact
latent KV tokens per layer, about 8x cache compression, and high retention on
long-context MCQs.

The base LLM is not fine-tuned. Training updates only the compactor that turns a
full KV cache into a smaller KV cache.

## What The Model Does

For a normal decoder-only transformer, a long context produces one KV cache per
layer:

```text
layer_keys, layer_values: [batch, kv_heads, source_tokens, head_dim]
```

The compactor replaces the `source_tokens` dimension with a smaller fixed
number of learned latent slots:

```text
compact_keys, compact_values: [batch, kv_heads, compact_tokens, head_dim]
beta:                         [batch, kv_heads, compact_tokens]
```

For the article-style pure configuration:

```text
source_tokens = 8192
compact_tokens = num_latents = 1024
compression = 8192 / 1024 = 8x
```

`beta` is an additive attention bias. During the student forward pass the
patched Hugging Face attention modules add beta to the attention mask, so the
model can learn which compact slots should receive more or less attention.

## Compactor Architecture

The main modules are:

- `StillCompactor` in `src/neural_kv/compactor.py`
- `StillLayerCompactor` in `src/neural_kv/compactor.py`
- `CompactKVCache` in `src/neural_kv/cache.py`
- beta attention patching in `src/neural_kv/attention_bias.py`
- training/evaluation helpers in `src/neural_kv/hf_training.py`

Each transformer layer is compacted independently unless
`--layer-compactor-groups` is used to share compactors across depth.

Per layer, the flow is:

1. Take the frozen model's full source-context keys and values.
2. Undo RoPE on the keys with inverse RoPE at the original token positions.
3. Concatenate unrotated keys and values into one feature vector:

   ```text
   kv_input = [K_unrotated; V]
   latent_dim = 2 * head_dim
   ```

4. Start from learned latent query tables. With `--head-specific-latents`, each
   KV head gets its own latent query table; otherwise all KV heads share one.
5. Cross-attend from latent slots into the full KV input. Latent slots use
   evenly spaced virtual positions across the source sequence.
6. Run self-attention over the latent slots so compact slots can coordinate.
7. Project latent states into compact keys, compact values, and beta biases.
8. Re-apply RoPE to compact keys at the latent virtual positions.
9. Return a `CompactKVCache` that can be converted to a Transformers
   `DynamicCache`.

The initialization is intentionally close to an identity cache copy:

- key and value heads initially select the key/value halves of the latent state
- the first cross-attention block has an active identity-style path
- `--beta-base zero` starts beta at zero, which matches the current article-style
  training path in this repo

## Optional Exact Tokens

The pure STILL setup uses only learned latent slots:

```bash
--sink-tokens 0 --exact-tokens 0
```

The code also supports exact uncompressed tokens for experiments:

- `--sink-tokens N`: prepend the first N source KV tokens exactly
- `--exact-tokens N`: prepend additional exact source KV tokens
- `--exact-strategy prefix|even|kv_norm|lexical`

`lexical` is query-guided and useful as a practical hybrid baseline, but it is
not the pure article method because the compactor sees the query when selecting
exact tokens.

## Training Objective

Training is teacher-student distillation.

For each MCQ row:

1. Encode the source/context into `context_ids`.
2. Run the frozen full model on the context with `use_cache=True`.
3. Run the teacher continuation against a fresh copy of the full cache.
4. Compact the original full context cache.
5. Run the student continuation against the compact cache and beta biases.
6. Match the teacher's logits with KL loss, optionally plus reverse KL and
   ground-truth CE.

The important implementation detail is the fresh cache copy in
`_fresh_dynamic_cache`. Hugging Face `DynamicCache.update(...)` can mutate the
cache object even when `use_cache=False`; the teacher continuation therefore
uses a fresh wrapper so the source cache passed into the compactor remains a
clean context-only cache.

Supported training modes:

- `--loss-mode token`: distill the teacher distribution over target tokens
- `--loss-mode letter`: distill only the four MCQ answer-letter logits
- `--target-mode teacher_response`: train on generated full-cache teacher
  response traces
- `--target-mode letter`: train on the gold MCQ letter
- `--target-mode choice_text`: train on the gold answer text

The strongest article-style path in this repo is token-level distillation over
teacher responses:

```text
--loss-mode token
--target-mode teacher_response
--kl-weight 1.0
--reverse-kl-weight 0.5
--ce-weight 0.1
--aux-letter-loss-weight 0.05
```

## ROCm Environment

Install dependencies with uv:

```bash
.uv-bootstrap/bin/uv sync --extra train --extra dev
```

Check ROCm visibility:

```bash
.uv-bootstrap/bin/uv run python scripts/check_rocm_env.py
```

On AMD/ROCm, PyTorch exposes HIP devices through the CUDA API. That means repo
commands should normally use:

```text
--device cuda
```

Select the AMD GPU with `HIP_VISIBLE_DEVICES`:

```bash
HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/check_rocm_env.py
```

The Docker wrapper sets project-local caches under `data/` so model downloads,
HF cache, uv cache, compiler cache, and generated traces are included in the
storage accounting.

## Storage Guardrail

The user constraint is that total project storage should stay under 10TB.
Scripts that can create large artifacts accept:

```text
--max-storage 10TB
```

Check current project-controlled usage:

```bash
.uv-bootstrap/bin/uv run python scripts/storage_report.py
```

The storage check covers the main generated roots such as `data`,
`checkpoints`, `artifacts`, uv bootstrap/cache paths, and local virtualenvs.

## Build Data

There are two dataset builders.

For quick public-corpus smoke tests, use Gutenberg data:

```bash
.uv-bootstrap/bin/uv run python scripts/prepare_data.py \
  --source gutenberg \
  --output-dir data/mcq \
  --raw-dir data/raw \
  --train-docs 512 \
  --eval-docs 128 \
  --questions-per-doc 20 \
  --max-storage 10TB
```

For the stronger long-context financial MCQ setup, use SEC companyfacts:

```bash
.uv-bootstrap/bin/uv run python scripts/prepare_sec_facts_mcq.py \
  --output-dir data/sec_facts_random_visible_3k \
  --context-chars 50000 \
  --train-rows 3000 \
  --validation-rows 400 \
  --test-rows 400 \
  --target-placement random_visible \
  --visible-target-chars 22000 \
  --max-storage 10TB \
  --user-agent "neural-kv-compressor research contact@example.com"
```

Each JSONL row contains:

- `context`: long document text
- `question`: MCQ question
- `choices`: four answer choices
- `answer_index`: zero-based gold choice
- `answer`: gold answer text

## Generate Teacher Traces

`--target-mode teacher_response` requires rows with
`teacher_response_token_ids`. Generate them with the full frozen model:

```bash
HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/generate_teacher_traces.py \
  --model Qwen/Qwen3-4B \
  --input-file data/sec_facts_random_visible_3k/train.jsonl \
  --output-file data/sec_facts_random_visible_3k/train_traces_3000_512.jsonl \
  --context-length 8192 \
  --max-new-tokens 512 \
  --device cuda \
  --dtype bfloat16 \
  --enable-thinking \
  --max-storage 10TB
```

For long runs, shard trace generation so progress is recoverable:

```bash
HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/generate_teacher_traces.py \
  --model Qwen/Qwen3-4B \
  --input-file data/sec_facts_random_visible_3k/train.jsonl \
  --output-file data/sec_facts_random_visible_3k/train_traces_part00.jsonl \
  --start-index 0 \
  --end-index 375 \
  --context-length 8192 \
  --max-new-tokens 512 \
  --device cuda \
  --dtype bfloat16 \
  --enable-thinking \
  --max-storage 10TB
```

After shards complete, combine them in numeric order into one train trace file.
The validation and test files do not need traces for MCQ accuracy evaluation,
but CE utilization evaluation does require traced rows.

```bash
cat data/sec_facts_random_visible_3k/train_traces_part*.jsonl \
  > data/sec_facts_random_visible_3k/train_traces_3000_512.jsonl
```

## Smoke Train

Use a tiny model to verify the training loop:

```bash
.uv-bootstrap/bin/uv run python scripts/train_still.py \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --train-file data/mcq/train.jsonl \
  --eval-file data/mcq/validation.jsonl \
  --output-dir checkpoints/smoke \
  --num-latents 16 \
  --context-length 256 \
  --steps 5 \
  --device auto \
  --dtype bfloat16 \
  --max-storage 10TB
```

## Article-Style Qwen3-4B Train

This is the pure 8x learned-latent configuration: 8192 context tokens, 1024
latents, no exact anchors, beta initialized at zero, and no query-guided exact
token selection.

```bash
HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/train_still.py \
  --model Qwen/Qwen3-4B \
  --train-file data/sec_facts_random_visible_3k/train_traces_3000_512.jsonl \
  --eval-file data/sec_facts_random_visible_3k/validation.jsonl \
  --output-dir checkpoints/qwen3_4b_sec_random_visible_3k_pure1024 \
  --num-latents 1024 \
  --context-length 8192 \
  --steps 2900 \
  --batch-size 2 \
  --learning-rate 0.00001 \
  --kl-weight 1.0 \
  --reverse-kl-weight 0.5 \
  --ce-weight 0.1 \
  --eval-every 100 \
  --save-every 100 \
  --eval-limit 128 \
  --loss-mode token \
  --target-mode teacher_response \
  --aux-letter-loss-weight 0.05 \
  --balanced-answer-sampling \
  --enable-thinking \
  --score-mode letter \
  --beta-base zero \
  --device cuda \
  --dtype bfloat16 \
  --max-storage 10TB
```

The key output files are:

- `metrics.jsonl`: metrics for every training step
- `step_<N>.pt`: periodic checkpoints when `--save-every` is set
- `final.pt`: final checkpoint
- `summary.json`: final metrics summary

To continue a compatible run, pass `--init-checkpoint` and keep all architecture
flags the same:

```bash
HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/train_still.py \
  --init-checkpoint checkpoints/qwen3_4b_sec_random_visible_3k_pure1024/step_1000.pt \
  --model Qwen/Qwen3-4B \
  --train-file data/sec_facts_random_visible_3k/train_traces_3000_512.jsonl \
  --eval-file data/sec_facts_random_visible_3k/validation.jsonl \
  --output-dir checkpoints/qwen3_4b_sec_random_visible_3k_pure1024_continue \
  --num-latents 1024 \
  --context-length 8192 \
  --steps 1000 \
  --batch-size 2 \
  --learning-rate 0.000005 \
  --kl-weight 1.0 \
  --reverse-kl-weight 0.5 \
  --ce-weight 0.1 \
  --eval-every 100 \
  --save-every 100 \
  --eval-limit 128 \
  --loss-mode token \
  --target-mode teacher_response \
  --aux-letter-loss-weight 0.05 \
  --balanced-answer-sampling \
  --enable-thinking \
  --score-mode letter \
  --beta-base zero \
  --device cuda \
  --dtype bfloat16 \
  --max-storage 10TB
```

## Useful Training Variants

Head-specific latent tables:

```text
--head-specific-latents
```

This gives each KV head group separate latent queries. It increases compactor
parameters but can reduce interference between heads.

Latent-only specialization from an existing checkpoint:

```text
--trainable-scope latents
```

Other trainable scopes are `all`, `beta`, `heads`, `beta_heads`, `latents`, and
`latents_beta`.

Hybrid exact-token baseline:

```text
--exact-tokens 128 --exact-strategy lexical --num-latents 896
```

This still gives 1024 total compact tokens, but 128 of them are exact
query-selected source tokens. Treat this as a practical baseline, not as the
pure article method.

## Evaluate Accuracy

Evaluate a checkpoint on validation:

```bash
HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/qwen3_4b_sec_random_visible_3k_pure1024/final.pt \
  --eval-file data/sec_facts_random_visible_3k/validation.jsonl \
  --limit 400 \
  --score-mode letter \
  --device cuda \
  --dtype bfloat16 \
  --max-storage 10TB
```

Evaluate on test:

```bash
HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/qwen3_4b_sec_random_visible_3k_pure1024/final.pt \
  --eval-file data/sec_facts_random_visible_3k/test.jsonl \
  --limit 400 \
  --score-mode letter \
  --device cuda \
  --dtype bfloat16 \
  --max-storage 10TB
```

Use an explicit row count for `--limit`. In the current
`evaluate_checkpoint.py`, `--limit 0` does not mean all rows.

Important metrics:

- `compact_accuracy`: accuracy with compact cache
- `full_accuracy`: accuracy with the original full cache
- `no_context_accuracy`: accuracy without the cached context
- `mean_compression`: source tokens divided by compact cache tokens
- `compact_prediction_counts`: answer-letter distribution for quick collapse checks

The article-style target is roughly:

```text
compact_accuracy >= 0.85
mean_compression >= 8.0
```

Run the target gate on a training summary:

```bash
.uv-bootstrap/bin/uv run python scripts/check_performance_target.py \
  checkpoints/qwen3_4b_sec_random_visible_3k_pure1024/summary.json
```

## Evaluate CE Utilization

CE utilization compares compact-cache cross entropy to full-cache and
no-context cross entropy on teacher-response tokens:

```text
ce_utilization = (no_context_ce - compact_ce) / (no_context_ce - full_ce)
```

Generate teacher traces for validation or test first, then run:

```bash
HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/evaluate_ce_utilization.py \
  --checkpoint checkpoints/qwen3_4b_sec_random_visible_3k_pure1024/final.pt \
  --eval-file data/sec_facts_random_visible_3k/validation_traces_400_512.jsonl \
  --limit 400 \
  --device cuda \
  --dtype bfloat16 \
  --enable-thinking \
  --max-storage 10TB
```

The article-style CE utilization target is about:

```text
ce_utilization >= 0.90
```

## Reading Training Progress

During training, `metrics.jsonl` receives one JSON object per optimizer step.
Useful fields:

- `step`
- `loss`
- `kl`
- `reverse_kl`
- `ce`
- `compression`
- `compact_accuracy`, when `--eval-every` fires
- `full_accuracy`, when `--eval-every` fires
- `no_context_accuracy`, when `--eval-every` fires
- `mean_compression`, when `--eval-every` fires

Tail the latest metrics:

```bash
tail -n 5 checkpoints/qwen3_4b_sec_random_visible_3k_pure1024/metrics.jsonl
```

Pretty-print the final summary:

```bash
python -m json.tool checkpoints/qwen3_4b_sec_random_visible_3k_pure1024/summary.json
```

## Troubleshooting

If ROCm is not visible:

- run `scripts/check_rocm_env.py`
- confirm `/dev/kfd` and `/dev/dri` are passed through Docker
- confirm the process has access to the `render` group
- keep using `--device cuda` inside ROCm PyTorch

If compact accuracy is near no-context accuracy:

- verify the teacher trace file contains `teacher_response_token_ids`
- verify `--context-length 8192` and `--num-latents 1024`
- verify the evaluation uses the same chat/thinking format as training
- inspect `compact_prediction_counts` for answer collapse
- compare `full_accuracy` against `compact_accuracy`; if full accuracy is low,
  the dataset or prompt format is the problem, not the compactor

If continuation training from a checkpoint fails:

- `--init-checkpoint` requires matching model, context length, number of latents,
  exact-token settings, number of blocks, beta base, and layer sharing settings
- only use older shared-latent checkpoints with `--head-specific-latents` when
  the script can expand the latent table shape

If storage grows too quickly:

- run `scripts/storage_report.py`
- delete obsolete checkpoint directories only after confirming they are not the
  best known run
- keep `--max-storage 10TB` on data, trace, train, and eval commands
