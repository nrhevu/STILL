# Neural KV Compressor

This branch implements a uv-managed scaffold for Baseten's STILL idea: compress each frozen LLM layer's KV cache with a learned perceiver bottleneck, then train the compactor by matching full-cache teacher behavior.

The article target is Qwen3-4B, 8192-token contexts, 1024 compact latents, about 8x KV compression, and roughly 85%+ extractive MCQ accuracy with CE utilization above 0.90. The code exposes that configuration, but reproducing those numbers needs a real GPU training run and the full generated MCQ dataset scale. The default scripts use smaller public data slices for smoke tests and iteration.

Sources used for the implementation direction:

- Baseten Research, "Towards infinite context windows: neural KV cache compaction" (April 1, 2026).
- `research.md`, especially the recommendations to use KL distillation instead of pure reconstruction, evaluate retrieval/MCQ retention, keep compression adaptive, and leave room for iterative compaction.

## Environment

`uv` is expected. If it is not on `PATH`, this repo can use the local bootstrap installed at `.uv-bootstrap/bin/uv`.

This project targets AMD GPUs with PyTorch ROCm. `pyproject.toml` pins Linux Torch packages to the official PyTorch ROCm 7.2 wheel index via uv. Do not let uv resolve the default PyPI Torch wheel, because that can install CUDA packages on this host.

```bash
.uv-bootstrap/bin/uv sync --extra train --extra dev
```

Validate the ROCm environment:

```bash
.uv-bootstrap/bin/uv run python scripts/check_rocm_env.py
```

If this reports `/dev/kfd` access failure, the current user needs membership in the `render` group before ROCm compute will work. `rocm-smi` can list devices without that compute permission, but PyTorch cannot train without it. On this host that means an admin needs to add the user and start a fresh login session:

```bash
sudo usermod -aG render "$USER"
```

## Data

The data preparation script defaults to downloading a bounded public-domain Project Gutenberg corpus and creating deterministic extractive MCQs for train/validation/test. A Hugging Face Datasets path is still available with `--source hf`. The script checks the project storage footprint against a configurable quota before and after download. The default quota is `10TB`; the check includes `data`, `checkpoints`, `artifacts`, `.venv`, `.uv-bootstrap`, and uv's shared cache.

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

The generated JSONL files are ignored by git because they can be regenerated and may become large.

## Train

Small smoke run:

```bash
.uv-bootstrap/bin/uv run python scripts/train_still.py \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --train-file data/mcq/train.jsonl \
  --eval-file data/mcq/validation.jsonl \
  --num-latents 16 \
  --context-length 256 \
  --steps 5 \
  --output-dir checkpoints/smoke
```

Article-scale run:

```bash
.uv-bootstrap/bin/uv run python scripts/train_still.py \
  --model Qwen/Qwen3-4B \
  --train-file data/mcq/train.jsonl \
  --eval-file data/mcq/validation.jsonl \
  --num-latents 1024 \
  --head-specific-latents \
  --context-length 8192 \
  --steps 2900 \
  --kl-weight 1.0 \
  --reverse-kl-weight 0.0 \
  --ce-weight 0.1 \
  --output-dir checkpoints/qwen3_4b_8x \
  --max-storage 10TB
```

Target gate after an article-scale run:

```bash
.uv-bootstrap/bin/uv run python scripts/check_performance_target.py \
  checkpoints/qwen3_4b_8x/summary.json
```

## What Is Implemented

- Per-layer STILL perceiver compactor.
- Optional per-KV-head latent query tables to avoid forcing all KV heads through a shared latent basis.
- RoPE-aware key unrotation, internal positional cross-attention, and compact-key re-rotation.
- Identity-style initialization so latents begin as position-local cache copies.
- Learned per-layer/per-KV-head beta attention biases.
- HF model patching via a context manager so beta can be added as an additive attention mask during student forwards.
- KL teacher-student loss on answer tokens, optional reverse KL for bidirectional distillation, and optional exact-answer CE.
- Storage quota checks for data and artifacts.
- A deterministic public-data MCQ builder for train/test bootstrapping.

## Current Limitations

- The Baseten-quality target is not proven until a Qwen3-4B 8K/1024-latent training run is completed and evaluated on held-out MCQs. This code is the runnable path toward that target, not a completed benchmark claim.
- The public MCQ generator is a substitute for the article's Claude-generated on-policy MCQs. It is useful for engineering, but final accuracy should be measured on stronger generated or human-authored extractive QA.
