# Neural KV Compressor

This branch implements a uv-managed scaffold for Baseten's STILL idea: compress each frozen LLM layer's KV cache with a learned perceiver bottleneck, then train the compactor by matching full-cache teacher behavior.

The article target is Qwen3-4B, 8192-token contexts, 1024 compact latents, about 8x KV compression, and roughly 85%+ extractive MCQ accuracy with CE utilization above 0.90. The code exposes that configuration, but reproducing those numbers needs a real GPU training run and the full generated MCQ dataset scale. The default scripts use smaller public data slices for smoke tests and iteration.

For the detailed model and training procedure, see [docs/model_and_training.md](docs/model_and_training.md).

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

## Validate A Hugging Face Checkpoint

Use this path to validate the best SEC 6k checkpoint downloaded from Hugging Face.
The validation command uses ROCm through `scripts/rocm_docker_run.sh`; PyTorch
still sees AMD GPUs as `cuda`.

First, create the SEC validation split. If `data/sec_facts_random_visible_6k/validation.jsonl`
already exists, this step can be skipped. The script writes all split files, so
the train/test counts below are kept at one row to minimize validation-only
storage while preserving the 600-row validation split.

```bash
.uv-bootstrap/bin/uv run python scripts/prepare_sec_facts_mcq.py \
  --output-dir data/sec_facts_random_visible_6k \
  --raw-dir data/sec_facts/raw \
  --context-chars 50000 \
  --train-rows 1 \
  --validation-rows 600 \
  --test-rows 1 \
  --target-placement random_visible \
  --visible-target-chars 22000 \
  --max-storage 10TB \
  --user-agent "neural-kv-compressor validation contact@example.com"
```

Set the Hugging Face repo that contains `step_800.pt`. For the default private
repo created by the upload helper, the repo name is
`<hf-user>/neural-kv-compressor-qwen3-4b-sec-6k-best`.

```bash
export HF_REPO_ID="<hf-user>/neural-kv-compressor-qwen3-4b-sec-6k-best"
export HF_CHECKPOINT_DIR="checkpoints/hf_qwen3_4b_sec_6k_best"
read -rsp "HF token: " HF_TOKEN; echo
export HF_TOKEN
```

Download the checkpoint into the project checkpoint directory:

```bash
mkdir -p "$HF_CHECKPOINT_DIR"
.uv-bootstrap/bin/uv run python - <<'PY_HF_DOWNLOAD'
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id=os.environ["HF_REPO_ID"],
    filename="step_800.pt",
    repo_type="model",
    local_dir=os.environ["HF_CHECKPOINT_DIR"],
    token=os.environ.get("HF_TOKEN"),
)
print(path)
PY_HF_DOWNLOAD
```

If the base model is not already cached in `data/hf_cache`, download it before
entering the ROCm container. This keeps the model cache project-local and under
the same `10TB` storage check.

```bash
HF_HOME=data/hf_cache .uv-bootstrap/bin/uv run python scripts/download_model.py \
  --model Qwen/Qwen3-4B \
  --max-storage 10TB
```

After the Hugging Face downloads finish, remove the token from the shell:

```bash
unset HF_TOKEN
```

Run the 600-row MCQ validation. Do not pass `--enable-thinking`; this matches
the validation path used in the completed report.

```bash
HIP_VISIBLE_DEVICES=0 scripts/rocm_docker_run.sh python scripts/evaluate_checkpoint.py \
  --checkpoint "$HF_CHECKPOINT_DIR/step_800.pt" \
  --eval-file data/sec_facts_random_visible_6k/validation.jsonl \
  --limit 600 \
  --score-mode letter \
  --device cuda \
  --dtype bfloat16 \
  --max-storage 10TB
```

The completed SEC 6k report measured `step_800.pt` at `0.966667` compact
validation accuracy, `0.996667` full-cache validation accuracy, and `8.0x`
compression. See `reports/performance_2026-06-08_6k_rocm.md` for the full
validation/test report.

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
