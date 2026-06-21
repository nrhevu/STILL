# Neural KV

Clean, extensible implementation of STILL-style neural KV-cache compaction for frozen Hugging Face causal language models. The rewrite keeps the legacy reproduction path intact while separating model modules, data preparation, training, evaluation, and utilities.

## Layout

```text
config/                 YAML defaults and experiment configs
data/                   Reproducible data instructions; generated data is ignored
src/neural_kv/models/   Compactor and HF export/load wrappers
src/neural_kv/modules/  RoPE, cache containers, beta attention bias
src/neural_kv/data/     MCQ builders, SEC/SQuAD/Gutenberg prep, teacher traces
src/neural_kv/training/ LightningModule, losses, metrics, callbacks
src/neural_kv/eval/     MCQ and CE-utilization evaluation
src/neural_kv/utils/    Config, ROCm, storage, model download helpers
scripts/                Thin command-line entrypoints
```

Compatibility wrappers such as `neural_kv.compactor` and `neural_kv.hf_training` remain so legacy tests, scripts, and checkpoints can migrate gradually.

## Environment

Use `uv` as the dependency manager. On this AMD server, keep Torch pinned to the explicit ROCm wheel index in `pyproject.toml`.

```bash
uv sync --extra train --extra dev
uv run python scripts/check_rocm_env.py --show-utilization
```

PyTorch exposes ROCm/HIP GPUs through the `cuda` API. Before any GPU run, check utilization and only use an idle GPU. The training launcher can enforce this and prefers GPU 7 by default through `runtime.preferred_gpu: 7` and `runtime.require_idle_gpu: true`.

## Prepare Data

Smoke data:

```bash
uv run python scripts/prepare_data.py \
  --source gutenberg \
  --output-dir data/mcq \
  --raw-dir data/raw \
  --train-docs 32 \
  --eval-docs 8 \
  --questions-per-doc 4 \
  --max-storage 10TB
```

SEC 6k reproduction data:

```bash
uv run python scripts/prepare_sec_facts_mcq.py \
  --output-dir data/sec_facts_random_visible_6k \
  --raw-dir data/sec_facts/raw \
  --context-chars 50000 \
  --train-rows 6000 \
  --validation-rows 600 \
  --test-rows 600 \
  --target-placement random_visible \
  --visible-target-chars 22000 \
  --max-storage 10TB \
  --user-agent "neural-kv research contact@example.com"
```

Generate teacher traces for token distillation:

```bash
HIP_VISIBLE_DEVICES=7 uv run python scripts/generate_teacher_traces.py \
  --model Qwen/Qwen3-4B \
  --input-file data/sec_facts_random_visible_6k/train.jsonl \
  --output-file data/sec_facts_random_visible_6k/train_traces_6000_512.jsonl \
  --context-length 8192 \
  --max-new-tokens 512 \
  --enable-thinking \
  --max-storage 10TB
```

## Train

Lightning smoke run:

```bash
uv run python scripts/train.py --config config/experiment/smoke_tiny_llama.yaml
```

Qwen3-4B SEC 6k reproduction run:

```bash
uv run python scripts/train.py --config config/experiment/sec_6k_qwen3_4b_8x_repro.yaml
```

The reproduction config covers Qwen/Qwen3-4B, 8192 context tokens, 1024 learned compact latents, pure 8x compression, `beta_base=zero`, token KL + reverse KL + CE, and auxiliary letter loss.

## Evaluate

```bash
uv run python scripts/evaluate.py \
  --checkpoint checkpoints/sec_6k_qwen3_4b_8x_repro/last.ckpt \
  --eval-file data/sec_facts_random_visible_6k/validation.jsonl \
  --limit 600 \
  --score-mode letter \
  --device cuda \
  --dtype bfloat16
```

CE utilization requires teacher trace rows:

```bash
uv run python scripts/evaluate_ce_utilization.py \
  --checkpoint checkpoints/sec_6k_qwen3_4b_8x_repro/last.ckpt \
  --eval-file data/sec_facts_random_visible_6k/validation_traces_600_512.jsonl \
  --limit 200 \
  --enable-thinking \
  --device cuda \
  --dtype bfloat16
```

## Export

Export a legacy `.pt` or Lightning `.ckpt` compactor checkpoint to an HF-style directory:

```bash
uv run python scripts/export_hf_checkpoint.py \
  --checkpoint checkpoints/sec_6k_qwen3_4b_8x_repro/last.ckpt \
  --output-dir checkpoints/sec_6k_qwen3_4b_8x_repro_hf
```

Load it from Python:

```python
from neural_kv.models.hf import NeuralKVForCausalLM

model = NeuralKVForCausalLM.from_pretrained(
    "checkpoints/sec_6k_qwen3_4b_8x_repro_hf",
    device="cuda",
    dtype="bfloat16",
)
```

## Reproduction Targets

Legacy report target for the selected SEC 6k Qwen3-4B checkpoint:

- validation compact accuracy near `0.966667`
- test compact accuracy near `0.958333`
- CE utilization near `0.938850` validation and `0.930875` test
- compression `8.0x`
