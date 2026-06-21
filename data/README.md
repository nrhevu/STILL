# Data

This directory stores generated and downloaded data. The files are intentionally ignored by git because reproduction datasets, teacher traces, HF caches, and checkpoints can be large.

## SEC 6k Reproduction Data

The reported legacy result uses SEC companyfacts MCQs with random-visible target placement:

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

Generate full-cache teacher traces for token distillation. On the 8-GPU AMD server, shard these jobs manually after checking utilization and only use idle GPUs:

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

Validation and test traces are generated the same way from `validation.jsonl` and `test.jsonl`.

## Smoke Data

For quick local testing, build a small public-domain Gutenberg MCQ set:

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

## Storage

All data commands check project-controlled storage with `--max-storage`. Model caches should stay under `data/hf_cache` when possible so they are included in quota reports.
