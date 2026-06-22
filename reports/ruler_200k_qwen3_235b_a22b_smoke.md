# RULER 200k Qwen3-235B-A22B Smoke

Local model snapshot:

`/scratch/vunguyen13/KVCacheProject/NeuralKVCompressor/data/hf_cache/hub/models--Qwen--Qwen3-235B-A22B/snapshots/8efa61729e24bd65b1d152b5ab5409052aa80e65`

Observed config:

- `model_type: qwen3_moe`
- `num_hidden_layers: 94`
- `hidden_size: 4096`
- `num_attention_heads: 64`
- `num_key_value_heads: 4`
- `head_dim: 128`
- `max_position_embeddings: 40960`
- tokenizer warns above `131072` tokens, so this local snapshot is not native 200k.

Smoke command:

```bash
.venv/bin/python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/ruler_200k_qwen3_235b_a22b_8x/initial_step0.pt \
  --eval-file data/ruler_200k/test.jsonl \
  --limit 1 \
  --score-mode letter \
  --device auto \
  --dtype bfloat16 \
  --device-map-auto \
  --summary-file outputs/ruler_200k_qwen3_235b_a22b_8x/smoke_summary.json
```

Smoke result:

```json
{
  "compact_accuracy": 1.0,
  "full_accuracy": 0.0,
  "mean_compression": 8.328128253175098,
  "no_context_accuracy": 1.0,
  "rows": 1,
  "task_accuracy": {
    "niah_single": 1.0
  },
  "task_counts": {
    "niah_single": 1
  }
}
```

This proves the 235B sharded load, full-context prefill, exact-token compact cache, STILL beta hook, and letter scoring can execute end-to-end. It does not satisfy acceptance because it is only one row and the local A22B snapshot is not the native 200k `2507` model.


## Diagnostic after opaque choices and exact_beta

After changing RULER choices to opaque shuffled `*_CHOICE_*` values and setting
`exact_beta: 8.0`, a 3-row diagnostic on the local non-native-200k A22B snapshot produced:

```json
{
  "compact_accuracy": 1.0,
  "full_accuracy": 0.3333333333333333,
  "mean_compression": 8.29009803933621,
  "no_context_accuracy": 0.0,
  "rows": 3
}
```

The local A22B snapshot still is not an acceptance model because it is not the native
200k `2507` target, but this diagnostic shows the compact exact-token path can
retrieve labels from evidence while no-context leakage is removed.

## 12-row compact-only diagnostic

A longer compact-only diagnostic on the same local non-native-200k A22B snapshot completed across two examples per RULER-style task:

```json
{
  "compact_accuracy": 1.0,
  "mean_compression": 8.294783751074755,
  "rows": 12,
  "task_accuracy": {
    "common_words_extraction": 1.0,
    "niah_multikey": 1.0,
    "niah_multivalue": 1.0,
    "niah_single": 1.0,
    "qa": 1.0,
    "variable_tracking": 1.0
  }
}
```

This run used `--compact-only`, so it validates the compact cache scoring path and 8x evidence retention diagnostic, not the strict acceptance gate. Strict acceptance still requires the native 200k `Qwen/Qwen3-235B-A22B-Instruct-2507` model and the full 64-row non-compact-only runner.

