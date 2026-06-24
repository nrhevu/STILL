# VLM Compactor Acceptance Archives

This directory stores accepted Qwen3-VL compactor benchmark runs. A run is accepted only when compact-cache accuracy is at least 95% of the matching full-cache accuracy, both in aggregate and for each enabled dataset/resolution/image-token sweep group.

Use the acceptance runner so evaluation, target checking, and archiving happen as one reproducible workflow:

```bash
uv run --extra train --extra dev python scripts/run_vlm_compactor_acceptance.py \
  --checkpoint /path/to/qwen3_vl_compactor.ckpt \
  --run-name qwen3-vl-95pct
```

Each accepted archive must contain:

- `summary.json`: evaluator summary with `compact_vs_full_accuracy >= 0.95`.
- `details.jsonl`: per-example full-cache and compact-cache results.
- `archive_manifest.json`: target checks that passed at archive time.
- `README.md`: human-readable run summary.

Run preflight before reserving GPU time:

```bash
uv run --extra train --extra dev python scripts/run_vlm_compactor_acceptance.py \
  --checkpoint /path/to/qwen3_vl_compactor.ckpt \
  --preflight-only
```
