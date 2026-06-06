# Experiment Results Report

Date: 2026-06-06 UTC

## Scope

This report summarizes the latest known results for the Neural KV Cache Compaction experiments in this workspace. The target setup is:

- Model: Qwen/Qwen3-4B
- Hardware/runtime: AMD GPU with PyTorch ROCm
- Context length: 8192 tokens
- Compact cache size: 1024 latent tokens
- Compression ratio: 8.0x
- Storage constraint: keep total experiment storage under 10 TB

## Storage

The last recorded storage snapshot was well below the 10 TB limit:

- `data`: about 11 GB
- `checkpoints`: about 28 GB
- `.cache`: about 1.6 GB
- Quota report: about 68 GB used out of 10 TB

No new storage-heavy operation was run while writing this report.

## Data

Current main dataset:

- `data/sec_facts_random_visible_3k/train.jsonl`: 3000 examples
- `data/sec_facts_random_visible_3k/validation.jsonl`: 400 examples
- `data/sec_facts_random_visible_3k/test.jsonl`: 400 examples
- `data/sec_facts_random_visible_3k/train_traces_3000_512.jsonl`: 3000 teacher traces

Trace validation previously showed:

- 3000 rows
- train ids matched the expected prefix
- 2999 unique ids
- 2984 examples contained the gold answer near the tail
- token count median: 234
- token count max: 512

Validation/test teacher traces for the 3k split were not yet recorded as completed.

## Best Results So Far

### Strongest Pure Post-Fix Run

Checkpoint directory:

`checkpoints/qwen3_4b_sec_random_visible_3k_cachefix_pure1024_identity_400step_b2_lr1e5_w05`

This is the most promising pure compaction run after fixing cache mutation during teacher continuations.

Known eval-limit 128 results:

| Step | Compact Accuracy | Full Accuracy | No-Context Accuracy | Compression |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.7109375 | 1.0 | 0.3359375 | 8.0 |
| 200 | 0.8828125 | 1.0 | 0.3359375 | 8.0 |

Interpretation:

- Step 200 reached 88.28% compact accuracy on eval-limit 128.
- This already exceeds the rough article target of about 85% MCQ accuracy on the small validation slice.
- Full-context accuracy remained 100%, while no-context stayed near 33.6%, so the task still depends on retrieved/compressed context.
- The remaining required confirmation is full validation/test evaluation, not just eval-limit 128.

### Best Hybrid Lexical Anchor Run

Checkpoint:

`checkpoints/qwen3_4b_sec_random_visible_lexical128_latent896_trace1200_400step_b2_lr1e5/final.pt`

Results:

| Split | Compact Accuracy | Full Accuracy | No-Context Accuracy | Compression |
| --- | ---: | ---: | ---: | ---: |
| Validation 400 | 0.9975 | 0.9975 | 0.315 | 8.0 |
| Test 400 | 1.0 | 1.0 | 0.33 | 8.0 |

Interpretation:

- This result is very strong locally.
- It is not a pure Baseten/STILL-style result because it uses query-guided exact lexical anchors.
- It should be treated as a useful upper-bound/control, not the main article-faithful result.

## Earlier Pure Results

Before the cache-mutation fix, the best pure checkpoint was:

`checkpoints/qwen3_4b_sec_random_visible_trace1200_aux_from500_900step_b2_lr1e7_w05_eval128/step_550.pt`

Known results:

| Evaluation | Compact Accuracy | Full Accuracy | No-Context Accuracy |
| --- | ---: | ---: | ---: |
| Eval-limit 128 | 0.578125 | 1.0 | 0.3359375 |
| Full validation 400 | 0.56 | not recorded here | not recorded here |

Larger-data pure continuation before the cache fix did not improve:

- Step 600: 0.5625
- Step 700: 0.5546875
- Step 800: 0.546875

Reverse KL before the cache fix:

- Best step 650: 0.5703125
- Final step 750: 0.5546875

Head-specific latents before the cache fix:

- Best eval-limit 128 at step 650: 0.5859375
- Full validation 400: 0.5725

These older results are now less reliable as indicators because the training objective was affected by cache contamination.

## Important Fix

A significant bug was found and fixed in the training path:

- Hugging Face `DynamicCache.update()` can mutate the original cache even when `use_cache=False`.
- The teacher continuation path previously reused `full_outputs.past_key_values`.
- That allowed prompt/target tokens to be appended into the cache before compaction.
- As a result, older training runs could learn from a contaminated cache rather than a clean context-only cache.

The fix introduced fresh read-only cache wrappers before continuation forwards. Unit tests passed after the fix:

- `28 passed`

This makes the post-fix fresh identity run the key result to trust next.

## Current Accuracy Snapshot

Latest known pure post-fix compact accuracy:

- 0.8828125 at step 200 on eval-limit 128

Latest known full-context baseline:

- 1.0 on the same eval slice

Latest known no-context baseline:

- 0.3359375 on the same eval slice

Latest known hybrid validation/test accuracy:

- Validation 400: 0.9975
- Test 400: 1.0

## Gaps Before Claiming Article-Level Success

The current pure post-fix result is promising but not yet fully validated. Remaining evidence needed:

1. Evaluate the best pure checkpoint on the full 400-example validation split.
2. Evaluate the same checkpoint on the 400-example test split.
3. Generate validation/test teacher traces for the 3k split if CE utilization needs held-out measurement.
4. Run CE utilization evaluation and confirm utilization is above 0.9.
5. Confirm final storage usage remains under 10 TB after any additional trace/checkpoint generation.

## Bottom Line

The project has a strong post-fix pure result: 88.28% compact accuracy at 8x compression on a 128-example validation slice. This is the first result that appears to reach the article-level MCQ accuracy target under the pure compaction setup, but it still needs full validation/test and CE utilization before it should be considered complete.

