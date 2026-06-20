# Changes From The Original Design

Date: 2026-06-20

This document compares the original design note in `docs/original.md` with the
current repository state. It uses four evidence streams:

- source code under `src/neural_kv/`, `scripts/`, `configs/`, and `tests/`
- git history on `main`
- current documentation under `README.md`, `docs/`, and `reports/`
- recorded experiment reports and checkpoint workflow notes

Important context: at the time of this analysis, `docs/original.md` is an
untracked baseline document. Git history therefore does not show it evolving
directly. The committed history starts with `research.md`, then turns the idea
into an implementation.

## Executive Summary

The original design is a Baseten STILL explainer: compress each layer's full KV
cache from an 8192-token context into 1024 learned latent KV slots, keep the base
LLM frozen, and train the compactor so compact-cache logits match full-cache
teacher logits.

The current repo still implements that core idea, but it has moved far beyond
the original design note:

- It is now a runnable Python package, not just a research/design description.
- The implementation targets Hugging Face decoder-only models, especially
  `Qwen/Qwen3-4B`, on AMD ROCm.
- Training now supports teacher-response trace distillation, reverse KL,
  auxiliary letter loss, letter-only MCQ scoring, chat-template handling, and
  scoped fine-tuning of subsets of compactor parameters.
- The architecture now includes optional per-KV-head latent query tables, grouped
  layer compactors, exact-token hybrid anchors, sink tokens, configurable beta
  initialization, and beta-based pruning.
- The evaluation story has changed from citing Baseten multi-domain results to
  local SEC company-facts experiments, including a completed 6k ROCm run with
  8x compression and strong MCQ/CE results.
- Some older docs are stale: they still describe target reproduction as future
  work even though later reports document a completed SEC 6k run.

The key caveat is scope. The current strongest result is a local SEC benchmark,
not a reproduction of Baseten's full latent-scaling, multi-domain, and
cross-domain transfer matrix.

## What Stayed The Same

The central STILL design survived.

The current implementation keeps the base LLM frozen and trains a separate
compactor. `load_model_and_tokenizer` freezes model parameters, and
`training_forward` runs the same model in two cache modes: full cache as teacher
and compact cache as student.

The compactor is still per-layer by default. `StillCompactor` builds a
`StillLayerCompactor` for each transformer layer unless grouped layer sharing is
enabled.

The Perceiver-style bottleneck is still the main architecture:

- learned latent slots
- cross-attention from latents into full KV-derived inputs
- self-attention among latents
- output heads for compact keys, compact values, and beta biases

The RoPE-aware path from the original design is implemented:

- undo RoPE on source keys
- use positional RoPE inside latent cross-attention
- re-apply RoPE to compact keys at evenly spaced virtual latent positions

The original training concept is also still present. The code computes
teacher-student KL on answer tokens, optionally adds CE, and reports MCQ,
full-cache, compact-cache, no-context, compression, KL, and CE-utilization
metrics.

The original warning about "infinite context" also remains true. The current
mainline implementation compresses a fixed context into a smaller cache. It does
not implement recursive or iterative compaction over an unbounded stream.

## Major Architecture Changes

### 1. The design became a concrete Hugging Face cache interface

Original design:

```text
Full KV cache -> Perceiver compactor -> compact K, compact V, beta
```

Current implementation:

```text
Transformers past_key_values
  -> normalize_past_key_values
  -> StillCompactor
  -> CompactKVCache
  -> DynamicCache.from_legacy_cache(...)
  -> frozen HF model continuation
```

The repo now has a `CompactKVCache` container with serialization, device moves,
byte accounting, compression-ratio accounting, and conversion back to
Transformers `DynamicCache`.

Files:

- `src/neural_kv/cache.py`
- `src/neural_kv/compactor.py`
- `src/neural_kv/hf_training.py`

### 2. Beta became an actual patched attention-mask mechanism

The original document mentions `beta_l` as part of the compact cache. The
current implementation makes beta operational by patching Hugging Face
attention modules.

`enable_still_attention_bias` wraps attention layers that expose `layer_idx` and
an `attention_mask` argument. During student forwards, `still_biases(...)`
stores the compact-cache beta tensors in a context variable, and the patch adds
beta into the additive attention mask.

This is a practical implementation choice that avoids changing frozen model
weights or replacing attention kernels.

Files:

- `src/neural_kv/attention_bias.py`
- `scripts/train_still.py`
- `scripts/evaluate_checkpoint.py`
- `scripts/evaluate_ce_utilization.py`

### 3. Beta initialization is now configurable

Original design emphasizes identity-style initialization.

Current code supports two beta bases:

- `zero`
- `log_compression`

The library class accepts both. The training script defaults to `zero`, which
matches the later identity/cache-fix training path documented in the repo.

This matters because older checkpoints may have used `log_compression`, while
the strongest documented pure path uses `--beta-base zero`.

### 4. Per-KV-head latent query tables were added

Original design describes learned latent queries generally. Current code can use
either:

- one shared latent table across KV heads
- one latent query table per KV head group

The `--head-specific-latents` option was added to reduce interference between
KV heads. This is a real architectural extension beyond the original baseline.

Files:

- `src/neural_kv/compactor.py`
- `scripts/train_still.py`
- `tests/test_compactor.py`

Related commit:

- `05c0b66 Add head-specific latent queries`

### 5. Grouped layer compactors were added

Original design describes independent per-layer compression.

Current code can share compactors across depth with `--layer-compactor-groups`.
When enabled, multiple transformer layers map to a smaller number of compactor
modules.

This is a parameter-sharing experiment, not part of the original pure per-layer
STILL description.

Related commit:

- `2d7cfa7 Add grouped layer compactor option`

### 6. Exact-token hybrid modes were added

This is one of the biggest divergences.

Original pure design:

```text
full KV cache -> learned latent KV cache
```

Current repo additionally supports:

- `--sink-tokens N`: prepend exact prefix KV tokens
- `--exact-tokens N`: prepend selected exact source KV tokens
- `--exact-strategy prefix|even|kv_norm|lexical`

The lexical strategy is query-guided. It selects context lines/tokens that match
the question and keeps their KV entries exactly.

This improves retrieval/factual behavior but changes the scientific claim. A
hybrid exact-anchor run is not a pure STILL compaction result because part of
the cache is retained rather than synthesized.

The reports make this distinction clearly: the near-perfect hybrid lexical
anchor results are useful controls, but the pure 6k SEC claim uses 1024 learned
latents and no exact-anchor tokens.

Files:

- `src/neural_kv/compactor.py`
- `src/neural_kv/hf_training.py`
- `scripts/train_still.py`
- `docs/experiment_results_report.md`

Related commits:

- `88d25bf Add exact sink tokens to compact cache`
- `cd7ae4f Add optional exact-anchor cache tokens`
- `ecdc3a1 Add lexical query-guided exact anchors`

### 7. Beta-based post-pruning was added

`prune_cache_by_beta` can keep a fraction of compact slots based on learned beta
scores. This is an adaptive-budget hook inspired by later research notes.

It is not part of the original Baseten/STILL design summary.

File:

- `src/neural_kv/cache.py`

## Major Training Changes

### 1. The objective expanded beyond simple KL

Original design centers on KL divergence between:

- teacher: frozen LLM with full KV cache
- student: same frozen LLM with compact KV cache

Current training supports:

- token-level KL
- reverse KL
- ground-truth CE
- letter-only MCQ KL/CE
- auxiliary direct-letter loss
- teacher-response trace distillation

The completed SEC 6k run used:

```text
token KL + reverse KL 0.5 + CE 0.1 + auxiliary letter loss 0.05
```

This is materially different from the original simpler training description.
The added losses are engineering responses to MCQ answer stability and teacher
distribution matching.

Files:

- `src/neural_kv/hf_training.py`
- `scripts/train_still.py`

Related commits:

- `899b0ba Add batched MCQ letter distillation`
- `fe6b7b7 Add auxiliary direct-letter training loss`
- `7666e29 Add bidirectional KL distillation option`

### 2. Teacher-response trace distillation was added

Original design says answer text should come from the model being compacted, so
the compactor learns the target model's distribution rather than another model's
style.

Current repo operationalizes this by generating full-cache teacher responses and
training against `teacher_response_token_ids`.

The trace path creates JSONL rows containing:

- `teacher_response`
- `teacher_response_token_ids`
- `teacher_response_token_count`
- `teacher_response_gold_letter`

This changed the project from direct MCQ answer training toward continuation
distillation from cached teacher traces.

Files:

- `scripts/generate_teacher_traces.py`
- `src/neural_kv/hf_training.py`
- `docs/model_and_training.md`

Related commit:

- `f9e6c53 Add teacher-response trace distillation path`

### 3. Qwen chat formatting and thinking flags were added

Original design is model-agnostic and does not specify Qwen chat-template
behavior.

Current code handles:

- Qwen chat templates
- system-context prompts
- no-context prompts
- `--enable-thinking`
- `--eval-enable-thinking`
- no-think instructions when thinking is disabled

This became necessary because the target model is `Qwen/Qwen3-4B`, and prompt
formatting affects answer extraction and distillation.

Related commits:

- `bc2e32f Align Qwen chat evaluation and RoPE compaction`
- `96f10f9 Separate training and eval thinking flags`

### 4. Training can resume and fine-tune selected parameter scopes

The repo now supports:

- `--init-checkpoint`
- compatibility checks for architecture flags
- expansion from shared latents to head-specific latents
- `--trainable-scope all|beta|heads|beta_heads|latents|latents_beta`

This is a workflow and experimentation layer absent from the original design.

Related commits:

- `2be356b Allow resuming STILL compactor training`
- `90747aa Add scoped compactor fine-tuning`
- `5d00048 Add latent-only training scopes`

### 5. Cache mutation was found and fixed

This is a major implementation discovery not present in the original design.

The problem:

- Hugging Face `DynamicCache.update()` can mutate a cache object.
- Teacher continuation previously reused `full_outputs.past_key_values`.
- Prompt/target continuation tokens could be appended into the context cache.
- The compactor could then train on a contaminated cache rather than the clean
  context-only cache.

The fix:

- `_fresh_dynamic_cache` wraps the legacy cache before teacher continuation.
- The compactor receives the original context-only full cache.

Docs now treat pre-fix results as less reliable.

Files:

- `src/neural_kv/hf_training.py`
- `docs/model_architecture_summary.md`
- `docs/experiment_results_report.md`

Related commit:

- `9c0812f Isolate read-only cache continuations`

## Data And Benchmark Changes

### 1. The project no longer only mirrors Baseten's reported benchmark setup

Original design summarizes Baseten results:

- Qwen3-4B
- 8192-token contexts
- 1024 latents for 8x compression
- latent scaling from 128 to 8192 latents
- domain-specific results for Financial, Legal, Code, Gutenberg
- cross-domain transfer matrix

Current repo implements a practical local benchmark path centered on SEC
company-facts MCQs.

The strongest documented local run uses:

```text
Model:       Qwen/Qwen3-4B
Context:     8192 tokens
Latents:     1024
Compression: 8.0x
Dataset:     SEC company-facts random-visible MCQs
Train rows:  6000
Val rows:    600
Test rows:   600
Runtime:     AMD ROCm
```

This is a narrower benchmark scope than the original Baseten multi-domain
story, even though the local result is numerically stronger on its own SEC task.

### 2. Public smoke data builders were added

The repo has deterministic MCQ builders for:

- Gutenberg/public text smoke tests
- Hugging Face dataset based smoke paths
- SQuAD exact-answer MCQs
- SEC company-facts long-context MCQs

The original document discusses extractive MCQs conceptually. The current repo
turns that into scripts with storage checks and deterministic generation.

Files:

- `scripts/prepare_data.py`
- `scripts/prepare_sec_facts_mcq.py`
- `scripts/prepare_squad_mcq.py`
- `src/neural_kv/data.py`

### 3. ROCm and storage constraints became first-class

The original design does not cover runtime operations.

Current repo targets AMD GPUs with ROCm and uv-managed dependencies. It includes:

- ROCm Torch pinning through uv
- `scripts/check_rocm_env.py`
- `scripts/rocm_docker_run.sh`
- project-local Hugging Face, uv, TorchInductor, and COMGR caches
- a default `10TB` storage guardrail
- storage accounting across `data`, `checkpoints`, `artifacts`, caches, and envs

Files:

- `pyproject.toml`
- `scripts/check_rocm_env.py`
- `scripts/rocm_docker_run.sh`
- `src/neural_kv/storage.py`
- `README.md`

Related commits:

- `f134c1f Switch training environment to ROCm Torch`
- `2329c58 Add ROCm runner and bounded long-context data builders`

## Evaluation Changes

### 1. The repo now has direct evaluation scripts

Original design describes MCQ accuracy, KL, CE, and utilization metrics.

Current repo implements:

- checkpoint MCQ evaluation
- no-context, full-cache, and compact-cache comparison
- generation and letter-scoring modes
- CE utilization on teacher-response traces
- performance target checking from summaries

Files:

- `scripts/evaluate_checkpoint.py`
- `scripts/evaluate_ce_utilization.py`
- `scripts/check_performance_target.py`
- `src/neural_kv/hf_training.py`

Related commits:

- `14c2ca2 Add STILL target performance gate`
- `bf1321d Add CE utilization evaluation`
- `3525830 Add calibrated letter scoring mode`

### 2. Local SEC 6k result replaced "target not yet proven" for one benchmark

The original design reports Baseten's approximate 8x target:

```text
1024 latents / 8192 context
~85% MCQ accuracy
~0.93 CE utilization
```

Earlier repo docs said these numbers still needed a real GPU training run. Later
reports document that the SEC 6k run met or exceeded those local targets.

From `reports/performance_2026-06-08_6k_rocm.md`:

```text
Best checkpoint: step_800.pt
Validation compact accuracy: 0.966667
Validation full accuracy:    0.996667
Validation no-context acc:   0.318333
Test compact accuracy:       0.958333
Test full accuracy:          1.000000
Test no-context acc:         0.330000
Compression:                 8.0x
Validation CE utilization:   0.938850
Test CE utilization:         0.930875
```

This is a meaningful implementation milestone, but it should be stated as:

> The current repo has a strong SEC company-facts result at 8x compression.

It should not be overstated as:

> The current repo reproduced every Baseten scaling, domain, and transfer
> result.

### 3. Some docs are now stale

Two examples:

- `README.md` still says reproducing the target numbers needs a real GPU
  training run, but later in the same README and in the performance report it
  documents a completed SEC 6k checkpoint.
- `docs/experiment_results_report.md` says full validation/test and CE
  utilization are still needed, but `reports/performance_2026-06-08_6k_rocm.md`
  and `docs/neural_kv_compressor_scientific_report.md` provide those results.

This new document should be treated as the current design-delta summary. Older
reports remain useful as historical snapshots.

## Git Timeline

The important committed changes on `main` are:

| Commit | Change |
| --- | --- |
| `ecbda4e` | Added `research.md`, expanding from Baseten-specific explanation into a broader KV compression landscape. |
| `7785eaa` | Implemented the first runnable STILL neural KV compactor scaffold: package, scripts, tests, config, README, uv project. |
| `14c2ca2` | Added target performance gate. |
| `f134c1f` | Switched training environment to ROCm Torch. |
| `2329c58` | Added ROCm runner, model download, SEC/SQuAD long-context data builders. |
| `bc2e32f` | Aligned Qwen chat evaluation and RoPE compaction. |
| `899b0ba` | Added batched MCQ letter distillation. |
| `0faa818` | Sharpened identity initialization. |
| `f9e6c53` | Added teacher-response trace distillation. |
| `96f10f9` | Separated training and eval thinking flags. |
| `2be356b` | Added checkpoint resume path. |
| `bd28574` | Added zero beta option. |
| `fe6b7b7` | Added auxiliary direct-letter training loss. |
| `52eb553` | Added balanced answer sampling. |
| `2d7cfa7` | Added grouped layer compactors. |
| `3525830` | Added calibrated letter scoring. |
| `90747aa` | Added scoped compactor fine-tuning. |
| `88d25bf` | Added exact sink tokens. |
| `5b0835b` | Added random-visible SEC packing mode. |
| `cd7ae4f` | Added optional exact-anchor cache tokens. |
| `ecdc3a1` | Added lexical query-guided exact anchors. |
| `bf1321d` | Added CE utilization evaluation. |
| `7666e29` | Added bidirectional KL distillation. |
| `05c0b66` | Added head-specific latent queries. |
| `5d00048` | Added latent-only training scopes. |
| `9c0812f` | Isolated read-only cache continuations, fixing cache contamination. |
| `cd659cd` | Added 6k ROCm performance report. |
| `78e9478` | Documented 6k ROCm data and training workflow. |
| `57d3a8f` | Documented Hugging Face checkpoint validation workflow. |

There are also side branches after `main` for Qwen3-32B and Qwen3-235B NIAH
experiments. Those branches show the project is exploring longer-context
retrieval-style evaluation, but they are not part of the current `main` branch
implementation summarized here.

## What Is Still Not Implemented From The Original Roadmap

The current repo does not implement iterative compaction for true unbounded
context. The original design framed iterative compaction as the step where
"infinite context" starts to matter, but also described it as future work. That
is still future work here.

The current repo does not include a complete automated reproduction of the
Baseten result suite:

- latent-count scaling curve
- fixed 8x scaling across multiple context lengths
- Financial/Legal/Code/Gutenberg domain table
- cross-domain transfer matrix
- long multi-pass drift evaluation

The current tests are mostly unit-level. They cover RoPE, attention bias,
compactor shape/init behavior, exact-token paths, data generation, storage, and
loss functions. They do not verify an end-to-end Qwen3-4B quality target in CI.

## Current Design, In One Sentence

The project changed from a Baseten STILL design summary into a ROCm-oriented,
Hugging Face compatible training and evaluation system for learned KV cache
compaction, with pure learned-latent results on SEC company-facts plus several
engineering extensions that go beyond the original pure STILL design.

## Practical Reading Guide

Read these files in this order:

1. `docs/original.md` for the original baseline and Baseten result framing.
2. `docs/changes_from_original_design.md` for the delta from that baseline.
3. `src/neural_kv/compactor.py` for the current architecture.
4. `src/neural_kv/hf_training.py` for teacher/student training and evaluation
   behavior.
5. `docs/model_and_training.md` for operational training commands.
6. `reports/performance_2026-06-08_6k_rocm.md` for the strongest completed SEC
   6k result.

