# Neural KV Compressor: Methodology, Training, and Empirical Results

Date: 2026-06-13

## Abstract

This document describes the Neural KV Compressor implemented in this repository as a scientific-style report. The system trains a learned cache compactor for a frozen decoder-only large language model. Instead of fine-tuning the base model, the method learns a per-layer neural module that maps a full long-context key-value (KV) cache into a shorter synthetic KV cache that can be consumed by the same Hugging Face transformer during decoding.

The main experimental configuration uses `Qwen/Qwen3-4B`, an 8192-token context, and 1024 compact latent KV slots per layer, giving an 8.0x reduction in cache sequence length. The compactor is trained by teacher-student distillation: the teacher is the frozen model using the original full KV cache, and the student is the same frozen model using the compact cache. On the completed SEC company-facts multiple-choice benchmark, the best validation-selected checkpoint achieved 0.966667 validation accuracy and 0.958333 test accuracy at 8.0x compression, with CE utilization above 0.93 on held-out teacher-response traces.

## 1. Problem Statement

Transformer inference over long contexts is constrained by the KV cache. During prefill, each layer stores key and value tensors for all context tokens. During decoding, every new token attends over this cache. For a decoder-only model, a layer cache has the conceptual shape:

```text
K_l, V_l: [batch, kv_heads, source_tokens, head_dim]
```

When `source_tokens` is large, the cache can become a dominant memory cost. This repository targets the following question:

> Can a learned neural module compress the full KV cache of a frozen LLM into a smaller synthetic KV cache while preserving the behavior of the original full-cache model?

The proposed answer is a STILL-style neural compactor. It does not select a subset of original tokens as the primary mechanism. Instead, it synthesizes new compact keys and values through latent cross-attention over the full cache. These synthetic cache slots are then passed back into the original transformer as `past_key_values`.

The target configuration is:

```text
Base model:       Qwen/Qwen3-4B
Context length:   8192 tokens
Compact length:   1024 latent KV slots
Compression:      8192 / 1024 = 8.0x
Base LLM update:  none
Trainable module: neural KV compactor only
```

## 2. Core Hypothesis

The method is based on three assumptions.

First, not every token-level KV entry is equally necessary for a downstream query. A full long-context cache contains redundancy across positions, layers, and heads. A smaller set of learned latent slots may preserve enough information for the decoder to produce the same answer distribution as the full-cache model.

Second, a compressed cache must be represented in the native interface of the target model. A generic text summary or embedding summary is insufficient because the decoder expects per-layer key and value tensors with the correct head structure, head dimension, dtype, positional geometry, and attention-mask behavior. Therefore, the compactor directly produces synthetic per-layer K and V tensors.

Third, matching the final task label is too weak as the only learning signal. The compactor should preserve the teacher model's behavior over answer tokens. The training objective therefore uses full-cache teacher-student distillation, including token-level KL divergence, reverse KL, cross-entropy to the target response, and an auxiliary letter-level loss for multiple-choice tasks.

## 3. Methodology

### 3.1 System Overview

For each training or evaluation example, the system executes four conceptual stages:

1. The frozen base model reads the long context and produces a full context-only KV cache.
2. The compactor transforms that full KV cache into a compact cache with fewer positions.
3. The frozen base model decodes the question and answer continuation using either the full cache or the compact cache.
4. The compact-cache student is optimized to match the full-cache teacher.

The teacher and student are not separate language models. They are the same frozen `Qwen/Qwen3-4B` model under two cache conditions:

```text
Teacher path: context -> full KV cache -> continuation logits
Student path: context -> full KV cache -> compactor -> compact KV cache -> continuation logits
```

Only the compactor parameters are updated.

### 3.2 Full Cache Representation

Let the frozen model have `L` transformer layers. For layer `l`, the context prefill produces:

```text
K_l, V_l in R^[B x H_kv x T x D]
```

where:

```text
B    = batch size
H_kv = number of key-value heads
T    = source context length, 8192 in the main experiment
D    = per-head dimension
```

The compactor replaces the `T` dimension with `M` latent slots:

```text
Kc_l, Vc_l in R^[B x H_kv x M x D]
beta_l     in R^[B x H_kv x M]
```

For the main pure configuration:

```text
T = 8192
M = 1024
T / M = 8.0
```

The beta tensor is a learned additive attention bias that is injected into the target model's attention mask during student decoding.

### 3.3 Per-Layer Neural Compactor

The primary architecture is implemented by `StillCompactor` and `StillLayerCompactor`. Each layer is compacted independently unless a grouped layer-sharing option is explicitly enabled. In the completed pure run, the conceptually important path is one learned compactor per transformer layer.

For a single layer, the compaction pipeline is:

1. Receive full-cache keys and values:

   ```text
   K_l, V_l: [B, H_kv, T, D]
   ```

2. Undo rotary positional embedding on the keys:

   ```text
   K_unrotated = inverse_rope(K_l, token_positions)
   ```

3. Concatenate the unrotated key and value vectors:

   ```text
   X = concat(K_unrotated, V_l)
   X: [B, H_kv, T, 2D]
   ```

4. Flatten batch and head dimensions for compaction:

   ```text
   X: [B * H_kv, T, 2D]
   ```

5. Initialize `M` learned latent query slots:

   ```text
   Z_0: [B * H_kv, M, 2D]
   ```

6. Apply one or more perceiver-style blocks. Each block cross-attends from latent slots into the full KV-derived sequence, then applies latent self-attention:

   ```text
   Z'  = Z + CrossAttention(RMSNorm(Z), X)
   Z'' = Z' + SelfAttention(RMSNorm(Z'))
   ```

7. Project the final latent states into compact keys, values, and beta scores:

   ```text
   Kc = W_k Z
   Vc = W_v Z
   beta = W_beta Z
   ```

8. Re-apply RoPE to compact keys at evenly spaced virtual latent positions:

   ```text
   Kc = rope(Kc, latent_positions)
   ```

9. Reshape back into the Hugging Face cache layout:

   ```text
   Kc_l, Vc_l: [B, H_kv, M, D]
   beta_l:     [B, H_kv, M]
   ```

The use of inverse RoPE followed by re-application of RoPE is important. The compactor first reads keys in a position-normalized space so that latent cross-attention can mix information across positions, then restores positional structure for the compact cache that the base model will consume.

### 3.4 Latent Cross-Attention

Each latent slot acts as a learned query into the source cache. The cross-attention module uses:

```text
Q = rope(W_q Z, latent_positions)
K = rope(W_k X, token_positions)
V = W_v X
Attention(Q, K, V) = softmax(Q K^T / sqrt(2D)) V
```

The latent positions are evenly distributed over the original source length. This makes each compact slot behave as a virtual memory position in the compressed context. The decoder later attends over these positions as if they were the past sequence.

### 3.5 Latent Self-Attention

After cross-attention, the compact slots communicate through self-attention. This stage allows the latents to coordinate instead of behaving as independent pooling queries. In practice, this is useful because one slot may specialize in a local fact, another may capture a unit or fiscal period, and another may carry disambiguating document structure. Self-attention lets these summaries be mutually adjusted before projection into K and V.

### 3.6 Identity-Oriented Initialization

The compactor is initialized to be close to a stable cache-copying behavior rather than a random destructive transformation. The implementation uses:

```text
latent_dim = 2 * head_dim
key_head initially reads the key half of the latent state
value_head initially reads the value half of the latent state
the first cross-attention output path starts active
self-attention output projection starts at zero
beta_head starts at zero
```

For the article-style pure run, `beta-base zero` is used. This avoids adding an initial log-compression bias to attention and lets the model learn beta from the distillation objective.

This initialization is not intended to make the untrained compactor good. Its purpose is to avoid unstable early training dynamics by making the first updates operate near a recognizable cache geometry.

### 3.7 Beta Attention Bias

In addition to compact keys and values, the compactor emits a per-layer beta vector:

```text
beta_l: [B, H_kv, M]
```

During student decoding, the repository patches Hugging Face attention modules so beta can be merged into the additive attention mask. Since a transformer may have more query heads than KV heads, beta is repeated across query-head groups:

```text
beta_l -> [B, attention_heads, query_length, M]
```

The attention logits for compact slots therefore become:

```text
logits = Q Kc^T / sqrt(D) + attention_mask + beta
```

This gives the compactor a direct way to up-weight or down-weight compact slots without changing the frozen model's attention kernels or model weights.

### 3.8 Pure and Hybrid Cache Modes

The repository supports both pure learned compaction and hybrid exact-token variants.

The pure setup uses only learned latent slots:

```text
sink_tokens = 0
exact_tokens = 0
num_latents = 1024
```

This is the main methodology for article-faithful claims. All main 6k results reported below use the pure setup.

The hybrid setup can prepend exact source KV entries before the learned latent cache. Supported exact-token strategies include prefix, evenly spaced positions, KV norm, and lexical query-guided selection. The lexical strategy produced very strong local results, but it is not considered a pure compaction result because the exact anchors can depend on the query. It is best interpreted as an upper-bound or engineering control.

### 3.9 Cache Compatibility

The compact cache is wrapped in a `CompactKVCache` object and converted into a Transformers `DynamicCache`. The frozen model then receives the compact cache through the standard `past_key_values` interface. This design is deliberate: the base model does not need to know whether the cache positions correspond to real tokens or synthetic latent memory slots.

The compatibility condition is:

```text
same number of layers
same number of KV heads
same head dimension
shorter sequence dimension
valid positional IDs and attention mask
```

Because the compactor preserves these structural requirements, the model can continue decoding normally.

## 4. Training Procedure

### 4.1 Data Construction

The completed main run uses SEC company-facts data converted into extractive multiple-choice questions. Each row contains:

```text
context
question
choices
answer_index
answer
```

The 6k benchmark split is:

```text
Train:       6000 rows
Validation:   600 rows
Test:         600 rows
```

The context builder uses long SEC-derived text and places the target evidence at random visible positions. The main context length for the model is capped at 8192 tokens after formatting.

### 4.2 Teacher Trace Generation

For token-level distillation, the training rows are augmented with teacher responses generated by the full-cache model. These traces provide target continuation token IDs:

```text
train_traces_6000_512.jsonl
validation_traces_600_512.jsonl
test_traces_600_512.jsonl
```

The completed 6k run recorded:

```text
Train teacher traces:      6000 rows, 5973 tail-gold hits
Validation teacher traces:  600 rows,  599 tail-gold hits
Test teacher traces:        600 rows,  596 tail-gold hits
```

The trace generation path uses the frozen `Qwen/Qwen3-4B` model with 8192-token context and up to 512 generated tokens. These traces are especially important for CE-utilization evaluation because they define the held-out teacher-response continuations.

### 4.3 Encoding

For each MCQ example, the input is separated into:

```text
context_ids: [1, context_length]
prompt_ids:  question and choices
target_ids:  answer letter, answer text, or teacher response tokens
```

The main training path uses:

```text
target_mode = teacher_response
loss_mode   = token
```

This means the compactor is trained to preserve the full teacher response distribution over the generated target sequence, not merely to predict a single MCQ label.

### 4.4 Teacher and Student Forward Passes

For each row, the training forward pass performs:

1. Full context prefill:

   ```text
   full_outputs = model(context_ids, use_cache=True)
   full_cache = full_outputs.past_key_values
   ```

2. Teacher continuation with a fresh full-cache wrapper:

   ```text
   teacher_logits = model(prompt + target_prefix, past_key_values=fresh(full_cache))
   ```

3. Cache compaction:

   ```text
   compact_cache = compactor(full_cache)
   ```

4. Student continuation using compact cache and beta biases:

   ```text
   student_logits = model(
       prompt + target_prefix,
       past_key_values=compact_cache,
       attention_bias=beta
   )
   ```

5. Loss computation on answer or teacher-response token positions.

A critical implementation detail is the fresh cache wrapper. Hugging Face `DynamicCache.update()` can mutate the cache object during continuation forwards. If the original context cache were reused directly, teacher continuation tokens could contaminate the cache that is later given to the compactor. The corrected training path creates a fresh dynamic wrapper before continuation, preserving a clean context-only cache for compaction.

### 4.5 Objective Function

The primary token-level training loss is:

```text
L = w_kl  * KL(p_teacher || p_student)
  + w_rkl * KL(p_student || p_teacher)
  + w_ce  * CE(target_ids, p_student)
```

where:

```text
p_teacher = softmax(teacher_logits)
p_student = softmax(student_logits)
```

The completed main run used:

```text
w_kl  = 1.0
w_rkl = 0.5
w_ce  = 0.1
```

An auxiliary letter-level loss is also used for MCQ calibration:

```text
L_total = L_token + 0.05 * L_letter
```

The auxiliary loss distills the full-cache model's distribution over the four answer labels `A`, `B`, `C`, and `D`. This gives the training process a direct signal on the benchmark decision rule while still using token-level teacher-response distillation as the primary objective.

### 4.6 Optimization

The completed 6k ROCm run used the following settings:

```text
Model:             Qwen/Qwen3-4B
Device path:       AMD ROCm through Torch cuda/HIP API
Context length:    8192
Latents:           1024
Compression:       8.0x
Exact anchors:     0
Batch size:        2
Learning rate:     5e-6
Optimizer:         AdamW
Gradient clipping: 1.0
Continuation steps: 1200
Global steps:      1500
Training time:     4811.43 seconds
Precision:         bfloat16 model execution
```

The 6k run continued from a corrected 3k pure checkpoint at global step 300:

```text
init checkpoint:
checkpoints/qwen3_4b_sec_random_visible_3k_cachefix_pure1024_identity_400step_b2_lr1e5_w05/step_300.pt

output directory:
checkpoints/qwen3_4b_sec_random_visible_6k_cachefix_pure1024_identity_resume1200_b2_lr5e6_w05
```

Balanced answer sampling was enabled so that training batches sample answer letters more uniformly.

### 4.7 Evaluation Protocol

Three evaluation paths are used.

First, MCQ accuracy is measured for the compact-cache model. The prediction is scored by answer letter, using the same frozen base model and the learned compact cache.

Second, the full-cache baseline is evaluated with the same prompt and scoring method. This measures the ceiling imposed by the base model and dataset formatting.

Third, a no-context baseline is evaluated. This removes the context and tests whether the benchmark can be solved from priors or prompt artifacts. In the main result, no-context accuracy is near 0.32 to 0.33, while compact-cache accuracy is near 0.96, confirming that the compact cache carries task-relevant context.

The main MCQ evaluation uses no-thinking letter scoring to match the sampled validation path used during training.

CE utilization is computed on held-out teacher-response traces. It measures how much of the cross-entropy gap between no-context and full-context decoding is recovered by the compact cache:

```text
CE utilization = (CE_no_context - CE_compact) / (CE_no_context - CE_full)
```

A value near 1.0 indicates that compact-cache decoding approaches the full-cache teacher in continuation likelihood. A value near 0.0 indicates little improvement over no-context decoding.

## 5. Results

### 5.1 Main Completed 6k ROCm Run

The best sampled-validation checkpoint is:

```text
checkpoints/qwen3_4b_sec_random_visible_6k_cachefix_pure1024_identity_resume1200_b2_lr5e6_w05/step_800.pt
```

This is the recommended validation-selected checkpoint. The final checkpoint is also strong, but `step_800.pt` is preferred because it has the highest recorded full validation accuracy among the evaluated checkpoints.

### 5.2 Sampled Validation During Training

Sampled validation used 128 validation examples. Full-cache accuracy stayed at 1.0 and no-context accuracy stayed at 0.335938 across the sampled evaluations. Compact-cache accuracy improved rapidly after continuation on the 6k trace set.

| Global step | Compact accuracy | Full accuracy | No-context accuracy | KL | CE | Reverse KL | Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 0.953125 | 1.000000 | 0.335938 | 0.086548 | 0.117078 | 0.205291 | 0.204255 |
| 500 | 0.921875 | 1.000000 | 0.335938 | 0.058867 | 0.095768 | 0.092088 | 0.114489 |
| 600 | 0.976563 | 1.000000 | 0.335938 | 0.066150 | 0.107203 | 0.106256 | 0.130207 |
| 700 | 0.929688 | 1.000000 | 0.335938 | 0.069947 | 0.117658 | 0.180029 | 0.171743 |
| 800 | 0.984375 | 1.000000 | 0.335938 | 0.081762 | 0.105518 | 0.310424 | 0.319094 |
| 900 | 0.976563 | 1.000000 | 0.335938 | 0.050755 | 0.097214 | 0.073245 | 0.099117 |
| 1000 | 0.968750 | 1.000000 | 0.335938 | 0.047853 | 0.079493 | 0.067183 | 0.089405 |
| 1100 | 0.968750 | 1.000000 | 0.335938 | 0.041995 | 0.066146 | 0.108552 | 0.102886 |
| 1200 | 0.953125 | 1.000000 | 0.335938 | 0.027820 | 0.058106 | 0.076118 | 0.071733 |
| 1300 | 0.960938 | 1.000000 | 0.335938 | 0.037835 | 0.061608 | 0.159722 | 0.123862 |
| 1400 | 0.960938 | 1.000000 | 0.335938 | 0.037172 | 0.073038 | 0.090111 | 0.089558 |
| 1500 | 0.960938 | 1.000000 | 0.335938 | 0.038485 | 0.064262 | 0.083023 | 0.086455 |

The best sampled compact accuracy was 0.984375 at global step 800.

### 5.3 Full MCQ Evaluation

The full MCQ evaluation used 600 validation rows and 600 test rows.

| Checkpoint | Split | Rows | Compact accuracy | Full accuracy | No-context accuracy | Compression |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `step_800.pt` | Validation | 600 | 0.966667 | 0.996667 | 0.318333 | 8.0 |
| `step_800.pt` | Test | 600 | 0.958333 | 1.000000 | 0.330000 | 8.0 |
| `final.pt` | Validation | 600 | 0.965000 | 0.996667 | 0.318333 | 8.0 |
| `final.pt` | Test | 600 | 0.965000 | 1.000000 | 0.330000 | 8.0 |

These results indicate that the pure learned latent cache preserves most of the full-context model's task performance while reducing the cache length by a factor of 8. The compact model is far above the no-context baseline, which shows that the answer signal is not primarily coming from world knowledge or answer-choice priors.

### 5.4 CE Utilization

CE utilization was measured on a fixed 200-row prefix sample per split using thinking-enabled teacher-response traces. This matches the formatting used when generating the teacher traces.

| Checkpoint | Split | Rows | Target tokens | Full CE | Compact CE | No-context CE | Compact KL | CE utilization | Compression |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `step_800.pt` | Validation traces | 200 | 47,314 | 0.024683 | 0.094268 | 1.162619 | 0.064519 | 0.938850 | 8.0 |
| `step_800.pt` | Test traces | 200 | 47,352 | 0.025552 | 0.106161 | 1.191687 | 0.077178 | 0.930875 | 8.0 |

The compact cache recovers more than 93 percent of the cross-entropy improvement that the full context provides over the no-context baseline. This is a stronger result than MCQ accuracy alone because it evaluates the likelihood of full teacher-response continuations, not only the final answer letter.

### 5.5 Earlier Pure Result After Cache Fix

Before the larger 6k continuation, the corrected 3k pure run reached:

```text
Checkpoint directory:
checkpoints/qwen3_4b_sec_random_visible_3k_cachefix_pure1024_identity_400step_b2_lr1e5_w05

Step 100 compact accuracy on eval-limit 128: 0.7109375
Step 200 compact accuracy on eval-limit 128: 0.8828125
Full accuracy on same slice:                 1.0
No-context accuracy on same slice:           0.3359375
Compression:                                  8.0x
```

This result was important because it showed that the corrected cache-isolated training path could exceed the rough 0.85 MCQ target even before the larger 6k continuation run.

### 5.6 Hybrid Lexical Anchor Control

A hybrid lexical-anchor run achieved near-perfect validation and test performance:

| Split | Compact accuracy | Full accuracy | No-context accuracy | Compression |
| --- | ---: | ---: | ---: | ---: |
| Validation 400 | 0.9975 | 0.9975 | 0.315 | 8.0 |
| Test 400 | 1.0000 | 1.0000 | 0.330 | 8.0 |

This result should not be used as the main pure compaction claim because lexical exact anchors are query-guided. It is nevertheless useful as a control showing that retaining selected exact evidence can make the benchmark almost lossless under the same nominal compression ratio.

## 6. Interpretation

The main result supports the central hypothesis: a frozen decoder-only LLM can consume a learned synthetic KV cache that is substantially shorter than the original context cache while retaining most full-context performance.

Several observations are important.

First, the model is not merely exploiting dataset artifacts. The no-context baseline is close to random multiple-choice behavior, while the compact-cache student approaches the full-cache teacher. This gap indicates that the compact cache carries relevant evidence from the long context.

Second, the pure compactor performs well without exact anchors. The main `step_800.pt` checkpoint uses 1024 learned latent slots and zero exact tokens. Therefore, its performance is attributable to neural cache synthesis rather than retrieval of raw source positions.

Third, CE utilization confirms behavioral preservation beyond final answer accuracy. The compact cache does not simply learn a shortcut to the correct letter; it also recovers most of the teacher-response likelihood gap between no-context and full-context decoding.

Fourth, cache isolation was essential. Earlier runs were affected by cache mutation during teacher continuation. The corrected method ensures that the compactor receives a context-only cache, making the post-fix results substantially more reliable.

## 7. Limitations

The results are strong on the current SEC multiple-choice benchmark, but they do not yet prove universal long-context compression.

The benchmark is extractive and structured around SEC company facts. Additional validation is needed on broader long-context tasks such as multi-hop QA, multi-turn shared-context evaluation, summarization, code retrieval, and adversarial needle-in-a-haystack tests.

The current result measures cache sequence-length compression. End-to-end latency and throughput improvements depend on implementation details such as compactor runtime, attention kernel behavior, batch size, and whether compact caches are reused across queries.

The compactor is trained for a specific base model and cache geometry. Porting to another model family may require retraining and validation of RoPE handling, head grouping, attention-mask patching, and tokenizer formatting.

Finally, CE utilization was measured on 200-row prefix samples per split. This is useful evidence but should be expanded to full held-out trace sets when reporting final benchmark claims.

## 8. Conclusion

This repository implements a learned neural KV cache compressor for a frozen Hugging Face decoder-only language model. The method converts each layer's full KV cache into a smaller synthetic cache through RoPE-aware perceiver-style latent cross-attention, latent self-attention, and learned projections into compact keys, values, and attention biases. Training uses full-cache teacher-student distillation, with token-level KL, reverse KL, cross-entropy, and an auxiliary MCQ letter objective.

The completed 6k SEC experiment demonstrates that the pure learned-latent compactor can achieve 8.0x cache compression while maintaining high task performance. The validation-selected `step_800.pt` checkpoint reached 0.966667 validation accuracy, 0.958333 test accuracy, and CE utilization above 0.93. These results indicate that neural KV cache compaction can preserve much of the behavior of full-context inference while substantially reducing the number of cached positions used during decoding.

