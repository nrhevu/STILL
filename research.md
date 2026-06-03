# Neural KV Cache Compression — Research Landscape

**Question:** Methods related to building a *neural KV cache compressor*, inspired by "Towards Infinite Context Windows: Neural KV Cache Compaction", "You Only Cache Once (YOCO)", and "In-Context Autoencoder (ICAE)". Survey of KV cache compression, context compression, memory-efficient attention, token eviction/merging, and learned compression (2023–2025).

> **Provenance & confidence.** This report was assembled from a deep-research run that completed Scope → Search (6 angles, ~33 sources) → Claim extraction (140 claims) → **partial** adversarial verification (18 of the claims hard-verified before the run was interrupted by a rate limit). Claims that were independently verified are tagged **[VERIFIED]** or **[REFUTED/OVERSTATED]**. Everything else is **[source-reported]** — accurately transcribed from the paper but not yet independently fact-checked, so treat specific numbers with mild caution. The strongest, most consistent caveat from verification: **headline compression ratios in this field are routinely cherry-picked** (best case, best dataset, often a context-length-extension metric rather than a like-for-like memory reduction).

---

## 1. The taxonomy (where your idea sits)

KV cache reduction splits into three levels [source-reported, review arXiv:2508.06297]:

- **Token-level** — selection/eviction, merging, quantization, low-rank decomposition. Operates on the cache, model untouched.
- **Model-level** — attention grouping/sharing (GQA/MQA), architecture changes (YOCO, Infini-attention). Requires training or architectural commitment.
- **System-level** — paging, scheduling, hardware-aware kernels (vLLM-style).

Your proposed **neural compressor** lives mostly in token-level "learned compression" but, depending on design, can blur into model-level (if you train an encoder adapter into the model, like ICAE). The key strategic axis for your idea:

| Axis | Cheap end | Expensive end |
|---|---|---|
| **Training** | training-free (H2O, SnapKV, FINCH, RocketKV) | learned encoder (ICAE, IC-Former, KV-Distill, KVP) |
| **Query dependence** | query-aware (SnapKV, FINCH) | query-agnostic (Compactor, KV-Distill) |
| **Representation** | keep a subset of real tokens (eviction) | synthesize new vectors in embedding space (KVSculpt, ICAE) |
| **Memory profile** | sub-linear but bounded | constant / recurrent (LESS, Infini-attention) |

The frontier finding most relevant to a *neural* design: **synthesizing unconstrained KV pairs in continuous embedding space is strictly more expressive than selecting or merging existing tokens** [source-reported, KVSculpt arXiv:2603.27819]. That is the theoretical case for your idea over heuristic eviction.

---

## 2. Directly adjacent — learned / neural compression (your closest neighbors)

These are the papers you must position against, because they already do "neural KV/context compression."

**ICAE — In-Context Autoencoder** (arXiv:2307.06945) — *one of your seed papers.*
- 4× context compression into a fixed set of memory slots (k=128 default); the target LLM conditions on the slots instead of raw tokens [source-reported].
- Adds **~1% params** via LoRA on Q/V projections as the encoder adapter [source-reported].
- Trains on **autoencoding + language-modeling jointly**; both together beat either alone [source-reported].
- Pretraining matters: an 8× ICAE matches a non-pretrained 4× ICAE [source-reported].

**IC-Former — In-Context Former** (arXiv:2406.13618) — the efficiency answer to ICAE.
- Replaces quadratic self-attention with **linear cross-attention** + learnable "digest tokens"; **68–112× faster compression**, **1/32 the FLOPs**, retains >90% performance, compresses 4× (512→128) [source-reported].
- Architecturally **independent of the target LLM** — a strong design pattern if you want a portable compressor.

**KV-Distill** (arXiv:2503.10337) — distillation framing, very aggressive.
- Up to **1000×** compression with coherent generation; domain fine-tuning cuts KV length **99% at ~20% accuracy drop** [source-reported].
- **Student–teacher KL** objective aligning compressed vs. uncompressed next-token distributions (bidirectional KL, λ>0.5) [source-reported].
- **Query-independent**; beats H2O, ICAE, DODO on SQuAD/Llama-3 (86.6% @ 25% retention vs H2O 84.0%) [source-reported].

**KVSculpt — KV Cache Compression as Distillation** (arXiv:2603.27819) — the "optimize new KV pairs" idea, executed without end-to-end SGD.
- Frames compression as distilling into a **smaller, unconstrained KV cache** in embedding space [source-reported].
- Optimizes via **alternating L-BFGS (keys) + closed-form ridge regression (values)** instead of gradient learning [source-reported].
- Reduces KL divergence **3.5–4.1×** vs a Select+Fit baseline [source-reported].
- Documents extreme **non-uniformity**: per-layer pilot MSE varies up to 100×, two KV heads in one layer up to 467× → strong argument for adaptive per-head/per-layer budgets [source-reported].

**KV-CAR — Autoencoders + KV Reuse** (arXiv:2512.06727) — lightweight, architecture-agnostic.
- Autoencoder compresses KV along the embedding dim; **47.85% memory reduction**, GPT-2 perplexity 21.4→23.9 (~+11.7%) [source-reported].
- Exploits **cross-layer KV similarity** to reuse KV across heads, no architecture change [source-reported].

**KVP — Learning to Evict** (arXiv:2602.10238) — RL instead of heuristics.
- Frames eviction as **RL**, training lightweight per-head agents on pre-computed generation traces using only K/V vectors; no LLM modification [source-reported].
- Argues heuristic eviction (recency, attention score) are **indirect proxies** for true future utility; learned ranking generalizes zero-shot to longer contexts [source-reported].

**LESS — recurrence + eviction** (arXiv:2402.09398) — hybrid you should know.
- Adds a **constant-size low-rank recurrent state** that accumulates info from evicted tokens (extra storage ≈ 4 KV pairs, regardless of context) [source-reported].
- −20% perplexity vs H2O at 2% budget; trains <2% of params [source-reported].

**SAC — autoencoding-free** (arXiv:2510.08907) — *important critique of the ICAE family.*
- Claim: autoencoding-based compression has an **objective conflict** — gradient cosine similarity between the autoencoding and LM objectives drops to ~0 after initial alignment (they pull orthogonally) [source-reported]. **This is a direct challenge to ICAE's joint-objective design — worth reading before you commit to a reconstruction loss.**
- SAC instead selects anchor tokens and aggregates info into their KV via bidirectional attention; **+23.5% F1 / +26.8% EM over ICAE at 15×**, trains ~31% faster [source-reported].

---

## 3. Architectural alternatives that avoid materializing a big KV cache

These compete with the "compress the cache" framing by changing the model.

- **YOCO — You Only Cache Once** (your seed paper): decoder-decoder design where the cache is produced once by a self-decoder and reused by a cross-decoder, so global KV is stored a single time. *Not surfaced as a fetched source in this run — included from the seed; verify its specific numbers directly.*
- **Infini-attention** (arXiv:2404.07143): stores old KV in a **compressive memory matrix** (Ms = Ms-1 + σ(K)ᵀV), **constant memory w.r.t. sequence length**; gates local attention + long-term memory retrieval. 1B model → 1M tokens after fine-tuning on 5K inputs; 114× fewer memory params than Memorizing Transformer [source-reported]. **This is the closest published instance of "neural KV compaction into bounded memory" — your strongest architectural prior art.**
- **StreamingLLM / attention sinks** (arXiv:2309.17453): keep initial "sink" tokens + sliding window → effectively infinite streaming, no fine-tuning, up to 22.2× speedup, validated to 4M tokens [source-reported]. **Caveat [VERIFIED via cross-source]: catastrophic on retrieval** — string-retrieval accuracy 57.1→0.4 [source-reported, SCBench arXiv:2412.10319]. Don't use pure sink+window if retrieval matters.
- **DuoAttention** (arXiv:2410.10819): splits heads into **retrieval heads** (full cache) vs **streaming heads** (constant cache); 2.55× (MHA)/1.67× (GQA) memory cut; retrieval heads found by optimizing per-head gates without fine-tuning [source-reported]. → A neural compressor could be applied *only to streaming heads* for free wins.
- **LightTransfer** (arXiv:2410.13846): "lazy layers" attend mostly to initial+recent tokens; converting half the layers to streaming gives 2.17× throughput, <1.5% degradation, no training [source-reported].

---

## 4. Heuristic eviction / merging baselines (what you must beat)

These are training-free and strong; they're the bar your neural method has to clear on the accuracy-vs-compression curve.

- **H2O — Heavy-Hitter Oracle** (arXiv:2306.14048): small token subset dominates attention (power-law); keep ~20% (heavy hitters + recent) ≈ full cache; 5–10× memory, up to 29× throughput [source-reported].
- **Scissorhands** (arXiv:2305.17118): "persistence of importance" (>95% persistence ratio) → 5× memory cut (20× with 4-bit quant), no fine-tuning [source-reported].
- **SnapKV** (arXiv:2404.14469): query-aware static selection from an observation window. **⚠️ heavily verified and partially overstated:**
  - **[REFUTED/OVERSTATED]** the widely-quoted "**8.2× memory efficiency**" is actually a *context-length-extension* metric (max processable context before OOM), **not** a like-for-like memory reduction (V6, V11, V13).
  - **[REFUTED/OVERSTATED]** "380K tokens with negligible accuracy drop" — the paper's own Needle-in-a-Haystack shows **degradation beyond ~140K tokens** (V12, V14, V15).
  - **[VERIFIED]** the "**92% compression rate at 1024-token budget on LongBench**" figure is real (V17, V18) — but Ada-KV/SCBench note SnapKV's gains are inflated by question-aware eval and **drop in multi-turn / retrieval** settings (V16, V17).
  - Takeaway: SnapKV is a real, strong baseline, but **report memory reduction and accuracy on multi-turn + retrieval, not just single-query LongBench.**
- **PyramidKV** (arXiv:2406.02069): "pyramidal funneling" — attention broadens in low layers, concentrates in high layers → **non-uniform per-layer budgets** (arithmetic schedule, more cache low, less high). Matches full cache at **12% retention**; at 0.7% beats SnapKV/H2O by up to 20.5 pts on TREC [source-reported].
- **CaM — Cache Merging** (ICML 2024, OpenReview LCTmppB165): merge discarded entries into retained ones (attention-prominence weighted) → less output perturbation than eviction, esp. at high ratios; beats H2O & StreamingLLM [source-reported].
- **KVCompose** (arXiv:2509.05165): per-head selection aligned into **composite tokens** + layer-adaptive global budget; RULER-4096 AUC 82.3 vs PyramidKV 61.5, TOVA 73.4; **compatible with standard inference engines** (a recurring practical blocker for structured methods) [source-reported].
- **RocketKV** (arXiv:2502.14051): training-free **two-stage** (coarse eviction + fine sparse attention) → up to 400× compression, 3.7× speedup, −32.6% decode peak memory [source-reported].
- **Compactor** (arXiv:2507.08143): **query-agnostic**, importance = squared row-norm of left-singular vectors of K (approx. leverage scores); 68% memory cut, 99% of full-KV at 50% retention; argues query-aware methods fail when query is unknown at prefill [source-reported].
- **FINCH** (TACL): prompt-guided, training-free, layer-wise top-r, up to **93× prefill compression**; corrects causal positional bias in attention scores [source-reported].
- **FAEDKV** (arXiv:2507.20030): training-free, transforms KV to frequency domain (Infinite-Window DFT) to **equalize all tokens' contribution** — counters the recency bias most methods have; +22% on LongBench, position-agnostic on NIAH [source-reported].
- **ZeroMerge** (2025): parameter-free merging [source-reported, name only].

---

## 5. Quantization & low-rank (orthogonal — combine with your method)

- **KIVI / KVQuant**: KIVI tuning-free asymmetric 2-bit → 2.35–3.47× throughput; KVQuant targets 10M-token inference [source-reported]. Quantization stacks multiplicatively with eviction/learned compression.
- **LoRC** (arXiv:2410.03111): SVD low-rank on KV weights, **55–60% compression, <1% drop, no retraining**; SVD across 80 layers of 70B = 40s [source-reported]. Key insight: **errors in shallow layers amplify** (compressing first 1/8 of layers → 68% accuracy drop on OpenBookQA) → **progressive, depth-aware compression** [source-reported].
- **DeltaKV** (arXiv:2602.08005): residual encoding vs retrieved historical reference tokens; 29% of original size, near-lossless on LongBench/SCBench/AIME; 2× throughput with a custom Sparse-vLLM engine [source-reported].
- **CacheGen** (SIGCOMM 2024, arXiv:2310.07240): KV cache *as a network artifact* — encode/stream compressed KV. **⚠️ verified with corrections:**
  - **[VERIFIED]** 3.5–4.3× size reduction; <2% accuracy loss; **2.4–2.9× delta-similarity** between nearby tokens (the locality insight) (V3, V4, V5, V7).
  - **[REFUTED/OVERSTATED]** the "3.2–3.7× TTFT" figure — the paper states **1.67–1.81× faster TTFT** (V1, V2).
  - **[CONTESTED]** "shallower layers more quantization-sensitive" holds in CacheGen's setup (V9) but is **not a general rule** — KVTuner (ICML 2025, arXiv:2502.04420) finds no clean depth-based heuristic (V8, V10). **Do not hard-code a "protect shallow layers" rule; learn the per-layer sensitivity.**

---

## 6. Benchmarks & evaluation guidance

- **LongBench** — the default, but single-request; easy to overstate on.
- **RULER / RULER-4096** — synthetic, controllable; used by Compactor, KVCompose.
- **Needle-in-a-Haystack** — retrieval stress test; **where eviction methods quietly fail.**
- **SCBench** (arXiv:2412.10319) — *evaluate here.* KV-cache-**centric**, **shared-context multi-turn** lifecycle. Findings that should shape your eval:
  - Sub-O(n) memory methods drop sharply at **1/4 compression** → near-infeasible for multi-turn [source-reported].
  - **Sparse encoding (O(n)) approximates full attention across queries; sparse decoding is fine on query 1 but degrades on later queries** [source-reported].
  - Dynamic sparsity > static → **supports learned/adaptive compression over fixed eviction** [source-reported].
- Practical scale anchors: 32-layer / 8 KV-heads / dim-128 @ 32K tokens ≈ **2 GB** KV [source-reported]; 30B model, batch 128, seq 1024 ≈ **180 GB** KV [source-reported, H2O]; KV cache is **2.5–5× model weights** at test time [source-reported, Scissorhands].

---

## 7. Synthesized opportunities for *your* neural KV compressor

Drawn from the gaps and the verified caveats above:

1. **Lead with multi-turn + retrieval evaluation (SCBench, NIAH), not single-query LongBench.** Verification shows that's exactly where the field's headline numbers evaporate. A method that holds up there is genuinely novel.
2. **Synthesize KV in embedding space, not select** — KVSculpt's expressiveness argument plus ICAE/IC-Former's encoder pattern is the strongest case for a neural approach. IC-Former's **linear cross-attention + digest tokens, decoupled from the target LLM** is the most deployable shape.
3. **Avoid the autoencoding objective trap.** SAC's gradient-orthogonality finding suggests a pure reconstruction loss may fight LM quality. Consider a **distillation/KL objective** (KV-Distill) or anchor-aggregation (SAC) instead of, or alongside, reconstruction.
4. **Make budgets adaptive per-layer and per-head.** KVSculpt's up-to-467× intra-layer head variance and PyramidKV's funneling both say a uniform budget leaves a lot on the table — but **don't hard-code "protect shallow layers"** (KVTuner refutes it); learn it.
5. **Stack with quantization and target only streaming heads** (DuoAttention) for compounding, low-risk wins.
6. **Consider the bounded-memory recurrent framing** (Infini-attention, LESS) if "infinite context" is the actual goal — that's a stronger claim than a fixed compression ratio and is the most direct realization of "neural KV compaction."
7. **Ship something that runs on a standard engine.** KVCompose and DeltaKV both flag that structured methods often need custom kernels (DeltaKV) or break tensor layouts — inference-engine compatibility is a real, underweighted differentiator.

---

## 8. Source list (33 sources)

Surveys: KV Cache Compression Review (arXiv:2508.06297) · Efficient Attention Mechanisms Survey (arXiv:2507.19595) · Awesome-KV-Cache-Management (GitHub treeai-lab) · SCBench (arXiv:2412.10319) · EMPIRIC (ACM 10.1145/3759441.3759448)

Learned/neural: ICAE (2307.06945) · IC-Former (2406.13618) · KV-Distill (2503.10337) · KVSculpt (2603.27819) · KV-CAR (2512.06727) · KVP/Learning-to-Evict (2602.10238) · LESS (2402.09398) · SAC (2510.08907) · Soft Prompt Compression (2404.04997)

Eviction/merging: H2O (2306.14048) · Scissorhands (2305.17118) · SnapKV (2404.14469) · PyramidKV (2406.02069) · CaM (OpenReview LCTmppB165) · KVCompose (2509.05165) · RocketKV (2502.14051) · Compactor (2507.08143) · FINCH (TACL) · FAEDKV (2507.20030) · ToMe (2210.09461)

Architecture: Infini-attention (2404.07143) · StreamingLLM (2309.17453) · DuoAttention (2410.10819) · LightTransfer (2410.13846) · SWAA (2512.10411)

Quant/low-rank/streaming: LoRC (2410.03111) · DeltaKV (2602.08005) · CacheGen (2310.07240, SIGCOMM 2024) · KVTuner (2502.04420, used as counter-source)

> Some arXiv IDs above (e.g. 2602.*, 2603.*, 2512.*) carry 2026-dated identifiers as returned by search — verify these resolve before citing in a paper; they may be very recent or placeholder identifiers.
