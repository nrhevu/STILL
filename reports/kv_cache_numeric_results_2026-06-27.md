# KV Cache Compression Numeric Results

Ngay thu thap: 2026-06-27 UTC.

File nay chi gom cac con so can trich dan/so sanh. Cot `Full/Base` la full-cache, original-context, FP16/BF16, hoac baseline duoc paper dung. Cot `Compressed/Method` la ket qua sau nen. `Delta` la `Compressed - Full/Base`, nen metric accuracy/F1/ROUGE cao-hon-la-tot thi delta am la giam chat luong; voi PPL/CE thap-hon-la-tot thi delta duong la xau hon.

## Learned / Neural KV And Context Compression

| Paper | Dataset / metric | Setting | Full/Base | Compressed/Method | Delta | Compression / speed / memory | Source |
|---|---:|---|---:|---:|---:|---|---|
| Still | MCQ accuracy | Qwen3-4B, 8192 tokens to 1024 latents | not tabled | ~85-86% | not tabled | 8x cache sequence compression; CE utilization ~0.93; KL ~0.15 | [Baseten](https://www.baseten.co/research/towards-infinite-context-windows-neural-kv-cache-compaction/), [arXiv:2606.07878](https://arxiv.org/abs/2606.07878) |
| Still | MCQ accuracy | Qwen3-4B, 8192 tokens to 128 latents | no-context ~20-22% | ~58-60% | +36 to +40 pp vs no-context | 64x cache sequence compression | [Baseten](https://www.baseten.co/research/towards-infinite-context-windows-neural-kv-cache-compaction/) |
| Still | Code MCQ accuracy | 8x compression | not tabled | 89.6% | not tabled | 1024 latents / 8192 tokens | [local summary](/scratch/vunguyen13/KVCacheProject/STILL/docs/still_original.md) |
| Still | Financial MCQ accuracy | 8x compression | not tabled | 86.0% | not tabled | 1024 latents / 8192 tokens | [local summary](/scratch/vunguyen13/KVCacheProject/STILL/docs/still_original.md) |
| Still | Gutenberg MCQ accuracy | 8x compression | not tabled | 79.2% | not tabled | 1024 latents / 8192 tokens | [local summary](/scratch/vunguyen13/KVCacheProject/STILL/docs/still_original.md) |
| Still | Legal MCQ accuracy | 8x compression | not tabled | 75.6% | not tabled | 1024 latents / 8192 tokens | [local summary](/scratch/vunguyen13/KVCacheProject/STILL/docs/still_original.md) |
| ICAE | Pile continuation PPL | Llama-7B, 512 to 128 slots | 9.01 | 9.50 | +0.49 PPL | 4x context compression; +~1% params | [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| ICAE | Pile continuation PPL | Llama-2-7B, 512 to 128 slots | 8.81 | 9.18 | +0.37 PPL | 4x context compression; +~1% params | [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| ICAE | Pile continuation PPL | Llama-2-13B, 512 to 128 slots | 8.15 | 8.45 | +0.30 PPL | 4x context compression; +~1% params | [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| ICAE | Autoencoding BLEU | Llama-7B, 512 to 128 slots | n/a | 99.1 | n/a | 4x context compression | [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| ICAE | Autoencoding loss | Llama-7B, 512 to 128 slots | n/a | 0.017 | n/a | 4x context compression | [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| ICAE | Autoencoding BLEU | Llama-2-13B, 512 to 128 slots | n/a | 99.8 | n/a | 4x context compression | [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| ICAE | Latency | Llama-7B, 8 x 2048 total | 24.0s | 7.3s | -16.7s | 3.3x speedup | [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| ICAE | Latency | Llama-7B, 8 x 512 total | 9.3s | 4.3s | -5.0s | 2.2x speedup | [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| ICAE | Latency | Llama-7B, 32 x 512 total | 24.3s | 6.8s | -17.5s | 3.6x speedup | [arXiv:2307.06945](https://arxiv.org/abs/2307.06945) |
| IC-Former | PwC ROUGE-L F1 | IC-Former vs ICAE | 0.519 | 0.482 | -0.037 | 512 tokens to 128 digest vectors | [arXiv:2406.13618](https://arxiv.org/abs/2406.13618) |
| IC-Former | PwC ROUGE-1 F1 | IC-Former vs ICAE | 0.555 | 0.516 | -0.039 | 512 tokens to 128 digest vectors | [arXiv:2406.13618](https://arxiv.org/abs/2406.13618) |
| IC-Former | Reconstruction BLEU-4 | length 500, IC-Former vs ICAE | 0.9654 | 0.9689 | +0.0035 | 512 tokens to 128 digest vectors | [arXiv:2406.13618](https://arxiv.org/abs/2406.13618) |
| IC-Former | Compression FLOPs | compression stage | 8.50e12 | 2.62e11 | -8.238e12 | ~1/32 FLOPs | [arXiv:2406.13618](https://arxiv.org/abs/2406.13618) |
| IC-Former | Total time | 8 x 2048 | 1.845s | 0.343s | -1.502s | 5.3x speedup; compression 68x-112x faster than ICAE | [arXiv:2406.13618](https://arxiv.org/abs/2406.13618) |
| KV-Distill | SQuAD accuracy | Llama-3, 25% KV | 87.6% | 86.6% | -1.0 pp | 75% KV removal | [arXiv:2503.10337](https://arxiv.org/abs/2503.10337) |
| KV-Distill | SQuAD accuracy | Llama-3, 20% KV | 87.6% | 86.0% | -1.6 pp | 80% KV removal | [arXiv:2503.10337](https://arxiv.org/abs/2503.10337) |
| KV-Distill | SQuAD accuracy | Llama-3, H2A baseline at 25% KV | 87.6% | 84.0% | -3.6 pp | 75% KV removal | [arXiv:2503.10337](https://arxiv.org/abs/2503.10337) |
| KV-Distill | GovReport ROUGE-L | Llama-3 finetuned, 1% KV | 23.7 | 22.8 | -0.9 | 99% KV removal | [arXiv:2503.10337](https://arxiv.org/abs/2503.10337) |
| KVSculpt | KL vs full logits | PG19, Qwen2.5-1.5B, r=0.3 | 0.233 Select+Fit | 0.0575 | -0.1755 | 4.1x lower KL; keeps 30% budget | [arXiv:2603.27819](https://arxiv.org/abs/2603.27819) |
| KVSculpt | KL vs full logits | PG19, Qwen2.5-1.5B, r=0.5 | 0.186 Select+Fit | 0.0463 | -0.1397 | 4.0x lower KL; keeps 50% budget | [arXiv:2603.27819](https://arxiv.org/abs/2603.27819) |
| KVSculpt | KL vs full logits | PG19, Qwen2.5-1.5B, r=0.7 | 0.125 Select+Fit | 0.0358 | -0.0892 | 3.5x lower KL; keeps 70% budget | [arXiv:2603.27819](https://arxiv.org/abs/2603.27819) |
| KVSculpt | KL vs full logits | adaptive allocation, r=0.3 | 0.0575 uniform | 0.0431 | -0.0144 | 25% KL reduction; ~170s/context on A100 | [arXiv:2603.27819](https://arxiv.org/abs/2603.27819) |
| KV-CAR | Wikitext PPL | GPT-2 774M, AE-only | 21.4 | 23.3 | +1.9 PPL | 41.6% KV memory reduction | [arXiv:2512.06727](https://arxiv.org/abs/2512.06727) |
| KV-CAR | Wikitext PPL | GPT-2 774M, AE+head reuse | 21.4 | 23.9 | +2.5 PPL | 47.85% KV memory reduction | [arXiv:2512.06727](https://arxiv.org/abs/2512.06727) |
| KV-CAR | PIQA accuracy | GPT-2 774M, AE+head reuse | 0.6262 | 0.5936 | -0.0326 | 47.85% KV memory reduction | [arXiv:2512.06727](https://arxiv.org/abs/2512.06727) |
| KV-CAR | Wikitext PPL | TinyLLaMA AE-only | not tabled here | +2.04 PPL vs full | +2.04 PPL | architecture-agnostic KV compression | [arXiv:2512.06727](https://arxiv.org/abs/2512.06727) |
| KV-CAR | Winogrande accuracy | TinyLLaMA AE-only | not tabled here | -0.0111 vs full | -0.0111 | architecture-agnostic KV compression | [arXiv:2512.06727](https://arxiv.org/abs/2512.06727) |
| KVP | Prefill FLOPs/token | Qwen2.5-7B-Chat, 112 per-head agents | 14.00 GFLOPs | 14.15 GFLOPs | +0.15 GFLOPs | ~1% prefill overhead; 0 decode FLOP overhead | [arXiv:2602.10238](https://arxiv.org/abs/2602.10238) |
| KVP | Compression time | 10k context, single layer | 404ms full-model prefill | 0.71ms compression | -403.29ms | paper plots quality but no tabled accuracy drop | [arXiv:2602.10238](https://arxiv.org/abs/2602.10238) |
| LESS | Wikitext PPL | Llama2-7B, H2O 2% budget | 8.791 | 10.745 | +1.954 PPL | ~98% sparse token KV reduction plus LR state | [arXiv:2402.09398](https://arxiv.org/abs/2402.09398) |
| LESS | Wikitext PPL | Llama2-7B, H2O baseline 2% | 8.791 | 13.333 | +4.542 PPL | same 2% sparse budget | [arXiv:2402.09398](https://arxiv.org/abs/2402.09398) |
| LESS | PG-19 PPL | Llama2, H2O 5% | 23.787 | 27.089 | +3.302 PPL | 5% sparse cache + ~4 token-equivalent LR state | [arXiv:2402.09398](https://arxiv.org/abs/2402.09398) |
| LESS | PG-19 PPL | Llama2, H2O baseline 5% | 23.787 | 27.939 | +4.152 PPL | 5% sparse cache | [arXiv:2402.09398](https://arxiv.org/abs/2402.09398) |
| LESS | CNN/DM ROUGE-1 | Falcon7B | 25.92 | 23.22 | -2.70 | H2O+LESS | [arXiv:2402.09398](https://arxiv.org/abs/2402.09398) |
| LESS | CNN/DM ROUGE-1 | Falcon7B, H2O baseline | 25.92 | 21.26 | -4.66 | H2O | [arXiv:2402.09398](https://arxiv.org/abs/2402.09398) |
| LESS | Latency | Llama2-13B, A100, 5000+5000 | 257.3s | 204.7s | -52.6s | 1.26x speedup | [arXiv:2402.09398](https://arxiv.org/abs/2402.09398) |
| LESS | Throughput | Llama2-7B, larger batch enabled | 421.2 tok/s | 699.2 tok/s | +278.0 tok/s | 1.66x throughput | [arXiv:2402.09398](https://arxiv.org/abs/2402.09398) |
| SAC | MRQA ID F1 | 15x, Llama-3.2-1B, SAC vs Full-FT | 71.51 | 54.95 | -16.56 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | MRQA ID EM | 15x, Llama-3.2-1B, SAC vs Full-FT | 56.85 | 39.67 | -17.18 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | MRQA ID F1 | 15x, SAC vs EPL | 51.52 | 54.95 | +3.43 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | MRQA ID EM | 15x, SAC vs EPL | 36.65 | 39.67 | +3.02 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | MRQA ID F1 | 15x, SAC vs ICAE | 44.50 | 54.95 | +10.45 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | MRQA ID EM | 15x, SAC vs ICAE | 31.28 | 39.67 | +8.39 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | MRQA OOD F1 | 15x, SAC vs Full-FT | 52.51 | 39.26 | -13.25 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | MRQA OOD EM | 15x, SAC vs Full-FT | 36.87 | 26.02 | -10.85 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | MRQA OOD F1 | 15x, SAC vs EPL | 36.74 | 39.26 | +2.52 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | MRQA OOD EM | 15x, SAC vs EPL | 23.83 | 26.02 | +2.19 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | Long summarization ROUGE-1 | 15x, SAC vs EPL | 17.61 | 18.49 | +0.88 | 15x compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | Training time | SAC vs ICAE | 3.85h | 2.66h | -1.19h | ~31% faster | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SAC | Compression latency | SAC vs 500x baseline | 257.43ms/batch | 243.87ms/batch | -13.56ms | faster compression | [arXiv:2510.08907](https://arxiv.org/abs/2510.08907) |
| SoftPromptComp | Cost/time | CNN/DailyMail | 12.33 | 3.37 | -8.96 | -77.9% processing cost/time | [arXiv:2404.04997](https://arxiv.org/abs/2404.04997) |
| SoftPromptComp | Cost/time | SST-2 | 4.22 | 1.86 | -2.36 | -63.9% processing cost/time | [arXiv:2404.04997](https://arxiv.org/abs/2404.04997) |
| SoftPromptComp | Cost/time | AG News | 42.41 | 15.51 | -26.90 | -78.5% processing cost/time | [arXiv:2404.04997](https://arxiv.org/abs/2404.04997) |
| SoftPromptComp | Cost/time | SQuAD2.0 | 2.14 | 0.42 | -1.72 | -80.1% processing cost/time | [arXiv:2404.04997](https://arxiv.org/abs/2404.04997) |

## Training-Free Eviction / Merging / Selection

| Paper | Dataset / metric | Setting | Full/Base | Compressed/Method | Delta | Compression / speed / memory | Source |
|---|---:|---|---:|---:|---:|---|---|
| H2O | COPA accuracy | OPT-30B, 20% KV budget | 85.0 | 84.0 | -1.0 | 5x KV reduction | [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) |
| H2O | OpenBookQA accuracy | OPT-30B, 20% KV budget | 43.2 | 43.0 | -0.2 | 5x KV reduction | [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) |
| H2O | PiQA accuracy | OPT-30B, 20% KV budget | 78.51 | 78.45 | -0.06 | 5x KV reduction | [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) |
| H2O | Winogrande accuracy | OPT-30B, 20% KV budget | 70.24 | 69.06 | -1.18 | 5x KV reduction | [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) |
| H2O | Throughput | OPT-6.7B/30B systems | baseline | up to 29x vs DeepSpeed/Accelerate; 3x vs FlexGen | n/a | latency up to 1.9x lower on A100 | [arXiv:2306.14048](https://arxiv.org/abs/2306.14048) |
| Scissorhands | HellaSwag accuracy | OPT-6B, 2x compression | 0.702 | 0.706 | +0.004 | 2x KV reduction | [arXiv:2305.17118](https://arxiv.org/abs/2305.17118) |
| Scissorhands | HellaSwag accuracy | OPT-6B, 2x + 4-bit | 0.702 | 0.704 | +0.002 | 2x token compression plus 4-bit | [arXiv:2305.17118](https://arxiv.org/abs/2305.17118) |
| Scissorhands | HellaSwag accuracy | OPT-13B, 2x compression | 0.720 | 0.720 | 0.000 | 2x KV reduction | [arXiv:2305.17118](https://arxiv.org/abs/2305.17118) |
| SnapKV | LongBench average | Mistral, cap 1024 | 52.88 | 51.87 | -1.01 | ~92% prompt-KV compression on avg 13k input | [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) |
| SnapKV | LongBench average | Mistral, cap 2048 | 52.88 | 52.01 | -0.87 | prompt-KV cap 2048 | [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) |
| SnapKV | LongBench average | Mistral, cap 4096 | 52.88 | 52.65 | -0.23 | ~68% prompt-KV compression on avg 13k input | [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) |
| SnapKV | NIAH score | Command-R | 9.866 | 9.819 | -0.047 | cap setting paper-reported | [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) |
| SnapKV | RAG F1 | paper-reported | baseline | baseline -1.2 pp | -1.2 pp | reports 3.6x generation speed; 8.2x memory efficiency is context extension metric | [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) |
| SnapKV | End-to-end RAG F1 | paper-reported | baseline | baseline -2.1 pp | -2.1 pp | reports 3.6x generation speed | [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) |
| PyramidKV | LongBench average | LLaMA-3-8B, KV=64 | 41.46 | 34.76 | -6.70 | very low retention | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| PyramidKV | LongBench average | LLaMA-3-8B, KV=2048 | 41.46 | 41.49 | +0.03 | paper says matches full around 12% retention | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| PyramidKV | NIAH accuracy | LLaMA-3-70B, KV=128 | 100.0 | 100.0 | 0.0 | KV=128 entries | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| PyramidKV | KV memory | LLaMA-3-8B, seq8192, full | 6848MB | n/a | n/a | reference full memory | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| PyramidKV | KV memory | LLaMA-3-8B, seq8192, cache512 | 6848MB | 428MB | -6420MB | ~16.0x smaller | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| PyramidKV | KV memory | LLaMA-3-8B, seq8192, cache1024 | 6848MB | 856MB | -5992MB | ~8.0x smaller | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| PyramidKV | KV memory | LLaMA-3-8B, seq8192, cache2048 | 6848MB | 1712MB | -5136MB | ~4.0x smaller | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| PyramidKV | TREC accuracy | KV=64, vs SnapKV | 38.5 SnapKV | 58.0 | +19.5 pp | low KV budget | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| PyramidKV | TREC accuracy | KV=64, vs H2O | 38.0 H2O | 58.0 | +20.0 pp | low KV budget | [arXiv:2406.02069](https://arxiv.org/abs/2406.02069) |
| CaM | OpenBookQA accuracy | LLaMA-7B + StreamingLLM | 30.0 | 31.8 | +1.8 | 20% KV budget | [OpenReview](https://openreview.net/forum?id=LCTmppB165) |
| CaM | COPA accuracy | LLaMA-7B + StreamingLLM | 64.0 | 68.0 | +4.0 | 20% KV budget | [OpenReview](https://openreview.net/forum?id=LCTmppB165) |
| CaM | RTE accuracy | LLaMA-7B + StreamingLLM | 49.5 | 54.6 | +5.1 | 20% KV budget | [OpenReview](https://openreview.net/forum?id=LCTmppB165) |
| CaM | OpenBookQA accuracy | vs H2O LLaMA-7B | H2O | H2O +4.8 | +4.8 pp | 20% KV budget | [OpenReview](https://openreview.net/forum?id=LCTmppB165) |
| CaM | COPA accuracy | vs H2O LLaMA-7B | H2O | H2O +9.0 | +9.0 pp | 20% KV budget | [OpenReview](https://openreview.net/forum?id=LCTmppB165) |
| CaM | RTE accuracy | vs H2O LLaMA-7B | H2O | H2O +0.9 | +0.9 pp | 20% KV budget | [OpenReview](https://openreview.net/forum?id=LCTmppB165) |
| CaM | Latency | RTX 3090, PG-19, budget 1024 | 26.1 ms/token StreamingLLM | 27.3 ms/token CaM | +1.2 ms/token | Full OOM | [OpenReview](https://openreview.net/forum?id=LCTmppB165) |
| CaM | Memory | RTX 3090, PG-19, budget 1024 | 19.1GB StreamingLLM | 19.3GB CaM | +0.2GB | Full OOM | [OpenReview](https://openreview.net/forum?id=LCTmppB165) |
| KVCompose | RULER-4096 AUC avg | vs TOVA | 73.4 | 82.3 | +8.9 | max compression under 20% tolerance avg 79.8% | [arXiv:2509.05165](https://arxiv.org/abs/2509.05165) |
| KVCompose | RULER-4096 AUC avg | vs SnapKV | 65.8 | 82.3 | +16.5 | engine-compatible structured compression | [arXiv:2509.05165](https://arxiv.org/abs/2509.05165) |
| KVCompose | RULER-4096 AUC avg | vs PyramidKV | 61.5 | 82.3 | +20.8 | engine-compatible structured compression | [arXiv:2509.05165](https://arxiv.org/abs/2509.05165) |
| RocketKV | LongBench average | LLaMA3.1-8B, budget256 | 52.2 | 51.1 | -1.1 | token budget 256 | [arXiv:2502.14051](https://arxiv.org/abs/2502.14051) |
| RocketKV | LongBench average | LLaMA3.1-8B, budget512 | 52.2 | 51.9 | -0.3 | token budget 512 | [arXiv:2502.14051](https://arxiv.org/abs/2502.14051) |
| RocketKV | NIAH accuracy | budget256 | 100.0 | 100.0 | 0.0 | up to 400x compression on NIAH 109K/256 | [arXiv:2502.14051](https://arxiv.org/abs/2502.14051) |
| RocketKV | Speedup | A100 | 1.0x | up to 3.7x | +2.7x | peak memory saving up to 32.6% | [arXiv:2502.14051](https://arxiv.org/abs/2502.14051) |
| RocketKV | Speedup | H100 | 1.0x | up to 3.3x | +2.3x | decode-phase compression | [arXiv:2502.14051](https://arxiv.org/abs/2502.14051) |
| Compactor | RULER score recovery | 50% retention | 100% baseline | 99% baseline | -1% recovery | 50% KV retention | [arXiv:2507.08143](https://arxiv.org/abs/2507.08143) |
| Compactor | RULER score recovery | 25% retention | 100% baseline | 87.1% baseline | -12.9% recovery | 75% KV reduction | [arXiv:2507.08143](https://arxiv.org/abs/2507.08143) |
| Compactor | RULER score recovery | 10% retention | 100% baseline | 68.0% baseline | -32.0% recovery | 90% KV reduction | [arXiv:2507.08143](https://arxiv.org/abs/2507.08143) |
| Compactor | LongBench score | LLaMA, 50% retention | 42.4 | 44.9 | +2.5 | 50% KV retention | [arXiv:2507.08143](https://arxiv.org/abs/2507.08143) |
| Compactor | LongBench score | LLaMA, 25% retention | 42.4 | 42.5 | +0.1 | 75% KV reduction | [arXiv:2507.08143](https://arxiv.org/abs/2507.08143) |
| Compactor | LongBench score | LLaMA, 10% retention | 42.4 | 38.8 | -3.6 | 90% KV reduction | [arXiv:2507.08143](https://arxiv.org/abs/2507.08143) |
| Compactor | LongBench score | LLaMA, 5% retention | 42.4 | 33.9 | -8.5 | 95% KV reduction | [arXiv:2507.08143](https://arxiv.org/abs/2507.08143) |
| Compactor | Context-calibrated LongBench | average | full | within 0.1 of full | ~-0.1 | retains 32% KV; 68% memory reduction | [arXiv:2507.08143](https://arxiv.org/abs/2507.08143) |
| FINCH | SQuADv2 reference accuracy | compression 1.53x | 100% reference | 99% reference | -1% reference | 1.53x compression | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FINCH | SQuADv2 reference accuracy | compression 3.76x | 100% reference | 90% reference | -10% reference | 3.76x compression | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FINCH | NarrativeQA score | LongBench LLaMA2, target 512 | 16.69 Vanilla | 19.10 | +2.41 | 93.17x reported compression in this setting | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FINCH | NarrativeQA KV memory | n=4096, sigma=2 | 4.52GB | 2.38GB | -2.14GB | 1.90x smaller | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FINCH | NarrativeQA KV memory | n=4096, sigma=4 | 4.52GB | 1.30GB | -3.22GB | 3.48x smaller | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FINCH | NarrativeQA KV memory | n=4096, sigma=8 | 4.52GB | 0.60GB | -3.92GB | 7.53x smaller | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FINCH | LongBench single-doc | LLaMA2 target 512 | baseline | baseline +2.75 | +2.75 | 4-bit quant setting | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FINCH | LongBench multi-doc | LLaMA2 target 512 | baseline | baseline +4.07 | +4.07 | 4-bit quant setting | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FINCH | LongBench TREC | LLaMA2 target 512 | baseline | baseline +8.75 | +8.75 | 4-bit quant setting | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FINCH | LongBench PassageCount | LLaMA2 target 512 | 4.25 | 2.45 | -1.80 | failure case | [TACL](https://aclanthology.org/2024.tacl-1.83/) |
| FAEDKV | LongBench average | LLaMA3-8B, r=0.094 | 33.42 | 25.21 | -8.21 | ~90.6% middle-cache reduction | [arXiv:2507.20030](https://arxiv.org/abs/2507.20030) |
| FAEDKV | LongBench average | LLaMA3-8B, r=0.125 | 33.42 | 29.45 | -3.97 | ~87.5% middle-cache reduction | [arXiv:2507.20030](https://arxiv.org/abs/2507.20030) |
| FAEDKV | LongBench average | LLaMA3-8B, r=0.25 | 33.42 | 33.24 | -0.18 | ~75% middle-cache reduction | [arXiv:2507.20030](https://arxiv.org/abs/2507.20030) |
| FAEDKV | LongBench average | r=0.25, vs SnapKV | SnapKV | SnapKV +0.64 | +0.64 | position-agnostic frequency compression | [arXiv:2507.20030](https://arxiv.org/abs/2507.20030) |
| ZSMerge | XSum ROUGE-1 | LLaMA2, 5% cache | 30.59 | 30.60 | +0.01 | 20:1 compression | [arXiv:2503.10714](https://arxiv.org/abs/2503.10714) |
| ZSMerge | XSum ROUGE-2 | LLaMA2, 5% cache | 11.34 | 11.67 | +0.33 | 20:1 compression | [arXiv:2503.10714](https://arxiv.org/abs/2503.10714) |
| ZSMerge | XSum ROUGE-L | LLaMA2, 5% cache | 25.50 | 25.72 | +0.22 | 20:1 compression | [arXiv:2503.10714](https://arxiv.org/abs/2503.10714) |
| ZSMerge | XSum ROUGE-1 | Falcon, 5% cache | 27.06 | 15.04 | -12.02 | 20:1 compression; failure case | [arXiv:2503.10714](https://arxiv.org/abs/2503.10714) |
| ZSMerge | XSum ROUGE-2 | Falcon, 5% cache | 8.79 | 3.29 | -5.50 | 20:1 compression; failure case | [arXiv:2503.10714](https://arxiv.org/abs/2503.10714) |
| ZSMerge | XSum ROUGE-L | Falcon, 5% cache | 22.39 | 12.73 | -9.66 | 20:1 compression; failure case | [arXiv:2503.10714](https://arxiv.org/abs/2503.10714) |
| ZSMerge | Throughput | 2048+2048, batch16, 7B | 133.1 tok/s | 281.9 tok/s | +148.8 tok/s | 2.12x throughput; reports 82% VRAM reduction at 54K | [arXiv:2503.10714](https://arxiv.org/abs/2503.10714) |
| ToMe | Video top-1 | ViT-L MAE, r65 | 84.7 | 84.5 | -0.2 | 7.3 to 16.3 clips/s, 2.2x speedup | [arXiv:2210.09461](https://arxiv.org/abs/2210.09461) |
| ToMe | Audio mAP | trained r40 | 46.4 | 46.0 | -0.4 | 103 to 200 samples/s, 1.94x speedup | [arXiv:2210.09461](https://arxiv.org/abs/2210.09461) |

## Architecture / System Alternatives And Benchmark Caveats

| Paper | Dataset / metric | Setting | Full/Base | Compressed/Method | Delta | Compression / speed / memory | Source |
|---|---:|---|---:|---:|---:|---|---|
| YOCO | LM Eval Harness average | YOCO-3B gRet | 0.636 at 1.6T | 0.645 after 1M extension | +0.009 | architecture-level cache reuse | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | Multi-needle 128K | N=1 | n/a | 0.98 | n/a | long-context extension | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | Multi-needle 128K | N=2 | n/a | 0.98 | n/a | long-context extension | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | Multi-needle 128K | N=4 | n/a | 0.84 | n/a | long-context extension | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | Multi-needle 128K | N=8 | n/a | 0.56 | n/a | long-context extension | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | KV memory | 65B, claimed reduction | Transformer | YOCO | n/a | ~80x KV reduction | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | KV memory | 1M context | YOCO 12.4GB | Transformer 9.4x more | n/a | YOCO much lower KV memory | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | Prefill | 1M context | Transformer 1.0x | YOCO 71.8x | +70.8x | architecture speedup | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | Prefill | 32K context | Transformer 1.0x | YOCO 2.87x | +1.87x | architecture speedup | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | Throughput | 512K context | 4.5 tok/s Transformer | 43.1 tok/s YOCO | +38.6 tok/s | 9.58x throughput | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| YOCO | PPL | small ablation, YOCO-gRet vs Transformer | 3.564 Transformer | 3.530 YOCO-gRet | -0.034 | not plug-in compression | [arXiv:2405.05254](https://arxiv.org/abs/2405.05254) |
| Infini-attention | PPL | PG19, Infini Linear vs Memorizing Transformer | 11.37 | 9.65 | -1.72 | bounded compressive memory | [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) |
| Infini-attention | PPL | Arxiv-math, Infini Linear vs Memorizing Transformer | 2.26 | 2.24 | -0.02 | bounded compressive memory | [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) |
| Infini-attention | PPL | PG19, Linear+Delta vs Memorizing Transformer | 11.37 | 9.67 | -1.70 | bounded compressive memory | [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) |
| Infini-attention | PPL | Arxiv-math, Linear+Delta vs Memorizing Transformer | 2.26 | 2.23 | -0.03 | bounded compressive memory | [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) |
| Infini-attention | Memory params | vs Memorizing Transformer | 183M | 1.6M | -181.4M | 114x smaller memory params | [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) |
| Infini-attention | 1M passkey | zero-shot Linear+Delta, start/middle/end | n/a | 7 / 6 / 97 | n/a | needs fine-tune for stable retrieval | [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) |
| Infini-attention | BookSum Overall | vs PRIMERA+Unlimiformer | 17.2 | 18.5 | +1.3 | 500K eval after FT | [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) |
| StreamingLLM | PG19 PPL | Llama-2-13B, 0+1024 window vs 4+1020 sink+window | 5158.07 | 5.40 | -5152.67 | constant-size KV | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| StreamingLLM | Speedup | vs sliding-window recomputation | 1.0x | up to 22.2x | +21.2x | constant cache | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| StreamingLLM | SCBench multi-turn avg | Llama-3.1-8B | 48.7 full | 15.5 | -33.2 | retrieval/shared-context failure | [SCBench](https://arxiv.org/abs/2412.10319) |
| StreamingLLM | SCBench multi-request avg | Llama-3.1-8B | 37.2 full | 14.5 | -22.7 | retrieval/shared-context failure | [SCBench](https://arxiv.org/abs/2412.10319) |
| StreamingLLM | SCBench Retr.String | Llama-3.1-8B | 57.1 full | 0.4 | -56.7 | exact retrieval collapse | [SCBench](https://arxiv.org/abs/2412.10319) |
| DuoAttention | MMLU | Llama-3-70B short task | 79.38 | 79.35 | -0.03 | memory reduction up to 2.55x MHA / 1.67x GQA | [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) |
| DuoAttention | MBPP | Llama-3-70B short task | 47.85 | 47.09 | -0.76 | decode speed up to 2.18x MHA / 1.50x GQA | [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) |
| DuoAttention | MT-Bench | Llama-3-70B short task | 8.93 | 9.14 | +0.21 | prefill latency up to 1.73x MHA / 1.63x GQA | [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) |
| DuoAttention | Context length | Llama-3-8B + quantization on A100-80G | baseline | 3.30M tokens | n/a | quantization + retrieval/streaming heads | [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) |
| LightTransfer | LongBench avg | Mistral | 47.6 | 47.4 | -0.2 | 50% lazy layers on LongBench | [arXiv:2410.13846](https://arxiv.org/abs/2410.13846) |
| LightTransfer | LongBench avg | Llama3-8B | 40.8 | 38.7 | -2.1 | layer-level streaming conversion | [arXiv:2410.13846](https://arxiv.org/abs/2410.13846) |
| LightTransfer | LongBench avg | Llama3-70B | 44.8 | 43.6 | -1.2 | layer-level streaming conversion | [arXiv:2410.13846](https://arxiv.org/abs/2410.13846) |
| LightTransfer | NIAH single-key | 32K | 96.7 | 96.6 | -0.1 | 25% layers reduced | [arXiv:2410.13846](https://arxiv.org/abs/2410.13846) |
| LightTransfer | NIAH multi-key | 32K | 78.2 | 78.9 | +0.7 | 25% layers reduced | [arXiv:2410.13846](https://arxiv.org/abs/2410.13846) |
| LightTransfer | Throughput | Mistral/RULER 4K | 1.0x | 1.44x | +0.44x | BF16 A100 | [arXiv:2410.13846](https://arxiv.org/abs/2410.13846) |
| LightTransfer | Throughput | Mistral/RULER 8K | 1.0x | 1.78x | +0.78x | BF16 A100 | [arXiv:2410.13846](https://arxiv.org/abs/2410.13846) |
| LightTransfer | Throughput | Mistral/RULER 16K | 1.0x | 2.17x | +1.17x | BF16 A100 | [arXiv:2410.13846](https://arxiv.org/abs/2410.13846) |
| LightTransfer | Throughput | Mistral/RULER 32K | 1.0x | 1.75x | +0.75x | BF16 A100 | [arXiv:2410.13846](https://arxiv.org/abs/2410.13846) |
| SWAA | LongMemEval | Qwen3-4B Thinking, naive 2K SWA | 73.0 | 3.2 | -69.8 | naive SWA collapse | [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) |
| SWAA | Ruler | Qwen3-4B Thinking, naive 2K SWA | 85.6 | 0.0 | -85.6 | naive SWA collapse | [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) |
| SWAA | LongMemEval | best combined SFT+FA decode+interleaving | 74.6 | 73.2 | -1.4 | quality-preserving config | [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) |
| SWAA | LongBench-V2 | best combined SFT+FA decode+interleaving | 37.9 | 38.3 | +0.4 | quality-preserving config | [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) |
| SWAA | Ruler | best combined SFT+FA decode+interleaving | 88.2 | 74.0 | -14.2 | retrieval remains hard | [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) |
| SWAA | Throughput | vLLM/H100 full vs pure 2K SWA | 3.74k tok/s | 30.72k tok/s | +26.98k tok/s | pure SWA fast but poor accuracy | [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) |
| SCBench | Multi-turn avg | Llama-3.1-8B, SnapKV tau=1/32 | 48.7 full | 20.9 | -27.8 | O(k) KV dropping | [arXiv:2412.10319](https://arxiv.org/abs/2412.10319) |
| SCBench | Multi-request avg | Llama-3.1-8B, SnapKV tau=1/32 | 37.2 full | 15.2 | -22.0 | O(k) KV dropping | [arXiv:2412.10319](https://arxiv.org/abs/2412.10319) |
| SCBench | Retr.String | Llama-3.1-8B, SnapKV | 57.1 full | 6.1 | -51.0 | exact retrieval collapse | [arXiv:2412.10319](https://arxiv.org/abs/2412.10319) |
| SCBench | Multi-turn avg | Llama-3.1-8B, PyramidKV tau=1/32 | 48.7 full | 21.0 | -27.7 | O(k) KV dropping | [arXiv:2412.10319](https://arxiv.org/abs/2412.10319) |
| SCBench | Multi-request avg | Llama-3.1-8B, PyramidKV tau=1/32 | 37.2 full | 15.3 | -21.9 | O(k) KV dropping | [arXiv:2412.10319](https://arxiv.org/abs/2412.10319) |

## Quantization / Low-Rank / Network-Oriented Compression

| Paper | Dataset / metric | Setting | Full/Base | Compressed/Method | Delta | Compression / speed / memory | Source |
|---|---:|---|---:|---:|---:|---|---|
| KIVI | LongBench avg | Llama2-7B, 2-bit KV | 44.52 | 44.27 | -0.25 | peak memory 2.6x lower; throughput 2.35x-3.47x | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| KIVI | LongBench avg | Llama2-13B, 2-bit KV | 44.85 | 44.69 | -0.16 | peak memory 2.6x lower; throughput 2.35x-3.47x | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| KIVI | LongBench avg | Mistral-7B, 2-bit KV | 46.58 | 45.85 | -0.73 | peak memory 2.6x lower; throughput 2.35x-3.47x | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| KIVI | GSM8K accuracy | Llama2-13B, 2-bit KV | 22.67 | 20.77 | -1.90 | 2-bit asymmetric KV | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| KIVI | GSM8K accuracy | Mistral, 2-bit KV | 38.36 | 36.01 | -2.35 | 2-bit asymmetric KV | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| KVQuant | Wikitext-2 PPL | LLaMA-7B, 3bit-1% | 5.68 | 5.75 | +0.07 PPL | 4.8x KV memory compression | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| KVQuant | LongBench avg | LLaMA-2-7B-32K | 31.96 | 31.21 | -0.75 | 3bit-1% / 4.8x KV compression | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| KVQuant | LongBench avg | vs KIVI same comparison | 30.04 KIVI | 31.21 | +1.17 | KVQuant higher quality | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| KVQuant | RULER avg | LLaMA-2-7B-32K | 56.40 | 53.65 | -2.75 | 3bit-1% / 4.8x KV compression | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| KVQuant | RULER avg | vs KIVI same comparison | 39.78 KIVI | 53.65 | +13.87 | KVQuant higher quality | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| KVQuant | Key matvec latency | 4-bit kernel | 1.0x | 1.2x-1.6x faster | +0.2x to +0.6x | kernel speed | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| KVQuant | Value latency | 4-bit kernel | 1.0x | 1.3x-1.7x faster | +0.3x to +0.7x | kernel speed | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| KVQuant | Context capacity estimate | LLaMA-7B, A100-80GB | baseline | 1M tokens | n/a | 1 GPU | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| KVQuant | Context capacity estimate | LLaMA-7B, 8 GPUs | baseline | 10M tokens | n/a | 8 GPUs | [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| LoRC | OpenBookQA accuracy | LLaMA-3-8B, progressive 60% | 78.0 | 77.4 | -0.6 | keeps 60% KV budget | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| LoRC | OpenBookQA accuracy | LLaMA-3-70B, progressive 60% | 91.2 | 91.2 | 0.0 | keeps 60% KV budget | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| LoRC | KV usage | LLaMA-2-13B | 50G | 27.5G | -22.5G | 45% memory reduction | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| LoRC | KV usage | LLaMA-3-8B | 8G | 4.8G | -3.2G | 40% memory reduction | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| LoRC | KV usage | LLaMA-3-70B | 20G | 11G | -9G | 45% memory reduction | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| LoRC | Average performance drop | LLaMA-2-13B | full | compressed | -0.47% | paper-reported avg | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| LoRC | Average performance drop | LLaMA-3-8B | full | compressed | -0.92% | paper-reported avg | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| LoRC | Average performance drop | LLaMA-3-70B | full | compressed | -0.22% | paper-reported avg | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| LoRC | OpenBookQA drop | bad shallow-block compression, 70B | full | compressed | -68.0 pp | failure case; sensitivity warning | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| LoRC | SVD preprocessing time | 70B, 80 layers | n/a | ~40s | n/a | no retraining | [arXiv:2410.03111](https://arxiv.org/abs/2410.03111) |
| DeltaKV | LongBench avg | Llama-3.1-8B | 50.0 | 50.2 | +0.2 | residual KV compression | [arXiv:2602.08005](https://arxiv.org/abs/2602.08005) |
| DeltaKV | LongBench avg | Llama-3.1-8B, DeltaKV+4bit KR 29% | 50.0 | 50.3 | +0.3 | KV memory to 29% original | [arXiv:2602.08005](https://arxiv.org/abs/2602.08005) |
| DeltaKV | SCBench avg | Llama-3.1-8B, DeltaKV+4bit | 50.4 | 46.8 | -3.6 | KV memory to 29% original | [arXiv:2602.08005](https://arxiv.org/abs/2602.08005) |
| DeltaKV | AIME accuracy | DeepSeek-Qwen-7B | 50.0 | 43.3 | -6.7 | residual KV compression | [arXiv:2602.08005](https://arxiv.org/abs/2602.08005) |
| DeltaKV | Throughput | Sparse-vLLM, 128k context | 143.2 tok/s | 187.0 tok/s | +43.8 tok/s | 1.31x speedup | [arXiv:2602.08005](https://arxiv.org/abs/2602.08005) |
| DeltaKV | Throughput | Sparse-vLLM, 512k context | 33.1 tok/s | 67.7 tok/s | +34.6 tok/s | 2.05x speedup | [arXiv:2602.08005](https://arxiv.org/abs/2602.08005) |
| CacheGen | Transmitted KV size | vs default quantization | 1.0x | 3.5x-4.3x smaller | n/a | network artifact compression | [arXiv:2310.07240](https://arxiv.org/abs/2310.07240) |
| CacheGen | TTFT | vs prefill from text | 1.0x | 3.1x-4.7x faster | n/a | network + KV streaming | [arXiv:2310.07240](https://arxiv.org/abs/2310.07240) |
| CacheGen | TTFT | vs default quantization | 1.0x | 3.2x-3.7x faster | n/a | network + KV streaming | [arXiv:2310.07240](https://arxiv.org/abs/2310.07240) |
| CacheGen | TTFT | vs 8-bit quantization | 1.0x | 1.67x-1.81x faster | n/a | network + KV streaming | [arXiv:2310.07240](https://arxiv.org/abs/2310.07240) |
| CacheGen | Accuracy degradation | lossy compression | full | compressed | <=2% accuracy loss | not mainly GPU-resident memory | [arXiv:2310.07240](https://arxiv.org/abs/2310.07240) |
| CacheGen | F1 degradation | lossy compression | full | compressed | <0.1% F1 loss | not mainly GPU-resident memory | [arXiv:2310.07240](https://arxiv.org/abs/2310.07240) |
| CacheGen | PPL degradation | lossy compression | full | compressed | <0.1 PPL | not mainly GPU-resident memory | [arXiv:2310.07240](https://arxiv.org/abs/2310.07240) |
| KVTuner | GSM8K avg | Llama-3.1-8B, C3.25 | 0.8038 BF16 | 0.7925 | -0.0113 | 3.25-bit mixed precision | [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) |
| KVTuner | GSM8K avg | Qwen2.5-7B, naive KV4 | 0.7755 BF16 | 0.0078 | -0.7677 | failure case | [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) |
| KVTuner | GSM8K avg | Qwen2.5-7B, KVTuner-C4.0 | 0.7755 BF16 | 0.7559 | -0.0196 | 4-bit searched config | [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) |
| KVTuner | LongBench | Qwen2.5-7B, KVTuner-C3.92 | 0.7956 BF16 | 0.7903 | -0.0053 | 3.92-bit searched config | [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) |
| KVTuner | LongBench | Qwen2.5-7B, per-token-asym C4.0 | 0.7956 BF16 | 0.7960 | +0.0004 | 4-bit config | [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) |
| KVTuner | Throughput | Llama-3.1-8B, KIVI kernel | KIVI-KV8 | KVTuner-C3.25 | +16.79%-21.25% | offline searched, no online overhead | [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) |

## Rows Without Verified Numeric Tables

| Source | What is missing | Usable status |
|---|---|---|
| KVP | Quality numbers are mainly plotted, not tabulated in extracted source | Use qualitative claim plus overhead rows above unless manually digitizing figures |
| KVSculpt | No downstream accuracy/PPL/F1/EM table | Use KL/logit-preservation rows above |
| EMPIRIC | ACM full text was not accessible in this run | Metadata/abstract only; do not cite eval numbers until PDF is read |
| KV Cache Compression Review | No original eval | Use as taxonomy/source discovery, not result table |
| Efficient Attention Mechanisms Survey | No original KV compression eval | Use as background only |

