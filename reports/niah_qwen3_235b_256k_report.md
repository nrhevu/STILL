# Qwen3-235B 256k NIAH KV Compression Report

## Summary

- Model: `Qwen/Qwen3-235B-A22B`
- Checkpoint: `none`
- Records: 3
- Compact success rate: 1.0000
- Compact exact rate: 1.0000
- Mean compression: 875.4698x
- Full-cache success rate: 1.0000
- Compact/full success ratio: 1.0000 (PASS, target 0.9500)
- Note: 256k uses empirical YaRN factor 8; official model-card example validates 131072.
- Note: This result uses an untrained num_latents=0 sink plus lexical exact-token baseline, not a trained latent compactor checkpoint.

## By Context And Depth

| Context | Task | Depth % | Trials | Compact | Full | Compact/Full | Compression | Decode s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262144 | multi_needle | 100.0000 | 1 | 1.0000 | 1.0000 | 1.0000 | 825.4606 | 1.6234 |
| 262144 | single | 0.0000 | 1 | 1.0000 | 1.0000 | 1.0000 | 1022.1602 | 1.7967 |
| 262144 | two_hop | 50.0000 | 1 | 1.0000 | 1.0000 | 1.0000 | 778.7887 | 1.7553 |
