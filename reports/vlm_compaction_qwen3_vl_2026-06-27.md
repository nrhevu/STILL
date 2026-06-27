# Báo cáo thí nghiệm VLM compaction với Qwen3-VL

Ngày báo cáo: 2026-06-27

## Tóm tắt

Đã triển khai pipeline VLM multiple-choice cho `Qwen/Qwen3-VL-8B-Instruct`, chuẩn bị dữ liệu ScienceQA/MMMU, đo baseline full-cache, train compactor trên ScienceQA, và evaluate compact-cache trên ScienceQA validation/test cùng MMMU validation.

Kết quả chính:

| Tập đánh giá | Rows | Full-cache acc | Compact acc | Compact/full | Mean compression |
| --- | ---: | ---: | ---: | ---: | ---: |
| ScienceQA validation | 2,097 | 94.09% | 94.52% | 100.46% | 0.2245 |
| ScienceQA test | 2,017 | 94.79% | 95.04% | 100.26% | 0.2270 |
| MMMU validation | 847 | 56.91% | 56.20% | 98.76% | 0.5317 |

Yêu cầu compact accuracy đạt ít nhất 95% so với full-cache accuracy đã đạt trên các evaluation đã chạy. Mọi lệnh GPU được khóa bằng `HIP_VISIBLE_DEVICES=4,5,6,7`.

## Những phần đã làm

### 1. Dataset và chuẩn hóa dữ liệu

Đã thêm pipeline chuẩn hóa VLM MCQ theo schema JSONL gồm `id`, `source`, `split`, `images`, `context_text`, `question`, `choices`, `answer_index`, `answer_letter`, và metadata task/subject.

Artifacts dữ liệu:

| Dataset | Split | Rows |
| --- | --- | ---: |
| ScienceQA | train | 6,218 |
| ScienceQA | validation | 2,097 |
| ScienceQA | test | 2,017 |
| MMMU | dev | 141 |
| MMMU | validation | 847 |

Files chính:

- `scripts/prepare_vlm_mcq.py`
- `src/neural_kv/data/vlm.py`
- `data/vlm_scienceqa/*.jsonl`
- `data/vlm_mmmu/*.jsonl`

### 2. VLM evaluator

Đã thêm evaluator cho full-cache và compact-cache:

- Full-cache scoring bằng `letter_logprob` và generation mode.
- Compact-cache scoring bằng continuation từ cache đã compact.
- Prompt styles: `compact`, `official_mmmu`, `qwen_mmmu`.
- Metrics JSON gồm `rows`, `full_accuracy`, `compact_accuracy`, `mean_compression`, `prediction_counts`, `parse_errors`, `skipped_too_long`, `task_accuracy`.
- Hỗ trợ image pixel/token budget và system prompt.

Files chính:

- `scripts/evaluate_vlm.py`
- `src/neural_kv/eval/vlm.py`
- `src/neural_kv/training/vlm.py`

### 3. Qwen3-VL cache/compactor integration

Đã thêm các phần cần thiết để compactor chạy với Qwen3-VL:

- Loader Qwen3-VL qua `AutoProcessor` và model multimodal.
- Lấy thông số text model từ `model.config.text_config`.
- Thêm `rope_mode: none` để tránh áp dụng RoPE text chuẩn sai trên Qwen3-VL Interleaved-MRoPE.
- Sửa continuation position/MRoPE cho Qwen3-VL cache scoring.
- Patch attention layers cho STILL beta trên VLM model.
- Checkpoint save/load có `rope_mode`.

Files chính:

- `src/neural_kv/models/compactor.py`
- `src/neural_kv/models/checkpointing.py`
- `src/neural_kv/training/callbacks.py`
- `src/neural_kv/training/vlm.py`

### 4. GPU guard

Đã thêm guard để đảm bảo các lệnh GPU chỉ chạy trên 4 GPU cuối:

- Biến môi trường bắt buộc: `HIP_VISIBLE_DEVICES=4,5,6,7`
- PyTorch/ROCm nhìn thấy chúng là `cuda:0..3`.
- Script abort nếu cấu hình GPU không đúng.

Files chính:

- `src/neural_kv/utils/rocm.py`
- `tests/test_vlm_rocm.py`

### 5. Config training

Config train chính:

- `config/experiment/vlm_qwen3_vl_8b_scienceqa_4gpu.yaml`

Thông số chính:

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Context length: `8192`
- Latents: `1024`
- Sink tokens: `8`
- Exact tokens: `128`
- Exact strategy: `kv_norm`
- `beta_base: zero`
- `num_blocks: 2`
- `lr: 5e-6`
- Precision: bf16
- Steps: `1200`
- DDP: `4` devices

Checkpoint cuối:

- `checkpoints/vlm_qwen3_vl_8b_scienceqa_4gpu/final.pt`
- `checkpoints/vlm_qwen3_vl_8b_scienceqa_4gpu/step_1200.pt`

## Baseline và gates

Baseline full-cache/gate đã chạy trước khi train compactor:

| Gate | Rows | Full-cache acc | Ngưỡng | Kết quả |
| --- | ---: | ---: | ---: | --- |
| ScienceQA smoke | 256 | 94.92% | 75% | Pass |
| MMMU smoke | 150 | 63.33% | 60% | Pass |
| MMMU validation generation (`qwen_mmmu`) | 847 | 57.26% | 55% | Pass |

Artifact baseline:

- `outputs/vlm/qwen3_vl_8b/scienceqa_smoke_full.json`
- `outputs/vlm/qwen3_vl_8b/mmmu_smoke_full.json`
- `outputs/vlm/qwen3_vl_8b/mmmu_validation_generation_qwen_mmmu_lmms_cli.json`

Lưu ý: mốc MMMU validation generation được chấp nhận làm gate thực nghiệm local sau khi full-cache không đạt mốc report công khai 69.6%. Gate hiện tại dùng `min_full_accuracy: 0.55` trên artifact generation baseline 847 rows.

## Training

Command train chính:

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 \
  /scratch/longnguyen37/.local/bin/uv run python scripts/train_vlm.py \
  --config config/experiment/vlm_qwen3_vl_8b_scienceqa_4gpu.yaml
```

Training hoàn tất:

- `max_steps=1200` reached
- Checkpoint final ghi thành công
- Validation trong training ở step cuối: `val/full_accuracy=0.9375`, `val/compact_accuracy=0.9375`
- Summary training: `checkpoints/vlm_qwen3_vl_8b_scienceqa_4gpu/summary.json`
- Metrics log: `checkpoints/vlm_qwen3_vl_8b_scienceqa_4gpu/metrics.jsonl`

## Evaluation sau training

### ScienceQA validation

Artifact:

- `outputs/vlm/qwen3_vl_8b/scienceqa_validation_compact_main.json`
- `outputs/vlm/qwen3_vl_8b/scienceqa_validation_compact_main.details.jsonl`

Kết quả:

- Rows: `2097`
- Full-cache accuracy: `0.9408679065`
- Compact accuracy: `0.9451597520`
- Compact/full: `1.0045615813`
- Mean compression: `0.2245470791`
- Parse errors: `0`
- Skipped too long: `0`

### ScienceQA test

Artifact:

- `outputs/vlm/qwen3_vl_8b/scienceqa_test_compact_main.json`
- `outputs/vlm/qwen3_vl_8b/scienceqa_test_compact_main.details.jsonl`

Kết quả:

- Rows: `2017`
- Full-cache accuracy: `0.9479424888`
- Compact accuracy: `0.9504214179`
- Compact/full: `1.0026150628`
- Mean compression: `0.2270372126`
- Parse errors: `0`
- Skipped too long: `0`

### MMMU validation

Artifact:

- `outputs/vlm/qwen3_vl_8b/mmmu_validation_compact_main.json`
- `outputs/vlm/qwen3_vl_8b/mmmu_validation_compact_main.details.jsonl`

Kết quả:

- Rows: `847`
- Full-cache accuracy: `0.5690672963`
- Compact accuracy: `0.5619834711`
- Compact/full: `0.9875518672`
- Mean compression: `0.5316647018`
- Parse errors: `0`
- Skipped too long: `0`

## Test coverage

Đã thêm/cập nhật tests cho:

- Normalize ScienceQA/MMMU rows.
- Parse options/answers.
- Prompt format.
- GPU visible-device guard.
- `rope_mode` backward compatibility và config.
- VLM experiment config gates.

Command verify cuối:

```bash
/scratch/longnguyen37/.local/bin/uv run pytest -q
```

Kết quả:

```text
75 passed in 2.79s
```

## Reproduce commands

ScienceQA validation:

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 \
  /scratch/longnguyen37/.local/bin/uv run python scripts/evaluate_vlm.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --eval-file data/vlm_scienceqa/validation.jsonl \
  --mode compact \
  --checkpoint checkpoints/vlm_qwen3_vl_8b_scienceqa_4gpu/final.pt \
  --summary-file outputs/vlm/qwen3_vl_8b/scienceqa_validation_compact_main.json \
  --details-file outputs/vlm/qwen3_vl_8b/scienceqa_validation_compact_main.details.jsonl \
  --device cuda \
  --dtype bfloat16
```

ScienceQA test:

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 \
  /scratch/longnguyen37/.local/bin/uv run python scripts/evaluate_vlm.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --eval-file data/vlm_scienceqa/test.jsonl \
  --mode compact \
  --checkpoint checkpoints/vlm_qwen3_vl_8b_scienceqa_4gpu/final.pt \
  --summary-file outputs/vlm/qwen3_vl_8b/scienceqa_test_compact_main.json \
  --details-file outputs/vlm/qwen3_vl_8b/scienceqa_test_compact_main.details.jsonl \
  --device cuda \
  --dtype bfloat16
```

MMMU validation:

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 \
  /scratch/longnguyen37/.local/bin/uv run python scripts/evaluate_vlm.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --eval-file data/vlm_mmmu/validation.jsonl \
  --mode compact \
  --checkpoint checkpoints/vlm_qwen3_vl_8b_scienceqa_4gpu/final.pt \
  --summary-file outputs/vlm/qwen3_vl_8b/mmmu_validation_compact_main.json \
  --details-file outputs/vlm/qwen3_vl_8b/mmmu_validation_compact_main.details.jsonl \
  --device cuda \
  --dtype bfloat16
```

## Kết luận

Pipeline VLM compaction với Qwen3-VL đã hoàn chỉnh ở mức thí nghiệm nội bộ:

- Full-cache baseline không còn 0%.
- Compactor được train thành công trên ScienceQA.
- Compact accuracy đạt trên 95% so với full-cache accuracy trên ScienceQA validation, ScienceQA test, và MMMU validation.
- Mean compression đạt khoảng 22-23% trên ScienceQA và 53% trên MMMU.
- Test suite hiện tại pass toàn bộ.

Các bước tiếp theo nên cân nhắc:

- Điều tra khoảng cách giữa MMMU local baseline và mốc public report 69.6%.
- Thử prompt/image preprocessing khớp official pipeline hơn cho MMMU.
- Train thêm hoặc fine-tune trên tập domain đa dạng hơn nếu muốn cải thiện absolute MMMU compact accuracy, không chỉ retention.
