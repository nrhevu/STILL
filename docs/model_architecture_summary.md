# Model Architecture Summary

Date: 2026-06-06 UTC

## Mục Tiêu

Mô hình trong repo này không train lại toàn bộ LLM. Thay vào đó, nó học một module phụ để nén KV cache của một mô hình nền đã có sẵn.

Thiết lập chính:

- Base model: Qwen/Qwen3-4B
- Runtime: PyTorch ROCm trên AMD GPU
- Context gốc: 8192 tokens
- Context nén: 1024 latent tokens
- Tỷ lệ nén: 8x
- Mục tiêu: sinh kết quả gần với full-context model nhưng dùng KV cache nhỏ hơn nhiều

Ý tưởng cốt lõi: thay vì giữ toàn bộ KV cache của 8192 token context, hệ thống học cách tạo ra một KV cache nén gồm 1024 latent slots. Sau đó LLM tiếp tục decode như bình thường, nhưng attention sẽ nhìn vào cache nén này.

## Thành Phần Chính

### 1. Frozen Base LLM

Base model là Qwen3-4B. Trọng số chính của LLM được giữ nguyên.

Base model vẫn làm các việc sau:

- đọc context dài
- tạo hidden states và KV cache ở từng transformer layer
- decode câu trả lời dựa trên prompt/query

Phần được train chủ yếu là compactor, không phải toàn bộ Qwen.

### 2. Full KV Cache

Khi chạy full context, mỗi transformer layer tạo ra KV cache:

- Key tensor
- Value tensor

KV cache này thường có shape khái niệm:

```text
layers x batch x kv_heads x sequence_length x head_dim
```

Với context length 8192, cache này lớn. Nếu decode tiếp với full cache, model có khả năng truy cập toàn bộ context nhưng chi phí bộ nhớ cao.

### 3. STILL / Neural KV Compactor

Compactor là module học được. Nó nhận KV cache dài của từng layer và tạo KV cache ngắn hơn.

Với mỗi layer, compactor tạo:

- compact keys
- compact values

Thay vì giữ 8192 vị trí, nó giữ 1024 latent vị trí.

Về mặt khái niệm:

```text
Full KV cache:
8192 context tokens

Compactor:
8192 KV positions -> 1024 latent KV positions

Compact KV cache:
1024 latent tokens
```

LLM sau đó dùng compact KV cache như thể đó là cache quá khứ thật.

### 4. Learned Latent Queries

Compactor dùng các latent queries học được để đọc thông tin từ full KV cache.

Mỗi latent slot đóng vai trò như một truy vấn học được:

```text
latent query -> attends over full context KV -> compact latent KV
```

Nói đơn giản:

- full context chứa rất nhiều token
- mỗi latent query học cách chọn/lọc/trộn thông tin quan trọng từ toàn bộ context
- kết quả là một slot trong KV cache nén

Các thí nghiệm gần đây có thêm biến thể head-specific latents:

- shared latent: cùng latent table dùng chung cho KV heads
- head-specific latent: mỗi KV head có latent queries riêng

Head-specific latent tăng độ linh hoạt nhưng cũng tăng số tham số.

### 5. Beta / Residual Parameters

Một số cấu hình có thêm beta hoặc cơ chế điều chỉnh output của compactor.

Mục đích là giúp compact KV cache dễ học hơn bằng cách cho module một đường điều chỉnh learned/residual thay vì chỉ phụ thuộc vào latent attention thô.

Trong các run gần đây, `beta-base zero` được dùng cho fresh identity/cache-fix run. Điều này giúp bắt đầu từ trạng thái đơn giản, tránh mang theo hành vi bị nhiễm từ checkpoint cũ.

## Luồng Hoạt Động Thực Tế

### Bước 1: Chạy Full Context Prefill

Input gồm context dài, ví dụ 8192 tokens.

Base Qwen chạy prefill trên context này và sinh ra full KV cache:

```text
context tokens -> Qwen prefill -> full past_key_values
```

Full cache này là nguồn thông tin để compactor học cách nén.

### Bước 2: Nén Cache Theo Layer

Compactor xử lý từng layer.

Ở mỗi layer:

```text
full K_l, full V_l -> layer compactor -> compact K_l, compact V_l
```

Kết quả cuối cùng là một compact cache có cùng cấu trúc layer/head/head_dim như cache thật, nhưng sequence length nhỏ hơn:

```text
8192 positions -> 1024 latent positions
```

### Bước 3: Decode Với Compact Cache

Sau khi có compact KV cache, model decode prompt/query và câu trả lời.

Quan trọng: base LLM không cần biết cache này đến từ token thật hay latent tokens. Nó chỉ nhận `past_key_values` có shape hợp lệ và tiếp tục attention như bình thường.

Luồng khái niệm:

```text
compact past_key_values + question prompt -> Qwen decode -> answer logits
```

Nếu compactor tốt, logits từ compact-cache model sẽ gần với logits của full-cache model.

### Bước 4: So Sánh Với Teacher

Teacher là cùng base Qwen nhưng dùng full KV cache.

Student là cùng base Qwen nhưng dùng compact KV cache.

Training tối ưu để student bắt chước teacher:

```text
teacher: full cache -> logits_teacher
student: compact cache -> logits_student
loss: make logits_student close to logits_teacher
```

Các loss chính:

- KL loss: ép phân phối token của student gần teacher
- reverse KL tùy chọn: thêm chiều ngược lại để giảm lệch phân phối
- CE loss tùy chọn: ép model sinh target đúng
- auxiliary letter loss: dùng cho tác vụ multiple-choice, tập trung vào token chữ cái đáp án

## Điểm Quan Trọng Trong Cách Train

### Teacher Và Student Dùng Chung Base Model

Không có hai LLM khác nhau. Teacher và student là cùng Qwen3-4B, khác nhau ở cache:

- teacher dùng full KV cache
- student dùng compact KV cache

Do đó compactor học cách làm cache nén sao cho base model vẫn hành xử giống full-context path.

### Base Model Chủ Yếu Được Freeze

Mục tiêu không phải fine-tune Qwen thành model mới. Mục tiêu là học bộ nén cache.

Điều này giúp:

- giảm số tham số train
- giảm chi phí
- giữ hành vi gốc của base model
- tách rõ bài toán nén context khỏi bài toán train LLM

### Cache Mutation Bug Đã Được Sửa

Một bug quan trọng đã được phát hiện:

- Hugging Face `DynamicCache.update()` có thể mutate cache gốc
- teacher continuation trước đây có thể làm full context cache bị append thêm prompt/target tokens
- điều này khiến compactor học từ cache không còn là context-only cache

Sau khi sửa, continuation forward dùng fresh cache wrapper để tránh mutate cache gốc.

Vì vậy các kết quả sau cache-fix đáng tin hơn các kết quả cũ.

## Vì Sao Mô Hình Có Thể Nén Được Context

Trong tác vụ hiện tại, không phải mọi token trong 8192-token context đều cần thiết cho câu trả lời. Compactor học cách gom các thông tin cần thiết vào số latent slots nhỏ hơn.

Về mặt trực giác:

- full context là bộ nhớ thô
- latent cache là bộ nhớ đã được chưng cất
- mỗi latent slot chứa hỗn hợp thông tin từ nhiều vị trí context
- decoder chỉ cần latent cache đủ tốt để ra cùng đáp án với full context

Điểm khó là latent cache phải tương thích với attention của base model. Nó không chỉ là embedding summary thông thường; nó phải có dạng key/value đúng với từng layer của transformer.

## Khác Biệt Giữa Pure Và Hybrid

### Pure Compaction

Pure setup chỉ dùng learned latent KV compactor.

Đây là hướng gần với bài Baseten/STILL:

```text
full KV cache -> learned compactor -> compact KV cache
```

Không dùng truy xuất exact anchor theo query.

### Hybrid Lexical Anchor

Hybrid setup giữ lại một số exact lexical anchors từ context, sau đó kết hợp với learned latents.

Kết quả hybrid rất cao, gần 100%, nhưng không phải pure compaction:

```text
selected exact context anchors + learned latents -> compact cache
```

Hybrid hữu ích như upper-bound/control, nhưng không nên dùng để claim kết quả pure STILL.

## Kết Quả Liên Quan Đến Architecture

Kết quả pure post-fix tốt nhất đã ghi nhận:

- checkpoint: `qwen3_4b_sec_random_visible_3k_cachefix_pure1024_identity_400step_b2_lr1e5_w05`
- step 100 compact accuracy: 0.7109375
- step 200 compact accuracy: 0.8828125
- full accuracy: 1.0
- no-context accuracy: 0.3359375
- compression: 8.0x

Điều này cho thấy architecture có tín hiệu học thật: compact cache giúp vượt xa no-context baseline và tiến gần full-context teacher trên eval slice 128.

## Sơ Đồ Tóm Tắt

```text
Long context, 8192 tokens
        |
        v
Frozen Qwen prefill
        |
        v
Full KV cache at all layers
        |
        v
Neural KV Compactor
        |
        v
Compact KV cache, 1024 latent positions
        |
        v
Frozen Qwen decode with compact cache
        |
        v
Answer logits / MCQ answer
```

Training signal:

```text
Full-cache Qwen logits    = teacher
Compact-cache Qwen logits = student

Train compactor so student logits match teacher logits.
```

## Tóm Tắt Ngắn

Architecture này là một hệ nén KV cache cho transformer. Qwen3-4B vẫn là model chính và gần như được giữ nguyên. Module train được là compactor: nó đọc full KV cache từ context dài 8192 tokens và tạo compact KV cache gồm 1024 latent positions. Sau đó Qwen decode tiếp bằng compact cache. Training dùng full-cache path làm teacher và compact-cache path làm student, tối ưu KL/CE để student bắt chước teacher.

Điểm quan trọng nhất: đây không phải là summary text thông thường. Nó học trực tiếp một cache nén ở dạng key/value của từng transformer layer, nên base LLM có thể sử dụng nó qua attention như một `past_key_values` hợp lệ.

