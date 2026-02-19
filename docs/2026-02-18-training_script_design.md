# KawaiiLLM Training Script Design

**Context**
Data formatting for incremental pretraining is complete (33GB across 7 sources). Now we need training scripts to implement the core architecture: **MemE (Qwen3-Embedding-4B) → Projector → LLM (Qwen3-8B-Base)**, enabling the LLM to read compressed memory vectors and reconstruct/continue text.

> **Hardware:** 8× A800 80GB single node.
> **Framework:** HF Trainer + DeepSpeed ZeRO-2.

---

## File Structure

```text
src/train/
├── __init__.py
├── arguments.py       # CLI argument dataclasses
├── model.py           # KawaiiLLM model (MemE + Projector + LLM)
├── dataset.py         # Dataset with byte-offset index + curriculum sampling
├── collator.py        # DataCollator (pad + sample n_mem per batch)
├── trainer.py         # KawaiiTrainer (custom save/load, curriculum callback)
├── train.py           # Entry point
├── build_index.py     # Standalone: scan formatted JSONL → index file
configs/
├── ds_zero2.json      # DeepSpeed ZeRO-2 config for 8×A800
└── train_8xa800.sh    # Launch script
```

---

## Step 1: Arguments (`src/train/arguments.py`)

Three `@dataclass` groups parsed by `HfArgumentParser`:

### ModelArguments

* `meme_model_name_or_path`: `str` — Path to Qwen3-Embedding-4B
* `llm_model_name_or_path`: `str` — Path to Qwen3-8B-Base
* `num_mem_tokens`: `int = 128` — Max MEM tokens (`N_max`)
* `freeze_meme` / `freeze_llm` / `freeze_projector`: `bool = False`
* `meme_lr`: `Optional[float] = None` — Per-component LR for MemE (falls back to global LR)
* `llm_lr`: `Optional[float] = None` — Per-component LR for LLM
* `projector_lr`: `Optional[float] = None` — Per-component LR for projector + mem_embeddings

### DataArguments

* `data_dirs`: `List[str]` — 7 formatted directories
* `index_path`: `Optional[str]` — Pre-built index JSON path
* `context_max_length`: `int = 4096`
* `target_max_length`: `int = 4096`

### TrainingArguments (extends `transformers.TrainingArguments`)

* `remove_unused_columns`: `bool = False` (**Critical** — prevents Trainer from dropping custom fields)
* `gradient_checkpointing`: `bool = True`

---

## Step 2: Build Index (`src/train/build_index.py`)

Standalone script. Scans all formatted JSONL files in binary mode (`rb`) and records byte offset per line.
Scanning 33GB JSONL ≈ 60s I/O. Output index file ~200-400MB.

**Output Format** (`data/train_index.json`):

```json
{
  "entries": [
    {"source": "novels", "id": "123", "split": 0, "tokens": 4056, "file": "/abs/path.jsonl", "offset": 0},
    ...
  ],
  "continuation_pairs": [[idx_A, idx_B], ...],
  "total_entries": N,
  "total_continuation_pairs": M
}
```

* `continuation_pairs`: pairs where `B.split == A.split + 1`, same `source` + `id`.

**CLI Command:**

```bash
python src/train/build_index.py \
  --data_dirs data/novels/formatted data/bilibili/formatted ... \
  --output_path data/train_index.json
```

---

## Step 3: Model (`src/train/model.py`) — `KawaiiLLMModel`

Plain `nn.Module` wrapping three components (not a `PreTrainedModel` subclass — avoids config registration complexity for a multi-model composition).

**Qwen3-Embedding-4B 实际参数 (已验证):**

| 属性 | Qwen3-Embedding-4B (MemE) | Qwen3-8B-Base (LLM) |
| :--- | :--- | :--- |
| `hidden_size` | **2560** | **4096** |
| `num_hidden_layers` | 36 | 36 |
| `num_attention_heads` | 32 | 32 |
| `num_key_value_heads` | 8 | 8 |
| `intermediate_size` | 9728 | 12288 |

> **注意**: 原计划中 hidden_size 写为 3584（那是 Qwen2.5-7B 的值），已根据 HuggingFace config.json 修正为 2560。代码中通过 `self.meme.config.hidden_size` 动态读取，不硬编码。

### Special Tokens

动态注册以下特殊 token，并 resize MemE 和 LLM 的 embedding 层：

| Token | 作用 | 所在侧 |
| :--- | :--- | :--- |
| `<mem>` / `</mem>` | 标记 memory token 序列的开始/结束边界 | MemE + LLM |
| `<mempad>` | 占位符，运行时被替换为可学习的 Memory Query Embeddings | MemE + LLM |
| `<AE>` | 提示 LLM 进入"文本重建"模式 | LLM |

```python
SPECIAL_TOKENS = ["<mem>", "<mempad>", "</mem>", "<AE>"]
```

### Components

```python
class KawaiiLLMModel(nn.Module):
    self.meme            # AutoModel.from_pretrained("Qwen3-Embedding-4B")  [4B, hidden=2560]
    self.mem_embeddings  # nn.Embedding(128, meme_hidden) — learnable MEM tokens, init N(0, 0.02)
    self.projector       # RMSNorm(meme_hidden) → Linear(meme_hidden, meme_hidden*4) → GELU → Linear(meme_hidden*4, llm_hidden)
    self.llm             # AutoModelForCausalLM.from_pretrained("Qwen3-8B-Base") [8B, hidden=4096]
```

> **Projector 设计**: 参考 PCC Converter + Qwen2.5-VL Merger 的共识设计。RMSNorm 稳定输入分布，4× expansion 提供足够表达能力。维度从 `model.config.hidden_size` 动态读取。

### Forward Pass Logic

1. **Encode Context (`encode_context`)**
```python
# 构建 MemE 输入: [text, <mem>, Q_1, ..., Q_n, </mem>]
text_embeds   = meme.get_input_embeddings()(context_ids)      # [B, L, meme_hidden]
mem_start_emb = meme.get_input_embeddings()(mem_token_id)      # [1, meme_hidden]
mem_end_emb   = meme.get_input_embeddings()(mem_end_token_id)  # [1, meme_hidden]
mem_embeds    = mem_embeddings.weight[:n_mem].expand(B)         # [B, n, meme_hidden]

combined = cat([text_embeds, mem_start_emb, mem_embeds, mem_end_emb], dim=1)
# [B, L+1+n+1, meme_hidden]

# 若 freeze_meme=True，在 torch.no_grad() 下运行，输出 detach 后 requires_grad_(True)
outputs     = meme(inputs_embeds=combined, attention_mask=extended_mask)
# 提取 MEM token 位置的 hidden states (跳过 <mem>，取 n 个 Q，跳过 </mem>)
mem_hidden  = outputs.last_hidden_state[:, -(n_mem+1):-1, :]  # [B, n, meme_hidden]
```

2. **Project**
```python
projected = projector(mem_hidden)  # [B, n, llm_hidden]
```

3. **Decode with LLM**

`<AE>` 信号由 dataset 层放入 `input_ids`（重建任务在 target_ids 前加 `<AE>`），model.forward() 使用固定 prefix，无需 task_type 分支：

```python
# LLM prefix: 固定结构 [<mem>, h_1, ..., h_N, </mem>]
prefix = cat([mem_start_emb, projected, mem_end_emb], dim=1)

# input_ids 由 dataset 构造：
#   重建: [<AE>, t1, t2, ..., tn]  (labels: [IGNORE, t1, t2, ..., tn])
#   续写: [t1, t2, ..., tn]        (labels: [t1, t2, ..., tn])
target_embeds = llm.get_input_embeddings()(input_ids)
llm_input = cat([prefix, target_embeds], dim=1)

labels = cat([IGNORE × prefix_len, target_labels], dim=1)
return llm(inputs_embeds=llm_input, attention_mask=..., labels=labels)
```

> **Labels 对齐说明**: HF 的 `ForCausalLM.forward()` 内部会自动做 `shift_labels = labels[..., 1:]`。因此 `cat([IGNORE*prefix_len, target_labels])` 的效果是：prefix 最后一个 token（`</mem>`）学习预测第一个 target token（重建时为 `<AE>`，续写时为 `t1`），无需手动移位。
>
> **`<AE>` per-sample 设计**: 将 `<AE>` 放入 `input_ids` 而非在 model.forward() 中按 batch 注入。这样同一 batch 中可以混合重建和续写样本，无需 grouped batch sampler。

**Additional Methods:**

* `gradient_checkpointing_enable()`: 仅对未冻结的组件启用（frozen MemE 跳过）。
* `gradient_checkpointing_disable()`: 同上，仅对未冻结组件操作。
* `enable_input_require_grads()`: gradient checkpointing 与冻结参数共存所必需。
* `config` property: 返回 `self.llm.config`（HF Trainer 兼容）。
* `dtype` / `device` properties: 代理到 LLM。
* `save_checkpoint(output_dir)`: 分别保存 meme/、projector/、llm/。
* `from_checkpoint(cls, ...)`: 从保存的 checkpoint 目录加载。

---

## Step 4: Dataset (`src/train/dataset.py`) — `KawaiiDataset`

**Two Tasks:**

1. **Reconstruction (AE):** `context` = text → MemE → latent → LLM（带 `<AE>` 信号）→ `same text`
2. **Continuation (AR):** `split_x` → MemE → latent → LLM → `split_{x+1}`

### Task Assignment — 线性 warmup + 合成续写

续写概率在训练前 10% 线性增长到 50%，之后保持 50%。如果选择续写但无自然续写对，则在 `\n` 处切分当前文本合成续写对：

```python
def _get_task_type(idx, progress):
    if progress < 0.1:
        cont_prob = 0.5 * (progress / 0.1)  # 0% -> 50%
    else:
        cont_prob = 0.5

    if random.random() < cont_prob:
        return "continuation"
    return "reconstruction"
```

续写来源（在 `__getitem__` 中处理）：
1. **自然续写**: 有 continuation pair → 使用下一 chunk
2. **合成续写**: 无 continuation pair → 在 `\n` 处切分当前文本（优先选最接近中点的 `\n`）

效果：50% 续写比例中，有自然续写对的样本使用真实对，无续写对的样本使用合成切分。所有样本都能参与续写任务。

**`<AE>` per-sample 实现**: 重建任务在 `input_ids` 前端插入 `<AE>` token（label 为 IGNORE），续写任务不加。这样每个样本独立携带任务信号，无需 per-batch task_type。

### Data Access

* Load index from `data/train_index.json`.
* **Random Access:** 使用 binary mode (`rb`) 打开文件，`file.seek(offset)` + `readline()` + `.decode("utf-8")`，确保与 build_index.py 的字节偏移一致。
* **File Handle Cache:** Per-file，带 `__del__` 清理。
* **Worker Safety:** 提供 `worker_init_fn` 静态方法，DataLoader 每个 worker 独立重置文件句柄并 seed RNG。
* `set_training_progress(float)`: Called by trainer callback.
* **`__getitem__` returns:** `{"context_ids": Tensor, "input_ids": Tensor, "labels": Tensor}`.
* *Note: `n_mem` is NOT sampled per-sample — it is sampled per-batch in the collator. `task_type` 不在返回 dict 中——通过 `<AE>` token 在 input_ids 中隐式编码。*

---

## Step 5: Collator (`src/train/collator.py`) — `KawaiiDataCollator`

* **`context_ids`:** Left-padded (matching MemE's `padding_side='left'`), ensuring MEM tokens always sit at the far right.
* **`input_ids` / `labels`:** Right-padded (standard causal LM convention).
* **Samples single `n_mem`** for the entire batch from **线性插值**课程分布（先多后少）。
* **Attention mask 基于实际长度构建**（而非 `.ne(pad_token_id)`），避免当 `pad_token == eos_token` 时真实 EOS token 被错误 mask 的问题。
* Returns dict with all fields + `n_mem` (scalar int).

**n_mem 采样 — 先多后少，线性插值（无阶梯跳变）:**

```python
def _sample_n_mem(self) -> int:
    p = self._training_progress  # 0 → 1
    N = self.num_mem_tokens_max  # 128

    # 各区间采样概率随 p 线性变化
    p_high   = 0.7 - 0.6 * p   # P(64-128): 70% → 10%
    p_mid    = 0.2 + 0.1 * p   # P(16-64):  20% → 30%
    p_low    = 0.1 + 0.2 * p   # P(2-16):   10% → 30%
    p_single = 0.3 * p         # P(1):       0% → 30%

    r = random.random()
    if r < p_high:
        return random.randint(64, N)
    elif r < p_high + p_mid:
        return random.randint(16, 64)
    elif r < p_high + p_mid + p_low:
        return random.randint(2, 16)
    else:
        return 1
```

**Attention Mask 实现 (length-based, 非 token identity-based):**

```python
# Context: 左填充，根据原始长度从右端标记为 1
context_attention_mask = torch.zeros_like(context_ids, dtype=torch.long)
for i, ids in enumerate(context_ids_list):
    real_len = ids.shape[0]
    context_attention_mask[i, max_ctx_len - real_len:] = 1

# Target: 右填充，根据原始长度从左端标记为 1
attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
for i, real_len in enumerate(target_lengths):
    attention_mask[i, :real_len] = 1
```

---

## Step 6: Trainer (`src/train/trainer.py`) — `KawaiiTrainer`

Extends `transformers.Trainer`.

### CurriculumCallback

```python
def on_train_begin(self, args, state, control, **kwargs):
    # 从 checkpoint 恢复时，根据 global_step/max_steps 恢复 curriculum progress
    if state.global_step > 0 and state.max_steps > 0:
        progress = state.global_step / state.max_steps
        self.dataset.set_training_progress(progress)
        self.collator.set_training_progress(progress)

def on_step_begin(self, args, state, control, **kwargs):
    progress = state.global_step / state.max_steps
    self.dataset.set_training_progress(progress)
    self.collator.set_training_progress(progress)
```

### KawaiiTrainer Logic

* `get_train_dataloader()`: 注入 `KawaiiDataset.worker_init_fn`，确保多 worker 下文件句柄独立。
* `compute_loss()`: Extracts `n_mem` from inputs, calls `model.forward()` with all fields.
* `create_optimizer()`: **Per-component learning rate groups** — 为 meme、llm、projector、mem_embeddings 分别创建 optimizer param groups，区分 weight_decay 和 no_decay 参数。
* `_save()`: Unwrap DeepSpeed/DDP wrapper, 调用 `model.save_checkpoint()` 分别保存组件 + tokenizer:

```text
checkpoint-N/
├── meme/           # meme.save_pretrained()
├── projector/      # projector.pt + mem_embeddings.pt
├── llm/            # llm.save_pretrained()
└── tokenizer files
```

---

## Step 7: Entry Point (`src/train/train.py`)

**Flow:** `parse args` → `load tokenizer` → **`register special tokens`** → `verify tokenizer compatibility` → `build/load model` → **`resize embeddings`** → `enable gradient checkpointing` → `build dataset` → `build collator` → `CurriculumCallback` → `KawaiiTrainer` → `train` → `save`.

* **Tokenizer:** Load from LLM path (Qwen3 family shared tokenizer), set `pad_token = eos_token`.
* **Special Token 注册:** 调用 `tokenizer.add_special_tokens({"additional_special_tokens": ["<mem>", "<mempad>", "</mem>", "<AE>"]})` 注册 4 个特殊 token，并 resize MemE 和 LLM 的 embedding 层。
* **Tokenizer 兼容性验证:** 加载 MemE tokenizer 对比 vocab_size，不匹配时打印 warning。
* **Embedding Resize:** `model.meme.resize_token_embeddings(len(tokenizer))` 和 `model.llm.resize_token_embeddings(len(tokenizer))`。
* **Checkpoint Resume:** Detect `checkpoint-*` dirs, call `KawaiiLLMModel.from_checkpoint()`, Trainer 同时传入 `resume_from_checkpoint`。
* **Gradient Checkpointing:** 启用时同时调用 `enable_input_require_grads()` 以兼容冻结参数。
* **Logging:** Log total/trainable param counts.

---

## Step 8: Config (`configs/ds_zero2.json`)

```json
{
  "bf16": {"enabled": true},
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1000000000,
    "reduce_bucket_size": "auto",
    "allgather_bucket_size": 500000000
  },
  "gradient_clipping": 1.0
}
```

**Memory Analysis (8× A800 80GB, ZeRO-2, bf16):**

* **Model weights:** ~12B parameters × 2 bytes ≈ **24GB** (full copy per GPU).
* **Optimizer states:** ~48GB / 8 GPUs ≈ **6GB**.
* **Gradients:** ~24GB / 8 GPUs ≈ **3GB**.
* **Static Total:** ~33GB per GPU.
* **MemE 激活 (gradient checkpointing):** 36 layers × `[2, 4224, 2560]` checkpoint input ≈ **1.5GB**.
* **LLM 激活 (gradient checkpointing):** 36 layers × `[2, 4224, 4096]` checkpoint input ≈ **2.5GB**.
* **Remaining:** ~40GB headroom (safe with gradient checkpointing).
* **Effective Batch Size:** 2 per GPU × 8 accum × 8 GPUs = **128**.

---

## Step 9: Launch Script (`configs/train_8xa800.sh`)

```bash
deepspeed --num_gpus 8 src/train/train.py \
  --deepspeed configs/ds_zero2.json \
  --meme_model_name_or_path /path/to/Qwen3-Embedding-4B \
  --llm_model_name_or_path /path/to/Qwen3-8B-Base \
  --data_dirs data/novels/formatted data/bilibili/formatted \
    data/moegirl/formatted data/games/formatted \
    data/general/formatted data/math/formatted data/code/formatted \
  --index_path data/train_index.json \
  --output_dir output/kawaii_v1 \
  --num_mem_tokens 128 \
  --freeze_meme False \
  --projector_lr 1e-3 \
  --meme_lr 1e-6 \
  --llm_lr 1e-6 \
  --bf16 True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --gradient_checkpointing True \
  --save_strategy steps --save_steps 2000 --save_total_limit 3 \
  --logging_steps 10 \
  --dataloader_num_workers 4 \
  --report_to wandb \
  --run_name kawaii_v1
```

> **Phase 1 = 全参数训练**: `--freeze_meme False`，MemE backbone 参与训练但使用极低学习率 (`1e-6`)。

---

## Verification Plan

1. **Smoke test:** 100-entry test JSONL, single GPU, 10 steps, `n_mem=1` — verify forward pass completes.
2. **Gradient flow:** Check `mem_embeddings.weight.grad`, `projector[0].weight.grad`, MemE/LLM grads are non-zero after backward.
3. **Loss sanity:** Initial loss should be `≈ ln(151936) ≈ 11.93`; anything much higher or NaN is a bug.
4. **Checkpoint round-trip:** Save → Load → Same input → Verify loss matches.
5. **Curriculum logging:** Log `n_mem` and task type per step, verify distribution matches phase.
6. **Memory profiling:** `torch.cuda.max_memory_allocated()` per GPU, ensure no OOM at `batch_size=2`.
7. **Multi-GPU:** Test on 2 GPUs first, then scale to 8.

## Known Risks

1. **Tokenizer compatibility:** 代码中已加入 vocab_size 校验。MemE (Qwen3-Embedding-4B) 的 vocab_size 可能为 151665 而非 LLM 的 151936，需在实际加载时确认影响。
2. **DeepSpeed with nn.Module:** HF Trainer expects `model.config`. This is solved by providing a `@property` returning `self.llm.config`.
3. **MemE Padding Side:** 使用左填充。Collator 中 attention mask 基于实际长度构建，避免 `pad_token == eos_token` 导致的 mask 错误。
4. **EOS == PAD 问题:** 因为 `pad_token = eos_token`，若使用 `.ne(pad_token_id)` 构建 attention mask 会将数据中真实 EOS token 误标为 padding。实现中改为基于序列实际长度构建 mask。
5. **多 Worker 文件句柄:** DataLoader fork 后子进程共享父进程文件句柄。通过 `worker_init_fn` 在每个 worker 中重置文件句柄和 RNG seed。
6. **Curriculum 恢复:** 从 checkpoint 恢复训练时，`CurriculumCallback.on_train_begin` 根据 `global_step / max_steps` 恢复 curriculum progress，避免前几个 batch 退回 early phase。

---

## 计划 vs 实现差异总结

| 项目 | 原计划 | 实际实现 | 原因 |
| :--- | :--- | :--- | :--- |
| MemE hidden_size | 3584 | **2560** (动态读取) | 原计划误用了 Qwen2.5-7B 的值；代码通过 `meme.config.hidden_size` 动态获取 |
| **Special tokens** | 无 | `<mem>`, `</mem>`, `<mempad>`, `<AE>` 动态注册 + embedding resize | README 设计要求区分任务模式和标记 memory 边界 |
| **Projector 架构** | `Linear→GELU→Linear` | `RMSNorm(meme_dim) → Linear(meme_dim, meme_dim*4) → GELU → Linear(meme_dim*4, llm_dim)` | 参考 PCC/Qwen2.5-VL 共识设计，RMSNorm 稳定输入 |
| **n_mem 课程** | 3-phase 阶梯 (先少后多) | 线性插值 (先多后少): P(64-128)=0.7-0.6p, P(1)=0.3p | README "先多后少" 设计 + 避免阶梯跳变 |
| **续写任务分配** | 概率采样 (30%/50%) | 线性 warmup (0→50% over 10%) + 合成续写（无自然对时按 `\n` 切分） | 所有样本均可续写；`<AE>` 放入 input_ids 实现 per-sample 任务信号 |
| **Phase 1 freeze_meme** | `--freeze_meme True` | `--freeze_meme False` | README 阶段 1 = 全参数训练，MemE 用极低 LR |
| Per-component LR | 未包含 | `meme_lr`, `llm_lr`, `projector_lr` in `ModelArguments`; `create_optimizer()` 分组实现 | 随机初始化组件和预训练组件需差异化 LR |
| Frozen MemE 处理 | `gradient_checkpointing_enable()` 对所有组件 | Frozen MemE 在 `torch.no_grad()` 下运行，跳过 gradient checkpointing；输出 `detach().requires_grad_(True)` | 冻结模型无需 checkpointing |
| Attention mask 构建 | `.ne(pad_token_id)` | 基于实际序列长度构建 | `pad_token == eos_token` 时 `.ne()` 会把真实 EOS 误标为 padding |
| 文件 I/O 模式 | 未明确 | build_index.py 用 `rb`，dataset.py 也用 `rb` + `.decode()` | 字节偏移必须在同一模式下才正确 |
| Worker 安全 | 仅提到 "forks independently" | `__del__` 清理 + `worker_init_fn` 重置句柄/seed + trainer `get_train_dataloader()` 注入 | Fork 后文件句柄实际共享，需显式重置 |
| Curriculum 恢复 | 未涉及 | `CurriculumCallback.on_train_begin` 从 `global_step/max_steps` 恢复 | 防止 checkpoint 恢复后 curriculum 重置为 0 |
| 显存估算 | 剩余 ~47GB | 修正为 ~40GB（加入 MemE/LLM 激活估算各 1.5/2.5GB） | 原计划遗漏 gradient checkpointing 下的激活开销 |

## Implementation Order

1. `arguments.py` — Straightforward dataclasses.
2. `build_index.py` — Standalone, testable immediately.
3. `model.py` — Core architecture, test with dummy input.
4. `dataset.py` + `collator.py` — Data pipeline.
5. `trainer.py` — Custom save/load + curriculum callback.
6. `train.py` — Wire everything together.
7. `configs/` — DeepSpeed config + launch script.
8. Smoke test on single GPU.
9. Multi-GPU validation.
