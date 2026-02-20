# KawaiiLLM 训练 NaN 调试历史

## 问题描述

**现象**：第 0 步所有微批次 loss 正常（0.28–3.29），logits 无 NaN/Inf；第 1 步（第一次 optimizer.step() 之后）所有微批次 loss = NaN，logits has_nan = True。

**环境**：8×A800 80GB，DeepSpeed ZeRO-2，bf16，gradient_accumulation_steps=8，freeze_meme=False。

---

## 诊断时间线

### 阶段 1 — 初始症状：loss = 1,415,945.2，grad_norm = nan

训练日志显示 loss 极大，grad_norm 为 nan。

**初步分析**：可能是 logits 过大或 labels 对齐错误。

**已应用修复**：
1. Projector 最后一层使用接近零的初始化（std=1e-4），防止初始投影值过大
2. `encode_context` 中前缀 attention mask 由 0 改为 1（避免 dummy_mem 完全被 mask 掉）
3. 边界 label masking（prefix memory token 位置 label 设为 -100）

**结论**：修复后 step 0 的 loss 恢复正常（0.28–3.29），但 step 1 仍然全部 NaN。

---

### 阶段 2 — 精确定位：step 0 健康，step 1 全部 NaN

在 `compute_loss` 中添加 debug 日志（前 3 个 optimizer step），输出：

```
DEBUG step=0 | loss=X.XXXX | logits shape=... dtype=bfloat16 min=... max=... has_nan=False ...
DEBUG step=1 | loss=nan   | logits shape=... dtype=bfloat16 min=nan max=nan has_nan=True  ...
```

**结论**：第一次 optimizer.step() 破坏了模型权重。问题在 backward/optimizer 中，不在 forward。

---

### 阶段 3 — 假设 1：左填充 context 导致 MemE 中 softmax NaN（已排除）

**假设路径**：
- MemE 的 context 使用左填充（padding_side='left'），position 0 是 padding
- `extended_mask[:, 0] = 0` → 因果注意力中 position 0 只能关注自身但被 mask 掉
- softmax 输入全为 -inf → softmax 输出 NaN
- backward：`softmax_backward = output * (grad - sum(grad * output))`，NaN * 0 = NaN（IEEE 754）

**修复尝试**：`extended_mask[:, 0] = 1`（commit 8cc444a）

**结果**：step 1 依然全部 NaN。**该假设已排除。**

---

### 阶段 4 — 假设 2：禁用 MemE GC → CUDA OOM（已排除）

**假设路径**：Qwen3-Embedding 自定义 attention + GC re-run 产生 NaN 梯度。

**修复尝试（commit d4f7adc）**：禁用 MemE gradient checkpointing，同时添加 NaNDetectorCallback 和梯度 NaN hook。

**实际结果**：CUDA OOM。禁用 MemE GC 后激活值全部驻留显存，backward 申请 3.32 GiB 时 OOM。诊断代码根本没机会运行。

**结论**：该方向已排除，MemE GC 已恢复（commit 13626c3）。

---

### 阶段 5 — 梯度 NaN hook 诊断：定位到 LLM backward（已确认）

**commit 13626c3** 恢复 MemE GC 并保留所有诊断代码后的运行结果：

**关键 hook 输出**（触发顺序即反向拓扑顺序）：

```
fwd=1: llm_input NaN  → 最先触发，NaN 起源于此
fwd=1: projected NaN  → llm_input NaN 沿 prefix 路径向上游传播
fwd=1: mem_hidden NaN → projected NaN 继续向上游传播至 MemE 输出
fwd=1: target_embeds NaN → llm_input NaN 沿 target 路径向上游传播
fwd=2: NTP_dummy_proj NaN → NTP 路径也受影响（与 MemE 无关）
fwd=2: NTP_combined_embeds NaN → NTP 路径也受影响
```

**推断链**（NaN 流向为 backward 方向）：
```
LLM backward 内部产生 NaN
        ↓
llm_input 梯度 = NaN
        ↓                    ↓
prefix 路径: projected → mem_hidden     target 路径: target_embeds
```

NTP 路径（最简单的 LLM forward，无复杂 prefix）也出现 NaN，排除了 prefix 结构本身是问题所在。
**结论：NaN 起源于 LLM backward 内部，与 MemE / projector 无关。**

---

### 阶段 6 — 假设 3：Flash Attention + reentrant GC 的兼容性问题（已排除）

**假设路径**：

Qwen3-8B 默认使用 Flash Attention。Flash Attention 的 backward kernel 需要 forward 时保存的 `log_sum_exp` 张量来计算注意力权重的梯度。当 gradient checkpointing 使用 `use_reentrant=True` 时，GC re-run 的 `log_sum_exp` 可能与原始 forward 的值不一致 → NaN 梯度。

**修复尝试（commit 539502c）**：

1. **`attn_implementation="eager"`**（model.py，MemE 和 LLM 加载时均添加）：关闭 Flash Attention，使用标准 PyTorch attention（`torch.matmul` + `F.softmax`）
2. **`use_reentrant=False`**（train.py，`gradient_checkpointing_enable` 调用时传入）：GC 改用非重入模式

**运行结果**：NaN 模式完全不变。所有 8 个 GPU 的第一次 backward（fwd=1）均产生 NaN：

```
GRAD_NaN fwd=1 llm_input shape=[2, 2314, 4096] nan=True inf=False
GRAD_NaN fwd=1 llm_input shape=[2, 3008, 4096] nan=True inf=False
... (全部 8 个 GPU)
GRAD_NaN fwd=1 projected shape=[2, 125, 4096] nan=True inf=False
GRAD_NaN fwd=1 mem_hidden shape=[2, 113, 2560] nan=True inf=False
GRAD_NaN fwd=1 target_embeds shape=[2, 2890, 4096] nan=True inf=False
... (全部 8 个 GPU)
GRAD_NaN fwd=2 NTP_combined_embeds shape=[2, 3394, 4096] nan=True inf=False
GRAD_NaN fwd=2 NTP_dummy_proj shape=[1, 1, 4096] nan=True inf=False
```

Step 0 所有微批次 forward 仍然健康（loss 1.66–3.17，无 NaN/Inf）。

**代码级确认（排除假设的依据）**：

通过审查 transformers 源码，确认该假设不成立：

1. **Eager attention 确实生效**：Qwen3 的 `Qwen3Attention.forward` 通过 `ALL_ATTENTION_FUNCTIONS.get_interface("eager", eager_attention_forward)` 正确分派到 `eager_attention_forward`，使用 `torch.matmul` + `F.softmax(dtype=torch.float32)` 计算，**不使用** `F.scaled_dot_product_attention`（源码 `modeling_qwen3.py:195-218`）
2. **`use_reentrant=False` 确实生效**：`GradientCheckpointingLayer.__call__` 通过 `self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)` 调用 `torch.utils.checkpoint.checkpoint(use_reentrant=False)`（源码 `modeling_layers.py:92`）
3. **DeepSpeed bf16 不使用 loss scaling**：`CreateLossScaler` 对非 fp16 类型始终返回 `loss_scale=1.0`，并有 `assert self.loss_scaler.cur_scale == 1.0` 验证（源码 `loss_scaler.py:216`, `stage_1_and_2.py:589-592`）

**结论：Flash Attention + reentrant GC 假设已排除。** eager attention + use_reentrant=False 未改变任何行为，说明根因不在 attention 实现或 GC 重入模式。

---

### 阶段 7 — 方案 A 结果：禁用 GC 后 NaN 仍在（GC 已排除）

**运行**：`bash scripts/train_debug_no_gc.sh`（Qwen3-Embedding-0.6B + Qwen3-4B-Base，freeze_meme=True，gradient_checkpointing=False）

**结果**：NaN 梯度从第一次 backward（fwd=1）起在所有 8 个 GPU 上持续出现。GC 不是根因。

```
GRAD_NaN fwd=1 NTP_combined_embeds shape=[1, 1950, 2560] nan=True inf=False
GRAD_NaN fwd=1 NTP_combined_embeds shape=[1, 2714, 2560] nan=True inf=False
GRAD_NaN fwd=1 NTP_dummy_proj shape=[1, 1, 2560] nan=True inf=False
...
GRAD_NaN fwd=7 llm_input shape=[1, 1347, 2560] nan=True inf=False norm=nan
GRAD_NaN fwd=7 projected shape=[1, 10, 2560] nan=True inf=False norm=nan
GRAD_NaN fwd=7 mem_hidden shape=[1, 10, 1024] nan=True inf=False norm=nan
GRAD_NaN fwd=7 target_embeds shape=[1, 1335, 2560] nan=True inf=False norm=nan
```

同时 forward 完全健康（loss 0.25–4.3，logits 无 NaN/Inf）。

---

### 阶段 8 — 根因确认：零嵌入 + mask=1 导致 RMSNorm 梯度爆炸（已修复）

**已排除的全部假设**：

| 假设 | 排除依据 |
|------|----------|
| Flash Attention backward 不兼容 GC | eager attention 同样 NaN |
| GC `use_reentrant=True` 的 re-run 不一致 | `use_reentrant=False` 同样 NaN |
| Loss scaling 导致梯度溢出 | bf16 下 loss_scale = 1.0，无放大 |
| Softmax 输入全为 -inf（全 mask 行） | `prefix_attn[:, 0] = 1` 保证每行至少有 1 个有效 key |
| Packed sequence 检测误触发 | 仅当 `attention_mask is None` 时触发，而我们传递了 attention_mask |
| MemE / projector 是 NaN 源头 | hook 顺序和 NTP 路径均排除 |
| **Gradient Checkpointing 导致 NaN** | **禁用 GC 后 NaN 仍在（方案 A 结果）** |

**根因分析**：

NTP 路径和 mixed 路径中，LLM 输入的 padding 位置使用 **全零嵌入** + `mask=1`：

```python
# NTP 路径 (model.py:373)
prefix_embeds = target_embeds.new_zeros(B, 1, llm_hidden)  # 全零！

# Mixed 路径 (model.py:483)
torch.zeros(max_prefix_len, llm_hidden, ...)  # 全零！

# 两种路径都设置 mask=1：
prefix_mask = attention_mask.new_ones(B, 1)  # NTP
prefix_attn[:, 0] = 1                         # Mixed
```

**NaN 产生机制**（4 步级联）：

1. **零输入传播**：position 0 全零嵌入 → 因果注意力只能自注意 → Q=K=V=0 → 输出=0 → 残差=0 → MLP(0)=0 → 逐层传播，position 0 在所有 36 层均为零向量。

2. **RMSNorm 梯度放大**：RMSNorm backward 在 x=0 处的梯度为 `dy/dx = weight / sqrt(eps)`，eps=1e-6 → **每层放大 1000×**。

3. **跨层指数爆炸**：position 0 的梯度来自 position 1+ 的注意力权重对 V[0] 的需求。梯度经过残差流逐层累积 RMSNorm 放大，约 1000^L 量级。36 层后：1000^12 > 3.4e38（bfloat16 max）→ 溢出为 **inf**。

4. **inf × 0 = NaN**：attention backward 计算 `d_loss/d_alpha[j,0] = d_loss/d_output[j] · V[0]^T`。`d_loss/d_output[j]` 包含 inf（从 step 2-3），`V[0] = 0`（零输入的 value）→ **inf × 0 = NaN**（IEEE 754）。NaN 进入 softmax backward 后污染**所有**位置的梯度。

**修复**：将所有 mask=1 的 padding 位置从 `zeros()` 替换为 `llm_embed(pad_token_id).detach()`。pad token 的学习嵌入非零 → RMSNorm backward 的放大倍数有界（≈ 1/norm ≈ O(1)），不会爆炸。

**修改的文件**：
- `src/train/model.py`：NTP 路径和 mixed 路径的 prefix 均改用 pad embedding
- `src/train/train.py`：向 `set_special_token_ids` 传递 `pad_token_id`

---

## 下一步

1. **验证修复**：重新运行 `bash scripts/train_debug_no_gc.sh`，确认 GRAD_NaN hook 不再触发
2. **恢复完整训练**：确认修复后，使用 `configs/train_8xa800.sh` 启动正式训练（含 GC）
3. **清理诊断代码**：训练稳定后删除 `_grad_hook_count`、debug logging 等（见下方清理清单）

---

## 验证标准

训练成功的标志：

```
# 1. 梯度 hook 不应出现 NaN
# 不应看到：GRAD_NaN fwd=X ...

# 2. NaNDetectorCallback 应报告正常
All parameters clean after step 0

# 3. DEBUG step=1 应有正常 loss（不应是 nan）
DEBUG step=1 | loss=X.XXXX | has_nan=False has_inf=False ...

# 4. grad_norm 应有限
{'loss': X.XX, 'grad_norm': Y.YY, 'learning_rate': ..., 'epoch': ...}
```

---

## 关键代码位置

| 文件 | 关键行为 |
|------|---------|
| `src/train/model.py:64` | MemE 加载 |
| `src/train/model.py:73` | LLM 加载 |
| `src/train/model.py:385-389` | **NTP prefix：pad embedding 替代零嵌入（NaN 修复）** |
| `src/train/model.py:477-482` | **Mixed prefix：pad embedding 替代零嵌入（NaN 修复）** |
| `src/train/model.py:265` | `extended_mask[:, 0] = 1`：防止左填充 position 0 的 softmax NaN |
| `src/train/trainer.py:19` | `NaNDetectorCallback`：step 1 后检查参数 NaN |
| `src/train/trainer.py:108` | `compute_loss()`：debug 日志（前 3 个 optimizer step） |

---

## Commit 历史

| Commit | 说明 | 结果 |
|--------|------|------|
| 8cc444a | 修复 encode_context 中 extended_mask position 0 | step 1 仍 NaN，假设 1 排除 |
| d4f7adc | 禁用 MemE GC + NaNDetectorCallback + 梯度 NaN hook | OOM，假设 2 排除，诊断代码就位 |
| 13626c3 | 恢复 MemE GC（fix OOM），保留诊断代码 | NaN 复现，hook 成功采集数据 |
| 539502c | `attn_implementation="eager"` + `use_reentrant=False` | NaN 模式不变，假设 3 排除 |
| 6f24817 | 移除 `attn_implementation="eager"`（已排除，且导致 OOM） | 消除 eager attention OOM |
| c7892e6 | 根因修复：零嵌入 → pad embedding（NTP path + 混合 path padding） | 部分修复：NTP path 干净，但 n_mem>0 的混合 path 仍有 GRAD_NaN |

---

### 阶段 9 — 二次定位：近零 projected tokens 的 RMSNorm 梯度放大

**输入**：应用 pad-embedding 修复后的训练日志（`scripts/train_debug_no_gc.sh`）

**观察**：
- Forward 全部干净（所有 logits 有限，无 NaN/Inf）
- GRAD_NaN 仅在 `fwd=7`（n_mem=10）触发，128 个微批次中仅 1 个
- NaN 出现在 `target_embeds` 梯度中 — 说明 NaN 源于 LLM backward 内部
- eager attention 已在阶段 5 排除（commit 539502c），Flash Attention 不是原因

**根因分析**：

Projector 最后一层 init std=1e-4，产生输出 std ≈ 0.003。正常 LLM embedding RMS ≈ 0.02。

RMSNorm backward 梯度放大：`d_rmsnorm/d_x ∝ 1/RMS(x)`
- 正常 embedding (RMS ≈ 0.02)：放大 ≈ 50×
- 近零 projected (RMS ≈ 0.003)：放大 ≈ 333×（7× 高于正常值）

虽然残差连接阻止了跨层乘性放大，但 7× 的额外放大因子对特定数据模式在 bfloat16 精度下足以触发 NaN（1/128 的概率说明是边界情况）。

**修复：残差嵌入初始化 (Residual Embedding Init)**

在 projected tokens 上叠加 detached 的 pad token embedding 作为常数基底：
```python
pad_embed_vec = llm_embed(pad_token_id).detach()
projected = projected + pad_embed_vec  # pad_embed (~0.02) + projector (~0.003)
```

效果：
- Projected tokens 初始 RMS ≈ 0.02（与正常 embedding 相同）
- RMSNorm backward 放大因子从 333× 降至 ≈ 50×（与正常 embedding 一致）
- Projector 输出作为 learned perturbation，训练中逐渐增大
- 不影响梯度方向（常数偏移的导数为 0），只改变 RMSNorm 的工作点

同时添加：
- `register_nan_gradient_hooks()`：参数级 NaN → 0 安全网，防止单个坏微批次毒化梯度累积
- `attn_implementation` 参数：虽然 eager attention 已排除为根因，保留选项用于未来调试

| (待定) | 残差嵌入 + NaN 梯度安全网 | 待验证 |

---

## 关键代码位置

| 位置 | 说明 |
|------|------|
| `src/train/model.py:516-524` | 残差嵌入：`projected = projected + pad_embed_vec` |
| `src/train/model.py:199-234` | `register_nan_gradient_hooks()`：参数级 NaN → 0 |
| `src/train/model.py:59` | `attn_implementation` 参数 |
| `src/train/model.py:265` | `extended_mask[:, 0] = 1`：MemE 左填充 NaN 防护 |
| `src/train/model.py:574` | `prefix_attn[:, 0] = 1`：LLM 前缀左填充 NaN 防护 |
| `src/train/train.py:140` | `model.register_nan_gradient_hooks()` 调用 |

---

## 已确认正常的部分

- **NCCL allreduce 顺序**：NTP 批次通过 dummy MemE+projector forward 解决 bucket 顺序问题
- **Step 0 forward pass**：所有微批次 loss 正常（1.66–3.17），logits 无 NaN/Inf
- **Label 对齐**：`cat([IGNORE * n_mem, target_labels])` 正确，valid label 数量合理
- **Projector 初始化**：最后一层 std=1e-4，输出接近零，不会导致 logits overflow
- **n_mem 混合批次**：同一批次内允许 n_mem 不同（如 [0, 82]），forward 逻辑正确处理
- **NaN 不来自 MemE**：MemE backward 本身是干净的（mem_hidden NaN 是 LLM NaN 的反向传播结果）
- **Loss 计算**：`ForCausalLMLoss` 将 logits upcast 至 float32，使用标准 `F.cross_entropy`
- **Loss scaling**：DeepSpeed bf16 使用 loss_scale=1.0，不存在梯度放大
- **Gradient Checkpointing**：不是 NaN 的根因（禁用后 NaN 仍在）
- **Flash/Eager Attention**：不是 NaN 的根因（eager attention NaN 模式不变）
- **NTP path 零嵌入**：已修复（pad embedding 替代 torch.zeros）

---

### 阶段 10 — NaN 梯度安全网对 inf 盲视（已修复）

**输入**：应用残差嵌入修复（阶段 9，commit `fe3004d`）后的训练日志

**观察**：
- Step 0 forward 完全干净（loss=2.5917，logits 有限）
- NaN gradient guard 在 step 0 期间无任何 `occurrence` 日志 → 梯度不是 NaN
- 中间张量诊断 hook 无 NaN、无 inf → 输出梯度有限
- 但 `grad_norm: nan` 在 step 0 → 梯度计算和 optimizer 之间出了问题
- `NaNDetectorCallback`：step 0 后所有 404 个参数权重均为 NaN
- `learning_rate: 0.0`（warmup）→ 但 `0.0 * NaN = NaN`（IEEE 754）

**根因分析：inf 盲视导致的级联**

NaN gradient guard（`register_nan_gradient_hooks`）只检查 `torch.isnan()`，遗漏了 `torch.isinf()`。完整级联链：

```
bf16 参数梯度 matmul 溢出 → inf（不是 NaN）
→ guard 只检查 isnan → inf 通过
→ DeepSpeed 梯度裁剪：norm = inf, scale = max_norm / inf = 0
→ 裁剪后梯度：inf * 0 = NaN（IEEE 754）
→ optimizer：w = w - lr * NaN = w - NaN = NaN（0 * NaN = NaN, IEEE 754）
→ 所有权重变为 NaN → 训练崩溃
```

**关键解释**：为什么中间张量 hook 没有捕获 inf？

中间张量 hook（`projected.register_hook`、`llm_input.register_hook` 等）监控的是**输出梯度**（`d_loss/d_output`）。参数梯度是通过 `activation^T @ output_gradient` matmul 计算的，这个矩阵乘法可以在 bf16 中独立溢出为 inf，即使输入的两个操作数都是有限的。

**修复**：

1. `src/train/model.py`：`register_nan_gradient_hooks()` 中 `_hook` 函数同时检查 `torch.isnan()` 和 `torch.isinf()`
2. `src/train/trainer.py`：`NaNDetectorCallback.on_step_end` 同时检查权重中的 NaN 和 inf

---

## 后续清理（训练稳定后）

确认 step 1 loss 正常、grad_norm 有限后，可以删除以下诊断代码：

1. `src/train/model.py`：删除 `_grad_hook_count` 和所有 `_make_hook` / `_make_ntp_hook` 代码块
2. `src/train/trainer.py`：删除 `NaNDetectorCallback` 类和 `compute_loss` 中的 debug logging（`if self.state.global_step < 3` 代码块）
3. `src/train/model.py`：如果 NaN 梯度安全网从未触发，可以移除 `register_nan_gradient_hooks()`
3. `src/train/train.py`：删除 `NaNDetectorCallback` 的 import 和注册
