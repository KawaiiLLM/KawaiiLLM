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

### 阶段 7 — 深入调查：GC backward 仍然产生 NaN（进行中）

**已排除的假设汇总**：

| 假设 | 排除依据 |
|------|----------|
| Flash Attention backward 不兼容 GC | eager attention 同样 NaN |
| GC `use_reentrant=True` 的 re-run 不一致 | `use_reentrant=False` 同样 NaN |
| Loss scaling 导致梯度溢出 | bf16 下 loss_scale = 1.0，无放大 |
| Softmax 输入全为 -inf（全 mask 行） | `prefix_attn[:, 0] = 1` 保证每行至少有 1 个有效 key |
| Packed sequence 检测误触发 | 仅当 `attention_mask is None` 时触发，而我们传递了 attention_mask |
| MemE / projector 是 NaN 源头 | hook 顺序和 NTP 路径均排除 |

**关键新发现**：

1. **DeepSpeed bf16 默认不检查梯度溢出**：`DeepSpeedBF16Config` 中 `check_grad_overflow` 默认为 `False`。这意味着 NaN 梯度会直接参与 optimizer.step()，无保护机制。这解释了为什么 NaN 梯度会导致 step 1 的权重被破坏。

2. **NaN 特征**：`nan=True inf=False` — 梯度中出现 NaN 但不出现 Inf。若是数值溢出，通常先出现 Inf 再传播为 NaN。直接出现 NaN 提示可能是 `0/0`、`inf - inf` 或 `0 * inf` 类操作。

3. **NaN 出现在最首次 backward**：fwd=1（每个 GPU 的第一个微批次 backward）就产生 NaN。此时模型权重为干净的预训练权重，尚未执行过任何 optimizer.step()。说明这是一个结构性问题，不是数值积累导致。

**当前状态**：根因仍在 gradient checkpointing 与 Qwen3 LLM backward 的交互中，但具体是哪一层、哪个操作产生 NaN 尚未确定。

---

## 下一步诊断计划

### 方案 A：禁用 GC 验证（确认 GC 是否为根因）— 已实施

**脚本**：`scripts/train_debug_no_gc.sh`

```bash
# 关键变更 vs train_8xa800.sh:
--freeze_meme True               # 无 MemE 梯度，省 ~16GB optimizer states
--gradient_checkpointing False   # 核心测试：完全禁用 GC
--per_device_train_batch_size 1  # 减半以容纳无 GC 时的激活
--gradient_accumulation_steps 16 # 翻倍保持有效 batch = 1*16*8 = 128
--context_max_length 2048        # 减半以降低激活内存
--target_max_length 2048         # 减半以降低激活内存
--max_steps 50                   # 短期诊断，只需观察 step 0-2
--save_strategy no               # 不保存 checkpoint
--logging_steps 1                # 每步都记录
--report_to none                 # 不需要 tensorboard
```

**预期内存**（MemE 冻结，无 GC）：
- MemE 权重（bf16，无梯度）：~8GB
- LLM 权重（bf16）：~16GB
- LLM 优化器状态（ZeRO-2，/8）：~8GB
- LLM 梯度（ZeRO-2，/8）：~2GB
- LLM 激活（无 GC，batch=1，seq≈2k）：~25GB
- 总计 ~60GB < 80GB → 可行

**运行**：`bash scripts/train_debug_no_gc.sh`

**判断标准**：
- 若 backward 无 NaN → 确认 GC 是根因，继续方案 B 定位具体层
- 若 NaN 仍在 → 根因不在 GC，需要重新审视其他方向

### 方案 B：逐层 backward hook（定位产生 NaN 的具体层）

在 model.py 中添加 per-layer backward hook：

```python
# 在 forward() 中 LLM 调用前添加
if self.training and self._grad_hook_count < 5:
    for i, layer in enumerate(self.llm.model.layers):
        def make_layer_hook(layer_idx):
            def hook(module, grad_input, grad_output):
                for j, g in enumerate(grad_output):
                    if g is not None and torch.isnan(g).any():
                        logger.error(
                            "LAYER_NaN layer=%d grad_output[%d] nan=True shape=%s",
                            layer_idx, j, list(g.shape),
                        )
            return hook
        layer.register_full_backward_hook(make_layer_hook(i))
```

**预期输出**：找到第一个（从模型顶部算起）报告 NaN 的层号。backward 从最后一层向第一层传播，最后一个报告 NaN 的层就是 NaN 的起源层。

### 方案 C：立即防护措施

在 `configs/ds_zero2.json` 的 bf16 配置中加入：
```json
"bf16": {
    "enabled": true,
    "check_grad_overflow": true
}
```
使 DeepSpeed 在 optimizer.step() 前检查 NaN/Inf，跳过包含异常梯度的步骤，防止权重被破坏。

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
| `src/train/model.py:68` | MemE 加载：`attn_implementation="eager"` |
| `src/train/model.py:78` | LLM 加载：`attn_implementation="eager"` |
| `src/train/model.py:193` | `encode_context()`：MemE 编码，extended_mask 构建 |
| `src/train/model.py:271` | `extended_mask[:, 0] = 1`：防止左填充 position 0 的 softmax NaN |
| `src/train/model.py:173` | `gradient_checkpointing_enable()`：MemE + LLM 均启用 GC |
| `src/train/model.py:534` | 梯度 NaN hook 注册（前 20 次 forward，诊断用） |
| `src/train/train.py:129` | GC 启用：`use_reentrant=False` |
| `src/train/trainer.py:19` | `NaNDetectorCallback`：step 1 后检查参数 NaN |
| `src/train/trainer.py:108` | `compute_loss()`：debug 日志（前 3 个 optimizer step） |

---

## 已确认正常的部分

- **NCCL allreduce 顺序**：NTP 批次通过 dummy MemE+projector forward 解决 bucket 顺序问题
- **Step 0 forward pass**：所有微批次 loss 正常（1.66–3.17），logits 无 NaN/Inf
- **Label 对齐**：`cat([IGNORE * n_mem, target_labels])` 正确，valid label 数量合理
- **Projector 初始化**：最后一层 std=1e-4，输出接近零，不会导致 logits overflow
- **n_mem 混合批次**：同一批次内允许 n_mem 不同（如 [0, 82]），forward 逻辑正确处理
- **NaN 不来自 MemE**：MemE backward 本身是干净的（mem_hidden NaN 是 LLM NaN 的反向传播结果）
- **Eager attention 实现**：Qwen3 正确使用 `torch.matmul` + `F.softmax(dtype=float32)`，非 SDPA
- **Loss 计算**：`ForCausalLMLoss` 将 logits upcast 至 float32，使用标准 `F.cross_entropy`
- **Loss scaling**：DeepSpeed bf16 使用 loss_scale=1.0，不存在梯度放大

---

## Commit 历史

| Commit | 说明 | 结果 |
|--------|------|------|
| 8cc444a | 修复 encode_context 中 extended_mask position 0 | step 1 仍 NaN，假设 1 排除 |
| d4f7adc | 禁用 MemE GC + NaNDetectorCallback + 梯度 NaN hook | OOM，假设 2 排除，诊断代码就位 |
| 13626c3 | 恢复 MemE GC（fix OOM），保留诊断代码 | NaN 复现，hook 成功采集数据 |
| 539502c | `attn_implementation="eager"` + `use_reentrant=False` | NaN 模式不变，假设 3 排除 |

---

## 后续清理（训练稳定后）

确认 step 1 loss 正常、grad_norm 有限后，可以删除以下诊断代码：

1. `src/train/model.py`：删除 `_grad_hook_count` 和所有 `_make_hook` / `_make_ntp_hook` 代码块
2. `src/train/trainer.py`：删除 `NaNDetectorCallback` 类和 `compute_loss` 中的 debug logging（`if self.state.global_step < 3` 代码块）
3. `src/train/train.py`：删除 `NaNDetectorCallback` 的 import 和注册
