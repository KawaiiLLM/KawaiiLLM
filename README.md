# KawaiiLLM

ACGN 领域大语言模型，具备层次化记忆压缩能力。

**核心目标**：
1. **ACGN 角色扮演** — 学习二次元语气与人物理解，而非标签化复读
2. **记忆压缩** — 将长上下文压缩为连续向量，实现分层记忆与角色向量

## 架构

```
Memory Text ──→ MemE (Qwen3-Embedding-4B) ──→ Projector ──→ LLM (Qwen3-8B-Base) ──→ Generated Text
               encoder, hidden=2560         2-layer MLP     decoder, hidden=4096
               causal attention +           RMSNorm+GELU    接收 [<mem>..tokens..</mem>]
               learnable MEM tokens         2560 → 4096     prefix 后自回归生成
```

MemE 将文本压缩为 1~128 个记忆向量，Projector 映射到 LLM 空间，LLM 基于记忆向量生成文本。

## 快速开始

### 环境

```bash
conda create -n kawaii python=3.10 -y && conda activate kawaii
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers deepspeed accelerate datasets pandas numpy
pip install fastapi uvicorn  # 推理服务
```

### 训练

```bash
# 1. 构建训练索引
bash scripts/build_index.sh

# 2. 启动训练 (8x A800)
bash scripts/train_multi_node.sh
```

### 推理

```bash
# 启动后端 API
CHECKPOINT_DIR=output/kawaii_v1 bash scripts/serve.sh

# 启动前端 (另一个终端)
cd web && npm install && npm run dev
```

生产部署：`VITE_API_BASE=http://gpu-server:8000 npm run build`，将 `dist/` 部署到任意 Web 服务器。

## 项目结构

```
KawaiiLLM/
├── src/
│   ├── train/                  # 训练代码
│   │   ├── model.py            #   KawaiiLLMModel (MemE + Projector + LLM)
│   │   ├── dataset.py          #   KawaiiDataset (byte-offset 索引访问)
│   │   ├── collator.py         #   DataCollator (左填充 context, 右填充 target)
│   │   ├── trainer.py          #   KawaiiTrainer + 监控回调
│   │   ├── train.py            #   训练入口
│   │   ├── build_index.py      #   构建训练索引 (扫描/上采样/合并/分割)
│   │   └── arguments.py        #   命令行参数定义
│   ├── inference/              # 推理代码
│   │   ├── engine.py           #   KawaiiInferenceEngine (训练一致的前向传播)
│   │   └── server.py           #   FastAPI + SSE 流式服务
│   ├── novels/                 # 轻小说数据处理
│   ├── bilibili/               # B站专栏数据处理
│   ├── moegirl/                # 萌娘百科数据处理
│   ├── games/                  # 游戏剧本数据处理
│   ├── general/                # 通用语料处理
│   ├── math/                   # 数学数据处理
│   ├── code/                   # 代码数据处理
│   ├── merge_and_shuffle.py    # 多源数据合并混洗
│   └── utils/
│       └── chunking.py         # 统一分层切分模块
├── web/                        # React 前端 (Vite + Tailwind)
│   └── src/
│       ├── App.jsx             #   主布局 (侧边栏 + 对话 + 记忆面板)
│       ├── api.js              #   API 客户端 (SSE 流式)
│       └── components/         #   UI 组件
├── configs/                    # DeepSpeed 配置
│   └── ds_zero2.json           #   ZeRO-2 配置
├── scripts/                    # 启动脚本
│   ├── train_multi_node.sh     #   多节点训练
│   ├── build_index.sh          #   构建索引
│   └── serve.sh                #   推理服务
└── docs/                       # 文档
    ├── design.md               #   项目设计文档 (详细架构与数据说明)
    ├── 2026-02-16-data-formatting-design.md
    │                           #   数据格式化设计
    ├── 2026-02-18-training_script_design.md
    │                           #   训练脚本设计 (含监控指标说明)
    ├── debug_nan_training.md   #   NaN 调试记录
    ├── bilibili/               #   B站数据清洗策略
    ├── novels/                 #   小说格式化实现
    └── code/                   #   代码处理日志
```

## 文档索引

| 文档 | 内容 |
|:-----|:-----|
| [docs/design.md](docs/design.md) | **项目总设计文档** — 层次记忆向量理论、架构设计、数据处理全流程、训练目标与任务分配、训练流程 |
| [docs/2026-02-16-data-formatting-design.md](docs/2026-02-16-data-formatting-design.md) | 数据格式化模块设计 — 各数据源处理方案 |
| [docs/2026-02-18-training_script_design.md](docs/2026-02-18-training_script_design.md) | 训练脚本设计 — 模型/数据集/Collator/Trainer 实现、TensorBoard 监控指标解读 |
| [docs/2026-02-24-training-run-analysis.md](docs/2026-02-24-training-run-analysis.md) | 第一轮预训练分析 — 梯度能量相变、recon/ntp 交叉里程碑、continuation 停滞诊断 |
| [docs/debug_nan_training.md](docs/debug_nan_training.md) | NaN 训练调试 — 诊断与修复过程记录 |
| [docs/bilibili/bilibili_cleaning_strategy.md](docs/bilibili/bilibili_cleaning_strategy.md) | B站专栏清洗策略 — 规则清洗 + 模型清洗 |
| [docs/novels/2026-02-16-novels-formatting-implementation.md](docs/novels/2026-02-16-novels-formatting-implementation.md) | 轻小说格式化实现细节 |
| [CLAUDE.md](CLAUDE.md) | Claude Code 项目指引 (环境、命令、架构速查) |

## 训练数据

| 来源 | 语言 | 体积 | 作用 |
|:-----|:-----|:-----|:-----|
| Z-Library 轻小说 | 中/日/英 | 5.16 GB | 长上下文、写作风格 |
| 萌娘百科 | 中文 | 790 MB | ACGN 领域知识 |
| Bilibili 专栏 | 中文 | 11.6 GB | 社区知识、评论风格 |
| 游戏剧本 | 中/英 | 730 MB | 对话、心理描写 |
| Ultra-FineWeb | 中/英 | 2.45 GB | 通用知识 |
| UltraData-Math | 英文 | 1.31 GB | 逻辑推理 |
| StarCoderData | 编程语言 | 1.27 GB | 逻辑推理 |

## 训练任务

三种任务确定性轮换，同时学习领域知识和记忆能力：

| 任务 | MemE 参与 | 输入 → 输出 |
|:-----|:----------|:------------|
| **NTP** | 否 | 文本前缀 → 下一 token |
| **重建** | 是 | `[<mem>..记忆..</mem>] [<AE>]` → 还原原文 |
| **续写** | 是 | `[<mem>..前段记忆..</mem>]` → 生成下一段 |

## 参考

- [C3-Context-Cascade-Compression](https://github.com/liufanfanlff/C3-Context-Cascade-Compression) — 代码基础
- [PCC (ACL 2025)](https://aclanthology.org/2025.acl-long.1394/) — 双任务预训练、memory 边界标记
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) — 分阶段训练策略

## License

MIT
