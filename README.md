### 1. 项目简介
KawaiiLLM 是一个构建 ACGN（动画、漫画、游戏、小说）领域大语言模型（LLM）的项目。
核心目标有两个：
1. 实现ACGN风格的角色扮演：LLM学会二次元风格的语气，对人物有深刻理解，而不是标签化。
2. 实现记忆压缩：可以大幅压缩上下文，甚至实现人格向量，即通过抽象向量控制LLM性格。

#### 1.1 层次记忆向量
记忆编码模型可以输出不同数量的记忆向量，适应不用详细程度的记忆。 
- **高度抽象内容（情景记忆）**：对于高度抽象的内容（如：心情、感觉、性格）应该能使用很少的向量（如1个）描述，而且比原子化的自然语言描述（开心、 难过、热情）更合适，相比离散的自然语言，是更高效的一种“状态”描述。
- **具体细节内容（语义记忆）**：对于某些不重要、较久远的记忆，应该使用较少的向量描述。 但对重要的、需要仔细回忆的场景，应该使用较多的向量描述。 
- **模糊记忆**：分层次的记忆可以实现模糊记忆，并控制模糊程度。以写小说为例，当前段落的编写更多依赖于上一个章节，但也依赖于更久之前的一些情节、整个小说的主题。 临近的章节可以用更多的向量描述详细细节，而之前的章节用少量向量描述大概经历。还可以额外用1个向量维护整个小说的主题。 
- **DAST**：DAST这个工作已经提出了类似的思想，但DAST强调的是为不同信息密度分配不同数量的token，而我想的是为同一文章建立不同模糊程度/抽象层次的记忆向量组。
#### 1.2 角色向量
为了实现为了实现更好的角色扮演，是否可以将高度压缩、搞抽象度的记忆向量作为角色个性向量？ 基于高度抽象的记忆向量，建立描述某个角色个性的向量，该向量基于某个角色的经历，包含了角色的喜好、性格、言语风格等等。 
- 自然语言的提示词人设存在如下劣势： 
	- 刻板标签（原子性）： 当你在Prompt中写下“傲娇（Tsundere）”这个词时，你实际上并没有定义“这个特定的角色”，而是激活了模型训练数据中所有“傲娇”角色的统计平均值。 后果： 模型会立刻调用最俗套的特征（比如红脸、口是心非、双马尾），因为这是“傲娇”这个离散符号在潜在空间（Latent Space）中对应的概率最高的区域。你的角色瞬间失去了独特性，变成了一个“大众脸”**。
	- 语义漂移（模糊性）： 你心中的“冷酷”和模型理解的“冷酷”可能只是“亲戚关系”，并不完全重合。你可能想要的是“只有在杀敌时才冷酷，平时是发呆”，但模型读到“冷酷”可能理解为“对谁都爱答不理”。
	- 精度丢失（离散性）： 无论你用多少形容词去修饰（Explication），你都是在用有限的离散符号去逼近一个无限复杂的连续性格。这就像用马赛克（离散像素）去拼凑一张超高清照片，细节注定会丢失。 
- 角色向量的优势：
	- 连续化：Latent Tokens 不需要使用“傲娇”这个词。它可以在高维空间中找到一个精确的坐标点。
		- 优势： 这个点可以位于“傲娇”和“病娇”之间，偏向“忧郁”维度的 0.375 处，同时混合了“维多利亚时代礼仪”的特征。
	- 抽象化：维特根斯坦认为，很多内在体验（如微妙的情绪、复杂的直觉）是语言无法表达的。但在大模型中，这些“不可言说”的东西往往对应着潜在空间中某些难以被Token化的特征方向。Latent Tokens 是更高效的表示方法。
		- 例如：一个训练好的软向量可能包含了一种“说话时句尾微微上扬但并不表示疑问”的语调特征。这种特征用文字很难描述准确（描述了模型也不一定听得懂），但向量可以直接“激活”相关的神经元层。
一言以蔽之，自然语言是角色外在表现的描述，一旦人设转化为自然语言，就一定存在信息损失，而软向量是直接定义角色内在本质或“灵魂”。 


**主要流程：**
- 继续预训练：收集ACGN领域语料（主要是中文、日语、英语），训练Encoder编码高维语义信息为抽象记忆的能力，训练LLM读取抽象记忆并学习ACGN领域知识与语言风格。
- 后训练（暂定）：规范LLM角色扮演的输出格式（如思维链内为心理活动）、理解多组记忆输入。
- 强化学习（暂定）：训练记忆的动态读取、角色扮演的优化
### 2. 模型架构

### 2.1 Memory Embedder (MemE)
记忆表示编码器（通常是较小的Embedder）负责将长文本（例如4k tokens）压缩为高维隐向量。
这个过程相比"文本压缩"更像是"感觉还原"，即把文本还原为LLM的"体验"，就像多模态LLM中的图片蕴涵了语言难以完整表达的信息量。这里MemE类似ViT。
#### 2.2 LLM
较大的LLM，以接收并读取编码器输出的高维隐向量，从中获得读到原始文章后的"体验"。
#### 2.3 Projector
- 将编码器的隐藏状态映射到解码器的维度
- 作为编码器-解码器之间的桥梁，允许两者拥有不同的隐藏层大小
### 3. 完整流程
#### 3.1 增量预训练
通过增量预训练，注入ACGN领域知识，同时训练模型实现记忆的向量高效表示编码与读取。
#### 3.1.1 数据准备

所有收集的原始数据在目录`/Volumes/moedb/Datasets`内，包含多个来源的数据。

**数据总览**

| 来源 | 语言 | 原始体积 | 格式化输出 | 作用 |
|------|------|----------|------------|------|
| Z-Library 轻小说 | 中/日/英 | 5.16 GB | `data/novels/formatted/` | 长上下文、写作风格 |
| 萌娘百科 | 中文 | 790 MB | `data/moegirl/formatted/` | ACGN 领域知识 |
| Bilibili 专栏 | 中文 | 11.6 GB | `data/bilibili/formatted/` | 社区知识、评论风格 |
| 游戏剧本 | 中/英 | 730 MB | `data/games/formatted/` | 对话、心理描写 |
| Ultra-FineWeb | 中/英 | 2.45 GB | `data/general/formatted/` | 通用知识 |
| UltraData-Math | 英文 | 1.31 GB | `data/math/formatted/` | 逻辑推理 |
| StarCoderData | 编程语言 | 1.27 GB | `data/code/formatted/` | 逻辑推理 |

**共享分层切分模块**

所有格式化脚本使用统一的 `src/utils/chunking.py`，切分策略：
1. 段落切分（`\n\n`）
2. 行切分（`\n`）
3. 句子切分（`。！？…；!?;.`），代码模式跳过此步
4. 字符级二分查找硬切（兜底）

每个 chunk 不超过 **4096 tokens**（Qwen3-0.6B tokenizer），累积阶段在 >90% 上限时进行精确校验。

---

##### A. Z-Library 轻小说
**基本信息**：
* **语言**：中文、日语、英语
* **作用**：提供高质量长上下文，学习写作风格、记忆压缩与读取
* **数据量**：5.16GB
 **处理流程**：
1. **出版社过滤**：元数据缺失轻小说标签，观察发现轻小说多数会在标题中带有"文庫"字样，基于该特征寻找轻小说出版社，并手动过滤非轻小说出版社；
	- 脚本：`/src/utils/data_extract.py`
	- 索引：`/Volumes/moedb/Datasets/light-novels/selected_index.txt`
2. **电子书下载**：基于轻小说出版社名单抓取一批轻小说书单，下载对应`epub`文件，并将`epub`格式的轻小说转换为`txt`格式；
	- epub转txt脚本：`/src/utils/data_extract.py`
	- epub 数据：`/Volumes/moedb/Datasets/light-novels/epub`
	- txt 数据：`/Volumes/moedb/Datasets/light-novels/txt`
3. **关键词去重**：使用MinHash+LSH+BM25过滤存在大量文本重复的轻小说，得到5.3GB文本；
	- 去重脚本：`src/novels/data_dedup.py`
	- 输出：`data/novels/deduped`
4. **统一格式**：标题拼入文本后，使用分层切分为 ≤4096 token 的 chunk。
```bash
.venv/bin/python src/novels/format_novels.py \
  --input_dir data/novels/deduped \
  --output_file data/novels/formatted/novels_formatted.jsonl
```
```json
{
	"meta": {
		"id": "18625732",
		"title": "刀剑神域1 艾恩葛朗特",
		"language": "zh-cn",
		"source_filename": "18625732.txt",
		"total_words": 119435,
		"total_pages": 30
	},
	"texts": {
		"0": {
		"words": 107,
		"text": "\nSword Art Online刀剑神域[第一卷]\n川原砾简介\n一场死亡游戏\n即将揭开序幕\n\nSAO玩家-桐人，以完全攻略为目标，\n在游戏舞台「艾恩葛朗特」城堡里展开一连串严酷的冒险。\n途中与女剑士-亚丝娜的相遇，也为桐人带来命中注定的契机——\n\n§个人网站超过650万阅览人数，传说中的小说磅礴登场!\n"
		}
	},
	...
}
```
##### B. 萌娘百科
**基本信息**：
* **语言**：中文
* **作用**：提供高质量领域知识，学习ACGN领域知识。
* **数据量**：790MB
**处理流程**：
1. 原始数据包含大量wiki标签，HuggingFace上有处理好的文本，直接使用。
	- 数据位置：`data/moegirl/cleaned/MoeGirlPedia_zh_cleaned_latest.jsonl`
2. **统一格式**：标题若不在首行则自动前置，分层切分为 ≤4096 token 的 chunk。
```bash
.venv/bin/python src/moegirl/format_moegirl.py \
  --input_file data/moegirl/cleaned/MoeGirlPedia_zh_cleaned_latest.jsonl \
  --output_file data/moegirl/formatted/moegirl_formatted.jsonl
```
```json
{
	"title": "傲娇",
	"text": "傲娇（tsundere，ツンデレ）是一种人物性格，也是ACG次文化中的萌属性之一。..."
}
```
##### C. Bilibili 专栏
**基本信息**：
- **语言**：中文
- **作用**：提供ACGN社区文章与评论，学习ACGN领域高频知识、话题与社区风格。
- **数据量**：11.6GB
 **处理流程**：
 1. **数据收集**：通过爬虫爬取B站专栏文章以及热门评论；
 2. **标签过滤**：过滤筛选`category_id`为泛ACGN领域的文章及对应评论，得到21GB文本数据；
	 - 文章数据：`/Volumes/moedb/Datasets/bilibili-articles/articles.jsonl`
	 - 评论数据：`/Volumes/moedb/Datasets/bilibili-articles/comments.jsonl`
	 - 文章-评论合并数据：`data/bilibili/merged_articles.jsonl`
 3. **规则清洗**：清洗长度过短（<300字）、浏览量过低（view < 20）、命中广告关键词的文章；
	 - 脚本：`src/bilibili/bilibili_clean_fast.py`
	 - 输出：`data/bilibili/cleaned/articles_cleaned_v3.jsonl`
 4. **模型清洗**：训练轻量级LLM（Qwen3-0.6B 知识蒸馏），清洗低质量文章；
	 - 蒸馏数据：`data/bilibili/score_model/train.json`（由 Gemini 生成）
	 - 过滤脚本：`src/bilibili/filter_q1_articles.py`
	 - 输出：`data/bilibili/cleaned/articles_q1_merged.jsonl`
 5. **统一格式**：标题+元数据+正文+精选评论（Top5热评，每条最多2条回复）拼接后分层切分。
```bash
.venv/bin/python src/bilibili/format_bilibili.py \
  --input_path data/bilibili/cleaned \
  --output_file data/bilibili/formatted/bilibili_formatted.jsonl
```
```json
{
	"cvid": 151, 
	"link": "https://www.bilibili.com/read/cv151/", 
	"title": "为死去的角色办一场追悼会......", 
	"publish_time": "2017年06月23日 02:19", 
	"category_id": 4, 
	"category_parent_id": 2, 
	"category_name": "动漫杂谈", 
	"content": "四月新番《Re:Creators》是一部争议性很大的作品，不同二次元作品中的人物穿越到现实中，仅仅是这个脑洞设定就已经吸引了不少漫迷的关注。最近在第9话中魔法少女茉美香成了这场二次元角色反穿现实世界混战的第一个牺牲的角色，网友吐槽茉美香的退场都是「队友话多」的锅...", 
	"words": 857, 
	"mid": 144900177, 
	"author_name": "哔哩哔哩专栏", 
	"banner_url": "https://i0.hdslb.com/bfs/article/8cb53150173bfb7d2b4fe6a160d60797fb2be315.jpg", 
	"in_list": null, 
	"pre_num": null, 
	"next_num": null, 
	"type": 0, 
	"view": 1242, 
	"favourite": 7, 
	"like": 16, 
	"dislike": 0, 
	"reply": 6, 
	"share": 1, 
	"coin": 8, 
	"dynamic": 0, 
	"tags": "[\"北斗神拳\", \"Re:Creators\", \"茉美香\"]", 
	"comment_ids": "[\"1293530604\", \"622891074\", \"2647969894\"]", 
	"add_ts": "2025-04-01T00:19:17"
}
```
##### D. 游戏剧本
**基本信息**：
- **语言**：中文、英文。
- **作用**：学习游戏剧情发展，特别是对话与心理描写。
- **数据量**：730MB
 **处理流程**：
 1. **数据收集**：基于b-corpus数据集，提取其中的中/英文游戏剧本（对话形式）。数据包含galgame、手游、文字AVG等。
	 - 原始数据位置：`/Volumes/moedb/Datasets/b-corpus`
	 - 提取数据位置：`data/games/raw`
 2. **统一格式**：从路径提取游戏名+章节名作为标题前置，分层切分。
```bash
# 中文
.venv/bin/python src/games/format_games.py \
  --input_dir data/games/raw/zh \
  --output_file data/games/formatted/games_zh_formatted.jsonl

# 英文
.venv/bin/python src/games/format_games.py \
  --input_dir data/games/raw/en \
  --output_file data/games/formatted/games_en_formatted.jsonl
```
```json
木馨：「这里是木馨!」
木馨：「从今天开始,就是你的女朋友啦!」
木馨：「今后我们要在一起哦!」
木馨：「一起看书、一起回家,一起吃路边的小吃——不管做什么,都要在一起!」
旁白：——那当然!
旁白：当然要在一起。
旁白：因为你是我女朋友啊!!
旁白：嘿嘿嘿......我终于有女朋友了。
旁白：真幸福。
旁白：...... ?
旁白：怎么有闹钟的声音?
旁白：————!!
旁白：糟......糟糕了!!
旁白：不好了,不好了。
旁白：要迟到了!!
...
```
##### E. 通用语料
**基本信息**：
- **语言**：中文、英文。
- **作用**：保持通用领域知识。
- **数据量**：2.45GB。
**处理流程**：
 1. **数据收集**：基于Ultra-FineWeb数据集，抽取部分整理好的中文、英文通用语料。
	 - 提取数据位置：`data/general`
 2. **统一格式**：Parquet 读取后分层切分；英文保留 `url`/`date` 元数据。
```bash
# 中文（526,966 chunks）
.venv/bin/python src/general/format_general.py \
  --input_dir data/general/Ultra-FineWeb-zh \
  --output_file data/general/formatted/general_zh_formatted.jsonl \
  --lang zh

# 英文（416,867 chunks）
.venv/bin/python src/general/format_general.py \
  --input_dir data/general/ultrafineweb_en_v1_4 \
  --output_file data/general/formatted/general_en_formatted.jsonl \
  --lang en
```
```json
{
	"source": "general",
	"id": "eba702a360f3e9f23151b121ad456b1b",
	"split": 0,
	"tokens": 1914,
	"text": "上帝也爱香奈儿,仙女们怎么能停止买包包\n\n..."
}
```
##### F. 数学
**基本信息**：
- **语言**：英文
- **作用**：保持逻辑推理能力。
- **数据量**：1.31GB
**处理流程**：
 1. **数据收集**：基于[UltraData-Math](https://huggingface.co/datasets/openbmb/UltraData-Math)数据集，选取 L3 教材习题合成数据。
	 - 提取数据位置：`data/math`
 2. **统一格式**：Parquet 读取后分层切分（1,042,041 chunks）。
```bash
.venv/bin/python src/math/format_math.py \
  --input_dir data/math \
  --output_file data/math/formatted/math_formatted.jsonl
```
 ```json
 {
	"source": "math",
	"id": "352589f0-cd90-4dad-9a95-dc785c5b283c",
	"split": 0,
	"tokens": 1090,
	"text": "\\section{Explanation}\n\nIn this section, we will explore..."
}
 ```
##### G. 代码
**基本信息**：
- **语言**：英文、编程语言
- **作用**：保持逻辑推理能力。
- **数据量**：1.27GB
**处理流程**：
 1. **数据收集**：基于[starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)数据集，抽取部分整理好的代码语料。
	 - 提取数据位置：`data/code`
	 - **语言分布**:
		- Python: 25% (250MB)
		- C++: 20% (200MB)
		- C#: 20% (200MB)
		- Java: 10% (100MB)
		- JavaScript: 10% (100MB)
		- TypeScript: 10% (100MB)
		- Go: 5% (50MB)
		- Rust: 5% (50MB)
		- Lua: 3% (30MB)
		- SQL: 2% (20MB)
 2. **统一格式**：使用代码切分模式（跳过标点切分），保留 repo 元数据。
```bash
.venv/bin/python src/code/format_code.py \
  --input_dir data/code \
  --output_dir data/code/formatted
```
```json
{
	"source": "code",
	"id": "python_0",
	"split": 0,
	"tokens": 1550,
	"text": "<reponame>MTES-MCT/sparte\nfrom rest_framework_gis import serializers\n...",
	"meta": {
		"repo_name": "MTES-MCT/sparte",
		"path": "public_data/serializers.py",
		"stars": 0,
		"lang": "python"
	}
}
```
#### 3.1.2 数据格式
所有数据源格式化后，统一为以下 JSONL 格式：
```json
{"source": "bilibili", "id": "84", "split": 0, "tokens": 4056, "text": "document"}
{"source": "bilibili", "id": "84", "split": 1, "tokens": 2589, "text": "document"}
{"source": "novels",   "id": "11860530", "split": 0, "tokens": 4080, "text": "document"}
{"source": "novels",   "id": "11860530", "split": 1, "tokens": 3958, "text": "document"}
```
字段说明：
- `source`：数据类型（novels / moegirl / bilibili / games / general / math / code）
- `id`：文档唯一标识（优先使用原始 id，否则自建）
- `split`：同一文档切分后的 chunk 索引
- `tokens`：该 chunk 的 token 数（≤ 4096）
- `text`：训练文本
- `meta`（可选）：附加元数据（英文通用语料的 url/date，代码的 repo 信息）

#### 3.1.3 数据合并与混洗

将所有格式化后的 JSONL 文件合并、混洗、分片，输出预训练数据集。

**策略**：
- **交错读取**：从多个文件中每次各读 1000 行，避免同源数据扎堆
- **流式 Shuffle**：500,000 行 buffer，每次 flush 50%，保证内存可控
- **分片输出**：每 100,000 行一个分片 `pretrain_v1_part_{idx:03d}.jsonl`
- **验证集**：随机抽取 0.5% 写入 `validation.jsonl`

```bash
.venv/bin/python src/merge_and_shuffle.py \
  --input_dirs \
    data/novels/formatted \
    data/bilibili/formatted \
    data/moegirl/formatted \
    data/games/formatted \
    data/general/formatted \
    data/math/formatted \
    data/code/formatted \
  --output_dir data/pretrain \
  --shard_size 100000 \
  --buffer_size 500000
```

#### 3.1.4 模型准备
* **MemE**：
	* [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) 作为初始模型，拥有较好的语义理解与压缩能力。
* **Projector**：
	* 参考 Qwen2.5-VL 和 PCC 的 2层MLP + RMSNorm + GELU，维度从 `model.config.hidden_size` 动态读取（MemE=2560, LLM=4096），expansion factor ~4×。
* **LLM**：
	* [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) 作为初始模型。
	* 备选Qwen3-VL-8B-Instruct，因为训练过理解图片向量输入，可能迁移性更好。
#### 3.1.5 模型训练

##### 特殊 Token 设计

动态注册以下特殊 token（避免 C3 硬编码 token ID 的问题），并 resize MemE 和 LLM 的 embedding 层：

|Token|作用|所在侧|
|---|---|---|
|`<mem>` / `</mem>`|标记 memory token 序列的开始/结束边界|MemE + LLM|
|`<mempad>`|占位符，运行时被替换为可学习的 Memory Query Embeddings|MemE + LLM|
|`<AE>`|提示 LLM 进入"文本重建"模式（参考 PCC）|LLM|

```python
special_tokens = {"additional_special_tokens": ["<mem>", "<mempad>", "</mem>", "<AE>"]}
tokenizer.add_special_tokens(special_tokens)
meme_model.resize_token_embeddings(len(tokenizer))
llm_model.resize_token_embeddings(len(tokenizer))
```

**是否需要 `<AE>` 类特殊 token 区分任务？**

这两个目标对 latent tokens 编码的信息的"读取方式"是不同的。重建需要 LLM 把 latent 当作"压缩档案"来解压，续写需要 LLM 把 latent 当作"前情提要"来延伸。如果不给信号，模型只能从 latent 本身猜意图，这会导致：
- latent 表示被迫同时编码"内容"和"意图"，降低信息容量
- 训练初期 loss 收敛更慢，因为模型要先自己学会区分两种模式
- 推理时无法主动控制行为
如果不加：模型也能学会——毕竟重建和续写的目标分布差异很大，梯度信号足够模型隐式区分。但加了之后收敛更快、更可控，代价仅仅是多几个特殊 token。没有理由不加。

##### 数据流

**MemE（Memory Embedder）**：

在输入文本末尾注入 N 个 `<mempad>` 占位符（参考 C3 / PCC），运行时替换为可学习的 Memory Query Embeddings。利用 causal attention 的性质，每个 MEM token 能 attend 到前面的所有 text tokens，从而提取压缩信息。取 MEM token 位置在最后一层的 hidden states 作为输出。

```
MemE 输入: [text tokens (≤4096)] [<mem>] [Q_1, Q_2, ..., Q_N] [</mem>]
                                           ↑ 运行时替换为可学习的 Query Embeddings
MemE 输出: 取 Q_i 位置的 last hidden states → [v_1, v_2, ..., v_N]
```

其中 N 在训练时动态采样（见下文"可变 Latent Token 采样策略"），最大值 128，即 Memory Query Embeddings 定义为最大长度，训练时截取前 n 个：

```python
self.Q = nn.Parameter(torch.randn(128, hidden_size) * 0.02)  # 最大 128

def forward(self, input_ids, attention_mask, num_latent):
    active_Q = self.Q[:num_latent]  # 动态截取
    # ... 替换占位符 → forward → 提取 hidden states
```

利用 causal attention 的信息梯度特性：`Q_1` 只能 attend 到 text tokens（无其他 MEM token 干扰），天然适合作为最核心、最抽象的 Persona Vector；`Q_N` 能 attend 到所有前序 token，承载最多细节。

**Projector**：

两层 MLP（参考 PCC Converter + Qwen2.5-VL Merger 的共识设计）。MemE hidden_size=2560，LLM hidden_size=4096，Projector 同时负责维度映射和语义空间对齐。维度从 `model.config.hidden_size` 动态读取，不硬编码。

```
设 meme_dim = MemE.config.hidden_size (2560), llm_dim = LLM.config.hidden_size (4096)
Projector: RMSNorm(meme_dim) → Linear(meme_dim, meme_dim*4) → GELU → Linear(meme_dim*4, llm_dim)
输入: [v_1, ..., v_N]  →  输出: [h_1, ..., h_N]
```

**LLM**：

接收 Projector 输出的向量，替换 LLM 侧的 `<mempad>` 占位符，结合训练目标生成文本。

```
LLM 输入（重建模式）: [<mem>] [h_1, ..., h_N] [</mem>] [<AE>] → 输出 split_x 原文
LLM 输入（续写模式）: [<mem>] [h_1, ..., h_N] [</mem>]       → 输出 split_y（下一段）
```

多记忆输入场景（后训练阶段），使用 `<mem>` / `</mem>` 边界标记区分不同记忆段（PCC 验证了边界标记对多段 memory 的必要性）：

```
[<mem>] [latent_1_1, ..., latent_1_n] [</mem>]   ← 第一段记忆（完整情景）
[<mem>] [latent_2_1, ..., latent_2_m] [</mem>]   ← 第二段记忆（模糊情景）
[<mem>] [latent_3_1]                  [</mem>]   ← 第三段记忆（仅 Persona Vector）
[当前对话 prompt tokens...]
```

##### 训练目标

三种训练任务，同时学习领域知识和记忆压缩/读取能力：

**纯 NTP (Next Token Prediction)**：标准语言建模，无 MemE 参与，LLM 直接预测下一 token。使 LLM 通过常规自回归训练内化领域知识。

```
L_NTP = -1/t * Σ log P(x_i | x_1, ..., x_{i-1})
```

**文本重建（Auto-Encoding）**：使用特殊 token `<AE>` 显式提示 LLM 进入重建模式，还原 MemE 输入长文本中的每个 token。

```
L_TR = -1/t * Σ log P(x_i | h_e, <AE>, x_1, ..., x_{i-1})
```

**文本续写（Auto-Regression）**：LLM 基于前一段的 memory 表示，续写下一连续文本块。

```
L_TC = -1/(n-k) * Σ log P(x_i | h_e, x_k, ..., x_{i-1})
```

三种任务通过确定性轮换均衡分配（各 1/3），详见下文"任务分配"。

##### 任务分配 — 2/3-task 确定性轮换

任务分配根据样本是否有下一 chunk 区分：

**有下一 chunk 的样本** — 3-task 轮转（NTP / 重建 / 续写）：
```python
TASK_TYPES_3 = ["ntp", "reconstruction", "continuation"]
task_idx = (sample_idx + epoch + epoch // 3) % 3
```
- 每 3-epoch 周期内完整覆盖三种任务
- `epoch // 3` 项使不同周期的起始任务不同

**无下一 chunk 的样本** — 2-task 轮转（NTP / 重建）：
```python
TASK_TYPES_2 = ["ntp", "reconstruction"]
task_idx = (sample_idx + epoch) % 2
```
- 每 2-epoch 周期内完整覆盖两种任务
- 续写只分配给有真实 continuation pair 的样本，无 fallback 逻辑

性质：
- 无 warmup，从训练开始即所有任务均衡
- 全局任务比例取决于有 continuation pair 的样本占比（合并前约 12.2%，合并短孤立 chunk 后比例上升至 27.1%，因为分母减小而分子不变）

**Per-sample n_mem**

| 任务类型 | n_mem | 说明 |
|:---|:---|:---|
| NTP | 0 | 无 MemE 参与，纯语言建模 |
| 重建 | uniform [1, 128] | 全范围，学习不同压缩率的重建 |
| 续写 | uniform [1, 128] | 全范围，学习不同压缩率的上下文利用 |

NTP 独立承担领域知识内化，重建/续写专注于记忆压缩与读取能力训练。

模型通过左填充 prefix 处理 batch 内 variable-length latent tokens：
```
NTP sample:              [target...]           (无 prefix)
Sample A (n_mem=3):  [pad...] [<mem>] [h1, h2, h3] [</mem>] [target...]
Sample B (n_mem=50): [pad...] [<mem>] [h1, ..., h50] [</mem>] [target...]
```
Pure NTP batch（`max_n_mem=0`）跳过 MemE 计算；混合 batch 仅对非 NTP 样本运行 MemE。

**`<AE>` per-sample 任务信号**：重建任务在 LLM 的 `input_ids` 前端插入 `<AE>` token，续写和 NTP 不加。同一 batch 内可混合三种任务。

**EOS Token 策略：仅在文本真正结束时添加**

EOS 只在文本真正结束时添加。合并 chunk（人工 `\n\n` 拼接边界）不加 EOS；重建任务始终加 EOS（完整还原）；NTP 和续写根据是否有后续 chunk 决定。

| 任务类型 | 条件 | EOS |
|:---|:---|:---|
| **NTP** | merged entry | **不加**（人工拼接边界） |
| | 非 merged，有后续 chunk | **不加**（文本还没结束） |
| | 非 merged，无后续 chunk | **加 EOS**（文档自然结束） |
| **重建** | merged entry | **不加**（同上） |
| | 非 merged | **加 EOS**（完整还原，含结束信号） |
| **续写** | target 有后续 chunk | **不加**（文本继续） |
| （仅非 merged 参与） | target 无后续 chunk | **加 EOS**（文档结束） |

**三任务设计空间**

```
          NTP (无 MemE)        重建 (MemE + <AE>)       续写 (MemE)
      ┌──────────────────┬──────────────────────┬──────────────────────┐
 目的 │ LLM 内化领域知识 │ 学习压缩→解压能力    │ 学习压缩→续写能力    │
      │ (标准自回归训练)  │ (latent 当"压缩档案") │ (latent 当"前情提要") │
      ├──────────────────┼──────────────────────┼──────────────────────┤
 n_mem│ 0 (无 latent)    │ uniform [1, 128]     │ uniform [1, 128]     │
      ├──────────────────┼──────────────────────┼──────────────────────┤
 比例 │ 1/3              │ 1/3                  │ 1/3                  │
      └──────────────────┴──────────────────────┴──────────────────────┘
```

存在下一连续chunck的数据统计：

|**数据源 (Data Source)**|**总 entries**|**文档数**|**单 chunk 文档**|**多 chunk 文档**|**多 chunk 文档均分块数**|**有下一 chunk的 entries**|**续写可用比例**|
|---|---|---|---|---|---|---|---|
|**novels**|352,059|14,631|18|14,613|24.1|337,428|**95.8%**|
|**games**|45,938|913|27|886|51.0|45,025|**98.0%**|
|**code**|264,157|232,667|220,993|11,674|3.7|31,490|11.9%|
|**moegirl**|170,404|153,584|144,473|9,111|2.8|16,820|9.9%|
|**bilibili**|2,016,658|1,889,091|1,794,006|95,085|2.3|127,567|6.3%|
|**general**|943,816|913,909|893,567|20,342|2.4|29,907|3.2%|
|**math**|1,042,041|1,040,205|1,038,971|1,234|2.5|1,836|0.2%|
|**合计**|**4,835,073**|—|—|—|—|**590,073**|**12.2%**|

##### 数据索引优化

`build_index.py` 在构建索引时支持两种优化：

**上采样小数据源**：萌娘百科等高价值但数据量小的 source 可按倍率上采样，增加 NTP 曝光频率以加强知识记忆。上采样的 entries 在不同 index 位置会分配到不同任务，保持多样性。

```bash
python src/train/build_index.py --upsample moegirl:3 games:2
```

**合并短孤立 chunk**：孤立样本（无 prev 且无 next chunk）中 token 数低于阈值的，按同 source 贪心合并为一个 entry（用 `\n\n` 拼接），减少 padding 浪费。合并后的 entry 自然走 2-task 轮转（NTP + 重建）。

```bash
python src/train/build_index.py --merge_max_tokens 3500 --merge_short_threshold 2048
```
##### 训练流程

Qwen3-Embedding 已经过对比学习预训练，具备强语义理解能力，无需像 Qwen2.5-VL 那样从头训练 ViT。但新加入的 128 个 MEM token embeddings 和 Projector 是随机初始化的，需要差异化学习率策略来处理冷启动问题（类似 LLaVA / Qwen-VL 早期版本的做法）。

**阶段 1：记忆-语言联合训练**

解冻所有参数（MemE、Projector、LLM），采用差异化学习率：

|训练步数|MEM Embeddings|Projector|MemE Backbone|LLM|
|---|---|---|---|---|
|前 ~1000 步（warmup）|`1e-3`（高）|`1e-3`（高）|`1e-6`（近冻结）|`1e-6`（近冻结）|
|warmup 结束后|`5e-4`|`5e-4`|`1e-5`|`5e-6`|

> 前 1000 步 MEM embeddings 和 Projector 快速从随机值找到有意义的方向，而 backbone 和 LLM 几乎不动。等映射关系初步建立后再一起训练。

同时加入上面提到的课程学习部分。

Forward / Backward 数据流：

```python
# Forward Pass
# 重建任务示例：
split_x + [<mem>, Q_1, ..., Q_N, </mem>] → MemE → [v_1, ..., v_N]
    → Projector → [h_1, ..., h_N]
    → LLM 输入: [<mem>, h_1, ..., h_N, </mem>, <AE>] → 输出: split_x

# 续写任务示例：
split_x + [<mem>, Q_1, ..., Q_N, </mem>] → MemE → [v_1, ..., v_N]
    → Projector → [h_1, ..., h_N]
    → LLM 输入: [<mem>, h_1, ..., h_N, </mem>] → 输出: split_y  (y = x + 1)

# Compute Loss
loss = λ * L_TC + (1 - λ) * L_TR

# Backward Pass (梯度链路完整，端到端训练)
∂loss/∂LLM → ∂loss/∂Projector → ∂loss/∂MemE (含 MEM Embeddings + Backbone)

# Update (差异化学习率)
mem_embeddings  -= lr_high * ∂loss/∂mem_embeddings
projector       -= lr_high * ∂loss/∂projector
meme_backbone   -= lr_low  * ∂loss/∂meme_backbone
llm             -= lr_low  * ∂loss/∂llm
```

**阶段 2：后训练**（暂定）

冻结 MemE（此时 MemE 已稳定，继续训练可能破坏已学到的表示，参考 Qwen2.5-VL 在指令微调阶段冻结 ViT 的做法），微调 Projector + LLM。

目标：规范角色扮演输出格式、理解多组记忆输入。

**阶段 3：强化学习**（暂定）

训练记忆的动态读取、角色扮演的优化。

##### 训练代码

推荐直接在 C3 代码基础上改造，而非硬改 LlamaFactory（C3 本身即基于 HuggingFace Transformers + DeepSpeed ZeRO-2 训练，与本项目架构高度一致）。

技术栈：**HuggingFace Transformers + Accelerate + DeepSpeed ZeRO-2**

改造要点（基于 C3 代码）：

|改动项|说明|
|---|---|
|替换模型|`llm1` → Qwen3-Embedding-4B，主 LLM → Qwen3-8B-Base|
|动态 latent token|Q-Embeddings 支持动态截取前 n 个，DataCollator 传入当前 step 的 `num_latent`|
|双任务数据流|支持重建（+`<AE>`）和续写两种模式，替换 C3 的问答格式|
|多 context 支持|改造数据集类，解除 C3 单 context 限制，支持多段 `<mem>...</mem>`|
|课程学习调度器|在 Trainer 中加入阶段切换逻辑和 `sample_num_latent_tokens`|
|差异化学习率|为 MEM embeddings / Projector / MemE backbone / LLM 设置独立学习率|
|分模型保存|参考 C3 的 `C3Trainer`，MemE 和 LLM 独立保存 checkpoint|

# 参考资料

- [C3-Context-Cascade-Compression](https://github.com/liufanfanlff/C3-Context-Cascade-Compression)（代码基础）
- [LlamaFactory](https://github.com/hiyouga/LlamaFactory)（数据处理工具参考）
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)（分阶段训练策略）
- [PCC (ACL 2025)](https://aclanthology.org/2025.acl-long.1394/)（双任务预训练、压缩率实验、memory 边界标记）
# 参考讨论

**记忆压缩架构（MemE → Projector → LLM）** 本质上是将多模态 LLM 的视觉编码器范式迁移到"文本→隐向量→文本"的回路中。这个类比是合理的——Qwen2.5-VL 已经验证了 ViT + Projector + LLM 联合训练的可行性。但关键区别在于：图像天然携带"语言难以完整表达"的信息（空间、纹理等），而文本→向量→文本这条路径中，MemE 的输入和 LLM 的输出同属文本模态，模型是否真的能学到"超越文本的压缩表示"而非简单的有损摘要，这一点存疑。尤其是 One Token 策略——用单个向量重建 4k tokens 的信息，信息瓶颈极其严重，模型大概率会退化为主题分类器。

**维特根斯坦的哲学论证**作为动机很有趣，但需要警惕：在实际训练中，如果损失函数是 NTP（下一 token 预测），模型优化目标本质上还是让输出逼近离散 token 序列——隐向量中"不可言说"的部分如果对 token 预测没有贡献，就不会被保留。换言之，训练信号会把隐空间的表示拉向"可以用 token 表达的方向"。

[^1]: 
