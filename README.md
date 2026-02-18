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
	* 参考Qwen2.5-VL的两层MLP，MLP Layer 1 → GELU → MLP Layer 2
* **LLM**：
	* [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) 作为初始模型。
	* 备选Qwen3-VL-8B-Instruct，因为训练过理解图片向量输入，可能迁移性更好。
#### 3.1.5 模型训练
##### 数据流
* **MemE**：输入长文本（4k discrete tokens），输出一组向量（128 latent tokens）。
* **Projector**: 输入 MemE 输出的向量，输出转换后的向量（与 LLM 向量维度一致）。
* **LLM**：输入 Projector 输出的向量，输出自然语言文本。
##### 训练目标
* 通过 NTP 学习领域知识，并学会压缩、读取向量记忆。分为两个任务（参考PCC）：
	* 文本重建：LLM 还原原始 MemE 输入长文本中的每个 token。
	* 文本续写：LLM 续写原始 MemE 输入长文本的下一连续文本块的每个 token。
* 为实现不同程度的记忆清晰度（压缩率），输入 LLM 的 Latent token 数量不固定：
	* One Token：仅使用 **Projector** 输出的第一个 latent token 进行训练，强调主题与模糊记忆，甚至抽象出人格向量（Persona Vector）。
	* Multiple Tokens：使用前几个 latent token 进行训练，逐步加入细节的还原，强调不同清晰程度的情景记忆。
##### 训练流程
以下流程参考Qwen2.5-VL。Qwen3-Embedding 已经经过对比学习预训练，无需单独训练。
* ***阶段1：记忆-语言联合训练**
	解冻所有参数（MemE、Projector、LLM）进行全面训练。Qwen2.5-VL 的案例证明**ViT(视觉编码器) + Projector + LLM** 的联合训练是可行的
	1. 前期：采样`split`为 0 的数据，使用文本重建 + Multiple Token 策略，即 `split_0` → MemE → [v] → Projector → [h1]  → LLM → `split_0`
	2. 中后期：慢慢加入`AD`类型数据（长文本，`split`大概率非0），此类型数据使用文本续写训练目标，更多的 Latent Tokens，即 `split_x`+ [m1, m2,...] → MemE → [v1, v2, ...] → Projector → [h1, h2, ...] → LLM → `split_y`，其中`y = x + 1`
```python
# 训练过程（简化版）

Step 1: Forward Pass
context → MemE → [v1, v2, ..., vn] → Projector → [h1, h2, ..., hn] → LLM → output
                ↑ 记忆表示             ↑ 当前映射           ↑ 生成结果

Step 2: Compute Loss
loss = CrossEntropy(output, target)  # 如"傲娇（tsundere，ツンデレ）是一种人物性格..."

Step 3: Backward Pass (关键!)
∂loss/∂LLM → ∂loss/∂Projector → ∂loss/∂MemE
     ↓              ↓                  ↓
  "需要更好    "需要这样映射"    "需要这种特征!"
   的记忆表示"

Step 4: Update All (同时更新!)
MemE.weights      -= lr * ∂loss/∂MemE
Projector.weights -= lr * ∂loss/∂Projector
LLM.weights      -= lr * ∂loss/∂LLM
```
* ***阶段2：长上下文预训练***
	暂时不考虑。
##### 训练代码
- 基于 [LlamaFactory](https://github.com/hiyouga/LlamaFactory)，借鉴 [C3-Context-Cascade-Compression](https://github.com/liufanfanlff/C3-Context-Cascade-Compression)的思路进行修改。
- 或则搭建独立的训练脚本。

# 参考讨论
## 1. 核心创新点评估

**记忆压缩架构（MemE → Projector → LLM）** 本质上是将多模态 LLM 的视觉编码器范式迁移到"文本→隐向量→文本"的回路中。这个类比是合理的——Qwen2.5-VL 已经验证了 ViT + Projector + LLM 联合训练的可行性。但关键区别在于：图像天然携带"语言难以完整表达"的信息（空间、纹理等），而文本→向量→文本这条路径中，MemE 的输入和 LLM 的输出同属文本模态，模型是否真的能学到"超越文本的压缩表示"而非简单的有损摘要，这一点存疑。尤其是 One Token 策略——用单个向量重建 4k tokens 的信息，信息瓶颈极其严重，模型大概率会退化为主题分类器。

**维特根斯坦的哲学论证**作为动机很有趣，但需要警惕：在实际训练中，如果损失函数是 NTP（下一 token 预测），模型优化目标本质上还是让输出逼近离散 token 序列——隐向量中"不可言说"的部分如果对 token 预测没有贡献，就不会被保留。换言之，训练信号会把隐空间的表示拉向"可以用 token 表达的方向"。
## 2. 模型架构与训练

**模型选择合理：** Qwen3-Embedding-4B 作为 MemE、Qwen3-8B-Base 作为 LLM，都是当前开源模型中的优质选择。4B 编码 + 8B 解码的参数比例也较为均衡。

**主要技术风险：**

1. **128 latent tokens 的瓶颈问题。** 4096 tokens → 128 latent tokens 是 32 倍压缩。对比 PCC 和 C3 的实验，这个压缩率下的文本重建质量通常会显著下降。建议先做消融实验，验证不同压缩率下重建和续写的 loss 曲线。
2. **Qwen3-Embedding-4B 的适配问题。** 该模型是为对比学习训练的 embedding 模型，输出的是单向量句子表示。要把它改造成输出 128 个 latent tokens 的序列编码器，需要显著修改其 pooling 策略。
3. **训练课程（curriculum）的设计风险。** 从"短文本重建 + One Token"到"长文本续写 + Multiple Tokens"的课程转换，节奏把控很关键。如果切换太快，模型可能学不到稳定的压缩表示；太慢则训练效率低。建议设置明确的阶段切换指标（比如重建 loss 降到某阈值后再引入续写任务）。
### Qwen3-Embedding 适配

Qwen3-Embedding-4B 原生只输出**1个向量**（EOS token 的 hidden state），而你需要**可变数量的 latent tokens**，且希望不同数量承载不同语义粒度。这其实是两个子问题：**如何提取多个 latent tokens** 和 **如何让不同 token 位置承载层次化语义**。


有三条主要技术路线：

**方案 A：C3 式占位符注入**

在 MemE 的输入序列末尾插入 N 个可学习的特殊 token `[MEM_1]...[MEM_N]`，取它们在最后一层的 hidden states 作为 latent tokens。

```
输入: [text tokens...] [MEM_1] [MEM_2] ... [MEM_128]
输出: 取 [MEM_i] 位置的 hidden states → Projector → LLM
```

优点是实现简单、与 Qwen3-Embedding 的 causal attention 兼容（每个 MEM token 能 attend 到前面所有 text tokens）。缺点是 128 个额外 token 会增加 MemE 的计算量，且 causal mask 导致 `MEM_1` 只能看到 text + 自己，`MEM_128` 能看到所有——**天然形成信息量从少到多的梯度**，这恰好符合你的设计意图。

**方案 B：Q-Former / Cross-Attention 式提取**

在 MemE 顶部加一个轻量 cross-attention 层，用 N 个可学习的 query tokens 去 attend MemE 最后一层的所有 hidden states。这本质上是 BLIP-2 的 Q-Former 思路。

```python
class MemoryExtractor(nn.Module):
    def __init__(self, num_latent=128, hidden_dim=3584):  # Qwen3-Emb-4B dim
        super().__init__()
        # 可学习的 query tokens
        self.latent_queries = nn.Parameter(
            torch.randn(num_latent, hidden_dim) * 0.02
        )
        # 轻量 cross-attention (可堆叠2-4层)
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=16, 
                dim_feedforward=hidden_dim * 4, batch_first=True
            ) for _ in range(2)
        ])
    
    def forward(self, encoder_hidden_states):
        # encoder_hidden_states: [B, seq_len, hidden_dim]
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.cross_attn_layers:
            queries = layer(queries, encoder_hidden_states)
        return queries  # [B, num_latent, hidden_dim]
```

优点：query tokens 的数量与输入长度解耦，推理时可以灵活控制；每个 query 都能全局 attend 所有位置，信息提取更均匀。缺点：引入额外参数（但 2 层 cross-attn 相对于 4B 模型来说很轻量）。

**方案 C：Strided Pooling / 降采样**

直接取 MemE 最后一层所有 hidden states，通过 1D 卷积或 strided average pooling 降采样到 128 个向量。实现最简单，但语义层次性最差。
### 记忆层次化

不管选哪个方案，**关键在于训练目标如何引导不同位置的 latent token 承载不同粒度的信息**。你的直觉是对的——通过 One Token vs Multiple Tokens 的训练策略来实现这一点。具体建议：

**对 latent tokens 施加"信息瓶颈递增"的约束：**

```
位置 1 (Persona Vector):  训练时最频繁被单独使用 → 被迫编码最核心的身份/主题/风格
位置 2-8 (粗粒度记忆):    训练时中等频率使用 → 编码主要事件/情感基调
位置 9-128 (细粒度记忆):  训练时偶尔全部使用 → 编码具体细节/对话内容
```

训练时的采样策略：

```python
def sample_num_latent_tokens(step, max_steps):
    """课程学习：随训练推进，逐步增大使用的 latent token 数"""
    # 前30%步数：80%概率用1个token, 20%概率用2-8个
    # 中间40%步数：40%用1个, 40%用2-16个, 20%用16-64个
    # 后30%步数：20%用1个, 30%用2-16个, 30%用16-64个, 20%用64-128个
    phase = step / max_steps
    if phase < 0.3:
        return random.choices([1, randint(2,8)], weights=[0.8, 0.2])[0]
    elif phase < 0.7:
        return random.choices([1, randint(2,16), randint(16,64)], 
                              weights=[0.4, 0.4, 0.2])[0]
    else:
        return random.choices([1, randint(2,16), randint(16,64), 128], 
                              weights=[0.2, 0.3, 0.3, 0.2])[0]
```

这样 **位置 1 在所有训练步骤中都会被用到**，自然会被优化为承载最鲁棒、最抽象的信息（Persona/Theme Vector）。后续位置只在需要更多细节时才被使用，信息呈递增分布。

**推荐方案 A + 上述采样策略**。理由是：C3 式占位符方案中，causal attention 的天然信息梯度与你的层次化设计完美契合——`MEM_1` 物理上只能 attend 到所有 text tokens（没有其他 MEM 的干扰），天然适合做 Persona Vector。而方案 B 中所有 query 都全局 attend，需要额外机制来保证层次性。

如果选方案 A，对 Qwen3-Embedding-4B 的改动很小：

```python
# 改动点：
# 1. 在 tokenizer 中添加 [MEM_1]...[MEM_128] 特殊 token
# 2. 扩展 embedding 层
# 3. 原本取 EOS hidden state → 改为取 MEM token 位置的 hidden states

class ModifiedQwen3Embedding(Qwen3EmbeddingModel):
    def __init__(self, config, num_mem_tokens=128):
        super().__init__(config)
        self.num_mem_tokens = num_mem_tokens
        # 可学习的 memory token embeddings
        self.mem_embeddings = nn.Embedding(num_mem_tokens, config.hidden_size)
    
    def forward(self, input_ids, attention_mask, num_active_tokens=None):
        # 原始 text encoding
        text_embeds = self.embed_tokens(input_ids)
        
        # 拼接 memory tokens
        n = num_active_tokens or self.num_mem_tokens
        mem_embeds = self.mem_embeddings.weight[:n].unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([text_embeds, mem_embeds], dim=1)
        
        # 更新 attention mask
        mem_mask = torch.ones(B, n, device=device)
        combined_mask = torch.cat([attention_mask, mem_mask], dim=1)
        
        # Forward through transformer
        hidden_states = self.transformer(combined, combined_mask)
        
        # 提取 memory token 位置的输出
        mem_hidden = hidden_states[:, -n:, :]  # [B, n, hidden_dim]
        return mem_hidden
```

## 3. 训练代码与框架

### 需要改动的层面

|改动项|难度|工作量|说明|
|---|---|---|---|
|**自定义模型架构**|⭐⭐⭐⭐|3-5天|LlamaFactory 假设单一模型，需要注入 MemE+Projector+LLM 三组件的 forward 逻辑|
|**自定义数据流**|⭐⭐⭐|2-3天|需要 DataCollator 同时产出 MemE 输入和 LLM 目标，并支持动态 latent token 数量|
|**训练循环改造**|⭐⭐⭐⭐|3-5天|多模型联合训练的梯度回传、不同组件的学习率、课程学习的阶段切换|
|**DeepSpeed 适配**|⭐⭐⭐⭐⭐|3-7天|这是最大的坑|
|**评估与调试**|⭐⭐⭐|持续|需要自定义 metrics（重建质量、续写困惑度等）|

### DeepSpeed 适配是最大难点

LlamaFactory 原生支持 DeepSpeed，但针对的是**单模型**训练。你的场景是 **MemE(4B) → Projector → LLM(8B)**的多模型联合训练，会遇到以下具体问题：

**ZeRO Stage 选择困境：**

- ZeRO-2：每张卡保存完整参数，12B+ 模型至少需要 ~24GB（bf16），加上激活值和梯度，单卡 80G A100 勉强够用。但你还要存 4k tokens 的 MemE 激活用于反向传播。
- ZeRO-3：参数分片到多卡，显存效率高，但 MemE → Projector → LLM 的链式 forward 中，每一步都需要 all-gather 参数，通信开销大。而且 ZeRO-3 对自定义模型的 `forward()` 有严格要求——所有参数必须在 forward 中被访问到，否则会 hang。

**梯度回传链路的问题：** `loss.backward()` 需要从 LLM 一路回传到 MemE。如果 MemE 和 LLM 被 DeepSpeed 包装为不同的 engine，梯度链会断裂。解决方案是把整个 pipeline 包装为一个 `nn.Module`：

```python
class KawaiiLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.meme = ModifiedQwen3Embedding(...)   # 4B
        self.projector = Projector(...)            # ~几十M
        self.llm = Qwen3ForCausalLM(...)          # 8B
    
    def forward(self, meme_input_ids, meme_attention_mask, 
                llm_input_ids, llm_labels, num_latent):
        # MemE 编码
        mem_hidden = self.meme(meme_input_ids, meme_attention_mask, num_latent)
        # 投影
        projected = self.projector(mem_hidden)
        # LLM forward: 将 projected 作为前缀嵌入
        llm_embeds = self.llm.get_input_embeddings()(llm_input_ids)
        combined_embeds = torch.cat([projected, llm_embeds], dim=1)
        outputs = self.llm(inputs_embeds=combined_embeds, labels=llm_labels)
        return outputs.loss
```

这样 DeepSpeed 只看到一个模型，梯度链路完整。但代价是 LlamaFactory 的很多上层逻辑（模型加载、checkpoint 保存、LoRA 注入等）需要大量改写。

### 务实建议

**不要硬改 LlamaFactory，而是借鉴其数据处理和训练工具，搭建独立的训练脚本。** 理由如下：

1. LlamaFactory 的核心价值在于标准化的 SFT/RLHF 流程和数据模板，但你的场景是**非标准的增量预训练**，LlamaFactory 的抽象层反而会成为阻碍。
2. C3 的核心代码（压缩 token 的注入、多段训练）约 500-800 行，直接参考移植到独立脚本比嵌入 LlamaFactory 更可控。
3. DeepSpeed 的集成用 `accelerate` 或直接调 `deepspeed.initialize()` 即可，不需要 LlamaFactory 的封装。

推荐的技术栈：

```
训练框架：HuggingFace Transformers + Accelerate + DeepSpeed (ZeRO-2)
参考代码：C3 的压缩逻辑 + LlamaFactory 的数据处理
自定义部分：KawaiiLLM 模型类、DataCollator、课程学习调度器
```

预估工作量：一个熟练的工程师大约 **2-3 周**可以跑通第一个端到端实验（不含调参）。如果坚持改 LlamaFactory，工期可能翻倍，且后续维护成本高。

# 参考资料
Qwen2.5-VL
Pretraining Context Compressor (PCC)
Context Cascade Compression (C3)
