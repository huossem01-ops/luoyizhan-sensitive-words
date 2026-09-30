# 罗翼展敏感词库

这是一个围绕“恋爱、异性关系、校园青春、婚恋压力和社会比较”的中文语义检索实验。当前版本不再使用 LLM 或模板批量生成词条，而是从公开中文词/短语向量词表中，以人工种子做语义召回，再通过规则过滤、去重和人工抽检形成候选集。

这里的“敏感词”是虚构化 persona 的幽默设定，用来描述围绕特定主题形成的语义联想，不是医学或心理诊断标签。

## 当前状态

- 已暂停并隔离旧的 48,000 条生成方案；旧数据仅保留在 `data/legacy_generated/` 供审计。
- MVP 使用腾讯 AI Lab 中文词向量的轻量镜像（143,613 个词/短语、200 维）进行召回。
- 已完成 Top 5,000 召回、三层抽样标注和 BGE 重排实验；仍不补齐 48,000 条。
- `term` 只允许词和固定/常用短语；当前自动过滤上限为 16 字，30 字以上绝不进入正式候选。
- 核心保守子集为 160 条；随后把旧 Generated 数据仅作为“概念缺口清单”，从腾讯真实词表定向召回并审核新增 150 条。当前扩展审核版共 310 条，边界词和未审核词不进入。

## 数据流

```text
人工种子词
  -> 腾讯中文词/短语向量词表
  -> 余弦相似度召回
  -> 词形与句子化过滤
  -> 规范化去重/近重复报告
  -> Top 5,000 候选与人工抽检样本
```

正式候选输出到：

- `data/final/sensitive_words.csv`
- `data/final/sensitive_words.jsonl`

经过明确审核的保守子集输出到：

- `data/final/conservative_sensitive_words.csv`
- `data/final/conservative_sensitive_words.jsonl`

加入校园物件、外貌细节、纸面心意、校园空间和约会意象后的当前扩展审核版：

- `data/final/expanded_sensitive_words.csv`
- `data/final/expanded_sensitive_words.jsonl`

BGE 与融合排序实验保存在 `data/reranked/`，不会覆盖原始腾讯召回顺序。

每条记录包括词条、规范化形式、长度、综合相似度、排名、最相近种子、最佳类别、类别相似度、暂定语义等级和数据源。字段约束见 `schema/data.schema.json`。

## 运行

建议使用 Python 3.10–3.12：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\download_tencent_vectors.py --source light-mirror
.venv\Scripts\python src\tencent_retrieval.py
.venv\Scripts\python src\deduplicate.py
.venv\Scripts\python src\validate.py
.venv\Scripts\python src\qa.py
.venv\Scripts\python src\label_manual_samples.py
.venv\Scripts\python src\bge_rerank.py
.venv\Scripts\python src\label_bge_samples.py
.venv\Scripts\python src\calibrate_fusion.py
.venv\Scripts\python src\build_conservative.py
.venv\Scripts\python src\targeted_expansion.py
.venv\Scripts\python src\rerank_expansion.py
.venv\Scripts\python src\label_expansion_samples.py
.venv\Scripts\python src\build_expanded_lexicon.py
```

腾讯官方完整词向量及轻量镜像的来源、许可和限制见 `docs/data_sources.md`。轻量镜像只用于跑通 MVP；后续若替换为官方完整词表，必须重新运行召回、去重和抽检。

## 质量原则

- 语义多样性来自不同概念节点，不来自添加时间、地点、人物或程度副词。
- 2–6 字为核心词汇，7–12 字为常用短语；超过 12 字须重点人工审核。
- 测试句子与词库严格分开，完整句子放在 `data/test_sentences.jsonl`。
- 相似度和 `semantic_level` 只表示相对于当前种子集的机器召回顺序，不等同于人工确认的触发强度。
- 近重复报告只提供人工复核线索，不会把语义角色不同的短语盲目合并。

## 局限性

- 轻量镜像只含官方大词表的一个子集，召回覆盖和排名不能代表完整腾讯词向量。
- 分布式词向量会召回人名、作品名、机构名和只有特定语境才相关的词，必须人工审核。
- BGE 可以提高头部集中度，但会把部分成语和固定恋爱表达错误降权，因此只作为第二路信号，与腾讯排名融合使用。
- 旧 Generated 数据只用于发现遗漏的概念节点和设计种子，不会直接回流正式词库；扩展版的每个新增词都必须真实存在于腾讯词表并经过复核。
- 种子词覆盖会直接影响类别分布；类别和暂定等级不是客观心理测量。
- 互联网语言持续变化，中英文混用词需定期复核。

## 隐私与许可

数据不得加入真实个人的私密记录、联系方式或可识别学校信息。代码沿用本仓库 MIT License；第三方向量不随仓库再分发，并受其自身许可约束。腾讯 AI Lab 页面将原始词向量标为 CC BY 3.0 Unported，并说明仅供研究使用，具体见 `docs/data_sources.md`。
