# 罗翼展敏感词库

> 据说一个人真正的敏感词并不只有一个，而是一整片语义空间。

一个带有幽默设定的中文主题词库与语义检索实验。项目围绕恋爱、校园青春、亲密关系、婚恋压力与社会比较等概念，研究如何从少量人工种子出发，构建可解释、可审核、可复现的中文词与短语集合。

“罗翼展”是项目中的虚构化 persona；“敏感词”指容易引发主题联想的语义触发词，不是医学或心理诊断标签。

## 项目状态

当前处于**小规模、人工审核优先**阶段。项目曾探索大规模生成路线，但发现句子化、伪重复和语义注水问题后，已停止以数量为目标的扩充。旧数据隔离保存，仅用于审计和发现概念缺口，不会直接进入正式词库。

| 数据层 | 规模 | 用途 | 审核状态 |
| --- | ---: | --- | --- |
| 腾讯召回候选 | 5,000 | 排序实验、误差分析 | 部分抽检 |
| 保守核心集 | 160 | 高置信基线 | 全部保留项经过人工审核 |
| 扩展审核集 | 310 | 当前推荐版本 | 核心 160 + 定向扩展 150，全部经过人工审核 |

当前推荐使用 [`data/final/expanded_sensitive_words.csv`](data/final/expanded_sensitive_words.csv) 或对应的 JSONL 文件。310 条记录均已去除完全重复和规范化重复，平均长度 2.46 字，中位数 2 字，最长 5 字。

项目的远期规模目标仍可作为研究方向，但不会在召回质量、标注一致性和来源完整性得到验证前机械扩容。

## 方法概览

```text
人工设计种子与类别
        │
        ▼
腾讯中文词/短语向量词表 ──► 余弦相似度召回
        │
        ▼
词形过滤、规范化与近重复检测
        │
        ├──► BGE 语义重排
        │
        ▼
分层抽样标注与融合阈值校准
        │
        ▼
人工审核的保守集与扩展集
```

项目不使用 LLM 直接生产正式词条。历史生成数据只承担“概念雷达”的作用：例如它可以提示校园信物、纸面心意或毕业意象等遗漏方向；候选词仍须由真实词表召回，并通过人工审核后才能收录。

## 数据文件

### 推荐数据

- [`data/final/expanded_sensitive_words.csv`](data/final/expanded_sensitive_words.csv)：当前完整审核版，适合直接分析。
- [`data/final/expanded_sensitive_words.jsonl`](data/final/expanded_sensitive_words.jsonl)：同一数据的流式处理格式。
- [`schema/expanded_data.schema.json`](schema/expanded_data.schema.json)：扩展审核版字段约束。

### 研究中间产物

- `data/final/conservative_sensitive_words.*`：160 条保守核心集。
- `data/final/sensitive_words.*`：腾讯向量召回的 Top 5,000 候选，不应视为全部已确认词条。
- `data/reranked/`：BGE 排序与腾讯+BGE 融合结果。
- `data/expansion/`：按概念缺口进行的定向召回候选。
- `data/legacy_generated/`：隔离的历史生成数据，仅用于审计，不建议用于训练或评测。
- `reports/`：抽样标注、阈值校准、重复检测与统计报告。

## 字段说明

推荐数据集的主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `term` | 原始词或固定/常用短语 |
| `normalized_term` | 用于去重的规范化形式 |
| `category` | 人工定义的主题类别 |
| `best_seed` | 召回时最接近的种子词 |
| `tencent_similarity` | 腾讯词向量余弦相似度 |
| `bge_similarity` | BGE 重排相似度 |
| `inclusion_route` | 进入审核集的流程路径 |
| `review_status` | 人工审核状态；正式扩展版固定为 `reviewed_keep` |

相似度反映模型与当前种子集之间的关系，不等同于客观触发强度，也不应脱离语境解释。

## 快速开始

建议使用 Python 3.10–3.12，并在仓库根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

下载当前可复现的腾讯轻量词向量镜像：

```powershell
.venv\Scripts\python scripts\download_tencent_vectors.py --variant light-mirror
```

运行基础召回和质量检查：

```powershell
.venv\Scripts\python src\tencent_retrieval.py
.venv\Scripts\python src\deduplicate.py
.venv\Scripts\python src\validate.py
.venv\Scripts\python src\qa.py
```

运行重排、融合和保守集构建：

```powershell
.venv\Scripts\python src\label_manual_samples.py
.venv\Scripts\python src\bge_rerank.py
.venv\Scripts\python src\label_bge_samples.py
.venv\Scripts\python src\calibrate_fusion.py
.venv\Scripts\python src\build_conservative.py
```

运行概念缺口扩展流程：

```powershell
.venv\Scripts\python src\targeted_expansion.py
.venv\Scripts\python src\rerank_expansion.py
.venv\Scripts\python src\label_expansion_samples.py
.venv\Scripts\python src\build_expanded_lexicon.py
.venv\Scripts\python src\validate_expanded.py
```

标注脚本会生成或读取审核文件。重新运行完整流程前，请先保留已有人工标注，避免覆盖审计结果。

## 质量控制

项目遵循以下数据约束：

- `term` 必须是词或固定/常用短语，不收录完整叙述句。
- 语义多样性来自不同概念节点，不依靠添加时间、地点、人物或程度副词制造变体。
- 2–6 字为核心长度；7–12 字允许收录常用短语；超过 12 字必须重点人工审核；超过 30 字禁止进入正式数据。
- 测试句子与词库分离，完整句子只用于检索测试。
- 完全重复、规范化重复和高相似近重复分别检查，近重复报告只提供人工复核线索。
- 边界项和未审核项不进入推荐数据集。

融合排序目前基于 423 条明确标签进行校准，分层交叉验证 ROC AUC 为 0.916、平均精确率为 0.889；这些数字来自按排名分层抽取的审核样本，不是独立同分布的正式测试集，因此只能用于内部阈值选择。完整口径见 [`docs/phase2_reranking.md`](docs/phase2_reranking.md) 和 [`reports/fusion_calibration.json`](reports/fusion_calibration.json)。

## 数据来源与可复现性

当前 MVP 使用腾讯 AI Lab 中文词向量的轻量镜像：143,613 个词/短语、200 维。BGE 重排使用 `BAAI/bge-small-zh-v1.5`，并固定模型 revision。模型文件和第三方词向量不随仓库分发。

数据来源、版本、校验方式、许可说明和当前下载限制见 [`docs/data_sources.md`](docs/data_sources.md)；类别设计见 [`docs/taxonomy.md`](docs/taxonomy.md)；完整流程见 [`docs/methodology.md`](docs/methodology.md)。

## 已知局限

- 当前轻量词表只是完整腾讯词向量 vocabulary 的子集，存在明显的未登录词问题，不能代表最终召回率。
- 分布式表示可能召回人名、作品名、机构名或仅在特定语境相关的词，模型高分不能替代人工判断。
- BGE 有助于提高头部主题集中度，但也可能错误降低成语和固定表达的排名，因此目前仅作为融合信号。
- 类别分布受种子设计影响，310 条审核集仍然偏小，不适合作为通用中文关系语义基准。
- 当前标注主要用于第一轮筛选，尚未完成多标注者一致性评估。

## 路线图

- 接入可验证来源的完整中文词/短语 vocabulary，补足轻量词表中的未登录表达。
- 为边界样本引入双人标注和一致性统计。
- 建立独立保留测试集，重新评估排序阈值与类别覆盖。
- 按概念节点扩展词库，同时维持短语长度、来源和审核约束。
- 发布版本化数据卡与变更记录。

## 使用边界

本项目主要用于幽默创作、语言学实验、文本分类、语义检索与 NLP 方法研究。数据不包含现实个人的私密记录、联系方式或可识别学校信息，也不应被用于推断个人心理状态、实施骚扰或对现实人物作自动化评价。

## 许可

仓库代码采用 [MIT License](LICENSE)。第三方模型与词向量受各自许可约束，不因本仓库的代码许可而改变；使用数据或复现实验前，请阅读 [`docs/data_sources.md`](docs/data_sources.md)。
