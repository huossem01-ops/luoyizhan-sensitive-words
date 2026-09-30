# 数据来源与许可

正式检索结果只从真实词表或预训练词/短语向量的 vocabulary 中召回。LLM 不作为候选词来源。

## Tencent AI Lab Embedding Corpus for Chinese Words and Phrases

- 官方说明：https://ailab.tencent.com/ailab/nlp/en/embedding.html
- 官方旧版下载地址：https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz
- 上游版本：官方页面标注中文语料 v0.2.0（2021-12-24），提供 100/200 维、超过 1,200 万中文词和短语。
- 许可：官方页面标注 Creative Commons Attribution 3.0 Unported，并同时声明仅供研究使用。本项目按更严格的“研究用途”限制处理。
- 署名要求：引用 Yan Song, Shuming Shi, Jing Li, and Haisong Zhang, *Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings*, NAACL 2018。
- 原始规模：官方最新版超过 12,000,000 条；旧版资料记载 8,824,330 条、200 维。
- 原始文件：不提交到本仓库，由用户本地下载。

### 当前 MVP 实际输入

- 镜像页：https://huggingface.co/shibing624/text2vec-word2vec-tencent-chinese
- 固定 revision：`b7b9fccfd5dd34cfc340607c58986ac2970e3a13`
- 文件：`light_Tencent_AILab_ChineseEmbedding.bin`
- 镜像标注：Apache-2.0；但其内容源自腾讯向量，本项目仍沿用上游 CC BY 3.0 / 研究用途约束，不接受镜像方对底层数据的重新许可作为唯一依据。
- 原始词条数量：143,613；维度：200。
- 最终采用数量：由每次 QA 后的 `reports/statistics.json` 记录；第一阶段最多 5,000。
- 限制：轻量镜像不是腾讯完整 vocabulary，只用于验证 pipeline 和第一轮检索质量，不能代表完整语义空间的最终召回率。

官方旧下载端点在 2026-10-01 的本地检查中返回 HTML 页面而不是压缩文件，因此下载脚本会校验文件类型并拒绝把 HTML 当作向量包。待官方端点恢复或用户提供已下载的官方文件后，可直接切换配置运行完整 vocabulary。

## BGE（第二阶段已使用）

- 模型：https://huggingface.co/BAAI/bge-small-zh-v1.5
- 许可：MIT。
- 固定 revision：`7999e1d3359715c523056ef9478215996d62a620`。
- 输出维度：512。
- 用途：对当前腾讯 Top 5,000 候选重新编码和排序，不生成新词，也不对腾讯千万级完整 vocabulary 重新编码。
- 编码设置：词条和种子均视为对称的短文本语义相似度任务，不添加 query instruction，向量进行 L2 normalize 后计算 cosine similarity。
- 实验结论：BGE 提升了 Top 100 的主题集中度，但会把部分稳定成语和固定表达降到尾部，因此最终使用腾讯+BGE 融合信号，不直接替换腾讯排序。
