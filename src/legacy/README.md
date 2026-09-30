# Deprecated generation pipeline

本目录保留早期 LLM/规则组合式生成代码，仅供结果对照和错误复盘。它不是正式数据来源，也不得用于补齐 48,000 条目标。

新的主流程位于 `src/tencent_retrieval.py`：真实腾讯词/短语向量 vocabulary → 多 seed cosine 检索 → 清洗/去重 → QA。
