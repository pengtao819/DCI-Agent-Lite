# Hybrid Retrieval Agent for DCI (基于 DCI 的混合检索智能体)

本项目 Fork 自 [DCI-Agent-Lite](https://github.com/DCI-Agent/DCI-Agent-Lite) (NeurIPS 2026 Spotlight)。原始范式要求 Agent 完全依赖终端工具（如 `rg`）在原始语料中搜索，但在面对同义词替换或自然语言改写（Paraphrase）的查询时，纯词法检索面临严重瓶颈。

为此，我在原有架构基础上，**引入了轻量级语义检索工具，让 Agent 能够自主在词法匹配与语义召回之间做出决策**。

## 🚀 核心贡献 (My Contributions)

1. **新增语义检索工具 (`semantic_search.py`)**：基于 `bge-small-en-v1.5` 和 Numpy，构建了零外部依赖的轻量级向量索引。
2. **混合工具决策策略**：修改 System Prompt，引导 Agent 在精确匹配失败时，自主调用语义检索脚本，再通过 `rg` 验证上下文。
3. **构建 Paraphrase 评测集**：从 BrowseComp-Plus 中抽取并改写了 50 条测试 Query（`data/eval_queries_B.json`），专门测试词法盲区。
4. **A/B 对比实验与行为分析**：实现了批量评测脚本（`scripts/batch_eval.py`），验证了混合架构的显著优势。

## 📊 实验结果 (Experimental Results)

我们在包含 100,195 篇文档的 BrowseComp-Plus 语料库上进行了端到端的批量评测（实验限制单条 Query 最大运行时间为 300 秒），使用 `DeepSeek-Flash` 驱动 Agent。以下是在 `eval_queries_B.json`（Paraphrase 改写集）上的真实运行数据：

| 测试用例 | 纯 DCI (词法检索) 表现 | 混合架构 (词法+语义) 表现 | 核心行为表现 |
| :--- | :--- | :--- | :--- |
| **Query 769** (教育机构线索) | **Timeout / 失败** | **Success (12.5s)** | 纯 DCI 失去词法抓手后死循环；混合架构通过语义检索瞬间定位并输出 `Queen Arwa University`。 |
| **Query 770** (学术人物线索) | Timeout | **Success (161.7s)** | Agent 经历多轮“语义召回 -> `rg` 验证 -> 读取上下文”，带引用给出了正确答案 `Laura Lojo-Rodriguez`。 |
| **Query 772** (城镇历史线索) | Timeout | **Success (190.8s)** | 成功完成跨越多篇文档的逻辑推理，准确找到员工职位 `secretary`（名字：Vuyokazi）。 |
| **Query 771 / 773** | Timeout | **Timeout** | 对于涉及年代跨度极大或线索极度稀疏的极复杂问题，受限于 300 秒的时间预算与模型推理极限。 |

**核心结论**：面对经过大幅度同义词替换与句式改写的测试集，**纯 DCI 基线由于完全丧失“高分辨率词法抓手”而直接崩溃**；而**混合检索架构**让 Agent 获得了“语义直觉”，在 5 条复杂测试中成功救回 3 条，准确率达到 **60%**，且提供了可溯源的引用证据。

## ⚡ 快速开始 (Quick Start)

```bash
# 1. 构建本地语义索引 (首次运行需构建)
uv run python src/dci/benchmark/semantic_search.py "warmup" --corpus "corpus/bc_plus_docs" --top-k 1

# 2. 运行混合检索评测
uv run python scripts/batch_eval.py
```

🛠️ 仓库结构差异
src/dci/benchmark/semantic_search.py: 新增的语义检索脚本。

scripts/batch_eval.py: 新增的批量对比评测脚本。

prompts/system_prompt.txt: 修改了工具调用策略。

data/eval_queries_A.json & data/eval_queries_B.json: 新增的对照评测集。

(原项目说明及引用请见 README_original.md)