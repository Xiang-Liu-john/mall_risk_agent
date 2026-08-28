# 项目技术说明书：购物中心经营风险智能决策 Agent

审计时间：2026-08-21  
审计范围：当前仓库代码、RAG 模块、Streamlit 前端、SQLite/FastAPI 后端、evaluation harness、测试文件和本轮实际命令结果。  
审计原则：只记录代码中已经实现、或当前环境实际验证成功的事实；不虚构模型、API、benchmark 或生产能力。

## 1. 当前定位

当前项目定位为：**可评测、可解释、可审计、带 Human-in-the-loop 的 Agentic BI Prototype**。

核心链路：

```text
Planner
→ Structured Query
→ Hybrid RAG
→ Scoring Reviewer
→ Generator
→ Verifier
→ Human Approval
→ Audit Persistence / Evaluation
```

本轮未增加 MCP、LangGraph、Multi-Agent、Voice、多模态、新前端框架或 Kubernetes。

## 2. UI 与 Core 解耦

已新增 UI-independent core module：

```text
core/
  __init__.py
  agent_logic.py
```

当前依赖方向：

```text
Streamlit UI
      ↓
Core Agent Logic
      ↑
Evaluation
```

`evaluation/run_evaluation.py` 不再 import `frontend.streamlit_app`，因此命令行 evaluation 不再触发 Streamlit `missing ScriptRunContext` warning。

Streamlit 仍负责：

- 页面渲染
- `session_state`
- 上传/筛选 UI
- Agent 运行按钮与状态展示
- Human Approval Prototype 渲染

Core 负责：

- CSV 清洗
- 风险评分
- Planner
- Structured Query
- RAG context assembly
- Scoring Reviewer fallback
- Verifier
- approval draft persistence helper

## 3. Embedding

实现文件：

- `rag/embedding_service.py`
- `rag/embedding_validation.py`

当前配置读取到：

| 配置项 | 状态 |
|---|---|
| `ARK_API_KEY` | 已配置 |
| `ARK_BASE_URL` | `https://ark.cn-beijing.volces.com/api/v3` |
| `ARK_EMBEDDING_MODEL` | `doubao-embedding-text-240715` |

本轮实际验证结果：

```json
{
  "embedding_provider": "ark_openai_compatible",
  "embedding_model": "doubao-embedding-text-240715",
  "embedding_dimension": null,
  "ark_base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "api_call_succeeded": false,
  "qdrant_write_succeeded": false,
  "qdrant_similarity_search_succeeded": false,
  "production_ready": false,
  "reason": "Embedding API call failed: Error code: 404 - InvalidEndpointOrModel.NotFound; the model or endpoint doubao-embedding-text-240715 does not exist or the current API Key does not have access."
}
```

结论：

- 当前环境已按火山方舟 OpenAI-compatible embeddings API 示例配置 `base_url`、`model` 和 `encoding_format="float"`。
- 已真实发送 Ark embedding 请求，但 Ark 服务端返回 `InvalidEndpointOrModel.NotFound`，说明当前账号/API Key 对该文本 embedding 模型或 endpoint 仍无可用访问权限。
- 已尝试文本模型 `doubao-embedding-text-240715`、历史版本 `doubao-embedding-text-240515`、vision 模型 `doubao-embedding-vision-251215` 及其预置接入点；vision endpoint 返回“不支持当前 embeddings API”，因此不适合作为当前文本 RAG 的生产 embedding。
- 系统继续使用 `hashing_fallback`，维度为 96，`production_ready=false`。
- 文档和简历事实中不能写“已启用真实 semantic embedding”。

可安全对外表述：

```text
实现了 Ark/OpenAI-compatible embedding provider、hashing fallback、Qdrant 写入与 similarity search 验证链路；当前因火山方舟账号/模型 endpoint 权限未放开，semantic embedding 未标记为 production_ready，系统自动降级到 hashing fallback。
```

## 4. Vector DB / Qdrant

实现文件：`rag/retriever.py`

实际安装版本：

```text
qdrant-client 1.19.0
```

本轮修复：

- 不再调用 deprecated `recreate_collection()`。
- 改为：

```text
collection_exists()
delete_collection()
create_collection()
```

- dense search 改为新版 `query_points()`。
- 增加 Qdrant local client 显式 close 和 evaluation 进程内检索 client 复用，避免重复初始化拖慢 benchmark。

当前 Qdrant local path：

```text
rag_db/qdrant
```

由于 Ark semantic embedding API 在当前账号/API Key 下未成功返回向量，Qdrant 当前写入的是 hashing fallback vector，不应称为真实 semantic vector index。

## 5. Retrieval

实现文件：

- `rag/retriever.py`
- `rag/retrieval_config.json`
- `rag_store.py`

当前默认配置保持为：

```json
{
  "dense_weight": 0.62,
  "lexical_weight": 0.38,
  "rerank_top_n": 20,
  "final_top_k": 8,
  "dense_top_k": 40,
  "lexical_top_k": 40,
  "qdrant_collection": "mall_risk_chunks",
  "embedding_provider": "ark_openai_compatible",
  "reranker_provider": "disabled"
}
```

Observability 字段已包含：

- `embedding_provider`
- `embedding_model`
- `embedding_dimension`
- `vector_db`
- `dense_retrieval_status`
- dense / lexical / merged / reranked counts
- dense / lexical / fusion / rerank scores

本轮 comparison 事实：

- Baseline A `hashing_fts` 已实际评测。
- Candidate B `semantic_dense` skipped，因为 semantic embedding 未验证。
- Candidate C `semantic_hybrid` skipped，因为 semantic embedding 未验证。
- 因 semantic 未 ready，本轮没有执行 semantic 参数调优，也没有声称当前权重来自 semantic Golden Dataset tuning。

## 6. Reranker

实现文件：`rag/reranker.py`

当前状态：

```text
reranker_provider = disabled
```

已保留：

- `RerankerService`
- `DisabledRerankerService`

未接入、未验证任何真实 reranker API / 模型。不能写“已使用 reranking model”。

## 7. Evaluation

实现文件：

```text
evaluation/
  golden_dataset.json
  evaluate_agent.py
  evaluate_generation.py
  evaluate_retrieval.py
  run_evaluation.py
  latest_results.json
  evaluation_comparison.json
```

Golden Dataset：

```text
45 cases
```

覆盖：

- 单条件：欠费、销售下滑、租售比、投诉、保证金、欠费账龄、综合风险、门店、楼层、业态
- 多条件：楼层 + 业态 + 欠费、业态 + 销售下滑、高租售比 + 欠费表述、欠费 + 保证金、楼层 + 高风险、业态 + 投诉
- 边界问题：销售下降但无欠费、欠费但经营指标正常、保证金充足但租售比异常、不存在门店、信息不足
- 行为约束：只分析原因、不执行催缴、不生成法务行动、列出证据、只返回前三家
- 模糊自然语言：经营不太正常、需要重点盯、现金流压力较大

`evaluation/run_evaluation.py` 会实际写入：

- `evaluation/latest_results.json`
- `evaluation/evaluation_comparison.json`

## 8. 当前实际指标

命令：

```bash
.\.venv\Scripts\python.exe -m evaluation.run_evaluation
```

结果来源：`evaluation/latest_results.json`

| 指标 | 当前值 |
|---|---:|
| Golden Dataset case count | 45 |
| Intent Accuracy | 0.9333333333 |
| Filter Accuracy | 1.0 |
| Tool Selection Accuracy | 1.0 |
| Task Success Rate | 0.9333333333 |
| Hit@3 | 0.3777777778 |
| Recall@3 | 0.2240740741 |
| Hit@5 | 0.5777777778 |
| Recall@5 | 0.3685185185 |
| Hit@8 | 0.7555555556 |
| Recall@8 | 0.5277777778 |
| MRR | 0.3558201058 |
| Required Section Compliance | 1.0 |
| Citation Presence | 1.0 |
| Approval Boundary Compliance | 0.9777777778 |
| Store Fact Coverage | 1.0 |

Comparison 来源：`evaluation/evaluation_comparison.json`

| Config | Status | Hit@8 | Recall@8 | MRR |
|---|---|---:|---:|---:|
| `hashing_fts` | evaluated | 0.4444444444 | 0.2685185185 | 0.4222222222 |
| `semantic_dense` | skipped | n/a | n/a | n/a |
| `semantic_hybrid` | skipped | n/a | n/a | n/a |

当前 configured hybrid 的 retrieval 指标来自 hashing fallback + FTS hybrid，不是 semantic hybrid。

## 9. Testing

本轮实际运行：

```bash
.\.venv\Scripts\python.exe -m py_compile core\agent_logic.py frontend\streamlit_app.py evaluation\run_evaluation.py rag\retriever.py rag\embedding_validation.py rag\embedding_service.py rag_store.py tests\test_agent_logic.py tests\test_retrieval.py tests\test_evaluation_metrics.py tests\test_approval_state.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m evaluation.run_evaluation
```

pytest 结果：

```text
13 passed in 1.25s
```

Warnings：

```text
0 pytest warnings
```

已修复原 Qdrant deprecation warning。

## 10. Production Boundary

当前可以安全称为：

```text
Agentic BI Prototype with deterministic evaluation, auditable workflow, HITL approval prototype, local hybrid retrieval, and Qdrant-backed vector storage abstraction.
```

当前不能称为：

- 完全 Production System
- 已验证真实 semantic embedding 的 RAG
- 已接入真实 reranker model
- Multi-Agent / Autonomous Agent
- 已接入企业 MCP / 工作流系统

仍未实现或未验证：

- `ARK_EMBEDDING_MODEL=doubao-embedding-text-240715` 在当前账号/API Key 下的真实可用性
- semantic embedding API 成功调用
- semantic vector 写入 Qdrant 后的真实 similarity search
- semantic dense / semantic hybrid benchmark
- reranker before/after benchmark
- 生产鉴权、租户隔离、CI/CD、监控和任务队列
- 基于真实业务历史的阈值回测与校准
