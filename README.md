# 购物中心经营风险智能决策 Agent

这是一个面向购物中心运营、财务、法务和招商团队的经营风险智能决策 Agent。系统目标不是简单展示风险看板，而是基于门店经营数据、历史趋势、合同履约、巡检记录、投诉记录和知识库证据，持续识别经营风险、解释风险来源、生成受审批约束的处置建议，并将每次 Agent 运行过程完整入库，支持审计、复盘和规则迭代。

当前版本已经具备：

- 100 家门店经营数据看板
- 12 个月门店经营历史数据和近 3/6 个月趋势特征
- 可解释风险评分与分级
- 风险触发依据和可解释诊断
- 面向管理层的自然语言查询
- SQLite 持久化 RAG 数据库和结构化 Prompt
- OpenAI tool calling：结构化风险查询、RAG 检索、行动草稿工具
- 后端审计入库：会话、消息、Agent run、步骤、证据和模型版本
- Verifier：格式、数值、审批、引用和角色权限校验
- 待审批行动草稿与人工确认机制
- CSV 上传与数据导出

当前阶段正在从“规则原型”升级为“数据校准 + AI 受控推理”的生产化版本：

- 已生成并接入 12 个月历史经营数据，用于识别连续下滑、季节波动和同业态基准。
- 将评分规则从代码中抽离为规则配置表，记录阈值来源、版本、owner 和审批记录。
- 基于历史分布动态生成阈值，例如同业态 P75/P90、租金倍数、欠费账龄和保证金覆盖率。
- 让大模型参与证据推理和评分调整建议，但最终调整受规则引擎、Verifier 和人工审批约束。
- 将 prompt 改造为结构化推理协议，先输出 JSON，再由系统渲染管理层报告。

## 1. 环境准备

建议使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 启动

在项目目录中执行：

```bash
streamlit run frontend/streamlit_app.py
```

浏览器通常会自动打开：

```text
http://localhost:8501
```

前端主入口只保留 `frontend/streamlit_app.py`，根目录 `app.py` 仅作为兼容启动器转发到该文件。

## 3. 可选后端服务与审计入库

当前系统已加入轻量后端记录层：

- `backend/store.py`：本地 SQLite 入库模块，自动保存用户会话、对话消息、Agent run、工作流步骤和 RAG 证据。
- `backend/api.py`：FastAPI 后端 API 原型，可用于后续生产架构拆分。
- `backend_db/mall_agent_backend.sqlite3`：运行时自动生成，已加入 `.gitignore`，不会提交用户对话记录。

Streamlit 页面可单独运行，并会直接写入后端 SQLite。若需要单独启动后端 API：

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

可访问：

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/stats
http://127.0.0.1:8000/runs/recent
```

多用户并发设计上，每个浏览器会话会生成独立 `user_id` 和 `session_id`，每次运行 Agent 都会生成 `run_id`。生产版本可将 SQLite 替换为 PostgreSQL，将长任务放入 Celery/RQ 队列，并通过 SSE/WebSocket 给前端返回流式进度。

## 4. 可选接入真实 AI

当前 AI 助理已经支持 SQLite 持久化 RAG：会先检索门店经营数据、合同条款样例、运营 SOP、招商策略和 `interview_source_extract.txt` 中的运营知识，再生成回答。未配置 API key 时，系统会使用本地规则兜底，仍可展示引用证据。

启用真实 AI 后，系统会优先使用 OpenAI tool calling。当前注册的工具包括：

- `query_structured_risk_data`：查询门店结构化风险数据。
- `search_rag_evidence`：检索本地 RAG 知识库证据。
- `create_action_draft`：生成受审批约束的行动草稿。

如果 tool calling 或模型服务不可用，会自动回退到原本的本地 Agentic Workflow。

Verifier 当前包含五类规则校验：

- 格式校验：检查固定标题和结论段引用编号。
- 数值校验：检查重点门店、风险得分、欠费金额是否能对应结构化查询结果。
- 审批校验：催缴函、解约、保证金扣划、品牌替换等敏感动作必须标注“需人工审批”。
- 引用校验：关键证据需要绑定 RAG evidence 或结构化结果引用编号。
- 权限校验：根据当前用户角色限制法务、招商、财务等敏感动作输出边界。

如需启用真实模型生成，创建 `.streamlit/secrets.toml`。使用火山方舟/豆包时：

```toml
ARK_API_KEY = "你的 Ark API key"
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "你的方舟模型 ID 或 Endpoint ID"
ARK_MODELS = "你的方舟模型 ID,备用模型 ID"
ARK_EMBEDDING_MODEL = "你的方舟 Embedding 模型 ID"
ARK_VISION_MODEL = "你的方舟视觉模型 ID 或 Endpoint ID"
```

也可以继续使用 OpenAI 兼容配置：

```toml
OPENAI_API_KEY = "你的 OpenAI API key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "你的模型名"
OPENAI_MODELS = "你的模型名,备用模型名"
OPENAI_EMBEDDING_MODEL = "你的 Embedding 模型名"
```

## 5. RAG 数据库与知识文件

RAG 知识源：

- `data/购物中心100家门店经营数据_优化命名.csv`：默认门店经营数据
- `data/门店经营月度历史数据_12个月.csv`：门店 12 个月历史经营数据，用于趋势分析和动态阈值校准
- `interview_source_extract.txt`：Agent Design Proposal Draft
- `rag_docs/`：合同条款样例、催缴规则、巡检 SOP、招商替换策略
- `rag_sources/`：可直接投放 `.txt/.md/.csv/.xlsx/.docx/.pdf/.png/.jpg` 等外部资料，然后在页面点击“重建 RAG 数据库”。图片会优先调用 `ARK_VISION_MODEL`、`ARK_MODEL`、`OPENAI_VISION_MODEL` 或 `OPENAI_MODEL` 做 OCR/视觉解析，失败时保留图片元数据。

应用启动后会构建：

- `rag_db/mall_rag.sqlite3`

数据库包含 `chunks` 表和 `chunks_fts` 全文索引。检索时会结合本地哈希向量相似度和 SQLite FTS 关键词召回。

## 6. 使用自己的 CSV

可以直接在侧边栏上传 CSV。至少需要下列字段：

- 门店ID
- 门店名称
- 业态分类
- 楼层
- 本月销售额
- 销售环比(%)
- 进店率(%)
- 成交转化率(%)
- 租售比(%)
- 欠费总额(元)
- 水电费波动(%)

原始文件中的“风险得分”和“风险等级”可以保留，但系统会重新计算 `计算风险得分` 和 `计算风险等级`，方便对比。

## 7. 当前风险评分引擎

当前版本使用可解释专家规则作为初始评分引擎，目标是让每个风险分都能追溯到具体指标和触发依据：

- 欠费金额：0–35 分
- 租售比：0–30 分
- 销售环比：0–25 分
- 水电费异常下降：0–20 分
- 进店率偏低：8 分
- 转化率偏低：7 分

这些阈值不是最终生产规则，而是第一版专家规则基线。生产版本会将规则迁移到 `risk_rule_config` 或数据库规则表中，记录：

- 指标名称、阈值条件、分值影响
- 阈值来源：历史分位数、业务专家、租金倍数、合同条款或监管要求
- 规则版本、owner、启用时间和审批记录
- 回测效果：命中率、误报率、漏报率和人工反馈

当前仓库已加入 `risk_rule_config.json` 和 `risk_rules.py`，用于记录并读取现有基线规则、阈值来源说明、未来动态阈值计算方法和 AI 调整分约束。`risk_score()` 已开始从硬编码迁移为读取配置，欠费金额、租售比、销售环比和水电异常规则已经走配置化阈值。

最终分数封顶 100：

- 0–29：低风险
- 30–54：中风险
- 55–74：高风险
- 75–100：极高风险

## 8. 历史数据与动态阈值路线

经营风险不应只依赖单月截面。生产版本至少需要 12 个月历史数据：

- 3 个月只能观察短期波动，容易受节假日、活动档期和天气影响。
- 6 个月可以识别连续下滑、欠费累积和租售比压力，但季节性仍不充分。
- 12 个月可以覆盖淡旺季、季度活动和同比变化，是形成相对稳定风险判断的最低建议周期。
- 如果有 24 个月数据，可以进一步做同比、季节性校准和品牌生命周期分析。

动态阈值会按业态、楼层、租金等级、面积和经营场景分组计算，例如：

- 欠费阈值：同业态近 12 个月欠费金额 P80/P90，叠加欠费账龄和保证金覆盖率。
- 租售比阈值：同业态、同楼层近 12 个月租售比分布 P75/P90。
- 销售下滑阈值：门店自身历史波动区间 + 同业态基准。
- 投诉阈值：同业态投诉率均值和标准差，识别异常偏离。

AI 不直接自由决定最终风险分，而是在规则基础分上给出受控调整建议：

- `base_score`：规则或统计模型计算。
- `reasoning_summary`：模型基于评分标准、门店指标、历史趋势和证据生成可审计推理摘要。
- `adjustment_proposal`：模型提出有限幅度调整，例如 `-10` 到 `+10`，不得突破配置上限。
- `final_score`：由规则引擎和 Verifier 校验后生成，必要时进入人工审批。

当前前端已加入 `Scoring Reviewer` 节点：规则引擎先计算基础分，AI 再读取 `risk_rule_config.json`、结构化门店指标、近 3/6 个月历史趋势和 RAG evidence，输出建议调整分、建议分、置信度、证据依据和可审计推理摘要。系统不展示模型隐藏 chain-of-thought，而展示可复核的业务推理摘要；如果建议分导致风险等级变化，会标记人工复核。

## 9. 上传到 GitHub 前检查

当前 `.gitignore` 已排除本地虚拟环境、密钥文件、运行时数据库、向量库、缓存、日志和评估最新结果。上传前建议再确认：

- 不提交 `.streamlit/secrets.toml`、`.env` 或任何包含真实 API key 的文件。
- `backend_db/` 和 `rag_db/` 是运行时生成目录，不需要提交。
- `evaluation/latest_results.json` 只代表本机最近一次评估结果，建议通过命令重新生成，不作为固定项目资料提交。
- `data/`、`rag_docs/` 和 `rag_sources/` 当前用于演示数据和样例知识库；如果替换为真实商户、合同、财务或投诉资料，先完成脱敏。
- `.vscode/` 等个人编辑器配置默认忽略；如果已经被 Git 跟踪，可按需执行 `git rm --cached .vscode/settings.json` 后再提交。

基础自检命令：

```bash
python -m pytest
python -m evaluation.run_evaluation
```

## 10. 项目定位

当前版本使用生成数据和专家规则基线来验证完整系统链路：数据感知、动态风险判断、证据检索、AI 受控推理、行动草稿、人工审批和审计入库。当前已接入 12 个月历史数据和部分配置化评分规则，后续迭代重点是引入历史回测、模型评估集、动态阈值计算服务，并接入真实工单、合同、财务和招商品牌系统。
