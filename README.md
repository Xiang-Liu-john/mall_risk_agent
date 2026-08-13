# 购物中心经营风险预警 Agent Demo

这是一个可本地运行的 Streamlit MVP，展示：

- 100 家门店经营数据看板
- 确定性风险评分与分级
- 风险触发依据和可解释诊断
- 面向管理层的自然语言查询
- SQLite 持久化 RAG 数据库和结构化 Prompt
- 待审批行动草稿与人工确认机制
- CSV 上传与数据导出

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
streamlit run app.py
```

浏览器通常会自动打开：

```text
http://localhost:8501
```

## 3. 可选接入真实 AI

当前 AI 助理已经支持 SQLite 持久化 RAG：会先检索门店经营数据、模拟合同条款、运营 SOP、招商策略和 `interview_source_extract.txt` 中的运营知识，再生成回答。未配置 API key 时，系统会使用本地规则兜底，仍可展示引用证据。

如需启用真实模型生成，创建 `.streamlit/secrets.toml`：

```toml
OPENAI_API_KEY = "你的 OpenAI API key"
OPENAI_BASE_URL = "https://x.ailzd.com/v1"
OPENAI_MODEL = "gpt-5.4-mini"
OPENAI_MODELS = "gpt-5.4-mini,gpt-5.4,gpt-5.2"
```

## 4. RAG 数据库与知识文件

RAG 知识源：

- `data/购物中心100家门店经营数据_优化命名.csv`：默认门店经营数据
- `interview_source_extract.txt`：Agent Design Proposal Draft
- `rag_docs/`：模拟合同条款、催缴规则、巡检 SOP、招商替换策略
- `rag_sources/`：可直接投放 `.txt/.md/.csv/.xlsx/.docx/.pdf/.png/.jpg` 等外部资料，然后在页面点击“重建 RAG 数据库”。图片会优先调用 `OPENAI_VISION_MODEL` 做 OCR/视觉解析，失败时保留图片元数据。

应用启动后会构建：

- `rag_db/mall_rag.sqlite3`

数据库包含 `chunks` 表和 `chunks_fts` 全文索引。检索时会结合本地哈希向量相似度和 SQLite FTS 关键词召回。

## 5. 使用自己的 CSV

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

## 6. 风险评分规则

当前版本为可解释 MVP，使用固定规则：

- 欠费金额：0–35 分
- 租售比：0–30 分
- 销售环比：0–25 分
- 水电费异常下降：0–20 分
- 进店率偏低：8 分
- 转化率偏低：7 分

最终分数封顶 100：

- 0–29：低风险
- 30–54：中风险
- 55–74：高风险
- 75–100：极高风险

## 7. 面试时应说明

当前版本使用模拟数据和确定性规则，目标是验证“数据感知—风险判断—原因解释—行动草稿—人工审批”的产品闭环。生产版本需要接入历史时序数据、同业态基准、合同与保证金、巡店日志、RAG 知识库及真实工单系统。
