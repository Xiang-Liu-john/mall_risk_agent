from __future__ import annotations

import json
import os
import re
import time
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from backend.store import create_approval
from rag_store import build_rag_database, search_rag_database_with_debug
from risk_rules import load_rule_config, threshold_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
SECRETS_PATH = PROJECT_DIR / ".streamlit" / "secrets.toml"

DEFAULT_CSV = PROJECT_DIR / "data" / "购物中心100家门店经营数据_优化命名.csv"

REQUIRED_COLUMNS = [
    "门店ID",
    "门店名称",
    "业态分类",
    "楼层",
    "本月销售额",
    "销售环比(%)",
    "进店率(%)",
    "成交转化率(%)",
    "租售比(%)",
    "欠费总额(元)",
    "水电费波动(%)",
]

NUMERIC_COLUMNS = [
    "租赁面积(sqm)",
    "本月销售额",
    "销售环比(%)",
    "进店率(%)",
    "成交转化率(%)",
    "租售比(%)",
    "欠费总额(元)",
    "欠费天数",
    "保证金覆盖率(%)",
    "近3月平均销售",
    "近6月平均销售",
    "近3月销售环比均值",
    "近6月最高欠费",
    "近6月平均租售比",
    "连续下滑月数",
    "水电费波动(%)",
    "近90天投诉数",
    "退款申请数",
    "续费率(%)",
    "安全巡检分",
    "家长投诉率(‰)",
    "风险得分",
]

LEVEL_ORDER = ["极高", "高", "中", "低"]
LEVEL_ICON = {"极高": "🔴", "高": "🟠", "中": "🟡", "低": "🟢"}
HIGH_RISK_LEVELS = ["高", "极高"]
LEVEL_COLORS = {
    "低": "#2E7D32",
    "中": "#F9A825",
    "高": "#F57C00",
    "极高": "#C62828",
}
TOP_RISK_COLORS = ["#FFF1F2", "#FDA4AF", "#F43F5E", "#B91C1C"]
CATEGORY_RISK_COLORS = ["#E3F2FD", "#64B5F6", "#1E88E5", "#0D47A1"]
CATEGORY_COLORS = {
    "餐饮": "#3B82F6",
    "精品零售": "#10B981",
    "儿童配套": "#F59E0B",
    "主力店": "#EF4444",
    "生活服务": "#8B5CF6",
    "潮流运动": "#06B6D4",
}


@dataclass(frozen=True)
class Action:
    owner: str
    task: str
    approval_required: bool
    deadline: str


@dataclass(frozen=True)
class RagDoc:
    source: str
    title: str
    content: str
    row_id: str | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class AgentStep:
    name: str
    detail: str
    status: str = "完成"


@dataclass(frozen=True)
class AgentPlan:
    intent: str
    intent_label: str
    filters: dict[str, str]
    sort_column: str
    ascending: bool
    limit: int
    needs_policy: bool
    needs_approval_guard: bool


@dataclass(frozen=True)
class AIScoreAssessment:
    store_name: str
    base_score: int
    adjustment: int
    adjusted_score: int
    adjusted_level: str
    reasoning_summary: str
    evidence_basis: str
    confidence: int
    requires_human_review: bool


def read_csv_robust(source) -> pd.DataFrame:
    """读取 UTF-8/GBK CSV，并给出清晰错误。"""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"无法识别 CSV 编码：{last_error}")


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("CSV 缺少必要字段：" + "、".join(missing))

    result = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    required_numeric = [c for c in NUMERIC_COLUMNS if c in REQUIRED_COLUMNS]
    bad_rows = result[required_numeric].isna().any(axis=1)
    if bad_rows.any():
        result = result.loc[~bad_rows].copy()
        warnings.warn(f"已忽略 {int(bad_rows.sum())} 行关键数值缺失或格式错误的数据。", RuntimeWarning, stacklevel=2)

    result["门店ID"] = result["门店ID"].astype(str)
    result["门店名称"] = result["门店名称"].astype(str)
    result["业态分类"] = result["业态分类"].astype(str)
    result["楼层"] = result["楼层"].astype(str)
    optional_defaults = {
        "经营场景": "未标注",
        "欠费天数": 0,
        "保证金覆盖率(%)": 100,
        "近3月平均销售": 0,
        "近6月平均销售": 0,
        "近3月销售环比均值": 0,
        "近6月最高欠费": 0,
        "近6月平均租售比": 0,
        "连续下滑月数": 0,
        "近90天投诉数": 0,
        "退款申请数": 0,
        "续费率(%)": 100,
        "安全巡检分": 100,
        "家长投诉率(‰)": 0,
    }
    for column, default in optional_defaults.items():
        if column not in result.columns:
            result[column] = default
    return result


def optional_float(row: pd.Series, column: str, default: float = 0) -> float:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def risk_score(row: pd.Series) -> tuple[int, list[str], dict[str, int]]:
    reasons: list[str] = []
    business_score = 0
    financial_score = 0
    operation_score = 0
    contract_score = 0

    arrears = float(row["欠费总额(元)"])
    arrears_score = threshold_score("arrears_amount", arrears)
    financial_score += arrears_score
    if arrears >= 50_000:
        reasons.append(f"欠费金额达到 ¥{arrears:,.0f}，超过 5 万元")
    elif arrears >= 10_000:
        reasons.append(f"欠费金额达到 ¥{arrears:,.0f}，超过 1 万元")
    elif arrears > 0:
        reasons.append(f"存在欠费 ¥{arrears:,.0f}")

    rent_ratio = float(row["租售比(%)"])
    financial_score += threshold_score("rent_to_sales_ratio", rent_ratio)
    if rent_ratio > 35:
        reasons.append(f"租售比 {rent_ratio:.1f}%，严重高于警戒区间")
    elif rent_ratio > 25:
        reasons.append(f"租售比 {rent_ratio:.1f}%，超过 25% 警戒线")
    elif rent_ratio >= 18:
        reasons.append(f"租售比 {rent_ratio:.1f}%，进入观察区间")

    mom = float(row["销售环比(%)"])
    business_score += threshold_score("sales_mom_decline", mom)
    if mom <= -30:
        reasons.append(f"销售环比下降 {abs(mom):.1f}%")
    elif mom <= -20:
        reasons.append(f"销售环比下降 {abs(mom):.1f}%")
    elif mom <= -10:
        reasons.append(f"销售环比下降 {abs(mom):.1f}%")

    utility = float(row["水电费波动(%)"])
    operation_score += threshold_score("utility_drop", utility)
    if utility <= -40:
        reasons.append(f"水电费下降 {abs(utility):.1f}%，需核查异常营业或撤店迹象")
    elif utility <= -20:
        reasons.append(f"水电费下降 {abs(utility):.1f}%")

    consecutive_declines = optional_float(row, "连续下滑月数")
    recent_mom_avg = optional_float(row, "近3月销售环比均值")
    if consecutive_declines >= 3:
        business_score += 8
        reasons.append(f"近3个月连续销售下滑，近3月环比均值 {recent_mom_avg:.1f}%")
    elif consecutive_declines >= 2 and recent_mom_avg < 0:
        business_score += 4
        reasons.append(f"近3个月中有 {consecutive_declines:.0f} 个月销售下滑")

    six_month_rent_ratio = optional_float(row, "近6月平均租售比")
    if six_month_rent_ratio >= 30 and rent_ratio >= 25:
        financial_score += 4
        reasons.append(f"近6月平均租售比 {six_month_rent_ratio:.1f}%，租金压力具有持续性")

    entry = float(row["进店率(%)"])
    if entry < 5:
        business_score += 6
        reasons.append(f"进店率仅 {entry:.1f}%")

    conversion = float(row["成交转化率(%)"])
    if conversion < 8:
        business_score += 6
        reasons.append(f"成交转化率仅 {conversion:.1f}%")

    complaints_90d = optional_float(row, "近90天投诉数")
    if complaints_90d >= 8:
        operation_score += 6
        reasons.append(f"近90天投诉数达到 {complaints_90d:.0f} 次")
    elif complaints_90d >= 4:
        operation_score += 3
        reasons.append(f"近90天投诉数达到 {complaints_90d:.0f} 次")

    if row["业态分类"] == "儿童配套":
        refund_count = optional_float(row, "退款申请数")
        renewal_rate = optional_float(row, "续费率(%)", 100)
        safety_score = optional_float(row, "安全巡检分", 100)
        parent_complaint_rate = optional_float(row, "家长投诉率(‰)")

        if refund_count >= 6:
            business_score += 4
            reasons.append(f"退款申请数达到 {refund_count:.0f} 次")
        if renewal_rate < 55:
            business_score += 6
            reasons.append(f"续费率降至 {renewal_rate:.1f}%")
        if safety_score < 82:
            operation_score += 5
            reasons.append(f"安全巡检分仅 {safety_score:.1f}")
        if parent_complaint_rate >= 5:
            operation_score += 4
            reasons.append(f"家长投诉率达到 {parent_complaint_rate:.1f}‰")

    arrears_days = optional_float(row, "欠费天数")
    if arrears_days >= 60:
        contract_score += 6
        reasons.append(f"欠费账龄达到 {arrears_days:.0f} 天")
    elif arrears_days >= 30:
        contract_score += 4
        reasons.append(f"欠费账龄达到 {arrears_days:.0f} 天")

    deposit_cover = optional_float(row, "保证金覆盖率(%)", 100)
    if deposit_cover < 60:
        contract_score += 4
        reasons.append(f"保证金覆盖率仅 {deposit_cover:.1f}%")
    elif deposit_cover < 90:
        contract_score += 2
        reasons.append(f"保证金覆盖率 {deposit_cover:.1f}%，低于稳健区间")

    sub_scores = {
        "经营风险分": min(40, business_score),
        "财务风险分": min(30, financial_score),
        "运营风险分": min(20, operation_score),
        "合同风险分": min(10, contract_score),
    }
    total_score = min(100, sum(sub_scores.values()))
    return total_score, reasons, sub_scores


def score_to_level(score: int) -> str:
    if score >= 75:
        return "极高"
    if score >= 55:
        return "高"
    if score >= 30:
        return "中"
    return "低"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    calculated = result.apply(risk_score, axis=1)
    result["计算风险得分"] = [x[0] for x in calculated]
    result["触发依据"] = [x[1] for x in calculated]
    for score_column in ["经营风险分", "财务风险分", "运营风险分", "合同风险分"]:
        result[score_column] = [x[2][score_column] for x in calculated]
    result["计算风险等级"] = result["计算风险得分"].map(score_to_level)
    result["坪效(元/sqm)"] = (
        result["本月销售额"] / result.get("租赁面积(sqm)", pd.Series(1, index=result.index)).replace(0, pd.NA)
    ).fillna(0)
    result["欠费标记"] = result["欠费总额(元)"] > 0
    return result.sort_values("计算风险得分", ascending=False).reset_index(drop=True)


def diagnose(row: pd.Series) -> tuple[str, int, list[Action]]:
    arrears = float(row["欠费总额(元)"])
    rent_ratio = float(row["租售比(%)"])
    mom = float(row["销售环比(%)"])
    entry = float(row["进店率(%)"])
    conversion = float(row["成交转化率(%)"])
    utility = float(row["水电费波动(%)"])

    if arrears >= 50_000:
        issue = "欠费金额较高，已形成明显现金流与履约风险"
    elif rent_ratio > 35:
        issue = "租售比严重偏高，当前销售能力难以覆盖租赁成本"
    elif utility <= -40:
        issue = "水电使用异常下降，需要排查缩短营业、停业或撤店迹象"
    elif mom <= -20 and entry < 5:
        issue = "销售与进店率同步下降，品牌吸引力或店面展示可能弱化"
    elif mom <= -20 and conversion < 8:
        issue = "销售明显下降且转化率偏低，问题更可能位于货品、价格或导购环节"
    elif mom <= -10:
        issue = "销售连续承压，需要进入重点观察与经营约谈"
    else:
        issue = "当前经营指标整体稳定，暂未发现需要立即干预的重大风险"

    reason_count = len(row["触发依据"])
    confidence = min(95, 45 + reason_count * 9)

    actions: list[Action] = []
    if arrears > 0:
        actions.append(Action("财务/法务", "核查欠费账龄并生成催缴函草稿", True, "1 个工作日"))
    if mom <= -10:
        actions.append(Action("楼层经理", "完成商户经营约谈并提交原因记录", False, "3 个工作日"))
    if entry < 5:
        actions.append(Action("运营团队", "核查橱窗陈列、店面形象和所在区域动线", False, "3 个工作日"))
    if conversion < 8:
        actions.append(Action("品牌店长", "提交人员排班、货品结构和转化整改方案", False, "5 个工作日"))
    if row["计算风险等级"] == "极高":
        actions.append(Action("招商团队", "建立同品类、同定位的备选品牌清单", True, "5 个工作日"))
    if not actions:
        actions.append(Action("楼层经理", "保持月度观察，无需立即干预", False, "下月复核"))

    return issue, confidence, actions


def format_store_line(row: pd.Series) -> str:
    issue, _, _ = diagnose(row)
    return (
        f"{LEVEL_ICON[row['计算风险等级']]} **{row['门店名称']}**｜"
        f"{row['计算风险等级']}风险 {int(row['计算风险得分'])} 分｜{issue}"
    )


def hash_dataframe_for_cache(data: pd.DataFrame) -> str:
    cache_safe = data.copy()
    for column in cache_safe.columns:
        if cache_safe[column].map(lambda value: isinstance(value, (list, dict, set))).any():
            cache_safe[column] = cache_safe[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            )
    hash_value = int(pd.util.hash_pandas_object(cache_safe, index=True).sum())
    return hash_value.to_bytes(8, "little", signed=False).hex()


def row_to_rag_doc(row: pd.Series) -> RagDoc:
    issue, confidence, actions = diagnose(row)
    action_text = "；".join(
        f"{action.owner}：{action.task}（{action.deadline}，{'需审批' if action.approval_required else '可直接建任务'}）"
        for action in actions
    )
    content = (
        f"门店 {row['门店名称']}，ID {row['门店ID']}，业态 {row['业态分类']}，楼层 {row['楼层']}，"
        f"经营场景 {row.get('经营场景', '未标注')}。风险等级 {row['计算风险等级']}，风险得分 {int(row['计算风险得分'])}/100。"
        f"销售环比 {float(row['销售环比(%)']):.1f}%，进店率 {float(row['进店率(%)']):.1f}%，"
        f"成交转化率 {float(row['成交转化率(%)']):.1f}%，租售比 {float(row['租售比(%)']):.1f}%，"
        f"欠费 {money(float(row['欠费总额(元)']))}，欠费天数 {optional_float(row, '欠费天数'):.0f} 天，"
        f"水电费波动 {float(row['水电费波动(%)']):.1f}%，近90天投诉 {optional_float(row, '近90天投诉数'):.0f} 次。"
        f"触发依据：{'；'.join(row['触发依据']) if row['触发依据'] else '未触发明显风险规则'}。"
        f"诊断：{issue}，置信度 {confidence}%。建议动作：{action_text}。"
    )
    return RagDoc("门店经营数据", str(row["门店名称"]), content, str(row["门店ID"]))


_LAST_RETRIEVAL_DEBUG: dict[str, object] | None = None


def _set_last_retrieval_debug(payload: dict[str, object]) -> None:
    global _LAST_RETRIEVAL_DEBUG
    _LAST_RETRIEVAL_DEBUG = payload


def get_last_retrieval_debug() -> dict[str, object] | None:
    return dict(_LAST_RETRIEVAL_DEBUG) if _LAST_RETRIEVAL_DEBUG is not None else None


def retrieve_context(question: str, data: pd.DataFrame, top_k: int = 8) -> list[RagDoc]:
    search_results, debug = search_rag_database_with_debug(question, top_k=top_k)
    if debug is not None:
        _set_last_retrieval_debug({
            "retrieval_latency_ms": debug.retrieval_latency_ms,
            "dense_count": debug.dense_count,
            "lexical_count": debug.lexical_count,
            "merged_count": debug.merged_count,
            "reranked_count": debug.reranked_count,
            "embedding_provider": debug.embedding_provider,
            "embedding_model": debug.embedding_model,
            "embedding_dimension": debug.embedding_dimension,
            "vector_db": debug.vector_db,
            "dense_retrieval_status": debug.dense_retrieval_status,
            "config": debug.config,
        })
    db_docs = [
        RagDoc(
            source=result.source,
            title=result.title,
            content=result.content,
            row_id=str(result.metadata.get("store_id", "")) or None,
            dense_score=result.metadata.get("dense_score"),
            lexical_score=result.metadata.get("lexical_score"),
            fusion_score=result.metadata.get("fusion_score"),
            rerank_score=result.metadata.get("rerank_score"),
        )
        for result in search_results
    ]
    if db_docs:
        return db_docs

    if data is None or data.empty:
        return []
    return [row_to_rag_doc(row) for _, row in data.head(top_k).iterrows()]


def build_rag_prompt(question: str, evidence: list[RagDoc], structured_snapshot: str) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{index}] 来源：{doc.source}｜标题：{doc.title}\n{doc.content}"
        for index, doc in enumerate(evidence, start=1)
    )
    system = (
        "# Role\n"
        "你是购物中心经营风险预警 Agent，服务对象是购物中心运营、招商、财务和法务团队。\n\n"
        "# Task\n"
        "根据用户问题和 RAG 检索证据，输出可执行但受审批约束的风险诊断。\n\n"
        "# Context\n"
        "你会收到结构化查询结果和 RAG 检索证据。结构化查询结果来自当前门店数据计算，RAG 证据可能来自门店经营数据、合同条款样例、运营 SOP、招商策略、图片 OCR 结果和 Agent 设计方案草稿。\n\n"
        "# Rules\n"
        "1. 只能依据给定证据回答，不得编造合同条款、外部新闻、概率或数据库中不存在的事实。\n"
        "2. 涉及催缴、法务函、合同解除、保证金扣划、锁铺、清场、品牌替换的事项，必须标注“需人工审批”。\n"
        "3. 如果证据不足，明确列出还需要补充的数据。\n"
        "4. 先判断风险，再解释证据，最后给行动建议；不要只给泛泛建议。\n"
        "5. 结论段必须用自然中文表达，不要出现 [1]、[2] 这类引用编号串。\n"
        "6. 引用编号只用于“关键证据”和“引用证据”两个段落；不要在结论、风险判断、建议动作中堆叠编号。\n"
        "7. 保持中文、专业、适合管理层阅读。\n\n"
        "# Output Format\n"
        "请严格使用以下结构：\n"
        "### 结论\n"
        "用 2-4 句话回答用户问题，不写引用编号。\n\n"
        "### 关键证据\n"
        "- 证据点，引用 [编号]\n\n"
        "### 风险判断\n"
        "- 风险等级/风险类型/主要原因\n\n"
        "### 建议动作\n"
        "- 责任部门｜动作｜时限｜是否需人工审批\n\n"
        "### 需要补充的数据\n"
        "- 如果没有缺口，写“暂无”。\n\n"
        "### 引用证据\n"
        "- [1] 来源｜标题\n"
        "- [2] 来源｜标题"
    )
    user = (
        "# User Question\n"
        f"{question}\n\n"
        "# Retrieved Context\n"
        f"{context}\n\n"
        "# Structured Data Snapshot\n"
        f"{structured_snapshot}\n\n"
        "# Instruction\n"
        "请只基于 Structured Data Snapshot、Retrieved Context 和 User Question 生成最终答复，并按 system message 中的 Output Format 输出。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_tool_calling_prompt(question: str) -> list[dict[str, str]]:
    system = (
        "你是购物中心经营风险预警 Agent。你可以调用工具查询结构化门店风险数据、检索 RAG 证据、生成待审批行动草稿。\n"
        "必须先基于工具结果回答，不得编造工具没有返回的数据。涉及催缴、法务函、合同解除、保证金扣划、锁铺、清场或品牌替换时，必须标注需人工审批。\n"
        "最终回答使用中文，并包含：### 结论、### 关键证据、### 风险判断、### 建议动作、### 需要补充的数据、### 引用证据。"
    )
    user = (
        "# User Question\n"
        f"{question}\n\n"
        "# Instruction\n"
        "请根据问题选择必要工具。通常先调用 query_structured_risk_data，再按需要调用 search_rag_evidence 和 create_action_draft。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


AGENT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_structured_risk_data",
            "description": "查询当前门店结构化经营风险数据，支持按用户问题自动识别意图、业态、楼层或门店范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户原始问题或用于查询的业务问题。",
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_rag_evidence",
            "description": "检索本地 RAG 知识库，返回合同条款、SOP、招商策略、图片 OCR 或门店证据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或完整问题。"},
                    "top_k": {"type": "integer", "description": "返回证据条数，默认 6，最大 10。"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_action_draft",
            "description": "根据风险意图和审批边界生成行动草稿，敏感动作只生成待审批建议。",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "description": "风险意图，例如 arrears、top_risk、sales_decline。"},
                    "needs_approval_guard": {"type": "boolean", "description": "是否需要人工审批护栏。"},
                    "target_summary": {"type": "string", "description": "重点门店或风险对象摘要。"},
                },
                "required": ["intent", "needs_approval_guard", "target_summary"],
                "additionalProperties": False,
            },
        },
    },
]


def dataframe_records_for_tool(data: pd.DataFrame, limit: int = 8) -> list[dict[str, str | int | float | None]]:
    display_columns = [
        "门店ID",
        "门店名称",
        "业态分类",
        "楼层",
        "经营场景",
        "计算风险得分",
        "计算风险等级",
        "本月销售额",
        "销售环比(%)",
        "进店率(%)",
        "成交转化率(%)",
        "租售比(%)",
        "欠费总额(元)",
        "欠费天数",
        "保证金覆盖率(%)",
        "近90天投诉数",
        "水电费波动(%)",
    ]
    display_columns = [column for column in display_columns if column in data.columns]
    records: list[dict[str, str | int | float | None]] = []
    for _, row in data.head(limit).iterrows():
        item: dict[str, str | int | float | None] = {}
        for column in display_columns:
            value = row[column]
            if pd.isna(value):
                item[column] = None
            elif isinstance(value, (int, float, str)):
                item[column] = value
            else:
                item[column] = str(value)
        records.append(item)
    return records


def create_action_draft(intent: str, needs_approval_guard: bool, target_summary: str) -> dict:
    actions = [
        {
            "owner": "运营/楼层经理",
            "task": f"复核重点对象现场经营状态与沟通记录：{target_summary}",
            "deadline": "1-3 个工作日",
            "approval_required": False,
        },
        {
            "owner": "财务",
            "task": "复核欠费明细、账龄、保证金覆盖率和历史回款记录",
            "deadline": "1 个工作日",
            "approval_required": False,
        },
    ]
    if needs_approval_guard or intent in {"arrears", "top_risk", "executive_summary"}:
        actions.append(
            {
                "owner": "财务/法务",
                "task": "生成催缴函、合同风险或品牌替换预案草稿，并提交人工审批",
                "deadline": "1-2 个工作日",
                "approval_required": True,
            }
        )
    return {
        "intent": intent,
        "approval_boundary": "敏感动作只生成待审批草稿，不自动发送或执行。",
        "actions": actions,
    }


def execute_agent_tool(tool_name: str, arguments: dict, data: pd.DataFrame) -> tuple[dict, pd.DataFrame | None, list[RagDoc]]:
    if tool_name == "query_structured_risk_data":
        question = str(arguments.get("question") or "")
        plan = build_agent_plan(question, data)
        result, query_summary = query_structured_risk_data(plan, data)
        structured_snapshot = build_structured_snapshot(plan, query_summary, result)
        return (
            {
                "intent": plan.intent,
                "intent_label": plan.intent_label,
                "filters": plan.filters,
                "sort_column": plan.sort_column,
                "query_summary": query_summary,
                "structured_snapshot": structured_snapshot,
                "records": dataframe_records_for_tool(result),
                "needs_approval_guard": plan.needs_approval_guard,
            },
            result,
            [],
        )
    if tool_name == "search_rag_evidence":
        query = str(arguments.get("query") or "")
        top_k = min(10, max(1, int(arguments.get("top_k") or 6)))
        evidence = retrieve_context(query, data, top_k=top_k)
        return (
            {
                "evidence": [
                    {
                        "index": index,
                        "source": doc.source,
                        "title": doc.title,
                        "content": doc.content[:900],
                        "row_id": doc.row_id,
                    }
                    for index, doc in enumerate(evidence, start=1)
                ]
            },
            None,
            evidence,
        )
    if tool_name == "create_action_draft":
        return (
            create_action_draft(
                str(arguments.get("intent") or "top_risk"),
                bool(arguments.get("needs_approval_guard")),
                str(arguments.get("target_summary") or "重点风险门店"),
            ),
            None,
            [],
        )
    return ({"error": f"unknown tool: {tool_name}"}, None, [])


def get_secret_value(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value not in ("", None):
        return str(value)
    if SECRETS_PATH.exists():
        try:
            data = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
            value = data.get(key, default)
        except Exception:
            value = default
    else:
        value = default
    if value in ("", None):
        return default
    return str(value)


def get_openai_api_key() -> str | None:
    return get_secret_value("ARK_API_KEY") or get_secret_value("OPENAI_API_KEY")


def get_openai_base_url() -> str | None:
    return get_secret_value("ARK_BASE_URL") or get_secret_value("OPENAI_BASE_URL")


def get_openai_models() -> list[str]:
    configured = (
        get_secret_value("ARK_MODELS")
        or get_secret_value("ARK_MODEL")
        or get_secret_value("OPENAI_MODELS")
        or get_secret_value("OPENAI_MODEL", "gpt-4o-mini")
    )
    models = [model.strip() for model in configured.split(",") if model.strip()]
    return models or ["gpt-4o-mini"]


def get_ai_provider_name() -> str:
    if get_secret_value("ARK_API_KEY"):
        return "Ark/豆包"
    return "OpenAI 兼容"


def sanitize_ai_error(error: Exception) -> str:
    message = str(error)
    message = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", message)
    return message


def is_non_retryable_ai_error(error: Exception) -> bool:
    message = str(error).lower()
    non_retryable_markers = [
        "does not support the requested model",
        "permission_error",
        "model_not_found",
        "invalid model",
        "group_deleted",
    ]
    return any(marker in message for marker in non_retryable_markers)


def clamp_int(value: int, low: int, high: int) -> int:
    return min(high, max(low, value))


def build_scoring_review_prompt(plan: AgentPlan, result: pd.DataFrame, evidence: list[RagDoc]) -> list[dict[str, str]]:
    rule_config = load_rule_config()
    max_delta = int(rule_config.get("score_adjustment_policy", {}).get("max_delta_per_run", 10))
    scoring_rules = json.dumps(rule_config, ensure_ascii=False, indent=2)
    store_payload = []
    columns = [
        "门店名称",
        "业态分类",
        "经营场景",
        "计算风险得分",
        "计算风险等级",
        "经营风险分",
        "财务风险分",
        "运营风险分",
        "合同风险分",
        "本月销售额",
        "销售环比(%)",
        "近3月销售环比均值",
        "连续下滑月数",
        "租售比(%)",
        "近6月平均租售比",
        "欠费总额(元)",
        "欠费天数",
        "保证金覆盖率(%)",
        "水电费波动(%)",
        "近90天投诉数",
        "触发依据",
    ]
    columns = [column for column in columns if column in result.columns]
    for _, row in result.head(5).iterrows():
        item = {}
        for column in columns:
            value = row[column]
            item[column] = serializable(value)
        store_payload.append(item)

    evidence_payload = [
        {
            "id": index,
            "source": doc.source,
            "title": doc.title,
            "content": doc.content[:700],
        }
        for index, doc in enumerate(evidence[:8], start=1)
    ]
    system = (
        "你是购物中心经营风险 Agent 的 AI Scoring Reviewer。\n"
        "你的任务不是替代规则引擎，而是在规则基础分上做受控复核。\n"
        "你必须基于输入的评分标准、门店指标和证据，输出可审计推理摘要。\n"
        "不要输出隐藏推理链或长篇思维过程，只输出面向审计的 reasoning_summary。\n"
        f"单次 score_adjustment 必须在 {-max_delta} 到 {max_delta} 之间。\n"
        "如果证据不足，调整分必须为 0，并说明需要补充什么数据。\n"
        "只返回 JSON，不要包 Markdown。"
    )
    user = {
        "task": plan.intent_label,
        "score_rules": rule_config,
        "stores": store_payload,
        "evidence": evidence_payload,
        "output_schema": {
            "items": [
                {
                    "store_name": "门店名称",
                    "base_score": "规则基础分",
                    "score_adjustment": f"整数，范围 {-max_delta} 到 {max_delta}",
                    "adjustment_reason": "为什么建议调整，必须引用具体指标或证据编号",
                    "reasoning_summary": "3-5条可审计推理摘要，不展示隐藏思维链",
                    "evidence_ids": "使用到的证据编号数组",
                    "confidence": "0-100",
                    "requires_human_review": "是否需要人工复核，尤其是等级变化或敏感动作",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_scoring_review_json(raw_text: str, result: pd.DataFrame) -> list[AIScoreAssessment]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S)
    payload = json.loads(cleaned)
    items = payload.get("items", payload if isinstance(payload, list) else [])
    score_map = {
        str(row["门店名称"]): (int(row["计算风险得分"]), str(row["计算风险等级"]))
        for _, row in result.iterrows()
    }
    max_delta = int(load_rule_config().get("score_adjustment_policy", {}).get("max_delta_per_run", 10))
    assessments: list[AIScoreAssessment] = []
    for item in items:
        store_name = str(item.get("store_name") or item.get("门店名称") or "")
        if not store_name or store_name not in score_map:
            continue
        base_score, base_level = score_map[store_name]
        raw_adjustment = int(float(item.get("score_adjustment", item.get("adjustment", 0)) or 0))
        adjustment = clamp_int(raw_adjustment, -max_delta, max_delta)
        adjusted_score = clamp_int(base_score + adjustment, 0, 100)
        adjusted_level = score_to_level(adjusted_score)
        reasoning = str(item.get("reasoning_summary") or item.get("adjustment_reason") or "AI 未返回有效推理摘要。")
        evidence_ids = item.get("evidence_ids", [])
        evidence_basis = "、".join(f"[{eid}]" for eid in evidence_ids) if isinstance(evidence_ids, list) else str(evidence_ids)
        requires_review = bool(item.get("requires_human_review")) or adjusted_level != base_level
        assessments.append(
            AIScoreAssessment(
                store_name=store_name,
                base_score=base_score,
                adjustment=adjustment,
                adjusted_score=adjusted_score,
                adjusted_level=adjusted_level,
                reasoning_summary=reasoning,
                evidence_basis=evidence_basis or "结构化指标",
                confidence=clamp_int(int(float(item.get("confidence", 60) or 60)), 0, 100),
                requires_human_review=requires_review,
            )
        )
    return assessments


def local_scoring_review(result: pd.DataFrame) -> list[AIScoreAssessment]:
    assessments: list[AIScoreAssessment] = []
    for _, row in result.head(5).iterrows():
        adjustment = 0
        reasons = []
        if optional_float(row, "连续下滑月数") >= 3:
            adjustment += 4
            reasons.append("连续下滑月数达到 3，说明风险不是单月波动。")
        if optional_float(row, "近6月平均租售比") >= 35 and optional_float(row, "租售比(%)") >= 30:
            adjustment += 3
            reasons.append("近6月平均租售比和当月租售比均偏高，财务压力具有持续性。")
        if optional_float(row, "欠费天数") >= 60 and optional_float(row, "保证金覆盖率(%)", 100) < 60:
            adjustment += 3
            reasons.append("欠费账龄较长且保证金覆盖不足，履约缓冲偏弱。")
        if not reasons:
            reasons.append("未发现足以突破规则基础分的额外趋势证据，保持基础分。")
        adjustment = clamp_int(adjustment, -10, 10)
        base_score = int(row["计算风险得分"])
        adjusted_score = clamp_int(base_score + adjustment, 0, 100)
        assessments.append(
            AIScoreAssessment(
                store_name=str(row["门店名称"]),
                base_score=base_score,
                adjustment=adjustment,
                adjusted_score=adjusted_score,
                adjusted_level=score_to_level(adjusted_score),
                reasoning_summary="；".join(reasons),
                evidence_basis="结构化指标",
                confidence=65,
                requires_human_review=score_to_level(adjusted_score) != str(row["计算风险等级"]),
            )
        )
    return assessments


def call_ai_scoring_review(
    plan: AgentPlan,
    result: pd.DataFrame,
    evidence: list[RagDoc],
    status_callback: Callable[[str], None] | None = None,
) -> tuple[list[AIScoreAssessment], str]:
    api_key = get_openai_api_key()
    if result.empty:
        return [], "无结构化结果，跳过 AI 评分复核"
    if not api_key:
        return local_scoring_review(result), "AI 不可用，使用本地趋势复核"
    try:
        from openai import OpenAI

        base_url = get_openai_base_url()
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=35) if base_url else OpenAI(api_key=api_key, timeout=35)
        for model in get_openai_models():
            try:
                if status_callback:
                    status_callback(f"Scoring Reviewer：正在让 AI 复核基础分：{model}")
                response = client.chat.completions.create(
                    model=model,
                    messages=build_scoring_review_prompt(plan, result, evidence),
                    temperature=0.1,
                )
                assessments = parse_scoring_review_json(response.choices[0].message.content or "", result)
                if assessments:
                    return assessments, f"AI 已基于评分标准复核基础分：{model}"
            except Exception:
                continue
    except Exception:
        pass
    return local_scoring_review(result), "AI 评分复核失败，使用本地趋势复核"


def format_scoring_assessments(assessments: list[AIScoreAssessment]) -> str:
    if not assessments:
        return "暂无 AI 评分复核结果。"
    lines = ["### AI 评分复核过程"]
    for item in assessments:
        review_note = "，等级变化需人工复核" if item.requires_human_review else ""
        lines.append(
            f"- {item.store_name}：规则基础分 {item.base_score}，AI 建议调整 {item.adjustment:+d}，"
            f"建议分 {item.adjusted_score}（{item.adjusted_level}），置信度 {item.confidence}%{review_note}。"
            f"依据：{item.evidence_basis}。推理摘要：{item.reasoning_summary}"
        )
    return "\n".join(lines)


def call_openai_rag(
    question: str,
    evidence: list[RagDoc],
    structured_snapshot: str,
    status_callback: Callable[[str], None] | None = None,
) -> str | None:
    api_key = get_openai_api_key()
    if not api_key:
        return None
    errors: list[str] = []
    try:
        from openai import OpenAI

        base_url = get_openai_base_url()
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30) if base_url else OpenAI(api_key=api_key, timeout=30)
        for model in get_openai_models():
            for attempt in range(2):
                try:
                    if status_callback:
                        status_callback(f"正在调用 AI 模型：{model}（第 {attempt + 1} 次）")
                    response = client.chat.completions.create(
                        model=model,
                        messages=build_rag_prompt(question, evidence, structured_snapshot),
                        temperature=0.2,
                    )
                    if status_callback:
                        status_callback(f"AI 模型返回成功：{model}")
                    return response.choices[0].message.content
                except Exception as exc:
                    errors.append(f"{model} 第 {attempt + 1} 次：{sanitize_ai_error(exc)}")
                    if status_callback:
                        status_callback(f"{model} 暂不可用，准备尝试备用路径")
                    if is_non_retryable_ai_error(exc):
                        break
                    if attempt == 0:
                        time.sleep(1.2)
    except Exception as exc:
        errors.append(sanitize_ai_error(exc))
    tried = "；".join(errors[-3:]) if errors else "未知错误"
    warnings.warn(f"AI 生成暂不可用，已切换为本地 RAG 兜底。最近错误：{tried}", RuntimeWarning, stacklevel=2)
    return None


def call_openai_with_tools(
    question: str,
    data: pd.DataFrame,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[str | None, pd.DataFrame | None, list[RagDoc], list[AgentStep]]:
    api_key = get_openai_api_key()
    if not api_key:
        return None, None, [], []
    errors: list[str] = []
    try:
        from openai import OpenAI

        base_url = get_openai_base_url()
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=45) if base_url else OpenAI(api_key=api_key, timeout=45)
        for model in get_openai_models():
            for attempt in range(2):
                tool_trace: list[AgentStep] = []
                tool_result_df: pd.DataFrame | None = None
                tool_evidence: list[RagDoc] = []
                try:
                    if status_callback:
                        status_callback(f"Tool Agent：正在调用模型选择工具：{model}（第 {attempt + 1} 次）")
                    messages = build_tool_calling_prompt(question)
                    first_response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=AGENT_TOOL_SCHEMAS,
                        tool_choice="auto",
                        temperature=0.1,
                    )
                    assistant_message = first_response.choices[0].message
                    messages.append(assistant_message)
                    tool_calls = assistant_message.tool_calls or []
                    if not tool_calls:
                        if status_callback:
                            status_callback("Tool Agent：模型未请求工具，直接返回生成内容")
                        return assistant_message.content, None, [], [AgentStep("Tool Agent", "模型未调用工具，直接生成回答")]

                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            arguments = json.loads(tool_call.function.arguments or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                        if status_callback:
                            status_callback(f"Tool Agent：调用工具 {tool_name}")
                        tool_output, maybe_df, maybe_evidence = execute_agent_tool(tool_name, arguments, data)
                        if maybe_df is not None:
                            tool_result_df = maybe_df
                        if maybe_evidence:
                            tool_evidence.extend(maybe_evidence)
                        tool_trace.append(
                            AgentStep(
                                f"Tool: {tool_name}",
                                f"参数={json.dumps(arguments, ensure_ascii=False)}；返回字段={', '.join(tool_output.keys())}",
                            )
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": json.dumps(tool_output, ensure_ascii=False),
                            }
                        )

                    if status_callback:
                        status_callback(f"Tool Agent：工具完成，正在生成最终回答：{model}")
                    final_response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.2,
                    )
                    if status_callback:
                        status_callback(f"Tool Agent：AI 模型返回成功：{model}")
                    tool_trace.append(AgentStep("Generator", f"已使用 tool calling 生成：{model}"))
                    return final_response.choices[0].message.content, tool_result_df, tool_evidence, tool_trace
                except Exception as exc:
                    errors.append(f"{model} 第 {attempt + 1} 次：{sanitize_ai_error(exc)}")
                    if status_callback:
                        status_callback(f"Tool Agent：{model} 暂不可用，准备尝试备用路径")
                    if is_non_retryable_ai_error(exc):
                        break
                    if attempt == 0:
                        time.sleep(1.2)
    except Exception as exc:
        errors.append(sanitize_ai_error(exc))
    tried = "；".join(errors[-3:]) if errors else "未知错误"
    warnings.warn(f"Tool calling 暂不可用，已切换为原 Agent 工作流。最近错误：{tried}", RuntimeWarning, stacklevel=2)
    return None, None, [], []


def prepare_rag_database(data: pd.DataFrame) -> dict:
    return build_rag_database(data)


def build_agent_plan(question: str, data: pd.DataFrame) -> AgentPlan:
    q = question.strip()
    low_terms = ["最低", "最少", "最差", "最弱", "较低", "偏低", "低"]
    high_terms = ["最高", "最多", "最好", "最佳", "最强", "较高", "偏高", "高"]
    filters: dict[str, str] = {}
    matched_category = next((c for c in data["业态分类"].dropna().unique() if str(c) in q), None)
    matched_floor = next((f for f in data["楼层"].dropna().unique() if str(f).upper() in q.upper()), None)
    matched_store = next((s for s in data["门店名称"].dropna().unique() if str(s) in q), None)
    if matched_category:
        filters["业态分类"] = str(matched_category)
    if matched_floor:
        filters["楼层"] = str(matched_floor)
    if matched_store:
        filters["门店名称"] = str(matched_store)

    if any(k in q for k in ["最低风险", "风险最低", "最安全", "最稳健", "低风险门店"]):
        return AgentPlan("lowest_risk", "全场最低风险", filters, "计算风险得分", True, 8, False, False)
    if any(k in q for k in ["欠费", "催缴", "催款", "逾期", "法务函", "律师函"]):
        return AgentPlan("arrears", "欠费与履约风险", filters, "欠费总额(元)", False, 8, True, True)
    if any(k in q for k in ["上涨", "增长", "提升", "上升", "涨幅", "增幅", "销售最好", "销售最佳"]):
        return AgentPlan("sales_growth", "销售上涨表现", filters, "销售环比(%)", False, 8, False, False)
    if any(k in q for k in ["销售下滑", "销售下降", "下滑", "下降", "跌幅", "负增长"]):
        return AgentPlan("sales_decline", "销售下滑风险", filters, "销售环比(%)", True, 8, False, False)
    if any(k in q for k in ["销售额", "营收", "业绩"]) and any(k in q for k in high_terms):
        return AgentPlan("sales_amount_high", "销售额最高门店", filters, "本月销售额", False, 8, False, False)
    if any(k in q for k in ["销售额", "营收", "业绩"]) and any(k in q for k in low_terms):
        return AgentPlan("sales_amount_low", "销售额最低门店", filters, "本月销售额", True, 8, False, False)
    if "租售比" in q:
        ascending = any(k in q for k in low_terms)
        return AgentPlan("rent_ratio", "租售比压力" if not ascending else "租售比较低门店", filters, "租售比(%)", ascending, 8, True, False)
    if any(k in q for k in ["投诉", "客诉"]):
        ascending = any(k in q for k in low_terms)
        return AgentPlan("complaints", "投诉最多门店" if not ascending else "投诉最少门店", filters, "近90天投诉数", ascending, 8, False, False)
    if "进店率" in q:
        ascending = any(k in q for k in low_terms)
        return AgentPlan("entry_rate", "进店率最低门店" if ascending else "进店率最高门店", filters, "进店率(%)", ascending, 8, False, False)
    if any(k in q for k in ["成交转化率", "转化率"]):
        ascending = any(k in q for k in low_terms)
        return AgentPlan("conversion_rate", "成交转化率最低门店" if ascending else "成交转化率最高门店", filters, "成交转化率(%)", ascending, 8, False, False)
    if "保证金覆盖率" in q or "保证金" in q:
        ascending = any(k in q for k in low_terms)
        return AgentPlan("deposit_cover", "保证金覆盖率最低门店" if ascending else "保证金覆盖率最高门店", filters, "保证金覆盖率(%)", ascending, 8, True, False)
    if "欠费天数" in q or "账龄" in q:
        ascending = any(k in q for k in low_terms)
        return AgentPlan("arrears_days", "欠费账龄最长门店" if not ascending else "欠费账龄最短门店", filters, "欠费天数", ascending, 8, True, False)
    if any(k in q for k in ["水电", "撤店", "停业", "闭店", "异常营业"]):
        ascending = not any(k in q for k in ["上涨", "增长", "提升", "最高", "最多"])
        return AgentPlan("operation_anomaly", "经营异常核查", filters, "水电费波动(%)", ascending, 8, True, False)
    if any(k in q for k in ["管理层", "摘要", "报告", "汇报"]):
        return AgentPlan("executive_summary", "管理层摘要", filters, "计算风险得分", False, 8, True, True)
    if any(k in q for k in ["图片", "OCR", "ocr", "催款函", "催缴函"]):
        return AgentPlan("document_qa", "图片/OCR 与资料问答", filters, "计算风险得分", False, 5, True, True)
    return AgentPlan("top_risk", "综合最高风险", filters, "计算风险得分", False, 8, True, True)


def apply_agent_filters(data: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    pool = data.copy()
    for column, value in filters.items():
        if column in pool.columns:
            pool = pool[pool[column].astype(str) == str(value)]
    return pool


def query_structured_risk_data(plan: AgentPlan, data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    pool = apply_agent_filters(data, plan.filters)
    if pool.empty:
        return pool, "当前筛选条件下没有匹配门店。"
    if plan.intent == "arrears":
        result = pool[pool["欠费总额(元)"] > 0].sort_values(plan.sort_column, ascending=plan.ascending).head(plan.limit)
        total = int((pool["欠费总额(元)"] > 0).sum())
        amount = float(pool["欠费总额(元)"].sum())
        return result, f"共发现 {total} 家欠费门店，欠费总额 {money(amount)}，已按欠费金额降序选择重点对象。"
    if plan.intent == "executive_summary":
        high_count = int(pool["计算风险等级"].isin(HIGH_RISK_LEVELS).sum())
        amount = float(pool["欠费总额(元)"].sum())
        avg_mom = float(pool["销售环比(%)"].mean())
        result = pool.sort_values(plan.sort_column, ascending=plan.ascending).head(plan.limit)
        return result, f"高/极高风险 {high_count} 家，欠费总额 {money(amount)}，平均销售环比 {avg_mom:.1f}%。"
    if plan.intent == "lowest_risk":
        result = pool.sort_values(["计算风险得分", "欠费总额(元)", "销售环比(%)"], ascending=[True, True, False]).head(plan.limit)
        min_score = int(result["计算风险得分"].min()) if not result.empty else 0
        tied_count = int((pool["计算风险得分"] == min_score).sum())
        return result, f"全场最低风险得分为 {min_score}/100，共 {tied_count} 家门店并列该分值；已按风险得分升序选择最低风险梯队。"
    if plan.intent == "sales_growth":
        result = pool.sort_values(plan.sort_column, ascending=plan.ascending).head(plan.limit)
        positive_count = int((pool["销售环比(%)"] > 0).sum())
        best_growth = float(result["销售环比(%)"].max()) if not result.empty else 0.0
        return result, f"全场共有 {positive_count} 家门店销售环比为正，最高销售环比为 {best_growth:.1f}%；已按销售环比降序选择上涨表现最好的门店。"
    if plan.sort_column in pool.columns:
        result = pool.sort_values(plan.sort_column, ascending=plan.ascending).head(plan.limit)
        direction = "升序" if plan.ascending else "降序"
        if result.empty:
            return result, f"已按 {plan.sort_column} {direction}查询，但没有匹配对象。"
        top_value = result.iloc[0][plan.sort_column]
        if isinstance(top_value, (int, float)):
            value_text = f"{float(top_value):,.1f}"
        else:
            value_text = str(top_value)
        return result, f"已按 {plan.sort_column} {direction}选择 {len(result)} 个重点对象，首位门店该指标为 {value_text}。"
    result = pool.sort_values("计算风险得分", ascending=False).head(plan.limit)
    return result, f"未找到指定排序字段，已退回综合风险得分排序。"


def build_structured_snapshot(plan: AgentPlan, query_summary: str, result: pd.DataFrame) -> str:
    filters = "、".join(f"{key}={value}" for key, value in plan.filters.items()) or "全场"
    lines = [
        "### Agent 结构化查询结果",
        f"- 任务类型：{plan.intent_label}",
        f"- 查询范围：{filters}",
        f"- 查询摘要：{query_summary}",
        "- 重点对象：",
    ]
    if result.empty:
        lines.append("  - 暂无匹配门店")
    else:
        for index, (_, row) in enumerate(result.iterrows(), start=1):
            lines.append(f"  {index}. {format_store_line(row)}")
    return "\n".join(lines)


def clean_ai_answer(answer: str, evidence: list[RagDoc]) -> str:
    sections = re.split(r"(###\s+)", answer)
    if len(sections) >= 3:
        rebuilt: list[str] = [sections[0]]
        for i in range(1, len(sections), 2):
            prefix = sections[i]
            body = sections[i + 1] if i + 1 < len(sections) else ""
            if body.startswith("结论"):
                body = re.sub(r"\s*\[\d+\](?:\[\d+\]|\s*)*", "", body)
            rebuilt.append(prefix + body)
        answer = "".join(rebuilt)

    if "### 引用证据" not in answer and evidence:
        refs = "\n".join(
            f"- [{index}] {doc.source}｜{doc.title}" for index, doc in enumerate(evidence[:5], start=1)
        )
        answer = f"{answer.rstrip()}\n\n### 引用证据\n{refs}"
    return answer


CONTROLLED_ACTION_TERMS = ["催缴", "催缴函", "催款函", "法务函", "律师函", "解除合同", "保证金扣划", "锁铺", "清场", "品牌替换"]
REQUIRED_ANSWER_SECTIONS = ["### 结论", "### 关键证据", "### 风险判断", "### 建议动作", "### 需要补充的数据", "### 引用证据"]
ACTION_TYPE_MAP = {
    "催缴": "催缴",
    "催缴函": "催缴",
    "催款函": "催缴",
    "法务函": "法务函",
    "律师函": "法务函",
    "解除合同": "合同解除",
    "保证金扣划": "保证金扣划",
    "锁铺": "锁铺",
    "清场": "清场",
    "品牌替换": "品牌替换",
}


def extract_answer_section(answer: str, section_title: str) -> str:
    pattern = rf"###\s*{re.escape(section_title)}\s*(.*?)(?=\n###\s|\Z)"
    match = re.search(pattern, answer, flags=re.S)
    return match.group(1).strip() if match else ""


def normalize_text_value(value: object) -> str:
    return str(value).strip() if value not in (None, "") else ""


def verify_agent_answer(
    answer: str,
    plan: AgentPlan,
    evidence: list[RagDoc],
    result: pd.DataFrame | None = None,
    user_role: str = "管理层",
) -> list[str]:
    issues: list[str] = []

    missing_sections = [section for section in REQUIRED_ANSWER_SECTIONS if section not in answer]
    if missing_sections:
        issues.append(f"格式校验：缺少固定标题 {', '.join(missing_sections)}。")

    conclusion = extract_answer_section(answer, "结论") or answer.split("### 关键证据", 1)[0]
    if re.search(r"\[\d+\]", conclusion):
        issues.append("格式校验：结论段包含证据编号，已自动清理。")

    if result is not None and not result.empty:
        store_names = {normalize_text_value(name) for name in result["门店名称"].dropna()} if "门店名称" in result.columns else set()
        mentioned_known_store = any(name and name in answer for name in store_names)
        if store_names and not mentioned_known_store:
            issues.append("数值校验：回答未提及结构化查询返回的重点门店。")

        for _, row in result.head(8).iterrows():
            store_name = normalize_text_value(row.get("门店名称", ""))
            if not store_name or store_name not in answer:
                continue
            score = row.get("计算风险得分")
            if pd.notna(score) and f"{int(float(score))}" not in answer:
                issues.append(f"数值校验：{store_name} 的风险得分未在回答中体现。")
            arrears = float(row.get("欠费总额(元)", 0) or 0)
            if arrears > 0:
                arrears_texts = {f"{arrears:,.0f}", f"{arrears / 10000:.1f}万", f"{arrears / 10000:.0f}万"}
                if not any(text in answer for text in arrears_texts):
                    issues.append(f"数值校验：{store_name} 存在欠费，但回答未体现欠费金额。")

    has_controlled_action = any(term in answer for term in CONTROLLED_ACTION_TERMS)
    if plan.needs_approval_guard and has_controlled_action:
        if "需人工审批" not in answer:
            issues.append("审批校验：涉及受控动作但未标注需人工审批。")

    if evidence and "### 引用证据" not in answer:
        issues.append("引用校验：缺少引用证据段，已自动补充。")

    key_evidence = extract_answer_section(answer, "关键证据")
    referenced_numbers = {int(value) for value in re.findall(r"\[(\d+)\]", key_evidence)}
    if key_evidence and evidence and not referenced_numbers:
        issues.append("引用校验：关键证据段未绑定引用编号。")
    invalid_refs = [number for number in sorted(referenced_numbers) if number < 1 or number > len(evidence)]
    if invalid_refs:
        issues.append(f"引用校验：存在超出证据范围的引用编号 {invalid_refs}。")

    role_action_limits = {
        "运营": ["法务函", "律师函", "解除合同", "保证金扣划", "锁铺", "清场", "品牌替换"],
        "招商": ["法务函", "律师函", "解除合同", "保证金扣划", "锁铺", "清场"],
        "财务": ["解除合同", "锁铺", "清场", "品牌替换"],
        "法务": ["品牌替换"],
    }
    blocked_terms = role_action_limits.get(user_role, [])
    blocked_hits = [term for term in blocked_terms if term in answer]
    if blocked_hits and "需人工审批" not in answer:
        issues.append(f"权限校验：{user_role} 角色涉及 {', '.join(blocked_hits)}，必须转为待审批草稿。")

    return issues


def create_pending_approvals_for_run(run_id: str, answer: str, result: pd.DataFrame, plan: AgentPlan) -> list[str]:
    if not plan.needs_approval_guard:
        return []
    action_types = sorted({ACTION_TYPE_MAP[term] for term in ACTION_TYPE_MAP if term in answer})
    if not action_types:
        action_types = ["受控动作草稿"]
    if result is not None and not result.empty and "门店名称" in result.columns:
        targets = "、".join(str(name) for name in result["门店名称"].head(3).tolist())
    else:
        targets = plan.intent_label
    approval_ids = []
    for action_type in action_types:
        approval_ids.append(create_approval(run_id, action_type=action_type, target=targets, status="pending"))
    return approval_ids




def agent_answer(
    question: str,
    data: pd.DataFrame,
    use_llm: bool = True,
    user_role: str = "管理层",
    status_callback: Callable[[str], None] | None = None,
) -> tuple[str, pd.DataFrame, list[RagDoc], list[AgentStep]]:
    trace: list[AgentStep] = []
    if use_llm:
        tool_answer, tool_result, tool_evidence, tool_trace = call_openai_with_tools(
            question,
            data,
            status_callback=status_callback,
        )
        if tool_answer:
            result_df = tool_result if isinstance(tool_result, pd.DataFrame) else pd.DataFrame()
            tool_plan = build_agent_plan(question, data)
            if not result_df.empty:
                scoring_assessments, scoring_status = call_ai_scoring_review(
                    tool_plan,
                    result_df,
                    tool_evidence,
                    status_callback=status_callback,
                )
                scoring_review_text = format_scoring_assessments(scoring_assessments)
                if "### AI 评分复核过程" not in tool_answer:
                    tool_answer = f"{tool_answer.rstrip()}\n\n{scoring_review_text}"
                tool_trace.append(AgentStep("Scoring Reviewer", scoring_status))
            answer = clean_ai_answer(tool_answer, tool_evidence)
            verification_issues = verify_agent_answer(
                answer,
                tool_plan,
                tool_evidence,
                result=result_df,
                user_role=user_role,
            )
            tool_trace.append(
                AgentStep(
                    "Verifier",
                    "；".join(verification_issues) if verification_issues else "未发现明显格式或审批边界问题",
                    "已修正" if verification_issues else "完成",
                )
            )
            return answer, result_df, tool_evidence, [AgentStep("Planner", "已启用 OpenAI tool calling")] + tool_trace

    if status_callback:
        status_callback("Planner：正在识别任务类型、筛选条件和审批边界")
    plan = build_agent_plan(question, data)
    filter_text = "、".join(f"{key}={value}" for key, value in plan.filters.items()) or "全场"
    trace.append(AgentStep("Planner", f"任务={plan.intent_label}；范围={filter_text}；排序={plan.sort_column}"))

    if status_callback:
        status_callback("Structured Query：正在查询门店经营数据和风险指标")
    result, query_summary = query_structured_risk_data(plan, data)
    trace.append(AgentStep("Structured Query", query_summary))

    if result.empty:
        if status_callback:
            status_callback("Retriever：结构化数据为空，改用通用 RAG 检索")
        evidence = retrieve_context(question, data)
        trace.append(AgentStep("Retriever", f"命中 {len(evidence)} 条通用知识证据"))
        answer = "没有找到符合条件的门店，请调整问题或筛选范围。"
        return answer, result, evidence, trace

    if status_callback:
        status_callback("Retriever：正在混合检索结构化门店证据、制度/SOP 和外部资料")
    evidence_query = f"{question}\n{query_summary}\n{build_structured_snapshot(plan, query_summary, result)}"
    evidence = retrieve_context(evidence_query, result, top_k=10)
    source_counts: dict[str, int] = {}
    for doc in evidence:
        source_counts[doc.source] = source_counts.get(doc.source, 0) + 1
    source_text = "，".join(f"{source} {count} 条" for source, count in source_counts.items()) or "无"
    trace.append(AgentStep("Retriever", f"命中 {len(evidence)} 条证据：{source_text}"))
    if status_callback:
        status_callback(f"Retriever：命中 {len(evidence)} 条证据（{source_text}）")

    if status_callback:
        status_callback("Scoring Reviewer：正在基于评分标准、历史趋势和证据复核基础分")
    scoring_assessments, scoring_status = call_ai_scoring_review(plan, result, evidence, status_callback=status_callback)
    scoring_review_text = format_scoring_assessments(scoring_assessments)
    trace.append(AgentStep("Scoring Reviewer", scoring_status))

    structured_snapshot = build_structured_snapshot(plan, query_summary, result)
    top_rows = list(result.iterrows())[:3]
    key_evidence = "\n".join(
        f"- {row['门店名称']}：风险得分 {int(row['计算风险得分'])}/100，欠费 {money(float(row['欠费总额(元)']))}，"
        f"销售环比 {float(row['销售环比(%)']):.1f}%，租售比 {float(row['租售比(%)']):.1f}% [{index}]"
        for index, (_, row) in enumerate(top_rows, start=1)
    )
    references = "\n".join(
        f"- [{index}] {doc.source}｜{doc.title}" for index, doc in enumerate(evidence[:5], start=1)
    )
    approval_note = (
        "；涉及催缴、法务函、合同解除、保证金扣划、锁铺、清场或品牌替换的动作必须保留人工审批"
        if plan.needs_approval_guard
        else ""
    )
    controlled_action_line = (
        "- 法务/商管负责人｜审核受控动作草稿，确认是否进入法务或合同风险流程｜2 个工作日｜需人工审批\n"
        if plan.needs_approval_guard
        else ""
    )
    fallback_answer = (
        f"### 结论\n"
        f"{query_summary}建议优先复核排序靠前的门店{approval_note}。\n\n"
        f"### 关键证据\n"
        f"{key_evidence or '- 暂无可展示证据'}\n\n"
        f"### 风险判断\n"
        f"- 风险类型：{plan.intent_label}\n"
        f"- 主要原因：当前结构化数据已按 {plan.sort_column} 排序，重点对象同时结合风险得分、欠费、销售环比、租售比和投诉等指标筛选。\n\n"
        f"{scoring_review_text}\n\n"
        f"### 建议动作\n"
        f"- 运营/财务｜核验重点门店数据、欠费账龄、沟通记录和现场经营状态｜1-3 个工作日｜否\n"
        f"{controlled_action_line}\n"
        f"### 需要补充的数据\n"
        f"- 近 3-6 个月销售、回款、保证金余额、巡检记录和商户沟通记录。\n\n"
        f"### 引用证据\n"
        f"{references}"
    )

    if status_callback:
        status_callback("Generator：正在调用 AI 生成结构化诊断")
    llm_snapshot = f"{structured_snapshot}\n\n{scoring_review_text}"
    llm_answer = call_openai_rag(question, evidence, llm_snapshot, status_callback=status_callback) if use_llm else None
    answer = llm_answer or fallback_answer
    if llm_answer and "### AI 评分复核过程" not in answer:
        answer = f"{answer.rstrip()}\n\n{scoring_review_text}"
    trace.append(AgentStep("Generator", "已使用真实 AI 生成" if llm_answer else "AI 不可用，已使用本地结构化兜底"))

    if status_callback:
        status_callback("Verifier：正在检查引用、审批边界和输出格式")
    verification_issues = verify_agent_answer(answer, plan, evidence, result=result, user_role=user_role)
    answer = clean_ai_answer(answer, evidence)
    if verification_issues:
        trace.append(AgentStep("Verifier", "；".join(verification_issues), "已修正"))
    else:
        trace.append(AgentStep("Verifier", "未发现明显格式或审批边界问题"))
    if status_callback:
        status_callback("Verifier：检查完成")
    return answer, result, evidence, trace


def money(value: float) -> str:
    if abs(value) >= 10_000:
        return f"¥{value / 10_000:,.1f}万"
    return f"¥{value:,.0f}"


def serializable(value) -> str | int | float | bool | list | None:
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def format_table_cell(row: pd.Series, col: str) -> str:
    value = row[col]
    if col in {"本月销售额", "欠费总额(元)"}:
        return f"¥{float(value):,.0f}"
    if isinstance(value, float):
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return str(value)


