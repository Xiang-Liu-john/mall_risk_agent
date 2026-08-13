from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from rag_store import build_rag_database, rag_database_stats, search_rag_database


st.set_page_config(
    page_title="智慧购物中心运营预警看板 AI Engine v4.0",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @media (min-width: 900px) {
        section[data-testid="stSidebar"] {
            width: 3.4rem !important;
            min-width: 3.4rem !important;
            max-width: 3.4rem !important;
            transition: width 0.22s ease, min-width 0.22s ease, max-width 0.22s ease;
            overflow: visible;
            border-right: 1px solid #E5E7EB;
        }

        section[data-testid="stSidebar"]:hover,
        section[data-testid="stSidebar"]:focus-within {
            width: 15rem !important;
            min-width: 15rem !important;
            max-width: 15rem !important;
            box-shadow: 12px 0 28px rgba(15, 23, 42, 0.10);
            z-index: 999;
        }

        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            width: 15rem !important;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            opacity: 0;
            pointer-events: none;
            transform: translateX(-0.7rem);
            transition: opacity 0.16s ease, transform 0.2s ease;
            overflow-x: hidden;
        }

        section[data-testid="stSidebar"]:hover [data-testid="stSidebarContent"],
        section[data-testid="stSidebar"]:focus-within [data-testid="stSidebarContent"] {
            opacity: 1;
            pointer-events: auto;
            transform: translateX(0);
        }

        section[data-testid="stSidebar"]::after {
            content: "筛选";
            position: absolute;
            top: 5rem;
            left: 0.85rem;
            writing-mode: vertical-rl;
            letter-spacing: 0;
            color: #334155;
            font-size: 0.8rem;
            font-weight: 700;
            pointer-events: none;
            transition: opacity 0.14s ease;
        }

        section[data-testid="stSidebar"]:hover::after,
        section[data-testid="stSidebar"]:focus-within::after {
            opacity: 0;
        }

        section[data-testid="stSidebar"] button[kind="header"] {
            display: none;
        }
    }

    .risk-table-wrap {
        min-height: 560px;
        max-height: 68vh;
        overflow: auto;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        background: #FFFFFF;
    }

    .risk-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
        color: #1F2937;
        white-space: nowrap;
    }

    .risk-table th,
    .risk-table td {
        padding: 0.66rem 0.78rem;
        border-bottom: 1px solid #EEF2F7;
        border-right: 1px solid #EEF2F7;
        text-align: left;
    }

    .risk-table th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #F8FAFC;
        color: #64748B;
        font-weight: 700;
    }

    .risk-table tr.is-selected {
        background: #FEF2F2;
    }

    .risk-link {
        color: #2563EB;
        font-weight: 700;
        text-decoration: none;
    }

    .risk-link:hover {
        color: #1D4ED8;
        text-decoration: underline;
    }

    .risk-score {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        min-width: 130px;
    }

    .risk-score-track {
        width: 86px;
        height: 7px;
        border-radius: 999px;
        background: #FEE2E2;
        overflow: hidden;
    }

    .risk-score-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #FDA4AF, #EF4444);
    }

    .diagnosis-drawer {
        position: fixed;
        top: 2.7rem;
        right: 0;
        z-index: 1000;
        width: min(50vw, 920px);
        height: calc(100vh - 2.7rem);
        padding: 1rem 1.55rem 2rem;
        overflow-y: auto;
        background: #FFFFFF;
        border-left: 1px solid #E5E7EB;
        box-shadow: -18px 0 36px rgba(15, 23, 42, 0.16);
    }

    .diagnosis-close {
        position: sticky;
        top: 0;
        float: right;
        color: #64748B;
        text-decoration: none;
        font-size: 1.25rem;
        line-height: 1;
        background: #FFFFFF;
        padding: 0.2rem 0.35rem;
    }

    .diagnosis-title {
        margin: 0.25rem 2rem 1rem 0;
        font-size: 1.25rem;
        font-weight: 800;
        color: #111827;
    }

    .diagnosis-metrics,
    .risk-breakdown {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.8rem 0 1rem;
    }

    .diagnosis-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem;
        background: #F8FAFC;
    }

    .diagnosis-card span {
        display: block;
        color: #64748B;
        font-size: 0.78rem;
        margin-bottom: 0.3rem;
    }

    .diagnosis-card strong {
        color: #111827;
        font-size: 1.05rem;
    }

    .diagnosis-note {
        border-radius: 8px;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E3A8A;
        padding: 0.8rem 0.9rem;
        margin: 0.8rem 0 1.1rem;
    }

    .drawer-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }

    .drawer-section h4 {
        margin: 0.4rem 0 0.7rem;
        color: #111827;
    }

    .drawer-section ul {
        margin: 0;
        padding-left: 1.15rem;
    }

    .action-card {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.65rem;
        background: #FFFFFF;
    }

    .action-card p {
        margin: 0.35rem 0;
    }

    .action-card small {
        color: #64748B;
    }

    @media (max-width: 1100px) {
        .diagnosis-drawer {
            width: 100vw;
        }
        .drawer-grid,
        .diagnosis-metrics,
        .risk-breakdown {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_CSV = Path(__file__).parent / "data" / "购物中心100家门店经营数据_优化命名.csv"

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
        st.warning(f"已忽略 {int(bad_rows.sum())} 行关键数值缺失或格式错误的数据。")

    result["门店ID"] = result["门店ID"].astype(str)
    result["门店名称"] = result["门店名称"].astype(str)
    result["业态分类"] = result["业态分类"].astype(str)
    result["楼层"] = result["楼层"].astype(str)
    optional_defaults = {
        "经营场景": "未标注",
        "欠费天数": 0,
        "保证金覆盖率(%)": 100,
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
    if arrears >= 50_000:
        financial_score += 22
        reasons.append(f"欠费金额达到 ¥{arrears:,.0f}，超过 5 万元")
    elif arrears >= 10_000:
        financial_score += 14
        reasons.append(f"欠费金额达到 ¥{arrears:,.0f}，超过 1 万元")
    elif arrears > 0:
        financial_score += 6
        reasons.append(f"存在欠费 ¥{arrears:,.0f}")

    rent_ratio = float(row["租售比(%)"])
    if rent_ratio > 35:
        financial_score += 10
        reasons.append(f"租售比 {rent_ratio:.1f}%，严重高于警戒区间")
    elif rent_ratio > 25:
        financial_score += 7
        reasons.append(f"租售比 {rent_ratio:.1f}%，超过 25% 警戒线")
    elif rent_ratio >= 18:
        financial_score += 4
        reasons.append(f"租售比 {rent_ratio:.1f}%，进入观察区间")

    mom = float(row["销售环比(%)"])
    if mom <= -30:
        business_score += 20
        reasons.append(f"销售环比下降 {abs(mom):.1f}%")
    elif mom <= -20:
        business_score += 16
        reasons.append(f"销售环比下降 {abs(mom):.1f}%")
    elif mom <= -10:
        business_score += 8
        reasons.append(f"销售环比下降 {abs(mom):.1f}%")

    utility = float(row["水电费波动(%)"])
    if utility <= -40:
        operation_score += 12
        reasons.append(f"水电费下降 {abs(utility):.1f}%，需核查异常营业或撤店迹象")
    elif utility <= -20:
        operation_score += 7
        reasons.append(f"水电费下降 {abs(utility):.1f}%")

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


def tokenize(text: str) -> set[str]:
    normalized = str(text).lower()
    words = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", normalized))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    words.update(chinese[i : i + 2] for i in range(max(0, len(chinese) - 1)))
    return {word for word in words if word.strip()}


def split_text(text: str, chunk_size: int = 420, overlap: int = 60) -> list[str]:
    clean = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


@st.cache_data(show_spinner=False)
def load_knowledge_chunks() -> list[RagDoc]:
    docs = [
        RagDoc(
            "内置规则",
            "风险评分边界",
            "风险等级：0-29 低风险，30-54 中风险，55-74 高风险，75-100 极高风险。涉及催缴、法务、终止合同或品牌替换的动作只生成待审批草稿，不由 Agent 直接执行。",
        ),
        RagDoc(
            "内置SOP",
            "经营风险处置SOP",
            "先核验数据异常，再对比同楼层和同业态表现；对欠费门店核查账龄、保证金覆盖率和沟通记录；对销售下滑门店复核客流、转化、陈列、货品和导购排班。",
        ),
        RagDoc(
            "内置SOP",
            "儿童配套专项指标",
            "儿童配套门店需要额外关注退款申请数、续费率、安全巡检分和家长投诉率；安全或投诉问题优先进入运营复核，不直接以销售指标替代服务质量判断。",
        ),
    ]
    source_path = Path(__file__).parent / "interview_source_extract.txt"
    if source_path.exists():
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        for index, chunk in enumerate(split_text(source_text), start=1):
            docs.append(RagDoc("Agent 设计方案草稿", f"Agent 设计草稿 #{index}", chunk))
    return docs


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


def retrieve_context(question: str, data: pd.DataFrame, top_k: int = 8) -> list[RagDoc]:
    forced_docs = [row_to_rag_doc(row) for _, row in data.iterrows()] if data is not None and not data.empty else []
    db_docs = [
        RagDoc(
            source=result.source,
            title=result.title,
            content=result.content,
            row_id=str(result.metadata.get("store_id", "")) or None,
        )
        for result in search_rag_database(question, top_k=top_k)
    ]
    if db_docs:
        merged_docs: list[RagDoc] = []
        seen: set[tuple[str, str]] = set()
        for doc in [*forced_docs, *db_docs]:
            key = (doc.source, doc.title)
            if key in seen:
                continue
            seen.add(key)
            merged_docs.append(doc)
        return merged_docs[:top_k]

    query_tokens = tokenize(question)
    docs = forced_docs + load_knowledge_chunks()
    scored: list[tuple[float, RagDoc]] = []
    for doc in docs:
        doc_tokens = tokenize(f"{doc.title} {doc.content}")
        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            score = 0.0
        else:
            score = overlap / max(6, len(query_tokens)) + min(0.25, len(doc_tokens) / 1200)
        if doc.source == "门店数据" and any(k in question for k in ["风险", "门店", "欠费", "销售", "租售比"]):
            score += 0.15
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored[:top_k] if score > 0] or [doc for _, doc in scored[:top_k]]


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
        "你会收到结构化查询结果和 RAG 检索证据。结构化查询结果来自当前门店数据计算，RAG 证据可能来自门店经营数据、模拟合同条款、运营 SOP、招商策略、图片 OCR 结果和 Agent 设计方案草稿。\n\n"
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


def get_secret_value(key: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(key, default)
    except Exception:
        return default
    if value in ("", None):
        return default
    return str(value)


def get_openai_api_key() -> str | None:
    return get_secret_value("OPENAI_API_KEY")


def get_openai_base_url() -> str | None:
    return get_secret_value("OPENAI_BASE_URL")


def get_openai_models() -> list[str]:
    configured = get_secret_value("OPENAI_MODELS") or get_secret_value("OPENAI_MODEL", "gpt-4o-mini")
    models = [model.strip() for model in configured.split(",") if model.strip()]
    return models or ["gpt-4o-mini"]


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
    st.warning(f"AI 生成暂不可用，已切换为本地 RAG 兜底。最近错误：{tried}")
    return None


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: hash_dataframe_for_cache})
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


def render_streaming_markdown(text: str, chunk_size: int = 12, delay: float = 0.01) -> None:
    placeholder = st.empty()
    rendered = ""
    for index in range(0, len(text), chunk_size):
        rendered += text[index : index + chunk_size]
        placeholder.markdown(rendered + "▌")
        time.sleep(delay)
    placeholder.markdown(text)


def verify_agent_answer(answer: str, plan: AgentPlan, evidence: list[RagDoc]) -> list[str]:
    issues: list[str] = []
    conclusion = answer.split("### 关键证据", 1)[0]
    if re.search(r"\[\d+\]", conclusion):
        issues.append("结论段包含证据编号，已自动清理。")
    if plan.needs_approval_guard and any(k in answer for k in ["催缴", "法务函", "律师函", "解除合同", "保证金扣划", "锁铺", "清场", "品牌替换"]):
        if "需人工审批" not in answer:
            issues.append("涉及受控动作但未标注需人工审批。")
    if evidence and "### 引用证据" not in answer:
        issues.append("缺少引用证据段，已自动补充。")
    return issues


def agent_answer(
    question: str,
    data: pd.DataFrame,
    use_llm: bool = True,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[str, pd.DataFrame, list[RagDoc], list[AgentStep]]:
    trace: list[AgentStep] = []
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
    llm_answer = call_openai_rag(question, evidence, structured_snapshot, status_callback=status_callback) if use_llm else None
    answer = llm_answer or fallback_answer
    trace.append(AgentStep("Generator", "已使用真实 AI 生成" if llm_answer else "AI 不可用，已使用本地结构化兜底"))

    if status_callback:
        status_callback("Verifier：正在检查引用、审批边界和输出格式")
    verification_issues = verify_agent_answer(answer, plan, evidence)
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


def render_store_diagnosis_component(data: pd.DataFrame, columns: list[str]) -> str:
    records = []
    for _, row in data.iterrows():
        issue, confidence, actions = diagnose(row)
        records.append(
            {
                "id": str(row["门店ID"]),
                "cells": {col: format_table_cell(row, col) for col in columns},
                "score": int(row["计算风险得分"]),
                "storeName": str(row["门店名称"]),
                "level": str(row["计算风险等级"]),
                "icon": LEVEL_ICON[row["计算风险等级"]],
                "metrics": {
                    "风险得分": f"{int(row['计算风险得分'])}/100",
                    "销售环比": f"{float(row['销售环比(%)']):.1f}%",
                    "租售比": f"{float(row['租售比(%)']):.1f}%",
                    "欠费": money(float(row["欠费总额(元)"])),
                },
                "issue": issue,
                "confidence": confidence,
                "scores": [
                    {"label": "经营风险", "value": f"{int(row['经营风险分'])} / 40"},
                    {"label": "财务风险", "value": f"{int(row['财务风险分'])} / 30"},
                    {"label": "运营风险", "value": f"{int(row['运营风险分'])} / 20"},
                    {"label": "合同风险", "value": f"{int(row['合同风险分'])} / 10"},
                ],
                "reasons": serializable(row["触发依据"]) or [],
                "actions": [
                    {
                        "owner": action.owner,
                        "task": action.task,
                        "approval": "需人工审批" if action.approval_required else "可直接创建内部任务",
                        "deadline": action.deadline,
                        "button": "生成待审批草稿" if action.approval_required else "加入任务清单",
                        "status": "待审批" if action.approval_required else "待执行",
                    }
                    for action in actions
                ],
            }
        )

    payload = json.dumps({"columns": columns, "records": records}, ensure_ascii=False)
    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #111827;
  background: #fff;
}}
.risk-shell {{
  position: relative;
  min-height: 860px;
  overflow: hidden;
}}
.risk-table-wrap {{
  height: 640px;
  min-height: 640px;
  max-height: 640px;
  overflow-y: auto;
  overflow-x: hidden;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  background: #FFFFFF;
}}
.risk-table {{
  width: 100%;
  table-layout: fixed;
  border-collapse: collapse;
  font-size: 12px;
  line-height: 1.25;
}}
.risk-table th,
.risk-table td {{
  padding: 8px 8px;
  border-bottom: 1px solid #EEF2F7;
  border-right: 1px solid #EEF2F7;
  text-align: left;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.risk-table th {{
  position: sticky;
  top: 0;
  z-index: 2;
  background: #F8FAFC;
  color: #64748B;
  font-weight: 700;
  cursor: pointer;
  user-select: none;
}}
.risk-table th:nth-child(1),
.risk-table td:nth-child(1) {{ width: 7.2%; }}
.risk-table th:nth-child(2),
.risk-table td:nth-child(2) {{ width: 5.4%; }}
.risk-table th:nth-child(3),
.risk-table td:nth-child(3) {{ width: 9.4%; }}
.risk-table th:nth-child(4),
.risk-table td:nth-child(4) {{ width: 5.2%; }}
.risk-table th:nth-child(5),
.risk-table td:nth-child(5) {{ width: 3.6%; }}
.risk-table th:nth-child(6),
.risk-table td:nth-child(6) {{ width: 5.8%; }}
.risk-table th:nth-child(7),
.risk-table td:nth-child(7) {{ width: 6.1%; }}
.risk-table th:nth-child(8),
.risk-table td:nth-child(8) {{ width: 5.1%; }}
.risk-table th:nth-child(9),
.risk-table td:nth-child(9) {{ width: 5.1%; }}
.risk-table th:nth-child(10),
.risk-table td:nth-child(10) {{ width: 6.3%; }}
.risk-table th:nth-child(11),
.risk-table td:nth-child(11) {{ width: 4.8%; }}
.risk-table th:nth-child(12),
.risk-table td:nth-child(12) {{ width: 6.3%; }}
.risk-table th:nth-child(13),
.risk-table td:nth-child(13) {{ width: 5.7%; }}
.risk-table th:nth-child(14),
.risk-table td:nth-child(14) {{ width: 6.2%; }}
.risk-table th:nth-child(15),
.risk-table td:nth-child(15) {{ width: 11.0%; }}
.risk-table th:nth-child(15) {{
  text-align: center;
}}
.risk-table td:nth-child(15) {{
  text-align: center;
  padding-left: 10px;
  padding-right: 10px;
}}
.risk-table th.no-sort {{
  cursor: default;
}}
.sort-mark {{
  color: #EF4444;
  margin-left: 4px;
}}
.risk-table tr.selected {{
  background: #FEF2F2;
}}
.risk-link {{
  appearance: none;
  border: 0;
  background: transparent;
  padding: 0;
  color: #2563EB;
  font: inherit;
  font-weight: 700;
  text-decoration: none;
  cursor: pointer;
}}
.risk-link:hover {{
  color: #1D4ED8;
  text-decoration: underline;
}}
.risk-score {{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  min-width: 0;
  width: 100%;
}}
.risk-score-track {{
  flex: 1 1 auto;
  width: auto;
  min-width: 64px;
  max-width: 110px;
  height: 6px;
  border-radius: 999px;
  background: #FEE2E2;
  overflow: hidden;
}}
.risk-score span {{
  flex: 0 0 30px;
  text-align: left;
}}
.risk-score-fill {{
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #FDA4AF, #EF4444);
}}
.drawer {{
  position: fixed;
  top: 0;
  right: 0;
  z-index: 20;
  width: min(50vw, 920px);
  height: min(640px, 100vh);
  max-height: 640px;
  padding: 14px 18px 18px;
  box-sizing: border-box;
  overflow-y: auto;
  overscroll-behavior: contain;
  background: #FFFFFF;
  border-left: 1px solid #E5E7EB;
  box-shadow: -18px 0 36px rgba(15, 23, 42, 0.16);
  transform: translateX(105%);
  transition: transform 0.18s ease;
  font-size: 13px;
  line-height: 1.45;
}}
.drawer.open {{
  transform: translateX(0);
}}
.drawer-close {{
  float: right;
  border: 0;
  background: transparent;
  color: #64748B;
  font-size: 20px;
  cursor: pointer;
}}
.drawer h3 {{
  margin: 2px 32px 12px 0;
  font-size: 17px;
  line-height: 1.35;
}}
.metric-grid,
.score-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin: 8px 0 12px;
}}
.metric-card {{
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 8px;
  background: #F8FAFC;
}}
.metric-card span {{
  display: block;
  color: #64748B;
  font-size: 11px;
  margin-bottom: 4px;
}}
.metric-card strong {{
  font-size: 13px;
}}
.note {{
  border-radius: 8px;
  background: #EFF6FF;
  border: 1px solid #BFDBFE;
  color: #1E3A8A;
  padding: 10px 12px;
  margin-bottom: 12px;
}}
.drawer-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}}
.drawer h4 {{
  margin: 8px 0 8px;
  font-size: 14px;
}}
.drawer li {{
  margin-bottom: 6px;
}}
.action-card {{
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 9px;
  margin-bottom: 8px;
}}
.action-card p {{
  margin: 6px 0;
}}
.action-card small {{
  color: #64748B;
}}
.action-list {{
  margin-top: 14px;
  padding-bottom: 12px;
}}
.action-list h4 {{
  margin: 0 0 8px;
  font-size: 15px;
}}
.action-list-table-wrap {{
  max-height: 190px;
  overflow: auto;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
}}
.action-list table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}}
.action-list th,
.action-list td {{
  padding: 7px 8px;
  border-bottom: 1px solid #E5E7EB;
  border-right: 1px solid #E5E7EB;
  text-align: left;
}}
.action-empty {{
  color: #64748B;
  font-size: 12px;
  padding: 10px 0 0;
}}
.action-list th {{
  position: sticky;
  top: 0;
  z-index: 1;
  background: #F8FAFC;
  color: #64748B;
}}
@media (max-width: 1100px) {{
  .drawer {{
    width: 100vw;
  }}
  .metric-grid,
  .score-grid,
  .drawer-grid {{
    grid-template-columns: 1fr;
  }}
}}
</style>
</head>
<body>
<div class="risk-shell">
  <div class="risk-table-wrap">
    <table class="risk-table" id="riskTable"></table>
  </div>
  <div class="action-list" id="actionList"></div>
  <aside class="drawer" id="drawer"></aside>
</div>
<script>
const data = {payload};
const table = document.getElementById("riskTable");
const drawer = document.getElementById("drawer");
const actionList = document.getElementById("actionList");
const actions = [];
let selectedId = null;
let sortState = {{ column: "计算风险得分", direction: "desc" }};
const levelOrder = {{ "低": 1, "中": 2, "高": 3, "极高": 4 }};

function esc(value) {{
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}}

function renderTable() {{
  const headerItems = ["AI诊断详情", ...data.columns];
  const headers = headerItems.map(col => {{
    const sortable = col !== "AI诊断详情";
    const mark = sortState.column === col ? `<span class="sort-mark">${{sortState.direction === "asc" ? "↑" : "↓"}}</span>` : "";
    return `<th class="${{sortable ? "sortable" : "no-sort"}}" data-column="${{esc(col)}}">${{esc(col)}}${{mark}}</th>`;
  }}).join("");
  const sortedRecords = [...data.records].sort(compareRecords);
  const rows = sortedRecords.map(record => {{
    const diagnosisCell = `<td title="查看智能诊断"><button class="risk-link diagnosis-link" data-id="${{esc(record.id)}}">查看诊断 →</button></td>`;
    const cells = data.columns.map(col => {{
      if (col === "计算风险得分") {{
        return `<td title="${{record.score}}%"><div class="risk-score"><div class="risk-score-track"><div class="risk-score-fill" style="width:${{record.score}}%;"></div></div><span>${{record.score}}%</span></div></td>`;
      }}
      return `<td title="${{esc(record.cells[col])}}">${{esc(record.cells[col])}}</td>`;
    }}).join("");
    const selected = record.id === selectedId ? " selected" : "";
    return `<tr class="${{selected}}">${{diagnosisCell}}${{cells}}</tr>`;
  }}).join("");
  table.innerHTML = `<thead><tr>${{headers}}</tr></thead><tbody>${{rows}}</tbody>`;
  table.querySelectorAll("th.sortable").forEach(header => {{
    header.addEventListener("click", () => toggleSort(header.dataset.column));
  }});
  table.querySelectorAll(".diagnosis-link").forEach(button => {{
    button.addEventListener("click", () => openDrawer(button.dataset.id));
  }});
}}

function parseSortValue(value) {{
  if (value === null || value === undefined) return "";
  const raw = String(value).trim();
  const numeric = Number(raw.replace(/[¥,%]/g, "").replaceAll(",", ""));
  return Number.isNaN(numeric) ? raw : numeric;
}}

function compareRecords(a, b) {{
  const col = sortState.column;
  const aValue = col === "计算风险得分" ? a.score : parseSortValue(a.cells[col]);
  const bValue = col === "计算风险得分" ? b.score : parseSortValue(b.cells[col]);
  let result = 0;
  if (col === "计算风险等级") {{
    result = (levelOrder[a.cells[col]] ?? 0) - (levelOrder[b.cells[col]] ?? 0);
  }} else if (typeof aValue === "number" && typeof bValue === "number") {{
    result = aValue - bValue;
  }} else {{
    result = String(aValue).localeCompare(String(bValue), "zh-Hans-CN", {{ numeric: true }});
  }}
  return sortState.direction === "asc" ? result : -result;
}}

function toggleSort(column) {{
  if (sortState.column === column) {{
    sortState.direction = sortState.direction === "asc" ? "desc" : "asc";
  }} else {{
    sortState = {{ column, direction: column === "门店ID" ? "asc" : "desc" }};
  }}
  renderTable();
}}

function card(label, value) {{
  return `<div class="metric-card"><span>${{esc(label)}}</span><strong>${{esc(value)}}</strong></div>`;
}}

function openDrawer(id) {{
  selectedId = id;
  const record = data.records.find(item => item.id === id);
  if (!record) return;
  renderTable();
  const metrics = Object.entries(record.metrics).map(([label, value]) => card(label, value)).join("");
  const scores = record.scores.map(item => card(item.label, item.value)).join("");
  const reasons = record.reasons.length
    ? `<ul>${{record.reasons.map(reason => `<li>${{esc(reason)}}</li>`).join("")}}</ul>`
    : "<p>未触发明显风险规则。</p>";
  const actionCards = record.actions.map((action, idx) => `
    <div class="action-card">
      <strong>${{idx + 1}}. ${{esc(action.owner)}}</strong>
      <p>${{esc(action.task)}}</p>
      <small>截止：${{esc(action.deadline)}}｜${{esc(action.approval)}}</small><br>
      <button class="risk-link add-action" data-index="${{idx}}">${{esc(action.button)}}</button>
    </div>
  `).join("");
  drawer.innerHTML = `
    <button class="drawer-close" id="drawerClose">×</button>
    <h3>${{esc(record.icon)}} ${{esc(record.storeName)}}｜${{esc(record.level)}}风险</h3>
    <div class="metric-grid">${{metrics}}</div>
    <div class="note"><strong>核心诊断：</strong>${{esc(record.issue)}}<br><br><strong>诊断置信度：</strong>${{esc(record.confidence)}}%</div>
    <h4>风险分项</h4>
    <div class="score-grid">${{scores}}</div>
    <div class="drawer-grid">
      <section><h4>触发依据</h4>${{reasons}}</section>
      <section><h4>建议行动</h4>${{actionCards}}</section>
    </div>
  `;
  drawer.classList.add("open");
  document.getElementById("drawerClose").addEventListener("click", closeDrawer);
  drawer.querySelectorAll(".add-action").forEach(button => {{
    button.addEventListener("click", () => addAction(record, Number(button.dataset.index)));
  }});
}}

function closeDrawer() {{
  selectedId = null;
  drawer.classList.remove("open");
  renderTable();
}}

function addAction(record, index) {{
  const action = record.actions[index];
  actions.push({{
    "门店": record.storeName,
    "责任人": action.owner,
    "任务": action.task,
    "审批状态": action.status
  }});
  renderActionList();
}}

function renderActionList() {{
  if (!actions.length) {{
    actionList.innerHTML = `<h4>本次演示行动清单</h4><div class="action-empty">还没有加入任务。请点击“查看诊断”，再在建议行动里加入任务。</div>`;
    return;
  }}
  const headers = ["门店", "责任人", "任务", "审批状态"];
  const head = headers.map(item => `<th>${{item}}</th>`).join("");
  const rows = actions.map(action => `<tr>${{headers.map(item => `<td>${{esc(action[item])}}</td>`).join("")}}</tr>`).join("");
  actionList.innerHTML = `<h4>本次演示行动清单</h4><div class="action-list-table-wrap"><table><thead><tr>${{head}}</tr></thead><tbody>${{rows}}</tbody></table></div>`;
}}

renderTable();
renderActionList();
</script>
</body>
</html>
"""


# --------------------------- 数据加载 ---------------------------
st.sidebar.title("数据与筛选")
uploaded = st.sidebar.file_uploader("上传门店经营 CSV", type=["csv"])

try:
    source = uploaded if uploaded is not None else DEFAULT_CSV
    raw_df = read_csv_robust(source)
    df = enrich(validate_and_clean(raw_df))
    rag_stats = prepare_rag_database(df)
except Exception as exc:
    st.error(f"数据加载失败：{exc}")
    st.stop()

categories = sorted(df["业态分类"].unique().tolist())
floors = sorted(df["楼层"].unique().tolist())
selected_categories = st.sidebar.multiselect("业态", categories, default=[], placeholder="全部业态")
selected_floors = st.sidebar.multiselect("楼层", floors, default=[], placeholder="全部楼层")
selected_levels = st.sidebar.multiselect("计算风险等级", LEVEL_ORDER, default=[], placeholder="全部等级")
search_text = st.sidebar.text_input("搜索门店", placeholder="门店名称或 ID")

filtered = df.copy()
if selected_categories:
    filtered = filtered[filtered["业态分类"].isin(selected_categories)]
if selected_floors:
    filtered = filtered[filtered["楼层"].isin(selected_floors)]
if selected_levels:
    filtered = filtered[filtered["计算风险等级"].isin(selected_levels)]
if search_text.strip():
    key = search_text.strip().lower()
    filtered = filtered[
        filtered["门店名称"].str.lower().str.contains(key, na=False)
        | filtered["门店ID"].str.lower().str.contains(key, na=False)
    ].copy()

st.sidebar.caption("说明：筛选器为空表示全部；看板中的风险等级由本地规则重新计算，不直接采用 CSV 中预先填写的结果。")

# --------------------------- 页面 ---------------------------
st.title("🏬 智慧购物中心运营预警看板 AI Engine v4.0")
st.caption("MVP：确定性风险计算 → 可解释诊断 → 行动草稿 → 人工审批")

if filtered.empty:
    st.warning("当前筛选条件下没有门店数据。")
    st.stop()

high_risk_count = int(filtered["计算风险等级"].isin(HIGH_RISK_LEVELS).sum())
arrears_store_count = int(filtered["欠费标记"].sum())
arrears_total = float(filtered["欠费总额(元)"].sum())
avg_mom = float(filtered["销售环比(%)"].mean())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("当前门店", f"{len(filtered)} 家")
m2.metric("高/极高风险", f"{high_risk_count} 家")
m3.metric("欠费门店", f"{arrears_store_count} 家")
m4.metric("欠费总额", money(arrears_total))
m5.metric("平均销售环比", f"{avg_mom:.1f}%")

tab_dashboard, tab_stores, tab_agent, tab_rules = st.tabs(
    ["经营总览", "风险门店诊断", "AI 经营助理", "规则与边界"]
)

with tab_dashboard:
    c1, c_mid, c2 = st.columns([1, 0.9, 1])
    with c1:
        category_counts = (
            filtered["业态分类"].value_counts().rename_axis("业态分类").reset_index(name="门店数")
        )
        category_counts["门店数标签"] = category_counts["门店数"].map(lambda value: f"{value}家")
        fig = px.pie(
            category_counts,
            names="业态分类",
            values="门店数",
            hole=0.58,
            title="业态分布",
            color="业态分类",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig.update_traces(
            text=category_counts["门店数标签"],
            textinfo="text",
            textposition="inside",
            hovertemplate="%{label}<br>门店数：%{value}<br>占比：%{percent}<extra></extra>",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
        )
        fig.update_layout(
            width=550,
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.12,
                xanchor="center",
                x=0.5,
            ),
            showlegend=True,
            uniformtext_minsize=11,
            uniformtext_mode="hide",
        )
        st.plotly_chart(fig, width="content", config={"responsive": False})

    with c_mid:
        category_risk = filtered.copy()
        category_risk["是否高风险"] = category_risk["计算风险等级"].isin(HIGH_RISK_LEVELS).astype(int)
        cat_summary = (
            category_risk.groupby("业态分类", as_index=False)
            .agg(
                门店数=("门店ID", "count"),
                高风险门店数=("是否高风险", "sum"),
                风险中位数=("计算风险得分", "median"),
                平均风险得分=("计算风险得分", "mean"),
            )
            .sort_values("高风险门店数", ascending=False)
        )
        cat_summary["高风险门店占比(%)"] = (
            cat_summary["高风险门店数"] / cat_summary["门店数"] * 100
        ).round(1)
        fig = px.scatter(
            cat_summary,
            x="门店数",
            y="高风险门店占比(%)",
            size="门店数",
            color="风险中位数",
            text="业态分类",
            title="各业态规模与高风险门店占比（当前筛选范围）",
            hover_data={
                "门店数": True,
                "高风险门店数": True,
                "高风险门店占比(%)": ":.1f",
                "风险中位数": ":.1f",
                "平均风险得分": ":.1f",
            },
            labels={"风险中位数": "风险中位数"},
            color_continuous_scale=CATEGORY_RISK_COLORS,
        )
        max_rate = max(1, float(cat_summary["高风险门店占比(%)"].max()))
        fig.update_traces(
            textposition="top center",
            textfont=dict(size=12, color="#334155"),
            cliponaxis=False,
        )
        fig.update_layout(
            width=520,
            height=380,
            margin=dict(l=10, r=10, t=70, b=10),
            yaxis_ticksuffix="%",
            yaxis_range=[0, max_rate * 1.28],
        )
        st.plotly_chart(fig, width="content", config={"responsive": False})

    with c2:
        level_counts = (
            filtered["计算风险等级"].value_counts().reindex(LEVEL_ORDER, fill_value=0).reset_index()
        )
        level_counts.columns = ["风险等级", "门店数"]
        fig = px.bar(
            level_counts,
            x="风险等级",
            y="门店数",
            color="风险等级",
            color_discrete_map=LEVEL_COLORS,
            title="风险等级分布",
            text="门店数",
            category_orders={"风险等级": LEVEL_ORDER},
        )
        level_max = max(1, int(level_counts["门店数"].max()))
        fig.update_traces(
            texttemplate="%{text}",
            textposition="outside",
            textfont=dict(size=12, color="#1F2937"),
            cliponaxis=False,
        )
        fig.update_layout(
            width=550,
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
            yaxis_range=[0, level_max * 1.18],
            uniformtext_minsize=11,
            uniformtext_mode="show",
        )
        st.plotly_chart(fig, width="content", config={"responsive": False})

    top_risk = filtered.nlargest(10, "计算风险得分").copy()
    fig = px.bar(
        top_risk.sort_values("计算风险得分"),
        x="计算风险得分",
        y="门店名称",
        color="计算风险得分",
        color_continuous_scale=TOP_RISK_COLORS,
        range_color=[0, 100],
        orientation="h",
        title="综合风险最高的 10 家门店",
        hover_data=["业态分类", "经营场景", "销售环比(%)", "租售比(%)", "欠费总额(元)", "近90天投诉数"],
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("管理层快速摘要")
    top3 = filtered.nlargest(3, "计算风险得分")
    st.markdown(
        f"当前筛选范围内共有 **{high_risk_count} 家高/极高风险门店**，"
        f"欠费总额为 **{money(arrears_total)}**，平均销售环比为 **{avg_mom:.1f}%**。"
    )
    for _, row in top3.iterrows():
        st.markdown(format_store_line(row))

with tab_stores:
    display_columns = [
        "门店ID",
        "门店名称",
        "业态分类",
        "楼层",
        "经营场景",
        "本月销售额",
        "销售环比(%)",
        "进店率(%)",
        "成交转化率(%)",
        "租售比(%)",
        "欠费总额(元)",
        "近90天投诉数",
        "水电费波动(%)",
        "计算风险得分",
        "计算风险等级",
    ]
    display_columns = [col for col in display_columns if col in filtered.columns]
    components.html(
        render_store_diagnosis_component(filtered, display_columns),
        height=890,
        scrolling=False,
    )

with tab_agent:
    st.markdown(
        "这里使用 Agentic Loop：Planner 先识别任务类型和审批边界，Structured Query 查询门店指标，"
        "Retriever 混合检索制度/SOP/图片 OCR 知识，Generator 生成诊断，Verifier 校验引用和人工审批边界；"
        "未配置 OpenAI API key 时自动使用本地结构化兜底，不会虚构真实预测概率。"
    )
    use_llm = st.toggle(
        "启用真实 AI 生成",
        value=bool(get_openai_api_key()),
        help="需要在 .streamlit/secrets.toml 中配置 OPENAI_API_KEY；关闭后使用本地 RAG 兜底。",
    )
    model_status = "，".join(get_openai_models())
    st.caption(
        f"AI 状态：{'已开启' if use_llm else '已关闭'}；"
        f"API key：{'已读取' if get_openai_api_key() else '未读取'}；"
        f"模型：{model_status}"
    )
    with st.expander("RAG 数据库状态", expanded=False):
        current_stats = rag_database_stats()
        design_draft_count = (
            current_stats.get("Agent 设计方案草稿", 0)
            or current_stats.get("Agent Design Proposal Draft", 0)
            or current_stats.get("访谈资料", 0)
        )
        known_stat_keys = {
            "db_path",
            "total_chunks",
            "门店经营数据",
            "制度与模板",
            "Agent 设计方案草稿",
            "访谈资料",
        }
        imported_count = sum(
            int(value)
            for key, value in current_stats.items()
            if key not in known_stat_keys and isinstance(value, int)
        )
        st.caption("这里展示的是本地 SQLite RAG 知识库的构建状态，用来确认 Agent 可以检索哪些知识源。")
        status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
        status_col1.metric("总知识片段", int(current_stats.get("total_chunks", 0)))
        status_col2.metric("门店经营数据", int(current_stats.get("门店经营数据", 0)))
        status_col3.metric("制度与合同模板", int(current_stats.get("制度与模板", 0)))
        status_col4.metric("Agent 设计方案草稿", int(design_draft_count))
        status_col5.metric("外部导入资料", imported_count)
        st.caption(f"数据库文件：{current_stats.get('db_path')}")
        source_rows = [
            {"知识源": key, "片段数": value}
            for key, value in current_stats.items()
            if key not in {"db_path", "total_chunks"} and isinstance(value, int)
        ]
        if source_rows:
            st.dataframe(pd.DataFrame(source_rows), hide_index=True)
        if st.button("重建 RAG 数据库", icon=":material/refresh:"):
            st.cache_data.clear()
            rag_stats = build_rag_database(df)
            st.success(f"已重建 RAG 数据库，共 {rag_stats}")
        with st.form("rag_debug_search", border=False):
            debug_query = st.text_input(
                "测试 RAG 检索",
                placeholder="例如：图片 跃动工场 欠费；催款函；OCR",
            )
            debug_submitted = st.form_submit_button("搜索 RAG", icon=":material/search:")
        if debug_submitted and debug_query.strip():
            debug_results = search_rag_database(debug_query.strip(), top_k=8)
            if not debug_results:
                st.info("没有检索到相关知识片段。")
            for index, result in enumerate(debug_results, start=1):
                with st.container(border=True):
                    st.markdown(f"**[{index}] {result.source}｜{result.title}｜得分 {result.score:.3f}**")
                    st.caption(result.content[:700] + ("..." if len(result.content) > 700 else ""))

    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    quick_questions = {
        "最高风险": "分析全场最高风险门店",
        "欠费风险": "分析欠费风险",
        "销售下滑": "分析销售下滑最严重的门店",
        "管理层摘要": "生成管理层摘要",
    }
    for col, (label, question) in zip(
        [quick_col1, quick_col2, quick_col3, quick_col4], quick_questions.items()
    ):
        if col.button(label, width="stretch"):
            st.session_state["agent_question"] = question

    question = st.text_area(
        "向经营风险助理提问",
        value=st.session_state.get("agent_question", "帮我分析餐饮业态中最需要关注的门店"),
        placeholder="例如：分析 F3 楼层欠费风险；找出租售比最高的门店",
        height=100,
    )
    if st.button("运行 Agent", type="primary"):
        with st.status("Agent 工作流正在运行", expanded=True) as status:
            status.write("启动 Agent：准备解析问题")

            def update_agent_status(message: str) -> None:
                status.write(message)

            answer, result_df, evidence, trace = agent_answer(
                question,
                filtered,
                use_llm=use_llm,
                status_callback=update_agent_status,
            )
            status.update(label="Agent 工作流已完成", state="complete", expanded=False)
        st.session_state["agent_answer"] = answer
        st.session_state["agent_result"] = result_df
        st.session_state["agent_evidence"] = evidence
        st.session_state["agent_trace"] = trace
        st.session_state["agent_answer_should_stream"] = True

    if "agent_answer" in st.session_state:
        if st.session_state.pop("agent_answer_should_stream", False):
            render_streaming_markdown(st.session_state["agent_answer"])
        else:
            st.markdown(st.session_state["agent_answer"])
        trace = st.session_state.get("agent_trace", [])
        if trace:
            with st.expander("查看 Agent 工作流轨迹", expanded=False):
                trace_df = pd.DataFrame(
                    [{"节点": step.name, "状态": step.status, "说明": step.detail} for step in trace]
                )
                st.dataframe(trace_df, hide_index=True, width="stretch")
        evidence = st.session_state.get("agent_evidence", [])
        if evidence:
            with st.expander("查看 RAG 引用证据", expanded=False):
                for index, doc in enumerate(evidence, start=1):
                    st.markdown(f"**[{index}] {doc.source}｜{doc.title}**")
                    st.caption(doc.content[:700] + ("..." if len(doc.content) > 700 else ""))
        result_df = st.session_state.get("agent_result")
        if isinstance(result_df, pd.DataFrame) and not result_df.empty:
            agent_columns = [
                "门店名称",
                "业态分类",
                "楼层",
                "经营场景",
                "计算风险得分",
                "计算风险等级",
                "销售环比(%)",
                "租售比(%)",
                "欠费总额(元)",
                "近90天投诉数",
            ]
            agent_columns = [col for col in agent_columns if col in result_df.columns]
            st.dataframe(
                result_df[agent_columns],
                hide_index=True,
            )

with tab_rules:
    st.subheader("当前风险评分")
    st.markdown(
        """
| 指标 | 条件 | 分值 |
|---|---:|---:|
| 经营风险 | 销售下滑、进店率低、转化率低；儿童配套补充退款申请、续费率 | 最高 40 |
| 财务风险 | 欠费金额、租售比 | 最高 30 |
| 运营风险 | 水电费异常、近90天投诉；儿童配套补充安全巡检、家长投诉率 | 最高 20 |
| 合同风险 | 欠费账龄、保证金覆盖率 | 最高 10 |

风险等级：0–29 低，30–54 中，55–74 高，75–100 极高。总分和四个分项均做上限保护。
"""
    )

    st.subheader("MVP 边界")
    st.warning(
        "当前数据只有单月截面，不能据此声称某商户会在多少天后现金流断裂，也不能输出真实违约概率。"
        "生产版本应补充至少 6–12 个月历史数据、同楼层/同业态基准、合同到期日、保证金、欠费天数、"
        "巡店日志、投诉文本和人工反馈。当前 MVP 已加入儿童配套的退款、续费、安全巡检和家长投诉模拟字段，"
        "生产版本仍应按业态配置更细的专属指标和阈值。"
    )

    st.subheader("下载计算结果")
    output_csv = df.drop(columns=["触发依据"]).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载带风险计算结果的 CSV",
        data=output_csv,
        file_name="门店经营风险计算结果.csv",
        mime="text/csv",
    )
