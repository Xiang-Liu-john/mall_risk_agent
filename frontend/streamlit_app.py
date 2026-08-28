from __future__ import annotations

import json
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from backend.store import (
    backend_stats,
    create_agent_run,
    create_approval,
    decide_approval,
    ensure_session,
    finish_agent_run,
    list_approvals,
    save_agent_step,
    save_message,
    save_rag_evidence,
)
from rag_store import build_rag_database, rag_database_stats, search_rag_database, search_rag_database_with_debug
from risk_rules import load_rule_config, threshold_score


st.set_page_config(
    page_title="智慧购物中心运营预警看板 AI Engine v4.0",
    page_icon="🏬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("backend_user_id", f"user_{uuid.uuid4().hex[:8]}")
st.session_state.setdefault("backend_session_id", f"sess_{uuid.uuid4().hex[:16]}")
ensure_session(
    st.session_state["backend_session_id"],
    st.session_state["backend_user_id"],
    title="购物中心经营风险 Agent 会话",
    source_channel="streamlit",
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

from core.agent_logic import (
    CATEGORY_COLORS,
    CATEGORY_RISK_COLORS,
    DEFAULT_CSV,
    HIGH_RISK_LEVELS,
    LEVEL_COLORS,
    LEVEL_ICON,
    LEVEL_ORDER,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TOP_RISK_COLORS,
    agent_answer,
    build_agent_plan,
    create_pending_approvals_for_run,
    diagnose,
    enrich,
    format_store_line,
    format_table_cell,
    get_ai_provider_name,
    get_last_retrieval_debug,
    get_openai_models,
    hash_dataframe_for_cache,
    money,
    prepare_rag_database,
    read_csv_robust,
    retrieve_context,
    validate_and_clean,
)


def render_streaming_markdown(text: str, chunk_size: int = 12, delay: float = 0.01) -> None:
    placeholder = st.empty()
    rendered = ""
    for index in range(0, len(text), chunk_size):
        rendered += text[index : index + chunk_size]
        placeholder.markdown(rendered + "▌")
        time.sleep(delay)
    placeholder.markdown(text)


def render_approval_center(current_run_id: str | None = None) -> None:
    approvals = list_approvals(run_id=current_run_id, limit=20) if current_run_id else list_approvals(limit=20)
    pending_count = sum(1 for item in approvals if item["status"] == "pending")
    st.caption(f"Human-in-the-loop 审批记录：待审批 {pending_count} 条。这里只保存审批决定，不调用外部执行系统。")
    if not approvals:
        st.info("暂无审批草稿。")
        return
    for item in approvals:
        with st.container(border=True):
            st.markdown(f"**{item['action_type']}｜{item['target']}**")
            st.caption(
                f"approval_id={item['approval_id']}｜run_id={item['run_id']}｜"
                f"状态={item['status']}｜创建={item['created_at']}｜决定={item.get('decided_at') or '未决定'}"
            )
            if item["status"] == "pending":
                col_approve, col_reject = st.columns(2)
                if col_approve.button("Approve", key=f"approve_{item['approval_id']}", icon=":material/check:"):
                    decide_approval(item["approval_id"], "approved", decided_by=st.session_state["backend_user_id"])
                    st.rerun()
                if col_reject.button("Reject", key=f"reject_{item['approval_id']}", icon=":material/close:"):
                    decide_approval(item["approval_id"], "rejected", decided_by=st.session_state["backend_user_id"])
                    st.rerun()


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
    actionList.innerHTML = `<h4>本次行动清单</h4><div class="action-empty">还没有加入任务。请点击“查看诊断”，再在建议行动里加入任务。</div>`;
    return;
  }}
  const headers = ["门店", "责任人", "任务", "审批状态"];
  const head = headers.map(item => `<th>${{item}}</th>`).join("");
  const rows = actions.map(action => `<tr>${{headers.map(item => `<td>${{esc(action[item])}}</td>`).join("")}}</tr>`).join("");
  actionList.innerHTML = `<h4>本次行动清单</h4><div class="action-list-table-wrap"><table><thead><tr>${{head}}</tr></thead><tbody>${{rows}}</tbody></table></div>`;
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
st.title("🏬 购物中心经营风险智能决策 Agent")
st.caption("数据感知 → 动态风险判断 → 证据推理 → 行动草稿 → 审批审计")

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

tab_dashboard, tab_stores, tab_agent, tab_architecture, tab_rules = st.tabs(
    ["经营总览", "风险门店诊断", "AI 经营助理", "Agent 架构说明", "规则与边界"]
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
        "启用真实 AI 时优先使用 OpenAI tool calling 调用结构化查询、RAG 检索和行动草稿工具；"
        "Verifier 会做格式、数值、审批、引用和角色权限校验；未配置 AI API key 或工具调用不可用时自动使用本地结构化兜底，不会虚构真实预测概率。"
    )
    use_llm = st.toggle(
        "启用真实 AI 生成",
        value=bool(get_openai_api_key()),
        help="需要在 .streamlit/secrets.toml 中配置 ARK_API_KEY 或 OPENAI_API_KEY；关闭后使用本地 RAG 兜底。",
    )
    user_role = st.selectbox(
        "当前用户角色",
        ["管理层", "运营", "招商", "财务", "法务"],
        index=0,
        help="用于 Verifier 做权限边界校验；生产环境应来自登录用户权限。",
    )
    model_status = "，".join(get_openai_models())
    st.caption(
        f"AI 状态：{'已开启' if use_llm else '已关闭'}；"
        f"提供方：{get_ai_provider_name()}；"
        f"API key：{'已读取' if get_openai_api_key() else '未读取'}；"
        f"模型：{model_status}"
    )
    with st.expander("后端记录状态", expanded=False):
        stats = backend_stats()
        st.caption(
            f"当前用户：{st.session_state['backend_user_id']}｜"
            f"当前会话：{st.session_state['backend_session_id']}｜"
            f"数据库：{stats['db_path']}"
        )
        backend_col1, backend_col2, backend_col3, backend_col4, backend_col5 = st.columns(5)
        backend_col1.metric("会话数", int(stats["sessions"]))
        backend_col2.metric("消息数", int(stats["messages"]))
        backend_col3.metric("Agent runs", int(stats["runs"]))
        backend_col4.metric("步骤记录", int(stats["steps"]))
        backend_col5.metric("证据记录", int(stats["evidence"]))
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
        retrieval_debug = st.session_state.get("last_retrieval_debug")
        if retrieval_debug:
            st.caption(
                "最近一次 Hybrid Retrieval："
                f"Dense {retrieval_debug['dense_count']}｜"
                f"FTS {retrieval_debug['lexical_count']}｜"
                f"Merge {retrieval_debug['merged_count']}｜"
                f"Rerank {retrieval_debug['reranked_count']}｜"
                f"{retrieval_debug['retrieval_latency_ms']} ms｜"
                f"{retrieval_debug['vector_db']}｜"
                f"{retrieval_debug['embedding_provider']}:{retrieval_debug['embedding_model']}｜"
                f"dim={retrieval_debug.get('embedding_dimension', 'n/a')}｜"
                f"dense={retrieval_debug.get('dense_retrieval_status', 'unknown')}"
            )
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
            debug_results, retrieval_debug_info = search_rag_database_with_debug(debug_query.strip(), top_k=8)
            if retrieval_debug_info is not None:
                st.session_state["last_retrieval_debug"] = {
                    "retrieval_latency_ms": retrieval_debug_info.retrieval_latency_ms,
                    "dense_count": retrieval_debug_info.dense_count,
                    "lexical_count": retrieval_debug_info.lexical_count,
                    "merged_count": retrieval_debug_info.merged_count,
                    "reranked_count": retrieval_debug_info.reranked_count,
                    "embedding_provider": retrieval_debug_info.embedding_provider,
                    "embedding_model": retrieval_debug_info.embedding_model,
                    "embedding_dimension": retrieval_debug_info.embedding_dimension,
                    "vector_db": retrieval_debug_info.vector_db,
                    "dense_retrieval_status": retrieval_debug_info.dense_retrieval_status,
                    "config": retrieval_debug_info.config,
                }
            if not debug_results:
                st.info("没有检索到相关知识片段。")
            for index, result in enumerate(debug_results, start=1):
                with st.container(border=True):
                    st.markdown(f"**[{index}] {result.source}｜{result.title}｜得分 {result.score:.3f}**")
                    st.caption(
                        f"dense={float(result.metadata.get('dense_score') or 0):.3f}｜"
                        f"lexical={float(result.metadata.get('lexical_score') or 0):.3f}｜"
                        f"fusion={float(result.metadata.get('fusion_score') or 0):.3f}｜"
                        f"rerank={result.metadata.get('rerank_score') if result.metadata.get('rerank_score') is not None else 'disabled'}"
                    )
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
        run_started = time.perf_counter()
        run_id = create_agent_run(
            st.session_state["backend_session_id"],
            st.session_state["backend_user_id"],
            question,
            use_llm,
            get_openai_models() if use_llm else [],
        )
        save_message(st.session_state["backend_session_id"], "user", question, run_id=run_id)
        with st.status("Agent 工作流正在运行", expanded=True) as status:
            status.write("启动 Agent：准备解析问题")

            def update_agent_status(message: str) -> None:
                status.write(message)

            answer, result_df, evidence, trace = agent_answer(
                question,
                filtered,
                use_llm=use_llm,
                user_role=user_role,
                status_callback=update_agent_status,
            )
            status.update(label="Agent 工作流已完成", state="complete", expanded=False)
        latency_ms = int((time.perf_counter() - run_started) * 1000)
        intent = trace[0].detail if trace else None
        for step in trace:
            save_agent_step(run_id, step.name, step.detail, step.status)
        for index, doc in enumerate(evidence, start=1):
            save_rag_evidence(
                run_id,
                index,
                doc.source,
                doc.title,
                doc.content[:700],
                row_id=doc.row_id,
                dense_score=doc.dense_score,
                lexical_score=doc.lexical_score,
                fusion_score=doc.fusion_score,
                rerank_score=doc.rerank_score,
            )
        plan_for_approval = build_agent_plan(question, filtered)
        approval_ids = create_pending_approvals_for_run(run_id, answer, result_df, plan_for_approval)
        for approval_id in approval_ids:
            save_agent_step(
                run_id,
                "Human Approval",
                f"已生成待审批动作草稿：{approval_id}",
                "pending",
            )
        save_message(
            st.session_state["backend_session_id"],
            "assistant",
            answer,
            run_id=run_id,
            model=",".join(get_openai_models()) if use_llm else "local-rag-fallback",
            latency_ms=latency_ms,
        )
        latest_retrieval_debug = get_last_retrieval_debug()
        if latest_retrieval_debug is not None:
            st.session_state["last_retrieval_debug"] = latest_retrieval_debug
        retrieval_debug = st.session_state.get("last_retrieval_debug", {})
        finish_agent_run(
            run_id,
            "completed",
            intent=intent,
            latency_ms=latency_ms,
            metrics={
                "retrieval_latency_ms": retrieval_debug.get("retrieval_latency_ms"),
                "total_latency_ms": latency_ms,
                "retrieved_chunk_count": len(evidence),
                "reranked_chunk_count": retrieval_debug.get("reranked_count"),
                "model_name": ",".join(get_openai_models()) if use_llm else "local-rag-fallback",
                "tool_calls": [step.name for step in trace if step.name.startswith("Tool:")],
                "tool_success_count": sum(1 for step in trace if step.name.startswith("Tool:") and step.status != "失败"),
                "tool_failure_count": sum(1 for step in trace if step.name.startswith("Tool:") and step.status == "失败"),
            },
        )
        st.session_state["agent_answer"] = answer
        st.session_state["agent_result"] = result_df
        st.session_state["agent_evidence"] = evidence
        st.session_state["agent_trace"] = trace
        st.session_state["agent_run_id"] = run_id
        st.session_state["agent_approval_ids"] = approval_ids
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
                    st.caption(
                        f"dense={doc.dense_score if doc.dense_score is not None else 0:.3f}｜"
                        f"lexical={doc.lexical_score if doc.lexical_score is not None else 0:.3f}｜"
                        f"fusion={doc.fusion_score if doc.fusion_score is not None else 0:.3f}｜"
                        f"rerank={doc.rerank_score if doc.rerank_score is not None else 'disabled'}"
                    )
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
        with st.expander("Human Approval Prototype", expanded=bool(st.session_state.get("agent_approval_ids"))):
            render_approval_center(st.session_state.get("agent_run_id"))

with tab_architecture:
    st.subheader("Agent 企业化流程")
    st.markdown(
        """
当前版本是单 Agent + tool calling + RAG + 后端审计入库的经营风险智能决策系统。整体链路如下：

1. 用户在 Streamlit 前端输入经营风险问题。
2. 系统创建 `session_id` 和 `run_id`，将用户消息写入后端 SQLite。
3. Agent 进入 tool calling：模型根据问题选择结构化查询、RAG 检索或行动草稿工具。
4. 工具返回门店指标、制度证据和待审批动作建议。
5. Scoring Reviewer 读取评分标准、历史趋势和证据，对规则基础分做受控复核。
6. Generator 基于工具结果和评分复核生成中文诊断报告。
7. Verifier 做格式、数值、审批、引用和权限校验。
8. 最终答案流式展示，同时将回答、步骤、证据、模型和耗时写入后端。
"""
    )

    st.subheader("核心模块职责")
    module_rows = [
        {
            "模块": "Planner",
            "作用": "识别用户问题意图、筛选条件和审批边界。",
            "当前实现": "根据关键词和门店/业态/楼层字段生成 AgentPlan。",
            "企业化扩展": "接入行业 taxonomy、同义词词典、权限上下文和历史偏好。",
        },
        {
            "模块": "Tool Calling",
            "作用": "让模型按需调用业务工具，而不是只接受固定上下文。",
            "当前实现": "注册结构化查询、RAG 检索、行动草稿 3 个工具。",
            "企业化扩展": "继续接入工单、合同、财务、招商品牌库和审批系统。",
        },
        {
            "模块": "Retriever / RAG",
            "作用": "为回答提供制度、合同、SOP、OCR 和门店证据。",
            "当前实现": "SQLite FTS + 本地哈希向量混合召回。",
            "企业化扩展": "替换为 pgvector/Milvus/Elasticsearch hybrid search，并增加 rerank。",
        },
        {
            "模块": "Scoring Reviewer",
            "作用": "让 AI 参与风险评分复核，但不允许自由覆盖规则分。",
            "当前实现": "规则引擎先算 base_score，AI 基于规则配置、历史趋势和证据输出调整建议、建议分和可审计推理摘要。",
            "企业化扩展": "接入历史回测、动态阈值服务和二次模型审查，等级变化进入人工复核。",
        },
        {
            "模块": "Generator",
            "作用": "基于工具和证据生成管理层可读的风险诊断。",
            "当前实现": "按固定结构输出结论、证据、判断、AI评分复核过程、动作、补充数据和引用。",
            "企业化扩展": "按用户角色生成不同颗粒度报告，并支持模板化审批材料。",
        },
        {
            "模块": "Verifier",
            "作用": "检查回答是否可追溯、格式稳定且不越权。",
            "当前实现": "格式、数值、审批、引用、角色权限五层规则校验。",
            "企业化扩展": "加入 schema 校验、事实一致性校验、二次模型审查和合规策略中心。",
        },
        {
            "模块": "Backend Audit",
            "作用": "保存每次会话、消息、run、步骤、证据、模型和耗时。",
            "当前实现": "Streamlit 直接写入 `backend_db/mall_agent_backend.sqlite3`，FastAPI 提供查询接口。",
            "企业化扩展": "迁移到 PostgreSQL，加入用户体系、租户隔离、队列和监控。",
        },
    ]
    st.dataframe(pd.DataFrame(module_rows), hide_index=True, width="stretch")

    st.subheader("当前注册的 Tools")
    tool_rows = [
        {
            "Tool": "query_structured_risk_data",
            "输入": "用户问题 question",
            "输出": "意图、筛选条件、排序字段、查询摘要、重点门店、审批护栏",
            "用途": "把自然语言问题落到门店经营数据查询，例如最高风险、欠费、销售下滑、租售比压力。",
        },
        {
            "Tool": "search_rag_evidence",
            "输入": "检索 query、top_k",
            "输出": "合同条款、SOP、招商策略、OCR、门店经营数据等证据片段",
            "用途": "为回答提供可引用证据，减少模型凭空判断。",
        },
        {
            "Tool": "create_action_draft",
            "输入": "intent、needs_approval_guard、target_summary",
            "输出": "责任部门、行动任务、时限、是否需人工审批",
            "用途": "把诊断转成行动草稿，并对催缴、解约、保证金扣划、品牌替换等动作加审批边界。",
        },
    ]
    st.dataframe(pd.DataFrame(tool_rows), hide_index=True, width="stretch")

    st.subheader("Verifier 五层校验")
    verifier_rows = [
        {"层级": "格式校验", "检查内容": "固定标题是否完整，结论段是否乱带引用编号。"},
        {"层级": "数值校验", "检查内容": "回答是否提到结构化结果中的重点门店、风险分和欠费金额。"},
        {"层级": "审批校验", "检查内容": "催缴函、解约、保证金扣划、品牌替换等敏感动作是否标注需人工审批。"},
        {"层级": "引用校验", "检查内容": "关键证据是否绑定引用编号，编号是否超出实际 evidence 范围。"},
        {"层级": "权限校验", "检查内容": "不同角色是否越权输出法务、招商、财务敏感动作。"},
    ]
    st.dataframe(pd.DataFrame(verifier_rows), hide_index=True, width="stretch")

    st.subheader("多用户与并发思路")
    st.info(
        "当前每个浏览器会话都有独立 user_id 和 session_id，每次运行 Agent 都生成 run_id。"
        "生产环境可将 SQLite 换成 PostgreSQL，把长任务放入 Celery/RQ 队列，并通过 SSE 或 WebSocket 返回流式进度。"
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

AI 不直接拍脑袋给最终分，而是在规则基础分上做 Scoring Reviewer：

1. 读取 `risk_rule_config.json` 中的评分标准和最大调整幅度。
2. 读取结构化门店指标、近 3/6 个月历史趋势和 RAG 证据。
3. 输出 `规则基础分、AI建议调整分、建议分、置信度、证据依据、可审计推理摘要`。
4. 单次调整幅度默认限制在 -10 到 +10。
5. 如果建议分导致风险等级变化，必须进入人工复核。

系统不会展示模型隐藏的 chain-of-thought，而是展示可审计推理摘要，便于业务方复核和留痕。
"""
    )

    st.subheader("当前版本边界与生产化路线")
    st.warning(
        "当前数据以生成样本为主，不能据此声称某商户会在多少天后现金流断裂，也不能输出未经校准的真实违约概率。"
        "生产版本应接入至少 12 个月历史数据，以覆盖淡旺季、季度活动和同比变化；"
        "阈值应按业态、楼层、面积、租金等级和经营场景动态校准。"
        "评分规则应从代码迁移到规则配置表，并保留版本、owner、阈值来源、审批记录和历史回测指标。"
    )

    st.subheader("下载计算结果")
    output_csv = df.drop(columns=["触发依据"]).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载带风险计算结果的 CSV",
        data=output_csv,
        file_name="门店经营风险计算结果.csv",
        mime="text/csv",
    )
