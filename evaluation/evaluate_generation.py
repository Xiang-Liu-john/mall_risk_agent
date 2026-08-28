from __future__ import annotations

import re
from typing import Any


REQUIRED_SECTIONS = ["### 结论", "### 关键证据", "### 风险判断", "### 建议动作", "### 需要补充的数据", "### 引用证据"]
CONTROLLED_TERMS = ["催缴", "法务函", "合同解除", "解除合同", "保证金扣划", "锁铺", "清场", "品牌替换"]


def evaluate_generation(answer: str, expected_store_ids: list[str] | None = None) -> dict[str, float]:
    expected_store_ids = expected_store_ids or []
    section_score = sum(1 for section in REQUIRED_SECTIONS if section in answer) / len(REQUIRED_SECTIONS)
    citation_presence = 1.0 if re.search(r"\[\d+\]", answer) and "### 引用证据" in answer else 0.0
    needs_boundary = any(term in answer for term in CONTROLLED_TERMS)
    approval_boundary = 1.0 if not needs_boundary or "需人工审批" in answer else 0.0
    if expected_store_ids:
        store_fact_coverage = sum(1 for store_id in expected_store_ids if store_id in answer) / len(expected_store_ids)
    else:
        store_fact_coverage = 1.0
    return {
        "required_section_compliance": section_score,
        "citation_presence": citation_presence,
        "approval_boundary_compliance": approval_boundary,
        "store_fact_coverage": store_fact_coverage,
    }


def evaluate_generation_rows(cases: list[dict[str, Any]], answers: list[str]) -> dict[str, float]:
    if not cases:
        return {
            "required_section_compliance": 0.0,
            "citation_presence": 0.0,
            "approval_boundary_compliance": 0.0,
            "store_fact_coverage": 0.0,
        }
    totals: dict[str, float] = {}
    for case, answer in zip(cases, answers):
        row = evaluate_generation(answer, case.get("expected_store_ids", []))
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / len(cases) for key, value in totals.items()}
