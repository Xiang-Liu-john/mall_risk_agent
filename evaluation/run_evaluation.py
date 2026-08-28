from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.evaluate_agent import evaluate_agent_rows
from evaluation.evaluate_generation import evaluate_generation_rows
from evaluation.evaluate_retrieval import evaluate_retrieval_rows
from rag.embedding_validation import validate_semantic_embedding
from rag.retriever import HybridRetriever, load_retrieval_config
from rag_store import RAG_DB_PATH, build_rag_database, close_search_retriever


DATASET_PATH = Path(__file__).with_name("golden_dataset.json")
OUTPUT_PATH = Path(__file__).with_name("latest_results.json")
COMPARISON_OUTPUT_PATH = Path(__file__).with_name("evaluation_comparison.json")


def load_cases() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def evaluate_retrieval_at_k(cases: list[dict[str, Any]], predictions: list[list[dict[str, Any]]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in (3, 5, 8):
        metrics.update(evaluate_retrieval_rows(cases, predictions, k=k))
    return metrics


def retrieval_predictions_for_config(cases: list[dict[str, Any]], config: dict[str, Any]) -> list[list[dict[str, Any]]]:
    retriever = HybridRetriever(RAG_DB_PATH, config=config)
    try:
        predictions = []
        for case in cases:
            rows, _debug = retriever.search(case["question"], top_k=8)
            predictions.append([{"source": row.source, "title": row.title} for row in rows])
        return predictions
    finally:
        retriever.close()


def metric_tuple(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (metrics.get("hit@8", 0.0), metrics.get("recall@8", 0.0), metrics.get("mrr", 0.0))


def build_deterministic_evaluation_answer(
    plan: Any,
    query_summary: str,
    result_df: Any,
    evidence: list[Any],
    money_fn: Any,
) -> str:
    if result_df is not None and not result_df.empty:
        key_rows = []
        for index, (_, row) in enumerate(result_df.head(3).iterrows(), start=1):
            arrears = float(row.get("欠费总额(元)", 0) or 0)
            key_rows.append(
                f"- {row.get('门店名称')}（{row.get('门店ID')}）：风险得分 "
                f"{int(float(row.get('计算风险得分', 0) or 0))}/100，"
                f"欠费 {money_fn(arrears)}，引用 [{min(index, max(1, len(evidence)))}]"
            )
        key_evidence = "\n".join(key_rows)
    else:
        key_evidence = "- 暂无匹配门店，引用 [1]" if evidence else "- 暂无匹配门店"

    references = "\n".join(
        f"- [{index}] {doc.source}｜{doc.title}" for index, doc in enumerate(evidence[:8], start=1)
    ) or "- 暂无"
    approval = "需人工审批" if plan.needs_approval_guard else "否"
    return (
        "### 结论\n"
        f"{query_summary}\n\n"
        "### 关键证据\n"
        f"{key_evidence}\n\n"
        "### 风险判断\n"
        f"- 风险类型：{plan.intent_label}\n\n"
        "### 建议动作\n"
        f"- 运营/财务｜复核结构化指标和证据来源｜1-3 个工作日｜{approval}\n\n"
        "### 需要补充的数据\n"
        "- 近 3-6 个月销售、回款、保证金余额、巡检记录和商户沟通记录。\n\n"
        "### 引用证据\n"
        f"{references}"
    )


def run_retrieval_comparison(cases: list[dict[str, Any]], semantic_ready: bool) -> dict[str, Any]:
    base_config = load_retrieval_config()
    baseline_config = {
        **base_config,
        "dense_weight": 0.0,
        "lexical_weight": 1.0,
        "final_top_k": 8,
        "dense_top_k": 40,
        "lexical_top_k": 40,
        "embedding_provider": "hashing_fallback",
    }
    baseline_predictions = retrieval_predictions_for_config(cases, baseline_config)
    retrieval_configs: list[dict[str, Any]] = [
        {
            "name": "hashing_fts",
            "status": "evaluated",
            "config": baseline_config,
            **evaluate_retrieval_at_k(cases, baseline_predictions),
        }
    ]

    parameter_candidates: list[dict[str, Any]] = []
    selected_config = base_config
    if semantic_ready:
        dense_config = {
            **base_config,
            "dense_weight": 1.0,
            "lexical_weight": 0.0,
            "final_top_k": 8,
        }
        dense_predictions = retrieval_predictions_for_config(cases, dense_config)
        retrieval_configs.append(
            {
                "name": "semantic_dense",
                "status": "evaluated",
                "config": dense_config,
                **evaluate_retrieval_at_k(cases, dense_predictions),
            }
        )

        for dense_weight, lexical_weight in ((0.5, 0.5), (0.6, 0.4), (0.7, 0.3)):
            for final_top_k in (5, 8, 10):
                candidate_config = {
                    **base_config,
                    "dense_weight": dense_weight,
                    "lexical_weight": lexical_weight,
                    "final_top_k": final_top_k,
                }
                predictions = retrieval_predictions_for_config(cases, candidate_config)
                metrics = evaluate_retrieval_at_k(cases, predictions)
                parameter_candidates.append(
                    {
                        "name": f"semantic_hybrid_{dense_weight:.1f}_{lexical_weight:.1f}_top{final_top_k}",
                        "status": "evaluated",
                        "config": candidate_config,
                        **metrics,
                    }
                )
        selected = max(parameter_candidates, key=lambda item: metric_tuple(item))
        selected_config = selected["config"]
        retrieval_configs.append({"name": "semantic_hybrid", **selected})
    else:
        reason = "Semantic embedding was not validated; semantic dense and semantic hybrid were not evaluated."
        retrieval_configs.extend(
            [
                {"name": "semantic_dense", "status": "skipped", "reason": reason},
                {"name": "semantic_hybrid", "status": "skipped", "reason": reason},
            ]
        )

    return {
        "dataset_size": len(cases),
        "semantic_ready": semantic_ready,
        "retrieval_configs": retrieval_configs,
        "parameter_candidates": parameter_candidates,
        "selected_config": selected_config,
        "selection_basis": (
            "Selected by Golden Dataset evaluation using Hit@8, Recall@8, then MRR."
            if semantic_ready
            else "Semantic parameter comparison was skipped because semantic embedding was not validated."
        ),
    }


def main() -> int:
    cases = load_cases()
    try:
        import pandas as pd  # noqa: F401
        from core.agent_logic import (
            DEFAULT_CSV,
            build_agent_plan,
            query_structured_risk_data,
            enrich,
            money,
            read_csv_robust,
            retrieve_context,
        )
    except Exception as exc:
        print(f"Evaluation dependencies unavailable: {exc}", file=sys.stderr)
        return 2

    semantic_validation = validate_semantic_embedding()
    data = enrich(read_csv_robust(DEFAULT_CSV))
    rag_build = build_rag_database(data)
    agent_predictions = []
    retrieval_predictions = []
    answers = []
    for case in cases:
        question = case["question"]
        plan = build_agent_plan(question, data)
        result_df, query_summary = query_structured_risk_data(plan, data)
        evidence_scope = result_df if not result_df.empty else data
        evidence = retrieve_context(question, evidence_scope, top_k=8)
        answer = build_deterministic_evaluation_answer(plan, query_summary, result_df, evidence, money)
        agent_predictions.append(
            {
                "intent": plan.intent,
                "filters": plan.filters,
                "tools": ["query_structured_risk_data", "search_rag_evidence"],
                "success": not result_df.empty or bool(evidence),
            }
        )
        retrieval_predictions.append([{"source": doc.source, "title": doc.title} for doc in evidence])
        answers.append(answer)
    retrieval_at_k = evaluate_retrieval_at_k(cases, retrieval_predictions)
    close_search_retriever()
    comparison = run_retrieval_comparison(cases, bool(semantic_validation.get("production_ready")))
    close_search_retriever()
    results = {
        "agent": evaluate_agent_rows(cases, agent_predictions),
        "retrieval": {key: value for key, value in retrieval_at_k.items() if key in {"hit@8", "recall@8", "mrr"}},
        "retrieval_at_k": retrieval_at_k,
        "generation": evaluate_generation_rows(cases, answers),
        "case_count": len(cases),
        "rag_build": rag_build,
        "semantic_embedding_validation": semantic_validation,
    }
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    COMPARISON_OUTPUT_PATH.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
