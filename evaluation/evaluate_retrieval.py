from __future__ import annotations

from typing import Any


def evaluate_retrieval_rows(cases: list[dict[str, Any]], predictions: list[list[dict[str, Any]]], k: int = 8) -> dict[str, float]:
    evaluated = 0
    hit_total = 0
    recall_total = 0.0
    rr_total = 0.0
    for case, result_rows in zip(cases, predictions):
        expected_sources = set(case.get("expected_evidence_sources", []))
        if not expected_sources:
            continue
        evaluated += 1
        top_rows = result_rows[:k]
        actual_sources = [row.get("source") for row in top_rows]
        hits = [source for source in actual_sources if source in expected_sources]
        hit_total += int(bool(hits))
        recall_total += len(set(hits)) / len(expected_sources)
        reciprocal_rank = 0.0
        for index, source in enumerate(actual_sources, start=1):
            if source in expected_sources:
                reciprocal_rank = 1.0 / index
                break
        rr_total += reciprocal_rank
    if evaluated == 0:
        return {f"hit@{k}": 0.0, f"recall@{k}": 0.0, "mrr": 0.0}
    return {f"hit@{k}": hit_total / evaluated, f"recall@{k}": recall_total / evaluated, "mrr": rr_total / evaluated}
