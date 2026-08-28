from __future__ import annotations

from typing import Any


def exact_match_dict(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(str(actual.get(key)) == str(value) for key, value in expected.items())


def evaluate_agent_rows(cases: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, float]:
    total = len(cases)
    if total == 0:
        return {"intent_accuracy": 0.0, "filter_accuracy": 0.0, "tool_selection_accuracy": 0.0, "task_success_rate": 0.0}
    intent_hits = 0
    filter_hits = 0
    tool_hits = 0
    task_hits = 0
    for case, pred in zip(cases, predictions):
        intent_ok = pred.get("intent") == case.get("expected_intent")
        filter_ok = exact_match_dict(case.get("expected_filters", {}), pred.get("filters", {}))
        expected_tools = set(case.get("expected_tool", []))
        actual_tools = set(pred.get("tools", []))
        tool_ok = expected_tools.issubset(actual_tools)
        intent_hits += int(intent_ok)
        filter_hits += int(filter_ok)
        tool_hits += int(tool_ok)
        task_hits += int(intent_ok and filter_ok and tool_ok and bool(pred.get("success")))
    return {
        "intent_accuracy": intent_hits / total,
        "filter_accuracy": filter_hits / total,
        "tool_selection_accuracy": tool_hits / total,
        "task_success_rate": task_hits / total,
    }
