from __future__ import annotations

import json
import operator
import re
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).parent
RULE_CONFIG_PATH = PROJECT_DIR / "risk_rule_config.json"


def load_rule_config(path: Path = RULE_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rule_by_id(rule_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_rule_config()
    for rule in config.get("rules", []):
        if rule.get("id") == rule_id:
            return rule
    raise KeyError(f"Unknown risk rule: {rule_id}")


def threshold_score(rule_id: str, value: float, config: dict[str, Any] | None = None) -> int:
    rule = rule_by_id(rule_id, config)
    matched_score = 0
    for threshold in rule.get("baseline_thresholds", []):
        if condition_matches(float(value), str(threshold["condition"])):
            matched_score = max(matched_score, int(threshold["score"]))
    return matched_score


def condition_matches(value: float, condition: str) -> bool:
    condition = condition.strip().replace(",", "")
    condition = condition.replace("%", "")
    match = re.fullmatch(r"(>=|<=|>|<|==)\s*(-?\d+(?:\.\d+)?)", condition)
    if not match:
        raise ValueError(f"Unsupported threshold condition: {condition}")
    op_text, raw_number = match.groups()
    operations = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
    }
    return operations[op_text](value, float(raw_number))


def rule_config_summary(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_rule_config()
    return {
        "rule_version": config.get("rule_version"),
        "status": config.get("status"),
        "minimum_history_months": config.get("minimum_history_months"),
        "llm_max_delta": config.get("score_adjustment_policy", {}).get("max_delta_per_run"),
        "rules": [rule.get("id") for rule in config.get("rules", [])],
    }
