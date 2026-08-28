from evaluation.evaluate_agent import evaluate_agent_rows
from evaluation.evaluate_generation import evaluate_generation
from evaluation.evaluate_retrieval import evaluate_retrieval_rows


def test_agent_metrics_are_computed_from_predictions():
    cases = [{"expected_intent": "arrears", "expected_filters": {"楼层": "F3"}, "expected_tool": ["query"]}]
    preds = [{"intent": "arrears", "filters": {"楼层": "F3"}, "tools": ["query"], "success": True}]
    metrics = evaluate_agent_rows(cases, preds)
    assert metrics["intent_accuracy"] == 1.0
    assert metrics["task_success_rate"] == 1.0


def test_retrieval_metrics_hit_recall_mrr():
    cases = [{"expected_evidence_sources": ["门店经营数据", "制度与模板"]}]
    preds = [[{"source": "其他"}, {"source": "制度与模板"}]]
    metrics = evaluate_retrieval_rows(cases, preds, k=2)
    assert metrics["hit@2"] == 1.0
    assert metrics["recall@2"] == 0.5
    assert metrics["mrr"] == 0.5


def test_generation_approval_boundary():
    answer = "### 结论\nx\n### 关键证据\n[1]\n### 风险判断\nx\n### 建议动作\n催缴，需人工审批\n### 需要补充的数据\n暂无\n### 引用证据\n[1] a"
    metrics = evaluate_generation(answer)
    assert metrics["required_section_compliance"] == 1.0
    assert metrics["approval_boundary_compliance"] == 1.0
