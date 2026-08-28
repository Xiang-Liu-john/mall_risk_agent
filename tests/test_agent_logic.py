import pytest


pd = pytest.importorskip("pandas")

from core.agent_logic import (
    REQUIRED_ANSWER_SECTIONS,
    build_agent_plan,
    query_structured_risk_data,
    risk_score,
    verify_agent_answer,
)


def sample_data():
    return pd.DataFrame(
        [
            {
                "门店ID": "S001",
                "门店名称": "测试餐饮",
                "业态分类": "餐饮",
                "楼层": "F3",
                "本月销售额": 100000,
                "销售环比(%)": -25,
                "进店率(%)": 4,
                "成交转化率(%)": 7,
                "租售比(%)": 38,
                "欠费总额(元)": 80000,
                "欠费天数": 65,
                "保证金覆盖率(%)": 50,
                "近3月平均销售": 90000,
                "近6月平均销售": 110000,
                "近3月销售环比均值": -18,
                "近6月最高欠费": 90000,
                "近6月平均租售比": 36,
                "连续下滑月数": 3,
                "水电费波动(%)": -45,
                "近90天投诉数": 9,
                "触发依据": [],
                "计算风险得分": 88,
                "计算风险等级": "极高",
            }
        ]
    )


def test_intent_routing_and_filters():
    plan = build_agent_plan("分析 F3 餐饮欠费风险", sample_data())
    assert plan.intent == "arrears"
    assert plan.filters == {"业态分类": "餐饮", "楼层": "F3"}
    assert plan.needs_approval_guard is True


def test_structured_query_returns_filtered_rows():
    plan = build_agent_plan("分析 F3 餐饮欠费风险", sample_data())
    result, summary = query_structured_risk_data(plan, sample_data())
    assert len(result) == 1
    assert "欠费" in summary


def test_risk_score_high_risk_row():
    score, reasons, sub_scores = risk_score(sample_data().iloc[0])
    assert score >= 75
    assert reasons
    assert set(sub_scores) == {"经营风险分", "财务风险分", "运营风险分", "合同风险分"}


def test_verifier_requires_sections_and_approval_boundary():
    plan = build_agent_plan("分析欠费风险", sample_data())
    issues = verify_agent_answer("建议生成催缴函。", plan, [], result=sample_data())
    assert any("格式校验" in issue for issue in issues)
    assert any("审批校验" in issue for issue in issues)


def test_verifier_accepts_required_sections():
    plan = build_agent_plan("分析欠费风险", sample_data())
    answer = "\n".join([f"{section}\n测试餐饮 88 ¥8.0万 需人工审批 [1]" for section in REQUIRED_ANSWER_SECTIONS])
    issues = verify_agent_answer(answer, plan, [], result=sample_data())
    assert not any("审批校验" in issue for issue in issues)
