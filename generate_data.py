from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from risk_rules import threshold_score


SEED = 20260718
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_CSV = Path(__file__).parent / "data" / "购物中心100家门店经营数据_优化命名.csv"
OUTPUT_HISTORY_CSV = Path(__file__).parent / "data" / "门店经营月度历史数据_12个月.csv"
HISTORY_MONTHS = pd.period_range("2025-08", "2026-07", freq="M")

CATEGORY_COUNTS = {
    "精品零售": 30,
    "餐饮": 26,
    "儿童配套": 16,
    "主力店": 10,
    "生活服务": 10,
    "潮流运动": 8,
}

SCENARIO_RATIOS = {
    "健康": 0.45,
    "短期波动": 0.15,
    "转化恶化": 0.10,
    "持续下滑": 0.10,
    "欠费承压": 0.08,
    "疑似撤店": 0.05,
    "区域性下滑": 0.07,
}

CATEGORY_BASE = {
    "精品零售": {"area": (90, 260), "sales": (160_000, 620_000), "floor": ["F1", "F2", "F3"]},
    "餐饮": {"area": (120, 520), "sales": (260_000, 1_250_000), "floor": ["F3", "F4", "F5"]},
    "儿童配套": {"area": (140, 420), "sales": (150_000, 680_000), "floor": ["F3", "F4"]},
    "主力店": {"area": (700, 2200), "sales": (1_600_000, 5_200_000), "floor": ["B1", "F1", "F5"]},
    "生活服务": {"area": (60, 180), "sales": (80_000, 320_000), "floor": ["B1", "F2", "F4"]},
    "潮流运动": {"area": (120, 360), "sales": (240_000, 900_000), "floor": ["F1", "F2", "F3"]},
}

STORE_NAMES = {
    "精品零售": [
        "森屿女装",
        "MONO集物",
        "梵木生活",
        "织造局",
        "NORM服饰",
        "小野饰品",
        "半糖香氛",
        "KIKI潮玩",
        "青禾眼镜",
        "方寸书店",
        "晨白美妆",
        "LUMA家居",
        "GLINT饰界",
        "素然衣橱",
        "岚裳集合",
        "初见花艺",
        "一家好物",
        "未央皮具",
        "松果文创",
        "星期八潮品",
        "ORIGIN男装",
        "MUSE女装",
        "山茶珠宝",
        "蓝盒数码",
        "拾光礼品",
        "北街鞋履",
        "PAPERMOOD文具",
        "芙鹿香氛",
        "AFTER RAIN服饰",
        "COTTON TREE内衣",
    ],
    "餐饮": [
        "川隐小馆",
        "渔火砂锅",
        "山城冒菜",
        "鹿港茶餐厅",
        "南洋椰子鸡",
        "松鹤烧肉",
        "京巷烤鸭",
        "米仓日料",
        "半山牛肉粉",
        "桂满陇",
        "花椒院子",
        "竹里火锅",
        "青柠泰厨",
        "云海米线",
        "烟火烤肉",
        "一碗兰州",
        "黑石牛排",
        "喜悦甜品",
        "茶庭书道",
        "林里柠檬茶",
        "野火披萨",
        "稻田寿司",
        "小满粥铺",
        "三巡咖啡",
        "沸点酸菜鱼",
        "谷雨面馆",
    ],
    "儿童配套": [
        "乐芽成长中心",
        "星宝亲子馆",
        "童学社",
        "小鲸游泳学院",
        "彩虹艺术课堂",
        "奇趣科学馆",
        "知禾阅读馆",
        "启星编程",
        "咕噜儿童摄影",
        "小小运动家",
        "贝塔积木工坊",
        "萌芽托育",
        "云朵舞蹈教室",
        "海豚音乐启蒙",
        "木马王国乐园",
        "南瓜少儿美术",
    ],
    "主力店": [
        "星幕影城",
        "悦购精品超市",
        "盒里鲜生",
        "万家生活超市",
        "云仓家居",
        "乐享健身中心",
        "知新书城",
        "极光电器",
        "都会百货",
        "NOVA数字生活馆",
    ],
    "生活服务": [
        "净衣坊洗护",
        "青柠美甲",
        "悦修手机维修",
        "安齿口腔",
        "云剪造型",
        "邻里花店",
        "轻氧皮肤管理",
        "白塔照相馆",
        "快印先生",
        "暖心宠物护理",
    ],
    "潮流运动": [
        "跃动工场",
        "风速跑步中心",
        "潮跑装备",
        "燃点篮球馆",
        "锋线足球社",
        "飞轮骑行",
        "MOTION LAB运动集合",
        "山野户外",
    ],
}


def allocate_scenarios_by_category(stores_df: pd.DataFrame) -> dict[str, str]:
    rng = random.Random(SEED)
    scenario_map: dict[str, str] = {}

    for _, group in stores_df.groupby("业态分类"):
        store_ids = group["门店ID"].tolist()
        rng.shuffle(store_ids)
        total = len(store_ids)
        assigned: list[str] = []

        for scenario, ratio in SCENARIO_RATIOS.items():
            assigned.extend([scenario] * int(total * ratio))

        fillers = ["健康", "短期波动", "健康"]
        filler_index = 0
        while len(assigned) < total:
            assigned.append(fillers[filler_index % len(fillers)])
            filler_index += 1

        assigned = assigned[:total]
        rng.shuffle(assigned)
        scenario_map.update(dict(zip(store_ids, assigned)))

    return scenario_map


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def build_store_base() -> pd.DataFrame:
    rows = []
    idx = 1
    rng = random.Random(SEED)

    for category, count in CATEGORY_COUNTS.items():
        config = CATEGORY_BASE[category]
        names = STORE_NAMES[category]
        if len(names) != count:
            raise ValueError(f"{category} needs {count} names, got {len(names)}")

        for name in names:

            rows.append(
                {
                    "门店ID": f"SHOP_{idx:03d}",
                    "门店名称": name,
                    "业态分类": category,
                    "楼层": rng.choice(config["floor"]),
                    "租赁面积(sqm)": rng.randint(*config["area"]),
                    "基准销售额": rng.randint(*config["sales"]),
                }
            )
            idx += 1

    stores = pd.DataFrame(rows)
    scenario_map = allocate_scenarios_by_category(stores)
    stores["经营场景"] = stores["门店ID"].map(scenario_map)
    return stores


def apply_scenario(row: pd.Series, rng: random.Random) -> dict[str, float | int | str]:
    scenario = row["经营场景"]
    category = row["业态分类"]
    base_sales = float(row["基准销售额"])

    values = {
        "销售环比(%)": rng.uniform(-3, 8),
        "进店率(%)": rng.uniform(9, 22),
        "成交转化率(%)": rng.uniform(14, 34),
        "租售比(%)": rng.uniform(10, 21),
        "欠费总额(元)": 0,
        "欠费天数": 0,
        "水电费波动(%)": rng.uniform(-5, 8),
        "近90天投诉数": rng.randint(0, 3),
        "保证金覆盖率(%)": rng.uniform(80, 160),
    }

    if scenario == "短期波动":
        values.update(
            {
                "销售环比(%)": rng.uniform(-12, -4),
                "进店率(%)": rng.uniform(7, 16),
                "成交转化率(%)": rng.uniform(11, 26),
                "租售比(%)": rng.uniform(14, 24),
                "水电费波动(%)": rng.uniform(-10, 6),
                "近90天投诉数": rng.randint(1, 5),
            }
        )
    elif scenario == "转化恶化":
        values.update(
            {
                "销售环比(%)": rng.uniform(-18, -7),
                "进店率(%)": rng.uniform(8, 18),
                "成交转化率(%)": rng.uniform(5, 10),
                "租售比(%)": rng.uniform(18, 30),
                "近90天投诉数": rng.randint(2, 7),
            }
        )
    elif scenario == "持续下滑":
        values.update(
            {
                "销售环比(%)": rng.uniform(-30, -16),
                "进店率(%)": rng.uniform(4, 10),
                "成交转化率(%)": rng.uniform(7, 15),
                "租售比(%)": rng.uniform(20, 34),
                "欠费总额(元)": rng.randint(0, 18_000),
                "欠费天数": rng.randint(0, 25),
                "水电费波动(%)": rng.uniform(-20, -4),
                "近90天投诉数": rng.randint(3, 8),
            }
        )
    elif scenario == "欠费承压":
        values.update(
            {
                "销售环比(%)": rng.uniform(-24, -8),
                "进店率(%)": rng.uniform(5, 14),
                "成交转化率(%)": rng.uniform(8, 18),
                "租售比(%)": rng.uniform(26, 40),
                "欠费总额(元)": rng.randint(18_000, 68_000),
                "欠费天数": rng.randint(20, 75),
                "水电费波动(%)": rng.uniform(-18, 4),
                "保证金覆盖率(%)": rng.uniform(35, 95),
                "近90天投诉数": rng.randint(2, 8),
            }
        )
    elif scenario == "疑似撤店":
        values.update(
            {
                "销售环比(%)": rng.uniform(-38, -20),
                "进店率(%)": rng.uniform(2, 7),
                "成交转化率(%)": rng.uniform(5, 12),
                "租售比(%)": rng.uniform(25, 42),
                "欠费总额(元)": rng.randint(22_000, 88_000),
                "欠费天数": rng.randint(35, 95),
                "水电费波动(%)": rng.uniform(-58, -32),
                "保证金覆盖率(%)": rng.uniform(20, 80),
                "近90天投诉数": rng.randint(3, 10),
            }
        )
    elif scenario == "区域性下滑":
        values.update(
            {
                "销售环比(%)": rng.uniform(-22, -10),
                "进店率(%)": rng.uniform(5, 12),
                "成交转化率(%)": rng.uniform(10, 22),
                "租售比(%)": rng.uniform(17, 29),
                "水电费波动(%)": rng.uniform(-16, 2),
                "近90天投诉数": rng.randint(1, 6),
            }
        )

    category_sales_factor = {
        "主力店": 1.10,
        "餐饮": 1.05,
        "儿童配套": 0.95,
        "生活服务": 0.85,
    }.get(category, 1.0)

    sales = base_sales * category_sales_factor * (1 + values["销售环比(%)"] / 100)
    values["本月销售额"] = int(max(30_000, sales))

    if category == "儿童配套":
        risk_push = max(0, -values["销售环比(%)"])
        values["退款申请数"] = int(max(0, round(rng.uniform(0, 4) + risk_push / 8 + values["近90天投诉数"] / 4)))
        values["续费率(%)"] = round(clamp(rng.uniform(58, 88) - risk_push * 0.7, 35, 92), 1)
        values["安全巡检分"] = round(clamp(rng.uniform(82, 98) - max(0, values["水电费波动(%)"] * -0.2), 70, 100), 1)
        values["家长投诉率(‰)"] = round(clamp(rng.uniform(0.5, 3.5) + values["近90天投诉数"] * 0.35, 0, 9), 2)
    else:
        values["退款申请数"] = 0
        values["续费率(%)"] = np.nan
        values["安全巡检分"] = np.nan
        values["家长投诉率(‰)"] = np.nan

    return values


def score_row(row: pd.Series) -> tuple[int, str]:
    business = 0
    financial = 0
    operation = 0
    contract = 0

    mom = row["销售环比(%)"]
    business += threshold_score("sales_mom_decline", float(mom))
    if row["进店率(%)"] < 5:
        business += 6
    if row["成交转化率(%)"] < 8:
        business += 6
    consecutive_declines = float(row.get("连续下滑月数", 0))
    recent_mom_avg = float(row.get("近3月销售环比均值", 0))
    if consecutive_declines >= 3:
        business += 8
    elif consecutive_declines >= 2 and recent_mom_avg < 0:
        business += 4
    if row["业态分类"] == "儿童配套":
        if row["退款申请数"] >= 6:
            business += 4
        if row["续费率(%)"] < 55:
            business += 6
    business = min(40, business)

    arrears = row["欠费总额(元)"]
    financial += threshold_score("arrears_amount", float(arrears))
    rent_ratio = row["租售比(%)"]
    financial += threshold_score("rent_to_sales_ratio", float(rent_ratio))
    six_month_rent_ratio = float(row.get("近6月平均租售比", 0))
    if six_month_rent_ratio >= 30 and rent_ratio >= 25:
        financial += 4
    financial = min(30, financial)

    utility = row["水电费波动(%)"]
    operation += threshold_score("utility_drop", float(utility))
    if row["近90天投诉数"] >= 8:
        operation += 6
    elif row["近90天投诉数"] >= 4:
        operation += 3
    if row["业态分类"] == "儿童配套":
        if row["安全巡检分"] < 82:
            operation += 5
        if row["家长投诉率(‰)"] >= 5:
            operation += 4
    operation = min(20, operation)

    if row["欠费天数"] >= 60:
        contract += 6
    elif row["欠费天数"] >= 30:
        contract += 4
    if row["保证金覆盖率(%)"] < 60:
        contract += 4
    elif row["保证金覆盖率(%)"] < 90:
        contract += 2
    contract = min(10, contract)

    total = min(100, business + financial + operation + contract)
    if total >= 75:
        level = "极高"
    elif total >= 55:
        level = "高"
    elif total >= 30:
        level = "中"
    else:
        level = "低"
    return total, level


def apply_seeded_risk_cases(df: pd.DataFrame) -> pd.DataFrame:
    cases = {
        "精品零售": {
            "经营场景": "转化恶化",
            "销售环比(%)": -32.0,
            "进店率(%)": 7.0,
            "成交转化率(%)": 5.8,
            "租售比(%)": 36.0,
            "欠费总额(元)": 12_000,
            "欠费天数": 32,
            "水电费波动(%)": -12.0,
            "近90天投诉数": 6,
        },
        "餐饮": {
            "经营场景": "持续下滑",
            "销售环比(%)": -32.0,
            "进店率(%)": 4.6,
            "成交转化率(%)": 10.0,
            "租售比(%)": 29.0,
            "欠费总额(元)": 26_000,
            "欠费天数": 32,
            "水电费波动(%)": -22.0,
            "近90天投诉数": 5,
        },
        "儿童配套": {
            "经营场景": "欠费承压",
            "销售环比(%)": -20.0,
            "进店率(%)": 6.5,
            "成交转化率(%)": 8.5,
            "租售比(%)": 28.0,
            "欠费总额(元)": 31_000,
            "欠费天数": 38,
            "保证金覆盖率(%)": 75.0,
            "水电费波动(%)": -10.0,
            "近90天投诉数": 6,
            "退款申请数": 7,
            "续费率(%)": 51.0,
            "安全巡检分": 80.0,
            "家长投诉率(‰)": 5.2,
        },
        "主力店": {
            "经营场景": "区域性下滑",
            "销售环比(%)": -26.0,
            "进店率(%)": 4.8,
            "成交转化率(%)": 13.0,
            "租售比(%)": 22.0,
            "欠费总额(元)": 0,
            "欠费天数": 0,
            "水电费波动(%)": -18.0,
            "近90天投诉数": 4,
        },
        "生活服务": {
            "经营场景": "欠费承压",
            "销售环比(%)": -24.0,
            "进店率(%)": 4.5,
            "成交转化率(%)": 9.0,
            "租售比(%)": 34.0,
            "欠费总额(元)": 52_000,
            "欠费天数": 64,
            "保证金覆盖率(%)": 55.0,
            "水电费波动(%)": -22.0,
            "近90天投诉数": 5,
        },
        "潮流运动": {
            "经营场景": "疑似撤店",
            "销售环比(%)": -36.0,
            "进店率(%)": 3.5,
            "成交转化率(%)": 6.0,
            "租售比(%)": 38.0,
            "欠费总额(元)": 66_000,
            "欠费天数": 72,
            "保证金覆盖率(%)": 48.0,
            "水电费波动(%)": -45.0,
            "近90天投诉数": 8,
        },
    }

    for category, updates in cases.items():
        index = df.index[df["业态分类"] == category][0]
        for column, value in updates.items():
            df.at[index, column] = value
        df.at[index, "本月销售额"] = int(
            max(30_000, df.at[index, "本月销售额"] * (1 + updates["销售环比(%)"] / 100))
        )
    return df


def scenario_monthly_adjustment(scenario: str, month_index: int, rng: random.Random) -> dict[str, float]:
    progress = month_index / max(1, len(HISTORY_MONTHS) - 1)
    noise = rng.uniform(-0.025, 0.025)
    if scenario == "健康":
        return {"sales_factor": 1.0 + 0.06 * progress + noise, "arrears_factor": 0.0, "utility_factor": 1.0 + rng.uniform(-0.04, 0.05)}
    if scenario == "短期波动":
        return {"sales_factor": 1.0 - 0.06 * progress + noise, "arrears_factor": 0.1 * progress, "utility_factor": 1.0 + rng.uniform(-0.08, 0.04)}
    if scenario == "转化恶化":
        return {"sales_factor": 1.0 - 0.16 * progress + noise, "arrears_factor": 0.25 * progress, "utility_factor": 1.0 + rng.uniform(-0.08, 0.02)}
    if scenario == "持续下滑":
        return {"sales_factor": 1.0 - 0.28 * progress + noise, "arrears_factor": 0.45 * progress, "utility_factor": 1.0 - 0.12 * progress + rng.uniform(-0.06, 0.02)}
    if scenario == "欠费承压":
        return {"sales_factor": 0.98 - 0.18 * progress + noise, "arrears_factor": 0.75 * progress, "utility_factor": 0.98 - 0.08 * progress + rng.uniform(-0.06, 0.03)}
    if scenario == "疑似撤店":
        return {"sales_factor": 0.96 - 0.38 * progress + noise, "arrears_factor": 1.0 * progress, "utility_factor": 0.96 - 0.34 * progress + rng.uniform(-0.08, 0.02)}
    if scenario == "区域性下滑":
        return {"sales_factor": 1.0 - 0.20 * progress + noise, "arrears_factor": 0.2 * progress, "utility_factor": 1.0 - 0.10 * progress + rng.uniform(-0.05, 0.02)}
    return {"sales_factor": 1.0 + noise, "arrears_factor": 0.0, "utility_factor": 1.0}


def build_monthly_history(current_df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED + 12)
    rows = []
    seasonality = {1: 0.92, 2: 0.88, 3: 0.98, 4: 1.02, 5: 1.05, 6: 1.00, 7: 0.96, 8: 0.98, 9: 1.02, 10: 1.08, 11: 1.12, 12: 1.18}
    for _, row in current_df.iterrows():
        base_sales = float(row["本月销售额"])
        base_arrears = float(row["欠费总额(元)"])
        base_complaints = float(row.get("近90天投诉数", 0))
        scenario = str(row["经营场景"])
        previous_sales: float | None = None
        cumulative_arrears_days = 0
        for month_index, period in enumerate(HISTORY_MONTHS):
            adjustment = scenario_monthly_adjustment(scenario, month_index, rng)
            sales = max(20_000, base_sales * adjustment["sales_factor"] * seasonality[int(period.month)])
            if month_index == len(HISTORY_MONTHS) - 1:
                sales = base_sales
            sales_mom = 0.0 if previous_sales in (None, 0) else (sales / previous_sales - 1) * 100
            previous_sales = sales

            arrears_amount = int(max(0, base_arrears * adjustment["arrears_factor"] + rng.uniform(-2500, 2500)))
            if month_index == len(HISTORY_MONTHS) - 1:
                arrears_amount = int(base_arrears)
            if arrears_amount > 0:
                cumulative_arrears_days = min(120, cumulative_arrears_days + rng.randint(4, 10))
            else:
                cumulative_arrears_days = max(0, cumulative_arrears_days - rng.randint(8, 18))
            if month_index == len(HISTORY_MONTHS) - 1:
                cumulative_arrears_days = int(row.get("欠费天数", cumulative_arrears_days))

            rent_to_sales = float(row["租售比(%)"]) * (base_sales / max(sales, 1))
            utility_change = (adjustment["utility_factor"] - 1) * 100 + rng.uniform(-3, 3)
            if month_index == len(HISTORY_MONTHS) - 1:
                utility_change = float(row["水电费波动(%)"])
            complaints = max(0, int(round(base_complaints * (0.45 + 0.55 * month_index / max(1, len(HISTORY_MONTHS) - 1)) + rng.uniform(-1, 1))))
            if month_index == len(HISTORY_MONTHS) - 1:
                complaints = int(row.get("近90天投诉数", complaints))

            rows.append(
                {
                    "月份": str(period),
                    "门店ID": row["门店ID"],
                    "门店名称": row["门店名称"],
                    "业态分类": row["业态分类"],
                    "楼层": row["楼层"],
                    "经营场景": row["经营场景"],
                    "租赁面积(sqm)": row["租赁面积(sqm)"],
                    "月销售额": int(sales),
                    "销售环比(%)": round(sales_mom, 1),
                    "租售比(%)": round(rent_to_sales, 1),
                    "欠费总额(元)": arrears_amount,
                    "欠费天数": cumulative_arrears_days,
                    "水电费波动(%)": round(utility_change, 1),
                    "近90天投诉数": complaints,
                    "进店率(%)": round(max(1.0, float(row["进店率(%)"]) * (0.9 + 0.2 * sales / max(base_sales, 1))), 1),
                    "成交转化率(%)": round(max(2.0, float(row["成交转化率(%)"]) * (0.92 + 0.15 * sales / max(base_sales, 1))), 1),
                    "保证金覆盖率(%)": round(max(10, float(row["保证金覆盖率(%)"]) - arrears_amount / 2000), 1),
                }
            )
    history = pd.DataFrame(rows)
    history["销售同比(%)"] = 0.0
    return history


def add_history_features(current_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    result = current_df.copy()
    tail = history_df.sort_values(["门店ID", "月份"]).groupby("门店ID").tail(6)
    features = (
        tail.groupby("门店ID")
        .agg(
            近3月平均销售=("月销售额", lambda series: round(series.tail(3).mean(), 1)),
            近6月平均销售=("月销售额", lambda series: round(series.tail(6).mean(), 1)),
            近3月销售环比均值=("销售环比(%)", lambda series: round(series.tail(3).mean(), 1)),
            近6月最高欠费=("欠费总额(元)", "max"),
            近6月平均租售比=("租售比(%)", lambda series: round(series.tail(6).mean(), 1)),
        )
        .reset_index()
    )
    declining = (
        history_df.sort_values(["门店ID", "月份"])
        .groupby("门店ID")
        .tail(3)
        .assign(是否下滑=lambda frame: frame["销售环比(%)"] < 0)
        .groupby("门店ID")["是否下滑"]
        .sum()
        .rename("连续下滑月数")
        .reset_index()
    )
    result = result.merge(features, on="门店ID", how="left").merge(declining, on="门店ID", how="left")
    result["连续下滑月数"] = result["连续下滑月数"].fillna(0).astype(int)
    return result


def main() -> None:
    rng = random.Random(SEED)
    stores = build_store_base()
    generated = []
    for _, row in stores.iterrows():
        values = row.to_dict()
        values.update(apply_scenario(row, rng))
        generated.append(values)

    df = pd.DataFrame(generated)
    df = df.drop(columns=["基准销售额"])
    df = apply_seeded_risk_cases(df)
    history_df = build_monthly_history(df)
    df = add_history_features(df, history_df)
    df["风险得分"], df["风险等级"] = zip(*df.apply(score_row, axis=1))

    column_order = [
        "门店ID",
        "门店名称",
        "业态分类",
        "楼层",
        "经营场景",
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
        "风险等级",
    ]
    df = df[column_order]
    numeric_round = [
        "销售环比(%)",
        "进店率(%)",
        "成交转化率(%)",
        "租售比(%)",
        "保证金覆盖率(%)",
        "近3月平均销售",
        "近6月平均销售",
        "近3月销售环比均值",
        "近6月平均租售比",
        "水电费波动(%)",
    ]
    df[numeric_round] = df[numeric_round].round(1)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    history_df.to_csv(OUTPUT_HISTORY_CSV, index=False, encoding="utf-8-sig")
    print(f"Generated {len(df)} rows -> {OUTPUT_CSV}")
    print(f"Generated {len(history_df)} monthly rows -> {OUTPUT_HISTORY_CSV}")


if __name__ == "__main__":
    main()
