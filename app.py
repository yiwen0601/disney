"""
面向上海迪士尼乐园运营的客流预测与商业洞察分析系统
"""

import json
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from urllib.parse import urlencode
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODELSCOPE_ACCESS_TOKEN = "ms-78a67a75-3b81-4796-b244-92ab0cf0b09c"
DEFAULT_MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1/"
DEFAULT_MODELSCOPE_MODEL = "ZhipuAI/GLM-5.1"
DEFAULT_MODELSCOPE_FALLBACK_MODELS = (
    "MiniMax/MiniMax-M2.7",
    "moonshotai/Kimi-K2.5",
    "Qwen/Qwen3.5-35B-A3B",
)
DEFAULT_AMAP_WEATHER_KEY = "04c837fddf6db740b94f04853ed9d266"
DEFAULT_AMAP_CITY_CODE = "310000"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

app = Flask(__name__)

# 加载模型
model_data = joblib.load(MODEL_DIR / "disney_attendance_model.joblib")
model = model_data["model"]
scaler = model_data["scaler"]
feature_columns = list(model_data["feature_columns"])

# 加载历史数据
historical_df = pd.read_csv(PROCESSED_DATA_DIR / "shanghai_disney_featured.csv")
historical_df["date"] = pd.to_datetime(historical_df["date"])
historical_df["date_key"] = historical_df["date"].dt.date
historical_df = historical_df.sort_values("date").reset_index(drop=True)
historical_by_date = historical_df.set_index("date_key")

HISTORICAL_ATTENDANCE_LOOKUP = {
    row_date: float(attendance)
    for row_date, attendance in zip(historical_df["date_key"], historical_df["attendance"])
}
LAST_HISTORY_DATE = historical_df["date_key"].max()
AVERAGE_ATTENDANCE = float(historical_df["attendance"].mean())
ATTENDANCE_Q35 = float(historical_df["attendance"].quantile(0.35))
ATTENDANCE_Q60 = float(historical_df["attendance"].quantile(0.60))
ATTENDANCE_Q82 = float(historical_df["attendance"].quantile(0.82))
PREDICTION_MIN = int(historical_df["attendance"].quantile(0.02))
PREDICTION_MAX = int(historical_df["attendance"].quantile(0.98) * 1.08)
MONTHLY_BASELINES = historical_df.groupby("month")["attendance"].mean().to_dict()
MONTH_WEEKDAY_BASELINES = historical_df.groupby(["month", "weekday"])["attendance"].mean().to_dict()
MONTHLY_CLIMATE = historical_df.groupby("month").agg(
    avg_temp=("temperature", "mean"),
    rain_probability=("is_rainy", "mean"),
)
SEASON_ENCODING = (
    historical_df.groupby("season")["season_encoded"]
    .agg(lambda series: int(series.mode().iloc[0]))
    .to_dict()
)

WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
HOLIDAY_NAME_MAP = {
    "new_year": "元旦",
    "spring_festival": "春节",
    "qingming": "清明假期",
    "labor_day": "劳动节",
    "dragon_boat": "端午节",
    "mid_autumn": "中秋节",
    "national_day": "国庆黄金周",
}
HOLIDAY_RULES = {
    "元旦": {"start": (1, 1), "duration": 3, "multiplier": 1.30},
    "春节": {"start": (2, 10), "duration": 7, "multiplier": 1.85},
    "清明假期": {"start": (4, 4), "duration": 3, "multiplier": 1.35},
    "劳动节": {"start": (5, 1), "duration": 5, "multiplier": 1.65},
    "端午节": {"start": (6, 10), "duration": 3, "multiplier": 1.30},
    "中秋节": {"start": (9, 15), "duration": 3, "multiplier": 1.32},
    "国庆黄金周": {"start": (10, 1), "duration": 7, "multiplier": 1.95},
}
SPECIAL_EVENTS = [
    {"name": "冬季节庆季", "start": (12, 15), "end": (1, 7)},
    {"name": "春日主题季", "start": (3, 15), "end": (5, 5)},
    {"name": "夏日庆典季", "start": (6, 20), "end": (8, 31)},
    {"name": "万圣狂欢季", "start": (10, 1), "end": (10, 31)},
]


class AIInsightError(RuntimeError):
    """AI 建议生成失败"""


class WeatherAPIError(RuntimeError):
    """天气接口调用失败"""


def get_modelscope_access_token():
    """优先使用环境变量，否则使用默认注入的 Token"""
    token = os.getenv("MODELSCOPE_ACCESS_TOKEN", DEFAULT_MODELSCOPE_ACCESS_TOKEN).strip()
    return token


def get_modelscope_model_candidates():
    """返回主模型与备用模型列表，自动去重"""
    primary_model = os.getenv("MODELSCOPE_MODEL", DEFAULT_MODELSCOPE_MODEL).strip()
    fallback_models = os.getenv(
        "MODELSCOPE_FALLBACK_MODELS",
        ",".join(DEFAULT_MODELSCOPE_FALLBACK_MODELS),
    )

    model_candidates = []
    for model_name in [primary_model, *fallback_models.split(",")]:
        cleaned = model_name.strip()
        if cleaned and cleaned not in model_candidates:
            model_candidates.append(cleaned)

    return model_candidates


def get_amap_weather_key():
    """优先使用环境变量，否则使用默认注入的高德 Key"""
    return os.getenv("AMAP_WEATHER_KEY", DEFAULT_AMAP_WEATHER_KEY).strip()


def get_amap_city_code():
    """默认使用上海天气，可通过环境变量覆盖"""
    return os.getenv("AMAP_CITY_CODE", DEFAULT_AMAP_CITY_CODE).strip()


def to_cn_date_label(date_str):
    """把 ISO 日期转成更适合中文生成的日期标签"""
    parsed = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{parsed.month}月{parsed.day}日"


def to_percent_label(value):
    """把 0-1 概率转成百分比整数"""
    return int(round(float(value) * 100))


def get_rain_risk_label(probability):
    """把降雨概率映射成更自然的风险描述"""
    if probability >= 0.6:
        return "降雨风险高"
    if probability >= 0.45:
        return "降雨风险较高"
    if probability >= 0.3:
        return "有一定降雨风险"
    return "降雨风险较低"


def get_park_hours_label(park_hours):
    """把营业时段转成更适合自然语言生成的标签"""
    open_time, close_time = park_hours.split("-")
    if open_time <= "08:00" and close_time >= "22:00":
        return "早开园且晚间延时运营"
    if close_time >= "21:30":
        return "晚间延时运营"
    return "常规运营时段"


def infer_rain_probability(dayweather, nightweather):
    """根据高德天气描述估算降雨概率"""
    weather_text = f"{dayweather or ''} {nightweather or ''}"
    weather_scores = [
        ("特大暴雨", 0.95),
        ("大暴雨", 0.92),
        ("暴雨", 0.88),
        ("大雨", 0.78),
        ("中雨", 0.68),
        ("小雨", 0.55),
        ("雷阵雨", 0.72),
        ("阵雨", 0.58),
        ("雨夹雪", 0.52),
        ("雪", 0.46),
        ("阴", 0.28),
        ("多云", 0.18),
        ("晴", 0.08),
    ]
    for keyword, score in weather_scores:
        if keyword in weather_text:
            return score
    return 0.15


def build_amap_weather_label(dayweather, nightweather):
    """把白天/夜间天气转成自然标签"""
    if dayweather and nightweather and dayweather != nightweather:
        return f"{dayweather}转{nightweather}"
    return dayweather or nightweather or "未知"


def fetch_amap_weather_forecast(city_code=None):
    """获取高德未来天气预报，默认上海"""
    api_key = get_amap_weather_key()
    if not api_key:
        raise WeatherAPIError("未配置可用的高德天气 Key。")

    params = {
        "city": city_code or get_amap_city_code(),
        "extensions": "all",
        "output": "JSON",
        "key": api_key,
    }
    request_url = f"{AMAP_WEATHER_URL}?{urlencode(params)}"

    try:
        with urlopen(request_url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise WeatherAPIError("高德天气接口请求失败。") from exc

    if payload.get("status") != "1":
        raise WeatherAPIError(payload.get("info") or "高德天气接口返回失败。")

    forecasts = payload.get("forecasts") or []
    if not forecasts:
        raise WeatherAPIError("高德天气接口未返回预报数据。")

    forecast = forecasts[0]
    weather_by_date = {}
    for cast in forecast.get("casts", []):
        date_str = str(cast.get("date", "")).strip()
        if not date_str:
            continue

        daytemp = float(cast.get("daytemp") or cast.get("daytemp_float") or 0)
        nighttemp = float(cast.get("nighttemp") or cast.get("nighttemp_float") or 0)
        avg_temp = round((daytemp + nighttemp) / 2, 1)
        dayweather = str(cast.get("dayweather", "")).strip()
        nightweather = str(cast.get("nightweather", "")).strip()
        rain_probability = infer_rain_probability(dayweather, nightweather)

        weather_by_date[date_str] = {
            "temperature_c": avg_temp,
            "rain_probability": round(rain_probability, 2),
            "is_rainy": rain_probability >= 0.52,
            "weather_label": build_amap_weather_label(dayweather, nightweather),
            "comfort_label": "基于高德天气预报",
            "risk_level": "medium" if rain_probability >= 0.45 else "low",
            "weather_day": dayweather,
            "weather_night": nightweather,
            "reporttime": forecast.get("reporttime"),
            "city": forecast.get("city"),
            "province": forecast.get("province"),
            "adcode": forecast.get("adcode"),
            "source": "amap",
        }

    return {
        "city": forecast.get("city"),
        "adcode": forecast.get("adcode"),
        "reporttime": forecast.get("reporttime"),
        "weather_by_date": weather_by_date,
    }


def load_amap_weather_forecast(city_code=None):
    """安全加载高德天气，失败时返回空映射"""
    try:
        return fetch_amap_weather_forecast(city_code)
    except WeatherAPIError:
        return {"weather_by_date": {}}


def month_day_in_range(day_value, start_tuple, end_tuple):
    """判断日期是否位于某个跨年或非跨年窗口内"""
    current = (day_value.month, day_value.day)
    if start_tuple <= end_tuple:
        return start_tuple <= current <= end_tuple
    return current >= start_tuple or current <= end_tuple


def get_holiday_info(day_value):
    """返回节假日标记、名称和乘数"""
    row = historical_by_date.loc[day_value] if day_value in historical_by_date.index else None
    if row is not None and bool(row["is_holiday"]):
        raw_name = str(row["holiday_name"])
        name = HOLIDAY_NAME_MAP.get(raw_name, raw_name)
        multiplier = float(row.get("holiday_multiplier", 1.0))
        return True, name, multiplier

    for name, rule in HOLIDAY_RULES.items():
        start_month, start_day = rule["start"]
        start_date = date(day_value.year, start_month, start_day)
        end_date = start_date + timedelta(days=rule["duration"] - 1)
        if start_date <= day_value <= end_date:
            return True, name, float(rule["multiplier"])

    return False, None, 1.0


def get_special_event_name(day_value):
    """返回特殊活动名称"""
    for event in SPECIAL_EVENTS:
        if month_day_in_range(day_value, event["start"], event["end"]):
            return event["name"]
    return None


def is_school_break(day_value):
    """判断是否处于寒暑假或长假前后窗口"""
    row = historical_by_date.loc[day_value] if day_value in historical_by_date.index else None
    if row is not None:
        return bool(row["is_school_break"])
    return (
        day_value.month in [1, 2, 7, 8]
        or month_day_in_range(day_value, (5, 1), (5, 5))
        or month_day_in_range(day_value, (10, 1), (10, 7))
    )


def get_season_name(day_value):
    """根据月份返回季节名称"""
    if day_value.month in [3, 4, 5]:
        return "spring"
    if day_value.month in [6, 7, 8]:
        return "summer"
    if day_value.month in [9, 10, 11]:
        return "autumn"
    return "winter"


def estimate_weather(day_value, amap_weather_by_date=None):
    """优先使用高德天气，否则根据月份和日期构造稳定的天气估计"""
    date_str = day_value.strftime("%Y-%m-%d")
    if amap_weather_by_date and date_str in amap_weather_by_date:
        return dict(amap_weather_by_date[date_str])

    row = historical_by_date.loc[day_value] if day_value in historical_by_date.index else None
    month_stats = MONTHLY_CLIMATE.loc[day_value.month]
    seed = int(day_value.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)

    if row is not None:
        temperature = round(float(row["temperature"]), 1)
        base_rain_probability = float(month_stats["rain_probability"])
        rain_probability = float(
            np.clip(
                base_rain_probability + (0.22 if bool(row["is_rainy"]) else -0.10),
                0.08,
                0.85,
            )
        )
    else:
        temperature = round(float(month_stats["avg_temp"]) + float(rng.normal(0, 1.8)), 1)
        rain_probability = float(
            np.clip(float(month_stats["rain_probability"]) + float(rng.normal(0, 0.08)), 0.08, 0.85)
        )

    is_rainy = rain_probability >= 0.52

    if temperature >= 33 and rain_probability >= 0.45:
        weather_label = "高温伴阵雨"
        comfort_label = "体感压力高"
        risk_level = "high"
    elif temperature >= 32:
        weather_label = "闷热"
        comfort_label = "中午体感偏热"
        risk_level = "medium"
    elif rain_probability >= 0.55:
        weather_label = "阵雨频发"
        comfort_label = "需关注室内项目拥挤"
        risk_level = "medium"
    elif temperature <= 8:
        weather_label = "湿冷"
        comfort_label = "早晚体感偏冷"
        risk_level = "low"
    else:
        weather_label = "舒适"
        comfort_label = "整体游玩体感较好"
        risk_level = "low"

    return {
        "temperature_c": temperature,
        "rain_probability": round(rain_probability, 2),
        "is_rainy": bool(is_rainy),
        "weather_label": weather_label,
        "comfort_label": comfort_label,
        "risk_level": risk_level,
        "source": "estimated",
    }


def get_baseline_attendance(day_value):
    """按月份和星期给出历史基准值"""
    key = (day_value.month, day_value.weekday())
    baseline = MONTH_WEEKDAY_BASELINES.get(key)
    if baseline is None:
        baseline = MONTHLY_BASELINES.get(day_value.month, AVERAGE_ATTENDANCE)
    return float(baseline)


def get_lag_features(day_value, attendance_lookup):
    """从历史 + 已预测序列中读取滞后特征"""
    lag1 = attendance_lookup.get(day_value - timedelta(days=1), AVERAGE_ATTENDANCE)
    lag7 = attendance_lookup.get(day_value - timedelta(days=7), AVERAGE_ATTENDANCE)
    rolling_values = [
        float(attendance_lookup[day_value - timedelta(days=offset)])
        for offset in range(1, 31)
        if day_value - timedelta(days=offset) in attendance_lookup
    ]
    rolling_30 = mean(rolling_values) if rolling_values else AVERAGE_ATTENDANCE
    return float(lag1), float(lag7), float(rolling_30)


def get_crowd_profile(prediction):
    """根据历史分位数返回拥挤级别"""
    if prediction < ATTENDANCE_Q35:
        return {"label": "轻松", "key": "relaxed", "signal": "低压运营"}
    if prediction < ATTENDANCE_Q60:
        return {"label": "适中", "key": "steady", "signal": "常规运营"}
    if prediction < ATTENDANCE_Q82:
        return {"label": "繁忙", "key": "busy", "signal": "高峰排班"}
    return {"label": "高压", "key": "peak", "signal": "峰值预警"}


def estimate_park_hours(day_value, is_holiday, special_event_name, crowd_key):
    """根据时段给出运营时长建议值"""
    open_time = "08:30"
    close_time = "20:30"

    if day_value.month in [7, 8] or special_event_name:
        close_time = "21:30"
    if is_holiday:
        open_time = "08:00"
        close_time = "22:00"
    elif crowd_key == "peak":
        close_time = "21:45"
    elif crowd_key == "busy":
        close_time = "21:15"

    return f"{open_time}-{close_time}"


def estimate_show_count(is_holiday, special_event_name, crowd_key):
    """估算演出及巡游资源密度"""
    base_count = 5
    if special_event_name:
        base_count += 1
    if crowd_key in {"busy", "peak"}:
        base_count += 1
    if is_holiday:
        base_count += 1
    return base_count


def estimate_confidence(day_value, rain_probability):
    """简单估计预测置信度"""
    if day_value <= LAST_HISTORY_DATE:
        base_confidence = 0.90
    else:
        days_beyond_history = (day_value - LAST_HISTORY_DATE).days
        base_confidence = 0.88 - min(days_beyond_history, 365) * 0.0004

    if rain_probability >= 0.55:
        base_confidence -= 0.05

    return round(max(0.58, min(base_confidence, 0.92)), 2)


def build_day_note(day_result):
    """生成单日业务描述"""
    fragments = []
    if day_result["is_holiday"]:
        fragments.append(f"{day_result['holiday_name']}带动客流上扬")
    elif day_result["is_weekend"]:
        fragments.append("周末效应会放大头部项目排队")

    if day_result["special_event_name"]:
        fragments.append(f"{day_result['special_event_name']}提升下午和晚间客流黏性")

    if day_result["weather"]["rain_probability"] >= 0.55:
        fragments.append("雨天风险会把需求集中到室内项目")
    elif day_result["weather"]["temperature_c"] >= 32:
        fragments.append("高温会抬升午间补水和遮阴区压力")

    if day_result["demand_delta_pct"] >= 12:
        fragments.append("预计明显高于历史同类日期")
    elif day_result["demand_delta_pct"] <= -8:
        fragments.append("低于历史同类日期，适合做体验优化")

    return "；".join(fragments[:3]) if fragments else "预计需求平稳，适合按照常规节奏组织游玩与运营资源。"


def build_ops_focus(day_result):
    """生成单日运营动作"""
    if day_result["crowd_level_en"] == "peak":
        return "建议在安检、检票口与热门项目入口配置弹性人手，并前置餐饮补货。"
    if day_result["weather"]["rain_probability"] >= 0.55:
        return "建议提前准备雨具售卖与室内排队疏导方案，避免动线拥堵。"
    if day_result["weather"]["temperature_c"] >= 32:
        return "建议加密补水点巡检与遮阳休息区引导，缓解中午热应激。"
    if day_result["crowd_level_en"] == "busy":
        return "建议把高峰运力集中在开园后两小时和晚间演出前。"
    return "可按常规排班运行，并用低峰时段承接会员转化和二次消费。"


def build_visitor_tip(day_result):
    """生成单日游客建议"""
    if day_result["crowd_level_en"] == "peak":
        return "建议开园前 45 分钟到达，优先处理头部项目和热门预约。"
    if day_result["weather"]["rain_probability"] >= 0.55:
        return "建议带轻便雨衣，先排室外项目，再转入室内馆。"
    if day_result["weather"]["temperature_c"] >= 32:
        return "建议上午冲热门项目，中午转室内演出和餐饮，避免连续暴晒。"
    if day_result["crowd_level_en"] == "relaxed":
        return "适合用来补漏项目、拍照打卡和安排更从容的餐饮节奏。"
    return "建议开园后尽早完成前两项重点项目，下午按体力灵活调整。"


def predict_single_day(day_value, attendance_lookup=None, amap_weather_by_date=None):
    """预测单日客流并生成结构化建议"""
    if attendance_lookup is None:
        attendance_lookup = dict(HISTORICAL_ATTENDANCE_LOOKUP)

    row = historical_by_date.loc[day_value] if day_value in historical_by_date.index else None
    weather = estimate_weather(day_value, amap_weather_by_date=amap_weather_by_date)
    is_holiday, holiday_name, holiday_multiplier = get_holiday_info(day_value)
    school_break = is_school_break(day_value)
    special_event_name = get_special_event_name(day_value)
    has_special_event = special_event_name is not None
    weekday = day_value.weekday()
    day_of_year = day_value.timetuple().tm_yday
    is_weekend = weekday >= 5
    attendance_lag1, attendance_lag7, attendance_rolling_30 = get_lag_features(day_value, attendance_lookup)
    season_name = get_season_name(day_value)
    season_encoded = int(SEASON_ENCODING.get(season_name, 3))

    if row is not None:
        holiday_multiplier = float(row.get("holiday_multiplier", holiday_multiplier))

    features = {
        "year": day_value.year,
        "month": day_value.month,
        "day": day_value.day,
        "weekday": weekday,
        "month_sin": np.sin(2 * np.pi * day_value.month / 12),
        "month_cos": np.cos(2 * np.pi * day_value.month / 12),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365),
        "is_weekend": int(is_weekend),
        "is_holiday": int(is_holiday),
        "is_school_break": int(school_break),
        "is_rainy": int(weather["is_rainy"]),
        "has_special_event": int(has_special_event),
        "weekend_or_holiday": int(is_weekend or is_holiday),
        "summer_weekend": int(school_break and is_weekend),
        "holiday_multiplier": holiday_multiplier,
        "temperature": weather["temperature_c"],
        "attendance_lag1": attendance_lag1,
        "attendance_lag7": attendance_lag7,
        "attendance_rolling_30": attendance_rolling_30,
        "season_encoded": season_encoded,
    }

    feature_vector = np.array([[features[column] for column in feature_columns]])
    scaled_vector = scaler.transform(feature_vector)
    raw_prediction = float(model.predict(scaled_vector)[0])
    prediction = int(round(float(np.clip(raw_prediction, PREDICTION_MIN, PREDICTION_MAX))))
    attendance_lookup[day_value] = prediction

    crowd_profile = get_crowd_profile(prediction)
    baseline_attendance = get_baseline_attendance(day_value)
    demand_delta_pct = round((prediction - baseline_attendance) / baseline_attendance * 100, 1)
    park_hours = estimate_park_hours(day_value, is_holiday, special_event_name, crowd_profile["key"])
    show_count = estimate_show_count(is_holiday, special_event_name, crowd_profile["key"])
    confidence = estimate_confidence(day_value, weather["rain_probability"])
    day_result = {
        "date": day_value.strftime("%Y-%m-%d"),
        "weekday": WEEKDAY_NAMES[weekday],
        "predicted_attendance": prediction,
        "baseline_attendance": int(round(baseline_attendance)),
        "demand_delta_pct": demand_delta_pct,
        "crowd_level": crowd_profile["label"],
        "crowd_level_en": crowd_profile["key"],
        "operational_signal": crowd_profile["signal"],
        "is_holiday": bool(is_holiday),
        "holiday_name": holiday_name,
        "is_weekend": bool(is_weekend),
        "is_school_break": bool(school_break),
        "special_event_name": special_event_name,
        "park_hours": park_hours,
        "show_count": show_count,
        "confidence": confidence,
        "weather": weather,
    }
    day_result["business_note"] = build_day_note(day_result)
    day_result["ops_focus"] = build_ops_focus(day_result)
    day_result["visitor_tip"] = build_visitor_tip(day_result)
    return day_result


def build_key_drivers(day_results, baseline_delta_pct):
    """区间关键驱动因素"""
    holiday_days = sum(1 for item in day_results if item["is_holiday"])
    weekend_days = sum(1 for item in day_results if item["is_weekend"])
    school_break_days = sum(1 for item in day_results if item["is_school_break"])
    special_event_days = sum(1 for item in day_results if item["special_event_name"])
    avg_temp = mean(item["weather"]["temperature_c"] for item in day_results)
    avg_rain_probability = mean(item["weather"]["rain_probability"] for item in day_results)

    drivers = []
    if holiday_days:
        drivers.append(f"{holiday_days} 天法定节假日将显著抬高入园需求和晚间停留时长。")
    if weekend_days >= max(1, len(day_results) // 2):
        drivers.append("周末占比较高，头部项目与餐饮高峰将更集中。")
    if school_break_days:
        drivers.append("学校假期窗口会增强家庭客群占比，午后与傍晚客流更具黏性。")
    if special_event_days:
        drivers.append("季节性主题活动会提升晚场表演、园区零售与餐饮的需求密度。")
    if avg_temp >= 31:
        drivers.append("高温天气会放大遮阴、补水和室内项目承接压力。")
    if avg_rain_probability >= 0.48:
        drivers.append("阵雨风险可能抑制部分临时客流，但会让室内项目排队更集中。")
    if baseline_delta_pct >= 10:
        drivers.append("整体需求高于历史同类日期，需按高峰运营节奏组织资源。")
    elif baseline_delta_pct <= -8:
        drivers.append("整体需求低于历史同类日期，适合利用低峰窗口做体验优化与转化。")

    return drivers[:4] if drivers else ["整体需求结构较平稳，主要受周内节奏与常规天气因素驱动。"]


def build_operational_recommendations(day_results, overview):
    """区间运营建议"""
    recommendations = []
    avg_temp = mean(item["weather"]["temperature_c"] for item in day_results)
    avg_rain_probability = mean(item["weather"]["rain_probability"] for item in day_results)

    if overview["peak_days"] > 0:
        recommendations.append("在高压日的 09:00-11:30 与 16:30-19:30 加强安检、检票口和热门项目入口排班。")
    if overview["busy_days"] >= 2:
        recommendations.append("提前锁定餐饮补货节奏，把高需求资源向头部项目周边与晚场核心动线倾斜。")
    if avg_rain_probability >= 0.48:
        recommendations.append("准备雨具售卖、室内项目导流和巡游变更预案，降低天气导致的局部拥堵。")
    if avg_temp >= 31:
        recommendations.append("增加补水点巡检与遮阴区引导，必要时强化中午时段的休息区服务。")
    if overview["baseline_delta_pct"] <= -8:
        recommendations.append("可利用相对低峰窗口推进会员活动、套餐转化和园区体验细节优化。")

    if not recommendations:
        recommendations.append("按常规周内运营节奏配置资源即可，重点关注午后餐饮与晚间演出前后的客流波动。")

    return recommendations[:4]


def build_visitor_recommendations(day_results, overview):
    """区间游客建议"""
    recommendations = []
    avg_temp = mean(item["weather"]["temperature_c"] for item in day_results)
    avg_rain_probability = mean(item["weather"]["rain_probability"] for item in day_results)

    if overview["peak_days"] > 0 or overview["busy_days"] >= 2:
        recommendations.append("建议至少提前 30-45 分钟抵达，优先完成最在意的头部项目。")
    if avg_rain_probability >= 0.48:
        recommendations.append("建议携带轻便雨衣，先安排室外项目，再把室内项目放到午后或阵雨时段。")
    if avg_temp >= 31:
        recommendations.append("建议上午冲热门项目，中午转向室内演出、餐饮和休息区，避免高温暴晒。")
    if overview["best_visit_days"]:
        best_days = "、".join(day["date"] for day in overview["best_visit_days"])
        recommendations.append(f"若行程可调，{best_days} 更适合安排拍照、补漏项目和轻松节奏。")

    if not recommendations:
        recommendations.append("整体适合按常规节奏游玩，建议开园后优先完成前两项重点项目。")

    return recommendations[:4]


def build_summary_payload(overview, day_results, drivers, operational_recommendations, visitor_recommendations):
    """给 AI 使用的结构化摘要数据"""
    peak_day = overview["peak_day"]
    calm_day = overview["calm_day"]
    best_days = [item["date"] for item in overview["best_visit_days"]]
    avg_temp = int(round(mean(item["weather"]["temperature_c"] for item in day_results)))
    avg_rain_probability = round(mean(item["weather"]["rain_probability"] for item in day_results), 2)

    return {
        "project_theme": "面向上海迪士尼乐园运营的客流预测与商业洞察分析",
        "report_scope": {
            "start_date": overview["start_date"],
            "end_date": overview["end_date"],
            "days": overview["day_count"],
        },
        "forecast": {
            "average_attendance": overview["average_attendance"],
            "total_attendance": overview["total_attendance"],
            "range_signal": overview["range_signal"],
            "busy_days": overview["busy_days"],
            "peak_days": overview["peak_days"],
            "baseline_delta_pct": overview["baseline_delta_pct"],
            "average_confidence": overview["average_confidence"],
        },
        "peak_day": {
            "date": peak_day["date"],
            "date_label": to_cn_date_label(peak_day["date"]),
            "predicted_attendance": peak_day["predicted_attendance"],
            "crowd_level": peak_day["crowd_level"],
            "weather_label": peak_day["weather"]["weather_label"],
        },
        "calm_day": {
            "date": calm_day["date"],
            "date_label": to_cn_date_label(calm_day["date"]),
            "predicted_attendance": calm_day["predicted_attendance"],
            "crowd_level": calm_day["crowd_level"],
        },
        "calendar_context": {
            "holiday_days": overview["holiday_days"],
            "weekend_days": overview["weekend_days"],
            "school_break_days": overview["school_break_days"],
            "special_event_days": overview["special_event_days"],
        },
        "weather_context": {
            "average_temperature_c": avg_temp,
            "average_rain_probability_pct": to_percent_label(avg_rain_probability),
            "average_rain_risk_label": get_rain_risk_label(avg_rain_probability),
            "high_rain_risk_days": overview["rainy_risk_days"],
        },
        "best_visit_days": [
            {"date": day_date, "date_label": to_cn_date_label(day_date)}
            for day_date in best_days
        ],
        "drivers": drivers,
        "recommendations": {
            "operations": operational_recommendations,
            "visitors": visitor_recommendations,
        },
        "daily_context": [
            {
                "date": item["date"],
                "date_label": to_cn_date_label(item["date"]),
                "weekday": item["weekday"],
                "predicted_attendance": item["predicted_attendance"],
                "baseline_attendance": item["baseline_attendance"],
                "demand_delta_pct": item["demand_delta_pct"],
                "crowd_level": item["crowd_level"],
                "is_holiday": item["is_holiday"],
                "holiday_name": item["holiday_name"],
                "is_school_break": item["is_school_break"],
                "special_event_name": item["special_event_name"],
                "park_hours": item["park_hours"],
                "park_hours_label": get_park_hours_label(item["park_hours"]),
                "weather_label": item["weather"]["weather_label"],
                "temperature_c": int(round(item["weather"]["temperature_c"])),
                "rain_probability_pct": to_percent_label(item["weather"]["rain_probability"]),
                "rain_risk_label": get_rain_risk_label(item["weather"]["rain_probability"]),
            }
            for item in day_results
        ],
    }


def extract_json_object(text):
    """从模型输出中提取 JSON 对象"""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise AIInsightError("AI 返回内容不是有效的 JSON。")
    return cleaned[start : end + 1]


CHINESE_NUMERAL_CHARS = "零〇一二两三四五六七八九十百千万亿"
CHINESE_DIGIT_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_UNIT_MAP = {
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
    "亿": 100000000,
}
DATE_NUMBER_PATTERN = re.compile(rf"([{CHINESE_NUMERAL_CHARS}]+)月([{CHINESE_NUMERAL_CHARS}]+)日")
RANGE_NUMBER_PATTERN = re.compile(rf"([{CHINESE_NUMERAL_CHARS}]+)(?=(?:至|到|-|~|～|/))")
LABOR_DAY_SHORTHAND_PATTERN = re.compile(r"五一(?=(?:假期|劳动节))")
TRAILING_LARGE_NUMBER_PATTERN = re.compile(rf"([{CHINESE_NUMERAL_CHARS}]*[十百千万亿][{CHINESE_NUMERAL_CHARS}]*)(?=(?:，|。|；|,|\s|$))")
ARABIC_COMPACT_NUMBER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(?:\s*)(万|亿)")
DECIMAL_PROBABILITY_PATTERN = re.compile(r"(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?=(?:\s*)(?:降雨概率|雨概率|降水概率|概率))")
UNIT_NUMBER_PATTERN = re.compile(
    rf"([{CHINESE_NUMERAL_CHARS}]+)(?=(?:人次|人|个|天|日|月|年|分钟|小时|项|条|倍|次|℃|度|点|分|秒|场|家|位|桌|米|公里|元|%))"
)


def chinese_numeral_to_int(text):
    """把常见中文数字转成整数"""
    normalized = (text or "").strip()
    if not normalized:
        raise ValueError("中文数字不能为空")

    if all(char in CHINESE_DIGIT_MAP for char in normalized):
        return int("".join(str(CHINESE_DIGIT_MAP[char]) for char in normalized))

    total = 0
    section = 0
    number = 0

    for char in normalized:
        if char in CHINESE_DIGIT_MAP:
            number = CHINESE_DIGIT_MAP[char]
            continue

        unit = CHINESE_UNIT_MAP.get(char)
        if unit is None:
            raise ValueError(f"无法识别的中文数字字符: {char}")

        if unit >= 10000:
            section = (section + number) * unit
            total += section
            section = 0
        else:
            if number == 0:
                number = 1
            section += number * unit
        number = 0

    return total + section + number


def replace_cn_number(match):
    """把匹配到的中文数字替换成阿拉伯数字"""
    return str(chinese_numeral_to_int(match.group(1)))


def replace_arabic_compact_number(match):
    """把 48万 / 1.2亿 这种写法展开成完整阿拉伯数字"""
    number = float(match.group(1))
    unit = match.group(2)
    multiplier = 10000 if unit == "万" else 100000000
    return str(int(round(number * multiplier)))


def replace_decimal_probability(match):
    """把 0.48 这类概率写法转成 48%"""
    value = float(match.group(1))
    if 0 <= value <= 1:
        return f"{int(round(value * 100))}%"
    return match.group(1)


def normalize_ai_numbers_in_text(text):
    """把 AI 输出中的中文数字尽量统一成阿拉伯数字"""
    normalized = (text or "").strip()
    if not normalized:
        return normalized

    normalized = ARABIC_COMPACT_NUMBER_PATTERN.sub(replace_arabic_compact_number, normalized)
    normalized = DECIMAL_PROBABILITY_PATTERN.sub(replace_decimal_probability, normalized)
    normalized = LABOR_DAY_SHORTHAND_PATTERN.sub("5月1日", normalized)
    normalized = DATE_NUMBER_PATTERN.sub(
        lambda match: f"{chinese_numeral_to_int(match.group(1))}月{chinese_numeral_to_int(match.group(2))}日",
        normalized,
    )
    normalized = RANGE_NUMBER_PATTERN.sub(replace_cn_number, normalized)
    normalized = UNIT_NUMBER_PATTERN.sub(replace_cn_number, normalized)
    normalized = TRAILING_LARGE_NUMBER_PATTERN.sub(replace_cn_number, normalized)
    return normalized


def coerce_text_list(value, field_name, minimum=1, maximum=4):
    """把 AI 输出统一为字符串列表"""
    if isinstance(value, str):
        items = [normalize_ai_numbers_in_text(value)]
    elif isinstance(value, list):
        items = [
            normalize_ai_numbers_in_text(str(item))
            for item in value
            if str(item).strip()
        ]
    else:
        items = []

    items = items[:maximum]
    if len(items) < minimum:
        raise AIInsightError(f"AI 返回的 {field_name} 数量不足。")
    return items


def coerce_summary_text(value):
    """把 AI 摘要统一为段落文本"""
    if isinstance(value, list):
        parts = [
            normalize_ai_numbers_in_text(str(item))
            for item in value
            if str(item).strip()
        ]
        if not parts:
            raise AIInsightError("AI 未返回有效摘要。")
        return "\n\n".join(parts[:4])

    if isinstance(value, str) and value.strip():
        return normalize_ai_numbers_in_text(value)

    raise AIInsightError("AI 未返回有效摘要。")


def merge_ai_daily_advice(day_results, daily_advice):
    """把 AI 生成的逐日建议回填到预测结果"""
    if not isinstance(daily_advice, list):
        raise AIInsightError("AI 未返回逐日建议列表。")

    advice_by_date = {}
    fallback_items = []
    for item in daily_advice:
        if not isinstance(item, dict):
            continue
        day_date = str(item.get("date", "")).strip()
        day_label = str(item.get("date_label", "")).strip()

        if day_date:
            advice_by_date[day_date] = item
        elif day_label:
            advice_by_date[day_label] = item

        fallback_items.append(item)

    for index, day_result in enumerate(day_results):
        ai_day = advice_by_date.get(day_result["date"])
        if not ai_day:
            ai_day = advice_by_date.get(to_cn_date_label(day_result["date"]))
        if not ai_day and index < len(fallback_items):
            ai_day = fallback_items[index]
        if not ai_day:
            raise AIInsightError(f"AI 未返回 {day_result['date']} 的逐日建议。")

        business_note = normalize_ai_numbers_in_text(str(ai_day.get("business_note", "")))
        ops_focus = normalize_ai_numbers_in_text(str(ai_day.get("ops_focus", "")))
        visitor_tip = normalize_ai_numbers_in_text(str(ai_day.get("visitor_tip", "")))

        if not business_note or not ops_focus or not visitor_tip:
            raise AIInsightError(f"AI 返回的 {day_result['date']} 逐日建议不完整。")

        day_result["business_note"] = business_note
        day_result["ops_focus"] = ops_focus
        day_result["visitor_tip"] = visitor_tip

    return day_results


def generate_ai_insights(summary_payload):
    """调用 AI 生成摘要、区间建议与逐日建议"""
    token = get_modelscope_access_token()
    if not token:
        raise AIInsightError("未配置可用的 ModelScope Token。")

    try:
        from openai import APITimeoutError, OpenAI
    except ImportError as exc:
        raise AIInsightError("当前环境缺少 openai 依赖。") from exc

    client = OpenAI(
        api_key=token,
        base_url=os.getenv("MODELSCOPE_BASE_URL", DEFAULT_MODELSCOPE_BASE_URL),
        timeout=80.0,
    )
    payload_text = json.dumps(summary_payload, ensure_ascii=False)

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名主题乐园商业分析顾问。"
                "请严格基于输入事实，用中文输出。"
                "不要虚构数据，不要补充输入里没有的日期、人数或活动。"
                "只返回 JSON 对象，不要使用 Markdown，不要输出代码块。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请根据以下 JSON，为上海迪士尼运营管理者生成结构化结果。"
                '返回 JSON，字段必须严格为："summary"、"drivers"、"operations"、"visitors"、"daily_advice"。'
                '"summary" 是 4 条数组，每条 40-70 字；'
                '"drivers"、"operations"、"visitors" 各返回 3 条，每条不超过 32 字；'
                '"daily_advice" 必须覆盖输入里的每一个 date，每项包含 "date"、"business_note"、"ops_focus"、"visitor_tip"，'
                '三个文本字段都不超过 28 字。'
                "如需引用日期，请优先使用输入中的 date_label 原样表达。"
                "所有数字必须使用阿拉伯数字，不要使用中文数字，例如 318937 人次、30-45 分钟、5月1日、22点。"
                "优先使用自然的 business 表达，例如“开园前后”“午后”“晚间闭园前”“降雨风险较高”。"
                "不要直接复述字段名，不要写 0.48 降雨概率；如需表达概率，请写成 48%。"
                "除非特别必要，避免机械地写 08:00 开园、0.48 降雨概率 这类机器式表达。"
                "所有文本都要简洁、可执行、适合 business 展示。\n\n"
                f"{payload_text}"
            ),
        },
    ]

    model_candidates = get_modelscope_model_candidates()
    parsed = None
    last_error = None
    for model_name in model_candidates:
        content = None
        for _ in range(2):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.15,
                    max_tokens=950,
                    messages=messages,
                )
            except APITimeoutError as exc:
                last_error = AIInsightError(f"{model_name} 生成建议超时，请稍后重试。")
                break
            except Exception as exc:
                last_error = AIInsightError(f"{model_name} 调用失败，请稍后重试。")
                break

            choices = getattr(response, "choices", None)
            if not choices:
                continue

            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None) if message else None
            if content:
                break

        if not content:
            last_error = AIInsightError(f"{model_name} 未返回有效内容。")
            continue

        try:
            parsed = json.loads(extract_json_object(content))
            break
        except (json.JSONDecodeError, AIInsightError) as exc:
            last_error = AIInsightError(f"{model_name} 返回内容无法解析。")
            continue

    if parsed is None:
        if len(model_candidates) > 1:
            raise AIInsightError("AI 未返回有效内容，已自动切换备用模型，请稍后重试。") from last_error
        raise AIInsightError("AI 未返回有效内容，请稍后重试。") from last_error

    return {
        "summary_text": coerce_summary_text(parsed.get("summary")),
        "key_drivers": coerce_text_list(parsed.get("drivers"), "关键驱动", minimum=3, maximum=4),
        "operational_recommendations": coerce_text_list(parsed.get("operations"), "运营建议", minimum=3, maximum=4),
        "visitor_recommendations": coerce_text_list(parsed.get("visitors"), "游客建议", minimum=3, maximum=4),
        "daily_advice": parsed.get("daily_advice"),
    }


def generate_rule_summary(overview, drivers, operational_recommendations):
    """无 AI 时的规则摘要"""
    peak_day = overview["peak_day"]
    best_days = "、".join(item["date"] for item in overview["best_visit_days"]) or "暂无明显低压日"
    driver_text = "；".join(drivers[:2])
    ops_text = "；".join(operational_recommendations[:2])

    return (
        f"在 {overview['start_date']} 至 {overview['end_date']} 的 {overview['day_count']} 天窗口中，"
        f"预计日均客流约 {overview['average_attendance']:,} 人，整体处于“{overview['range_signal']}”区间。"
        f"峰值出现在 {peak_day['date']}，预计约 {peak_day['predicted_attendance']:,} 人。\n\n"
        f"主要驱动因素包括：{driver_text}\n\n"
        f"若按当前窗口组织运营，建议重点关注 {overview['busy_days']} 个繁忙日与 "
        f"{overview['rainy_risk_days']} 个天气风险日，对开园早高峰和晚间演出前后做弹性调度。\n\n"
        f"行动建议：{ops_text}。对游客而言，优先考虑 {best_days} 作为更轻松的游玩窗口。"
    )


def analyze_date_range(start_date_str, end_date_str, enable_ai_summary=True):
    """主分析逻辑"""
    start_day = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_day = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    if end_day < start_day:
        raise ValueError("结束日期不能早于开始日期")

    day_count = (end_day - start_day).days + 1
    if day_count > 31:
        raise ValueError("单次分析最多支持 31 天，请缩短日期区间")

    attendance_lookup = dict(HISTORICAL_ATTENDANCE_LOOKUP)
    amap_weather_bundle = load_amap_weather_forecast()
    amap_weather_by_date = amap_weather_bundle.get("weather_by_date", {})
    day_results = []
    for offset in range(day_count):
        current_day = start_day + timedelta(days=offset)
        day_results.append(
            predict_single_day(
                current_day,
                attendance_lookup,
                amap_weather_by_date=amap_weather_by_date,
            )
        )

    total_attendance = int(sum(item["predicted_attendance"] for item in day_results))
    average_attendance = int(round(total_attendance / day_count))
    average_confidence = round(mean(item["confidence"] for item in day_results), 2)
    baseline_average = mean(item["baseline_attendance"] for item in day_results)
    baseline_delta_pct = round((average_attendance - baseline_average) / baseline_average * 100, 1)
    holiday_days = sum(1 for item in day_results if item["is_holiday"])
    weekend_days = sum(1 for item in day_results if item["is_weekend"])
    school_break_days = sum(1 for item in day_results if item["is_school_break"])
    special_event_days = sum(1 for item in day_results if item["special_event_name"])
    busy_days = sum(1 for item in day_results if item["crowd_level_en"] == "busy")
    peak_days = sum(1 for item in day_results if item["crowd_level_en"] == "peak")
    rainy_risk_days = sum(1 for item in day_results if item["weather"]["rain_probability"] >= 0.48)
    peak_day = max(day_results, key=lambda item: item["predicted_attendance"])
    calm_day = min(day_results, key=lambda item: item["predicted_attendance"])
    best_visit_days = sorted(
        day_results,
        key=lambda item: (
            item["predicted_attendance"],
            item["weather"]["rain_probability"],
            item["is_holiday"],
        ),
    )[:2]

    if peak_days > 0 or busy_days >= max(2, day_count // 3):
        range_signal = "需要精准计划"
    elif average_attendance <= ATTENDANCE_Q35:
        range_signal = "轻松窗口"
    else:
        range_signal = "稳态窗口"

    overview = {
        "start_date": start_date_str,
        "end_date": end_date_str,
        "day_count": day_count,
        "average_attendance": average_attendance,
        "total_attendance": total_attendance,
        "average_confidence": average_confidence,
        "baseline_delta_pct": baseline_delta_pct,
        "holiday_days": holiday_days,
        "weekend_days": weekend_days,
        "school_break_days": school_break_days,
        "special_event_days": special_event_days,
        "busy_days": busy_days,
        "peak_days": peak_days,
        "rainy_risk_days": rainy_risk_days,
        "peak_day": peak_day,
        "calm_day": calm_day,
        "best_visit_days": best_visit_days,
        "range_signal": range_signal,
    }
    key_drivers = build_key_drivers(day_results, baseline_delta_pct)
    operational_recommendations = build_operational_recommendations(day_results, overview)
    visitor_recommendations = build_visitor_recommendations(day_results, overview)
    summary_payload = build_summary_payload(
        overview,
        day_results,
        key_drivers,
        operational_recommendations,
        visitor_recommendations,
    )

    summary_source = "rule"
    summary_text = generate_rule_summary(overview, key_drivers, operational_recommendations)

    if enable_ai_summary:
        ai_insights = generate_ai_insights(summary_payload)
        day_results = merge_ai_daily_advice(day_results, ai_insights["daily_advice"])
        key_drivers = ai_insights["key_drivers"]
        operational_recommendations = ai_insights["operational_recommendations"]
        visitor_recommendations = ai_insights["visitor_recommendations"]
        summary_text = ai_insights["summary_text"]
        summary_source = "ai"

    return {
        "query": {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "day_count": day_count,
        },
        "weather_meta": {
            "city": amap_weather_bundle.get("city", "上海"),
            "adcode": amap_weather_bundle.get("adcode", get_amap_city_code()),
            "reporttime": amap_weather_bundle.get("reporttime"),
        },
        "overview": overview,
        "key_drivers": key_drivers,
        "operational_recommendations": operational_recommendations,
        "visitor_recommendations": visitor_recommendations,
        "summary": {
            "text": summary_text,
            "source": summary_source,
        },
        "summary_payload": summary_payload,
        "daily_predictions": day_results,
    }


@app.route("/")
def index():
    """主页"""
    return render_template("index.html")


@app.route("/analyze_range", methods=["POST"])
def analyze_range():
    """日期区间分析接口"""
    try:
        data = request.get_json() or {}
        start_date = data.get("start_date")
        end_date = data.get("end_date")

        if not start_date or not end_date:
            return jsonify({"success": False, "error": "请同时提供开始日期和结束日期"}), 400

        result = analyze_date_range(start_date, end_date, enable_ai_summary=data.get("enable_ai_summary", True))
        return jsonify({"success": True, "data": result})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except AIInsightError as exc:
        return jsonify({"success": False, "error": str(exc)}), 502
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """兼容旧版单日预测接口"""
    try:
        data = request.get_json() or {}
        date_str = data.get("date")
        if not date_str:
            return jsonify({"success": False, "error": "请提供日期"}), 400

        day_value = datetime.strptime(date_str, "%Y-%m-%d").date()
        amap_weather_bundle = load_amap_weather_forecast()
        result = predict_single_day(
            day_value,
            amap_weather_by_date=amap_weather_bundle.get("weather_by_date", {}),
        )
        return jsonify({"success": True, "data": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/predict_week", methods=["POST"])
def predict_week():
    """兼容旧版一周预测接口"""
    try:
        data = request.get_json() or {}
        start_date_str = data.get("start_date")
        if not start_date_str:
            return jsonify({"success": False, "error": "请提供开始日期"}), 400

        start_day = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_day = start_day + timedelta(days=6)
        result = analyze_date_range(
            start_day.strftime("%Y-%m-%d"),
            end_day.strftime("%Y-%m-%d"),
            enable_ai_summary=False,
        )
        return jsonify({"success": True, "data": result["daily_predictions"]})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/holidays")
def get_holidays():
    """获取节假日规则"""
    return jsonify({"success": True, "data": HOLIDAY_RULES})


if __name__ == "__main__":
    print("上海迪士尼运营洞察系统启动...")
    print("访问地址: http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
