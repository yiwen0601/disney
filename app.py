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
DEFAULT_MODELSCOPE_ACCESS_TOKEN = "sk-5923f50b586b49b9b08335dd9ee9fa52"
DEFAULT_MODELSCOPE_BASE_URL = "https://matrixllm.alipay.com/v1"
DEFAULT_MODELSCOPE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MODELSCOPE_FALLBACK_MODELS = (
    "Qwen/Qwen3.5-35B-A3B",
    "ZhipuAI/GLM-5.1",
    "MiniMax/MiniMax-M2.7",
)
DEFAULT_AI_MODEL_TIMEOUT_SECONDS = float(os.getenv("MODELSCOPE_MODEL_TIMEOUT_SECONDS", "10"))
DEFAULT_AI_MODEL_MAX_ATTEMPTS = int(os.getenv("MODELSCOPE_MODEL_MAX_ATTEMPTS", "1"))
DEFAULT_AI_OVERVIEW_MAX_TOKENS = int(os.getenv("MODELSCOPE_OVERVIEW_MAX_TOKENS", "1200"))
DEFAULT_AI_DAILY_MAX_TOKENS = int(os.getenv("MODELSCOPE_DAILY_MAX_TOKENS", "900"))
DEFAULT_AI_DAILY_CHUNK_SIZE = int(os.getenv("MODELSCOPE_DAILY_CHUNK_SIZE", "7"))
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

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOLIDAY_NAME_MAP = {
    "new_year": "New Year's Day",
    "spring_festival": "Spring Festival",
    "qingming": "Qingming Festival",
    "labor_day": "Labor Day Holiday",
    "dragon_boat": "Dragon Boat Festival",
    "mid_autumn": "Mid-Autumn Festival",
    "national_day": "National Day Golden Week",
}
HOLIDAY_RULES = {
    "New Year's Day": {"start": (1, 1), "duration": 3, "multiplier": 1.30},
    "Spring Festival": {"start": (2, 10), "duration": 7, "multiplier": 1.85},
    "Qingming Festival": {"start": (4, 4), "duration": 3, "multiplier": 1.35},
    "Labor Day Holiday": {"start": (5, 1), "duration": 5, "multiplier": 1.65},
    "Dragon Boat Festival": {"start": (6, 10), "duration": 3, "multiplier": 1.30},
    "Mid-Autumn Festival": {"start": (9, 15), "duration": 3, "multiplier": 1.32},
    "National Day Golden Week": {"start": (10, 1), "duration": 7, "multiplier": 1.95},
}
SPECIAL_EVENTS = [
    {"name": "Winter Celebration Season", "start": (12, 15), "end": (1, 7)},
    {"name": "Spring Festival Season", "start": (3, 15), "end": (5, 5)},
    {"name": "Summer Celebration Season", "start": (6, 20), "end": (8, 31)},
    {"name": "Halloween Season", "start": (10, 1), "end": (10, 31)},
]
AMAP_WEATHER_TERM_MAP = {
    "特大暴雨": "extreme rainstorm",
    "大暴雨": "severe rainstorm",
    "暴雨": "rainstorm",
    "大雨": "heavy rain",
    "中雨": "moderate rain",
    "小雨": "light rain",
    "雷阵雨": "thunderstorms",
    "阵雨": "showers",
    "雨夹雪": "sleet",
    "多云": "cloudy",
    "阴": "overcast",
    "晴": "sunny",
    "雪": "snow",
}


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


def to_display_date_label(date_str):
    """Convert an ISO date into an English date label."""
    parsed = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{MONTH_ABBR[parsed.month - 1]} {parsed.day}"


def to_percent_label(value):
    """把 0-1 概率转成百分比整数"""
    return int(round(float(value) * 100))


def get_rain_risk_label(probability):
    """把降雨概率映射成更自然的风险描述"""
    if probability >= 0.6:
        return "High rain risk"
    if probability >= 0.45:
        return "Elevated rain risk"
    if probability >= 0.3:
        return "Some rain risk"
    return "Low rain risk"


def get_park_hours_label(park_hours):
    """把营业时段转成更适合自然语言生成的标签"""
    open_time, close_time = park_hours.split("-")
    if open_time <= "08:00" and close_time >= "22:00":
        return "Early opening and extended evening hours"
    if close_time >= "21:30":
        return "Extended evening hours"
    return "Standard operating hours"


def translate_weather_text(text):
    """Translate Amap weather descriptions into English for display."""
    translated = (text or "").strip()
    if not translated:
        return ""

    for chinese, english in AMAP_WEATHER_TERM_MAP.items():
        translated = translated.replace(chinese, english)
    translated = translated.replace("转", " to ")
    return translated or "Unknown"


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
    day_label = translate_weather_text(dayweather)
    night_label = translate_weather_text(nightweather)
    if day_label and night_label and day_label != night_label:
        return f"{day_label} to {night_label}"
    return day_label or night_label or "Unknown"


def fetch_amap_weather_forecast(city_code=None):
    """获取高德未来天气预报，默认上海"""
    api_key = get_amap_weather_key()
    if not api_key:
        raise WeatherAPIError("No valid Amap weather API key is configured.")

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
        raise WeatherAPIError("The Amap weather request failed.") from exc

    if payload.get("status") != "1":
        raise WeatherAPIError("The Amap weather service returned an error.")

    forecasts = payload.get("forecasts") or []
    if not forecasts:
        raise WeatherAPIError("The Amap weather service returned no forecast data.")

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
            "comfort_label": "Based on the latest Amap forecast",
            "risk_level": "medium" if rain_probability >= 0.45 else "low",
            "weather_day": translate_weather_text(dayweather),
            "weather_night": translate_weather_text(nightweather),
            "reporttime": forecast.get("reporttime"),
            "city": "Shanghai",
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
        weather_label = "Hot with scattered showers"
        comfort_label = "Higher thermal stress"
        risk_level = "high"
    elif temperature >= 32:
        weather_label = "Hot and humid"
        comfort_label = "Warmer conditions around midday"
        risk_level = "medium"
    elif rain_probability >= 0.55:
        weather_label = "Frequent showers"
        comfort_label = "Indoor attractions may become more crowded"
        risk_level = "medium"
    elif temperature <= 8:
        weather_label = "Cold and damp"
        comfort_label = "Cooler mornings and evenings"
        risk_level = "low"
    else:
        weather_label = "Comfortable"
        comfort_label = "Generally pleasant park conditions"
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
        return {"label": "Relaxed", "key": "relaxed", "signal": "Low-pressure operations"}
    if prediction < ATTENDANCE_Q60:
        return {"label": "Moderate", "key": "steady", "signal": "Standard operations"}
    if prediction < ATTENDANCE_Q82:
        return {"label": "Busy", "key": "busy", "signal": "Peak staffing"}
    return {"label": "Peak", "key": "peak", "signal": "Peak alert"}


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
        fragments.append(f"{day_result['holiday_name']} is likely to lift demand")
    elif day_result["is_weekend"]:
        fragments.append("Weekend traffic is likely to intensify queues at headline attractions")

    if day_result["special_event_name"]:
        fragments.append(f"{day_result['special_event_name']} should support stronger afternoon and evening traffic")

    if day_result["weather"]["rain_probability"] >= 0.55:
        fragments.append("Rain risk may shift demand toward indoor attractions")
    elif day_result["weather"]["temperature_c"] >= 32:
        fragments.append("Hot weather is likely to raise pressure on hydration and shaded rest areas")

    if day_result["demand_delta_pct"] >= 12:
        fragments.append("Expected demand is materially above the historical baseline")
    elif day_result["demand_delta_pct"] <= -8:
        fragments.append("Demand is below comparable historical days, creating room for experience optimization")

    return "; ".join(fragments[:3]) if fragments else "Demand looks stable and supports a standard operations rhythm."


def build_ops_focus(day_result):
    """生成单日运营动作"""
    if day_result["crowd_level_en"] == "peak":
        return "Add flexible staffing at security, ticket entry, and top attraction entrances, and pre-position food replenishment."
    if day_result["weather"]["rain_probability"] >= 0.55:
        return "Prepare rain gear sales and indoor queue guidance in advance to reduce local bottlenecks."
    if day_result["weather"]["temperature_c"] >= 32:
        return "Increase checks at hydration points and guide guests toward shaded rest areas around midday."
    if day_result["crowd_level_en"] == "busy":
        return "Concentrate peak operating capacity in the first two hours after opening and before evening shows."
    return "A standard staffing plan should be sufficient, with quieter hours used for membership conversion and secondary spend."


def build_visitor_tip(day_result):
    """生成单日游客建议"""
    if day_result["crowd_level_en"] == "peak":
        return "Arrive about 45 minutes before opening and prioritize headline attractions and reservations."
    if day_result["weather"]["rain_probability"] >= 0.55:
        return "Bring a lightweight rain poncho, do outdoor rides first, and shift indoors later."
    if day_result["weather"]["temperature_c"] >= 32:
        return "Target major rides in the morning, then move to indoor shows and dining around midday."
    if day_result["crowd_level_en"] == "relaxed":
        return "This is a good day for lower-priority rides, photo stops, and a more relaxed dining pace."
    return "Finish your top two priorities early, then adjust the afternoon plan based on energy levels."


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
        drivers.append(f"{holiday_days} public holiday days are likely to raise park entry demand and evening dwell time.")
    if weekend_days >= max(1, len(day_results) // 2):
        drivers.append("A high weekend share should concentrate demand around headline attractions and dining peaks.")
    if school_break_days:
        drivers.append("School break timing is likely to increase the family segment and keep traffic elevated into the evening.")
    if special_event_days:
        drivers.append("Seasonal events should support stronger demand for evening shows, retail, and food and beverage.")
    if avg_temp >= 31:
        drivers.append("Higher temperatures are likely to increase pressure on shaded zones, hydration points, and indoor capacity.")
    if avg_rain_probability >= 0.48:
        drivers.append("Shower risk may soften some spontaneous visits while concentrating queues inside indoor attractions.")
    if baseline_delta_pct >= 10:
        drivers.append("Overall demand is above comparable historical dates and should be managed with a peak-style operating rhythm.")
    elif baseline_delta_pct <= -8:
        drivers.append("Overall demand is below comparable historical dates, creating room for experience upgrades and conversion activity.")

    return drivers[:4] if drivers else ["Demand looks relatively stable, driven mainly by the weekday mix and normal weather conditions."]


def build_operational_recommendations(day_results, overview):
    """区间运营建议"""
    recommendations = []
    avg_temp = mean(item["weather"]["temperature_c"] for item in day_results)
    avg_rain_probability = mean(item["weather"]["rain_probability"] for item in day_results)

    if overview["peak_days"] > 0:
        recommendations.append("Add extra staffing at security, ticket entry, and top attraction gates from 09:00-11:30 and 16:30-19:30 on peak days.")
    if overview["busy_days"] >= 2:
        recommendations.append("Lock in food replenishment timing early and shift high-demand resources toward headline attractions and evening traffic corridors.")
    if avg_rain_probability >= 0.48:
        recommendations.append("Prepare rain gear sales, indoor re-routing, and parade contingency plans to reduce weather-driven congestion.")
    if avg_temp >= 31:
        recommendations.append("Increase checks at hydration points and add clearer guidance to shaded rest zones around midday.")
    if overview["baseline_delta_pct"] <= -8:
        recommendations.append("Use lower-pressure windows for membership campaigns, bundled offers, and guest experience refinements.")

    if not recommendations:
        recommendations.append("A standard weekday operating plan should be sufficient, with extra attention on afternoon dining and pre-show traffic swings.")

    return recommendations[:4]


def build_visitor_recommendations(day_results, overview):
    """区间游客建议"""
    recommendations = []
    avg_temp = mean(item["weather"]["temperature_c"] for item in day_results)
    avg_rain_probability = mean(item["weather"]["rain_probability"] for item in day_results)

    if overview["peak_days"] > 0 or overview["busy_days"] >= 2:
        recommendations.append("Arrive at least 30-45 minutes early and complete your highest-priority attractions first.")
    if avg_rain_probability >= 0.48:
        recommendations.append("Bring a lightweight rain poncho and schedule outdoor attractions before moving indoors later in the day.")
    if avg_temp >= 31:
        recommendations.append("Target major rides in the morning, then switch to indoor shows, dining, and rest areas around midday.")
    if overview["best_visit_days"]:
        best_days = ", ".join(day["date"] for day in overview["best_visit_days"])
        recommendations.append(f"If your schedule is flexible, {best_days} should be better for photo stops, catch-up rides, and a lighter pace.")

    if not recommendations:
        recommendations.append("This window supports a standard touring pace; finish your top two priorities shortly after park opening.")

    return recommendations[:4]


def build_summary_payload(overview, day_results, drivers, operational_recommendations, visitor_recommendations):
    """给 AI 使用的结构化摘要数据"""
    peak_day = overview["peak_day"]
    calm_day = overview["calm_day"]
    best_days = [item["date"] for item in overview["best_visit_days"]]
    avg_temp = int(round(mean(item["weather"]["temperature_c"] for item in day_results)))
    avg_rain_probability = round(mean(item["weather"]["rain_probability"] for item in day_results), 2)

    return {
        "project_theme": "Shanghai Disneyland Operations Visitor Flow Forecasting and Business Insight Analysis",
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
            "date_label": to_display_date_label(peak_day["date"]),
            "predicted_attendance": peak_day["predicted_attendance"],
            "crowd_level": peak_day["crowd_level"],
            "weather_label": peak_day["weather"]["weather_label"],
        },
        "calm_day": {
            "date": calm_day["date"],
            "date_label": to_display_date_label(calm_day["date"]),
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
            {"date": day_date, "date_label": to_display_date_label(day_date)}
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
                "date_label": to_display_date_label(item["date"]),
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
        raise AIInsightError("The AI response was not valid JSON.")
    return cleaned[start : end + 1]


def dedupe_preserve_order(items):
    """Return a list without duplicates while preserving order."""
    deduped = []
    seen = set()
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def coerce_optional_text_list(value, maximum=4):
    """Normalize AI text output into a list without enforcing a minimum length."""
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
    return dedupe_preserve_order(items)[:maximum]


def merge_text_list_with_fallback(value, fallback_items, maximum=4):
    """Use AI items first and top up with fallback items when needed."""
    combined = coerce_optional_text_list(value, maximum=maximum) + [
        normalize_ai_numbers_in_text(str(item))
        for item in fallback_items
        if str(item).strip()
    ]
    merged = dedupe_preserve_order(combined)[:maximum]
    if not merged:
        raise AIInsightError("The AI response did not return any usable recommendation items.")
    return merged


def split_into_chunks(items, chunk_size):
    """Split a list into smaller chunks."""
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


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
        raise AIInsightError(f"The AI response did not include enough items for {field_name}.")
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
            raise AIInsightError("The AI response did not return a valid summary.")
        return "\n\n".join(parts[:4])

    if isinstance(value, str) and value.strip():
        return normalize_ai_numbers_in_text(value)

    raise AIInsightError("The AI response did not return a valid summary.")


def merge_ai_daily_advice(day_results, daily_advice):
    """把 AI 生成的逐日建议回填到预测结果"""
    if not isinstance(daily_advice, list):
        return day_results

    advice_by_date = {}
    for item in daily_advice:
        if not isinstance(item, dict):
            continue
        day_date = str(item.get("date", "")).strip()
        day_label = str(item.get("date_label", "")).strip()

        if day_date:
            advice_by_date[day_date] = item
        elif day_label:
            advice_by_date[day_label] = item

    for index, day_result in enumerate(day_results):
        ai_day = advice_by_date.get(day_result["date"])
        if not ai_day:
            ai_day = advice_by_date.get(to_display_date_label(day_result["date"]))
        if not ai_day:
            continue

        business_note = normalize_ai_numbers_in_text(str(ai_day.get("business_note", "")))
        ops_focus = normalize_ai_numbers_in_text(str(ai_day.get("ops_focus", "")))
        visitor_tip = normalize_ai_numbers_in_text(str(ai_day.get("visitor_tip", "")))

        if business_note:
            day_result["business_note"] = business_note
        if ops_focus:
            day_result["ops_focus"] = ops_focus
        if visitor_tip:
            day_result["visitor_tip"] = visitor_tip

    return day_results


def call_modelscope_json(client, messages, max_tokens):
    """Call the configured ModelScope models and return the first parseable JSON payload."""
    model_candidates = get_modelscope_model_candidates()
    model_errors = []
    max_attempts = max(1, DEFAULT_AI_MODEL_MAX_ATTEMPTS)

    for model_name in model_candidates:
        for _ in range(max_attempts):
            try:
                response = client.with_options(timeout=DEFAULT_AI_MODEL_TIMEOUT_SECONDS).chat.completions.create(
                    model=model_name,
                    temperature=0.15,
                    max_tokens=max_tokens,
                    messages=messages,
                )
            except Exception as exc:
                model_errors.append(f"{model_name}: API call failed")
                continue

            choices = getattr(response, "choices", None)
            if not choices:
                model_errors.append(f"{model_name}: empty choices")
                continue

            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None) if message else None
            if not content:
                model_errors.append(f"{model_name}: empty message content")
                continue

            try:
                return json.loads(extract_json_object(content)), model_name
            except (json.JSONDecodeError, AIInsightError):
                model_errors.append(f"{model_name}: returned non-parseable JSON")
                continue

    reason_text = "; ".join(model_errors[-6:]) if model_errors else "unknown model failure"
    raise AIInsightError(
        "The AI returned no valid content after automatically trying fallback models. "
        f"Recent failure details: {reason_text}."
    )


def generate_ai_insights(summary_payload):
    """Call AI for overview insights and chunked daily advice."""
    token = get_modelscope_access_token()
    if not token:
        raise AIInsightError("No valid ModelScope token is configured.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AIInsightError("The current environment is missing the openai dependency.") from exc

    client = OpenAI(
        api_key=token,
        base_url=os.getenv("MODELSCOPE_BASE_URL", DEFAULT_MODELSCOPE_BASE_URL),
        timeout=DEFAULT_AI_MODEL_TIMEOUT_SECONDS,
    )

    overview_payload = {
        "project_theme": summary_payload["project_theme"],
        "report_scope": summary_payload["report_scope"],
        "forecast": summary_payload["forecast"],
        "peak_day": summary_payload["peak_day"],
        "calm_day": summary_payload["calm_day"],
        "calendar_context": summary_payload["calendar_context"],
        "weather_context": summary_payload["weather_context"],
        "best_visit_days": summary_payload["best_visit_days"],
        "drivers": summary_payload["drivers"],
        "recommendations": summary_payload["recommendations"],
    }
    overview_messages = [
        {
            "role": "system",
            "content": (
                "You are a theme park business analytics consultant. "
                "Use only the facts in the input and write in natural English. "
                "Do not invent dates, visitor counts, activities, or assumptions that are not present. "
                "Return only a JSON object, with no Markdown and no code fences."
            ),
        },
        {
            "role": "user",
            "content": (
                "Based on the following JSON, generate a structured output for Shanghai Disneyland operations stakeholders. "
                'Return JSON with exactly these fields: "summary", "drivers", "operations", "visitors". '
                '"summary" must be an array of 4 concise English paragraphs. '
                '"drivers", "operations", and "visitors" must each contain 3 concise English bullet-style strings. '
                "If you reference a date, prefer the provided date_label. "
                "Use Arabic numerals only, for example 318,937 visitors, 30-45 minutes, May 1, and 10:00 PM. "
                "Use natural business wording such as park opening, midday, late afternoon, pre-show period, or elevated rain risk. "
                "Keep all text concise, practical, and presentation-ready.\n\n"
                f"{json.dumps(overview_payload, ensure_ascii=False)}"
            ),
        },
    ]
    overview_parsed, _ = call_modelscope_json(
        client,
        overview_messages,
        max_tokens=DEFAULT_AI_OVERVIEW_MAX_TOKENS,
    )

    daily_advice = []
    daily_chunks = split_into_chunks(summary_payload.get("daily_context", []), DEFAULT_AI_DAILY_CHUNK_SIZE)
    for chunk in daily_chunks:
        chunk_payload = {
            "project_theme": summary_payload["project_theme"],
            "report_scope": summary_payload["report_scope"],
            "forecast": summary_payload["forecast"],
            "daily_context": chunk,
        }
        daily_messages = [
            {
                "role": "system",
                "content": (
                    "You are a theme park business analytics consultant. "
                    "Use only the facts in the input and write in natural English. "
                    "Return only a JSON object."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Based on the following JSON, generate day-level guidance. "
                    'Return JSON with exactly one field: "daily_advice". '
                    '"daily_advice" must be an array that covers every input day exactly once. '
                    'Each item must include "date", "business_note", "ops_focus", and "visitor_tip". '
                    'The "date" field must copy the exact ISO date from the input, such as 2026-05-01. '
                    "Keep each text field concise, natural, and practical for a business presentation.\n\n"
                    f"{json.dumps(chunk_payload, ensure_ascii=False)}"
                ),
            },
        ]
        try:
            daily_parsed, _ = call_modelscope_json(
                client,
                daily_messages,
                max_tokens=DEFAULT_AI_DAILY_MAX_TOKENS,
            )
        except AIInsightError:
            continue

        chunk_advice = daily_parsed.get("daily_advice")
        if isinstance(chunk_advice, list):
            daily_advice.extend(chunk_advice)

    return {
        "summary_text": coerce_summary_text(overview_parsed.get("summary")),
        "key_drivers": merge_text_list_with_fallback(
            overview_parsed.get("drivers"),
            summary_payload.get("drivers", []),
            maximum=4,
        ),
        "operational_recommendations": merge_text_list_with_fallback(
            overview_parsed.get("operations"),
            summary_payload.get("recommendations", {}).get("operations", []),
            maximum=4,
        ),
        "visitor_recommendations": merge_text_list_with_fallback(
            overview_parsed.get("visitors"),
            summary_payload.get("recommendations", {}).get("visitors", []),
            maximum=4,
        ),
        "daily_advice": daily_advice,
    }


def generate_rule_summary(overview, drivers, operational_recommendations):
    """无 AI 时的规则摘要"""
    peak_day = overview["peak_day"]
    best_days = ", ".join(item["date"] for item in overview["best_visit_days"]) or "No clear lower-pressure day"
    driver_text = "; ".join(drivers[:2])
    ops_text = "; ".join(operational_recommendations[:2])

    return (
        f"Across the {overview['day_count']}-day window from {overview['start_date']} to {overview['end_date']}, "
        f"average daily attendance is forecast at about {overview['average_attendance']:,}, placing the period in the "
        f'"{overview["range_signal"]}" range. The highest-pressure day is {peak_day["date"]}, with an estimated '
        f'{peak_day["predicted_attendance"]:,} visitors.\n\n'
        f"Primary demand drivers include: {driver_text}\n\n"
        f"From an operations perspective, the key watchpoints are {overview['busy_days']} busy days and "
        f"{overview['rainy_risk_days']} weather-risk days, especially around park opening and the pre-show evening window.\n\n"
        f"Recommended actions: {ops_text}. For visitors, {best_days} should offer the lighter touring window."
    )


def analyze_date_range(start_date_str, end_date_str, enable_ai_summary=True):
    """主分析逻辑"""
    start_day = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_day = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    if end_day < start_day:
        raise ValueError("The end date cannot be earlier than the start date.")

    day_count = (end_day - start_day).days + 1
    if day_count > 31:
        raise ValueError("A single analysis supports up to 31 days. Please choose a shorter range.")

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
        range_signal = "Careful planning needed"
    elif average_attendance <= ATTENDANCE_Q35:
        range_signal = "Lower-pressure window"
    else:
        range_signal = "Stable window"

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
    ai_status = {
        "requested": bool(enable_ai_summary),
        "used": False,
        "error": None,
    }

    if enable_ai_summary:
        try:
            ai_insights = generate_ai_insights(summary_payload)
            day_results = merge_ai_daily_advice(day_results, ai_insights["daily_advice"])
            key_drivers = ai_insights["key_drivers"]
            operational_recommendations = ai_insights["operational_recommendations"]
            visitor_recommendations = ai_insights["visitor_recommendations"]
            summary_text = ai_insights["summary_text"]
            summary_source = "ai"
            ai_status["used"] = True
        except AIInsightError as exc:
            ai_status["error"] = str(exc)

    return {
        "query": {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "day_count": day_count,
        },
        "weather_meta": {
            "city": "Shanghai",
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
        "ai_status": ai_status,
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
            return jsonify({"success": False, "error": "Please provide both a start date and an end date."}), 400

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
            return jsonify({"success": False, "error": "Please provide a date."}), 400

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
            return jsonify({"success": False, "error": "Please provide a start date."}), 400

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
    print("Shanghai Disneyland operations insight app started.")
    print("Open: http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
