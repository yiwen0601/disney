"""
迪士尼游客数量预测 Web 应用
Flask 后端服务
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import holidays

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# 全局变量存储模型和特征列
model = None
feature_columns = None


def load_model():
    """加载训练好的模型"""
    global model, feature_columns
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    model_path = os.path.join(model_dir, 'best_model.pkl')
    features_path = os.path.join(model_dir, 'feature_columns.pkl')
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_columns = joblib.load(features_path)
        print(f"模型加载成功: {model_path}")
        print(f"特征数量: {len(feature_columns)}")
        return True
    else:
        print(f"模型文件不存在，请先运行 train_model.py 训练模型")
        return False


def get_season(month):
    """获取季节"""
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'


def get_season_encoded(season):
    """季节编码"""
    season_map = {'Fall': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
    return season_map.get(season, 0)


def prepare_features_for_prediction(date_str, temperature=None, rain_probability=None):
    """
    为预测准备特征
    
    参数:
        date_str: 日期字符串 (YYYY-MM-DD)
        temperature: 温度（可选）
        rain_probability: 降雨概率（可选）
    
    返回:
        DataFrame: 特征DataFrame
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # 美国节假日
    us_holidays = holidays.US(years=date.year)
    is_holiday = 1 if date in us_holidays else 0
    
    # 基础特征
    month = date.month
    day = date.day
    day_of_week = date.weekday()
    day_of_year = date.timetuple().tm_yday
    week_of_year = date.isocalendar()[1]
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # 季节特征
    season = get_season(month)
    season_encoded = get_season_encoded(season)
    
    # 高峰季节
    is_peak_season = 1 if month in [3, 4, 6, 7, 8, 12] else 0
    
    # 学校假期
    is_school_holiday = 1 if (
        month in [3, 4, 6, 7, 8] or 
        (month == 11 and day >= 20) or 
        (month == 12 and day >= 20)
    ) else 0
    
    # 季度
    quarter = (month - 1) // 3 + 1
    
    # 月初月末
    is_month_start = 1 if day == 1 else 0
    is_month_end = 1 if day >= 28 else 0
    
    # 温度（佛罗里达平均温度）
    if temperature is None:
        temp_map = {
            1: 65, 2: 68, 3: 72, 4: 78, 5: 82, 6: 86,
            7: 88, 8: 88, 9: 85, 10: 78, 11: 70, 12: 65
        }
        temperature = temp_map.get(month, 75)
    
    # 降雨概率
    if rain_probability is None:
        rain_map = {
            1: 0.15, 2: 0.18, 3: 0.20, 4: 0.22, 5: 0.28, 6: 0.45,
            7: 0.50, 8: 0.55, 9: 0.45, 10: 0.25, 11: 0.18, 12: 0.15
        }
        rain_probability = rain_map.get(month, 0.25)
    
    is_rainy = 1 if np.random.random() < rain_probability else 0
    
    # 年份归一化（假设训练数据年份范围是2018-2024）
    year_normalized = (date.year - 2018) / (2024 - 2018)
    
    # 滞后特征和移动平均（使用默认值）
    # 在实际应用中，这些应该从历史数据获取
    visitors_lag1 = 45000
    visitors_lag7 = 45000
    visitors_ma7 = 45000
    visitors_ma30 = 45000
    
    # 创建特征字典
    features = {
        'month': month,
        'day': day,
        'day_of_week': day_of_week,
        'day_of_year': day_of_year,
        'week_of_year': week_of_year,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'is_peak_season': is_peak_season,
        'is_school_holiday': is_school_holiday,
        'temperature': temperature,
        'rain_probability': rain_probability,
        'is_rainy': is_rainy,
        'quarter': quarter,
        'is_month_start': is_month_start,
        'is_month_end': is_month_end,
        'visitors_lag1': visitors_lag1,
        'visitors_lag7': visitors_lag7,
        'visitors_ma7': visitors_ma7,
        'visitors_ma30': visitors_ma30,
        'season_encoded': season_encoded,
        'year_normalized': year_normalized
    }
    
    return pd.DataFrame([features])


def get_travel_advice(predicted_visitors, date_str):
    """
    根据预测游客数量生成出行建议
    
    参数:
        predicted_visitors: 预测的游客数量
        date_str: 日期字符串
    
    返回:
        dict: 包含建议信息的字典
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    month = date.month
    day_of_week = date.weekday()
    
    # 游客等级划分
    if predicted_visitors < 30000:
        level = "低"
        color = "#27ae60"
        emoji = "✅"
        general_advice = "游客较少，是游玩的好时机！可以享受较短的排队时间。"
        tips = [
            "热门项目排队时间预计较短，建议优先体验",
            "可以考虑购买Genie+服务以进一步优化游玩体验",
            "建议提前预订餐厅，享受更悠闲的用餐体验"
        ]
    elif predicted_visitors < 45000:
        level = "中等"
        color = "#f39c12"
        emoji = "⚠️"
        general_advice = "游客数量适中，需要合理规划游玩路线。"
        tips = [
            "建议提前制定游玩计划，优先体验热门项目",
            "考虑使用Lightning Lane或Genie+服务",
            "避开高峰时段用餐，错峰就餐"
        ]
    elif predicted_visitors < 60000:
        level = "较高"
        color = "#e67e22"
        emoji = "🔶"
        general_advice = "游客较多，需要做好充分准备和规划。"
        tips = [
            "强烈建议购买Genie+或Lightning Lane服务",
            "提前30天预订餐厅",
            "利用Early Entry或Extended Evening Hours（如适用）",
            "准备充足的饮水和防晒用品"
        ]
    else:
        level = "高峰"
        color = "#e74c3c"
        emoji = "🔴"
        general_advice = "游客高峰期，需要做好长时间排队的心理准备。"
        tips = [
            "必须购买Genie+和Individual Lightning Lane",
            "提前60天预订餐厅",
            "考虑购买VIP导览服务",
            "利用酒店Early Entry提前入园",
            "避开最热门项目，选择冷门项目游玩",
            "考虑分两天游玩，减轻单日压力"
        ]
    
    # 时间建议
    time_advice = []
    
    # 基于星期几的建议
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    if day_of_week < 5:
        time_advice.append(f"今天是{weekday_names[day_of_week]}，相对周末游客较少")
    else:
        time_advice.append(f"今天是{weekday_names[day_of_week]}，属于周末，游客通常较多")
    
    # 基于月份的建议
    month_names = ['一月', '二月', '三月', '四月', '五月', '六月', 
                   '七月', '八月', '九月', '十月', '十一月', '十二月']
    
    if month in [6, 7, 8]:
        time_advice.append("正值暑期，是一年中的旅游高峰期")
    elif month in [3, 4]:
        time_advice.append("春假期间，游客数量较多")
    elif month in [9, 10]:
        time_advice.append("属于淡季，是游玩的好时机")
    elif month == 12 or month == 1:
        time_advice.append("节假日季节，请关注特殊活动和表演时间")
    
    return {
        'level': level,
        'color': color,
        'emoji': emoji,
        'general_advice': general_advice,
        'tips': tips,
        'time_advice': time_advice
    }


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        date_str = data.get('date')
        temperature = data.get('temperature')
        rain_probability = data.get('rain_probability')
        
        if not date_str:
            return jsonify({'error': '请提供日期'}), 400
        
        # 验证日期格式
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': '日期格式错误，请使用 YYYY-MM-DD 格式'}), 400
        
        # 准备特征
        features_df = prepare_features_for_prediction(date_str, temperature, rain_probability)
        
        # 预测
        prediction = model.predict(features_df)[0]
        prediction = int(max(5000, min(80000, prediction)))  # 限制在合理范围内
        
        # 获取出行建议
        advice = get_travel_advice(prediction, date_str)
        
        # 格式化日期显示
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        weekday_names = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
        formatted_date = f"{date_obj.year}年{date_obj.month}月{date_obj.day}日 {weekday_names[date_obj.weekday()]}"
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'formatted_date': formatted_date,
            'advice': advice,
            'features': {
                'temperature': features_df['temperature'].iloc[0],
                'rain_probability': features_df['rain_probability'].iloc[0],
                'is_weekend': features_df['is_weekend'].iloc[0],
                'is_holiday': features_df['is_holiday'].iloc[0],
                'is_peak_season': features_df['is_peak_season'].iloc[0]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测接口"""
    try:
        data = request.get_json()
        dates = data.get('dates', [])
        
        if not dates:
            return jsonify({'error': '请提供日期列表'}), 400
        
        results = []
        for date_str in dates:
            features_df = prepare_features_for_prediction(date_str)
            prediction = model.predict(features_df)[0]
            prediction = int(max(5000, min(80000, prediction)))
            
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            results.append({
                'date': date_str,
                'weekday': date_obj.weekday(),
                'prediction': prediction
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 加载模型
    if load_model():
        print("\n启动Web服务...")
        print("访问 http://localhost:5000 使用预测界面")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("请先运行 train_model.py 训练模型")