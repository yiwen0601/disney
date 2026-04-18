"""
迪士尼游客数据生成脚本
基于真实的年度统计数据和影响因素生成每日游客数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import os

def generate_disney_attendance_data(start_date='2018-01-01', end_date='2024-12-31', seed=42):
    """
    生成迪士尼每日游客数据
    
    参数:
        start_date: 开始日期
        end_date: 结束日期
        seed: 随机种子
    
    返回:
        DataFrame: 包含每日游客数据的DataFrame
    """
    np.random.seed(seed)
    
    # 日期范围
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 基础数据（基于真实统计）
    # Magic Kingdom 年均游客约1700-2100万
    # 日均约48000-57000人
    base_daily_visitors = 50000
    
    # 创建DataFrame
    df = pd.DataFrame({'date': dates})
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 美国节假日
    us_holidays = holidays.US(years=range(2018, 2025))
    df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays).astype(int)
    
    # 季节因素（佛罗里达气候）
    # 春季(3-5月): 高峰期 - 春假
    # 夏季(6-8月): 高峰期 - 暑假
    # 秋季(9-11月): 低谷期 - 开学季
    # 冬季(12-2月): 中等 - 假期和寒冷
    
    def get_season_factor(month):
        if month in [3, 4, 5]:  # 春季
            return 1.25
        elif month in [6, 7, 8]:  # 夏季
            return 1.35
        elif month in [9, 10, 11]:  # 秋季
            return 0.75
        else:  # 冬季
            return 0.90
    
    df['season_factor'] = df['month'].apply(get_season_factor)
    
    # 月份因素
    month_factors = {
        1: 0.85,   # 1月 - 新年后淡季
        2: 0.80,   # 2月 - 淡季
        3: 1.20,   # 3月 - 春假开始
        4: 1.30,   # 4月 - 春假高峰
        5: 1.10,   # 5月 - 春假结束
        6: 1.35,   # 6月 - 暑假开始
        7: 1.40,   # 7月 - 暑假高峰
        8: 1.25,   # 8月 - 暑假尾声
        9: 0.65,   # 9月 - 开学季淡季
        10: 0.75,  # 10月 - 淡季
        11: 0.85,  # 11月 - 感恩节
        12: 1.15   # 12月 - 圣诞新年
    }
    df['month_factor'] = df['month'].map(month_factors)
    
    # 星期因素
    weekday_factors = {
        0: 0.85,  # 周一
        1: 0.80,  # 周二
        2: 0.85,  # 周三
        3: 0.90,  # 周四
        4: 1.05,  # 周五
        5: 1.25,  # 周六
        6: 1.20   # 周日
    }
    df['weekday_factor'] = df['day_of_week'].map(weekday_factors)
    
    # 特殊节假日因素
    def get_special_day_factor(date):
        month, day = date.month, date.day
        
        # 元旦
        if month == 1 and day <= 3:
            return 1.3
        # 马丁路德金日（1月第三个周一）
        # 总统日（2月第三个周一）
        # 春假期间（3月中旬-4月中旬）
        if month == 3 and day >= 10 or month == 4 and day <= 20:
            return 1.4
        # 复活节周末
        # 独立日
        if month == 7 and 1 <= day <= 7:
            return 1.5
        # 劳动节（9月第一个周一）
        # 哥伦布日（10月第二个周一）
        # 万圣节
        if month == 10 and day == 31:
            return 1.2
        # 感恩节周
        if month == 11 and 20 <= day <= 30:
            return 1.3
        # 圣诞新年假期
        if month == 12 and day >= 20:
            return 1.5
        if month == 1 and day == 1:
            return 1.3
        
        return 1.0
    
    df['special_day_factor'] = df['date'].apply(get_special_day_factor)
    
    # 年份因素（考虑疫情和恢复）
    year_factors = {
        2018: 1.0,
        2019: 1.02,
        2020: 0.35,  # 疫情影响
        2021: 0.65,  # 恢复期
        2022: 0.85,
        2023: 0.95,
        2024: 1.0
    }
    df['year_factor'] = df['year'].map(year_factors)
    
    # 天气因素（模拟）
    # 佛罗里达夏季多雨和飓风季节
    def get_weather_factor(row):
        month = row['month']
        # 夏季飓风季节可能影响
        if month in [8, 9, 10]:
            return np.random.choice([0.7, 0.8, 0.9, 1.0, 1.0, 1.0], p=[0.05, 0.1, 0.15, 0.3, 0.2, 0.2])
        return np.random.choice([0.9, 0.95, 1.0, 1.0, 1.05], p=[0.1, 0.15, 0.4, 0.25, 0.1])
    
    df['weather_factor'] = df.apply(get_weather_factor, axis=1)
    
    # 计算最终游客数量
    df['visitors'] = (
        base_daily_visitors * 
        df['month_factor'] * 
        df['weekday_factor'] * 
        df['special_day_factor'] * 
        df['year_factor'] * 
        df['weather_factor']
    )
    
    # 添加随机噪声
    noise = np.random.normal(0, 0.08, len(df))
    df['visitors'] = df['visitors'] * (1 + noise)
    
    # 确保游客数量为整数且在合理范围内
    df['visitors'] = df['visitors'].astype(int)
    df['visitors'] = df['visitors'].clip(lower=5000, upper=80000)
    
    # 添加额外特征
    df['is_peak_season'] = ((df['month'].isin([3, 4, 6, 7, 8, 12]))).astype(int)
    df['is_school_holiday'] = (
        (df['month'] == 3) | 
        (df['month'] == 4) | 
        (df['month'] == 6) | 
        (df['month'] == 7) | 
        (df['month'] == 8) | 
        ((df['month'] == 11) & (df['day'] >= 20)) |
        ((df['month'] == 12) & (df['day'] >= 20))
    ).astype(int)
    
    # 添加温度特征（模拟佛罗里达气温）
    def get_temperature(month):
        # 佛罗里达平均温度（华氏度）
        temp_map = {
            1: 65, 2: 68, 3: 72, 4: 78, 5: 82, 6: 86,
            7: 88, 8: 88, 9: 85, 10: 78, 11: 70, 12: 65
        }
        base_temp = temp_map[month]
        return base_temp + np.random.randint(-5, 6)
    
    df['temperature'] = df['month'].apply(get_temperature)
    
    # 添加降雨概率
    def get_rain_probability(month):
        # 佛罗里达降雨概率
        rain_map = {
            1: 0.15, 2: 0.18, 3: 0.20, 4: 0.22, 5: 0.28, 6: 0.45,
            7: 0.50, 8: 0.55, 9: 0.45, 10: 0.25, 11: 0.18, 12: 0.15
        }
        return rain_map[month]
    
    df['rain_probability'] = df['month'].apply(get_rain_probability)
    df['is_rainy'] = (np.random.random(len(df)) < df['rain_probability']).astype(int)
    
    # 选择最终列
    final_columns = [
        'date', 'year', 'month', 'day', 'day_of_week', 'day_of_year', 
        'week_of_year', 'is_weekend', 'is_holiday', 'is_peak_season', 
        'is_school_holiday', 'temperature', 'rain_probability', 'is_rainy',
        'visitors'
    ]
    
    df = df[final_columns]
    
    return df


def main():
    """主函数"""
    # 生成数据
    print("正在生成迪士尼游客数据...")
    df = generate_disney_attendance_data()
    
    # 保存数据
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'disney_attendance.csv')
    df.to_csv(output_path, index=False)
    
    print(f"数据已保存到: {output_path}")
    print(f"数据形状: {df.shape}")
    print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"游客数量范围: {df['visitors'].min()} - {df['visitors'].max()}")
    print(f"\n数据统计:")
    print(df.describe())
    
    return df


if __name__ == '__main__':
    main()
