"""
数据处理脚本
替代notebook进行数据清洗和特征工程
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_data():
    """处理数据并保存"""
    # 设置路径
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, 'data', 'disney_attendance.csv')
    output_path = os.path.join(project_dir, 'data', 'disney_attendance_cleaned.csv')
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"数据形状: {df.shape}")
    
    # 检查缺失值
    print("\n检查缺失值...")
    missing = df.isnull().sum().sum()
    print(f"缺失值总数: {missing}")
    
    # 检查重复值
    duplicates = df.duplicated().sum()
    print(f"重复记录数: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"已删除 {duplicates} 条重复记录")
    
    # 特征工程
    print("\n进行特征工程...")
    
    # 添加季度特征
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # 添加季节特征
    def get_season(month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Winter'
    
    df['season'] = df['month'].apply(get_season)
    
    # 添加滞后特征
    df['visitors_lag1'] = df['visitors'].shift(1)
    df['visitors_lag7'] = df['visitors'].shift(7)
    
    # 添加移动平均特征
    df['visitors_ma7'] = df['visitors'].rolling(window=7).mean()
    df['visitors_ma30'] = df['visitors'].rolling(window=30).mean()
    
    # 填充NaN值
    df['visitors_lag1'] = df['visitors_lag1'].fillna(df['visitors'].mean())
    df['visitors_lag7'] = df['visitors_lag7'].fillna(df['visitors'].mean())
    df['visitors_ma7'] = df['visitors_ma7'].fillna(df['visitors'].mean())
    df['visitors_ma30'] = df['visitors_ma30'].fillna(df['visitors'].mean())
    
    # 确保数值列为整数
    int_columns = ['visitors', 'temperature', 'year', 'month', 'day', 'day_of_week', 
                   'day_of_year', 'week_of_year', 'quarter', 'is_month_start', 'is_month_end',
                   'visitors_lag1', 'visitors_lag7', 'visitors_ma7', 'visitors_ma30']
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # 保存处理后的数据
    df.to_csv(output_path, index=False)
    print(f"\n处理后的数据已保存到: {output_path}")
    print(f"数据形状: {df.shape}")
    
    # 数据摘要
    print("\n" + "=" * 60)
    print("数据摘要")
    print("=" * 60)
    print(f"日期范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"总记录数: {len(df):,}")
    print(f"\n游客数量统计:")
    print(f"  - 平均值: {df['visitors'].mean():,.0f} 人/天")
    print(f"  - 中位数: {df['visitors'].median():,.0f} 人/天")
    print(f"  - 最小值: {df['visitors'].min():,} 人/天")
    print(f"  - 最大值: {df['visitors'].max():,} 人/天")
    print(f"  - 标准差: {df['visitors'].std():,.0f}")
    
    return df


if __name__ == '__main__':
    process_data()