"""
迪士尼游客数量预测模型训练脚本
使用多种机器学习模型进行训练和评估
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import xgboost as xgb


def load_data(filepath):
    """加载数据"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_features(df):
    """准备特征"""
    # 选择特征列
    feature_columns = [
        'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year',
        'is_weekend', 'is_holiday', 'is_peak_season', 'is_school_holiday',
        'temperature', 'rain_probability', 'is_rainy',
        'quarter', 'is_month_start', 'is_month_end',
        'visitors_lag1', 'visitors_lag7', 'visitors_ma7', 'visitors_ma30'
    ]
    
    # 处理季节特征
    season_encoder = LabelEncoder()
    df['season_encoded'] = season_encoder.fit_transform(df['season'])
    feature_columns.append('season_encoded')
    
    # 添加年份特征（归一化）
    df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    feature_columns.append('year_normalized')
    
    X = df[feature_columns]
    y = df['visitors']
    
    return X, y, feature_columns


def train_evaluate_models(X_train, X_test, y_train, y_test):
    """训练和评估多个模型"""
    results = {}
    models = {}
    
    # 定义模型
    model_configs = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    print("开始训练模型...")
    print("=" * 60)
    
    for name, model in model_configs.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 计算评估指标
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        
        results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mape': test_mape
        }
        
        models[name] = model
        
        print(f"  训练集 RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
        print(f"  测试集 RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")
        print(f"  测试集 MAPE: {test_mape:.2f}%")
    
    return models, results


def tune_best_model(X_train, y_train, model_type='xgboost'):
    """对最佳模型进行超参数调优"""
    print("\n" + "=" * 60)
    print("进行超参数调优...")
    
    if model_type == 'xgboost':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    elif model_type == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 使用较小的网格搜索以节省时间
    small_param_grid = {k: v[:2] for k, v in param_grid.items()}
    
    grid_search = GridSearchCV(
        model, small_param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳得分 (负MSE): {grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_


def save_model(model, scaler, feature_columns, model_dir):
    """保存模型和预处理器"""
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(model_dir, 'best_model.pkl')
    joblib.dump(model, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存特征列
    features_path = os.path.join(model_dir, 'feature_columns.pkl')
    joblib.dump(feature_columns, features_path)
    print(f"特征列已保存到: {features_path}")
    
    # 保存模型信息
    model_info = {
        'model_type': type(model).__name__,
        'feature_count': len(feature_columns),
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    info_path = os.path.join(model_dir, 'model_info.pkl')
    joblib.dump(model_info, info_path)
    print(f"模型信息已保存到: {info_path}")


def main():
    """主函数"""
    # 设置路径
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_dir, 'data', 'disney_attendance_cleaned.csv')
    model_dir = os.path.join(project_dir, 'models')
    
    # 加载数据
    print("加载数据...")
    df = load_data(data_path)
    print(f"数据形状: {df.shape}")
    
    # 准备特征
    print("\n准备特征...")
    X, y, feature_columns = prepare_features(df)
    print(f"特征数量: {len(feature_columns)}")
    print(f"特征列表: {feature_columns}")
    
    # 分割数据集
    # 使用时间序列分割：用前面的数据训练，后面的数据测试
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练和评估模型
    models, results = train_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 找出最佳模型
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    print("\n" + "=" * 60)
    print(f"最佳模型: {best_model_name}")
    print(f"测试集 R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"测试集 RMSE: {results[best_model_name]['test_rmse']:.2f}")
    print(f"测试集 MAE: {results[best_model_name]['test_mae']:.2f}")
    
    # 超参数调优
    best_model = tune_best_model(X_train_scaled, y_train, model_type='xgboost')
    
    # 评估调优后的模型
    y_pred = best_model.predict(X_test_scaled)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_mae = mean_absolute_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)
    final_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print("\n" + "=" * 60)
    print("最终模型性能:")
    print(f"  RMSE: {final_rmse:.2f}")
    print(f"  MAE: {final_mae:.2f}")
    print(f"  R²: {final_r2:.4f}")
    print(f"  MAPE: {final_mape:.2f}%")
    
    # 保存模型
    save_model(best_model, scaler, feature_columns, model_dir)
    
    # 保存评估结果
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(project_dir, 'data', 'model_comparison.csv')
    results_df.to_csv(results_path)
    print(f"\n模型对比结果已保存到: {results_path}")
    
    return best_model, results


if __name__ == '__main__':
    main()