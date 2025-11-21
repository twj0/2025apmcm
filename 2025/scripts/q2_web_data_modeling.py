#!/usr/bin/env python3
"""
Q2建模分析: 基于web数据的汽车销量预测模型
"""

import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json

def load_web_data(file_path):
    """加载web数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data

def analyze_brand_trends(data):
    """分析品牌趋势"""
    brands = set(row['brand'] for row in data)
    trends = {}
    
    print('2. 品牌趋势分析:')
    for brand in sorted(brands):
        brand_data = [row for row in data if row['brand'] == brand]
        brand_data.sort(key=lambda x: x['year'])
        
        if len(brand_data) >= 2:
            sales_2024 = int([row for row in brand_data if row['year'] == '2024'][0]['sales_units'])
            sales_2025 = int([row for row in brand_data if row['year'] == '2025'][0]['sales_units'])
            growth_rate = (sales_2025 - sales_2024) / sales_2024 * 100
            trends[brand] = {
                'growth_rate': growth_rate,
                'sales_2024': sales_2024,
                'sales_2025': sales_2025
            }
            print(f'   - {brand}: {growth_rate:+.1f}%')
    
    return trends

def build_prediction_models(data, brand_name):
    """为指定品牌构建预测模型"""
    brand_data = [row for row in data if row['brand'] == brand_name]
    brand_data.sort(key=lambda x: x['year'])
    
    if len(brand_data) < 2:
        return None
    
    # 准备数据
    years = np.array([int(row['year']) for row in brand_data])
    sales = np.array([int(row['sales_units']) for row in brand_data])
    
    # 线性回归模型
    X = years.reshape(-1, 1)
    linear_model = LinearRegression()
    linear_model.fit(X, sales)
    
    # 多项式回归模型 (2次)
    if len(years) >= 3:
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, sales)
    else:
        poly_model = None
        poly_features = None
    
    # 模型评估
    linear_r2 = r2_score(sales, linear_model.predict(X))
    
    # 预测2026年
    future_year = np.array([[2026]])
    linear_pred = linear_model.predict(future_year)[0]
    
    if poly_model:
        poly_pred = poly_model.predict(poly_features.transform(future_year))[0]
        poly_r2 = r2_score(sales, poly_model.predict(X_poly))
    else:
        poly_pred = None
        poly_r2 = None
    
    return {
        'brand': brand_name,
        'linear_model': linear_model,
        'poly_model': poly_model,
        'linear_r2': linear_r2,
        'poly_r2': poly_r2,
        'linear_pred_2026': linear_pred,
        'poly_pred_2026': poly_pred,
        'years': years,
        'sales': sales
    }

def market_analysis(data):
    """整体市场分析"""
    # 按年份统计总销量
    sales_by_year = {}
    for row in data:
        year = int(row['year'])
        sales = int(row['sales_units'])
        if year not in sales_by_year:
            sales_by_year[year] = 0
        sales_by_year[year] += sales
    
    years = np.array(sorted(sales_by_year.keys()))
    total_sales = np.array([sales_by_year[year] for year in years])
    
    # 线性回归模型
    X = years.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, total_sales)
    
    # 预测2026年
    pred_2026 = model.predict(np.array([[2026]]))[0]
    r2 = r2_score(total_sales, model.predict(X))
    
    return {
        'years': years,
        'total_sales': total_sales,
        'model': model,
        'pred_2026': pred_2026,
        'r2': r2,
        'growth_2025': (sales_by_year[2025] - sales_by_year[2024]) / sales_by_year[2024] * 100
    }

def main():
    """主函数"""
    print('=== Q2建模分析: 基于web数据的汽车销量预测模型 ===')
    print()
    
    # 加载数据
    data = load_web_data('data/external/us_auto_sales_2024_2025_web.csv')
    
    # 数据预处理
    print('1. 数据预处理:')
    print('   - 原始数据记录数:', len(data))
    brands = set(row['brand'] for row in data)
    print('   - 品牌数量:', len(brands))
    years = set(row['year'] for row in data)
    print('   - 年份范围:', min(years), '-', max(years))
    
    # 品牌趋势分析
    trends = analyze_brand_trends(data)
    
    # 预测模型构建 (选择Toyota作为示例)
    print()
    print('3. 预测模型构建 (以Toyota为例):')
    toyota_model = build_prediction_models(data, 'Toyota')
    
    if toyota_model:
        print(f'   - 线性模型 R²: {toyota_model["linear_r2"]:.4f}')
        if toyota_model['poly_r2']:
            print(f'   - 多项式模型 R²: {toyota_model["poly_r2"]:.4f}')
        print(f'   - 2026年预测 (线性): {toyota_model["linear_pred_2026"]:,.0f}')
        if toyota_model['poly_pred_2026']:
            print(f'   - 2026年预测 (多项式): {toyota_model["poly_pred_2026"]:,.0f}')
    
    # 整体市场分析
    print()
    print('4. 整体市场预测:')
    market_data = market_analysis(data)
    
    print(f'   - 整体市场模型 R²: {market_data["r2"]:.4f}')
    print(f'   - 2025年市场增长率: {market_data["growth_2025"]:+.1f}%')
    print(f'   - 2026年整体市场预测: {market_data["pred_2026"]:,.0f}')
    
    # 保存结果
    results = {
        'brand_trends': trends,
        'toyota_prediction': toyota_model,
        'market_analysis': market_data,
        'summary': {
            'total_brands': len(brands),
            'year_range': f"{min(years)}-{max(years)}",
            'data_points': len(data)
        }
    }
    
    with open('results/q2_web_data_modeling_results.json', 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    elif hasattr(v, 'coef_'):  # sklearn模型
                        json_results[key][k] = str(v)
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print()
    print('5. 结果保存:')
    print('   - 建模结果已保存至: results/q2_web_data_modeling_results.json')
    
    return results

if __name__ == '__main__':
    main()