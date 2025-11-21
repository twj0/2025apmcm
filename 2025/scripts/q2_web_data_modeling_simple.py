#!/usr/bin/env python3
"""
Q2建模分析: 基于web数据的汽车销量预测模型 (简化版)
"""

import csv
import json

def load_web_data(file_path):
    """加载web数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data

def simple_linear_regression(x, y):
    """简单线性回归"""
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)
    
    # 计算回归系数
    b1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b0 = (sum_y - b1 * sum_x) / n
    
    return b0, b1

def calculate_r2(y_true, y_pred):
    """计算R²值"""
    y_mean = sum(y_true) / len(y_true)
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y_true, y_pred))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y_true)
    return 1 - (ss_res / ss_tot)

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
    years = [int(row['year']) for row in brand_data]
    sales = [int(row['sales_units']) for row in brand_data]
    
    # 线性回归模型
    b0, b1 = simple_linear_regression(years, sales)
    
    # 模型评估
    y_pred = [b0 + b1 * x for x in years]
    r2 = calculate_r2(sales, y_pred)
    
    # 预测2026年
    pred_2026 = b0 + b1 * 2026
    
    return {
        'brand': brand_name,
        'linear_r2': r2,
        'linear_pred_2026': pred_2026,
        'years': years,
        'sales': sales,
        'model_params': {'b0': b0, 'b1': b1}
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
    
    years = sorted(sales_by_year.keys())
    total_sales = [sales_by_year[year] for year in years]
    
    # 线性回归模型
    b0, b1 = simple_linear_regression(years, total_sales)
    
    # 模型评估
    y_pred = [b0 + b1 * x for x in years]
    r2 = calculate_r2(total_sales, y_pred)
    
    # 预测2026年
    pred_2026 = b0 + b1 * 2026
    
    growth_2025 = (sales_by_year[2025] - sales_by_year[2024]) / sales_by_year[2024] * 100
    
    return {
        'years': years,
        'total_sales': total_sales,
        'pred_2026': pred_2026,
        'r2': r2,
        'growth_2025': growth_2025,
        'model_params': {'b0': b0, 'b1': b1}
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
        print(f'   - 2026年预测 (线性): {toyota_model["linear_pred_2026"]:,.0f}')
    
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
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print()
    print('5. 结果保存:')
    print('   - 建模结果已保存至: results/q2_web_data_modeling_results.json')
    
    return results

if __name__ == '__main__':
    main()