#!/usr/bin/env python3
"""
Q2结果可视化: 创建图表展示趋势
"""

import csv
import json

def create_simple_chart(data, title, filename):
    """创建简单的ASCII图表"""
    # 准备数据
    brands = sorted(set(row['brand'] for row in data))
    
    # 2024和2025年数据
    sales_2024 = {}
    sales_2025 = {}
    
    for row in data:
        brand = row['brand']
        year = row['year']
        sales = int(row['sales_units'])
        
        if year == '2024':
            sales_2024[brand] = sales
        else:
            sales_2025[brand] = sales
    
    # 创建图表
    chart_lines = []
    chart_lines.append(f"=== {title} ===")
    chart_lines.append("")
    
    # 找出最大值用于缩放
    max_sales = max(max(sales_2024.values()), max(sales_2025.values()))
    scale = 50 / max_sales  # 50个字符宽度
    
    for brand in brands:
        sales24 = sales_2024.get(brand, 0)
        sales25 = sales_2025.get(brand, 0)
        
        bar24 = int(sales24 * scale)
        bar25 = int(sales25 * scale)
        
        chart_lines.append(f"{brand:12}")
        chart_lines.append(f"  2024: {'█' * bar24} {sales24:,}")
        chart_lines.append(f"  2025: {'░' * bar25} {sales25:,}")
        
        growth = ((sales25 - sales24) / sales24 * 100) if sales24 > 0 else 0
        chart_lines.append(f"  增长: {growth:+.1f}%")
        chart_lines.append("")
    
    # 保存图表
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chart_lines))
    
    return chart_lines

def create_growth_chart(trends, title, filename):
    """创建增长率图表"""
    # 排序品牌按增长率
    sorted_brands = sorted(trends.items(), key=lambda x: x[1]['growth_rate'])
    
    chart_lines = []
    chart_lines.append(f"=== {title} ===")
    chart_lines.append("")
    
    max_growth = max(abs(trends[brand]['growth_rate']) for brand in trends)
    scale = 30 / max_growth  # 30个字符宽度
    
    for brand, data in sorted_brands:
        growth = data['growth_rate']
        bar_length = int(abs(growth) * scale)
        
        if growth >= 0:
            bar = '▲' * bar_length
            chart_lines.append(f"{brand:12} {bar} {growth:+.1f}%")
        else:
            bar = '▼' * bar_length
            chart_lines.append(f"{brand:12} {bar} {growth:+.1f}%")
    
    # 保存图表
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chart_lines))
    
    return chart_lines

def create_market_share_chart(data, title, filename):
    """创建市场份额图表"""
    # 计算市场份额
    sales_by_brand = {}
    total_sales = 0
    
    for row in data:
        if row['year'] == '2025':  # 使用2025年数据
            brand = row['brand']
            sales = int(row['sales_units'])
            
            sales_by_brand[brand] = sales
            total_sales += sales
    
    # 排序
    sorted_brands = sorted(sales_by_brand.items(), key=lambda x: x[1], reverse=True)
    
    chart_lines = []
    chart_lines.append(f"=== {title} ===")
    chart_lines.append("")
    chart_lines.append(f"总销量: {total_sales:,} 辆")
    chart_lines.append("")
    
    # 创建饼图样式的文本表示
    max_share = max(sales_by_brand.values()) / total_sales * 100
    scale = 40 / max_share
    
    for brand, sales in sorted_brands:
        share = sales / total_sales * 100
        bar_length = int(share * scale)
        
        chart_lines.append(f"{brand:12} {'█' * bar_length} {share:.1f}% ({sales:,})")
    
    # 保存图表
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(chart_lines))
    
    return chart_lines

def main():
    """主函数"""
    print('=== Q2结果可视化 ===')
    print()
    
    # 加载数据
    with open('data/external/us_auto_sales_2024_2025_web.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # 加载建模结果
    with open('results/q2_web_data_modeling_results.json', 'r', encoding='utf-8') as f:
        modeling_results = json.load(f)
    
    print('1. 创建销量对比图表...')
    sales_chart = create_simple_chart(
        data, 
        "2024-2025年汽车品牌销量对比", 
        "results/q2_sales_comparison_chart.txt"
    )
    
    print('2. 创建增长率图表...')
    growth_chart = create_growth_chart(
        modeling_results['brand_trends'],
        "品牌增长率排名",
        "results/q2_growth_rate_chart.txt"
    )
    
    print('3. 创建市场份额图表...')
    share_chart = create_market_share_chart(
        data,
        "2025年市场份额分布",
        "results/q2_market_share_chart.txt"
    )
    
    print('4. 创建预测结果图表...')
    # 预测结果摘要
    pred_summary = []
    pred_summary.append("=== 2026年销量预测结果 ===")
    pred_summary.append("")
    pred_summary.append(f"Toyota预测销量: {modeling_results['toyota_prediction']['linear_pred_2026']:,.0f} 辆")
    pred_summary.append(f"整体市场预测: {modeling_results['market_analysis']['pred_2026']:,.0f} 辆")
    pred_summary.append(f"市场增长率: {modeling_results['market_analysis']['growth_2025']:+.1f}%")
    pred_summary.append("")
    pred_summary.append("主要趋势:")
    pred_summary.append("- 韩系品牌(Hyundai, Kia)增长强劲")
    pred_summary.append("- 日系品牌分化: Toyota增长, Honda/Nissan下滑")
    pred_summary.append("- 美系传统品牌面临挑战")
    pred_summary.append("- 整体市场预计继续下滑")
    
    with open('results/q2_prediction_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(pred_summary))
    
    print()
    print('5. 可视化结果汇总:')
    print('   - 销量对比图表: results/q2_sales_comparison_chart.txt')
    print('   - 增长率图表: results/q2_growth_rate_chart.txt')
    print('   - 市场份额图表: results/q2_market_share_chart.txt')
    print('   - 预测摘要: results/q2_prediction_summary.txt')
    
    # 显示部分图表内容
    print()
    print("销量对比图表预览:")
    print('\n'.join(sales_chart[:15]))  # 显示前15行
    print("...")
    
    print()
    print("增长率图表预览:")
    print('\n'.join(growth_chart))
    
    print()
    print("预测摘要:")
    print('\n'.join(pred_summary))

if __name__ == '__main__':
    main()