#!/usr/bin/env python3
"""
中国海关大豆进口数据获取工具 - 演示模拟版本

由于实际的海关网站存在SSL证书问题，本脚本提供模拟数据来演示
数据获取和分析的完整流程。
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_simulated_data(start_year=2022, end_year=2024):
    """生成模拟的大豆进口数据"""
    print("正在生成模拟数据...")
    
    data = []
    base_import = 8000000  # 基础进口量（吨）
    
    # 季节性模式：春季较低，秋季较高
    seasonal_pattern = {
        1: 0.85, 2: 0.80, 3: 0.75, 4: 0.85, 5: 0.90, 6: 0.95,
        7: 1.05, 8: 1.10, 9: 1.15, 10: 1.20, 11: 1.10, 12: 0.95
    }
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # 基础季节性调整
            seasonal_factor = seasonal_pattern[month]
            
            # 年度趋势（小幅增长）
            year_trend = 1.0 + (year - start_year) * 0.02
            
            # 随机波动
            random_factor = random.uniform(0.85, 1.15)
            
            # 计算月度进口量
            monthly_import = base_import * seasonal_factor * year_trend * random_factor
            
            # 添加一些特殊事件（如贸易政策变化）
            if year == 2023 and month == 3:
                monthly_import *= 1.3  # 政策刺激
            elif year == 2024 and month == 8:
                monthly_import *= 0.7  # 市场调整
            
            # 确保数据合理
            monthly_import = max(monthly_import, 2000000)  # 最小200万吨
            monthly_import = min(monthly_import, 15000000)  # 最大1500万吨
            
            # 创建数据记录
            data.append({
                'date_str': f"{year}-{month:02d}",
                'date': datetime(year, month, 1),
                'year': year,
                'month': month,
                'total_import': int(monthly_import),
                'source': '模拟数据',
                'query_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    df = pd.DataFrame(data)
    print(f"✓ 生成了 {len(data)} 个月的模拟数据")
    return df

def basic_analysis_demo(df):
    """基础分析演示"""
    print("\n" + "="*60)
    print("基础分析演示")
    print("="*60)
    
    # 基本统计
    total_import = df['total_import'].sum()
    avg_import = df['total_import'].mean()
    std_import = df['total_import'].std()
    
    print(f"数据期间: {df['date'].min().strftime('%Y年%m月')} - {df['date'].max().strftime('%Y年%m月')}")
    print(f"总进口量: {total_import:,.0f} 吨")
    print(f"平均月度进口量: {avg_import:,.0f} 吨")
    print(f"标准差: {std_import:,.0f} 吨")
    print(f"变异系数: {(std_import/avg_import)*100:.1f}%")
    
    # 按年份统计
    yearly_stats = df.groupby('year')['total_import'].agg(['sum', 'mean', 'count'])
    
    print("\n按年份统计:")
    for year in yearly_stats.index:
        total = yearly_stats.loc[year, 'sum']
        avg = yearly_stats.loc[year, 'mean']
        count = yearly_stats.loc[year, 'count']
        print(f"  {year}年: 总量 {total:,.0f} 吨, 平均 {avg:,.0f} 吨/月, 数据 {count} 个月")

def seasonal_analysis_demo(df):
    """季节性分析演示"""
    print("\n" + "="*60)
    print("季节性分析演示")
    print("="*60)
    
    monthly_stats = df.groupby('month')['total_import'].agg(['mean', 'std', 'count'])
    
    print("月度进口量统计:")
    for month in range(1, 13):
        if month in monthly_stats.index:
            mean_val = monthly_stats.loc[month, 'mean']
            std_val = monthly_stats.loc[month, 'std']
            count = monthly_stats.loc[month, 'count']
            month_name = f"{month}月"
            print(f"  {month_name:>4}: 平均 {mean_val:>8,.0f} 吨 (样本: {count}个月, 标准差: {std_val:,.0f})")
    
    # 找出进口量最高和最低的月份
    if len(monthly_stats) > 0:
        max_month = monthly_stats['mean'].idxmax()
        min_month = monthly_stats['mean'].idxmin()
        
        print(f"\n进口量最高的月份: {max_month}月 (平均 {monthly_stats.loc[max_month, 'mean']:,.0f} 吨)")
        print(f"进口量最低的月份: {min_month}月 (平均 {monthly_stats.loc[min_month, 'mean']:,.0f} 吨)")

def trend_analysis_demo(df):
    """趋势分析演示"""
    print("\n" + "="*60)
    print("趋势分析演示")
    print("="*60)
    
    # 计算移动平均线
    df = df.sort_values('date').copy()
    df['ma_3'] = df['total_import'].rolling(window=3).mean()
    df['ma_6'] = df['total_import'].rolling(window=6).mean()
    
    # 计算趋势
    first_half = df['total_import'][:len(df)//2].mean()
    second_half = df['total_import'][len(df)//2:].mean()
    
    trend_change = ((second_half - first_half) / first_half) * 100
    
    print(f"整体趋势变化: {trend_change:+.1f}%")
    print(f"前半期平均: {first_half:,.0f} 吨")
    print(f"后半期平均: {second_half:,.0f} 吨")
    
    if trend_change > 5:
        trend_desc = "显著上升"
    elif trend_change < -5:
        trend_desc = "显著下降"
    else:
        trend_desc = "相对稳定"
    
    print(f"趋势描述: {trend_desc}")
    
    # 显示最近几个月的数据
    print("\n最近几个月的数据:")
    recent_data = df.tail(6)[['date_str', 'total_import', 'ma_3', 'ma_6']]
    for _, row in recent_data.iterrows():
        print(f"  {row['date_str']}: {row['total_import']:>8,.0f} 吨 "
              f"(3个月平均: {row['ma_3']:>8,.0f} 吨, "
              f"6个月平均: {row['ma_6']:>8,.0f} 吨)")

def data_export_demo(df):
    """数据导出演示"""
    print("\n" + "="*60)
    print("数据导出演示")
    print("="*60)
    
    # 导出为CSV
    csv_file = "demo_soybean_import_data.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✓ 数据已导出到: {csv_file}")
    
    # 导出为Excel
    try:
        excel_file = "demo_soybean_import_data.xlsx"
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✓ 数据已导出到: {excel_file}")
    except ImportError:
        print("⚠ 未安装 openpyxl，无法导出Excel文件")
    
    # 创建汇总表
    summary_data = []
    
    # 按年份汇总
    yearly_summary = df.groupby('year').agg({
        'total_import': ['sum', 'mean', 'max', 'min', 'std']
    }).round(0)
    
    # 按月份汇总
    monthly_summary = df.groupby('month').agg({
        'total_import': ['mean', 'std']
    }).round(0)
    
    # 保存汇总表
    with pd.ExcelWriter('demo_soybean_analysis_summary.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='原始数据', index=False)
        yearly_summary.to_excel(writer, sheet_name='年度汇总')
        monthly_summary.to_excel(writer, sheet_name='月度汇总')
    
    print("✓ 汇总分析已保存到: demo_soybean_analysis_summary.xlsx")

def simulate_real_workflow():
    """模拟真实工作流程"""
    print("模拟真实工作流程")
    print("="*60)
    
    # 步骤1: 生成模拟数据
    print("步骤1: 生成模拟数据...")
    df = generate_simulated_data(2022, 2024)
    
    # 步骤2: 基础分析
    basic_analysis_demo(df)
    
    # 步骤3: 季节性分析
    seasonal_analysis_demo(df)
    
    # 步骤4: 趋势分析
    trend_analysis_demo(df)
    
    # 步骤5: 数据导出
    data_export_demo(df)
    
    print("\n" + "="*60)
    print("工作流程模拟完成!")
    print("="*60)
    print("\n生成的文件:")
    print("  - demo_soybean_import_data.csv: 原始数据")
    print("  - demo_soybean_import_data.xlsx: Excel格式数据")
    print("  - demo_soybean_analysis_summary.xlsx: 分析汇总")
    print("\n这些文件可以用于:")
    print("  - 数据可视化练习")
    print("  - 统计分析练习")
    print("  - 报告制作")
    print("  - 模型训练")

def interactive_demo():
    """交互式演示"""
    print("交互式演示模式")
    print("="*60)
    
    while True:
        print("\n选项:")
        print("1. 生成模拟数据")
        print("2. 基础分析")
        print("3. 季节性分析")
        print("4. 趋势分析")
        print("5. 数据导出")
        print("6. 完整工作流程演示")
        print("0. 退出")
        
        choice = input("\n请选择 (0-6): ").strip()
        
        if choice == "0":
            print("感谢使用演示程序!")
            break
        
        elif choice == "1":
            start_year = int(input("开始年份 (默认2022): ") or "2022")
            end_year = int(input("结束年份 (默认2024): ") or "2024")
            df = generate_simulated_data(start_year, end_year)
            print(f"生成了 {len(df)} 条数据")
            # 保存到全局变量供其他功能使用
            globals()['current_df'] = df
        
        elif choice in ["2", "3", "4", "5"]:
            if 'current_df' not in globals():
                print("请先生成数据 (选项1)")
                continue
            
            df = globals()['current_df']
            
            if choice == "2":
                basic_analysis_demo(df)
            elif choice == "3":
                seasonal_analysis_demo(df)
            elif choice == "4":
                trend_analysis_demo(df)
            elif choice == "5":
                data_export_demo(df)
        
        elif choice == "6":
            simulate_real_workflow()
        
        else:
            print("无效选择，请重试")

def main():
    """主函数"""
    print("中国海关大豆进口数据获取工具 - 演示模拟版本")
    print("="*60)
    print("由于实际的海关网站存在SSL证书验证问题，")
    print("本演示使用模拟数据来展示完整的数据分析流程。")
    print("="*60)
    
    # 询问用户选择模式
    print("\n选择演示模式:")
    print("1. 完整工作流程演示 (推荐)")
    print("2. 交互式演示")
    print("0. 退出")
    
    choice = input("\n请选择 (0-2): ").strip()
    
    if choice == "0":
        print("已取消演示")
        return
    elif choice == "1":
        simulate_real_workflow()
    elif choice == "2":
        interactive_demo()
    else:
        print("无效选择，将运行完整工作流程演示")
        simulate_real_workflow()
    
    print("\n演示完成!")
    print("\n注意: 本演示使用的是模拟数据，")
    print("实际使用时需要解决SSL证书问题或联系海关获取官方数据。")

if __name__ == "__main__":
    main()