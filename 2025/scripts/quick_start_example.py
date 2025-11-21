#!/usr/bin/env python3
"""
中国海关大豆进口数据获取工具 - 快速开始示例

本脚本提供了完整的使用示例，展示了如何使用不同的数据获取器
来获取和处理大豆进口数据。
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_customs_soybean_fetcher import SimpleCustomsDataFetcher
from ssl_customs_soybean_fetcher import SSLCustomsDataFetcher
from china_customs_soybean_fetcher import ChinaCustomsDataFetcher

def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)
    
    # 创建数据获取器
    fetcher = SimpleCustomsDataFetcher()
    
    # 获取2024年1月的数据
    year, month = 2024, 1
    print(f"正在获取 {year}年{month}月 的大豆进口数据...")
    
    data = fetcher.get_monthly_import_data(year, month)
    
    if data:
        print(f"✓ 成功获取数据!")
        print(f"  总进口量: {data['total_import']:,.0f} 吨")
        print(f"  数据来源: {data['source']}")
        print(f"  查询时间: {data['query_time']}")
        
        # 保存数据
        filename = f"example_soybean_import_{year}_{month:02d}.csv"
        if fetcher.save_data_to_csv(data, filename):
            print(f"  数据已保存到: {filename}")
        else:
            print("  数据保存失败")
    else:
        print("✗ 未能获取到数据")
        print("  可能原因: 数据尚未发布、网络问题、网站结构变化")
    
    print()

def example_ssl_usage():
    """SSL兼容版使用示例"""
    print("=" * 60)
    print("示例2: SSL兼容版使用")
    print("=" * 60)
    
    # 创建SSL兼容的获取器
    fetcher = SSLCustomsDataFetcher()
    
    # 获取2024年2月的数据
    year, month = 2024, 2
    print(f"正在获取 {year}年{month}月 的大豆进口数据 (SSL兼容模式)...")
    
    data = fetcher.get_monthly_import_data(year, month)
    
    if data:
        print(f"✓ 成功获取数据!")
        print(f"  总进口量: {data['total_import']:,.0f} 吨")
        print(f"  数据来源: {data['source']}")
    else:
        print("✗ 未能获取到数据")
    
    print()

def example_batch_processing():
    """批量处理示例"""
    print("=" * 60)
    print("示例3: 批量获取多个月份数据")
    print("=" * 60)
    
    fetcher = SimpleCustomsDataFetcher()
    
    # 定义要查询的月份范围
    months_to_query = [
        (2023, 10), (2023, 11), (2023, 12),
        (2024, 1), (2024, 2), (2024, 3)
    ]
    
    all_data = []
    success_count = 0
    
    print(f"计划获取 {len(months_to_query)} 个月的数据...")
    
    for year, month in months_to_query:
        print(f"  正在获取 {year}年{month}月 数据...", end=" ")
        
        data = fetcher.get_monthly_import_data(year, month)
        
        if data:
            all_data.append(data)
            print(f"✓ 成功 ({data['total_import']:,.0f} 吨)")
            success_count += 1
        else:
            print("✗ 失败")
    
    print(f"\n批量获取完成: 成功 {success_count}/{len(months_to_query)} 个月")
    
    # 保存批量数据
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = "example_soybean_import_batch.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"批量数据已保存到: {output_file}")
        print("\n数据概览:")
        print(df[['date_str', 'total_import', 'source']].to_string())
    
    print()

def example_data_analysis():
    """数据分析示例"""
    print("=" * 60)
    print("示例4: 数据分析和可视化")
    print("=" * 60)
    
    # 假设我们已经有了批量数据
    batch_file = "example_soybean_import_batch.csv"
    
    if os.path.exists(batch_file):
        df = pd.read_csv(batch_file)
        
        print("数据分析结果:")
        print(f"  总月份数: {len(df)}")
        print(f"  平均月度进口量: {df['total_import'].mean():,.0f} 吨")
        print(f"  最大月度进口量: {df['total_import'].max():,.0f} 吨")
        print(f"  最小月度进口量: {df['total_import'].min():,.0f} 吨")
        print(f"  标准差: {df['total_import'].std():,.0f} 吨")
        
        # 显示详细数据
        print("\n详细数据:")
        for _, row in df.iterrows():
            print(f"  {row['date_str']}: {row['total_import']:,.0f} 吨")
        
        # 创建简单的趋势分析
        if len(df) > 2:
            trend = "上升" if df['total_import'].iloc[-1] > df['total_import'].iloc[0] else "下降"
            print(f"\n趋势分析: 总体呈{trend}趋势")
    else:
        print("未找到批量数据文件，请先运行示例3")
    
    print()

def example_error_handling():
    """错误处理示例"""
    print("=" * 60)
    print("示例5: 错误处理和重试机制")
    print("=" * 60)
    
    fetcher = SimpleCustomsDataFetcher()
    
    # 尝试获取一个可能不存在的数据
    year, month = 2025, 12  # 未来的日期
    
    print(f"尝试获取 {year}年{month}月 数据 (预期会失败)...")
    
    # 重试机制
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        print(f"  尝试 {retry_count + 1}/{max_retries}...")
        
        data = fetcher.get_monthly_import_data(year, month)
        
        if data:
            print(f"  ✓ 成功获取数据!")
            break
        else:
            print(f"  ✗ 失败")
            retry_count += 1
            
            if retry_count < max_retries:
                print(f"  等待2秒后重试...")
                import time
                time.sleep(2)
    
    if retry_count >= max_retries:
        print(f"  在 {max_retries} 次尝试后仍未成功，放弃获取")
    
    print()

def main():
    """主函数"""
    print("中国海关大豆进口数据获取工具 - 快速开始示例")
    print("=" * 60)
    print()
    
    # 运行所有示例
    examples = [
        example_basic_usage,
        example_ssl_usage,
        example_batch_processing,
        example_data_analysis,
        example_error_handling
    ]
    
    print("本示例将演示以下功能:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example.__doc__.strip()}")
    print()
    
    # 询问用户是否继续
    response = input("按回车键开始运行示例，或输入 'q' 退出: ").strip().lower()
    if response == 'q':
        print("已取消运行示例")
        return
    
    print()
    
    # 运行示例
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"示例运行出错: {e}")
            print("继续下一个示例...")
            print()
    
    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
    print()
    print("下一步建议:")
    print("1. 查看生成的CSV文件了解数据结构")
    print("2. 根据实际需求修改脚本参数")
    print("3. 查看详细的使用指南: USAGE_GUIDE.md")
    print("4. 运行测试脚本验证环境: python simple_test_customs_fetcher.py")
    print()

if __name__ == "__main__":
    main()