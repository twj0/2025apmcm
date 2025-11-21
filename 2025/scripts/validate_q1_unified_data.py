"""
Q1数据单位统一验证脚本
验证统一后的Q1数据文件的单位一致性
"""

import pandas as pd
import numpy as np

def validate_q1_unified_data():
    """验证Q1统一数据文件的单位一致性"""
    
    print("=== Q1数据单位统一验证 ===\n")
    
    # 读取统一数据文件
    try:
        df = pd.read_csv('d:/Mathematical Modeling/2025APMCM/SPEC/2025/data/processed/q1/q1_unified_data.csv', 
                        comment='#', skip_blank_lines=True)
        print(f"✓ 成功读取统一数据文件，共{len(df)}条记录")
    except Exception as e:
        print(f"✗ 读取统一数据文件失败: {e}")
        return False
    
    # 1. 验证字段名称
    expected_columns = ['period', 'exporter', 'quantity_10k_tonnes', 'value_100m_usd', 
                         'unit_price_usd_per_ton', 'tariff_rate', 'data_source']
    actual_columns = list(df.columns)
    
    if actual_columns == expected_columns:
        print("✓ 字段名称验证通过")
    else:
        print(f"✗ 字段名称不匹配")
        print(f"  期望: {expected_columns}")
        print(f"  实际: {actual_columns}")
        return False
    
    # 2. 验证数据范围合理性
    print("\n=== 数据范围验证 ===")
    
    # 转换数据类型
    numeric_columns = ['quantity_10k_tonnes', 'value_100m_usd', 'unit_price_usd_per_ton', 'tariff_rate']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 数量验证 (万吨)
    quantity_min = df['quantity_10k_tonnes'].min()
    quantity_max = df['quantity_10k_tonnes'].max()
    print(f"数量范围: {quantity_min:.2f} - {quantity_max:.2f} 万吨")
    
    if quantity_min < 0:
        print("✗ 发现负数量，数据异常")
        return False
    elif quantity_max > 10000:  # 超过10亿吨可能异常
        print("⚠  警告: 最大数量超过1000万吨，请检查")
    else:
        print("✓ 数量范围合理")
    
    # 金额验证 (亿美元)
    value_min = df['value_100m_usd'].min()
    value_max = df['value_100m_usd'].max()
    print(f"金额范围: {value_min:.2f} - {value_max:.2f} 亿美元")
    
    if value_min < 0:
        print("✗ 发现负金额，数据异常")
        return False
    elif value_max > 500:  # 超过500亿美元可能异常
        print("⚠  警告: 最大金额超过500亿美元，请检查")
    else:
        print("✓ 金额范围合理")
    
    # 价格验证 (美元/吨)
    price_min = df['unit_price_usd_per_ton'].min()
    price_max = df['unit_price_usd_per_ton'].max()
    print(f"价格范围: {price_min:.2f} - {price_max:.2f} 美元/吨")
    
    if price_min < 100 or price_max > 2000:  # 大豆价格通常在100-2000美元/吨
        print("⚠  警告: 价格超出正常范围(100-2000美元/吨)")
    else:
        print("✓ 价格范围合理")
    
    # 关税验证
    tariff_min = df['tariff_rate'].min()
    tariff_max = df['tariff_rate'].max()
    print(f"关税范围: {tariff_min:.3f} - {tariff_max:.3f}")
    
    if tariff_min < 0 or tariff_max > 1:
        print("✗ 关税超出0-100%范围")
        return False
    else:
        print("✓ 关税范围合理")
    
    # 3. 验证数据一致性
    print("\n=== 数据一致性验证 ===")
    
    # 计算验证价格 = 金额/数量
    df['calculated_price'] = (df['value_100m_usd'] * 1e8) / (df['quantity_10k_tonnes'] * 1e4)
    
    # 比较计算价格与给定价格
    price_diff = abs(df['unit_price_usd_per_ton'] - df['calculated_price'])
    max_diff = price_diff.max()
    avg_diff = price_diff.mean()
    
    print(f"价格差异: 最大{max_diff:.2f}美元/吨, 平均{avg_diff:.2f}美元/吨")
    
    if max_diff > 50:  # 超过50美元/吨的差异可能有问题
        print("⚠  警告: 价格一致性偏差较大")
        # 显示差异较大的记录
        large_diff = df[price_diff > 50][['period', 'exporter', 'unit_price_usd_per_ton', 'calculated_price', 'quantity_10k_tonnes', 'value_100m_usd']].head()
        if len(large_diff) > 0:
            print("差异较大的前5条记录:")
            print(large_diff.to_string())
    else:
        print("✓ 价格一致性良好")
    
    # 4. 验证月度与年度数据重叠期
    print("\n=== 重叠期验证 (2015-2019) ===")
    
    # 提取年度数据的2015-2019年部分
    annual_data = df[df['data_source'] == 'annual'].copy()
    annual_overlap = annual_data[annual_data['period'].astype(str).str[:4].isin(['2015', '2016', '2017', '2018', '2019'])]
    
    if len(annual_overlap) > 0:
        print(f"找到{len(annual_overlap)}条年度重叠数据")
        
        # 按年份和出口商分组验证
        for year in ['2015', '2016', '2017', '2018', '2019']:
            annual_year = annual_overlap[annual_overlap['period'].astype(str).str[:4] == year]
            if len(annual_year) > 0:
                print(f"\n{year}年度数据概览:")
                for _, row in annual_year.iterrows():
                    print(f"  {row['exporter']}: {row['quantity_10k_tonnes']:.1f}万吨, {row['value_100m_usd']:.1f}亿美元")
    else:
        print("未找到重叠期数据")
    
    # 5. 数据完整性统计
    print("\n=== 数据完整性统计 ===")
    
    total_records = len(df)
    complete_records = len(df.dropna())
    missing_data = total_records - complete_records
    
    print(f"总记录数: {total_records}")
    print(f"完整记录数: {complete_records}")
    print(f"缺失数据记录: {missing_data}")
    
    if missing_data > 0:
        print("缺失数据字段统计:")
        missing_counts = df.isnull().sum()
        for field, count in missing_counts.items():
            if count > 0:
                print(f"  {field}: {count}")
    else:
        print("✓ 无缺失数据")
    
    # 6. 出口商统计
    print("\n=== 出口商统计 ===")
    exporter_stats = df['exporter'].value_counts()
    print("各出口商记录数:")
    for exporter, count in exporter_stats.items():
        print(f"  {exporter}: {count}")
    
    # 7. 数据源统计
    print("\n=== 数据源统计 ===")
    source_stats = df['data_source'].value_counts()
    print("各数据源记录数:")
    for source, count in source_stats.items():
        print(f"  {source}: {count}")
    
    print("\n=== 验证完成 ===")
    return True

if __name__ == "__main__":
    validate_q1_unified_data()