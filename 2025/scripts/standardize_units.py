import pandas as pd
import numpy as np

def standardize_units():
    """标准化数据单位到国际标准单位（吨，美元）"""
    
    # 读取补充关税后的数据
    df = pd.read_csv('data/processed/q1/q1_1_supplemented.csv')
    
    print('=== 单位标准化处理 ===')
    print(f'原始数据形状: {df.shape}')
    print(f'原始列名: {list(df.columns)}')
    
    # 创建新的标准化数据框
    standardized_df = df.copy()
    
    # 标准化重量单位：从千克转换为吨
    # netWgt列：千克 -> 吨 (1吨 = 1000千克)
    standardized_df['netWgt_tons'] = df['netWgt'] / 1000
    
    # 标准化价值单位：保持美元不变，但创建更清晰列名
    standardized_df['primaryValue_usd'] = df['primaryValue']
    
    # 验证现有单位列的准确性
    # 数量(万吨)列应该与netWgt转换后的吨数一致
    df['quantity_check'] = df['数量(万吨)'] * 10000  # 万吨 -> 吨
    
    # 检查netWgt与数量(万吨)的一致性
    weight_diff = abs(standardized_df['netWgt_tons'] - df['quantity_check'])
    max_diff = weight_diff.max()
    avg_diff = weight_diff.mean()
    
    print(f'\n=== 重量数据一致性检查 ===')
    print(f'netWgt(kg->吨) 与 数量(万吨->吨) 的最大差异: {max_diff:.2f} 吨')
    print(f'平均差异: {avg_diff:.2f} 吨')
    
    # 检查价值数据一致性
    # 金额(亿美元)列应该与primaryValue一致
    df['value_check'] = df['金额(亿美元)'] * 1e8  # 亿美元 -> 美元
    value_diff = abs(standardized_df['primaryValue_usd'] - df['value_check'])
    max_value_diff = value_diff.max()
    avg_value_diff = value_diff.mean()
    
    print(f'\n=== 价值数据一致性检查 ===')
    print(f'primaryValue 与 金额(亿美元) 的最大差异: {max_value_diff:.2f} 美元')
    print(f'平均差异: {avg_value_diff:.2f} 美元')
    
    # 创建标准化列
    standardized_df['quantity_tons'] = df['数量(万吨)'] * 10000  # 万吨 -> 吨
    standardized_df['value_usd'] = df['金额(亿美元)'] * 1e8  # 亿美元 -> 美元
    
    # 选择最终的标准化列
    final_columns = [
        'period', 'partnerDesc', 'netWgt_tons', 'primaryValue_usd', 
        'quantity_tons', 'value_usd', 'tariff_rate'
    ]
    
    final_df = standardized_df[final_columns]
    
    # 重命名列以更清晰地表示单位
    final_df.columns = [
        'period', 'partner_desc', 'net_weight_tons', 'primary_value_usd',
        'quantity_tons', 'value_usd', 'tariff_rate'
    ]
    
    # 保存标准化数据
    output_file = 'data/processed/q1/q1_1_standardized.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f'\n=== 标准化结果 ===')
    print(f'输出文件: {output_file}')
    print(f'最终数据形状: {final_df.shape}')
    print(f'标准化列名: {list(final_df.columns)}')
    
    # 生成标准化报告
    generate_standardization_report(final_df, max_diff, avg_diff, max_value_diff, avg_value_diff)
    
    return final_df

def generate_standardization_report(df, max_weight_diff, avg_weight_diff, max_value_diff, avg_value_diff):
    """生成单位标准化报告"""
    
    report_content = f"""# q1_1数据单位标准化报告

**处理时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**输入文件**: data/processed/q1/q1_1_supplemented.csv
**输出文件**: data/processed/q1/q1_1_standardized.csv

## 标准化概览

- 总记录数: {len(df)}
- 时间范围: {df['period'].min()} - {df['period'].max()}
- 贸易伙伴: {df['partner_desc'].nunique()}个国家

## 单位转换规则

1. **重量单位**: 千克 → 吨 (除以1000)
   - netWgt (千克) → net_weight_tons (吨)
   - 数量(万吨) → quantity_tons (吨) (乘以10000)

2. **价值单位**: 美元保持不变
   - primaryValue → primary_value_usd (美元)
   - 金额(亿美元) → value_usd (美元) (乘以1亿)

## 数据一致性验证

### 重量数据一致性
- net_weight_tons 与 quantity_tons 的最大差异: {max_weight_diff:.2f} 吨
- 平均差异: {avg_weight_diff:.2f} 吨
- 一致性评级: {'优秀' if max_weight_diff < 1 else '良好' if max_weight_diff < 10 else '需要检查'}

### 价值数据一致性  
- primary_value_usd 与 value_usd 的最大差异: {max_value_diff:.2f} 美元
- 平均差异: {avg_value_diff:.2f} 美元
- 一致性评级: {'优秀' if max_value_diff < 1 else '良好' if max_value_diff < 100 else '需要检查'}

## 标准化列说明

| 列名 | 含义 | 单位 | 数据来源 |
|------|------|------|----------|
| period | 时间期间 | YYYYMM | 原始数据 |
| partner_desc | 贸易伙伴 | 国家名称 | 原始数据 |
| net_weight_tons | 净重 | 吨 | netWgt转换 |
| primary_value_usd | 主要价值 | 美元 | primaryValue重命名 |
| quantity_tons | 数量 | 吨 | 数量(万吨)转换 |
| value_usd | 价值 | 美元 | 金额(亿美元)转换 |
| tariff_rate | 关税税率 | 小数形式 | 补充数据 |

## 质量标准

✅ **单位统一**: 所有重量数据统一为吨，价值数据统一为美元
✅ **格式规范**: 列名使用英文小写+下划线格式
✅ **精度保持**: 转换过程中保持原有精度
✅ **一致性检查**: 验证不同来源数据的一致性

---
*本报告由自动化脚本生成*
"""
    
    # 保存报告
    report_file = 'data/processed/q1/q1_1_standardized_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f'标准化报告已生成: {report_file}')

if __name__ == "__main__":
    standardized_df = standardize_units()
    
    # 显示数据样本
    print('\n=== 标准化数据样本 ===')
    print(standardized_df.head())
    
    print('\n=== 数据描述统计 ===')
    print(standardized_df.describe())