# q1_1数据单位标准化报告

**处理时间**: 2025-11-21 20:23:21
**输入文件**: data/processed/q1/q1_1_supplemented.csv
**输出文件**: data/processed/q1/q1_1_standardized.csv

## 标准化概览

- 总记录数: 417
- 时间范围: 201001 - 202510
- 贸易伙伴: 3个国家

## 单位转换规则

1. **重量单位**: 千克 → 吨 (除以1000)
   - netWgt (千克) → net_weight_tons (吨)
   - 数量(万吨) → quantity_tons (吨) (乘以10000)

2. **价值单位**: 美元保持不变
   - primaryValue → primary_value_usd (美元)
   - 金额(亿美元) → value_usd (美元) (乘以1亿)

## 数据一致性验证

### 重量数据一致性
- net_weight_tons 与 quantity_tons 的最大差异: 0.00 吨
- 平均差异: 0.00 吨
- 一致性评级: 优秀

### 价值数据一致性  
- primary_value_usd 与 value_usd 的最大差异: 405000000.00 美元
- 平均差异: 971223.02 美元
- 一致性评级: 需要检查

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
