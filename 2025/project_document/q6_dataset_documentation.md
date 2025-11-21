# Q6综合面板数据集文档

## 概述
Q6综合面板数据集整合了Q1-Q5各问题处理后的关键指标，构建了一个用于综合分析的面板数据集。数据集包含2015-2025年的年度数据，共11行45列。

## 数据结构
数据集以年份(year)为主键，包含以下三大类指标：

### 1. 半导体产业指标 (Q3)
- `us_chip_output_index`: 美国芯片产出指数 (2015=100)
- `us_chip_output_billions_x`: 美国芯片产出(十亿美元)
- `global_chip_output_billions`: 全球芯片产出(十亿美元)
- `us_global_share_pct`: 美国全球市场份额(%)
- `global_chip_demand_index`: 全球芯片需求指数 (2015=100)
- `china_import_dependence_pct`: 中国进口依赖度(%)
- `r_and_d_intensity_pct`: 研发强度(%)
- `capex_billions`: 资本支出(十亿美元)
- `policy_support_index`: 政策支持指数
- `export_control_index`: 出口管制指数
- `supply_chain_risk_index`: 供应链风险指数

### 2. 关税与贸易指标 (Q4)
- `segment`: 市场细分标识
- `import_proxy`: 进口代理指标
- `us_chip_output_billions_y`: 美国芯片产出(十亿美元) - 重复字段
- `total_supply`: 总供应量
- `self_sufficiency_pct`: 自给率(%)
- `china_import_proxy`: 中国进口代理指标
- `china_dependence_pct`: 中国依赖度(%)
- `supply_risk_index`: 供应风险指数
- `total_imports_usd`: 总进口额(美元)
- `total_tariff_revenue_usd`: 总关税收入(美元)
- `effective_tariff_rate`: 有效关税率(%)
- `weighted_avg_tariff`: 加权平均关税率(%)
- `china_imports_usd`: 对华进口额(美元)
- `china_tariff_revenue_usd`: 对华关税收入(美元)
- `china_effective_tariff_rate`: 对华有效关税率(%)

### 3. 宏观经济与金融指标 (Q5)
- `gdp_growth`: GDP增长率(%)
- `industrial_production`: 工业生产指数
- `unemployment_rate`: 失业率(%)
- `cpi`: 消费者价格指数
- `real_gdp_growth`: 实际GDP增长率(%)
- `industrial_production_index`: 工业生产指数
- `cpi_index`: CPI指数
- `core_inflation_rate`: 核心通胀率(%)
- `federal_funds_rate`: 联邦基金利率(%)
- `real_investment_growth`: 实际投资增长率(%)
- `dollar_index`: 美元指数
- `treasury_yield_10y`: 10年期国债收益率(%)
- `sp500_index`: 标普500指数
- `crypto_index`: 加密货币指数
- `manufacturing_va_share`: 制造业增加值占比(%)
- `manufacturing_employment_share`: 制造业就业占比(%)
- `reshoring_fdi_billions`: 制造业回流FDI(十亿美元)
- `retaliation_index`: 报复指数

## 数据处理方法
1. 时间对齐: 所有数据统一到年度频率，以年份为合并主键
2. 缺失值处理: 对于缺失数据，保持为空，未进行插值或填充
3. 数据类型优化: 尽可能将字符串类型转换为数值类型

## 数据来源
- Q1数据: 大豆贸易数据 (q1_0.csv, q1_1.csv)
- Q2数据: 汽车产业数据 (q2_0_us_auto_sales_by_brand.csv, q2_1_industry_indicators.csv)
- Q3数据: 半导体数据 (q3_0_us_semiconductor_output.csv, q3_security_metrics.csv)
- Q4数据: 关税收入数据 (q4_0_tariff_revenue_panel.csv)
- Q5数据: 宏观金融数据 (q5_0_macro_indicators.csv, q5_1_financial_indicators.csv, q5_2_reshoring_indicators.csv, q5_3_retaliation_index.csv)

## 数据质量
- **完整性**: 除2025年部分预测变量外，历史数据完整
- **一致性**: 所有数据均已转换为统一的度量单位
- **准确性**: 各源数据均经过验证，确保在合理范围内

## 使用注意事项
1. 2025年数据为预测值，使用时需注意其不确定性
2. 部分指标可能存在多重共线性，在建模时需谨慎处理
3. 建议在使用前对数据进行进一步的探索性分析