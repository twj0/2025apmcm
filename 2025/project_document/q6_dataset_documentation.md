# Q6综合面板数据集文档

## 概述
Q6综合面板数据集是整合了Q1-Q5问题关键指标的综合数据集，用于支持跨领域的综合分析。该数据集包含了2010年至2025年的数据，涵盖了大豆贸易、汽车产业、半导体行业、关税政策和宏观经济等多个维度。

## 数据集基本信息
- 文件路径: `data/processed/q6/q6_0_final_integrated_dataset.csv`
- 行数: 125行
- 列数: 75列
- 时间范围: 2010-2025年

## 数据结构

### 核心字段
- `year`: 年份（2010-2025）
- `net_weight_tons`: 净重（吨）
- `primary_value_usd`: 主要价值（美元）
- `quality_grade`: 质量等级
- `tariff_rate`: 关税率

### Q2汽车产业相关字段
- `brand`: 汽车品牌
- `total_sales`: 总销量
- `us_produced`: 美国生产数量
- `china_produced`: 中国生产数量
- `eu_produced`: 欧盟生产数量
- `row_produced`: 其他地区生产数量
- `production_location`: 主要生产地
- `us_produced_pct`: 美国生产占比
- `china_produced_pct`: 中国生产占比
- `eu_produced_pct`: 欧盟生产占比
- `row_produced_pct`: 其他地区生产占比
- `market_share_pct`: 市场份额百分比
- `total_light_vehicle_sales_million`: 轻型车辆总销量（百万）
- `total_light_vehicle_sales_index`: 轻型车辆总销量指数
- `industry_concentration_ratio`: 行业集中度比率
- `avg_vehicle_price_usd`: 平均车辆价格（美元）
- `production_cost_index`: 生产成本指数
- `rd_expenditure_pct`: 研发支出占比
- `export_intensity_pct`: 出口强度百分比
- `import_intensity_pct`: 进口强度百分比

### Q3半导体行业相关字段
- `us_chip_output_index`: 美国芯片产出指数
- `us_chip_output_billions`: 美国芯片产出（十亿美元）
- `global_chip_output_billions`: 全球芯片产出（十亿美元）
- `us_global_share_pct`: 美国全球份额百分比
- `global_chip_demand_index`: 全球芯片需求指数
- `china_import_dependence_pct`: 中国进口依赖百分比
- `r_and_d_intensity_pct`: 研发强度百分比
- `capex_billions`: 资本支出（十亿美元）
- `policy_support_index`: 政策支持指数
- `export_control_index`: 出口控制指数
- `supply_chain_risk_index`: 供应链风险指数
- `chips_subsidy_index`: 芯片补贴指数
- `export_control_index_policy`: 出口控制政策指数
- `reshoring_incentive_index`: 回流激励指数
- `rd_investment_billions`: 研发投资（十亿美元）
- `policy_uncertainty_index`: 政策不确定性指数
- `target_supply_chain_risk_score`: 目标供应链风险评分

### Q4关税收入相关字段
- `total_imports_usd`: 总进口额（美元）
- `total_tariff_revenue_usd`: 总关税收入（美元）
- `effective_tariff_rate`: 有效关税率
- `weighted_avg_tariff`: 加权平均关税
- `china_imports_usd`: 中国进口额（美元）
- `china_tariff_revenue_usd`: 中国关税收入（美元）
- `china_effective_tariff_rate`: 中国有效关税率
- `avg_tariff_rate`: 平均关税率
- `china_tariff_rate`: 中国关税率
- `row_tariff_rate`: 其他地区关税率
- `expected_revenue_billions`: 预期收入（十亿美元）
- `import_elasticity`: 进口弹性
- `target_tariff_revenue`: 目标关税收入
- `target_tariff_revenue_scenario`: 目标关税收入情景

### Q5宏观金融与制造业回流相关字段
- `gdp_growth`: GDP增长率
- `industrial_production`: 工业生产
- `unemployment_rate`: 失业率
- `cpi`: 消费者价格指数
- `real_gdp_growth`: 实际GDP增长率
- `industrial_production_index`: 工业生产指数
- `cpi_index`: CPI指数
- `core_inflation_rate`: 核心通胀率
- `federal_funds_rate`: 联邦基金利率
- `real_investment_growth`: 实际投资增长率
- `dollar_index`: 美元指数
- `treasury_yield_10y`: 10年期国债收益率
- `sp500_index`: 标普500指数
- `crypto_index`: 加密货币指数
- `manufacturing_va_share`: 制造业增加值份额
- `manufacturing_employment_share`: 制造业就业份额
- `reshoring_fdi_billions`: 制造业回流外商直接投资（十亿美元）
- `retaliation_index`: 报复指数
- `tariff_index`: 关税指数
- `china_tariff_index`: 中国关税指数
- `row_tariff_index`: 其他地区关税指数
- `china_tariff_coverage_pct`: 中国关税覆盖率百分比
- `z_manufacturing_va_share`: 制造业增加值份额Z值
- `z_manufacturing_employment_share`: 制造业就业份额Z值
- `z_reshoring_fdi_billions`: 制造业回流外商直接投资Z值
- `manufacturing_reshoring_index`: 制造业回流指数
- `target_manufacturing_reshoring_index`: 目标制造业回流指数

## 数据来源
- Q1: 大豆贸易数据 (data/processed/q1/q1_1.csv)
- Q2: 汽车产业数据 (data/processed/q2/q2_1.csv)
- Q3: 半导体行业数据 (data/processed/q3/q3_1.csv)
- Q4: 关税收入数据 (data/processed/q4/q4_1.csv)
- Q5: 宏观经济与制造业回流数据 (data/processed/q5/q5_1.csv)

## 数据处理说明
1. Q1数据通过将period字段转换为年份，并对年度数据进行聚合处理
2. Q2-Q5数据直接从对应的CSV文件中加载
3. 所有数据按年份进行合并，使用左连接方式保留所有年份数据
4. 数据类型进行了优化，数值字段转换为适当的数值类型
5. 数据按年份排序

## 注意事项
1. 部分字段可能存在缺失值，特别是在不同问题的数据时间范围不一致的情况下
2. 2025年的数据主要是预测数据，应谨慎使用
3. 某些字段可能在特定年份没有数据，显示为NaN或空值