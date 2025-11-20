# 2025 APMCM Problem C 数据可用性评估报告

**生成时间：2025-11-20**  
**项目路径：`2025APMCM/SPEC`**

## 摘要

本报告基于对项目数据目录中24个CSV文件的全面验证，评估了原始数据和通过爬虫获取的官方数据的可用性。总体而言，我们成功获取了多个关键的官方数据集，大部分数据质量良好，可直接用于分析建模。

### 关键发现：

1. **成功获取9个官方数据集**：通过FRED API成功获取了美国宏观经济、金融和行业指标数据，包括CPI、失业率、工业生产指数、GDP、半导体产出指数等。
2. **数据质量整体良好**：24个文件中，22个被评为"ready_to_use"（可直接使用），1个需要预处理，1个不推荐使用。
3. **主要问题集中在结构完整性**：部分文件缺少预期列或唯一标识符，少数文件存在异常值。
4. **UN Comtrade数据获取失败**：中国大豆进口数据由于API限制未能成功获取，仍需手动处理。

---

## 一、数据获取成果

### 1.1 成功获取的官方数据

通过FRED API成功获取以下9个官方数据集：

| 数据集 | 文件名 | 数据来源 | 时间范围 | 质量评分 |
|--------|--------|----------|----------|----------|
| 消费者价格指数 | us_cpi_official.csv | FRED (CPIAUCSL) | 2015-2024 | 100 |
| 失业率 | us_unemployment_rate_official.csv | FRED (UNRATE) | 2015-2024 | 90 |
| 工业生产指数 | us_industrial_production_official.csv | FRED (INDPRO) | 2015-2024 | 100 |
| 实际GDP | us_real_gdp_official.csv | FRED (GDPC1) | 2015-2024 | 100 |
| 半导体产出指数 | us_semiconductor_output_index_official.csv | FRED (IPUEN3344T300000000) | 2015-2024 | 100 |
| 汽车零售销售 | us_motor_vehicle_retail_sales_official.csv | FRED | 2015-2024 | 100 |
| 轻型汽车销售总量 | us_total_light_vehicle_sales_official.csv | FRED (TOTALSA) | 2015-2024 | 100 |
| 联邦基金利率 | us_federal_funds_rate_official.csv | FRED (FEDFUNDS) | 2015-2024 | 90 |
| 10年期国债收益率 | us_treasury_10y_yield_official.csv | FRED (DGS10) | 2015-2024 | 100 |
| 标普500指数 | us_sp500_index_official.csv | FRED (SP500) | 2015-2024 | 95 |

### 1.2 获取失败的数据

1. **中国大豆进口数据**：尝试通过UN Comtrade API获取，但遇到"Comtrade response was not JSON (likely gateway redirect)"错误。此数据仍需手动下载和处理。

---

## 二、数据质量评估

### 2.1 总体评估

- **总计文件数**：24个
- **可直接使用**：22个 (91.7%)
- **需要预处理**：1个 (4.2%)
- **不推荐使用**：1个 (4.2%)

### 2.2 高质量数据（评分90-100）

以下22个文件数据质量良好，可直接用于分析：

1. **官方数据集**（9个）：
   - us_cpi_official.csv (100分)
   - us_industrial_production_official.csv (100分)
   - us_real_gdp_official.csv (100分)
   - us_semiconductor_output_index_official.csv (100分)
   - us_motor_vehicle_retail_sales_official.csv (100分)
   - us_total_light_vehicle_sales_official.csv (100分)
   - us_treasury_10y_yield_official.csv (100分)
   - us_sp500_index_official.csv (95分)
   - us_unemployment_rate_official.csv (90分)
   - us_federal_funds_rate_official.csv (90分)

2. **项目原始数据**（13个）：
   - q4_avg_tariff_by_year.csv (100分)
   - retaliation_index.csv (100分)
   - us_auto_indicators.csv (100分)
   - us_chip_policies.csv (100分)
   - us_financial.csv (100分)
   - us_reshoring.csv (100分)
   - us_semiconductor_output.csv (100分)
   - china_imports_soybeans.csv (85分)
   - hs_autos.csv (92分)
   - hs_semiconductors_segmented.csv (82分)
   - hs_soybeans.csv (82分)
   - us_auto_sales_by_brand.csv (90分)

### 2.3 需要改进的数据

1. **需要预处理**：
   - us_macro.csv (70分)：存在多个异常值，包括GDP增长率、工业生产和失业率数据。

2. **不推荐使用**：
   - hs_to_sector.csv (54分)：缺失数据率高(16.1%)，且存在异常值。

---

## 三、主要问题分析

### 3.1 结构完整性问题

5个文件存在结构完整性问题：

1. **china_imports_soybeans.csv**：缺少预期列"importer"和"hs_code"
2. **hs_autos.csv**：缺少唯一标识符或时间戳列
3. **hs_semiconductors_segmented.csv**：缺少唯一标识符或时间戳列
4. **hs_soybeans.csv**：缺少唯一标识符或时间戳列
5. **hs_to_sector.csv**：缺少唯一标识符或时间戳列

### 3.2 数据准确性问题

7个文件存在数据准确性问题，主要表现为异常值：

1. **hs_semiconductors_segmented.csv**：hs_code列14.29%异常值
2. **hs_soybeans.csv**：hs_code列7.69%异常值
3. **hs_to_sector.csv**：hs_code列1.69%异常值
4. **us_macro.csv**：gdp_growth列18.18%异常值，industrial_production列9.09%异常值，unemployment_rate列9.09%异常值
5. **us_federal_funds_rate_official.csv**：value列20.00%异常值
6. **us_sp500_index_official.csv**：value列10.00%异常值
7. **us_unemployment_rate_official.csv**：value列10.00%异常值

### 3.3 缺失数据问题

2个文件存在缺失数据：

1. **hs_to_sector.csv**：16.1%缺失数据
2. **us_auto_sales_by_brand.csv**：少量缺失数据

---

## 四、可用性结论与建议

### 4.1 可用性结论

1. **高可用性数据**：已获取的9个FRED官方数据集质量良好，可直接用于Q2、Q3和Q5的分析建模。
2. **部分可用数据**：项目原始数据中大部分可用，但需注意结构完整性和异常值问题。
3. **关键数据缺失**：中国大豆进口官方数据未能获取，这是Q1分析的关键数据，需要优先解决。

### 4.2 建议

#### 即时行动

1. **优先获取中国大豆进口数据**：
   - 尝试通过World Bank WITS获取
   - 或手动从中国海关总署网站下载并处理
   - 确保数据包含年份、出口国、进口值和数量等关键字段

2. **修复结构完整性问题**：
   - 为china_imports_soybeans.csv添加缺少的importer和hs_code列
   - 为HS编码相关文件添加唯一标识符

#### 短期改进

1. **异常值处理**：
   - 检查并验证us_macro.csv中的异常值，确认是否为真实数据波动
   - 验证官方数据中的异常值，特别是2020年疫情前后的数据波动

2. **数据标准化**：
   - 统一日期格式和列名规范
   - 建立数据验证检查点
   - 创建数据清洗脚本

#### 长期策略

1. **建立自动化数据质量监控系统**：
   - 定期验证数据完整性、准确性和一致性
   - 设置数据质量阈值和警报机制

2. **制定数据收集和存储标准**：
   - 建立统一的数据格式和命名规范
   - 实施数据版本控制

3. **定期数据质量审计**：
   - 建立定期数据质量评估流程
   - 记录数据来源和处理历史

---

## 五、各问题数据集可用性评估

### 5.1 Q1：中国大豆进口数据

- **当前状态**：有示例数据，但缺少官方数据
- **可用性**：低（需优先获取官方数据）
- **建议**：立即获取官方数据，或至少确保示例数据结构完整

### 5.2 Q2：美国汽车数据

- **当前状态**：有示例数据和官方汽车销售数据
- **可用性**：高
- **建议**：可使用官方汽车销售数据，结合示例数据进行品牌细分

### 5.3 Q3：美国半导体产出数据

- **当前状态**：有示例数据和官方半导体产出指数
- **可用性**：高
- **建议**：优先使用官方半导体产出指数数据

### 5.4 Q4：平均关税数据

- **当前状态**：有关税数据
- **可用性**：高
- **建议**：可直接使用现有数据

### 5.5 Q5：美国宏观与金融数据

- **当前状态**：有示例数据和多个官方宏观金融数据
- **可用性**：高
- **建议**：优先使用官方数据，替换示例数据

---

## 六、结论

总体而言，项目数据可用性良好，特别是通过FRED API获取的官方数据质量高，可直接用于分析建模。主要缺口是中国大豆进口的官方数据，这是Q1分析的关键，需要优先解决。其他数据问题多为结构完整性和异常值问题，可通过数据清洗和预处理解决。

建议优先获取中国大豆进口数据，然后进行数据清洗和预处理，以确保所有分析模型基于高质量数据运行。