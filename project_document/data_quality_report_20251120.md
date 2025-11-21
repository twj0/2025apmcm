# 数据爬虫检查与质量评估报告

**生成时间**: 2025-11-20 20:00  
**检查范围**: `2025/src/utils/data_fetch.py`, `2025/src/utils/worldbank_wits.py`, `2025/data/external/*.csv`

---

## 执行摘要

已完成对两个数据获取脚本和所有外部数据文件的全面检查。发现并修复了代码重复定义问题，改进了数据清洗逻辑，创建了数据整合和质量检查工具。

**关键发现**:
- ✅ FRED数据获取成功，10个官方指标完整覆盖2015-2024年
- ❌ UN Comtrade API访问失败，Q1大豆数据获取受阻
- ⚠️ World Bank数据格式问题，存储了嵌套字典字符串
- ✅ 已创建替代方案和手工数据处理工具

---

## 1. 代码问题与修复

### 1.1 `data_fetch.py` - 重复定义问题 ✅ 已修复

**问题**: 在第357行和第622行存在完全重复的`DEFAULT_DATASETS`定义，包括辅助函数`_render_dataset_catalog`、`_select_dataset_names`和`run_cli`。

**影响**: 
- 代码冗余，难以维护
- 可能导致更新不同步
- 增加了约200行重复代码

**修复**: 删除了第622-884行的重复定义，保留第357-619行的原始定义。

### 1.2 `worldbank_wits.py` - 数据清洗改进 ✅ 已完成

**问题**: World Bank API返回的数据包含嵌套字典，CSV输出格式不佳：

```csv
indicator,country,value
"{'id': 'TM.TAX.MANF.SM.AR.ZS', 'value': 'Tariff rate...'}","{'id': 'CN', 'value': 'China'}",5.21
```

**改进**: 
- 添加了`clean_output`参数（默认为True）
- 自动提取字典中的`'value'`和`'id'`字段
- 支持解析字符串形式的字典（使用`ast.literal_eval`）
- 生成更清晰的CSV列名

**效果**: 现在输出为：
```csv
indicator,country,value
Tariff rate applied simple mean manufactured products (%),China,5.21
```

---

## 2. 数据获取状态

### 2.1 成功获取的数据 ✅

#### FRED官方数据（10个指标，2015-2024年）

| 数据集 | Series ID | 行数 | 质量 | 用途 |
|-------|-----------|------|------|------|
| US Real GDP | GDPC1 | 10 | ✅ EXCELLENT | Q5 宏观分析 |
| US CPI | CPIAUCSL | 10 | ✅ EXCELLENT | Q5 通胀分析 |
| US Unemployment Rate | UNRATE | 10 | ✅ EXCELLENT | Q5 劳动力市场 |
| US Industrial Production | INDPRO | 10 | ✅ EXCELLENT | Q5 制造业指标 |
| US Federal Funds Rate | FEDFUNDS | 10 | ✅ EXCELLENT | Q5 货币政策 |
| US Treasury 10Y Yield | DGS10 | 10 | ✅ EXCELLENT | Q5 金融市场 |
| US S&P 500 Index | SP500 | 10 | ✓ GOOD (1缺失值) | Q5 股市指标 |
| US Semiconductor Output | IPUEN3344T300000000 | 10 | ✅ EXCELLENT | Q3 半导体产出 |
| US Light Vehicle Sales | TOTALSA | 10 | ✅ EXCELLENT | Q2 汽车销量 |
| US Motor Vehicle Retail | MRTSSM441USN | 10 | ✅ EXCELLENT | Q2 汽车零售 |

**总体评估**: FRED数据质量优秀，时间序列完整，格式统一。

#### 已整合的宏观数据文件

已生成 `us_macro_consolidated.csv`，包含7个关键宏观指标的宽表格式：

```csv
year,gdp_real,cpi,unemployment,industrial_production,fed_funds_rate,treasury_10y,sp500
2015,18799.622,237.017,5.283333333,101.61825,0.135833333,2.1358,2061.072
2016,19141.672,240.007,4.875,101.69583333,0.397083333,1.8350,2094.651
...
```

### 2.2 获取失败的数据 ❌

#### Q1: 中国大豆进口数据（UN Comtrade）

**失败详情**（来自`failed_downloads.jsonl`）:
```json
{
  "dataset": "q1_china_soybean_imports",
  "error": "non_json_response",
  "snippet": "<!doctype html>...",
  "status": 200,
  "content_type": "text/html"
}
```

**分析**: 
- UN Comtrade API返回HTML页面而非JSON
- 可能原因：API密钥无效、需要注册、或API访问策略变更
- 环境变量`UN_COMTRADE_API_KEY`可能未设置或已过期

**替代方案**: ✅ 已实现
1. 手动下载工具: `china_soybeans_manual.py`
   - `template` - 生成数据录入模板
   - `process` - 处理GACC导出的CSV
   - `validate` - 验证数据完整性

2. 数据来源建议：
   - 中国海关总署（GACC）: http://www.customs.gov.cn/
   - UN Comtrade网页界面: https://comtradeplus.un.org/
   - World Bank WITS: https://wits.worldbank.org/

### 2.3 存在质量问题的数据 ⚠️

#### World Bank关税数据

**文件**: `wb_tariff_mean_china_2015_2024.csv`

**问题**:
- 缺失值：22个（27.5%）
- 2023-2024年数据缺失
- 列格式包含字典字符串

**建议**: 
- 使用改进的`worldbank_wits.py`重新获取
- 或使用USITC关税数据库作为替代

#### HS分类映射表

**文件**: `hs_to_sector.csv`

**问题**:
- 缺失值：38个（16.1%）
- 部分HS代码未映射到行业

**建议**: 补充缺失的HS-行业映射

---

## 3. 新创建的工具

### 3.1 数据整合工具 - `data_consolidate.py`

提供三个核心功能：

#### 功能1: 整合FRED数据
```bash
uv run python 2025/src/utils/data_consolidate.py consolidate --data-dir 2025/data/external
```

- 合并所有`*_official.csv`文件
- 支持长格式或宽格式输出（--wide）
- 自动提取指标名称

#### 功能2: 合并宏观数据
```bash
uv run python 2025/src/utils/data_consolidate.py merge-macro --data-dir 2025/data/external
```

- 生成宽表格式的综合宏观数据集
- 包含GDP、CPI、失业率、工业生产、利率等
- 适用于Q5经济影响分析

#### 功能3: 数据质量检查
```bash
uv run python 2025/src/utils/data_consolidate.py check --data-dir 2025/data/external
```

- 检查25个CSV文件
- 报告缺失值、重复行、时间覆盖范围
- 自动评级：EXCELLENT / GOOD / NEEDS REVIEW

### 3.2 中国大豆数据工具 - `china_soybeans_manual.py`

针对UN Comtrade API失败的替代方案：

#### 功能1: 创建数据模板
```bash
uv run python 2025/src/utils/china_soybeans_manual.py template --output 2025/data/raw/soybeans_template.csv
```

#### 功能2: 处理GACC数据
```bash
uv run python 2025/src/utils/china_soybeans_manual.py process <input_csv> --output 2025/data/external/china_imports_soybeans_official.csv
```

- 自动识别列名（支持中英文）
- 标准化国家名称（US/Brazil/Argentina）
- 计算关税税率（考虑贸易战期间的报复性关税）

#### 功能3: 验证数据完整性
```bash
uv run python 2025/src/utils/china_soybeans_manual.py validate <data_csv>
```

- 检查年份覆盖（2015-2024）
- 验证出口国完整性（US/Brazil/Argentina）
- 识别异常值和缺失数据

---

## 4. 当前数据准备度评估

基于实际检查结果，更新数据准备度评分：

### 4.1 按问题分类

| 问题 | 官方数据获取 | 数据完整性 | 准备度 | 状态 |
|------|-------------|-----------|--------|------|
| **Q1**: 大豆贸易 | ❌ 0% (UN Comtrade失败) | 示例数据可用 | 30/100 | 🔴 BLOCKED |
| **Q2**: 汽车产业 | ✅ 50% (总量数据) | 缺品牌细分 | 55/100 | 🟡 PARTIAL |
| **Q3**: 半导体 | ✅ 100% (FRED完整) | 完整 | 85/100 | 🟢 READY |
| **Q4**: 关税收入 | ✅ 100% (USITC数据) | 需计算处理 | 70/100 | 🟡 PARTIAL |
| **Q5**: 宏观影响 | ✅ 100% (FRED完整) | 完整+已整合 | 95/100 | 🟢 READY |

**整体准备度**: **67/100** (从60分提升)

**提升原因**:
- Q5数据已整合成可用格式 (+5分)
- 创建了Q1替代方案工具 (+2分)

### 4.2 优先级排序（P0级别任务）

#### 🔥 立即执行

1. **Q1大豆数据** - 最高优先级
   - **方案A**: 手动从GACC下载并使用`china_soybeans_manual.py`处理
   - **方案B**: 从UN Comtrade网页界面手动下载
   - **方案C**: 使用World Bank WITS bulk download
   - **预计时间**: 2-4小时（含数据查找、下载、清洗）

2. **Q4关税收入数据**
   - 基于USITC General_Import_Charges计算年度关税收入
   - 使用tariff_database_YYYY.csv计算平均税率
   - **预计时间**: 1-2小时

#### 🟡 重要但非紧急

3. **Q2品牌级汽车数据**
   - 总量数据已有，品牌细分可用市场份额估算
   - 可考虑标注为"基于市场份额估算"
   - **预计时间**: 2-3小时

4. **World Bank数据重新获取**
   - 使用改进的`worldbank_wits.py`
   - 填补2023-2024年缺失数据
   - **预计时间**: 30分钟

---

## 5. 数据文件清单

### 5.1 官方数据（*_official.csv）

✅ **完整且可用** (10个文件):
- `us_real_gdp_official.csv` - 2015-2024年实际GDP
- `us_cpi_official.csv` - 2015-2024年CPI
- `us_unemployment_rate_official.csv` - 2015-2024年失业率
- `us_industrial_production_official.csv` - 2015-2024年工业生产指数
- `us_federal_funds_rate_official.csv` - 2015-2024年联邦基金利率
- `us_treasury_10y_yield_official.csv` - 2015-2024年10年期国债收益率
- `us_sp500_index_official.csv` - 2015-2024年标普500指数
- `us_semiconductor_output_index_official.csv` - 2015-2024年半导体产出指数
- `us_total_light_vehicle_sales_official.csv` - 2015-2024年轻型车销量
- `us_motor_vehicle_retail_sales_official.csv` - 2015-2024年汽车零售额

### 5.2 示例数据（需替换）

⚠️ **非官方，仅供结构测试** (9个文件):
- `china_imports_soybeans.csv` - ❌ 需要官方数据
- `us_auto_sales_by_brand.csv` - ⚠️ 需要品牌细分数据
- `us_auto_indicators.csv` - ⚠️ 可用但建议验证
- `us_semiconductor_output.csv` - ✅ 已有官方版本
- `us_chip_policies.csv` - ⚠️ 需验证政策时间线
- `us_macro.csv` - ✅ 已生成整合版本
- `us_financial.csv` - ✅ 可用官方FRED数据
- `us_reshoring.csv` - ⚠️ 需要官方制造业数据
- `retaliation_index.csv` - ⚠️ 需基于官方政策文件

### 5.3 配置和映射数据

✅ **完整** (5个文件):
- `hs_soybeans.csv` - HS代码映射（大豆）
- `hs_autos.csv` - HS代码映射（汽车）
- `hs_semiconductors_segmented.csv` - HS代码映射（半导体，含分段）
- `hs_to_sector.csv` - ⚠️ 16%缺失，需补充
- `q4_avg_tariff_by_year.csv` - 平均关税（需验证计算方法）

### 5.4 新生成的整合数据

✅ **刚创建** (1个文件):
- `us_macro_consolidated.csv` - 7个宏观指标的宽表格式

---

## 6. 使用建议

### 6.1 对于建模团队

**立即可用的数据**:
- Q3半导体分析: 使用`us_semiconductor_output_index_official.csv`
- Q5宏观影响: 使用`us_macro_consolidated.csv`
- Q2汽车总量: 使用`us_*_vehicle_*_official.csv`

**需要手动补充的数据**:
- Q1大豆: 使用`china_soybeans_manual.py`工具处理GACC数据
- Q4关税: 基于USITC数据计算（脚本待开发）

### 6.2 数据更新流程

1. **检查数据质量**:
   ```bash
   uv run python 2025/src/utils/data_consolidate.py check
   ```

2. **重新获取FRED数据**（如需更新）:
   ```bash
   uv run python 2025/src/utils/data_fetch.py --groups Q5
   ```

3. **整合宏观数据**:
   ```bash
   uv run python 2025/src/utils/data_consolidate.py merge-macro
   ```

4. **验证大豆数据**（在手动输入后）:
   ```bash
   uv run python 2025/src/utils/china_soybeans_manual.py validate 2025/data/external/china_imports_soybeans_official.csv
   ```

### 6.3 API密钥配置

**FRED API** (已工作):
```bash
# 设置环境变量或在.env文件中配置
FRED_API_KEY=your_key_here
```

**UN Comtrade API** (当前失败，待修复):
```bash
UN_COMTRADE_API_KEY=your_key_here
```

注册地址: https://comtradeplus.un.org/

---

## 7. 下一步行动计划

### 本周必须完成（P0）

- [ ] **获取中国大豆进口官方数据**
  - 从GACC下载2015-2024年按来源国分解的大豆进口数据
  - 使用`china_soybeans_manual.py process`处理
  - 验证数据完整性

- [ ] **计算Q4关税收入数据**
  - 开发脚本从USITC General_Import_Charges提取年度关税收入
  - 使用tariff_database计算有效平均税率
  - 生成`q4_tariff_revenue_official.csv`

### 后续优化（P1-P2）

- [ ] 补充Q2品牌级汽车数据或说明估算方法
- [ ] 重新获取World Bank关税数据（填补2023-2024）
- [ ] 补充`hs_to_sector.csv`缺失映射
- [ ] 验证并更新Q3、Q5的政策和回流数据
- [ ] 创建数据来源元数据文档（记录所有数据源URL和获取日期）

---

## 8. 结论

### 8.1 主要成就

1. ✅ **修复了代码质量问题**: 删除重复定义，改进数据清洗
2. ✅ **建立了完整的FRED数据集**: 10个官方指标，质量优秀
3. ✅ **创建了数据整合工具**: 自动合并和质量检查
4. ✅ **开发了替代方案**: 应对UN Comtrade API失败

### 8.2 关键差距

1. ❌ **Q1大豆数据缺失**: UN Comtrade API不可用，需手动获取
2. ⚠️ **Q2品牌数据不完整**: 仅有总量，缺少细分
3. ⚠️ **部分示例数据未替换**: 仍有9个文件使用模拟数据

### 8.3 整体评估

**当前状态**: 🟡 **部分就绪 (67/100)**

**可进行的分析**:
- Q3半导体产业影响分析 - 完全就绪
- Q5宏观经济影响评估 - 完全就绪
- Q2汽车产业总体分析 - 部分就绪（缺品牌细分）
- Q4关税收入预测 - 数据可用但需处理
- Q1大豆贸易分析 - 受阻（需优先获取数据）

**建议策略**:
1. 先推进Q3和Q5分析（数据完备）
2. 并行获取Q1大豆数据（最高优先级）
3. Q2和Q4使用现有数据开始建模，后续用官方数据验证

---

**报告生成**: 2025-11-20 20:00  
**检查工具**: `data_consolidate.py`, `china_soybeans_manual.py`  
**数据覆盖**: 2025/data/external/ (25个CSV文件)
