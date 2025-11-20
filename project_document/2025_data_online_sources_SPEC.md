# 2025 APMCM Problem C 外部数据联网与官方数据源 SPEC

**生成时间：2025-11-20 (UTC+08)**  
**项目路径：`2025APMCM/SPEC`**  
**适用范围：Q1–Q5 外部数据 `*_official` 构建与接入**

---

## 1. 目标与总体策略

本 SPEC 文档用于规范 2025 APMCM Problem C 中，除题目提供的 USITC Tariff Data 之外的**外部官方/权威数据获取方案**，并指导在 `2025/data/external/` 目录下新增 `*_official` 数据文件，以替换当前 `external_data.py` 中自动生成的 SAMPLE 数据。

总体策略：

- **不覆盖现有 SAMPLE 文件**，在 `data/external/` 中新增 `*_official` 文件。  
- 在 `external_data.py` 中调整逻辑：
  - 优先尝试读取 `*_official` 文件；
  - 如不存在或数据明显异常，再回退到示例数据生成逻辑（仅用于开发调试）。
- 数据来源优先级：
  - 官方统计数据库（海关总署、UN Comtrade、BEA、Census、FRED 等）；
  - 其次为国际组织或权威行业协会（WTO、World Bank、SIA 等）；
  - 智库/学术构造的指数仅用于难以官方化的软变量（如反制指数、回流案例数）。

---

## 2. Q1：中国大豆进口数据

### 2.1 目标文件

- `2025/data/external/china_imports_soybeans_official.csv`

### 2.2 推荐数据源

- **UN Comtrade / World Bank WITS 封装（主数据源）**  
  - 示例页面：
    - `https://wits.worldbank.org/trade/comtrade/en/country/CHN/year/2024/tradeflow/Imports/partner/ALL/product/120100`
  - 通用查询入口：
    - UN Comtrade+：`https://comtradeplus.un.org/TradeFlow`
  - 查询建议：
    - Reporter：China (CHN)
    - Partner：All countries
    - Commodity：HS 1201 / 120100（Soya beans）
    - Trade Flow：Imports
    - Frequency：Annual
    - Years：2015–2024

> 说明：WITS 是对 UN Comtrade 的友好封装，适合浏览器交互导出 CSV；UN Comtrade+ 则提供更灵活的高级查询。

### 2.3 字段设计（与 Q1 模型兼容）

统一处理导出 CSV 后，转换为以下列：

- `year`  
  - 从原始列 `Period` 或等价字段中提取年份。
- `exporter`  
  - 对应原始列 `Partner`，如 `Brazil`, `United States`, `Argentina` 等。
- `import_value_usd`  
  - 对应 `Trade Value (US$)`。
- `import_quantity_tonnes`  
  - 如原数据提供 `Netweight (kg)`，则换算为吨：`tonnes = kg / 1000`。
- `tariff_cn_on_exporter`  
  - 由另一个关税表（基于中国海关/WTO 税则等来源）构建：
    - 对不同来源国、不同年份大豆加征的附加关税或总关税水平；
    - 可单独整理为 `cn_tariff_soybeans_retaliation.csv`，再在 Q1 数据预处理阶段按 `year + exporter` merge 进入本表。

---

## 3. Q2：美国汽车数据（销量与行业指标）

### 3.1 目标文件

- `2025/data/external/us_auto_sales_by_brand_official.csv`  
  - 如果无法获取品牌级数据，可退而使用总量级数据，改名为 `us_auto_sales_aggregate_official.csv`。  
- `2025/data/external/us_auto_indicators_official.csv`

### 3.2 官方/权威数据源

#### 3.2.1 BEA – 国民经济核算与行业数据

- 入口：`https://www.bea.gov/data`
- 关键词：`motor vehicles`, `auto`, `personal consumption expenditures motor vehicles and parts`。
- 典型表格：
  - NIPA 表 2.5.5、2.4.5（Personal Consumption Expenditures by Type of Product）中含
    - “Motor vehicles and parts” 相关系列。
- 用途：
  - 构造年度宏观层面的汽车需求/产出代理变量，如：
    - PCE on motor vehicles and parts (Billion USD)；
    - 相关行业产出指数（若使用 Industry Accounts 数据）。

#### 3.2.2 Census – 零售/批发贸易调查

- 入口：`https://www.census.gov/econ/currentdata/`
- 关注行业：`Motor vehicle and parts dealers`（NAICS 441）。
- 指标：
  - 零售销售额（Retail Sales）；
  - 可按月/季导出，再聚合为年度值。

#### 3.2.3 FRED – 辅助时间序列（便于 CSV 下载）

- FRED 搜索：`https://fred.stlouisfed.org/search?st=motor%20vehicle`
- 常用系列示例（根据最终建模需要选取）：
  - Total Vehicle Sales (`TOTALSA`)
  - Light Weight Vehicle Sales (`LTOTALNSA`)
  - 其他机动车相关工业产出/价格指数。

> 注：严格品牌级销量多依赖商业数据库（Ward’s, MarkLines 等），非严格意义“官方”。如果题目允许，可以使用总量+市场份额估算品牌拆分，并在论文中明确说明限制。

### 3.3 字段设计

#### 3.3.1 `us_auto_indicators_official.csv`

建议列：

- `year`  
- `pce_motor_vehicles_billion_usd`  
  - 来自 BEA NIPA，个人消费支出中“机动车及零部件”。
- `retail_sales_motor_vehicles_billion_usd`  
  - 来自 Census，NAICS 441 年度零售销售额。
- `total_light_vehicle_sales_million_units`  
  - 如使用 FRED `TOTALSA` 或 `LTOTALNSA`，可从月度/季节数据聚合到年度。

#### 3.3.2 `us_auto_sales_by_brand_official.csv`（如有）

建议列：

- `year`  
- `brand`（Toyota, Ford, GM, Honda, etc.）  
- `sales_units`  
- `market_share_percent`（如可计算）

---

## 4. Q3：美国半导体产出与 CHIPS 相关信息

### 4.1 目标文件

- `2025/data/external/us_semiconductor_output_official.csv`  
- `2025/data/external/us_chip_policies_official.csv`

### 4.2 半导体产出（FRED/BEA）

#### 4.2.1 FRED – Sectoral Output（推荐）

- 系列：`IPUEN3344T300000000`  
  - 名称：Sectoral Output for Manufacturing: Semiconductor and Other Electronic Component Manufacturing (NAICS 3344)。
  - 页面：`https://fred.stlouisfed.org/series/IPUEN3344T300000000`
- 特点：
  - 季度指数，覆盖 1987–今，可导出 CSV；
  - 可对 2015–2024 进行年度平均或末期值计算，构造年度产出指数。

#### 4.2.2 FRED – 其他半导体相关系列（可选）

- 通过标签页：`https://fred.stlouisfed.org/tags/series?t=semiconductors` 搜索：
  - Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing 等。

### 4.3 字段设计：`us_semiconductor_output_official.csv`

建议列：

- `year`  
- `output_index_naics3344`  
  - 由 `IPUEN3344T300000000` 聚合而来，基期与原系列保持一致；
- 可选扩展：
  - `ppi_semiconductors`（来自 PPI 对应系列，反映成本/价格）。

### 4.4 CHIPS 法案与政策变量

#### 4.4.1 信息源

- NSF CHIPS 页面（官方政策与项目介绍）：
  - `https://www.nsf.gov/chips`
- 美国商务部/CHIPS Program Office 公告：
  - 通过搜索“CHIPS for America awards”、“CHIPS funding announcement”等获取具体项目与金额。
- 半导体行业协会（SIA）报告：
  - 例如《2024 State of the U.S. Semiconductor Industry》等年度报告，包含：
    - 新增晶圆厂投资计划；
    - 各类 CHIPS 激励的预期规模等。

#### 4.4.2 字段设计：`us_chip_policies_official.csv`

本表侧重于构造可用于回归/情景分析的“政策虚拟变量/强度变量”：

- `year`  
- `chips_law_in_effect`  
  - 0/1 标记 CHIPS and Science Act 是否已生效、是否进入主要实施阶段。  
- `announced_chips_funding_billion_usd`  
  - 当年已公开的 CHIPS 相关资金总额（粗略汇总自官方公告/报告）；
- `new_fab_investments_billion_usd`  
  - 当年新增美国本土晶圆厂投资规模估计值（可从 SIA 报告的项目列表中手工汇总）。

> 说明：该表并非“官方统一时间序列”，而是基于官方公告与行业报告手工构造的政策强度变量，需在论文中说明构造方法与数据来源。

---

## 5. Q5：美国宏观、金融与回流/反制变量

### 5.1 目标文件

- `2025/data/external/us_macro_official.csv`  
- `2025/data/external/us_financial_official.csv`  
- `2025/data/external/us_reshoring_official.csv`  
- `2025/data/external/retaliation_index_official.csv`

### 5.2 宏观指标（FRED）

#### 5.2.1 推荐系列

- 实际 GDP：`GDPC1`  
- CPI（全体城市居民）：`CPIAUCSL`  
- 失业率：`UNRATE`  
- 工业生产指数：`INDPRO`  
- 实际个人消费支出：`PCEC96`

#### 5.2.2 获取与聚合

- 访问对应 FRED 页面（例如 `https://fred.stlouisfed.org/series/GDPC1`），使用 `Download Data` 导出 CSV；
- 将季度/月度数据聚合为年度：
  - 年均（平均）或年末值，按模型需求选择；
- 统一到 2015–2024 年度面板。

#### 5.2.3 字段设计：`us_macro_official.csv`

- `year`  
- `real_gdp_billion_2012usd`（来自 GDPC1，按 BEA/FRED 标注单位换算）  
- `cpi_index_1982_84_100`（CPIAUCSL 年均）  
- `unemployment_rate_percent`（UNRATE 年均）  
- `industrial_production_index`（INDPRO 年均）  
- 可视需要增加：`real_pce_billion_2012usd` 等。

### 5.3 金融指标（FRED）

#### 5.3.1 推荐系列

- 联邦基金有效利率：`FEDFUNDS`  
- 10 年期国债收益率：`DGS10`  
- 标普 500 指数：`SP500` 或 `SP500TR`（如需要总回报）。

#### 5.3.2 字段设计：`us_financial_official.csv`

- `year`  
- `fed_funds_rate_percent`（FEDFUNDS 年均）  
- `treasury_10y_yield_percent`（DGS10 年均）  
- `sp500_index`（年均或年末值）。

### 5.4 回流（Reshoring）与制造业指标

回流/再工业化难以直接从单一官方数据库获得完整时间序列，建议采用**代理指标 + 案例统计**结合的方式。

#### 5.4.1 制造业代理指标（FRED）

- 制造业就业：`MANEMP`  
- 其他可选：制造业工时、加班、制造业产出指数等。

#### 5.4.2 回流案例统计（Reshoring Initiative 等）

- Reshoring Initiative 提供面向美国的回流与外资制造项目数据库/年度统计；
- 可从其年度报告中提取每年项目数量、预期岗位数等信息，用于构造回流强度变量。

#### 5.4.3 字段设计：`us_reshoring_official.csv`

- `year`  
- `manufacturing_employment_thousands`  
  - 来自 FRED `MANEMP`，按年均或年末值换算为千人。  
- `reshoring_projects_count`  
  - 来自 Reshoring Initiative 等报告的年度项目数量（手工录入）；
- 可选：
  - `announced_jobs_from_reshoring`（回流预计创造岗位数）。

### 5.5 中方反制措施/报复性关税指数

#### 5.5.1 数据来源与形式

- 学术论文与智库报告中，通常提供 2018 年后中方对美加征关税清单：
  - 涵盖 HS 级别、加征幅度、覆盖的贸易额等；
- 这些表格可以从报告 PDF 中导出，再在本项目中手动整理。

#### 5.5.2 指数构造思路

将离散关税清单转化为年度/季度“反制强度指数”：

- 指标可包括：
  - 覆盖的美对华出口额占总出口额的比例；
  - 平均加征税率水平；
  - 重大反制事件虚拟变量（如 2018/2019 年大规模加征）；
- 最终归一化到 0–1 或其他适合建模的尺度。

#### 5.5.3 字段设计：`retaliation_index_official.csv`

- `year`  
- `retaliation_index`  
  - 综合上述覆盖比例和加征幅度的指数化结果；
- 可附加：
  - `retaliatory_tariff_event`（0/1，是否有大规模新一轮反制关税）；
  - `coverage_share`（被征收报复性关税的美对华出口占比，0–1）。

> 说明：该指数用于在宏观/贸易模型中捕捉“中方反制强度”的时间变化，属于研究者自构指数，需要在论文方法部分详细说明。

---

## 6. 与项目代码的集成建议

### 6.1 `external_data.py` 读取优先级

对现有的 `ensure_q1_external_data()` 至 `ensure_q5_external_data()`，建议统一采用如下逻辑：

1. 检查对应的 `*_official` 文件是否存在且数据非空/非模板：
   - 例如：`china_imports_soybeans_official.csv`、`us_auto_indicators_official.csv` 等；
2. 若存在，则直接读取并进行必要的字段重命名/单位换算，返回 DataFrame/ndarray；
3. 若不存在或检测到为 SAMPLE 特征（全 0、小样本、含明显警告字段），则：
   - 记录 WARNING 日志；
   - 回退到当前仓库中已有的 SAMPLE 生成逻辑，以保证代码可以运行。

### 6.2 数据清洗与文档记录

- 在 `2025/src/data/` 或 `2025/src/utils/` 中增加对应的清洗函数：
  - 负责从原始下载文件（可暂存于 `data/external/raw/`）转换为本 SPEC 约定的 `*_official` 结构；
- 在 `paper/` 或本 `project_document/` 目录中记录：
  - 每个字段的经济含义、原始数据源链接、单位、时间聚合方法；
  - 特别是自构指数（如 CHIPS 强度、反制指数、回流强度）的构造方法与权重设定。

---

## 7. 实施步骤建议

1. **浏览器手动下载原始数据**：
   - 按本 SPEC 中的链接和关键词，从 UN Comtrade/WITS、FRED、BEA、Census 等处下载 CSV/Excel；
   - 从 SIA/NSF/CHIPS、Reshoring Initiative、相关智库/论文中获取 PDF 和附表。
2. **在仓库中落地原始文件**：
   - 建议先放入 `2025/data/external/raw/`，保持原始结构与命名；
3. **编写数据清洗脚本**：
   - 使用 Python/`pandas` 将原始数据转换为本 SPEC 约定的 `*_official.csv` 格式；
4. **修改 `external_data.py`**：
   - 按本 SPEC 的文件名和字段结构，优先读取官方数据；
5. **在建模与论文中引用**：
   - 所有使用到的外部官方数据在论文中注明：
     - 数据源、表/系列 ID、下载日期、处理方法；
     - 对于自构指数，还需给出构造公式与敏感性分析说明。
