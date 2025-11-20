# 2025 APMCM Problem C 数据现状与联网获取计划报告

**生成时间：2025-11-20 12:42 (UTC+08)**  
**项目路径：`2025APMCM/SPEC`**

## 摘要

- **Tariff Data（USITC）**：已确认为 2025-08-07 从 USITC DataWeb 官方导出的年度数据，涵盖 2020–2025，且关税数据库 `tariff_data_2015–2025` 已由你运行脚本转换为 CSV，结构完整，可直接用于 Q1–Q5 分析。  
- **外部情景/宏观/行业数据**：当前大部分为 `external_data.py` 自动生成的示例数据（SAMPLE），需要按题意用官方数据替换，并通过新增 `*_official` 文件实现。  
- **中国大豆进口、美国汽车、半导体、CHIPS 法案、FRED 宏观、回流指标、中方反制措施** 等关键数据源已初步锁定对应的官方网站入口，但尚未完成批量抓取与清洗。  
- 建议在不覆盖原 CSV 的前提下，在 `2025/data/external/` 中新增 `*_official` 系列文件，并在 `external_data.py` 中优先读取官方文件，从而保证主分析流程基于真实数据运行。

---

## 一、当前数据与代码结构概览

### 1. Tariff Data（USITC 官方关税与贸易数据）

**路径**：`2025/problems/Tariff Data/`

**核心文件：**

- `DataWeb-Query-Import__General_Import_Charges.csv`
- `DataWeb-Query-Export__FAS_Value.csv`
- `DataWeb-Query-Import__Query_Parameters.csv`
- `DataWeb-Query-Export__Query_Parameters.csv`
- `tariff_data_2015/…/tariff_database_2015.csv`
- `…`
- `tariff_data_2025/tariff_database_2025.csv`

**Query_Parameters 关键信息：**

- Download Date：2025-08-07（Import 11:00 AM / Export 10:56 AM）
- Trade Flow：General Imports / Total Exports  
- Classification System：HTS Items  
- Data To Report：General Import Charges / FAS Value  
- Years：2020, 2021, 2022, 2023, 2024, 2025  
- Country：Use All Countries, Break Out Countries  
- Commodity：Use All Commodities，Aggregation Level = 2（HTS2）

**Tariff schedule 子目录：**

你已运行 `convert_xlsx_to_csv.py`，完成以下转化：

- `tariff_data_2015/tariff_database_2015.csv`
- `tariff_data_2016/tariff_database_2016.csv`
- `tariff_data_2017/tariff_database_2017.csv`
- `tariff_data_2018/tariff_database_2018.csv`
- `tariff_data_2019/2019_Tariff_Database_v11.csv`
- `tariff_data_2020/tariff_database_202010.csv`
- `tariff_data_2021/tariff database_202106.csv`
- `tariff_data_2022/tariff database_202207.csv`
- `tariff_data_2023/tariff database_202307.csv`
- `tariff_data_2024/tariff_database_202405.csv`
- `tariff_data_2025/tariff_database_2025.csv`

**结论（P0: Tariff Data 验证）：**

- 数据为 **USITC DataWeb 官方导出**，带有完整的查询参数记录，满足“官方、可追溯”的要求。  
- 时间覆盖 2020–2025，下载时间 2025-08-07，时效性对题目而言是“最新”的。  
- 关税数据库 2015–2025 年的 CSV 已生成，可支持更细粒度的有效税率计算、HS 映射等分析。

### 2. 项目中的外部数据加载逻辑

**配置文件**：`2025/src/utils/config.py`

- `DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"`
- `TARIFF_DATA_DIR = PROJECT_ROOT / "problems" / "Tariff Data"`

**外部数据工具**：`2025/src/utils/external_data.py`

- `ensure_q1_external_data()`：生成或读取 `china_imports_soybeans.csv`
- `ensure_q2_external_data()`：生成或读取 `us_auto_sales_by_brand.csv`、`us_auto_indicators.csv`
- `ensure_q3_external_data()`：生成或读取 `us_semiconductor_output.csv`、`us_chip_policies.csv`
- `ensure_q4_external_data()`：生成或读取 Q4 平均关税与弹性参数等 JSON/CSV
- `ensure_q5_external_data()`：生成或读取 `us_macro.csv`、`us_financial.csv`、`us_reshoring.csv`、`retaliation_index.csv`

**重要现状：**

- 以上 `ensure_*` 函数在 **检测到文件不存在或数据全为 0/模板特征** 时，会自动写入带有警告的 **SAMPLE 数据**。  
- 当前仓库中的这些 CSV 大多属于示例数据，目的是结构演示，**并非官方统计**。

---

## 二、按题目要求的数据有效性评估

### 1. 对解决问题“无效”或价值有限的数据

结合题目与代码逻辑，可以认为对主问题帮助有限的数据包括：

- **与大豆、汽车、半导体无关的 HTS 类别**（在 Tariff Data 中仍然存在，但对 Q1–Q3 的定向分析不直接使用）。
- **2025 年的“0 值”或尚未完整填报的观测**：
  - DataWeb 2025 年度数据可能是年内累计，不适合作为完整“历史事实”使用；更多是情景分析基准。  
- **非目标国家/地区的贸易数据**：
  - Q1 主要聚焦：中国从美国、巴西、阿根廷大豆进口。  
  - Q2 主要聚焦：美国市场上日本品牌、在墨西哥生产的车辆等。  
  - Q3 主要聚焦：美中及主要半导体生产/需求国。

这些数据依然可以保留在数据库中，用于敏感性分析或扩展研究，但在“主干模型”中可以先忽略。

### 2. 目前项目中为 SAMPLE 的关键外部数据

以下文件 **当前由 `external_data.py` 自动生成样例数据**，不满足“官方数据”要求：

- `data/external/china_imports_soybeans.csv`
- `data/external/us_auto_sales_by_brand.csv`
- `data/external/us_auto_indicators.csv`
- `data/external/us_semiconductor_output.csv`
- `data/external/us_chip_policies.csv`
- `data/external/us_macro.csv`
- `data/external/us_financial.csv`
- `data/external/us_reshoring.csv`
- `data/external/retaliation_index.csv`

对应的 Q1–Q5 模型目前是 **结构正确**，但实证结果依赖示例数据，需要通过新增 `*_official` 文件接入真实数据。

---

## 三、需要通过联网获取 / 人工下载的官方数据清单

本节按你的 P0 优先级整理。

### P0-1 中国大豆进口数据（替换 Q1 外部数据）

**目标文件（新增，不覆盖原文件）：**

- `data/external/china_imports_soybeans_official.csv`

**推荐字段结构（与 Q1 模型兼容）：**

- `year`  
- `exporter`（US / Brazil / Argentina 等）  
- `import_value_usd`  
- `import_quantity_tonnes`  
- `tariff_cn_on_exporter`（中国对该来源国大豆的关税/附加关税）

**首选官方来源：**

- 中国海关总署统计系统：
  - 总署门户首页 → **“数说海关” → “数据在线查询”**  
  - 或：总署概况 → 统计分析司 → 统计服务 → **数据查询**  
- 通过在线系统按 HS/商品名称“大豆”（通常对应 HS 1201）+ 年度 2015–2024 进行统计，导出 Excel/CSV。

**当前进展：**

- 已确定 **官方获取路径**（海关数据在线查询及相关指南页面）。  
- 受限于交互式网页与当前环境，自动批量抓取暂未完成，需要你先在浏览器中下载原始表格，我再在仓库内进行清洗、汇总与格式转换。

### P0-2 美国汽车数据（替换 Q2 外部数据）

**目标文件（新增）：**

- `data/external/us_auto_sales_by_brand_official.csv`（如果官方仅提供总体/厂商级，也可退而求其次为 `*_aggregate_official.csv`）
- `data/external/us_auto_indicators_official.csv`

**首选官方来源：**

- 美国商务部 / 经济分析局（BEA）：
  - 汽车及零部件出货量、个人消费支出中的“机动车辆及零部件”。
- 美国人口普查局（Census）：
  - 月度/年度批发与零售贸易调查中关于 motor vehicle & parts 的销售额。  
- 若品牌级数据官方统计缺失，可：
  - 使用总体销量 + 市场研究机构公开份额构造近似分解，或在建模时只使用总体层面的指标，并在文档中注明限制。

**当前进展：**

- 已通过联网检索锁定 BEA/Census 的相关统计作为主要候选；尚未将具体系列（table/series ID）映射到模型所需列格式。  
- 需要后续通过 FRED 或直接 CSV 下载来构造 2015–2024 的年度时间序列。

### P0-3 美国半导体产出数据（替换 Q3 外部数据）

**目标文件（新增）：**

- `data/external/us_semiconductor_output_official.csv`

**首选来源：**

- 美国半导体行业协会（SIA）：年度报告中的 **美国半导体销售/产出** 时间序列。  
- BEA 工业账户或相关行业指标，可作为补充。

**当前进展：**

- 已定位 SIA 报告为主要行业来源；尚需将报告中的图表/表格转换为结构化年度数据，并拆分为高/中/低端等段（若题目有明确分段要求）。

### P0-4 CHIPS 法案与 2025 半导体政策（更新 Q3 政策数据）

**目标文件（新增/补充）：**

- `data/external/us_chip_policies_official.csv`  
- `results/q3_trade_response_real.json`（或在 `results` 下新增真实政策驱动的 Q3 响应文件）。

**首选来源：**

- 白宫官网：CHIPS and Science Act 签署声明、实施细则相关 fact sheet。  
- Congress.gov：CHIPS 法案与 2025 年与半导体/供应链相关的成文法/议案。  
- 美国商务部：CHIPS 资金分配、制造项目审批公告。

**当前进展：**

- 已锁定 CHIPS Act 和若干 2025 年供应链相关法案/政策为主要文本来源；尚未完成政策条文到结构化“年度/政策强度指标”的编码，也尚未落地到 `q3_trade_response.json` 的真实参数上。

### P0-5 美国宏观与金融数据（替换 Q5 外部数据）

**目标文件（新增）：**

- `data/external/us_macro_official.csv`  
- `data/external/us_financial_official.csv`

**推荐字段示例：**

- 宏观：`year, gdp_growth, real_gdp_index, industrial_production_index, unemployment_rate, cpi_index`  
- 金融：`year, dollar_index, treasury_yield_10y, sp500_index, corporate_spread` 等

**首选来源（通过 FRED 统一获取）：**

- FRED（圣路易斯联储）：
  - 实际 GDP（BEA 源）
  - 工业生产指数（FRB 源）
  - 失业率（BLS 源）
  - CPI（BLS 源）
  - 美元指数、10 年期国债收益率、S&P 500 等

**当前进展：**

- 已确认 FRED 为统一官方接口，并定位诸如 UNRATE、GDPC1 等典型序列入口；受限于工具调用频次，尚未系统下载 2015–2024 的完整年度数据并落地成表。

### P0-6 制造业回流指标（更新 Q5 回流数据）

**目标文件（新增）：**

- `data/external/us_reshoring_official.csv`

**首选来源：**

- BLS：制造业就业总量/占比（可作为回流 proxy）。  
- BEA：制造业增加值占 GDP 的比重。  
- 商务部/BEA：制造业 FDI、在美投资项目数据。  
- 行业报告（例如 Kearney Reshoring Index）可作为辅助，但需在文档中标明“非官方”。

**当前进展：**

- 已识别 BLS 制造业就业作为一条相对稳健的官方 proxy；具体年度数据尚未下载与格式化。

### P0-7 中美贸易反制措施（更新 Q5 反制指数）

**目标文件（新增）：**

- `data/external/retaliation_index_official.csv`

**首选来源：**

- 中国商务部（MOFCOM）：
  - 对美加征关税、出口管制、反制清单的官方公告与 WTO 立场文件。  
- 国务院/海关相关公告：涉及关税调整的政策文件。

**当前进展：**

- 已通过联网检索锁定部分总结性文件和对美措施报告；需要基于政策时间线构建年度“反制强度指数”（如 0–10），并形成可与 Q5 宏观/金融数据联动的时间序列。

### P0-8 Q3 贸易响应实现（基于真实政策）

**目标文件（新增/覆盖 SAMPLE 结果）：**

- `results/q3_trade_response_real.json`（名称可根据你偏好调整）。

**目标：**

- 在现有 `q3_semiconductors.py` 估计结构基础上，用 **真实的政策时间序列（CHIPS + 出口管制 + 中方反制）** 驱动模型参数，并导出一个包含：
  - 各细分段（高/中/低端）估计系数；
  - 政策情景下的进出口、产出与福利变化；
  的 JSON 文件。

**当前进展：**

- 目前代码中仅有基于 SAMPLE 的政策指数与输出；真实政策尚未编码进模型参数，因此这部分需要在完成上文数据采集后重新估计/设定。

---

## 四、计划方案（含执行顺序与产出形式）

### 1. 文件命名与接入策略

为避免覆盖原题目数据，统一采用：

- 在 `data/external/` 中新增：
  - `*_official.csv`（真实统计）  
  - 视情况新增 `*_fred.csv`、`*_customs.csv` 等更具体后缀  
- 在 `results/` 中新增：
  - `q3_trade_response_real.json` 等基于官方政策的结果文件

**代码改动策略：**

- 在 `external_data.py` 中的 `ensure_q*_external_data()` 函数内：
  - **优先尝试读取 `*_official.csv`**；
  - 仅当官方文件不存在或为空时，才回退到原有 SAMPLE 生成逻辑（并保留警告日志）。
- 这样既满足“使用官方数据”的要求，又保持题目原始环境的可复现性。

### 2. P0 执行顺序建议

1. **Tariff Data 验证与清洗**（已完成验证，视需要可追加 HS 层面处理）。  
2. **中国大豆进口（GACC） → `china_imports_soybeans_official.csv`**：先由你导出原始表。  
3. **FRED 宏观/金融 → `us_macro_official.csv` & `us_financial_official.csv`**：可相对自动化批量下载。  
4. **BLS/BEA 制造业与 FDI → `us_reshoring_official.csv`**。  
5. **CHIPS/半导体政策文件 → `us_chip_policies_official.csv` & `q3_trade_response_real.json`**。  
6. **SIA/BEA 半导体 → `us_semiconductor_output_official.csv`**。  
7. **美国汽车数据 → `us_auto_*_official.csv`**（受制于品牌粒度的可得性，可能略微靠后）。  
8. **MOFCOM 反制措施 → `retaliation_index_official.csv`**。

### 3. P2/P3 后续工作（在 P0 基本完成之后）

- **数据一致性检查（P2）**：
  - 检查每个 `*_official` 文件的来源 URL、下载日期、单位、货币是否在元数据中完整记录。  
  - 确认所有时间序列对齐在 2015–2024（有必要时延长到 2025，用于预测情景）。

- **时间序列对齐与缺失值处理（P2）**：
  - 对年度和季度/月份数据进行一致化（例如先聚合到年度）。  
  - 对少量缺失值采用线性插值或官方推荐方法处理，并在文档中注明。

- **数据文档（P2）**：
  - 在 `data/` 或 `docs/` 下新增一个 `external_data_metadata.md`，逐个记录：
    - 文件名、字段含义、单位  
    - 官方来源、URL、获取日期  
    - 预处理步骤与任何非官方假设

- **敏感性分析（P3）**：
  - 在 Q1–Q5 模型上测试：对关键外生数据（大豆进口、半导体产出、宏观变量等）做 ±5%/±10% 扰动，评估结果的弹性。  
  - 对“非官方但必要的 proxy”（例如回流指数）进行特别的鲁棒性检查。

---

## 五、当前整体状态小结

- **Tariff Data**：
  - 已确认为最新 USITC 官方数据，关税数据库 CSV 已就绪。  
  - 可直接驱动 Q1–Q5 中所有需要 Tariff 相关的模块。

- **外部情景与行业、宏观数据**：
  - 目前仓库内主要是 SAMPLE 数据，结构上已与模型对接良好。  
  - 关键任务是用 `*_official` 文件替换数据源，而非推翻现有代码结构。

- **联网检索**：
  - 已初步锁定所有主要官方来源（GACC、USITC、FRED、SIA、BLS、BEA、MOFCOM、白宫、Congress.gov），但受限于工具调用配额与部分系统交互性，尚未大规模抓取。  
  - 需要你配合完成部分“在线查询系统 → Excel/CSV 导出”的环节。

- **下一步建议**：
  1. 你先从 **海关数据在线查询** 导出 2015–2024 中国按来源国分解的大豆进口量/额表格，放入 `data/raw/`。  
  2. 我在仓库内完成清洗与 `china_imports_soybeans_official.csv` 的生成，并改造 `external_data.ensure_q1_external_data()` 逻辑。  
  3. 并行启动 FRED 宏观/金融数据的系列下载与 `us_macro_official.csv`/`us_financial_official.csv` 生成。
