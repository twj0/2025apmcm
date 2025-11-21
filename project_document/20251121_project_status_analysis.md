# 2025 APMCM Problem C 项目综合状态分析（截至 2025-11-21）

## 1. 项目总体概览

- **题目背景**：2025 APMCM C 题围绕美国“互惠关税”政策，考察其对双边贸易、产业结构、财政收入以及宏观金融和制造业回流的影响，并要求考虑贸易伙伴的反制政策。
- **五个问题对应关系**：
  - **Q1 大豆贸易**：美中巴阿四国之间大豆贸易与中国反制关税下的贸易转移与市场份额重构。
  - **Q2 日本汽车**：日本品牌在美国市场的贸易结构、FDI 与产能迁移，以及关税对美国汽车产业的冲击。
  - **Q3 半导体**：高/中/低端芯片贸易、产出与安全性指标，分析关税、补贴和出口管制等组合政策下的效率–安全权衡。
  - **Q4 关税收入**：基于美国关税与进出口数据，构建静态与动态“拉弗曲线”，模拟特朗普第二任期关税路径对财政收入的影响。
  - **Q5 宏观 / 金融 / 制造业回流与反制**：互惠关税政策在宏观、金融市场和制造业回流层面的中短期影响，以及中方及其他贸易伙伴的反制路径。
- **代码架构概况**：
  - `2025/src/main.py` 作为统一入口，负责环境准备（目录、随机种子、外部数据检查）并串联 `run_q1`–`run_q5` 五个分析流程。
  - `2025/src/models/q1_soybeans.py` ~ `q5_macro_finance.py` 为五个问题的主模型类，每个类封装：数据加载 → 模型估计 → 情景模拟 → 结果输出与绘图。
  - `2025/src/utils/` 提供公共工具：
    - `config.py`：路径、随机种子与绘图风格。
    - `data_loader.py`：USITC 关税 & 进出口宽表 → 标准化长表（含 `duty_collected`）。
    - `mapping.py`：HS → 豆类、汽车、半导体（含 segment）标签；国家名称标准化。
    - `external_data.py`：为 Q1–Q5 生成结构化 **SAMPLE** 外部数据（LLM/规则生成），确保在缺乏官方数据时模型可运行。
    - `data_fetch.py`：FRED 与 UN Comtrade API 封装；前者已成功获取 10 条宏观/行业官方时间序列，后者受环境限制经常失败。
    - `data_consolidate.py`：整合 FRED 系列并生成 `us_macro_consolidated.csv`，同时提供 CSV 数据质量扫描与报告。
    - `worldbank_wits.py`：World Bank 指标抓取与 WITS 批量下载 CSV 标准化（含 `TM.TAX.MANF.SM.AR.ZS` 等平均关税率指标）。
    - `china_soybeans_manual.py`（未在本报告中详细展开）：为 Q1 设计的手动/半自动清洗中国大豆进口数据的工具。
- **文档体系**：`project_document/` 下已有多份高质量文档（数据计划、数据质量报告、适用性评估、数据来源说明等），与代码实现高度对齐，总体上形成了“题目 → 数据需求 → 数据现状 → 代码架构 → 适用性评估”的闭环。

总体结论：**项目架构与代码基本完备，可以一键跑通 Q1–Q5 的方法框架和情景分析；但从“严格基于官方统计的实证分析”角度看，目前最扎实的是 Q3 与 Q5 的宏观/金融部分，其次是 Q2 与 Q4，总体上 Q1 仍被关键数据可追溯性问题所阻塞。**

---

## 2. 数据盘点与状态

### 2.1 数据目录结构与主要文件

- **原始/中间数据**：
  - `2025/data/raw/`
    - `china_soybeans_official_2015_2024.csv`（中国大豆官方数据的原始/中间形态，需配合脚本进一步处理）。
  - `2025/data/processed/`
    - `q1/q1_0.csv`：Q1 的标准化面板数据（若存在则优先在 `SoybeanTradeModel.load_external_china_imports()` 中使用）。
    - `q2/q2_0_japan_brand_sales.csv`，`q2_0_us_auto_sales_by_brand.csv`，`q2_1_industry_indicators.csv`：Q2 的预处理结果，用于品牌销量与行业指标分析。

- **外部数据（external）主目录**：`2025/data/external/`
  - **官方宏观/金融/行业时间序列（来自 FRED 等）**：
    - `us_real_gdp_official.csv`
    - `us_cpi_official.csv`
    - `us_unemployment_rate_official.csv`
    - `us_industrial_production_official.csv`
    - `us_federal_funds_rate_official.csv`
    - `us_treasury_10y_yield_official.csv`
    - `us_sp500_index_official.csv`
    - `us_total_light_vehicle_sales_official.csv`
    - `us_motor_vehicle_retail_sales_official.csv`
    - `us_semiconductor_output_index_official.csv`
  - **宏观合并表**：
    - `us_macro_consolidated.csv`：由 `data_consolidate.merge_macro_datasets()` 基于上述官方文件生成，结构为 `year` + 7 指标列（`gdp_real, cpi, unemployment, industrial_production, fed_funds_rate, treasury_10y, sp500`）。
  - **USITC 关税/贸易相关辅助表**：
    - `hs_soybeans.csv`，`hs_autos.csv`，`hs_semiconductors_segmented.csv`，`hs_to_sector.csv`：HS → 产品/部门映射，支持 Q1–Q3 中按产品/segment 聚合。
  - **Q1 关键文件**：
    - `china_imports_soybeans.csv`：Q1 直接使用的中国大豆进口数据（若不存在或为模板，将由 `external_data.ensure_q1_external_data()` 生成 SAMPLE）。
    - `china_imports_soybeans_official.csv`：从 WITS/Comtrade/GACC 等官方数据标准化而来（需要结合 `worldbank_wits.py` 或手工处理脚本生成）。
    - `q1_china_imports_soybeans_wits_candidate.csv`：结构完备、命名表明来自 WITS 的候选数据，但文档中尚未精确记录其抓取方法与校验流程。
    - `china_soybean_tariffs_candidate.csv`：中国对美/巴/阿大豆关税候选路径，与题目中的反制叙事大体一致。
  - **Q2 关键文件**：
    - `us_auto_sales_by_brand.csv`：品牌 × 年销量及按产地分解（US/Mexico/Japan），目前为 `external_data.ensure_q2_external_data()` 生成的结构化 SAMPLE，用以驱动 Q2 结构分析与情景模拟。
    - `us_auto_indicators.csv`：美国汽车产量、就业、价格指数、GDP、燃油价格指数等行业指标，同样为 SAMPLE 数据。
    - `us_auto_brand_origin_mapping.csv`，`us_auto_official_indicators_2015_2024.csv` 等辅助文件（部分为计划/候选形态）。
  - **Q3 关键文件**：
    - `us_semiconductor_output.csv`：按 `year × segment`（high/mid/low）给出美国芯片产出（十亿美元）和全球需求指数，目前为 SAMPLE，但结构与模型预期高度匹配。
    - `us_chip_policies.csv`：`year, subsidy_index, export_control_china`，刻画 CHIPS 法案与出口管制政策强度，目前为结构化 SAMPLE。
    - `q3_us_semiconductor_policy_overall_candidate.csv`：以整体 segment=`overall` 提供产出与政策候选路径，与分段模型存在可对接空间。
  - **Q4 关键文件**：
    - `q4_avg_tariff_by_year.csv`：2015–2025 平滑插值生成的平均关税率路径（SAMPLE），被 `TariffRevenueModel.estimate_static_revenue_model()` 使用。
    - `q4_dynamic_import_params.json`：短期/中期弹性与调整速度参数模板，当前填入的是合理但非估计得到的样本值。
    - `q4_tariff_scenarios.json`：基准与 `reciprocal_tariff` 场景下 2025–2029 年关税路径与基准进口规模（目前为样本情景）。
    - `q4_us_tariff_revenue_gemini3_scenario.csv`，`q4_us_tariff_revenue_grok4_scenario.csv`：基于 LLM 情景的关税收入路径，主要用于展示，而非从 USITC 原始数据推导。
  - **Q5 关键文件（宏观/金融/回流/反制）**：
    - SAMPLE 基础：`us_macro.csv`，`us_financial.csv`，`us_reshoring.csv`，`retaliation_index.csv`（均由 `external_data.ensure_q5_external_data()` 生成）。
    - LLM 候选：
      - `q5_us_macro_from_grok4_candidate.csv`
      - `q5_us_financial_from_grok4_candidate.csv`
      - `q5_us_reshoring_from_grok4_candidate.csv`
      - `q5_retaliation_index_from_grok4_candidate.csv`
      - `q5_us_macro_reshoring_from_gemini3_candidate.csv`
    - 它们提供带经济直觉约束的情景数据（含制造业 VA 占比、就业占比、回流 FDI/就业、反制指数等），**但不属于官方统计，只能作为情景假设使用。**
  - **World Bank / WITS 指标**：
    - `wb_tariff_mean_china_2015_2024.csv`：World Bank 平均制造业关税率指标的历史下载版本，存在缺失与未解包的嵌套字段，需要用更新版 `worldbank_wits.fetch_worldbank_indicator_to_csv(clean_output=True)` 重新抓取。

- `project_document/` 下的多份数据报告（`data_plan_status_20251120_1242.md`，`data_quality_report_20251120.md`，`data_requirements_status_20251120_2014.md`，`202511202300对当前的csv文件进行评分.md` 等）对上述文件已经做了质量分级。总体上：
  - **高质量官方数据**：主要是 FRED 宏观/金融/行业指标与 USITC Tariff Data，评分约 8–9/10。
  - **结构化样本数据**：例如 `us_macro.csv`、`us_financial.csv`、`us_reshoring.csv`、`us_auto_sales_by_brand.csv`、`us_semiconductor_output.csv` 等，评分约 4–5/10，用于保证流水线可运行。
  - **候选/LLM 情景数据**：`*_candidate.csv` 系列，多为合理的情景路径，但来源与处理方法需要在文稿中单独说明。

### 2.2 时间范围与缺失值概览

结合 `data_consolidate.check_data_quality()` 的自动报告与人工抽样：

- **时间范围**：
  - 绝大多数指标覆盖 **2015–2024**，与题目要求及互惠关税政策发生时间窗口吻合。
  - 某些样本数据延伸到 2025，用于第二任期情景预测（特别是 Q4 关税收入与 Q5 宏观场景）。
- **缺失与异常**：
  - `us_sp500_index_official.csv` 在 2015 年存在个别缺失，需要在 VAR/回归前进行插值或删除。
  - `wb_tariff_mean_china_2015_2024.csv` 在 2023–2024 年缺失较多，并且 `indicator`/`country` 列仍是字典字符串；建议视为“草稿文件”，稍后重新拉取和清洗。
  - LLM 情景数据中没有显著缺失，但数值并不对应真实统计，需要在论文中明确定位为“假设场景”。

---

## 3. 模型框架与算法实现概览

### 3.1 Q1 SoybeanTradeModel（大豆贸易）

- **核心类**：`models/q1_soybeans.py` 中的 `SoybeanTradeModel`。
- **数据流程**：
  - 使用 `TariffDataLoader.load_exports()` + `HSMapper.tag_dataframe()` 从 USITC 出口数据中筛选大豆相关 HS 码，得到美国对中国的大豆出口记录，并构造 `export_value_millions` 等派生变量。
  - `load_external_china_imports()` 负责加载中国从美/巴/阿进口数据：
    - 优先使用 `data/processed/q1/q1_0.csv`；
    - 否则读取 `data/external/china_imports_soybeans.csv`，不存在时自动创建模板（全零），提示用户填充实际数据。
  - 构造 `unit_value`、`price_with_tariff` 等价格变量，并在 `prepare_panel_for_estimation()` 中加入 log 变换和市场份额相关变量：
    - `ln_import_value`, `ln_import_quantity`, `ln_price_with_tariff`, `ln_unit_value`；
    - `market_share`（按年归一化）；
    - 以美国为基准的 `ln_share_ratio` 与 `tariff_diff_vs_us`。
- **计量模型**：
  - **模型 1 – 价格弹性**：
    - `ln_import_value ~ ln_price_with_tariff + C(exporter)`，通过 OLS 估计大豆进口对含税价格的价格弹性，结果存入 `elasticities['price_elasticity']`。
  - **模型 2 – 市场份额与关税差异**：
    - 在去掉美国行后，估计 `ln_share_ratio ~ tariff_diff_vs_us + C(exporter)`，得到相对关税差异对份额的弹性 `share_elasticity`。
- **情景模拟**：
  - `simulate_tariff_scenarios()` 以最近年份为基准，对 baseline / reciprocal_tariff / full_retaliation 三种中国对美大豆加征关税情景模拟进口变化，输出各出口国的进口额与市场份额调整。
- **可视化**：
  - `plot_q1_results()` 输出不同情景下出口国市场份额和进口额对比图。

### 3.2 Q2 AutoTradeModel（日本汽车）

- **核心类**：`models/q2_autos.py` 中的 `AutoTradeModel`。
- **数据流程**：
  - `load_q2_data()`：通过 `TariffDataLoader.load_imports()` 与 `HSMapper.tag_dataframe()` 获取所有汽车相关 HS 进口数据，并按 `year × partner_country` 聚合 `duty_collected → auto_import_charges`。
  - `load_external_auto_data()`：
    - 读取或创建 `us_auto_sales_by_brand.csv` 与 `us_auto_indicators.csv`：若缺失则生成模板/SAMPLE；若存在则直接读取。
- **计量与结构模型**：
  - **进口结构模型**：
    - 目前以 `ln_import_share ~ year + C(partner_country)` 的趋势模型为占位，未来可扩展为含有效关税与宏观控制变量的更完整规格。
  - **产业传导模型**：
    - 基于 `industry_df`（包含产量、就业、GDP、燃油价格）与聚合后的 `auto_import_charges_total` 计算 `import_penetration`，并估计：
      - 若全部变量齐全：`us_auto_production ~ import_penetration + us_gdp_billions + fuel_price_index`；
      - 否则退化为 `us_auto_production ~ import_penetration + year`。
- **情景模拟**：
  - `simulate_japan_response_scenarios()` 设定 S0 无应对、S1 部分转移、S2 激进本地化三类场景，对日本直接出口、墨西哥生产与美国本地生产的权重进行调整，推算：
    - 进口渗透率、美国生产总量与就业的变化（采用简化线性假设，例如 1% 进口渗透上升 → 0.5% 产量下降）。

### 3.3 Q3 SemiconductorModel（半导体效率–安全权衡）

- **核心类**：`models/q3_semiconductors.py` 中的 `SemiconductorModel`。
- **数据流程**：
  - `load_q3_data()`：从 USITC 进口数据中通过 `HSMapper` 筛选半导体相关 HS，并按 `year × partner_country × segment` 聚合 `duty_collected → chip_import_charges`。
  - `load_external_chip_data()`：
    - `us_semiconductor_output.csv`：给出 `year × segment × us_chip_output_billions × global_chip_demand_index`；
    - `us_chip_policies.csv`：给出 `subsidy_index` 与 `export_control_china`，近似刻画 CHIPS 法案与出口管制政策强度。
- **计量模块**：
  - **贸易响应**：
    - 在每个 segment 上估计 `ln_import_charges ~ year + C(partner_country)` 的趋势模型（当前为占位，可扩展加入关税变量与出口管制指标）。
  - **产出响应**：
    - 将产出与政策表按年合并，并估计 `ln_output ~ subsidy_index + export_control_china + global_chip_demand_index`，将补贴系数视作统一的“补贴弹性”。
  - **安全性指标**：
    - `compute_security_metrics()` 通过进口 proxy 与产出值计算：
      - 自给率：`self_sufficiency_pct = output / (output + imports)`；
      - 对中国依赖度：`china_dependence_pct`；
      - `supply_risk_index`：综合自给率缺口与中国依赖度构造风险指数。
- **政策组合情景**：
  - `simulate_policy_combinations()` 构造“仅补贴”“仅关税”“补贴+关税+出口管制”三类政策并结合补贴弹性，估算不同 segment 上自给率的改善、成本指数与安全指数的权衡，输出 `efficiency_security_ratio` 指标。

### 3.4 Q4 TariffRevenueModel（关税收入与拉弗曲线）

- **核心类**：`models/q4_tariff_revenue.py` 中的 `TariffRevenueModel`。
- **数据流程**：
  - `load_q4_data()`：利用 `TariffDataLoader.load_imports()` 汇总到 `year × total_revenue`（即年度关税收入）。
  - `estimate_static_revenue_model()`：
    - 合并 `q4_avg_tariff_by_year.csv` 中的 `avg_tariff`，构造 `avg_tariff_sq` 与 `ln_revenue`，并估计 `ln_revenue ~ avg_tariff + avg_tariff_sq` 以拟合静态拉弗曲线；
    - 由系数计算理论上“收入最大化关税率”。
  - `estimate_dynamic_import_response()`：从 `q4_dynamic_import_params.json` 读取短期/中期弹性与调整速度（目前为设定值而非估计值）。
- **情景模拟**：
  - `simulate_second_term_revenue()` 读取 `q4_tariff_scenarios.json`，基于设定的基准进口值与不同场景下的关税路径，叠加（可选的）动态进口响应，生成 2025–2029 年的收入路径，并计算基准 vs 互惠关税情景的累计收入差异。

### 3.5 Q5 MacroFinanceModel（宏观 / 金融 / 回流 / 反制）

- **核心类**：`models/q5_macro_finance.py` 中的 `MacroFinanceModel`。
- **数据加载与整合**：
  - 通过 `us_macro_consolidated.csv` 获取官方宏观/金融核心指标（GDP、CPI、失业率、INDPRO、联邦基金利率、10 年期国债收益率、SP500）。
  - 结合 `us_reshoring.csv` 或对应的候选/情景文件（如 `q5_us_macro_reshoring_from_gemini3_candidate.csv`）补充制造业 VA 占比、就业占比与回流 FDI/就业。
  - 通过 `retaliation_index.csv` 或 `q5_retaliation_index_from_grok4_candidate.csv` 提供贸易伙伴反制强度的时间序列。
- **计量与时序模型**（根据代码结构与文档）：
  - **回归模块**：
    - 多元回归以 GDP 增长、工业产出、制造业回流指标等为被解释变量，以关税水平、反制指数、金融条件（利率、汇率、股指）为解释变量，评估互惠关税及反制对宏观与制造业回流的边际影响。
  - **VAR/SVAR 模块**：
    - 利用 VAR（或 SVAR）结构刻画关税冲击、反制冲击对 GDP、CPI、工业产出、股市等变量的脉冲响应函数（IRFs），特别关注：
      - 互惠关税情景下的短期 vs 中期产出与物价影响；
      - 在存在中方/多边反制情况下冲击幅度与持续时间的变化。
  - **回流事件研究**：
    - 以关键政策年份（2018 关税升级、2020–2021 疫情供应链冲击、2022 CHIPS 与 IRA 等）为事件窗口，对 `manufacturing_va_share`、`manufacturing_employment_share`、`reshoring_fdi/jobs` 进行事件研究，识别政策驱动的结构性变化。

---

## 4. 代码实现与数据流水线评估

### 4.1 总体流水线

1. **Tariff Data 预处理**：
   - `TariffDataLoader.validate_data_sources()` 检查 USITC 宽表与年度关税 schedule 的存在性与质量。
   - `TariffDataLoader.load_imports()` / `load_exports()` 标准化列名、将宽表按年份熔融为长表，并保留 `hs_code, partner_country, year, duty_collected/export_value`。

2. **产品/部门打标签**：
   - `HSMapper.tag_dataframe()` 基于 HS 前缀将记录打上 `is_soybean, is_auto, is_semiconductor, semiconductor_segment` 标签，支撑 Q1–Q3 的部门/segment 分析。

3. **外部数据准备**：
   - **当尚未获取官方外部数据时**：`external_data.ensure_q1_external_data()` 等函数自动生成结构化 SAMPLE CSV/JSON 文件，提供合理但虚构的时间路径；
   - **当存在 `*_official.csv` 或 `*_candidate.csv` 时**：当前 `external_data.py` 通过 `_is_all_zero()` 检查是否为模板，但尚未完全实现“优先使用 `*_official`，其次 `*_candidate`，最后 SAMPLE”的逻辑。

4. **宏观与 World Bank 指标**：
   - `data_fetch.fetch_fred_series()` 已在文档中确认成功拉取 10 条 FRED 官方序列并落盘为 `*_official.csv`；
   - `data_consolidate.merge_macro_datasets()` 整合成 `us_macro_consolidated.csv` 供 Q5、部分 Q2/Q3 使用；
   - `worldbank_wits.fetch_worldbank_indicator_to_csv()` 与 `standardize_wits_trade_csv()` 提供 World Bank 平均关税率与 WITS 贸易数据的抓取和标准化流程（目前只部分在仓库中执行，存在老版本中未清洗的 WB CSV）。

5. **问题级模型运行**：
   - `main.py` 中的 `run_q1_analysis()` ~ `run_q5_analysis()` 按顺序调用各模型类 API：
     - 数据准备 → 计量估计 → 情景模拟 → 结果落盘（CSV/JSON/图表）。

### 4.2 实现完备度与鲁棒性

- **优点**：
  - 模型接口统一、职责清晰，每个问题都有独立的类与 `run_q*_analysis()` 入口，便于单独调试与整体联动。
  - 日志记录充分（包括数据加载与模型估计过程中的关键信息与异常），便于在数据不完整时快速定位问题。
  - 对缺失外部数据的健壮性较好：`external_data.py` 会生成结构化 SAMPLE，避免流水线完全中断。

- **不足与技术债**：
  - `external_data.py` **仍以生成 SAMPLE 数据为主**，尚未自动识别并优先使用 `*_official`/`*_candidate` 文件；需要手工替换或在模型中显式指定官方路径。
  - Q4 的 `TariffRevenueModel` 目前 **没有直接从 USITC 原始数据计算 `avg_effective_tariff` 与 `total_revenue` 的完整管线**，而是依赖于外部 JSON/CSV 中设定的平均税率与弹性参数；这在结构上是合理的，但削弱了 “完全从官方数据出发”的可追溯性。
  - Q1 中国大豆进口官方数据的手动/半自动处理路径（`data/raw` + `china_soybeans_manual.py` + `scripts/preprocess_china_soybeans_official.py`）在设计上完备，但在当前仓库状态下尚未看到已经完全执行并写回 `china_imports_soybeans.csv` 的 **终态版本**。

---

## 5. 文档–代码–数据一致性评估

### 5.1 与数据计划和质量报告的一致性

- `data_plan_status_20251120_1242.md` 与 `data_requirements_status_20251120_2014.md` 中对各问题数据需求、优先级和缺口的描述，与实际代码与数据基本一致：
  - Q3、Q5 宏观/金融部分已通过 FRED 官方数据实现高质量覆盖；
  - Q2 品牌级数据、Q4 关税收入与有效平均税率、Q1 中国大豆进口官方数据被明确标为缺口或候选状态；
  - 文档明确提到“不要覆盖 SAMPLE 文件，而是新增 `*_official` 并在代码中优先使用”，与 `external_data.py` 的现有设计方向一致（但实现仍部分滞后）。

- `data_quality_report_20251120.md` 与 `202511202300对当前的csv文件进行评分.md` 对每个 CSV 做了定性–定量评分：
  - 官方 FRED 文件与 USITC 宽表被评为 EXCELLENT；
  - 由 `external_data.py` 生成的 SAMPLE 被评为 3–5/10；
  - 这一结论与当前代码中对 SAMPLE 的定位（“用于结构演示和敏感性分析，不宜作为最终实证基准”）一致。

### 5.2 与建模框架的契合程度

- `2025_data_online_sources_SPEC.md` 中对各问题的目标模型和指标选择，与模型类中的实现高度一致：
  - Q1 使用价格弹性、份额弹性与关税差异，配合中国进口数据与反制税率，重构源国结构；
  - Q2 使用进口结构 + 产业传导模型，结合美国汽车产业指标与日本品牌产能重配情景；
  - Q3 使用分段（高/中/低端）模型与安全性指标，自给率和对华依赖度的构造与代码完全一致；
  - Q4 中对动态进出口响应与拉弗曲线的描述与 `TariffRevenueModel` 中的方法完全匹配，只是目前仍在依赖样本参数；
  - Q5 中使用 VAR/SVAR 与回归/事件研究分析宏观、金融和回流的结构，与 `MacroFinanceModel` 中的接口和变量设计相符。

### 5.3 不一致与需要特别说明的点

1. **external_data 优先级逻辑尚未完全实现**：
   - 文档推荐：“当官方数据可用时，放入 `*_official`，再用脚本转换为模型所需结构，并保证不被 SAMPLE 覆盖”；
   - 代码现实：`external_data._is_all_zero()` 可避免覆写非零用户数据，但**尚未系统地检查并优先读取同名 `*_official`**。这意味着：
     - 若用户只是将官方数据命名为 `*_official.csv` 而未显式替换 base 文件，模型可能仍在使用 SAMPLE。

2. **Q1 WITS Candidate 与官方数据之间的边界**：
   - `q1_china_imports_soybeans_wits_candidate.csv` 在结构与数值路径上非常合理，也与 SPEC 中建议的字段完全一致，但项目文档中尚未记录其精确获取流程与来源；
   - 在未补充“可追溯元数据”前，将其视为 **高质量候选/情景数据** 而非“严格意义上的官方统计”。

3. **Q4 官方收入表尚未生成**：
   - 文档中多次提到可以从 USITC 宽表构建 `year, total_import_value, total_tariff_revenue, avg_effective_tariff` 等表，以支撑 Q4 模型；
   - 目前代码中已有能力（`TariffDataLoader.compute_effective_tariff()` 与 `compute_tariff_indices()`），但尚未看到仓库内已经存在完整的 `q4_tariff_revenue_official.csv`，`q4_avg_tariff_by_year.csv` 仍是样本曲线；
   - 这需要在论文撰写中清晰标注：当前 Q4 的量化结果更多基于“结构+参数化情景”，而非完全由官方数据估计得到。

4. **Q5 反制指数与回流指标的官方支持仍然有限**：
   - 文档中建议从 MOFCOM 公告、学术研究与新闻报道中构建“反制强度指数”，以及从 BLS/BEA/Reshoring Initiative 获取回流指标；
   - 目前仓库中的 `retaliation_index.csv` 与 `q5_retaliation_index_from_grok4_candidate.csv` 均为 **手工/LLM 构造**，尚未与正式统计或公开资料形成严密映射。

---

## 6. 差距分析与优先建议（聚焦 Q5 及互惠关税/反制情景）

本节按问题（Q1–Q5）与维度（数据、模型、情景）给出差距与优先建议，其中 **重点展开 Q5 及与其紧密相关的互惠关税与反制路径**。

### 6.1 Q1 大豆贸易

- **当前状态**：
  - 代码架构（`SoybeanTradeModel` + `TariffDataLoader` + `HSMapper` + `china_soybeans_manual.py`）已经足以支撑严肃的面板回归与情景模拟；
  - 数据层面，USITC Tariff Data 完备，但 **中国按来源国大豆进口官方数据尚未完全定稿**：
    - `china_imports_soybeans.csv` 很可能仍是 SAMPLE 或混合版本；
    - `q1_china_imports_soybeans_wits_candidate.csv` 虽结构合理，但缺乏可追溯元数据。

- **差距**：
  - 若论文需要声称“基于官方统计的历史数据识别贸易弹性与替代效应”，则必须：
    - 要么确认 candidate 文件的官方来源并记录获取与清洗流程；
    - 要么按照文档中的 GACC/WITS 路径从零重建官方大豆进口数据。

- **优先建议（P0–P1）**：
  - **P0**：
    - 若 candidate 文件确为从 WITS/Comtrade 官方导出：
      - 在 `data_sources_usage_*.md` 中补充其查询参数、下载日期与处理脚本；
      - 将其视为 `china_imports_soybeans_official.csv`，并通过 `preprocess_china_soybeans_official.py` 生成最终供模型使用的 `china_imports_soybeans.csv`；
      - 在 `external_data.ensure_q1_external_data()` 中加入逻辑：若存在 `china_imports_soybeans_official.csv` 且非模板，则不再生成 SAMPLE。
  - **P1**：
    - 若无法确认 candidate 来源，则执行：GACC/WITS 手动下载 → `worldbank_wits.standardize_wits_trade_csv` / `china_soybeans_manual.py` → `preprocess_china_soybeans_official.py` 的完整链条，并在 `project_document` 中撰写简明的数据处理说明。

### 6.2 Q2 日本汽车

- **当前状态**：
  - 使用 USITC 进口数据 + HS 映射可以较好刻画 **整体汽车进口关税与日本/墨西哥/其他来源国的相对地位**；
  - 但品牌 × 产地的细粒度销量与生产结构，目前主要依赖于 `us_auto_sales_by_brand.csv` 等 SAMPLE 数据；
  - 文档中已经规划了从 BEA/Census/FRED 或行业网站补充品牌级信息（例如 GoodCarBadCar、Ward's、OEM 年报等），但尚未系统落地到 CSV 与代码中。

- **差距**：
  - 若 Q2 的重点是“日本汽车在美国市场的份额变化与产能转移”，仅用 SAMPLE 的品牌结构会削弱结论的说服力；
  - 目前 `AutoTradeModel` 的计量部分仍为以年份为核的简化趋势模型，没有将实际关税变化与品牌结构正式联立建模。

- **优先建议**：
  - **数据层面**：
    - 最低目标：获取“日本品牌在美国市场的整体份额及本地/墨西哥生产占比”年份序列，即使做不到逐品牌；
    - 将这些系列整理为 `us_auto_sales_by_brand_official_or_proxy.csv`，并在 `load_external_auto_data()` 中优先读取；
    - 将 `us_total_light_vehicle_sales_official.csv` 与品牌份额结合，校准 SAMPLE 数值，使总量与 FRED 官方一致。
  - **模型层面**：
    - 在 `estimate_import_structure_model()` 中引入：
      - 有效关税（基于 HS + 年度 schedule）；
      - 可能的配套变量（燃油价格、GDP、汇率等，从 FRED 获取）。

### 6.3 Q3 半导体

- **当前状态**：
  - 代码结构已经支持高/中/低端分段分析与供应安全指数的构造；
  - 官方数据方面：FRED 的 `us_semiconductor_output_index_official.csv` 为整体半导体产出提供高质量时间序列；
  - 分段产出、政策强度和对华依赖等指标目前以 SAMPLE 或 `q3_us_semiconductor_policy_overall_candidate.csv` 为主。

- **差距与建议**：
  - 若时间有限，**建议优先完成“整体段模型 + 明确的政策指数构造说明”**：
    - 使用 FRED 官方指数作为 `us_chip_output_billions`（或其指数）基线；
    - 将 `q3_us_semiconductor_policy_overall_candidate.csv` 中的 `subsidy_index` 与 `export_control_china_index` 明确声明为“基于官方政策事件编码的指数”，并在 `project_document` 补充编码规则；
    - 以整体段为主，分段作为敏感性延伸。
  - 若有余力，可参考 SIA 报告或产业研究，将整体指数拆解为高/中/低端 proxy，并在 `us_semiconductor_output.csv` 中替换 SAMPLE。

### 6.4 Q4 关税收入与拉弗曲线

- **当前状态**：
  - USITC 宽表与 tariff schedule 已准备完毕，`TariffDataLoader` 具备将其转换为长表并计算有效关税的能力；
  - 但实际用于模型的 `q4_avg_tariff_by_year.csv` 与动态弹性参数、关税情景 JSON 均为样本设定。

- **核心差距**：
  - 尚未真正从 USITC 数据 **推导出历史上的 `avg_effective_tariff` 和 `total_revenue` 系列**，从而限制了静态与动态模型的实证基础；
  - 目前模拟的“第二任期互惠关税路径”在量值上是合理的，但更接近“政策设想情景”而非从数据回推的最优路径。

- **优先建议（P0）**：
  - 基于 `TariffDataLoader.create_imports_panel()` 和 `compute_effective_tariff()`：
    1. 生成 `imports_panel.parquet`，其中包含 `year, hs_code, partner_country, duty_collected, import_value?, effective_tariff`；
    2. 若仍缺少 `import_value`，可从 USITC 贸易额 CSV 中合并，或通过近似方法构造；
    3. 使用 `compute_tariff_indices(groupby=['year'])` 生成 `tariff_indices`，并落盘为 `q4_tariff_indices_official.csv`：
       - 字段：`year, tariff_index_weighted, tariff_index_simple, total_duty_collected`；
    4. 将 `total_duty_collected` 作为 `total_revenue`，`tariff_index_weighted` 或 `tariff_index_simple` 作为 `avg_tariff`，替换现有样本表。
  - 完成后，重新估计静态拉弗模型，使其真正反映历史数据（哪怕样本年份有限）。

### 6.5 Q5 宏观 / 金融 / 回流 / 反制（重点）

- **当前状态总结**：
  - **宏观与金融数据**：
    - FRED 官方时间序列（GDP、CPI、失业率、工业产出、联邦基金利率、10Y 国债收益率、SP500）已获取，并整合为 `us_macro_consolidated.csv`，覆盖 2015–2024 年；
    - 这些数据完全可以支持 Q5 中的回归与 VAR/SVAR 分析，是目前最强的官方数据基础之一。
  - **制造业回流指标**：
    - `us_reshoring.csv` 与 `q5_us_macro_reshoring_from_gemini3_candidate.csv` 等提供了制造业 VA 占比、就业占比与回流 FDI/就业的合理路径，但以 SAMPLE/LLM 情景为主；
    - 文档中建议从 BLS/BEA/Reshoring Initiative 获取更权威的回流数据，目前尚未完全落地。
  - **反制/报复指数**：
    - `retaliation_index.csv` 与 `q5_retaliation_index_from_grok4_candidate.csv` 分别提供基础与 LLM 候选的报复强度时间路径，均非官方统计；
    - 指标大体呈现“2018 后上升”的合理叙事，但缺乏明确、可重复的构造方法说明。
  - **模型实现**：
    - 回归、VAR/SVAR 与事件研究等模块已经在 `MacroFinanceModel` 中实现，接口设计合理、可直接调用。

- **与题目要求的契合度**：
  - 在“**结构建模与政策情景**”层面，Q5 完全可以给出：
    - 互惠关税与反制冲击下 GDP、CPI、工业产出与金融市场的冲击响应；
    - 制造业回流指标在关税与其他政策（补贴、监管）共同作用下的趋势分析；
    - 在第二任期互惠关税路径（与 Q4 关税情景联动）下的宏观与产业综合效应。
  - 在“**严格依托官方数据的历史识别**”层面：
    - 宏观与金融部分已满足要求；
    - 制造业回流与反制指数仍存在明显的官方数据缺口，当前更适合定位为“情景假设驱动的政策分析”。

- **关键差距**：
  1. **回流指标的官方化**：
     - 现有 `us_reshoring.csv` 与 LLM candidate 只能作为“合理的情景路径”；
     - 若希望在论文中给出“实证识别回流程度”的结论，需要从 BLS/BEA 等处获取：
       - 制造业就业占比（BLS）；
       - 制造业增加值占 GDP 比重（BEA）；
       - Reshoring Initiative 或类似数据库中的“回流项目数、岗位数或 FDI 金额统计”。
  2. **反制指数的可追溯构造**：
     - 目前指数虽然符合叙事（2018 后上升、2020–2022 期间持续较高），但缺乏基于 MOFCOM 公告、关税清单数量、涉及贸易额等“客观变量”的定量构造方法；
     - 这会在答辩或评审时被质疑为“随意设定”。

- **重点建议（P0）**：

  1. **锁定宏观/金融 VAR/回归的“官方数据版本”**：
     - 以 `us_macro_consolidated.csv` 为唯一输入源（必要时加入 `us_semiconductor_output_index_official.csv` 与 `us_total_light_vehicle_sales_official.csv` 数据），在 Q5 中：
       - 构建不依赖任何 SAMPLE/LLM 的 VAR/回归模型；
       - 将互惠关税与反制变量先以“事件/虚拟变量”的形式建模（例如 2018–2019 为互惠关税高强度期，2020–2021 为疫情+供应链冲击，共同体现在宏观变量上）。
     - 这样，即便回流与反制指标部分仍为 proxy/情景，也可以保障 **“宏观金融主结果是稳健且可追溯的”**。

  2. **将回流与反制指标拆分为“历史指标 + 情景路径”两个层级**：
     - 历史层面：
       - 从 BLS/BEA 拉取至少两三个简单指标：制造业就业占比、制造业增加值占比、制造业产出指数等；
       - 将这些指标与 VAR/回归一并建模，回答“截至目前，是否已观察到显著的制造业回流趋势”。
     - 情景层面：
       - 保留 `q5_us_macro_reshoring_from_gemini3_candidate.csv` 和 `q5_retaliation_index_from_grok4_candidate.csv` 作为“第二任期互惠关税 + 强/弱反制”情景下的假设路径；
       - 在论文中明确说明：这些路径由模型/LLM 根据经济直觉设定，用于评估政策可能的中期影响，而非对未来的预测。

  3. **与 Q4 关税情景联动**：
     - 将 Q4 中 `q4_tariff_scenarios.json` 的关税路径与 Q5 中的 VAR/回归结合：
       - 在 Q4 得到不同关税路径下的收入与平均税率变化；
       - 在 Q5 中将这些税率路径映射为宏观冲击（例如通过回归中的“关税/有效税率变量”或事件时间结构），从而实现在宏观/金融/回流维度上对互惠关税政策的联动评估。

---

## 7. 总结：整体可行性与下一步优先级

- **整体可行性**：
  - **架构与代码层面**：项目已具备高质量的整体框架，能够完整演示并实现 Q1–Q5 所要求的模型结构与政策情景；
  - **数据与实证层面**：
    - 官方数据最扎实的部分集中在宏观/金融（Q5）与部分行业（Q3 整体、Q2 总量）；
    - Q1 中国大豆、Q2 品牌级数据、Q4 历史关税收入/有效税率、Q5 的回流与反制指数仍需补完或在论文中明确降级为“情景层面”。

- **建议的整体优先级（结合时间与性价比）**：
  1. **P0：锁定 Q5 宏观/金融 + Q3 整体模型为“官方数据版核心结果”**：
     - 用 FRED 官方系列与 USITC Tariff Data 构造宏观 VAR/回归与半导体整体安全性指标；
     - 把互惠关税与反制作为事件/情景变量嵌入模型，得到稳健的主结论。
  2. **P0：构建 Q4 官方版 `tariff_indices_official.csv` 与 `q4_tariff_revenue_official.csv`**：
     - 从 USITC 数据出发，用现有工具生成年度关税收入与加权平均税率，替换当前样本曲线；
     - 这将大幅提升 Q4 与 Q5 交叉分析的可信度。
  3. **P1：解决 Q1 大豆官方数据路径问题**：
     - 要么确认并官方化 `q1_china_imports_soybeans_wits_candidate.csv`，要么按照 GACC/WITS 路径重建；
     - 这是回答“互惠关税下中国对美大豆需求与贸易重构”的关键。
  4. **P1：为 Q2 获取至少“日本品牌整体”的官方/半官方份额与产地结构**：
     - 不必面面俱到，但需有足够真实信息支撑“日本汽车对美国市场的结构影响”的分析。
  5. **P2：完善回流与反制指数的构造说明与来源**：
     - 哪怕完全基于二手资料或研究报道，也应给出定量化的打分规则与数据表，以便在论文中透明披露。

若按上述路径推进，**在不大幅增加新开发工作量的前提下，可以形成：以 Q3 + Q5 宏观/金融为主线，Q1/Q2/Q4 为支持性和情景性分析的整体答卷**，既符合题目对“互惠关税及反制政策的中短期综合影响”这一主线要求，又在数据可追溯性与模型严谨性之间取得较好平衡。
