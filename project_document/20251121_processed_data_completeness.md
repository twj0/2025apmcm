# Processed 数据完备性分析报告（2025/data/processed）

> 截至：2025-11-21  
> 目录：`2025/data/processed/` 及子目录（当前共 4 个 CSV 文件）

---

## 1. 数据发现与总览

本次分析使用 `uv run python` + pandas 在项目根目录执行，对 `2025/data/processed/` 及其子目录递归扫描，发现 4 个 CSV：

| # | 路径 | 关联问题 | 行数 (n_rows) | 列数 (n_cols) | 时间范围 (year) | 大小 (bytes) |
|---|------|----------|---------------|---------------|------------------|--------------|
| 1 | `2025/data/processed/q1/q1_0.csv` | Q1 大豆 | 30 | 6 | 2015–2024 | 1357 |
| 2 | `2025/data/processed/q2/q2_0_japan_brand_sales.csv` | Q2 汽车（日系品牌子集） | 50 | 7 | 2015–2024 | 2424 |
| 3 | `2025/data/processed/q2/q2_0_us_auto_sales_by_brand.csv` | Q2 汽车（全品牌） | 120 | 7 | 2015–2024 | 5636 |
| 4 | `2025/data/processed/q2/q2_1_industry_indicators.csv` | Q2 汽车（行业指标） | 10 | 4 | 2015–2024 | 345 |

> 说明：时间范围根据 `year` 列最小/最大值推断；所有文件均覆盖 2015–2024，完全满足题目要求的 2015–2024 主区间。

---

## 2. 单文件结构与完整性检查

### 2.1 `q1/q1_0.csv`（中国大豆进口标准面板）

- **路径**：`2025/data/processed/q1/q1_0.csv`
- **用途/关联代码**：
  - 在 `models/q1_soybeans.py` 中：
    - `processed_file = DATA_PROCESSED / 'q1' / 'q1_0.csv'`
    - 若存在，则作为 **标准 processed 数据** 使用，替代 `data/external/china_imports_soybeans.csv`。
- **结构与列信息**：
  - 行数：30（3 个出口国 × 10 年）
  - 列数：6
  - 列：
    - `year`：`int64`，无缺失，2015–2024；数值型，`[2015, 2024]`
    - `exporter`：`object`，无缺失（预期为 `US`, `Brazil`, `Argentina`）
    - `import_quantity_tonnes`：`int64`，无缺失，`[1,500,000, 74,500,000]`
    - `import_value_usd`：`int64`，无缺失，`[700,000,000, 43,800,000,000]`
    - `unit_price_usd_per_ton`：`int64`，无缺失，`[335, 609]`
    - `tariff_cn_on_exporter`：`float64`，无缺失，`[0.03, 0.33]`
- **缺失值与重复行**：
  - 每列缺失值：0（缺失率 0%）
  - 全行重复：0
  - 作为键的组合 `year + exporter`：重复行数 0（每年每出口国唯一样本）。
- **数值异常检查**：
  - 所有数值列均无负值；价格、数量、金额量级合理。
- **与模型代码匹配度**：
  - `load_external_china_imports()` 期望列：`year`, `exporter`, `import_value_usd`, `import_quantity_tonnes`, `tariff_cn_on_exporter`，可选 `unit_price_usd_per_ton`：
    - 文件中列名 **与代码完全一致**；
    - 若存在 `unit_price_usd_per_ton`，代码直接复制为 `unit_value`，否则按 `import_value_usd / import_quantity_tonnes` 计算；
    - 随后构造 `price_with_tariff`、`market_share`、`ln_*` 等派生变量。
  - 时间覆盖 2015–2024，满足以 2015 为起点、以近年作为情景基准年的需求。
- **来源与可追溯性（基于现有文档理解）**：
  - 按项目规划，该文件应来自：
    - `data/external/china_imports_soybeans.csv`（由 WITS/GACC/Comtrade 原始数据清洗而来），或
    - `q1_china_imports_soybeans_wits_candidate.csv` 的进一步标准化。
  - 但当前仓库中：
    - 尚未在文档中看到“已完成从官方源到 `q1_0.csv` 的最终清洗脚本与版本记录”；
    - `data_quality_report_20251120.md` 仍将 Q1 官方数据状态标为 **部分完成 / 候选**。

**小结（Q1）**：
- 结构完备度：**高**（无缺失、列名与模型完全匹配、键唯一、数值合理）。
- 数据来源严格性：**中等**（推测源自官方或 WITS 候选，但缺少最终“可追溯处理链”说明）。
- 建模就绪度（严格实证视角）：**NEEDS_REVIEW**  
  - 可直接用于方法展示与情景模拟；
  - 若需要在论文中声称“严格基于官方统计”，需补充：数据来源说明、清洗脚本与版本记录。

---

### 2.2 `q2/q2_0_japan_brand_sales.csv`（日本品牌销量子集）

- **路径**：`2025/data/processed/q2/q2_0_japan_brand_sales.csv`
- **用途/关联问题**：Q2，日本品牌在美国市场的销量与产地结构（子集）；当前 Q2 模型主流程并未直接读取该文件，但可作为 **日本品牌专题分析** 的标准面板。 
- **结构与列信息**：
  - 行数：50
  - 列数：7
  - 列：
    - `year`：`int64`，无缺失，2015–2024
    - `brand`：`object`，无缺失
    - `total_sales`：`int64`，无缺失，`[300,000, 1,989,000]`
    - `us_produced`：`int64`，无缺失，`[60,000, 436,000]`
    - `mexico_produced`：`int64`，无缺失，`[105,000, 696,150]`
    - `japan_imported`：`int64`，无缺失，`[135,000, 895,050]`
    - `origin`：`object`，无缺失
- **缺失值与重复行**：
  - 所有列缺失值数：0
  - 全行重复：0
  - 组合键 `year + brand`：重复行数 0（每品牌每年一行）。
- **数值异常检查**：
  - 所有销量/产量列为非负整数，量级在 10 万–200 万辆区间，符合整体市场规模的数量级；
  - 未检测到负值或明显异常极值。
- **与模型代码匹配度**：
  - `AutoTradeModel` 当前只直接使用：
    - USITC 关税与进口数据（`TariffDataLoader.load_imports()` → `autos_agg`），
    - 外部文件 `DATA_EXTERNAL / 'us_auto_sales_by_brand.csv'` 与 `DATA_EXTERNAL / 'us_auto_indicators.csv'`。
  - processed 文件 `q2_0_japan_brand_sales.csv` **没有直接被模型读取**，但其列结构与 `us_auto_sales_by_brand.csv` 模板高度一致：
    - `year`, `brand`, `total_sales`, `us_produced`, `mexico_produced`, `japan_imported`。
  - 若要在现有模型中使用，需要：
    - 在单独分析脚本中读入该 processed 文件；或
    - 在 `external_data.py` / `q2_autos.py` 中增加可选分支，优先读取 processed 文件。
- **来源与可追溯性**：
  - 结合 `external_data.ensure_q2_external_data()` 和文档描述，现阶段品牌级数据多为 **结构化 SAMPLE**：
    - 总量参考 `us_total_light_vehicle_sales_official.csv` 等官方数据；
    - 品牌与产地结构根据经验和行业报道设定，而非逐品牌官方统计。

**小结（Q2 日本品牌子集）**：
- 结构完备度：**高**（键唯一、无缺失、列名清晰、适合面板回归）。
- 数据来源严格性：**低–中**（高质量 SAMPLE，而非逐品牌官方统计）。
- 建模就绪度（严格实证视角）：**NEEDS_REVIEW**  
  - 适合作为结构和情景分析的基础（如日本品牌产能转移情景），但在论文中应明确为“构造数据”。

---

### 2.3 `q2/q2_0_us_auto_sales_by_brand.csv`（全品牌销量面板）

- **路径**：`2025/data/processed/q2/q2_0_us_auto_sales_by_brand.csv`
- **用途/关联问题**：Q2，美国市场所有品牌销量与产地结构标准面板；可用于：
  - 估计品牌层面的进口/本地化趋势；
  - 构建日本品牌 vs 其他品牌的对比分析。
- **结构与列信息**：
  - 行数：120
  - 列数：7（与日本子集一致）
  - 列：
    - `year`：`int64`，无缺失，2015–2024
    - `brand`：`object`，无缺失
    - `total_sales`：`int64`，无缺失，`[300,000, 1,989,000]`
    - `us_produced`：`int64`，无缺失，`[60,000, 1,417,500]`
    - `mexico_produced`：`int64`，无缺失，`[105,000, 696,150]`
    - `japan_imported`：`int64`，无缺失，`[135,000, 895,050]`
    - `origin`：`object`，无缺失
- **缺失值与重复行**：
  - 各列缺失值：0
  - 全行重复：0
  - `year + brand` 作为键：重复行数 0。
- **数值异常检查**：
  - 所有销量列非负；
  - 数值分布与日本子集类似，符合 2015–2024 美国轻型车销量大致量级。
- **与模型代码匹配度**：
  - 与 `q2_0_japan_brand_sales.csv` 一样，**当前 Q2 模型并不直接读取该 processed 文件**，而是读 `DATA_EXTERNAL / 'us_auto_sales_by_brand.csv'`；
  - 两者列结构一致，说明 processed 文件极可能是对 external 文件的整理或子集/补充版本。
  - 若在论文/说明里使用 processed 版本，需保证总销量与 FRED 官方 `us_total_light_vehicle_sales_official.csv` 一致（或给出差异说明）。

**小结（Q2 全品牌面板）**：
- 结构完备度：**高**（适合品牌×年份面板建模）。
- 数据来源严格性：**低–中**（SAMPLE + 参考官方总量）。
- 建模就绪度（严格实证视角）：**NEEDS_REVIEW**  
  - 适合结构分析与情景模拟；
  - 如在正式论文中用于定量结论，应在 `project_document` 中单独说明：数据构造方法与校准规则。

---

### 2.4 `q2/q2_1_industry_indicators.csv`（行业指标）

- **路径**：`2025/data/processed/q2/q2_1_industry_indicators.csv`
- **用途/关联问题**：Q2，美国汽车行业整体指标面板；可用于：
  - 描述性图表（行业趋势）；
  - 简化版的“进口渗透率 → 产量/就业”关系分析。
- **结构与列信息**：
  - 行数：10
  - 列数：4
  - 列：
    - `year`：`int64`，无缺失，2015–2024
    - `total_light_vehicle_sales_million`：`float64`，无缺失，`[14.23, 17.88]`
    - `us_auto_employment_thousands`：`float64`，无缺失，`[900.0, 990.8]`
    - `us_auto_price_index_1982_100`：`float64`，无缺失，`[145.8, 175.5]`
- **缺失值与重复行**：
  - 各列缺失值：0
  - 全行重复：0
  - `year` 作为键：重复行数 0。
- **数值异常检查**：
  - 所有数值列无负值；数值量级和趋势符合公开资料中美国轻型车销量、就业与价格指数概貌。
- **与模型代码匹配度**：
  - `AutoTradeModel.estimate_industry_transmission_model()` 当前依赖于 `DATA_EXTERNAL / 'us_auto_indicators.csv'`，期望列包括：
    - `year`, `us_auto_production`, `us_auto_employment`, `us_auto_price_index`, `us_gdp_billions`, `fuel_price_index`。
  - processed 文件使用的是 **更“出版友好”的列名与单位**：
    - `total_light_vehicle_sales_million`（单位：百万台）、
    - `us_auto_employment_thousands`（单位：千人）、
    - `us_auto_price_index_1982_100`（基期指数）。
  - 因此：
    - 该 processed 文件 **不能直接作为现有 Q2 代码的输入**；
    - 但非常适合作为论文中的图表与说明数据，或者用于派生出 `us_auto_production` 等变量（在额外脚本中转换单位/推算产量）。
- **来源与可追溯性**：
  - 结合 `data_quality_report_20251120.md` 的描述，这类 auto 行业总量多来自 FRED/BEA 官方时间序列，经过单位转换与汇总；
  - 若要将其升级为 Q2 主输入，需要在 `project_document` 中明确：
    - 对应的 FRED 系列 ID；
    - 单位转换与加总/差分规则；
    - 与外部 `us_auto_indicators.csv` 的关系。

**小结（Q2 行业指标）**：
- 结构完备度：**高**。
- 数据来源严格性：**中–高**（很可能来自官方统计系列，但当前仓库中未完全显式关联到 FRED/BEA ID）。
- 建模就绪度：
  - 对于 **描述性分析/图表**：**READY**；
  - 对于 **直接接入现有 Q2 代码**：**NEEDS_REVIEW**（需在数据脚本中添加列名和单位的映射/转换）。

---

## 3. 与模型使用的匹配度汇总

### 3.1 Q1：`q1_0.csv` 与 `SoybeanTradeModel`

- 模型代码：`models/q1_soybeans.py`
  - 优先读取 `DATA_PROCESSED / 'q1/q1_0.csv'`；
  - 若不存在，则回退至 `data/external/china_imports_soybeans.csv`（或生成模板）。
- 匹配结论：
  - 列名与预期完全一致；
  - 时间覆盖 2015–2024，满足建模需求；
  - 主键 `year + exporter` 唯一；
  - 可直接用于：
    - 面板回归（价格弹性与份额弹性）、
    - 情景模拟（互惠关税 + 全面报复情景）。

### 3.2 Q2：`q2_0_*` 与 `AutoTradeModel`

- 模型代码：`models/q2_autos.py`
  - 主数据流：
    - USITC 关税与进口数据（并非 processed 目录）；
    - 外部 `us_auto_sales_by_brand.csv` 和 `us_auto_indicators.csv`（位于 `data/external/`）。
  - processed 目录下三份 CSV **目前均未被该模块直接引用**：
    - `q2_0_japan_brand_sales.csv`、
    - `q2_0_us_auto_sales_by_brand.csv`、
    - `q2_1_industry_indicators.csv`。
- 匹配结论：
  - 三个 processed 文件的结构与 external 文件的模板高度匹配（或是其重新标度的版本）；
  - 若要在正式 Q2 流水线中使用 processed 文件，需要：
    - 在 `external_data.ensure_q2_external_data()` 或 `AutoTradeModel.load_external_auto_data()` 中优先读 processed 版本；
    - 或在专门的分析脚本中，以 processed 为主，external 为辅助。

---

## 4. 数据质量等级与建模就绪度评估

结合数据来源、结构完备度及与代码的一致性，为每个文件给出质量等级和“建模就绪度”标签（针对 **严格数学建模/论文级别实证要求**）：

| 路径 | 关联问题 | 质量等级 | 说明 | 建模就绪度 |
|------|----------|----------|------|------------|
| `q1/q1_0.csv` | Q1 大豆 | **Standardized (Candidate/Official mix)** | 结构完备，列与模型匹配，极可能来源于官方或 WITS 候选，但文档中尚未完全记录官方数据清洗链路 | **NEEDS_REVIEW**：可直接用于方法与情景分析；论文中若声称“官方数据”，需补充来源与脚本说明 |
| `q2/q2_0_japan_brand_sales.csv` | Q2 日系品牌 | **Standardized SAMPLE** | 结构完备，品牌×年份面板适配度高，但主要依据样本设定与行业经验构造 | **NEEDS_REVIEW**：适用于情景分析，不适合作为严格官方实证基础 |
| `q2/q2_0_us_auto_sales_by_brand.csv` | Q2 全品牌 | **Standardized SAMPLE** | 与 external 模板完全一致，应为样本/经验构造序列；总量需与 FRED 官方系列校验 | **NEEDS_REVIEW**：可用于结构/情景建模，须在文稿中标注为构造数据 |
| `q2/q2_1_industry_indicators.csv` | Q2 行业指标 | **Standardized from Official (likely)** | 指标形态与 FRED/BEA 官方序列高度一致，但仓库中尚未完全显式地记录来源 ID 与转换规则 | **READY（用于描述性分析）/ NEEDS_REVIEW（直接接入 Q2 模型）**：若映射到 `us_auto_indicators.csv` 所期望列，则可作为严格建模用输入 |

---

## 5. 详细问题清单

### 5.1 共性问题

1. **与 external 数据脚本的衔接不完全**：
   - Q2 相关 processed 文件尚未在 `external_data.py` 与 `AutoTradeModel` 中系统集成，导致：
     - 实际模型运行仍以 `data/external/` 目录为主；
     - processed 文件更接近“用于论文图表和补充分析的标准数据集”，而不是“主流水线的唯一真值来源”。

2. **官方数据可追溯性信息不足**：
   - 部分 processed 文件（特别是 `q1_0.csv` 与 `q2_1_industry_indicators.csv`）有较大概率源于官方统计，但：
     - 当前未在 `project_document` 中给出：源网站、下载日期、API 调用参数（如 FRED series ID）、中间脚本与版本标记等；
     - 对于论文级别严格复现要求，这是主要短板。

### 5.2 单文件问题

- **`q1/q1_0.csv`**：
  - 未在文档中看到“从 raw → external → processed”的完整 pipeline 说明；
  - 无法仅凭仓库内容判断是否完全对应官方统计，还是混入了 candidate/插值数据。

- **`q2/q2_0_japan_brand_sales.csv` & `q2/q2_0_us_auto_sales_by_brand.csv`**：
  - 数据量与结构完备，但来源为 SAMPLE；
  - 与 GoodCarBadCar、Ward's、OEM 年报等真实品牌统计之间的偏差目前不可知。

- **`q2/q2_1_industry_indicators.csv`**：
  - 指标命名与单位（million, thousands, 1982=100）更接近“对外展示版本”；
  - 与 `us_auto_indicators.csv`（Q2 模型当前使用的内部版本）之间的映射规则尚未在脚本中编码；
  - 若要统一，需要一个专门的转换脚本（例如 `scripts/q2_build_industry_panel.py`）。

---

## 6. 改进建议与操作性方案

### 6.1 Q1：`q1_0.csv`（中国大豆进口）

- **目标**：将其升级为“可追溯官方标准面板”，从而为 Q1 的价格与份额弹性估计提供坚实基础。
- **建议步骤**：
  1. 在 `project_document` 中补充一份短文档（例如 `q1_china_imports_soybeans_pipeline.md`）：
     - 列出原始数据源（WITS/Comtrade/GACC）、查询参数、下载日期；
     - 给出清洗脚本名（如 `worldbank_wits.py`, `china_soybeans_manual.py`, 可能存在的 `scripts/preprocess_china_soybeans_official.py`）；
     - 标注 `q1_0.csv` 的生成命令与版本时间戳。
  2. 在必要时重新执行 raw→processed 流水线，并在脚本中固定随机性与排序，保证可复现。
  3. 在 Q1 论文部分注明：
     - 面板样本国家与年份；
     - 是否进行了插值/外推；
     - 对比官方摘要（例如 GACC 年度统计）以验证总量一致性。

### 6.2 Q2：品牌销量面板 `q2_0_*`

- **目标**：在保证建模灵活性的同时，对“官方 vs 构造”的边界给出清晰说明。
- **建议步骤**：
  1. 将 `q2_0_us_auto_sales_by_brand.csv` 视为“主品牌面板”，`q2_0_japan_brand_sales.csv` 视为“日系子集视图”：
     - 在 `project_document` 中补充它们的构造规则：
       - 如何从官方总销量估算品牌份额；
       - 产地结构（US/Mexico/Japan）是依据何种行业经验或报道设定；
       - 是否对 2020 疫情等年份做了特殊调整。
  2. 若时间允许，可针对少量品牌（如 Toyota、Honda、Ford）从公开统计验证 processed 数据的合理区间，并在文档中给出“误差带”说明，而非逐点完全吻合。
  3. 在代码侧，可新增一个辅助分析脚本（不必修改主模型）：
     - 读取 `q2_0_us_auto_sales_by_brand.csv`；
     - 计算日系品牌份额、产地重心转换路径，与 Q2 现有情景模拟结果做对比图表。

### 6.3 Q2：行业指标 `q2_1_industry_indicators.csv`

- **目标**：
  - 保持其作为“展示友好”的指标表；
  - 同时建立与 `us_auto_indicators.csv`（模型内部使用）的双向映射。
- **建议步骤**：
  1. 新建一个简单转换脚本，例如 `2025/scripts/q2_build_industry_panel.py`：
     - 输入：FRED/BEA 官方原始系列；
     - 输出：
       - `data/external/us_auto_indicators.csv`（模型版，含 `us_auto_production`, `us_auto_employment`, `us_auto_price_index`, `us_gdp_billions`, `fuel_price_index`）；
       - `data/processed/q2/q2_1_industry_indicators.csv`（展示版，包含当前 3 列 +/− 更多派生指标）。
  2. 在文档中同步维护：两者之间的单位换算与公式，让评委可以追踪：
     - 从 FRED ID → us_auto_indicators → q2_1_industry_indicators 的完整链路。

---

## 7. 结论与优先级建议

1. **结构层面**：
   - 所有 processed CSV 在“列结构、缺失值、重复行、基本数值合理性”上均表现良好，可视为 **结构上 READY**；
   - 时间覆盖统一为 2015–2024，与题目所需时间窗口完全对齐。

2. **严格实证建模层面**：
   - **Q1 `q1_0.csv`** 是当前最接近“官方标准面板”的 processed 数据，适合优先补齐来源与脚本说明；
   - **Q2 三个 processed 文件** 为高质量结构化 SAMPLE，更适合作为情景和结构分析输入，而非“官方事实”本身，需要通过文档来限定其角色。

3. **推荐的短期工作优先级**：
   - **P0**：为 `q1_0.csv` 与 `q2_1_industry_indicators.csv` 补齐官方来源说明与处理脚本（至少文档级别）；
   - **P1**：在 `project_document` 中单独撰写 `q2_brand_panel_data_spec.md`，系统说明 `q2_0_*` 两个品牌面板的构造方法与用途；
   - **P2**：视时间和精力，再考虑将 processed 文件系统性纳入现有 Q1/Q2 模型流水线，形成“external → processed → model”的标准模式。

整体来看，`2025/data/processed/` 目录下的 CSV 已经具备 **较高的结构完备度与良好的一致性**，可以作为正式建模与论文撰写时的“标准化数据视图”。若能按上文建议补齐可追溯性与脚本说明，将显著提升项目在严格数学建模和复现性方面的说服力。
