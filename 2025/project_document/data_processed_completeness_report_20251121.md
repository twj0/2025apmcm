# 2025 APMCM C 题 — `2025/data/processed/` 标准数据集完备性分析报告（2025-11-21）

## 1. 项目概述与本报告目的

`2025/data/processed/` 目录下的 CSV 文件，是各子问题（尤其 Q1、Q2）在正式建模和数值分析阶段优先使用的 **标准化数据集**。本报告聚焦：

- 系统清点当前已存在的 processed CSV 文件；
- 从**结构完整性**、**时间覆盖**、**缺失与一致性**三个维度评估其质量；
- 结合 `src/models/q1_soybeans.py` 与 `src/models/q2_autos.py` 等代码的实际调用情况，判断其与建模代码的匹配度；
- 给出严格数学建模视角下的 **建模就绪度等级** 以及后续改进建议。

> 说明：本次分析仅覆盖 `2025/data/processed/` 目录（当前含 Q1/Q2 两类数据），不包括 Q4/Q5 的 Parquet 文件等时间序列。

---

## 2. 总览表：processed CSV 数据清单与状态

### 2.1 文件级统计

下表基于脚本 `2025/scripts/audit_processed_csvs.py` 的扫描结果（在根目录 `.venv` Python 环境下执行）整理而成。

| 文件 | 问题关联 | 行数 | 列数 | 年份范围 | 关键键唯一性 | 缺失值情况 | 数据来源与性质 | 数据质量等级 | 建模就绪度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `2025/data/processed/q1/q1_0.csv` | Q1 中国大豆进口 | 30 | 6 | **2015–2024** | `year, exporter` 唯一；无重复行 | 全列缺失率 0% | 来自海关/官方贸易数据，经 `china_soybeans_manual` 处理与人工校准，**标准化处理数据** | 结构完备、时间连续，无缺失与重复，值域合理 | **READY**（可直接用于 Q1 面板/回归建模） |
| `2025/data/processed/q2/q2_0_us_auto_sales_by_brand.csv` | Q2 美国市场按品牌汽车销量与产地结构 | 120 | 7 | **2015–2024** | `year, brand` 唯一；无重复行 | 全列缺失率 0% | 来自 goodcarbadcar.com 等公开来源汇总，结合产地结构假设拆分，**标准化处理数据** | 结构完备、无缺失，品牌×年份网格完整 | **READY（分析与情景模拟）/NEEDS_REVIEW（严格结构一致性）** |
| `2025/data/processed/q2/q2_0_japan_brand_sales.csv` | Q2 日本品牌在美销量与产地结构子集 | 50 | 7 | **2015–2024** | `year, brand` 唯一；无重复行 | 全列缺失率 0% | 从上一表中筛选 `origin=Japan` 得到的子面板，**标准化处理数据** | 结构完备，year×brand 网格完整 | **READY（日本品牌分析）** |
| `2025/data/processed/q2/q2_1_industry_indicators.csv` | Q2 汽车行业宏观指标（销量、就业、价格指数） | 10 | 4 | **2015–2024** | `year` 唯一；无重复行 | 全列缺失率 0% | 来自 FRED / BLS 等官方宏观指标合成，**官方统计 + 标准化处理数据** | 时间完整、值域合理，无缺失 | **READY**（可直接进入回归/情景分析） |

#### 等级说明

- **数据来源维度**：
  - 官方统计数据：直接源于 FRED、BLS、GACC、USITC 等权威来源；
  - 标准化处理数据：在官方或高可信外部数据基础上，经清洗、汇总、变换后形成的建模就绪版本；
  - 模板/占位数据：为方便代码开发/测试而设定的结构样例或合成数据。

- **建模就绪度维度**：
  - **READY**：结构、时间、缺失和基本一致性均满足严格建模要求，可直接进入回归、面板或情景分析；
  - **NEEDS_REVIEW**：整体结构完备，但存在来源假设或构造逻辑，建议在严肃论文中透明披露并做敏感性分析；
  - **INCOMPLETE**：缺失关键年份、核心变量或结构与代码期望不符，需要先补全/重构再建模。

本轮扫描下，`processed` 目录中 **没有发现 INCOMPLETE 级别的 CSV 文件**，这是一个明显的优点。

---

## 3. 按文件的结构与完整性详细分析

### 3.1 Q1 — `q1/q1_0.csv`（中国大豆进口标准化数据）

- **路径与规模**：
  - 路径：`2025/data/processed/q1/q1_0.csv`
  - 行数：30 行；列数：6 列；文件大小：约 1.3 KB。

- **列结构与类型**（来自审计脚本输出）：

  - `year`：`int64`，无缺失，范围 `[2015, 2024]`；
  - `exporter`：`object`（国家/地区名），无缺失；
  - `import_quantity_tonnes`：`int64`，无缺失；
  - `import_value_usd`：`int64`，无缺失；
  - `unit_price_usd_per_ton`：`int64`，无缺失；
  - `tariff_cn_on_exporter`：`float64`，无缺失，范围 `[0.03, 0.33]`。

- **时间覆盖**：
  - `year` 取值区间为 **2015–2024**，对应 10 个年份；
  - 行数=30，推断每年约 3 个出口国（典型为 US / Brazil / Argentina），与 Q1 文档和 `china_soybeans_manual.validate_soybean_data` 的设计一致。

- **缺失值情况**：
  - 所有列缺失值计数为 0，缺失率均为 0%。

- **重复与键唯一性**：
  - 全行重复：0 行；
  - 主键选择：`(year, exporter)`；
  - `(year, exporter)` 上重复行：0 行。

- **数值合理性（基于简单规则）**：
  - `import_quantity_tonnes`：最小值约 1,500,000 吨，最大值约 74,500,000 吨，均 **非负**，数量级与中国大豆进口市场规模相符；
  - `import_value_usd`：7 亿—438 亿美元区间，数量级合理；
  - `unit_price_usd_per_ton`：335–609 美元/吨，符合 2015–2024 年国际大豆价格水平；
  - `tariff_cn_on_exporter`：0.03–0.33（3%–33%），能体现加税前后差异，且无负数或异常极值。

- **与代码使用的匹配度**：
  - 在 `2025/src/models/q1_soybeans.py` 中：
    - `load_external_china_imports` 优先从 `DATA_PROCESSED / 'q1/q1_0.csv'` 读取；若不存在，则回退到 `data/external/china_imports_soybeans.csv`；
    - 回归与情景分析中，假定存在 `year`、`exporter`、数量/金额/关税等列；
  - 当前 processed 文件列名与上述期望完全匹配，因此与 Q1 代码高度一致。

- **质量评价与就绪度**：
  - 从结构、时间覆盖、缺失与键唯一性看，`q1_0.csv` 已达到 **严格面板数据建模** 要求；
  - 在来源上，虽有人工处理与标准化，但以官方贸易统计为基础，可视为高质量的标准化数据；
  - 评估：
    - **数据质量等级**：官方统计 + 标准化处理；
    - **建模就绪度**：**READY**。

- **潜在注意点（用于论文写作时披露）**：
  - 关税率 `tariff_cn_on_exporter` 的构造方式（例如加税时间点、是否按 HS code 加权）需要在方法章节透明说明；
  - 如有极少数年份/出口国依赖推断或插值，也建议在数据附录中标明。

---

### 3.2 Q2 — `q2_0_us_auto_sales_by_brand.csv`（美国市场各品牌销量与产地结构）

- **路径与规模**：
  - 路径：`2025/data/processed/q2/q2_0_us_auto_sales_by_brand.csv`；
  - 行数：120 行；列数：7 列；文件大小：约 5.6 KB。

- **列结构与类型**：

  - `year`：`int64`，无缺失，范围 `[2015, 2024]`；
  - `brand`：`object`，无缺失（品牌名）；
  - `total_sales`：`int64`，无缺失；
  - `us_produced`：`int64`，无缺失；
  - `mexico_produced`：`int64`，无缺失；
  - `japan_imported`：`int64`，无缺失；
  - `origin`：`object`，无缺失（品牌原籍，如 Japan / US / EU 等）。

- **时间覆盖与面板结构**：
  - `year` 范围为 **2015–2024**；
  - 行数 120 行 = 10 年 × 12 品牌（推断），说明 **year × brand 网格完整**；
  - `year, brand` 作为主键：无重复记录。

- **缺失值情况**：
  - 所有列缺失率均为 0%。

- **重复与键唯一性**：
  - 完整行重复：0 行；
  - `(year, brand)` 上重复：0 行；
  - 满足严格面板建模对键唯一性的要求。

- **数值合理性与潜在一致性问题**：
  - 所有销量/产量列最小值、最大值均为 **正数**，无负值；
  - 但本报告的自动脚本未直接检查 `us_produced + mexico_produced + japan_imported` 是否严格等于 `total_sales`，该一致性需要在后续人工或专门脚本中补充验证；
  - 根据 Q2 文档（如 `project_document/20251121_q2_data_sources_goodcarbadcar.md`）：
    - `total_sales` 来自 goodcarbadcar.com 等公开统计；
    - 三个产地分项是基于某种规则/假设拆分（例如按品牌在各国家生产比例估算），因此具有**构造型**成分。

- **与代码使用的匹配度**：
  - `2025/src/models/q2_autos.py` 当前主要读的是 `data/external/us_auto_sales_by_brand.csv` 和 `us_auto_indicators.csv`；
  - processed 版本 `q2_0_us_auto_sales_by_brand.csv` 与 external 版本列结构高度一致，适合作为：
    - 分析与画图的标准数据源；
    - 若后续将 Q2 模型替换为完全基于 processed 数据的实现，可作为 drop-in replacement。

- **质量评价与就绪度**：
  - 结构层面（时间、缺失、键唯一性）：满足严格面板数据要求，适合作回归或结构估计的输入；
  - 数据来源上，`total_sales` 接近官方统计，但各生产地分项为**基于假设的构造变量**；
  - 因此：
    - **数据质量等级**：标准化处理数据，含构造成分；
    - **建模就绪度**：
      - 就 **描述性分析 / 情景模拟** 而言，可视为 **READY**；
      - 若用于 **结构参数估计或政策评估**，则建议标记为 **NEEDS_REVIEW**，并在论文中说明构造假设与敏感性。

---

### 3.3 Q2 — `q2_0_japan_brand_sales.csv`（日本品牌在美销量与产地结构）

- **路径与规模**：
  - 路径：`2025/data/processed/q2/q2_0_japan_brand_sales.csv`；
  - 行数：50 行；列数：7 列；文件大小：约 2.4 KB。

- **列结构与类型**：与上一表完全相同：
  - `year` (`int64`)、`brand` (`object`)、`total_sales`、`us_produced`、`mexico_produced`、`japan_imported`（均为 `int64`）、`origin` (`object`)；
  - 全列缺失率 0%。

- **时间覆盖与面板结构**：
  - `year` 范围：**2015–2024**；
  - 行数 50 行，结合日本品牌数量（例如 Toyota, Honda, Nissan, Mazda, Subaru 等），推断为“年 × 日本品牌集合”的完整网格；
  - `(year, brand)` 上无重复，满足面板数据唯一性。

- **与上游数据及代码的关系**：
  - 文档与列结构表明，该文件是从 `q2_0_us_auto_sales_by_brand.csv` 中筛选 `origin=Japan` 得到的子集合；
  - 在 Q2 的情景分析中（特别是“日本汽车对美出口 vs 美国本土生产”的对比），日本品牌子集是关键视角；
  - 当前 `q2_autos.py` 仍以 external CSV 为主数据源，processed 日本品牌表可以作为：
    - 绘图与补充分析的数据；
    - 若将来 Q2 模型显式区分日本品牌与其他品牌时的直接数据输入。

- **质量评价与就绪度**：
  - 在结构、时间、缺失和键唯一性维度表现优秀；
  - 由于完全由上游 processed 表过滤得到，其数值逻辑依赖于 `q2_0_us_auto_sales_by_brand.csv` 的假设；
  - 评估：
    - **数据质量等级**：标准化处理数据（含构造成分）；
    - **建模就绪度**：**READY（用于日本品牌相关的描述性 & 回归分析）**。

---

### 3.4 Q2 — `q2_1_industry_indicators.csv`（美国汽车行业宏观指标）

- **路径与规模**：
  - 路径：`2025/data/processed/q2/q2_1_industry_indicators.csv`；
  - 行数：10 行；列数：4 列；文件大小：约 345 B。

- **列结构与类型**：

  - `year`：`int64`，无缺失，范围 `[2015, 2024]`；
  - `total_light_vehicle_sales_million`：`float64`，无缺失，范围约 `[14.23, 17.88]`；
  - `us_auto_employment_thousands`：`float64`，无缺失，范围约 `[900.0, 990.8]`；
  - `us_auto_price_index_1982_100`：`float64`，无缺失，范围约 `[145.8, 175.5]`。

- **时间覆盖与键唯一性**：
  - `year` 覆盖 **2015–2024**，每年 1 行；
  - `(year)` 作为唯一键无重复；
  - 满足典型时间序列回归或面板控制变量的要求。

- **数据来源与合理性**：
  - 指标定义与数值区间与 FRED / BLS 的汽车行业统计高度一致：
    - 轻型汽车销量（百万辆）；
    - 汽车行业就业（千人）；
    - 汽车价格指数（1982=100 基期）；
  - 属于“官方统计 + 稍微整理”的标准化宏观控制变量；
  - 无负值、无极端异常值。

- **与代码使用的匹配度**：
  - `q2_autos.py` 中宏观控制变量目前主要来自 external CSV；
  - 该 processed 表与 external 版本在列名和结构上高度一致，可以：
    - 作为可直接喂入回归/VAR 的数据源；
    - 支持更清晰的“数据来源与整理说明”。

- **质量评价与就绪度**：
  - 因其来源权威、结构简单且无缺失，质量非常高；
  - 评估：
    - **数据质量等级**：官方统计 + 标准化处理；
    - **建模就绪度**：**READY**。

---

## 4. 与建模代码的整体匹配度评估

### 4.1 Q1：`q1_soybeans.py` 与 `q1_0.csv`

- 代码逻辑（简要）：
  - `load_external_china_imports()`：优先读 `DATA_PROCESSED / 'q1/q1_0.csv'`，否则回退到 external 原始数据；
  - 后续估计中国对美大豆关税对进口量/来源结构的影响，需要：
    - 完整的 `(year, exporter)` 面板；
    - `import_quantity_tonnes`、`import_value_usd` 等数值列无缺失；
    - 稳健的 `tariff_cn_on_exporter` 时间路径。

- 当前 `q1_0.csv` 恰好满足上述要求：
  - 年份连续覆盖 2015–2024；
  - 出口国集合与文档设定（US/Brazil/Argentina）一致；
  - 无缺失、无重复；
  - 数值区间合理；

> 结论：在不考虑极端稳健性检验的前提下，`q1_0.csv` 已完全满足 Q1 模型的结构与数据要求，可视为 **主建模数据源**。

### 4.2 Q2：`q2_autos.py` 与 Q2 processed CSV

- 当前代码状态：
  - `load_q2_data()` 主要使用关税面板和贸易流量（imports_panel）；
  - `load_external_auto_data()` 使用 external：
    - `us_auto_sales_by_brand.csv`；
    - `us_auto_indicators.csv`；
  - processed CSV（`q2_0_*.csv`, `q2_1_industry_indicators.csv`）在模型中还没有完全替代 external 路径，但列结构匹配。

- 从数据侧看：
  - processed Q2 表相比 external：
    - 列名更统一，便于文档和分析；
    - 明确了年份范围（2015–2024）、品牌/年份网格完整性；
    - 将行业宏观指标集中到一个小表中，方便作为控制变量。

> 结论：
>
> - 若保持当前实现：processed Q2 CSV 是 **高质量的分析数据集**，适合绘制图表和事后验证；
> - 若未来希望将 Q2 模型完全基于 `data/processed` 统一输入，只需在 `q2_autos.py` 中将 external 路径替换为 processed 路径或增加一个 config 开关即可，结构层面无需额外改造。

---

## 5. 建模就绪度综合评估

结合前述分析，针对 `2025/data/processed/` 当前四个 CSV 文件的整体评价如下：

1. **结构与时间维度**：
   - 所有文件在其定义的业务范围内，年份都是 **2015–2024 连续覆盖**；
   - 每个文件都具有明确的主键（`year, exporter` 或 `year, brand` 或 `year`），且无重复；
   - 无任何列存在缺失值。

2. **数据来源与构造程度**：
   - Q1 `q1_0.csv`：以官方贸易数据为基础，属于“官方 + 标准化处理”；
   - Q2 `q2_1_industry_indicators.csv`：直接源于 FRED/BLS，属于可靠官方统计；
   - Q2 `q2_0_*_sales.csv`：销量总量接近官方或权威第三方统计，但产地分解具有构造成分，需要在论文中说明假设。

3. **严格数学建模视角的就绪度**：

   - **完全 READY（可直接作为严肃实证分析的基础数据）**：
     - `q1/q1_0.csv`；
     - `q2/q2_1_industry_indicators.csv`；
   - **READY（用于描述性与情景模拟）+ NEEDS_REVIEW（用于结构性因果推断前需额外说明）**：
     - `q2/q2_0_us_auto_sales_by_brand.csv`；
     - `q2/q2_0_japan_brand_sales.csv`。

4. **当前未发现的严重问题**：
   - 未发现 INCOMPLETE 类缺口（如年份缺失、大量 NA、键冲突）；
   - 未发现明显的负值或量纲级别错误。

---

## 6. 详细问题与潜在改进点清单

虽然当前 processed CSV 整体质量较高，但从“论文级严格性”和“可复现性”角度，仍存在若干需要记录或改善的点：

### 6.1 Q1 `q1_0.csv`

- **需要在文档中进一步澄清的点**：
  1. `tariff_cn_on_exporter` 的构造逻辑：
     - 是否按 HS code 聚合（如 1201 大豆）？
     - 中间是否经过贸易加权？
     - 对于同一出口国在不同年份是否存在插值或近似？
  2. 数据来源链路：
     - GACC / UN Comtrade 等原始数据的获取方式；
     - 与 `2025/data/raw/` 与 `data/external/china_imports_soybeans.csv` 的映射关系。

- **改进建议**：
  - 在 `project_document/` 中为 Q1 新增一份专门的数据说明（若尚未存在）：详细记录 `q1_0.csv` 的来源、处理步骤与质量校验结果（可复用 `china_soybeans_manual.validate_soybean_data` 的检查逻辑）。

### 6.2 Q2 `q2_0_us_auto_sales_by_brand.csv` & `q2_0_japan_brand_sales.csv`

- **结构性潜在问题（待核查）**：
  1. **产地分解和总销量的一致性**：
     - 需要验证是否在每一行满足 `us_produced + mexico_produced + japan_imported == total_sales`（或在数值上近似相等）；
     - 如存在系统性偏差，需要在数据说明中解释分解规则（例如四舍五入导致的小差异）。
  2. **品牌集合的稳定性**：
     - 检查 2015–2024 年间品牌集合是否稳定，是否存在中途进入/退出的品牌；
     - 如品牌集合发生变化，在建模时需要考虑 unbalanced panel 处理或对空缺年份做显式标记。

- **方法论层面的注意点**：
  - `origin` 和产地分解列属于**构造型变量**：
    - 多数来自对公开资料和常识假设的推断，而非官方逐品牌产地统计；
    - 在做因果推断时，应当将其视为一种“基于假设的情景设定”，而不是硬事实数据。

- **改进建议**：
  1. 实现一个小型验证脚本，对 `q2_0_*` 两个文件逐行校验产地分解与总销量的一致性，输出偏差分布；
  2. 在 Q2 数据说明文档中补充：
     - 产地分解的设定规则（如固定比例、时间趋势、逐品牌特征等）；
     - 对结果敏感性的讨论（例如测试不同分解方案对回归系数的影响）。

### 6.3 Q2 `q2_1_industry_indicators.csv`

- **潜在增强点**：
  1. 当前指标集中在销量、就业、价格指数三个维度，可以考虑补充：
     - 行业产值或增加值（value added）；
     - 行业资本存量或投资；
     - 更细的就业结构（如制造 vs 销售/服务）。
  2. 若未来需要更复杂的时间序列模型，可以将该表扩展为季度或月度数据，并在 `processed` 目录中增加相应版本。

---

## 7. 综合结论与后续建议

### 7.1 综合结论

- `2025/data/processed/` 目前仅包含 Q1 和 Q2 的四个 CSV 文件，但 **整体质量较高，结构完备，时间连续，无缺失与键冲突**；
- 从严格数学建模角度：
  - `q1_0.csv` 和 `q2_1_industry_indicators.csv` 可以视为 **完全 READY** 的建模数据集；
  - `q2_0_us_auto_sales_by_brand.csv` 与 `q2_0_japan_brand_sales.csv` 在描述性和情景分析上 READY，但由于包含基于规则构造的分解变量，在做因果推断前需要**明确披露假设并做敏感性检验**；
- 未发现需要立刻修复的“结构性错误”（如年份缺失、大规模 NA、主键不唯一等）。

### 7.2 建议的后续步骤（优先级排序）

1. **（高优先级）为 Q1/Q2 processed 数据撰写/完善数据说明文档**  
   - 在 `project_document/` 下新增或完善：
     - Q1：`q1_china_soybeans_data_notes.md`（建议文件名）；
     - Q2：已存在的 `20251121_q2_data_sources_goodcarbadcar.md` 可补充产地分解与敏感性说明；
   - 明确每个关键列的含义、来源、处理步骤和潜在局限。

2. **（中优先级）为 Q2 processed 数据实现一致性校验脚本**  
   - 在 `2025/scripts/` 下新增如 `validate_q2_processed_data.py`：
     - 检查 `us_produced + mexico_produced + japan_imported` 与 `total_sales` 的关系；
     - 输出偏离分布与最坏/最佳情形；
     - 可选地根据规则自动做微调（如按比例缩放以保证和为 `total_sales`）。

3. **（中优先级）在 Q2 模型中增加“processed vs external”的配置开关**  
   - 修改 `q2_autos.py`，允许通过 config 或参数选择：
     - 使用 external CSV（保持现状）；
     - 或统一使用 `data/processed/q2/*.csv`，以提升数据路径一致性。

4. **（后续增强）扩展 processed 层到 Q3/Q4/Q5**  
   - 目前 Q3/Q4/Q5 的核心数据大多以 external CSV 或 parquet 形式存在；
   - 若时间允许，可以为这些模块也建立清晰的 `data/processed/q3/`, `q4/`, `q5/` 标准化输出，并沿用本报告的审计流程，形成完整的数据质量闭环。

---

本报告可作为你在撰写论文“数据与方法”章节时的参考框架：
- 表 2.1 的总览表可以直接简化为论文中的“主要数据集一览表”；
- 第 3–6 节的内容可以融入数据描述和数据质量/稳健性讨论；
- 第 7 节的建议可演化为未来工作或方法扩展的论述。

