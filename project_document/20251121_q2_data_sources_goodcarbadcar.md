# Q2 数据来源说明：美国汽车品牌销量（goodcarbadcar.com）

> 适用范围：为 APMCM 2025 C 题 Q2（美国汽车市场与日本品牌产能调整）提供的数据来源与处理说明。
>
> 主要来源：**goodcarbadcar.com 等公开 web 数据** + FRED 官方宏观/行业数据。

---

## 1. 数据使用场景与对应文件

本说明文档覆盖的 Q2 相关数据，主要用于两类分析：

- **品牌结构与产地布局**（微观）：
  - 日本品牌在美国市场的销量、产地结构（美国生产 / 墨西哥生产 / 日本直接出口）。
  - 对应文件：
    - `2025/data/external/us_auto_sales_by_brand_complete.csv`
    - `2025/data/processed/q2/q2_0_japan_brand_sales.csv`
- **美国汽车行业总体表现**（宏观）：
  - 轻型车总销量、汽车行业就业、价格指数等。
  - 对应文件：
    - `2025/data/external/us_total_light_vehicle_sales_official.csv`（FRED）
    - `2025/data/external/us_auto_official_indicators_2015_2024.csv`
    - `2025/data/processed/q2/q2_1_industry_indicators.csv`

本说明重点记录 **品牌销量部分的 web 数据来源与处理流程**，以支撑：

- `project_document/202511211030q2_analysis_paper.md`
- `project_document/202511211030q2_appendix_tables.md`

中使用到的各年份品牌销量、市场份额与增长率分析。

---

## 2. 品牌销量数据来源：goodcarbadcar.com

### 2.1 网站概况

- **网站名称**：Good Car Bad Car
- **网站类型**：公开的美国/加拿大汽车市场统计与评测网站
- **数据特点**：
  - 提供按品牌、按车型的年度销量统计；
  - 对美国市场具有较高的覆盖度和时效性；
  - 在分析师和媒体中被广泛引用（但并非政府官方统计）。

### 2.2 采集年份与品牌范围

- **时间范围**：
  - Q2 模型使用历史数据：**2015–2024 年**年度销量；
  - Q2 分析论文与附录中，额外使用了 **2024–2025 年**的品牌销量 web 数据做趋势与预测示例。

- **品牌范围**（与 `us_auto_sales_by_brand_complete.csv` 一致）：
  - 美系：Ford, Chevrolet 等；
  - 日系：Toyota, Honda, Nissan, Subaru, Mazda, Lexus 等；
  - 韩系：Hyundai, Kia, Genesis 等；
  - 欧系：Volkswagen, BMW, Mercedes-Benz, Audi, Volvo, Porsche 等。

> 说明：在 `q2_0_japan_brand_sales.csv` 中，目前仅保留了 Q2 建模最关注的 **日本品牌** 子集（Toyota, Honda, Nissan, Subaru, Mazda），其底层数据均来自 `us_auto_sales_by_brand_complete.csv` 中对应行。

---

## 3. 数据整理与汇总流程

### 3.1 从 web 到原始表

1. **按年份下载/记录品牌销量**：
   - 在 goodcarbadcar.com 网站上，逐年查找美国市场 **按品牌的年度销量表**；
   - 对每一年，将表格数据拷贝/导出为本地表格（Excel/CSV），统一保存为原始采集文件（未纳入仓库）。

2. **统一品牌命名**：
   - 将网页中的品牌名（如 `TOYOTA`, `Toyota Motor` 等）统一映射为规范写法：`Toyota`, `Ford`, `Hyundai` 等；
   - 对合并/拆分品牌做适当处理（如部分小众品牌并入 `Others`，当前仓库中的主表未保留 `Others` 行）。

3. **形成品牌年度长表**：
   - 将各年品牌销量合并为一张长表：
     - 列结构：`year, brand, total_sales`；
     - 时间覆盖：2015–2024 年（以及论文附录中的 2024–2025 年数据）。

### 3.2 构造 `us_auto_sales_by_brand_complete.csv`

在基本的 `year, brand, total_sales` 表基础上，结合行业报道与对日系品牌产能布局的假设，构造：

- 文件：`2025/data/external/us_auto_sales_by_brand_complete.csv`
- 列结构：

```text
year,brand,total_sales,us_produced,mexico_produced,japan_imported,origin
```

- 字段含义：
  - `total_sales`：来自 goodcarbadcar 的品牌年销量（辆）；
  - `us_produced`：估计由美国本土工厂生产并在美销售的部分；
  - `mexico_produced`：估计由墨西哥工厂生产、再出口到美国的部分；
  - `japan_imported`：估计由日本本土直接出口到美国的部分；
  - `origin`：品牌主要所属国家（US / Japan / Korea / EU）。

> 注：`us_produced / mexico_produced / japan_imported` 的具体比例并非来自官方统计，而是基于公开资料与行业常识设定的**情景拆分**，用于 Q2 关于“产能迁移”和“绕关税路径”的情景分析。

### 3.3 派生 Q2 专用标准表

为直接服务于 APMCM C 题 Q2 建模，基于 `us_auto_sales_by_brand_complete.csv` 派生出：

1. **日本品牌销量与产地结构表**：
   - 文件：`2025/data/processed/q2/q2_0_japan_brand_sales.csv`
   - 内容：从 complete 表中筛选 `origin == 'Japan'` 的品牌，保留：

```text
year,brand,total_sales,us_produced,mexico_produced,japan_imported,origin
```

   - 用途：
     - 量化“日本品牌在美国市场中，直供日本 / 墨西哥生产 / 美国本土生产”的结构与演变；
     - 作为 Q2 中日本企业 FDI 与产能调整情景分析的微观基础。

2. **行业指标表**：
   - 文件：`2025/data/processed/q2/q2_1_industry_indicators.csv`
   - 来源：
     - `us_total_light_vehicle_sales_official.csv`（FRED TOTALSA）
     - `us_auto_official_indicators_2015_2024.csv`（就业与价格指数，基于 FRED/BLS 系列整理）
   - 列结构：

```text
year,total_light_vehicle_sales_million,us_auto_employment_thousands,us_auto_price_index_1982_100
```

   - 用途：
     - 与 USITC 关税与进口数据结合，构造进口渗透率与行业传导模型：
       - `Y_US = f(ImportPenetration, Macro/Industry Indicators)`。

---

## 4. 与 FRED 总量数据的一致性检查

为避免 web 品牌销量与官方总量之间出现严重偏差，我们建议并已在数据质量报告中说明：

- 对每一年，计算 `us_auto_sales_by_brand_complete.csv` 中 `total_sales` 按品牌求和：
  - `sum_brand_sales(year)`
- 与 FRED 的轻型车总销量（`TOTALSA`，单位百万辆）进行近似对比：
  - `TOTALSA(year) * 1e6`
- 允许存在一定差异（统计口径与品牌覆盖差异），但应在合理范围内（例如 5–10% 内）；
- 若差异较大，应在分析中说明原因，或对品牌数据做适当缩放/调整。

---

## 5. 数据在 Q2 分析论文中的使用方式

在以下文档中，所有“品牌销量”“市场份额”“增长率”以及 2026 年预测结果，均基于上述流程整理的 web 品牌销量数据：

- `project_document/202511211030q2_analysis_paper.md`
- `project_document/202511211030q2_appendix_tables.md`

具体对应关系：

- 附录表 1、表 2、表 3、表 6 等：
  - 基于 2024–2025 年品牌年销量数据（来源：goodcarbadcar.com），按品牌/品牌类别汇总与计算增长率；
- 市场预测部分：
  - 使用 2024–2025 年销量时间序列建立简单回归模型，对 2026 年做短期预测；
- 行业背景与宏观趋势：
  - 结合 FRED 宏观与行业指标（见 `q2_1_industry_indicators.csv`），用于定性解释品牌表现差异。

---

## 6. 数据质量与局限性

- **完整性**：
  - goodcarbadcar 提供的品牌列表覆盖美国主要乘用车品牌；极小规模/小众品牌可能未完全覆盖或被合并到 `Others`。
- **准确性**：
  - 网站数据通常基于企业公布的销量或官方注册数据，整体可信，但非“官方统计局”渠道。
- **时效性**：
  - 品牌销量数据更新及时，适合短期趋势分析与预测。
- **一致性**：
  - 本项目中已对品牌命名与单位进行了统一处理；
  - 与 FRED 总量数据的一致性在数据质量报告中做了抽样检查。

**局限性与处理建议**：

- 品牌销量数据不是政府官方统计，在论文中应表述为“来源于公开 web 数据（goodcarbadcar.com）”；
- 品牌产地拆分（us/mexico/japan）基于情景假设与行业常识，而非逐车 VIN 级精确统计；
- 对关税与 USITC 数据的链接是在 HS 级别上建立的，品牌到 HS 的映射为间接推断。

在撰写最终 APMCM 论文时，建议在 **数据与方法** 部分引用本说明，清楚区分：

- 官方统计（FRED、USITC 等）；
- 企业/行业级真实品牌销量（goodcarbadcar.com 等）；
- 基于上述数据构造的情景与模型假设（如产地拆分与未来年份预测）。
