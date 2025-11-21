# 2025 APMCM C 数据缺口与爬虫极限说明（截至 2025-11-20 20:14）

> 面向：本仓库内所有智能体（mathcoder / modeler / paperwriter / checker / prompter 等）
>
> 目的：统一说明**目前已经有哪些数据**、**爬虫和脚本已经做到什么程度**、**距离真正支撑五个问题还缺哪些关键数据**，以及**每种缺口数据的 CSV 结构和官方信息源**。

---

## 0. 全局结论（ TL;DR ）

- **架构层面**：数据目录分层、字段设计、代码流水线（Q1–Q5）已经完备，可以一键从 `2025/` 生成图和 JSON。
- **已有的高质量官方数据**：
  - 美国宏观 & 金融 & 半导体 & 部分汽车总量：通过 `data_fetch.py` + FRED，得到 10 条官方时间序列（2015–2024），质量优秀。
  - Tariff Data（USITC）：2020–2025 USITC DataWeb 官方贸易与关税数据已经在 `2025/problems/Tariff Data` 下完全就位。
- **核心缺口**：
  - **Q1**：缺少基于 **中国海关/UN Comtrade/WITS** 的中国大豆进口按来源国官方数据（当前只有 SAMPLE）。
  - **Q2**：缺少 **品牌/来源国结构** 的美国汽车市场数据（目前只有总量级 FRED 指标 + SAMPLE 品牌拆分）。
  - **Q3**：半导体产出有官方指数，但 **细分段（高/中/低端）结构** 和 **政策时间线编码** 仍为 SAMPLE。
  - **Q4**：关税收入需要由 USITC 原始数据 **二次计算**，现有 `q4_avg_tariff_by_year.csv` 为假设曲线。
  - **Q5**：宏观与金融官方数据质量高，但 **制造业回流指标** 与 **反制指数** 仍为估算结构。
- **爬虫极限**：
  - FRED 与 USITC 一类 REST/CSV API 已经被 `data_fetch.py` 等完全打通。
  - **UN Comtrade API 当前反复返回 HTML**（见 `failed_downloads.jsonl`），在本计算环境下已经到达可靠性上限。
  - WITS 和中国海关数据在线查询均为高交互网页，**难以自动爬取**，需要人工下载 + 仓库内清洗脚本配合。

下面按问题（Q1–Q5）系统列出：
1. 已有的数据及质量评价
2. 缺失或不可靠的数据
3. 推荐的 CSV 结构（字段名、类型、单位）
4. 推荐的官方数据源与获取方式

---

## 1. Q1 – 中国大豆进口与三国产业影响

### 1.1 已有数据与质量

- `2025/data/external/china_imports_soybeans.csv`
  - **性质**：由 `external_data.ensure_q1_external_data()` 自动生成的 **SAMPLE** 数据。
  - **结构**：
    - `year, exporter, import_value_usd, import_quantity_tonnes, tariff_cn_on_exporter`
  - **时间范围**：2020–2024
  - **质量评价**：
    - 结构正确，可用于代码演示和函数联调。
    - 数值为假设值，**不能作为论文实证依据**。
- `2025/data/external/hs_soybeans.csv`
  - 提供 HS 级别的大豆代码映射（用于和 USITC / Comtrade 对接）。
  - 结构简单，质量 **EXCELLENT**（无缺失）。

### 1.2 爬虫与脚本的当前极限

- `data_fetch.fetch_un_comtrade_soybeans()` 已实现 **UN Comtrade+ API 的标准调用流程**，包括：
  - 分页、多次重试、Content-Type 检查、JSON 解析、错误日志 `failed_downloads.jsonl`。
- 实际执行结果（见 `failed_downloads.jsonl`）：
  - 多次请求均返回 HTML 页面（登录页 / 网关重定向）。
  - Content-Type=`text/html`，`invalid_json` / `non_json_response` 错误频发。
  - 在当前环境下，**UN Comtrade API 基本不可用**。
- World Bank / WITS：
  - `worldbank_wits.py` 已提供 **WITS 本地 CSV 标准化工具**，但不包含自动登录和任务提交逻辑。
  - WITS 本身需要交互式登录 + 邮件下载链接，超出“纯代码爬虫”的可控范围。

### 1.3 仍然缺少的关键数据（Q1）

> 用于严肃回答 Q1 的“数据基座”，我们还缺：

1. **中国按来源国分解的大豆进口量与进口额**（年度）
   - 国家至少包括：US, Brazil, Argentina（可拓展至 Top N）。
   - 时间覆盖：**2015–2024**（与 USITC/FRED 对齐）。
2. **中国对不同来源国大豆的有效关税率**（含贸易战期加征）
   - 基本 MFN 税率
   - 针对美国的反制关税（2018–2020 及后续调整）

### 1.4 推荐 CSV 结构（权威标准）

目标文件：`2025/data/external/china_imports_soybeans_official.csv`

```csv
year,exporter,import_value_usd,import_quantity_tonnes,tariff_cn_on_exporter
```

**字段定义（严格）**：

- `year`  
  - int，四位年份，至少 2015–2024  全覆盖。
  - 与 `exporter` 共同形成主键（唯一约束）。

- `exporter`  
  - string，出口国英文名，核心三国：`US`, `Brazil`, `Argentina`。  
  - 建议固定写法，区分大小写。

- `import_value_usd`  
  - float，**美元金额**。  
  - 若源数据为人民币，需要统一汇率换算（并在文档中说明）。

- `import_quantity_tonnes`  
  - float，**公吨（metric tonnes）**。  
  - 若源数据为千克，需除以 1000。

- `tariff_cn_on_exporter`  
  - float，0–1 之间的比例税率。  
  - 含基本 MFN + 任何附加关税。

行级约束：
- `(year, exporter)` 唯一。
- 五列都不允许空值。

### 1.5 推荐官方数据源与获取方式

1. **中国海关总署（GACC）** – 推荐优先级：⭐⭐⭐⭐⭐
   - 网址：<http://www.customs.gov.cn/customs/302249/zfxxgk/2799825/302274/index.html>
   - 路径：主页 → “数说海关 / 统计数据” → “数据在线查询”或统计分析司页面。
   - 操作：
     - 选择商品 HS 1201（或“大豆”关键字）。
     - 选择 Reporter=China，按 **Partner** 分解。
     - 时间：2015–2024，频率=年度。
     - 导出 Excel/CSV，存入 `2025/data/raw/`，再通过 `china_soybeans_manual.py process` 清洗。

2. **UN Comtrade Web 界面（非 API）** – 推荐优先级：⭐⭐⭐⭐
   - 网址：<https://comtradeplus.un.org/>
   - 操作：Web GUI 选择：
     - Reporter=China；
     - Partner=US/Brazil/Argentina/All；
     - HS 1201；
     - Flow=Imports；
     - Years=2015–2024；
     - 下载 CSV；
     - 本地清洗为上述 CSV 结构。

3. **World Bank WITS Bulk Download** – 推荐优先级：⭐⭐⭐
   - 网址：<https://wits.worldbank.org/>
   - 通过 Advanced Query 使用 UN Comtrade 或 TRAINS 数据；
   - 下载 CSV 后用 `worldbank_wits.standardize_wits_trade_csv()` 清洗出：`year, partner, trade_value`，再映射到目标结构。

---

## 2. Q2 – 美国汽车市场与日本车地位

### 2.1 已有数据与质量

**官方数据（FRED，经 `data_fetch.py` 获得）：**

- `us_total_light_vehicle_sales_official.csv` (TOTALSA)  
- `us_motor_vehicle_retail_sales_official.csv` (MRTSSM441USN)

特征：
- `series_id,date,year,value`，2015–2024，全程无缺失，质量 **EXCELLENT**。

**示例数据（SAMPLE）：**

- `us_auto_sales_by_brand.csv`  
  - 品牌拆分（Toyota/Honda/Nissan/Ford）、美/墨/日生产结构，完全为脚本模拟。  
- `us_auto_indicators.csv`  
  - 美国汽车产量、就业、价格指数等，全部为示例。

### 2.2 仍然缺少的关键数据（Q2）

1. **美国汽车市场按品牌/源地结构的销量**：
   - 至少区分：
     - 日本品牌（Toyota, Honda, Nissan 等）；
     - 生产地：美国本土、墨西哥、日本出口；
   - 时间范围：2015–2024。
2. **美国汽车产业相关宏观 / 行业指标的官方版本**：
   - 汽车及零部件出货量（BEA）
   - 汽车零售/批发销售（Census）

### 2.3 推荐 CSV 结构

#### 2.3.1 品牌级销量（示例结构）

目标文件：`us_auto_sales_by_brand_official.csv`

```csv
year,brand,total_sales,us_produced,mexico_produced,japan_imported,other_imported
```

- 单位：辆（unit counts）。
- `year`：int，2015–2024。
- `brand`：string，品牌名（Toyota/Honda/Nissan/Ford/...）。
- 其余列：float/int，非负，**行内各产地之和 ≈ total_sales**。

#### 2.3.2 行业指标（宏观层面）

目标文件：`us_auto_indicators_official.csv`

```csv
year,us_auto_production,us_auto_employment,us_auto_price_index,us_auto_exports,us_auto_imports
```

- `us_auto_production`：年产量（辆）。
- `us_auto_employment`：汽车行业就业（人数）。
- `us_auto_price_index`：价格指数（2015=100）。
- `us_auto_exports` / `us_auto_imports`：贸易额（USD）。

### 2.4 推荐官方数据源

1. **FRED（统一入口）** – 指标已经部分接入
   - <https://fred.stlouisfed.org/>
   - 通过 Table/Series ID 映射 BEA/Census 数据。

2. **美国经济分析局（BEA）**
   - <https://www.bea.gov/>
   - 行业增加值、出货量等。

3. **美国人口普查局（Census）**
   - 零售/批发贸易统计，汽车及零部件子类。

4. **私营数据库/行业报告**
   - 用于品牌拆分时可作为**估算依据**，但须在论文中明确标注“非官方”。

---

## 3. Q3 – 半导体产业与关税/补贴/管制

### 3.1 已有数据

**官方指标（FRED）：**

- `us_semiconductor_output_index_official.csv`
  - Series: `IPUEN3344T300000000`
  - 结构：`series_id,date,year,value`，2015–2024，无缺失。

**示例数据：**

- `us_semiconductor_output.csv`（分段高/中/低）
- `us_chip_policies.csv`（CHIPS/出口管制指标）

### 3.2 缺口

1. **半导体产出按高/中/低端分段的真实数据或合理 proxy**。
2. **CHIPS 法案、2025 关税与出口管制等政策时间线** → 需转化为结构化指标（subsidy index / export_control_index 等）。

### 3.3 推荐 CSV 结构

#### 3.3.1 段别产出

目标文件：`us_semiconductor_output_official.csv`

```csv
year,segment,us_chip_output_billions,global_chip_demand_index
```

- `segment` ∈ {"high","mid","low"} 或更细分类别。
- `us_chip_output_billions`：美国该段产出（十亿美元）。
- `global_chip_demand_index`：全球需求指数（2015=100）。

#### 3.3.2 政策时间线

目标文件：`us_chip_policies_official.csv`

```csv
year,subsidy_index,export_control_china,tariff_on_chips
```

- `subsidy_index`：0–10，衡量政策补贴强度。
- `export_control_china`：0/1 或分级指数。
- `tariff_on_chips`：对芯片类商品的平均有效关税（比例）。

### 3.4 官方信息源

- 白宫 Fact Sheet / News：CHIPS & Science Act 相关材料。  
- Congress.gov：法案文本。  
- 美国商务部：CHIPS 资金分配、项目公告。  
- 半导体行业协会（SIA）：年度报告（销售/产出统计）。

---

## 4. Q4 – 关税收入与动态效应

### 4.1 已有数据

- `2025/problems/Tariff Data/` 目录：
  - USITC DataWeb 商品 × 国家 × 年度的进出口/关税数据（2020–2025）。
  - 2015–2025 年 Tariff Database CSV（已从官方 Excel 转换）。
- `q4_avg_tariff_by_year.csv`：
  - 目前为 **线性插值的假设曲线**，非官方结果。

### 4.2 缺口

1. **基于 USITC 数据计算的真实年度关税收入表**：
   - 按年份汇总所有 HS × 国家上的关税收入。 
2. **基于 HS 级别数据计算的贸易加权平均关税率**：
   - 可按产品组或总体给出。

### 4.3 推荐 CSV 结构

#### 4.3.1 年度关税收入

目标文件：`q4_tariff_revenue_official.csv`

```csv
year,total_import_value_usd,total_tariff_revenue_usd,avg_effective_tariff
```

- `total_import_value_usd`：美方总进口额（USD）。
- `total_tariff_revenue_usd`：总关税收入（USD）。
- `avg_effective_tariff` = 收入 / 进口额（比例）。

### 4.4 官方信息源

- USITC DataWeb：<https://dataweb.usitc.gov/>  
  - 已通过题目附件给出完整导出文件。
- Yale Budget Lab 报告可用于交叉验证。

---

## 5. Q5 – 宏观、金融、回流与反制

### 5.1 已有数据

**官方（FRED，经 data_fetch）：**
- us_real_gdp_official.csv
- us_cpi_official.csv
- us_unemployment_rate_official.csv
- us_industrial_production_official.csv
- us_federal_funds_rate_official.csv
- us_treasury_10y_yield_official.csv
- us_sp500_index_official.csv
- us_semiconductor_output_index_official.csv

**整合数据：**
- `us_macro_consolidated.csv`：已生成宽表，2015–2024，7 指标，质量 EXCELLENT。

**示例（SAMPLE）：**
- `us_macro.csv`, `us_financial.csv`, `us_reshoring.csv`, `retaliation_index.csv`

### 5.2 缺口

1. **制造业回流官方 proxy**：
   - 制造业增加值占 GDP 比重（BEA）。
   - 制造业就业占比（BLS）。
   - 制造业 FDI / Reshoring Index（可用官方加行业报告）。
2. **反制指数**：
   - 需要基于 MOFCOM 公告、海关加征关税文件，构造年度 0–10 的“反制强度”指数。

### 5.3 推荐 CSV 结构

#### 5.3.1 宏观（已基本实现）

目标文件：`us_macro_official.csv`

```csv
year,gdp_growth,real_gdp_index,industrial_production_index,unemployment_rate,cpi_index
```

#### 5.3.2 制造业回流

目标文件：`us_reshoring_official.csv`

```csv
year,manufacturing_va_share,manufacturing_employment_share,reshoring_fdi_billions
```

#### 5.3.3 反制指数

目标文件：`retaliation_index_official.csv`

```csv
year,retaliation_index
```

- `retaliation_index`：0–10 的年度反制强度分数，由政策事件人工编码。

### 5.4 官方信息源

- FRED（统一宏观/金融入口）。
- BEA/BLS：制造业增加值和就业统计。
- MOFCOM（商务部）：反制关税、出口管制公告。  
  - <https://www.mofcom.gov.cn/zcfb/blgg/index.html>

---

## 6. 对“所有智能体”的操作建议

1. **mathcoder / modeler**
   - 在使用任何 `*_sample.csv` 数据时，必须在模型说明中明确标注“示例数据，非官方统计”。
   - 对 Q1、Q2、Q4 的敏感结论，应在官方数据补齐后再次运行模型。

2. **paperwriter**
   - 引用数据时，优先使用 `*_official.csv` 和 `us_macro_consolidated.csv`。  
   - 在“数据与方法”章节中，明确列出：
     - 官方数据源（USITC/FRED/GACC 等）；
     - 仍为估算的数据（如回流指数、品牌拆分等）。

3. **checker**
   - 对所有使用到的 CSV，检查是否：
     - 存在 `_official` 对应文件；
     - 结构是否匹配本文件定义；
     - 是否误用 SAMPLE 数据作为政策结论依据。

4. **prompter**
   - 在生成与“爬虫”“数据获取”相关的 Prompt 时，显式提醒：
     - UN Comtrade API 已在本环境下达到极限；
     - WITS/GACC 需要人工导出 + 仓库内部清洗。

---

**时间戳**：本说明基于仓库状态和脚本运行结果，更新至 **2025-11-20 20:14 (UTC+08)**。  
如后续补齐了任何 `*_official.csv` 或新增脚本，请在本文件上游的 `data_plan_status_*.md` 中同步维护。
