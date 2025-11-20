# 数据来源与使用方式（2025-11-20）

本文件汇总当前项目使用或计划使用的主要数据来源、数据在模型中的作用，以及配套脚本/使用方法，便于团队成员快速定位高质量数据并保持可追溯性。

## 1. 官方贸易/关税数据

| 数据集 | 来源 | 作用 | 获取与使用方式 |
| --- | --- | --- | --- |
| **USITC Tariff Data** (`2025/problems/Tariff Data/**`) | 美国国际贸易委员会（DataWeb导出） | Q2–Q4 的进口关税、贸易额、有效税率计算 | 通过 `TariffDataLoader`（`src/utils/data_loader.py`）标准化列名、熔融年份列后生成 `imports_panel.parquet`、`exports_panel.parquet` 等供模型使用。未经处理的宽表**不要直接用于交互式分析**。 |
| **UN Comtrade / WITS 大豆进口数据** | 联合国 Comtrade 数据库；若 API 受限，可在 WITS 高级查询中批量下载 | Q1 的核心输入：按照年份×出口国划分的中国（HS1201）进口量/金额 | - 若 API 可用：运行 `python 2025/src/utils/data_fetch.py --datasets q1_china_soybean_imports`（需 `UN_COMTRADE_API_KEY`）。<br>- 若使用 WITS 下载：先在网页导出 CSV，然后运行 `uv run python 2025/src/utils/worldbank_wits.py wits <输入CSV> --output 2025/data/external/china_imports_soybeans_official.csv`；最后执行 `uv run python 2025/scripts/preprocess_china_soybeans_official.py` 生成 Q1 直接使用的 `china_imports_soybeans.csv`。 |
| **中国对出口国大豆关税** (`2025/data/external/china_soybean_tariffs.csv` 可选) | 自行整理（可来自 TRAINS/WITS 或政策资料） | Q1 中用于构造 `tariff_cn_on_exporter` 列，支持含税/不含税价格比较与政策情景 | 在 `preprocess_china_soybeans_official.py` 中通过 `--tariff-file` 传入；若缺失则按 0.0 填充并给出警告。 |

## 2. 宏观 / 金融 / 关税均值指标

| 数据集 | 来源 | 作用 | 获取与使用方式 |
| --- | --- | --- | --- |
| **FRED 官方时间序列** (`2025/data/external/*_official.csv`) | 美国圣路易斯联储 (FRED) | Q2–Q5 的宏观/行业指标（CPI、失业率、INDPRO、TOTALSA、SP500 等） | `src/utils/data_fetch.py` 中的 `fetch_fred_series` 已封装；运行 `python 2025/src/utils/data_fetch.py --datasets q5_cpi ...` 或 `--datasets all` 获取并落盘。注意 `us_sp500_index_official.csv` 2015 年存在缺失值，分析前需预处理。 |
| **World Bank 指标（如平均关税率）** (`2025/data/external/wb_*.csv`) | 世界银行 Open Data (`wbdata` API) | Q4/Q5 中的补充指标（如 `TM.TAX.MANF.SM.AR.ZS`） | 运行 `uv run python 2025/src/utils/worldbank_wits.py wb <indicator> --country CHN --start-year 2015 --end-year 2024 --output 2025/data/external/<file>.csv`。示例：`TM.TAX.MANF.SM.AR.ZS` 已生成 `wb_tariff_mean_china_2015_2024.csv`。 |

## 3. 项目内置的结构化样本数据 (`2025/data/external/*.csv`)

这些数据由 `src/utils/external_data.py` 自动生成，仅用于在没有官方数据时让模型“可运行、可展示”。使用时需在文档或论文中注明为“模拟/样本数据”。

- `china_imports_soybeans.csv`（若未被官方数据覆盖）
- `us_auto_sales_by_brand.csv`、`us_auto_indicators.csv`（Q2）
- `us_semiconductor_output.csv`、`us_chip_policies.csv`（Q3）
- `q4_avg_tariff_by_year.csv`、`q4_dynamic_import_params.json`、`q4_tariff_scenarios.json`
- `us_macro.csv`、`us_financial.csv`、`us_reshoring.csv`、`retaliation_index.csv`（Q5）

> **推荐做法**：当官方数据可用时，先放入 `2025/data/external/<name>_official.csv`，再用配套脚本转换至模型所需结构，避免样本数据覆盖真实数据。

## 4. 新增脚本速览

| 脚本 | 位置 | 功能 | 常用命令 |
| --- | --- | --- | --- |
| `worldbank_wits.py` | `2025/src/utils/` | - 子命令 `wb`: 通过 `wbdata` 拉世界银行指标。<br>- 子命令 `wits`: 读取并标准化 WITS 批量下载的 CSV。 | `uv run python 2025/src/utils/worldbank_wits.py wb TM.TAX.MANF.SM.AR.ZS --country CHN --start-year 2015 --end-year 2024 --output ...`<br>`uv run python 2025/src/utils/worldbank_wits.py wits 2025/data/raw/wits_xxx.csv --output 2025/data/external/china_imports_soybeans_official.csv` |
| `preprocess_china_soybeans_official.py` | `2025/scripts/` | 将官方/标准化大豆数据转换为 Q1 模型需要的 `china_imports_soybeans.csv`（含关税列）。 | `uv run python 2025/scripts/preprocess_china_soybeans_official.py --official-file ... --tariff-file ... --output ...` |

## 5. 建议的工作流程（以 Q1 大豆为例）

1. **获取官方数据**：
   - **优先**：修复 UN Comtrade API key 后运行 `data_fetch.py`；
   - **备选**：在 WITS Advanced Query 中导出 2015–2024 年中国 HS1201 进口，并保存 CSV。
2. **标准化**：若来自 WITS，先运行 `worldbank_wits.py wits ...` 提取 `year/partner/value`。
3. **预处理成 Q1 数据**：运行 `preprocess_china_soybeans_official.py`，得到 `china_imports_soybeans.csv`。
4. **模型使用**：`models/q1_soybeans.py` 默认读取 `data/external/china_imports_soybeans.csv`，自动识别这份官方数据并用于分析。
5. **记录与审计**：将最终 CSV 列入本文件、`data_plan_status` 等文档，方便论文撰写与审查。

如需扩展至其他数据源，请在本文件中追加条目，保持“来源—作用—使用方式”的结构。
