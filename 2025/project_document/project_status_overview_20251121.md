# 2025 APMCM Problem C 项目状态综合分析报告（聚焦 Q5）

## 1. 项目概述摘要

本项目围绕 2025 APMCM C 题，已按 Q1–Q5 构建了完整的 Python 建模与分析框架，涵盖大豆贸易、汽车、日本 FDI、半导体安全、关税收入以及宏观与金融影响及制造业回流。整体代码结构与 `spec/2025C_impl_spec.md`、`spec/2025C_modeling_outline.md` 基本一致，可通过 `uv run python 2025/src/main.py` 一键运行全流程。
在 Q5 方向，基于 FRED 官方宏观和金融时间序列以及手工构建的报复指数和回流指标，已经实现了“回归 + VAR + 事件研究”的宏观-金融-制造业综合分析框架，能够显式区分美国互惠关税与贸易伙伴报复行为的冲击效应。

## 2. 数据清单和状态（CSV）

> 注：下表仅列出与 Q5 高度相关、且在当前代码中已经使用或明确规划使用的核心 CSV；更完整的数据质量审计见 `spec/csv_data_integrity_assessment202511201651.md` 和 `project_document/data_quality_report_20251120.md`。

| 文件路径 | 主要变量 | 时间范围 | 官方/样本 | 缺失与质量 | 用途/关联问题 |
|---------|----------|----------|-----------|------------|----------------|
| `2025/data/external/us_macro.csv` | `year`, `gdp_growth`, `industrial_production`, `unemployment_rate`, `cpi` | 2015–2025（含 2020 疫情、2025 政策年份） | 样本结构，内容已按 FRED 口径填充 | 无缺失（当前版本）；列结构与 `MacroFinanceModel.load_q5_data` 完全一致 | Q5：回归与 VAR 中的宏观变量输入；也可作为论文图表数据源 |
| `2025/data/external/us_macro_consolidated.csv` | `year`, `gdp_real`, `cpi`, `unemployment`, `industrial_production`, `fed_funds_rate`, `treasury_10y`, `sp500` | 2015–2024 | FRED 官方序列整合，质量评估为 EXCELLENT | 无明显缺失；2025 年度可通过追加 FRED 最新数据扩展 | Q5：作为宏观数据黄金标准，可与 `us_macro.csv` 做一致性校验或替换；Q4：宏观背景描述 |
| `2025/data/external/us_financial.csv` | `year`, `dollar_index`, `treasury_yield_10y`, `sp500_index`, `crypto_index` | 2015–2025 | 样本结构 + 参考官方水平手工填充 | 无缺失；波动模式合理但需在论文中说明为构造数据 | Q5：金融市场响应（回归、VAR）输入 |
| `2025/data/external/us_reshoring.csv` | `year`, `manufacturing_va_share`, `manufacturing_employment_share`, `reshoring_fdi_billions` | 2015–2025 | 构造指标（基于回流趋势假设） | 无缺失；水平与趋势需在 `data_requirements_status` 中做额外解释 | Q5：制造业回流指标（回归 & 事件研究） |
| `2025/data/external/retaliation_index.csv` | `year`, `retaliation_index` | 2015–2025 | 构造指数，基于贸易伙伴报复与出口管制事件的等级编码 | 无缺失；指数刻度与含义需在方法章节中详细说明 | Q5：报复情景的时间序列度量，直接进入回归与 VAR；Q3：可作为政策变量扩展 |
| `2025/data/processed/tariff_indices.parquet` | `year`, `tariff_index_total`, … | 2015–2025（视 TariffDataLoader 处理结果而定） | 来源于 Tariff Data 官方 HS 级数据的加权计算 | 若文件存在，则列完整；缺失将被 `q5_macro_finance.py` 用随机占位补齐 | Q5：美国互惠关税强度的时间序列；Q2–Q4：可共享使用 |
| `2025/data/external/q5_us_macro_from_grok4_candidate.csv` 等候选文件 | 各类宏观、金融或回流候选指标 | 多为 2015–2024 或 2015–2025 | LLM 生成候选数据（Grok/Gemini） | 仅供结构参考或敏感性分析；不建议直接作为主结果数据源 | Q5：补充或压力测试用数据 |

整体上，**Q5 主体分析所需的宏观与金融时间序列已经齐备，且覆盖 2015–2024（部分扩展至 2025）**。关键不足在于：
- 制造业回流与反制指数目前仍为主观构造型数据，虽在结构和时间覆盖上完整，但需要在论文和说明文档中强调这一点；
- Q5 仍依赖 `tariff_indices.parquet` 是否成功由 `TariffDataLoader` 及其上游脚本生成，若缺失会退化为随机占位，对结论有潜在影响。

## 3. 建模方法摘要

### 3.1 整体题目（Q1–Q5）

- **Q1 大豆贸易**：基于 HS 1201 贸易数据与中国外贸统计，构建来源替代模型与价格/份额弹性估计，并在此基础上模拟“互惠关税”“完全报复”等情景下中、美、巴西、阿根廷之间的大豆贸易重分配。
- **Q2 汽车与 FDI**：利用 HS 8703 系列进口数据与美国汽车产业指标，构造进口结构模型和产业传导模型，并设置“无应对/部分产能转移/激进本地化”三类情景，评估日系车企对关税的应对与美国本土生产和就业的变化。
- **Q3 半导体安全**：以 HS 8541/8542 及政策变量为基础，对高/中/低三段半导体进行贸易与产出回归，计算自给率、对中依赖度与综合供应安全指数，并比较“补贴、关税、综合政策”三种方案下的效率-安全权衡。
- **Q4 关税收入**：在 Tariff Data 和外部情景设定基础上，构建静态 Laffer 曲线模型与动态进口响应框架，模拟第二任期（2025–2029）不同关税路径下的累计关税收入（基线 vs 互惠关税方案）。
- **Q5 宏观+金融+回流**：以贸易加权关税指数、反制指数以及宏观和金融变量为核心，采用线性回归、VAR 与事件研究方法来刻画关税政策对宏观、金融市场以及制造业回流的多维影响。

### 3.2 Q5 具体方法

参照 `spec/2025C_modeling_outline.md` 和 `spec/q5_macro_finance_modeling_method.md`，当前实现采用了**中尺度可执行框架**，而非完整 DSGE：

1. **回归模型**（`MacroFinanceModel.estimate_regression_effects`）
   - 形式：`Y_t = λ0 + λ1·TariffIndex_t + λ2·RetaliationIndex_t + ε_t`
   - 被解释变量：`gdp_growth`、`industrial_production`、`manufacturing_va_share`、`manufacturing_employment_share` 等。
   - 解释变量：`tariff_index_total`（互惠关税强度）、`retaliation_index`（贸易伙伴报复程度）。
   - 目标：粗略识别关税与报复对宏观和制造业回流指标的方向和大致弹性。

2. **VAR 模型**（`MacroFinanceModel.estimate_var_model`）
   - 变量集合（可根据可用性裁剪）：`[tariff_index_total, retaliation_index, gdp_growth, industrial_production, ...]`。
   - 使用 `statsmodels.tsa.api.VAR` 估计多元时间序列模型，提取以关税指数为冲击源的冲击响应函数（IRF）。
   - 目标：刻画关税冲击在宏观与实体变量间的动态传导，尤其是与贸易伙伴报复行为的联动效应。

3. **事件研究 / DID**（`MacroFinanceModel.evaluate_reshoring`）
   - 将 2025 年视为互惠关税及相关政策的“处理时点”，构造 `post_treatment = 1{year ≥ 2025}`。
   - 比较处理前后制造业增加值占比和制造业就业占比的均值与回归系数，评估回流强度是否显著上升。
   - 当前实现为单国、单时间轴的简化事件研究，未来可扩展为多国对照或分行业 DID。

在此基础上，文档中还设计了完整的 DSGE 蓝图（家庭—企业—政府—国际收支多个方程），但**尚未在代码中落地**，当前实现更接近于“实证 VAR + 经验回归 + 事件研究”的可执行版本。

## 4. 代码实现状态

根据 `2025/CODE_SUMMARY.md` 与 `2025/src` 目录：

- **整体结构**：
  - `src/main.py` 提供统一入口，可按 `--questions Q1 Q5` 选择性运行各模块；
  - `src/models/q1_soybeans.py` ~ `q5_macro_finance.py` 分别实现五个子问题的分析流程；
  - `src/utils/config.py`、`src/utils/data_loader.py`、`src/utils/mapping.py` 等为共享工具模块。

- **Q5 模块实现情况（`src/models/q5_macro_finance.py`）**：
  - `MacroFinanceModel.load_q5_data()`：
    - 读取 `tariff_indices.parquet`、`us_macro.csv`、`us_financial.csv`、`us_reshoring.csv`、`retaliation_index.csv`；
    - 使用 `year` 进行外连接合并，得到统一时间序列 DataFrame，覆盖至少 2015–2024；
    - 若关税指数文件缺失，则用随机数生成占位数据（便于调试，但需在正式分析前替换为真实计算结果）。
  - `estimate_regression_effects()`：
    - 对多组因变量（GDP 增长、工业生产、制造业 VA 和就业占比）分别拟合 OLS；
    - 结果写入 `2025/results/q5_regressions.json`。
  - `estimate_var_model()`：
    - 基于所选变量拟合 VAR，计算 10 期冲击响应；
    - 将关键指标（滞后阶数、AIC/BIC、IRF 序列）写入 `2025/results/q5_var_results.json`。
  - `evaluate_reshoring()`：
    - 构造前后样本，估计 `post_treatment` 系数及其显著性，结果写入 `2025/results/q5_reshoring_effects.json`。
  - `plot_q5_results()`：
    - 输出时间序列总览图 `q5_time_series_overview.pdf` 和 IRF 图 `q5_impulse_response.pdf`。

从实现程度看：**Q5 已经具备完整的“数据加载 → 回归/VAR → 事件研究 → 图表输出”的闭环流程**，与实现规格文档高度一致。

## 5. 一致性评估

- **数据 vs 代码**：
  - Q5 使用的核心 CSV（`us_macro.csv`, `us_financial.csv`, `us_reshoring.csv`, `retaliation_index.csv`）均已存在，列名与模型代码严格匹配；
  - `tariff_indices.parquet` 的生成依赖 TariffDataLoader 上游流程，目前在仓库中存在 Tariff Data 原始数据和 `compute_tariff_indices` 能力，但尚未看到自动批处理脚本；若未提前运行，将触发 Q5 中的随机占位逻辑，影响结果的可解释性。

- **代码 vs 规格文档（`spec/2025C_impl_spec.md`）**：
  - Q1–Q5 各模块的函数接口和输出文件名与实现规格文档保持高度一致；
  - Q5 部分特别要求的输出：`q5_regressions.json`, `q5_var_results.json`, `q5_reshoring_effects.json` 和对应图表均已在代码中实现。

- **代码 vs 方法文档（尤其 Q5）**：
  - `spec/q5_macro_finance_modeling_method.md` 中描述的 DSGE 框架目前仅停留在理论层面；
  - `spec/2025C_modeling_outline.md` 中给出的“VAR + 回归 + 事件研究”的可执行方案，已被当前 `MacroFinanceModel` 基本实现；
  - 互惠关税和贸易伙伴报复：在 Q5 中通过 `tariff_index_total` 与 `retaliation_index` 明确分别建模，且同时进入回归与 VAR，因此**“互惠关税情景”和“贸易伙伴报复行为”在模型中是有区分且被显式考虑的**。

- **代码 vs 论文草稿（`2025/paper/APMCM_2025C_paper_cn.md`）**：
  - 论文草稿中 Q5 章节对宏观、金融和制造业回流的描述与实现基本对应，个别地方为便于叙述对模型稍作概括，但未与代码产生明显冲突；
  - DSGE 部分在论文中可以作为“进一步研究方向”，而不是当前数值结果的来源。

总体结论：**数据、代码和文档在路径、变量和分析流程上的一致性较高**，主要不一致集中在“理论规划（DSGE） vs 实际实现（VAR/回归）”的层面，需要在撰写正式论文和答辩材料时明确区分。

## 6. 差距分析与后续建议

### 6.1 主要缺失组件与潜在风险

1. **Q5 中关税指数数据链尚未完全闭合**
   - 风险点：`tariff_indices.parquet` 若未由上游脚本正确生成，当前 Q5 会使用随机占位数据，仅能用于流程调试，不能用于严肃结论。
   - 影响：关税指数作为核心解释变量，随机占位将削弱回归与 VAR 的经济含义，使得“互惠关税情景”的量化结果失真。

2. **制造业回流与报复指数的数据仍为构造型**
   - 风险点：`us_reshoring.csv` 与 `retaliation_index.csv` 目前基于假设或情景编码构建，并非完全来自官方统计或可复现的事件数据库。
   - 影响：定量结果更适合作为“情景分析”而非“经验事实”，需要在文中强化这一定位，并在附录中说明构造方法和数据来源假设。

3. **Q1/Q2/Q4 的部分外部数据仍为模板或部分填充**
   - 虽然本报告聚焦 Q5，但整体论文中不同问题之间存在逻辑联动（例如 Q1 大豆和 Q3 半导体的经验结论会为 Q5 提供微观支撑），因此这些数据缺口会降低整体叙事的一致性。

4. **深度方法（DSGE、机器学习/强化学习）尚未落地**
   - Q5 方法文档中提出了完整的 DSGE 模型以及可能的深度强化学习框架（比如在多期政策路径选择问题上使用 DRL），但目前代码层面仍停留在 VAR + 回归 + 简化事件研究上。

### 6.2 建议的后续步骤（按优先级）

1. **优先级 A：打通关税指数真实计算链（影响 Q5 结论可靠性）**
   - 基于 `TariffDataLoader.create_imports_panel()` 和 `compute_effective_tariff()`，编写或完善一个批处理脚本：
     - 从 Tariff Data 原始 CSV（2015–2024）生成含 `effective_tariff` 的 imports panel；
     - 按年（以及可选的行业/伙伴维度）计算 trade-weighted 关税指数，将结果写入 `tariff_indices.parquet`；
     - 在 `project_document` 中补充一份“关税指数构造说明”文档，列出权重、截尾规则及与 World Bank/WITS 指标的对比。

2. **优先级 A：严格文档化并尽可能校准回流与报复指数**
   - 为 `us_reshoring.csv` 和 `retaliation_index.csv` 分别撰写数据说明：
     - 指明变量定义、编码规则、数据来源（新闻事件、政策公告、第三方数据库等）；
     - 在可行范围内，引入部分官方或半官方数据校准（如 Reshoring Initiative 报告、OECD FDI 数据、WTO 报复措施清单等）。
   - 在论文与报告中，将 Q5 回流和报复相关结论标记为“情景/机制演示”，避免被误读为精确预测。

3. **优先级 B：对 Q5 VAR 与回归结果做系统性稳健性检验**
   - 尝试：
     - 使用 `us_macro_consolidated.csv` 替换或联合 `us_macro.csv`，检查结果对宏观变量口径的敏感性；
     - 改变 VAR 滞后阶数、变量组合（加入通胀、利率、汇率等），测试冲击响应的稳健性；
     - 使用候选数据（`q5_us_macro_from_grok4_candidate.csv` 等）做压力测试，验证模型形式的鲁棒性。

4. **优先级 B：完善 Q1/Q2/Q4 的数据与模型，使整体故事闭环**
   - 按 `project_document/data_requirements_status_20251120_2014.md` 的建议，尽量补齐 Q1 中国大豆进口官方数据、Q2 品牌级汽车销售完整面板和 Q4 关税收入时间序列；
   - 更新相应的结果文件与图表，使 Q1–Q4 的结论可直接为 Q5 提供定性支撑和参数参考。

5. **优先级 C：规划深度模型与机器学习扩展（中长期任务）**
   - 在现有 VAR/回归框架基础上，进一步探讨：
     - 使用非线性时间序列模型（如随机森林、梯度提升、RNN）刻画关税与宏观变量间的非线性与阈值效应；
     - 将“政策路径选择”抽象为多期决策问题，引入强化学习（如 DQN、PPO）在模拟环境中优化关税与产业政策组合；
     - 基于当前 `MacroFinanceModel` 的结果，为 DSGE 校准提供先验区间和检验目标，但将 DSGE/DRL 的实现放在后续迭代。

---

**总结**：目前 2025C 项目在代码结构和数据准备上已经具备较高完整度，特别是 Q5 已实现兼顾互惠关税与贸易伙伴报复的宏观-金融-制造业回流分析框架。主要差距集中在部分关键指标仍为构造数据、关税指数计算链尚未完全自动化，以及深度理论模型尚处于规划阶段。建议先完成关税指数与回流/报复数据的实证化与文档化，再在现有 VAR 和事件研究的基础上逐步引入更复杂的 DSGE 与机器学习方法。
