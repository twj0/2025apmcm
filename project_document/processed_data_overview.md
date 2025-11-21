# Processed Data Overview (Q1–Q5)

**Purpose**  
This document summarizes the processed datasets under `2025/data/processed/` for Questions Q1–Q5. For each CSV, we briefly describe coverage, units, and key caveats.

---

## Q1 – Soybean Trade (China Imports)

### `q1/q1_0.csv`
- **Content**: Annual soybean import panel from China’s perspective by exporter.  
- **Main variables**: `year`, `exporter`, `import_quantity` (tons), `import_value` (USD), `unit_price` (USD/ton), `tariff_cn_on_exporter` (percent).  
- **Time coverage**: 2015–2024 (continuous, 1 row per exporter-year).  
- **Units**: Values in **USD**, quantities in **metric tons**; tariffs in **percent** (e.g. 30 = 30%).  
- **Usage**: Baseline input for Q1 econometric model and LSTM time series, used to estimate China’s import demand and tariff pass-through.  
- **Notes**: Some values are harmonized from UN Comtrade and official tariff schedules; minor smoothing may be applied to ensure monotonic tariff changes.

### `q1/q1_1.csv`
- **Content**: Monthly soybean import panel (China imports from US, Brazil, Argentina) with seasonality and market share.  
- **Main variables**: `date` (month), `year`, `month`, `exporter`, `import_quantity`, `import_value`, `unit_price`, `tariff_cn_on_exporter`, `tariff_rate`, `market_share`.  
- **Time coverage**: 2015–2024 (monthly).  
- **Units**: Values in **USD**, quantities in **metric tons**; tariffs in **percent**; market share as **fraction of monthly total import value**.  
- **Usage**: Input for Q1 LSTM model (sequence features) and event-study style analysis around tariff shocks.  
- **Notes**: When monthly tariff data is missing, annual `tariff_cn_on_exporter` is merged and forward-filled within each exporter.

---

## Q2 – Japanese Auto Market

### `q2/q2_0_us_auto_sales_by_brand.csv`
- **Content**: US auto sales by brand (including Japanese brands) for Q2 analysis.  
- **Main variables**: `year`, `brand`, `sales_units`, `market_share`, and possibly derived indicators (depending on version).  
- **Time coverage**: Typically 2015–2024; check header for exact range.  
- **Units**: Sales in **units of vehicles**; market share as **fraction or percent** (see column name).  
- **Usage**: Demand-side structure and substitution patterns in Q2 MARL / Transformer models.  
- **Notes**: Data combines GoodCarBadCar style market statistics with simple smoothing to avoid implausible jumps.

### `q2/q2_0_japan_brand_sales.csv`
- **Content**: US sales for Japanese brands only.  
- **Main variables**: `year`, `brand`, `sales_units`, `market_share_japan`.  
- **Time coverage**: Typically 2015–2024.  
- **Units**: Same as `q2_0_us_auto_sales_by_brand.csv`.  
- **Usage**: Focused analysis on Japanese makers’ exposure to US tariff and retaliation scenarios.  
- **Notes**: Market share is within the **US market**, not global.

### `q2/q2_1_industry_indicators.csv`
- **Content**: Sector-level auto industry indicators (production, capacity, macro controls).  
- **Main variables**: `year`, indices for production, capacity utilization, price indices, etc.  
- **Time coverage**: Typically 2015–2024.  
- **Units**: Mostly **indices (2015 or 2017=100)** or **percentages**.  
- **Usage**: Controls in Q2 econometric models and state variables in MARL environment.  
- **Notes**: Some series are interpolated/normalized from FRED and industry reports.

---

## Q3 – US Semiconductor Supply Chain

### `q3/q3_0_us_semiconductor_output.csv`
- **Content**: Annual US semiconductor industry indicators with CHIPS Act and export controls reflected.  
- **Main variables**:  
  - `year` (2015–2024)  
  - `us_chip_output_index` (2015=100)  
  - `us_chip_output_billions` (USD billions)  
  - `global_chip_output_billions` (USD billions)  
  - `us_global_share_pct` (US share of global production, %)  
  - `global_chip_demand_index` (2015=100)  
  - `china_import_dependence_pct` (US dependence on China for inputs, %)  
  - `r_and_d_intensity_pct` (R&D / revenue, %)  
  - `capex_billions` (capex, USD billions)  
  - `policy_support_index` (0–10)  
  - `export_control_index` (0–10)  
  - `supply_chain_risk_index` (0–100).  
- **Units**: As indicated above; monetary variables are in **USD billions**.  
- **Usage**: Node/feature time series for Q3 GNN model and policy counterfactuals.  
- **Notes**: Values are calibrated to match public SIA/WSTS/USITC trends, not exact official numbers.

### `q3/q3_1_chip_policies.csv`
- **Content**: Annual policy intensity indicators for CHIPS Act, export controls, and reshoring incentives.  
- **Main variables**: `year`, `chips_subsidy_index` (0–10), `export_control_index` (0–10), `reshoring_incentive_index` (0–10), `rd_investment_billions` (USD billions), `policy_uncertainty_index` (0–100).  
- **Time coverage**: 2015–2024.  
- **Usage**: Exogenous policy inputs in Q3 models; helps capture structural breaks in supply chain risk.  
- **Notes**: Indices are synthetic but aligned with real-world policy chronology (2018+ export controls, 2022 CHIPS jump).

### `q3/q3_2_supply_chain_segments.csv`
- **Content**: Segment-level (high/mid/low) US semiconductor production and trade flows.  
- **Main variables**: `year`, `segment` (`high`, `mid`, `low`), `us_production_billions`, `import_value_billions`, `export_value_billions`, `self_sufficiency_ratio`, `china_share_pct`.  
- **Time coverage**: 2015–2024, 3 segments per year.  
- **Units**: Monetary values in **USD billions**; `self_sufficiency_ratio` is dimensionless; `china_share_pct` in percent.  
- **Usage**: Graph node/edge attributes for Q3 GNN, plus scenario analysis on segment-specific reshoring.  
- **Notes**: `self_sufficiency_ratio > 1` indicates net exporter status for that segment.

---

## Q4 – Tariff Revenue & Laffer Curve

### `q4/q4_0_tariff_revenue_panel.csv`
- **Content**: Annual US tariff revenue and import panel, including China split.  
- **Main variables**:  
  - `year`  
  - `total_imports_usd` (USD, goods imports)  
  - `total_tariff_revenue_usd` (USD, total customs duties on goods)  
  - `effective_tariff_rate` (percent, revenue/imports*100)  
  - `weighted_avg_tariff` (percent, trade-weighted average)  
  - `china_imports_usd` (USD, goods imports from China)  
  - `china_tariff_revenue_usd` (USD, duties collected on China-origin goods)  
  - `china_effective_tariff_rate` (percent).  
- **Time coverage**: 2015–2024.  
- **Units**: All monetary values are **USD**; rates in **percent**.  
- **Usage**: Core input for Q4 static/dynamic Laffer models, dynamic import response estimation, and baseline revenue calibration.  
- **Notes**: Magnitudes are aligned with USITC/Treasury orders of magnitude; effective rates show clear rise during 2018–2020 trade war.

### `q4/q4_1_tariff_scenarios.csv`
- **Content**: Forward-looking tariff scenarios (baseline, reciprocal, escalation) for 2025–2030.  
- **Main variables**: `year`, `scenario`, `avg_tariff_rate`, `china_tariff_rate`, `row_tariff_rate`, `import_elasticity`, `expected_revenue_billions`.  
- **Time coverage**: 2025–2030.  
- **Units**: Rates in **percent**; `expected_revenue_billions` in **USD billions**.  
- **Usage**: Scenario input for Q4 second-term revenue simulations and DRL policy evaluation.  
- **Notes**: Elasticities and revenue paths are stylized but internally consistent; econometric model will recompute revenue paths given these tariff paths.

---

## Q5 – Macro, Financial, and Reshoring

### `q5/q5_0_macro_indicators.csv`
- **Content**: Core US macro indicators for 2015–2024.  
- **Main variables**: `year`, `real_gdp_growth`, `industrial_production_index`, `unemployment_rate`, `cpi_index`, `core_inflation_rate`, `federal_funds_rate`, `real_investment_growth`, etc.  
- **Time coverage**: 2015–2024.  
- **Units**: Growth rates in **percent**; indices normalized (e.g. 2017=100); rates in **percent**.  
- **Usage**: Left-hand variables in VAR/SVAR and regression models in Q5.  
- **Notes**: Data is partially approximated from FRED; signs and magnitudes follow known patterns (e.g. 2020 GDP shock).

### `q5/q5_1_financial_indicators.csv`
- **Content**: Financial market indicators.  
- **Main variables**: `year`, `treasury_10y_yield`, `sp500_index`, `dollar_index`, `vix_index`, `corporate_spread_bps`, `equity_risk_premium`.  
- **Time coverage**: 2015–2024 (plus possible 2025 row).  
- **Units**: Yields and premiums in **percent**; spreads in **basis points**; indices as level indices.  
- **Usage**: Financial channels in Q5 VAR and ML models; stress-test scenarios.  
- **Notes**: Some series are smoothed to reduce noise; used mainly for relative, not point, predictions.

### `q5/q5_2_reshoring_indicators.csv`
- **Content**: Manufacturing reshoring and capacity metrics.  
- **Main variables**: `year`, `manufacturing_va_share`, `manufacturing_employment_share`, `reshoring_announcements`, `reshoring_jobs_created`, `fdi_manufacturing_billions`, `capacity_utilization`.  
- **Time coverage**: 2015–2024 (plus possible 2025 row).  
- **Units**: Shares in **percent**; counts in **number of events/jobs**; FDI in **USD billions**.  
- **Usage**: Reshoring outcome variables for event study and ML prediction.  
- **Notes**: Where official counts are missing, orders of magnitude follow Reshoring Initiative style reports.

### `q5/q5_3_retaliation_index.csv`
- **Content**: Trade partner retaliation intensity.  
- **Main variables**: `year`, `retaliation_index`, `china_retaliation_index`, `eu_retaliation_index`, `wto_disputes_filed`, `retaliatory_tariff_coverage_billions`, `retaliation_products_count`.  
- **Time coverage**: 2015–2024 (plus possible 2025 row).  
- **Units**: Indices are **0–100** scale; coverage in **USD billions**; counts as integers.  
- **Usage**: Control for foreign responses in Q5 regression and VAR models.  
- **Notes**: Indices are composite, built to increase during trade war and peak when retaliation is strongest.

### `q5/q5_4_integrated_panel.csv`
- **Content**: Integrated macro–tariff–financial–reshoring panel for Q5 modeling.  
- **Main variables**: `year`, `tariff_index`, `gdp_growth`, `unemployment_rate`, `industrial_production`, `treasury_10y_yield`, `sp500_index`, `manufacturing_va_share`, `reshoring_jobs`, `retaliation_index`.  
- **Time coverage**: 2015–2024.  
- **Units**: Inherited from component files (tariff indices in percent, macro/financial as specified above).  
- **Usage**: Direct input to Q5 VAR, VAR-LSTM, and Transformer models.  
- **Notes**: Remaining missing values (mostly early-year tariff indices) are now filled via calibrated Q5 tariff indices.

### `q5/q5_tariff_indices_calibrated.csv`
- **Content**: Tariff indices aligned with Q4 effective tariff rates.  
- **Main variables**: `year`, `tariff_index` (mapped to `tariff_index_total` in code), `china_tariff_index`, `row_tariff_index`, `china_tariff_coverage_pct`.  
- **Time coverage**: 2015–2024.  
- **Units**: Indices in **percent**; coverage in **percent of US imports from China under additional tariffs**.  
- **Usage**: Preferred tariff index input in Q5 model (ensures consistency with Q4 revenue panel).  
- **Notes**: `tariff_index` approximately equals Q4 `effective_tariff_rate`; China index is higher post-2018.

### `q5/q5_tariff_indices_policy.csv`
- **Content**: Policy-strength version of tariff indices, emphasizing relative intensity rather than strict equality to revenue-based effective rates.  
- **Main variables**: Same as calibrated version.  
- **Time coverage**: 2015–2024.  
- **Usage**: Alternative input for sensitivity analysis, where larger differences between `tariff_index` and `china_tariff_index` emphasize policy shocks.  
- **Notes**: Use with caution if you require strict consistency with Q4 revenue data; default pipeline uses the calibrated version.

---

## General Notes

- **Directory**: All processed files live under `2025/data/processed/`.  
- **Encoding**: UTF-8 CSV with comma separator and header row.  
- **Missing Data**: Typically below 5% per column; imputation strategies include forward-fill (for policies) and interpolation (for continuous indices).  
- **Realism vs. Exactness**: Most series are calibrated to match public trends and orders of magnitude but are not literal official downloads; this is acceptable for modeling and competition purposes but should be described clearly in the paper.
