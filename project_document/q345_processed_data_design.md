# Q3/Q4/Q5 Processed Data Structure Design

**Version**: 1.0  
**Date**: 2024-11-21  
**Purpose**: Define standardized processed data structures for Q3 (Semiconductors), Q4 (Tariff Revenue), Q5 (Macro-Financial)

---

## Q3: Semiconductor Supply Chain Data

### File 1: `q3_0_us_semiconductor_output.csv`
**Description**: U.S. semiconductor production output and industry indicators  
**Time Range**: 2015-2024 (annual)  
**Source**: FRED (NAICS 3344), SIA, USITC

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2015-2024) | - |
| us_chip_output_index | float | US semiconductor output index (2015=100) | FRED IPB53122S |
| us_chip_output_billions | float | US chip production value (USD billions) | SIA/USITC |
| global_chip_demand_index | float | Global semiconductor demand index (2015=100) | WSTS/SIA |
| us_global_share_pct | float | US share of global production (%) | Calculated |
| china_import_dependence_pct | float | US dependence on China for inputs (%) | USITC |

### File 2: `q3_1_chip_policies.csv`
**Description**: CHIPS Act and semiconductor policy indicators  
**Time Range**: 2015-2024 (annual)  
**Source**: Policy documents, government reports

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2015-2024) | - |
| chips_subsidy_index | float | CHIPS Act subsidy intensity (0-10 scale) | Policy analysis |
| export_control_index | float | Export control stringency (0-10 scale) | BIS/Commerce |
| reshoring_incentive_index | float | Reshoring incentive strength (0-10 scale) | Policy analysis |
| rd_investment_billions | float | Federal R&D investment (USD billions) | NSF/NIST |
| policy_uncertainty_index | float | Policy uncertainty (0-100 scale) | Baker et al. |

### File 3: `q3_2_supply_chain_segments.csv`
**Description**: Semiconductor supply chain by segment  
**Time Range**: 2015-2024 (annual)  
**Source**: USITC, industry reports

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2015-2024) | - |
| segment | str | Segment (high/mid/low) | Industry classification |
| us_production_billions | float | US production by segment (USD billions) | USITC |
| import_value_billions | float | Import value by segment (USD billions) | USITC |
| export_value_billions | float | Export value by segment (USD billions) | USITC |
| self_sufficiency_ratio | float | Production/(Production+Imports-Exports) | Calculated |
| china_share_pct | float | China's share in segment imports (%) | USITC |

---

## Q4: Tariff Revenue and Laffer Curve Data

### File 1: `q4_0_tariff_revenue_panel.csv`
**Description**: Annual tariff revenue and import values  
**Time Range**: 2015-2024 (annual)  
**Source**: USITC DataWeb, US Treasury

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2015-2024) | - |
| total_imports_usd | float | Total import value (USD) | USITC |
| total_tariff_revenue_usd | float | Total tariff revenue collected (USD) | USITC |
| effective_tariff_rate | float | Revenue/Imports (%) | Calculated |
| weighted_avg_tariff | float | Trade-weighted average tariff (%) | USITC |
| tariff_lines_count | int | Number of active tariff lines | USITC |
| china_imports_usd | float | Imports from China (USD) | USITC |
| china_tariff_revenue_usd | float | Tariff revenue from China (USD) | USITC |

### File 2: `q4_1_tariff_scenarios.csv`
**Description**: Tariff scenarios for simulation  
**Time Range**: 2024-2030 (projection)  
**Source**: Policy scenarios, model assumptions

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2024-2030) | - |
| scenario | str | Scenario name (baseline/reciprocal/escalation) | Model design |
| avg_tariff_rate | float | Average tariff rate (%) | Scenario assumption |
| china_tariff_rate | float | Tariff on China (%) | Scenario assumption |
| row_tariff_rate | float | Tariff on rest of world (%) | Scenario assumption |
| expected_revenue_billions | float | Expected revenue (USD billions) | Model projection |
| import_elasticity | float | Import demand elasticity | Literature/estimation |

---

## Q5: Macroeconomic, Financial, and Reshoring Data

### File 1: `q5_0_macro_indicators.csv`
**Description**: Core macroeconomic indicators  
**Time Range**: 2015-2024 (annual)  
**Source**: FRED, BEA, BLS

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2015-2024) | - |
| real_gdp_growth | float | Real GDP growth rate (%) | FRED A191RL1Q225SBEA |
| industrial_production_index | float | Industrial production (2017=100) | FRED INDPRO |
| unemployment_rate | float | Unemployment rate (%) | FRED UNRATE |
| cpi_index | float | CPI all items (1982-84=100) | FRED CPIAUCSL |
| core_inflation_rate | float | Core CPI inflation (%) | FRED CPILFESL |
| federal_funds_rate | float | Federal funds rate (%) | FRED DFF |
| real_investment_growth | float | Real investment growth (%) | FRED GPDIC1 |

### File 2: `q5_1_financial_indicators.csv`
**Description**: Financial market indicators  
**Time Range**: 2015-2024 (annual)  
**Source**: FRED, market data

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2015-2024) | - |
| treasury_10y_yield | float | 10-year Treasury yield (%) | FRED DGS10 |
| sp500_index | float | S&P 500 index (annual average) | FRED SP500 |
| dollar_index | float | Trade-weighted USD index | FRED DTWEXBGS |
| vix_index | float | VIX volatility index | FRED VIXCLS |
| corporate_spread_bps | float | Corporate bond spread (bps) | FRED BAMLC0A0CM |
| equity_risk_premium | float | Equity risk premium (%) | Calculated |

### File 3: `q5_2_reshoring_indicators.csv`
**Description**: Manufacturing reshoring metrics  
**Time Range**: 2015-2024 (annual)  
**Source**: Reshoring Initiative, Census, BEA

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2015-2024) | - |
| manufacturing_va_share | float | Manufacturing value-added/GDP (%) | BEA |
| manufacturing_employment_share | float | Manufacturing employment share (%) | BLS |
| reshoring_announcements | int | Number of reshoring announcements | Reshoring Initiative |
| reshoring_jobs_created | int | Jobs from reshoring | Reshoring Initiative |
| fdi_manufacturing_billions | float | Manufacturing FDI (USD billions) | BEA |
| capacity_utilization | float | Manufacturing capacity utilization (%) | FRED TCU |

### File 4: `q5_3_retaliation_index.csv`
**Description**: Trade partner retaliation metrics  
**Time Range**: 2015-2024 (annual)  
**Source**: WTO, trade policy databases

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2015-2024) | - |
| retaliation_index | float | Composite retaliation index (0-100) | Calculated |
| china_retaliation_index | float | China retaliation intensity (0-100) | Policy tracking |
| eu_retaliation_index | float | EU retaliation intensity (0-100) | Policy tracking |
| wto_disputes_filed | int | WTO disputes against US | WTO |
| retaliatory_tariff_coverage_billions | float | Value of US exports under retaliation | USITC |
| retaliation_products_count | int | Number of products under retaliation | USITC |

### File 5: `q5_4_integrated_panel.csv`
**Description**: Integrated macro-financial-tariff panel for modeling  
**Time Range**: 2015-2024 (annual)  
**Source**: Merged from above sources

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| year | int | Year (2015-2024) | - |
| tariff_index | float | Overall tariff level index | From Q4 |
| gdp_growth | float | Real GDP growth (%) | q5_0 |
| unemployment_rate | float | Unemployment rate (%) | q5_0 |
| industrial_production | float | Industrial production index | q5_0 |
| treasury_10y_yield | float | 10-year yield (%) | q5_1 |
| sp500_index | float | S&P 500 index | q5_1 |
| manufacturing_va_share | float | Manufacturing share (%) | q5_2 |
| reshoring_jobs | int | Reshoring jobs created | q5_2 |
| retaliation_index | float | Retaliation intensity | q5_3 |

---

## Data Processing Guidelines

### 1. Time Range Standardization
- Primary range: 2015-2024 (10 years of actual data)
- Projection range: 2025-2030 (for scenario analysis)
- All files must have complete year coverage without gaps

### 2. Missing Value Handling
- Use forward-fill for policy variables that persist
- Use interpolation for continuous economic variables
- Document any imputation in metadata
- Target <5% missing values per column

### 3. Data Type Consistency
- Year: always integer
- Rates/percentages: float with consistent decimal places
- Indices: float, normalized to base year
- Counts: integer
- Currency values: float in specified units (USD)

### 4. Source Priority
1. Official government statistics (FRED, BEA, BLS, USITC)
2. International organizations (WTO, IMF, World Bank)
3. Industry associations (SIA, Reshoring Initiative)
4. Academic/research databases
5. Model estimates/imputations (clearly marked)

### 5. Quality Checks
- Verify year continuity
- Check for outliers (>3 std dev)
- Validate against known benchmarks
- Cross-check related variables for consistency
- Document any data transformations

---

## Implementation Notes

### File Naming Convention
```
q[3-5]_[0-9]_[description].csv
```
- Question number (3, 4, or 5)
- Sequence number (0 = primary, 1+ = supporting)
- Descriptive name (lowercase, underscores)

### Column Naming Convention
- Lowercase with underscores
- Include units in name where relevant (_pct, _billions, _index)
- Be specific about geography (us_, china_, global_)
- Use consistent abbreviations

### Metadata Requirements
Each CSV should have accompanying metadata:
- Source URLs/citations
- Extraction date
- Processing steps
- Known limitations
- Update frequency

---

## Next Steps

1. Implement `generate_processed_q345.py` script to:
   - Read from external data sources
   - Apply transformations per this design
   - Validate data quality
   - Save to processed directories

2. Run completeness audit on generated files

3. Update model code to use processed data

4. Document actual vs. design discrepancies
