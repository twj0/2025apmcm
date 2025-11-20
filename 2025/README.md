# APMCM 2025 Problem C: U.S. Tariff Policy Analysis

## Project Overview

This project analyzes the impact of U.S. reciprocal tariff policies (April 2025) on global trade patterns, U.S. tariff revenue, and the broader economy. It addresses five interconnected questions focusing on:

1. **Q1**: Soybean trade redistribution among U.S., Brazil, and Argentina exporters to China
2. **Q2**: Japanese automobile trade and U.S. auto industry impacts
3. **Q3**: Semiconductor trade, manufacturing, and national security trade-offs
4. **Q4**: Tariff revenue dynamics (Laffer curve analysis)
5. **Q5**: Macroeconomic, financial, and manufacturing reshoring effects

## Project Structure

```
2025/
├── src/
│   ├── utils/
│   │   ├── config.py           # Configuration and paths
│   │   ├── data_loader.py      # Tariff data loader (CU-1)
│   │   └── mapping.py          # HS code mappings (CU-2)
│   ├── models/
│   │   ├── q1_soybeans.py      # Q1 analysis
│   │   ├── q2_autos.py         # Q2 analysis
│   │   ├── q3_semiconductors.py # Q3 analysis
│   │   ├── q4_tariff_revenue.py # Q4 analysis
│   │   └── q5_macro_finance.py  # Q5 analysis
│   └── main.py                 # Main entry point
├── data/
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   ├── external/               # External data sources
│   └── interim/                # Intermediate results
├── results/
│   ├── predictions/            # Model predictions
│   ├── metrics/                # Performance metrics
│   ├── tables/                 # LaTeX tables
│   └── logs/                   # Execution logs
├── figures/                    # Output figures (PDF)
├── problems/                   # Problem statement and data
│   └── Tariff Data/            # U.S. tariff database
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- `uv` package manager (already configured in project root)

### Installation

From the repository root (`SPEC/`):

```bash
# Sync dependencies
uv sync

# Verify installation
uv run python --version
```

### Running the Analysis

**Run all questions:**
```bash
uv run python 2025/src/main.py
```

**Run specific questions:**
```bash
uv run python 2025/src/main.py --questions Q1 Q3
```

**Run individual question modules:**
```bash
uv run python 2025/src/models/q1_soybeans.py
uv run python 2025/src/models/q2_autos.py
# ... etc
```

### Expected Outputs

After running, you will find:

- **Results**: `2025/results/`
  - `q1_elasticities.json`, `q1_scenario_exports.csv`
  - `q2_import_structure.json`, `q2_scenario_industry.csv`
  - `q3_policy_scenarios.csv`, `q3_security_metrics.csv`
  - `q4_revenue_scenarios.csv`, `q4_static_laffer.json`
  - `q5_regressions.json`, `q5_var_results.json`

- **Figures**: `2025/figures/`
  - `q1_shares_before_after.pdf`
  - `q2_import_structure.pdf`, `q2_industry_impact.pdf`
  - `q3_efficiency_security_tradeoff.pdf`
  - `q4_revenue_time_path.pdf`, `q4_laffer_curve.pdf`
  - `q5_time_series_overview.pdf`, `q5_impulse_response.pdf`

## Data Requirements

### Primary Data (Already Provided)

Located in `2025/problems/Tariff Data/`:
- `DataWeb-Query-Import__General_Import_Charges.csv`
- `DataWeb-Query-Export__FAS_Value.csv`
- Annual tariff schedules: `tariff_data_20XX/`

### External Data (Templates Created Automatically)

The code creates template CSV files in `2025/data/external/` for:

**Q1 - Soybeans:**
- `china_imports_soybeans.csv` - Chinese imports from US/Brazil/Argentina

**Q2 - Autos:**
- `us_auto_sales_by_brand.csv` - Sales by Japanese brands
- `us_auto_indicators.csv` - U.S. auto industry metrics

**Q3 - Semiconductors:**
- `us_semiconductor_output.csv` - Domestic chip production
- `us_chip_policies.csv` - CHIPS Act subsidies and export controls

**Q5 - Macro/Finance:**
- `us_macro.csv` - GDP, industrial production, employment
- `us_financial.csv` - Dollar index, yields, equity/crypto prices
- `us_reshoring.csv` - Manufacturing reshoring indicators
- `retaliation_index.csv` - Trade partner retaliation measures

**Note**: These templates contain placeholder data. For accurate results, populate them with real data from sources like:
- Chinese Customs Statistics
- U.S. BEA, BLS, Federal Reserve
- Industry associations (Auto Manufacturers, SIA)

## Methodology

### Q1: Soybean Trade
- **Model**: Panel regression + Armington source-substitution
- **Key equation**: `ln(import_k) = α + β·ln(price_with_tariff) + controls`
- **Scenarios**: Baseline, reciprocal tariff, full retaliation

### Q2: Automobiles
- **Model**: Import structure regression + industry transmission
- **Key equations**:
  - Import: `ln(M_j / M_ROW) = δ + φ·τ_j + controls`
  - Industry: `Y_US = θ + θ1·ImportPenetration + controls`
- **Scenarios**: S0 (no response), S1 (partial relocation), S2 (aggressive localization)

### Q3: Semiconductors
- **Model**: Segment-specific (high/mid/low) trade and output regressions
- **Trade**: `ln(M_s,j) = α + β1·τ + β2·ExportControl + controls`
- **Output**: `ln(Q_US_s) = γ + δ1·Subsidy + δ2·τ + controls`
- **Metrics**: Self-sufficiency, dependence on sensitive suppliers

### Q4: Tariff Revenue
- **Static**: Laffer curve with `ln(R) = α + β1·τ + β2·τ²`
- **Dynamic**: Import response with lags `Δln(M) = φ0 + φ1·Δτ + φ2·Δτ_{-1}`
- **Simulation**: 5-year revenue projection under policy paths

### Q5: Macro/Financial
- **Regression**: `Y = λ0 + λ1·TariffIndex + λ2·RetaliationIndex + controls`
- **VAR/SVAR**: Multi-variable system with impulse responses
- **Reshoring**: Event study comparing pre/post 2025 trends

## Code Quality Standards

Following `.0mathcoder` specifications:

- **Reproducibility**: Random seed set to 42, deterministic operations
- **Documentation**: Comprehensive docstrings with Args/Returns
- **Modularity**: Separate modules for each question, shared utilities
- **Type hints**: Used throughout for clarity
- **Error handling**: Graceful fallbacks and informative logging
- **Output formats**: JSON for metrics, CSV for tables, PDF for figures

## Dependencies

Core packages (from `pyproject.toml`):
- Data: `pandas`, `numpy`
- Statistics: `statsmodels`, `scipy`, `pingouin`
- Time series: `pmdarima`, `prophet`
- Machine learning: `scikit-learn`, `xgboost`, `lightgbm`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Optimization: `cvxpy`, `pulp`

## Contributing

When modifying the code:

1. Follow the coding standards in `.0mathcoder` agent prompt
2. Add docstrings to all functions
3. Log major decisions in `2025/results/logs/`
4. Save intermediate results for debugging
5. Update this README if adding new features

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
- **Solution**: Run `uv sync` from repository root

**Issue**: `FileNotFoundError` for external data
- **Solution**: Check `2025/data/external/` for template files and populate with real data

**Issue**: Empty or minimal results
- **Solution**: Primary tariff data may have formatting issues. Check logs in `2025/results/logs/`

**Issue**: Plotting errors
- **Solution**: Ensure matplotlib backend is configured: `export MPLBACKEND=Agg` (Linux/Mac) or install GUI toolkit

## References

- Problem statement: `2025/problems/2025 APMCM Problem C.md`
- Modeling outline: `spec/2025C_modeling_outline.md`
- Implementation spec: `spec/2025C_impl_spec.md`
- Field definitions: `2025/problems/Tariff Data/td-fields.md`

## Contact

For questions about the code implementation, refer to:
- `.0mathcoder` agent prompt: `.claude/agents/mathcoder.md`
- Collaboration guide: `COLLABORATION_GUIDE.md`
- Project workflow: `WORKFLOW.md`

---

**Last Updated**: 2025-05-20  
**Version**: 1.0  
**Status**: Initial implementation complete, ready for data integration and testing
