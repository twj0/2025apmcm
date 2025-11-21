#!/usr/bin/env python
"""
Build Q5 tariff indices in two versions:
- q5_tariff_indices_calibrated.csv (aligned to Q4 effective tariff rates and China rates)
- q5_tariff_indices_policy.csv (policy strength index from provided series)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parents[1]
Q4_PANEL = ROOT / 'data' / 'processed' / 'q4' / 'q4_0_tariff_revenue_panel.csv'
Q5_DIR = ROOT / 'data' / 'processed' / 'q5'
Q5_DIR.mkdir(parents=True, exist_ok=True)

# 1) Calibrated version using Q4 panel
q4 = pd.read_csv(Q4_PANEL)
q4 = q4[(q4['year'] >= 2015) & (q4['year'] <= 2024)].copy()

# Weighted average decomposition: t = c_t*c_s + r_t*(1-c_s)  => r_t = (t - c_t*c_s) / (1-c_s)
c_s = q4['china_imports_usd'] / q4['total_imports_usd']
den = (1 - c_s).replace(0, 1e-12)
row_idx = (q4['effective_tariff_rate'] - q4['china_effective_tariff_rate'] * c_s) / den
row_idx = row_idx.clip(lower=0)

coverage_map = {
    2015: 0.0,
    2016: 0.0,
    2017: 0.0,
    2018: 40.0,
    2019: 65.0,
    2020: 65.0,
    2021: 62.0,
    2022: 60.0,
    2023: 58.0,
    2024: 55.0,
}

calibrated = pd.DataFrame({
    'year': q4['year'].astype(int),
    'tariff_index': q4['effective_tariff_rate'].astype(float),
    'china_tariff_index': q4['china_effective_tariff_rate'].astype(float),
    'row_tariff_index': row_idx.astype(float),
    'china_tariff_coverage_pct': q4['year'].map(coverage_map).astype(float),
})

calibrated.to_csv(Q5_DIR / 'q5_tariff_indices_calibrated.csv', index=False)

# 2) Policy strength version from provided series
policy = pd.DataFrame([
    [2015, 2.0,  2.0,  2.0, 0.0],
    [2016, 2.0,  2.0,  2.0, 0.0],
    [2017, 2.0,  2.0,  2.0, 0.0],
    [2018, 3.5, 12.0,  2.0, 40.0],
    [2019, 4.5, 19.0,  2.0, 65.0],
    [2020, 4.0, 19.5, 1.8, 65.0],
    [2021, 3.8, 20.0, 1.8, 62.0],
    [2022, 3.7, 20.0, 1.7, 60.0],
    [2023, 3.6, 20.5, 1.7, 58.0],
    [2024, 3.5, 21.0, 1.7, 55.0],
], columns=['year','tariff_index','china_tariff_index','row_tariff_index','china_tariff_coverage_pct'])

policy.to_csv(Q5_DIR / 'q5_tariff_indices_policy.csv', index=False)

print('Wrote:', Q5_DIR / 'q5_tariff_indices_calibrated.csv')
print('Wrote:', Q5_DIR / 'q5_tariff_indices_policy.csv')
