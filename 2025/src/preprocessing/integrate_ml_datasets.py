"""
Integrated ML-ready datasets for APMCM 2025 Problem C.

This script reads already processed CSVs in DATA_PROCESSED and builds
compact, machine-learning-friendly datasets for Q1–Q5:

- Q1: adds target_import_quantity to q1_1.csv
- Q2: q2_1.csv  (brand-year panel with relocation intensity)
- Q3: q3_1.csv  (semiconductor output + policy panel)
- Q4: q4_1.csv  (historical + scenario tariff revenue panel)
- Q5: q5_1.csv  (macro-financial-reshoring integrated panel with index target)

The original processed files are preserved; this script only creates
additional integrated views or adds extra columns.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).parents[1]))

from utils.config import DATA_PROCESSED, ensure_directories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _safe_read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    """Read CSV if exists; optionally raise if required."""
    if not path.exists():
        msg = f"Required file not found: {path}"
        if required:
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return pd.DataFrame()
    return pd.read_csv(path)

def _zscore(series: pd.Series) -> pd.Series:
    """Compute simple z-score, handling constant or empty series."""
    s = series.astype(float)
    if s.dropna().empty:
        return pd.Series(0.0, index=s.index)
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std

def integrate_q1_ml() -> None:
    """Add ML target column to q1_1.csv (monthly soybean imports)."""
    q1_dir = DATA_PROCESSED / "q1"
    path = q1_dir / "q1_1.csv"
    df = _safe_read_csv(path, required=False)
    if df.empty:
        logger.warning("Q1: q1_1.csv not found or empty; skipping")
        return

    if "netWgt" in df.columns:
        df["target_import_quantity"] = df["netWgt"]
        df.to_csv(path, index=False)
        logger.info("Q1: added target_import_quantity to q1_1.csv (%d rows)", len(df))
    else:
        logger.warning("Q1: netWgt column not found in q1_1.csv; no target added")

def build_q2_1() -> None:
    """Build q2_1.csv: brand-year auto panel with relocation intensity target."""
    q2_dir = DATA_PROCESSED / "q2"
    sales_path = q2_dir / "q2_0_us_auto_sales_by_brand.csv"
    ind_path = q2_dir / "q2_1_industry_indicators.csv"

    sales = _safe_read_csv(sales_path, required=False)
    if sales.empty:
        logger.warning("Q2: sales file %s not found or empty; skipping q2_1.csv", sales_path)
        return

    ind = _safe_read_csv(ind_path, required=False)
    if ind.empty:
        logger.warning("Q2: industry indicators %s not found; proceeding without controls", ind_path)

    df = sales.copy()

    required_cols = [
        "year",
        "brand",
        "total_sales",
        "us_produced",
        "mexico_produced",
        "japan_imported",
        "origin",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error("Q2: missing required columns in sales data: %s", missing)
        return

    total = df["total_sales"].replace(0, np.nan)
    df["share_us_produced"] = df["us_produced"] / total
    df["share_mexico_produced"] = df["mexico_produced"] / total
    df["share_japan_imported"] = df["japan_imported"] / total

    df["target_japan_relocation_intensity"] = df["us_produced"] / total

    if not ind.empty and "year" in ind.columns:
        df = df.merge(ind, on="year", how="left", suffixes=("", "_ind"))

    out_path = q2_dir / "q2_1.csv"
    df.to_csv(out_path, index=False)
    logger.info("Q2: saved integrated ML dataset to %s (%d rows, %d cols)", out_path, len(df), df.shape[1])

def build_q3_1() -> None:
    """Build q3_1.csv: semiconductor output + policy integrated panel."""
    q3_dir = DATA_PROCESSED / "q3"
    out_path = q3_dir / "q3_1.csv"

    output_path = q3_dir / "q3_0_us_semiconductor_output.csv"
    policies_path = q3_dir / "q3_1_chip_policies.csv"

    out_df = _safe_read_csv(output_path, required=False)
    if out_df.empty:
        logger.warning("Q3: %s not found or empty; skipping q3_1.csv", output_path)
        return

    pol_df = _safe_read_csv(policies_path, required=False)
    if pol_df.empty:
        logger.warning("Q3: %s not found; proceeding with output data only", policies_path)
        df = out_df.copy()
    else:
        if "year" not in out_df.columns or "year" not in pol_df.columns:
            logger.error("Q3: year column missing in output or policy data; cannot merge")
            return
        df = out_df.merge(pol_df, on="year", how="left", suffixes=("", "_policy"))

    if "supply_chain_risk_index" in df.columns:
        df["target_supply_chain_risk_score"] = df["supply_chain_risk_index"]
    else:
        logger.warning("Q3: supply_chain_risk_index not found; target column not added")

    df.to_csv(out_path, index=False)
    logger.info("Q3: saved integrated ML dataset to %s (%d rows, %d cols)", out_path, len(df), df.shape[1])

def build_q4_1() -> None:
    """Build q4_1.csv: historical + scenario tariff revenue panel."""
    q4_dir = DATA_PROCESSED / "q4"
    hist_path = q4_dir / "q4_0_tariff_revenue_panel.csv"
    scen_path = q4_dir / "q4_1_tariff_scenarios.csv"
    out_path = q4_dir / "q4_1.csv"

    hist = _safe_read_csv(hist_path, required=False)
    scen = _safe_read_csv(scen_path, required=False)

    if hist.empty and scen.empty:
        logger.warning("Q4: neither historical nor scenario files found; skipping q4_1.csv")
        return

    frames: List[pd.DataFrame] = []

    if not hist.empty:
        hist_df = hist.copy()
        hist_df["scenario"] = "historical"
        frames.append(hist_df)

    if not scen.empty:
        scen_df = scen.copy()
        if "scenario" not in scen_df.columns:
            scen_df["scenario"] = "scenario"
        frames.append(scen_df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if "total_tariff_revenue_usd" in combined.columns:
        combined["target_tariff_revenue"] = combined["total_tariff_revenue_usd"]

    if "expected_revenue_billions" in combined.columns:
        combined["target_tariff_revenue_scenario"] = combined["expected_revenue_billions"] * 1e9

    combined.to_csv(out_path, index=False)
    logger.info("Q4: saved integrated ML dataset to %s (%d rows, %d cols)", out_path, len(combined), combined.shape[1])

def build_q5_1() -> None:
    """Build q5_1.csv: macro-financial-reshoring integrated panel with index target."""
    q5_dir = DATA_PROCESSED / "q5"
    out_path = q5_dir / "q5_1.csv"

    macro_path = q5_dir / "q5_0_macro_indicators.csv"
    fin_path = q5_dir / "q5_1_financial_indicators.csv"
    resh_path = q5_dir / "q5_2_reshoring_indicators.csv"
    ret_path = q5_dir / "q5_3_retaliation_index.csv"
    tariff_path = q5_dir / "q5_tariff_indices_calibrated.csv"

    macro = _safe_read_csv(macro_path, required=False)
    if macro.empty:
        logger.warning("Q5: %s not found or empty; skipping q5_1.csv", macro_path)
        return

    df = macro.copy()

    fin = _safe_read_csv(fin_path, required=False)
    if not fin.empty and "year" in fin.columns:
        df = df.merge(fin, on="year", how="left", suffixes=("", "_fin"))

    resh = _safe_read_csv(resh_path, required=False)
    if not resh.empty and "year" in resh.columns:
        df = df.merge(resh, on="year", how="left", suffixes=("", "_resh"))

    ret = _safe_read_csv(ret_path, required=False)
    if not ret.empty and "year" in ret.columns:
        df = df.merge(ret, on="year", how="left", suffixes=("", "_ret"))

    tariff = _safe_read_csv(tariff_path, required=False)
    if not tariff.empty and "year" in tariff.columns:
        df = df.merge(tariff, on="year", how="left", suffixes=("", "_tariff"))

    comp_cols = ["manufacturing_va_share", "manufacturing_employment_share", "reshoring_fdi_billions"]
    missing = [c for c in comp_cols if c not in df.columns]
    if missing:
        logger.warning("Q5: missing components for reshoring index: %s", missing)
    else:
        z_va = _zscore(df["manufacturing_va_share"])
        z_emp = _zscore(df["manufacturing_employment_share"])
        z_fdi = _zscore(df["reshoring_fdi_billions"])
        df["z_manufacturing_va_share"] = z_va
        df["z_manufacturing_employment_share"] = z_emp
        df["z_reshoring_fdi_billions"] = z_fdi
        df["manufacturing_reshoring_index"] = (z_va + z_emp + z_fdi) / 3.0
        df["target_manufacturing_reshoring_index"] = df["manufacturing_reshoring_index"]

    df.to_csv(out_path, index=False)
    logger.info("Q5: saved integrated ML dataset to %s (%d rows, %d cols)", out_path, len(df), df.shape[1])

def main() -> None:
    logger.info("=" * 70)
    logger.info("APMCM 2025 Problem C - ML Dataset Integration")
    logger.info("=" * 70)

    ensure_directories()

    integrate_q1_ml()
    build_q2_1()
    build_q3_1()
    build_q4_1()
    build_q5_1()

    logger.info("All integrated ML datasets have been generated (where source data exists).")

if __name__ == "__main__":
    main()
