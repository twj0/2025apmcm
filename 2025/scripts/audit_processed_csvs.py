"""Utility to audit all CSV files under 2025/data/processed.

Prints summary information for each file:
- file path, size, row/column counts
- year range if a year column is present
- column dtypes and missing values
- duplicate rows and duplicate keys on common identifiers
- simple numeric anomaly flags (negative / extreme values)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes


def analyze_file(path: Path) -> None:
    df = pd.read_csv(path)

    print(f"FILE: {path}")
    print(f"  size_bytes: {path.stat().st_size}")
    print(f"  shape: {df.shape[0]} rows x {df.shape[1]} cols")

    cols_lower = {c.lower(): c for c in df.columns}
    if "year" in cols_lower:
        ycol = cols_lower["year"]
        try:
            year_min = df[ycol].min()
            year_max = df[ycol].max()
            print(f"  year range: {year_min} - {year_max}")
        except Exception:
            pass

    print("  columns:")
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        n_missing = int(s.isna().sum())
        missing_pct = float(s.isna().mean()) * 100.0 if len(df) else 0.0
        line = f"    - {col}: dtype={dtype}, missing={n_missing} ({missing_pct:.2f}%)"

        if ptypes.is_numeric_dtype(s):
            try:
                mn = s.min()
                mx = s.max()
                line += f", range=[{mn}, {mx}]"
            except Exception:
                pass

        print(line)

    # Duplicate checks
    n_dup = int(df.duplicated().sum())
    print(f"  duplicated rows (full row): {n_dup}")

    key = None
    lower_set = {c.lower() for c in df.columns}
    if {"year", "exporter"}.issubset(lower_set):
        key = ["year", "exporter"]
    elif {"year", "brand"}.issubset(lower_set):
        key = ["year", "brand"]
    elif "year" in lower_set:
        key = ["year"]

    if key is not None:
        key_cols = [cols_lower[k] for k in key]
        dup_key = int(df.duplicated(subset=key_cols).sum())
        print(f"  duplicated rows on key {key_cols}: {dup_key}")

    # Simple numeric anomaly flags
    anomalies: list[str] = []
    for col in df.columns:
        s = df[col]
        if not ptypes.is_numeric_dtype(s):
            continue
        try:
            mn = s.min()
            mx = s.max()
        except Exception:
            continue

        if pd.notna(mn) and mn < 0:
            anomalies.append(
                f"    - {col}: negative values present, range=[{mn}, {mx}]"
            )
        if pd.notna(mx) and abs(mx) > 1e12:
            anomalies.append(
                f"    - {col}: extreme magnitude values, range=[{mn}, {mx}]"
            )

    if anomalies:
        print("  anomalies:")
        for msg in anomalies:
            print(msg)

    print()


def main() -> None:
    base = Path("2025/data/processed")
    files = sorted(base.rglob("*.csv"))

    if not files:
        print(f"No CSV files found under {base}")
        return

    for path in files:
        analyze_file(path)


if __name__ == "__main__":
    main()

