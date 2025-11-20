"""Preprocess official China soybean import data into Q1-ready CSV.

This script converts an "official" China soybean imports dataset (e.g., from
UN Comtrade+ / WITS) into the standardized format expected by the Q1
soybeans model:

    year, exporter, import_value_usd, import_quantity_tonnes, tariff_cn_on_exporter

The script is intentionally tolerant to different source schemas. It will:

- Detect common column names for year, partner/exporter, trade value (USD),
  and quantity (typically net weight in kilograms).
- Normalize them into the above canonical columns.
- Optionally merge a tariff mapping file providing `tariff_cn_on_exporter`
  by (year, exporter).

By default it reads from:

    2025/data/external/china_imports_soybeans_official.csv

and writes to:

    2025/data/external/china_imports_soybeans.csv

so that the Q1 model can consume high-quality data instead of sample values.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)


def _find_column(df: pd.DataFrame, patterns: Dict[str, float], required: bool = True) -> Optional[str]:
    """Find a column whose lowercased name contains any of the patterns.

    Args:
        df: Input DataFrame.
        patterns: Mapping of pattern string to weight (unused, reserved for
            future scoring). Only keys are used.
        required: If True, raise ValueError when not found; otherwise return None.

    Returns:
        Name of the first matching column, or None.
    """

    lower_to_orig = {str(col).lower(): str(col) for col in df.columns}

    for pattern in patterns.keys():
        pattern_l = pattern.lower()
        # Exact match first
        if pattern_l in lower_to_orig:
            return lower_to_orig[pattern_l]
        # Then substring match
        for lower, orig in lower_to_orig.items():
            if pattern_l in lower:
                return orig

    if required:
        raise ValueError(
            f"Could not find any column matching patterns: {list(patterns.keys())}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_official_data(path: Path) -> pd.DataFrame:
    """Load the official China soybean import dataset.

    This function is schema-tolerant and supports two main cases:

    1. Already-standardized file produced by `fetch_un_comtrade_soybeans`,
       with columns like: year, exporter, import_value_usd,
       import_quantity_tonnes, hs_code, importer, flow.
    2. Raw CSV exported from WITS / UN Comtrade, where columns are more
       verbose (e.g., "Year", "Partner", "Trade Value (US$)",
       "Netweight (kg)").
    """

    if not path.exists():
        raise FileNotFoundError(f"Official data file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Official data file is empty: {path}")

    cols = {str(c).lower() for c in df.columns}

    # Case 1: Already in canonical form.
    if {"year", "exporter", "import_value_usd"}.issubset(cols):
        LOGGER.info("Detected standardized official schema with canonical columns.")
        # Ensure quantity column exists (may be missing).
        if "import_quantity_tonnes" not in cols:
            LOGGER.warning(
                "Official file lacks 'import_quantity_tonnes'; filling with NA. "
                "Consider recomputing from raw quantity if available."
            )
            df["import_quantity_tonnes"] = pd.NA
        canonical = df.copy()
        canonical["year"] = pd.to_numeric(canonical["year"], errors="coerce").astype("Int64")
        canonical["exporter"] = canonical["exporter"].astype("string")
        canonical["import_value_usd"] = pd.to_numeric(
            canonical["import_value_usd"], errors="coerce"
        )
        canonical["import_quantity_tonnes"] = pd.to_numeric(
            canonical["import_quantity_tonnes"], errors="coerce"
        )
        return canonical[["year", "exporter", "import_value_usd", "import_quantity_tonnes"]]

    # Case 2: Raw WITS / Comtrade-like schema.
    LOGGER.info("Detected non-standard schema; attempting flexible column mapping.")

    year_col = _find_column(
        df,
        {"year": 1.0},
        required=True,
    )
    exporter_col = _find_column(
        df,
        {
            "partner": 1.0,
            "partner name": 1.0,
            "partner_desc": 1.0,
        },
        required=True,
    )
    value_col = _find_column(
        df,
        {
            "trade value": 1.0,
            "trade_value": 1.0,
            "value (us$)": 1.0,
            "tradevalue": 1.0,
            "trade_value_us$": 1.0,
        },
        required=True,
    )
    quantity_col = _find_column(
        df,
        {
            "netweight": 1.0,
            "net weight": 1.0,
            "quantity": 1.0,
            "qty": 1.0,
        },
        required=False,
    )

    year = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    exporter = df[exporter_col].astype("string")
    import_value_usd = pd.to_numeric(df[value_col], errors="coerce")

    if quantity_col:
        qty_raw = pd.to_numeric(df[quantity_col], errors="coerce")
        col_lower = quantity_col.lower()
        # Heuristic: if column mentions kg, convert to tonnes.
        if "kg" in col_lower or "netweight" in col_lower:
            import_quantity_tonnes = qty_raw / 1000.0
        else:
            import_quantity_tonnes = qty_raw
    else:
        LOGGER.warning(
            "Could not detect quantity column; 'import_quantity_tonnes' will be NA."
        )
        import_quantity_tonnes = pd.Series([pd.NA] * len(df))

    out = pd.DataFrame(
        {
            "year": year,
            "exporter": exporter,
            "import_value_usd": import_value_usd,
            "import_quantity_tonnes": import_quantity_tonnes,
        }
    )

    # Drop rows without year or value
    out = out.dropna(subset=["year", "import_value_usd"])
    return out


def load_tariffs(tariff_path: Path) -> Optional[pd.DataFrame]:
    """Load optional tariff mapping file.

    Expected schema:
        year, exporter, tariff_cn_on_exporter

    If the file does not exist, returns None and logs a warning.
    """

    if not tariff_path.exists():
        LOGGER.warning(
            "Tariff mapping file not found at %s; "
            "'tariff_cn_on_exporter' will default to 0.0.",
            tariff_path,
        )
        return None

    df = pd.read_csv(tariff_path)
    required_cols = {"year", "exporter", "tariff_cn_on_exporter"}
    if not required_cols.issubset({c.lower() for c in df.columns}):
        raise ValueError(
            f"Tariff file {tariff_path} must contain columns {required_cols}, "
            f"got {list(df.columns)}"
        )

    # Normalize column names
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "year":
            rename_map[col] = "year"
        elif lower == "exporter":
            rename_map[col] = "exporter"
        elif lower == "tariff_cn_on_exporter":
            rename_map[col] = "tariff_cn_on_exporter"
    df = df.rename(columns=rename_map)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["exporter"] = df["exporter"].astype("string")
    df["tariff_cn_on_exporter"] = pd.to_numeric(
        df["tariff_cn_on_exporter"], errors="coerce"
    )

    return df[["year", "exporter", "tariff_cn_on_exporter"]]


def preprocess_soybeans(
    official_path: Path,
    output_path: Path,
    tariff_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Convert official China soybean imports to Q1-ready CSV.

    Args:
        official_path: Path to official data (from UN Comtrade+/WITS/etc.).
        output_path: Path where the Q1-ready CSV will be written.
        tariff_path: Optional path to tariff mapping CSV.

    Returns:
        Processed DataFrame with columns:
            year, exporter, import_value_usd, import_quantity_tonnes,
            tariff_cn_on_exporter
    """

    LOGGER.info("Loading official data from %s", official_path)
    base_df = load_official_data(official_path)

    if tariff_path is not None:
        tariffs = load_tariffs(tariff_path)
    else:
        tariffs = None

    if tariffs is not None and not tariffs.empty:
        LOGGER.info("Merging tariff mapping from %s", tariff_path)
        merged = base_df.merge(
            tariffs,
            on=["year", "exporter"],
            how="left",
        )
    else:
        merged = base_df.copy()
        merged["tariff_cn_on_exporter"] = 0.0

    # Fill any missing tariffs with 0.0 (no tariff) to keep Q1 pipeline running.
    if merged["tariff_cn_on_exporter"].isna().any():
        LOGGER.warning(
            "Some observations missing tariffs; filling NaN with 0.0. "
            "Consider providing a richer tariff mapping file."
        )
        merged["tariff_cn_on_exporter"] = merged["tariff_cn_on_exporter"].fillna(0.0)

    merged = merged.sort_values(["year", "exporter"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    LOGGER.info("Saved Q1-ready soybean data to %s", output_path)

    return merged


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess official China soybean import data into the format "
            "required by the Q1 analysis."
        )
    )
    parser.add_argument(
        "--official-file",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "china_imports_soybeans_official.csv"
        ),
        help=(
            "Path to the official China soybean imports CSV. "
            "Default: 2025/data/external/china_imports_soybeans_official.csv"
        ),
    )
    parser.add_argument(
        "--tariff-file",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "china_soybean_tariffs.csv"
        ),
        help=(
            "Optional tariff mapping CSV with columns: year, exporter, "
            "tariff_cn_on_exporter. If not found, tariffs default to 0.0."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "china_imports_soybeans.csv"
        ),
        help=(
            "Output CSV path for the Q1-ready data. "
            "Default: 2025/data/external/china_imports_soybeans.csv"
        ),
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    official_path = Path(args.official_file)
    tariff_path = Path(args.tariff_file) if args.tariff_file else None
    output_path = Path(args.output_file)

    preprocess_soybeans(
        official_path=official_path,
        output_path=output_path,
        tariff_path=tariff_path,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
