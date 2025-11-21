from pathlib import Path
import pandas as pd

PROCESSED = Path("2025/data/processed/q1/q1_0.csv")
MONTHLY = Path("2025/data/external/q1_soybean_imports_comtrade_monthly.csv")

processed = pd.read_csv(PROCESSED)
monthly = pd.read_csv(MONTHLY, dtype={"period": str})
monthly["year"] = monthly["period"].str[:4].astype(int)
monthly["exporter"] = monthly["partnerDesc"]
monthly["import_quantity_tonnes"] = monthly["netWgt"] / 1000
monthly["import_value_usd"] = monthly["primaryValue"]
agg = (
    monthly.groupby(["year", "exporter"], as_index=False)
    .agg({
        "import_quantity_tonnes": "sum",
        "import_value_usd": "sum",
    })
)

agg["unit_price_usd_per_ton"] = agg["import_value_usd"] / agg["import_quantity_tonnes"]

merged = processed.merge(
    agg,
    on=["year", "exporter"],
    how="left",
    suffixes=("", "_agg"),
)
for col in ["import_quantity_tonnes", "import_value_usd", "unit_price_usd_per_ton"]:
    merged[col] = merged[f"{col}_agg"].combine_first(merged[col])
    merged.drop(columns=f"{col}_agg", inplace=True)

merged.to_csv(PROCESSED, index=False)
print(f"Merged monthly aggregates ({len(agg)}) into processed q1_0.csv")
