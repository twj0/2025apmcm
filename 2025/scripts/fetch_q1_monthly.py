from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_uncomtrade import get_soybean_data


def main() -> None:
    df = get_soybean_data()
    if df is None:
        raise SystemExit("No data fetched from UN Comtrade")

    output_path = Path("2025/data/external/q1_soybean_imports_comtrade_monthly.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
