from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.q3_semiconductors import SemiconductorModel


def main() -> None:
    model = SemiconductorModel()

    trade_data = model.load_q3_data()
    output_data, policy_data = model.load_external_chip_data()
    security_metrics = model.compute_security_metrics()

    output_dir = Path("2025/data/q3")
    output_dir.mkdir(parents=True, exist_ok=True)

    trade_data.to_csv(output_dir / "q3_trade_charges.csv", index=False)
    output_data.to_csv(output_dir / "q3_output_by_segment.csv", index=False)
    policy_data.to_csv(output_dir / "q3_policies.csv", index=False)
    security_metrics.to_csv(output_dir / "q3_security_metrics.csv", index=False)

    print(f"Exported Q3 trade data ({len(trade_data)}) and summaries to {output_dir}")


if __name__ == "__main__":
    main()
