"""生成可直接放入课程报告的正式实验结果快照。"""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def metric(record: dict, key: str) -> tuple[float, float]:
    return record["mean"][f"test/{key}"], record["std"][f"test/{key}"]


def main() -> None:
    root = Path("outputs/revised")
    data = json.loads((root / "results/summary.json").read_text())
    lines = [
        "$ python -m src.cli run-all --config configs/default.yaml",
        "Protocol: 90-day input; isolated validation/test targets; 5 seeds",
        "Metric unit: daily energy (kWh)",
        "",
        "model          horizon     MSE (mean +/- sd)      MAE (mean +/- sd)",
        "--------------------------------------------------------------------",
    ]
    for horizon in (90, 365):
        for model in ("lstm", "transformer", "hcmrf"):
            record = data[f"{model}_h{horizon}"]
            mse, mse_sd = metric(record, "MSE")
            mae, mae_sd = metric(record, "MAE")
            lines.append(
                f"{model:<14} {horizon:<7} {mse:>8.3f} +/- {mse_sd:<7.3f} "
                f"{mae:>8.3f} +/- {mae_sd:<7.3f}"
            )

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    ax.axis("off")
    ax.text(
        0.025, 0.96, "\n".join(lines), va="top", ha="left",
        family="DejaVu Sans Mono", fontsize=12, color="#e5e7eb", linespacing=1.35,
    )
    output = root / "figures/result_snapshot.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
