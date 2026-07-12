"""汇总正式实验的报告指标、数据切分和模型参数量。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.train import build_model
from src.windows import prepare_windows


OUTPUT_ROOT = Path("outputs/revised")


def parameter_count(model_name: str, horizon: int) -> int:
    model = build_model(Config(model_name=model_name, horizon=horizon))
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    summary = json.loads((OUTPUT_ROOT / "results/summary.json").read_text())
    protocol = {}
    for horizon in (90, 365):
        prepared = prepare_windows("data/processed", 90, horizon, 1)
        split = prepared.split
        val_target_start = split.val_starts[0] + split.input_len
        test_target_start = split.test_starts[0] + split.input_len
        protocol[str(horizon)] = {
            "train_windows": len(split.train_starts),
            "validation_target": [
                str(prepared.dates.iloc[val_target_start].date()),
                str(prepared.dates.iloc[val_target_start + horizon - 1].date()),
            ],
            "test_target": [
                str(prepared.dates.iloc[test_target_start].date()),
                str(prepared.dates.iloc[test_target_start + horizon - 1].date()),
            ],
        }

    report = {
        "protocol": protocol,
        "parameters": {
            f"{name}_h{horizon}": parameter_count(name, horizon)
            for name in ("lstm", "transformer", "hcmrf")
            for horizon in (90, 365)
        },
        "results": summary,
    }
    (OUTPUT_ROOT / "results/report_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"protocol": protocol, "parameters": report["parameters"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
