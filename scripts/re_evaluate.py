"""重新评估所有已有 checkpoint — 使用修复后的逆归一化逻辑。

只评估带 scaler 与 manifest 的正式 checkpoint，结果保存到 outputs/revised/results/。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.evaluate import evaluate
from src.run import seasonal_naive_baseline

SEEDS = [42, 123, 456, 789, 2024]

EXPERIMENTS = [
    ("lstm", 90), ("lstm", 365),
    ("transformer", 90), ("transformer", 365),
    ("hcmrf", 90), ("hcmrf", 365),
    # 消融变体仅对 365d（90d 路径不池化、patch=1，消融退化为恒等操作）
    ("hcmrf_wo_MultiScale", 365),
    ("hcmrf_wo_Patch", 365),
    ("hcmrf_wo_DRD", 365),
]


def latest_checkpoint(model_name: str, horizon: int, seed: int) -> str | None:
    candidates = list(Path("outputs/revised/checkpoints").glob(f"{model_name}_h{horizon}_s{seed}*.ckpt"))
    candidates = [path for path in candidates if path.with_suffix(".artifacts.joblib").exists()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def main():
    os.makedirs("outputs/revised/results", exist_ok=True)

    # 1. 基线（日用电量 kWh）
    print("=" * 60)
    print("Seasonal Naive Baseline (daily kWh)")
    print("=" * 60)
    baseline = {}
    for horizon in [90, 365]:
        result = seasonal_naive_baseline("data/processed", Config().input_len, horizon)
        baseline[f"seasonal_naive_h{horizon}"] = {"mean": result, "std": {"MSE": 0.0, "MAE": 0.0}}
        print(f"  h{horizon}: MSE={result['MSE']:.2f}, MAE={result['MAE']:.2f}")

    with open("outputs/revised/results/baseline_summary.json", "w") as f:
        json.dump(baseline, f, indent=2)

    # 2. 深度学习模型（日用电量 kWh）
    summary = {}
    for model_name, horizon in EXPERIMENTS:
        print(f"\n{'='*60}\nModel: {model_name}, horizon={horizon}\n{'='*60}")
        metrics = []
        for seed in SEEDS:
            cfg = Config(model_name=model_name, horizon=horizon, seed=seed)
            ckpt_path = latest_checkpoint(model_name, horizon, seed)
            if ckpt_path is None:
                print(f"  SKIP: no checkpoint for {model_name} h{horizon} seed={seed}")
                continue
            print(f"  seed={seed}: evaluating...", end=" ", flush=True)
            result = evaluate(cfg, ckpt_path)
            print(f"MSE={result['test/MSE']:.2f}, MAE={result['test/MAE']:.2f}")
            metrics.append(result)

        if not metrics:
            print(f"  WARNING: no metrics collected")
            continue

        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics], ddof=1)) for k in metrics[0]}
        summary[f"{model_name}_h{horizon}"] = {"mean": avg, "std": std, "runs": metrics}
        print(f"  → mean MSE={avg['test/MSE']:.2f} ± {std['test/MSE']:.2f}")

    with open("outputs/revised/results/final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 3. 消融实验汇总
    ablation = {}
    for key, val in summary.items():
        if key.startswith("hcmrf"):
            ablation[key] = val
    with open("outputs/revised/results/ablation_summary.json", "w") as f:
        json.dump(ablation, f, indent=2)

    print("\n\nDone. Results saved to outputs/revised/results/")


if __name__ == "__main__":
    main()
