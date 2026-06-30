"""重新评估所有已有 checkpoint — 使用修复后的逆归一化逻辑。

直接调用 src.evaluate.evaluate()，将结果保存到 outputs/results/。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

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
    ("hcmrf_wo_Shared", 365),
]


def main():
    os.makedirs("outputs/results", exist_ok=True)

    # 1. 基线（原始kW量纲）
    print("=" * 60)
    print("Seasonal Naive Baseline (original kW)")
    print("=" * 60)
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    baseline = {}
    for horizon in [90, 365]:
        result = seasonal_naive_baseline(train_df, test_df, horizon)
        baseline[f"seasonal_naive_h{horizon}"] = {"mean": result, "std": {"MSE": 0.0, "MAE": 0.0}}
        print(f"  h{horizon}: MSE={result['MSE']:.2f}, MAE={result['MAE']:.2f}")

    with open("outputs/results/baseline_summary.json", "w") as f:
        json.dump(baseline, f, indent=2)

    # 2. 深度学习模型（原始kW量纲）
    summary = {}
    for model_name, horizon in EXPERIMENTS:
        print(f"\n{'='*60}\nModel: {model_name}, horizon={horizon}\n{'='*60}")
        metrics = []
        for seed in SEEDS:
            cfg = Config(model_name=model_name, horizon=horizon, seed=seed)
            ckpt_path = f"outputs/checkpoints/{model_name}_h{horizon}_s{seed}.ckpt"
            if not os.path.exists(ckpt_path):
                print(f"  SKIP: {ckpt_path} not found")
                continue
            print(f"  seed={seed}: evaluating...", end=" ", flush=True)
            result = evaluate(cfg, ckpt_path)
            print(f"MSE={result['test/MSE']:.2f}, MAE={result['test/MAE']:.2f}")
            metrics.append(result)

        if not metrics:
            print(f"  WARNING: no metrics collected")
            continue

        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]}
        summary[f"{model_name}_h{horizon}"] = {"mean": avg, "std": std}
        print(f"  → mean MSE={avg['test/MSE']:.2f} ± {std['test/MSE']:.2f}")

    with open("outputs/results/final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 3. 消融实验汇总
    ablation = {}
    for key, val in summary.items():
        if key.startswith("hcmrf"):
            ablation[key] = val
    with open("outputs/results/ablation_summary.json", "w") as f:
        json.dump(ablation, f, indent=2)

    print("\n\nDone. Results saved to outputs/results/")


if __name__ == "__main__":
    main()
