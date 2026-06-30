"""主实验运行脚本 — 依次执行所有实验并汇总结果到 JSON。

提供 run_all() 函数：
  1. 运行统计基线（季节性朴素预测，无需训练）
  2. 遍历所有模型（LSTM / Transformer / HCMRF + 4 个消融变体）× 2 个 horizon × 5 个随机种子
  3. 消融实验仅对 365d 运行（90d 路径不池化、patch=1，消融退化为恒等操作）
  4. 超参数消融（压缩因子、精修 kernel、粗预测周数）仅对 365d 运行
  5. 将结果写入 outputs/results/summary.json
"""
import json
import os

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import evaluate
from .train import train

# 5 个随机种子，用于可重复性实验
SEEDS = [42, 123, 456, 789, 2024]


def seasonal_naive_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int) -> dict:
    """季节性朴素基线 — 以去年同日的电力消耗作为预测（原始 kW 量纲）。"""
    train_map = dict(zip(train_df["Date"], train_df["Global_active_power"]))

    preds = []
    trues = []
    for _, row in test_df.head(horizon).iterrows():
        last_year_date = (pd.to_datetime(row["Date"]) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        if last_year_date in train_map:
            preds.append(train_map[last_year_date])
            trues.append(row["Global_active_power"])

    preds = np.array(preds)
    trues = np.array(trues)

    mse = float(np.mean((trues - preds) ** 2))
    mae = float(np.mean(np.abs(trues - preds)))
    return {"MSE": mse, "MAE": mae}


def _run_experiment(model_name: str, horizon: int, seeds: list[int]) -> dict:
    """运行单个模型的 5 轮实验，返回 {mean: ..., std: ...} 字典。"""
    metrics = []
    for seed in seeds:
        cfg = Config(model_name=model_name, horizon=horizon, seed=seed)
        print(f"  seed={seed}: training ...", end=" ", flush=True)
        ckpt = train(cfg)
        result = evaluate(cfg, ckpt)
        print(f"MSE={result['test/MSE']:.4f}, MAE={result['test/MAE']:.4f}")
        metrics.append(result)

    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    std = {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]}
    return {"mean": avg, "std": std}


def run_all():
    """运行所有实验并保存汇总结果到 JSON。"""
    os.makedirs("outputs/results", exist_ok=True)

    # ---- 主实验矩阵 ----
    # LSTM / Transformer: 两个 horizon 都跑
    # HCMRF: 两个 horizon 都跑
    # 消融变体: 仅 365d（90d 路径不池化、patch=1，消融退化为恒等操作）
    main_experiments = [
        ("lstm", 90), ("lstm", 365),
        ("transformer", 90), ("transformer", 365),
        ("hcmrf", 90), ("hcmrf", 365),
    ]
    ablation_models = [
        "hcmrf_wo_MultiScale",
        "hcmrf_wo_Patch",
        "hcmrf_wo_DRD",
        "hcmrf_wo_Shared",
    ]

    summary = {}

    # ---- 1. 统计基线 ----
    print("=" * 60)
    print("Statistical baseline (seasonal naive)")
    print("=" * 60)
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    for horizon in [90, 365]:
        result = seasonal_naive_baseline(train_df, test_df, horizon)
        summary[f"seasonal_naive_h{horizon}"] = {"mean": result, "std": {"MSE": 0.0, "MAE": 0.0}}
        print(f"  horizon={horizon}: MSE={result['MSE']:.4f}, MAE={result['MAE']:.4f}")

    # ---- 2. 主实验（LSTM / Transformer / HCMRF） ----
    for model_name, horizon in main_experiments:
        print("=" * 60)
        print(f"Model: {model_name}, horizon={horizon}")
        print("=" * 60)
        summary[f"{model_name}_h{horizon}"] = _run_experiment(model_name, horizon, SEEDS)
        avg = summary[f"{model_name}_h{horizon}"]["mean"]
        std = summary[f"{model_name}_h{horizon}"]["std"]
        print(f"  → mean MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # ---- 3. 消融实验（仅 365d） ----
    ablation_summary = {}
    for model_name in ablation_models:
        print("=" * 60)
        print(f"Ablation: {model_name}, horizon=365")
        print("=" * 60)
        result = _run_experiment(model_name, 365, SEEDS)
        ablation_summary[f"{model_name}_h365"] = result
        # 同时写入主 summary
        summary[f"{model_name}_h365"] = result
        avg = result["mean"]
        std = result["std"]
        print(f"  → mean MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # ---- 4. 超参数消融（仅 365d） ----
    hyperparam_summary = {}

    # 4a. 压缩因子消融
    print("\n" + "=" * 60)
    print("Hyperparameter ablation: compress_factor")
    print("=" * 60)
    for cf in [2, 3, 4]:
        label = f"hcmrf_cf{cf}"
        print(f"  compress_factor={cf}:")
        metrics = []
        for seed in SEEDS:
            cfg = Config(model_name="hcmrf", horizon=365, seed=seed,
                         hcmrf_hcm_compress_factor=cf, ckpt_prefix=label)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]}
        hyperparam_summary[f"compress_factor_{cf}"] = {"mean": avg, "std": std}
        print(f"    → MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # 4b. 精修 kernel 消融
    print("\n" + "=" * 60)
    print("Hyperparameter ablation: refine_kernel")
    print("=" * 60)
    for rk in [3, 5, 7]:
        label = f"hcmrf_rk{rk}"
        print(f"  refine_kernel={rk}:")
        metrics = []
        for seed in SEEDS:
            cfg = Config(model_name="hcmrf", horizon=365, seed=seed,
                         hcmrf_drd_refine_kernel=rk, ckpt_prefix=label)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]}
        hyperparam_summary[f"refine_kernel_{rk}"] = {"mean": avg, "std": std}
        print(f"    → MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # 4c. 粗预测周数消融
    print("\n" + "=" * 60)
    print("Hyperparameter ablation: coarse_weeks")
    print("=" * 60)
    for cw in [26, 52]:
        label = f"hcmrf_cw{cw}"
        print(f"  coarse_weeks={cw}:")
        metrics = []
        for seed in SEEDS:
            cfg = Config(model_name="hcmrf", horizon=365, seed=seed,
                         hcmrf_drd_coarse_weeks=cw, ckpt_prefix=label)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]}
        hyperparam_summary[f"coarse_weeks_{cw}"] = {"mean": avg, "std": std}
        print(f"    → MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # ---- 保存结果 ----
    with open("outputs/results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMain results saved to outputs/results/summary.json")

    with open("outputs/results/ablation_summary.json", "w") as f:
        # 加入完整模型作为基线
        ablation_with_full = {"hcmrf_h365": summary["hcmrf_h365"]}
        ablation_with_full.update(ablation_summary)
        json.dump(ablation_with_full, f, indent=2)
    print(f"Ablation results saved to outputs/results/ablation_summary.json")

    with open("outputs/results/hyperparam_ablation.json", "w") as f:
        json.dump(hyperparam_summary, f, indent=2)
    print(f"Hyperparameter ablation saved to outputs/results/hyperparam_ablation.json")


if __name__ == "__main__":
    run_all()
