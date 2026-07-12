"""主实验运行脚本 — 依次执行所有实验并汇总结果到 JSON。

提供 run_all() 函数：
  1. 运行统计基线（季节性朴素预测，无需训练）
  2. 遍历主模型（LSTM / Transformer / HCMRF）与 3 个正式消融变体 × 5 个随机种子
  3. 消融实验仅对 365d 运行（90d 路径不池化、patch=1，消融退化为恒等操作）
  4. 可选超参数消融（压缩因子、精修 kernel、粗预测周数）仅对 365d 运行
  5. 将结果写入 outputs/revised/results/summary.json
"""
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import Config
from .evaluate import evaluate
from .features import add_features
from .train import train
from .windows import load_daily_frame, make_window_split, prepare_windows

# 5 个随机种子，用于可重复性实验
SEEDS = [42, 123, 456, 789, 2024]


def seasonal_naive_baseline(data_dir: str, input_len: int, horizon: int) -> dict:
    """季节性朴素基线 — 以去年同日的日用电量作为预测（kWh 量纲）。"""
    daily = add_features(load_daily_frame(data_dir))
    split = make_window_split(len(daily), input_len, horizon, step_size=7)
    target = daily.iloc[split.test_target_start : split.test_target_start + horizon]
    history = dict(zip(daily["Date"], daily["Global_active_power"]))

    preds = []
    trues = []
    for _, row in target.iterrows():
        last_year_date = (pd.to_datetime(row["Date"]) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        if last_year_date in history:
            preds.append(history[last_year_date])
            trues.append(row["Global_active_power"])

    preds = np.array(preds)
    trues = np.array(trues)

    mse = float(np.mean((trues - preds) ** 2))
    mae = float(np.mean(np.abs(trues - preds)))
    return {"MSE": mse, "MAE": mae}


def ridge_direct_baseline(data_dir: str, input_len: int, horizon: int, step_size: int = 1) -> dict:
    """Ridge 直接多步基线：展平输入窗口并一次输出整个预测区间。"""
    prepared = prepare_windows(data_dir, input_len, horizon, step_size)
    split = prepared.split
    x_train = np.stack([
        prepared.values[start : start + input_len].reshape(-1)
        for start in split.train_starts
    ])
    y_train = np.stack([
        prepared.values[start + input_len : start + input_len + horizon, 0]
        for start in split.train_starts
    ])
    test_start = split.test_starts[0]
    x_test = prepared.values[test_start : test_start + input_len].reshape(1, -1)
    y_test = prepared.values[test_start + input_len : test_start + input_len + horizon, 0]

    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test).reshape(-1)

    n_features = prepared.values.shape[1]
    def inverse_target(values: np.ndarray) -> np.ndarray:
        dummy = np.zeros((values.size, n_features))
        dummy[:, 0] = values.reshape(-1)
        return prepared.scaler.inverse_transform(dummy)[:, 0]

    pred_original = inverse_target(y_pred)
    true_original = inverse_target(y_test)
    return {
        "MSE": float(np.mean((true_original - pred_original) ** 2)),
        "MAE": float(np.mean(np.abs(true_original - pred_original))),
    }


def _run_experiment(
    model_name: str,
    input_len: int,
    horizon: int,
    seeds: list[int],
    output_root: str,
    base_config: Config,
) -> dict:
    """运行单个模型的 5 轮实验，返回 {mean: ..., std: ...} 字典。"""
    metrics = []
    for seed in seeds:
        cfg = replace(
            base_config,
            model_name=model_name,
            input_len=input_len,
            horizon=horizon,
            seed=seed,
            output_root=output_root,
            ckpt_prefix="",
        )
        checkpoint_dir = Path(output_root) / "checkpoints"
        prefix = cfg.ckpt_prefix or model_name
        candidates = sorted(
            checkpoint_dir.glob(f"{prefix}_h{horizon}_s{seed}*.ckpt"),
            key=lambda path: path.stat().st_mtime,
        )
        reusable = []
        for path in candidates:
            manifest_path = path.with_suffix(".manifest.json")
            artifact_path = path.with_suffix(".artifacts.joblib")
            if not (manifest_path.exists() and artifact_path.exists()):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if manifest.get("config") == asdict(cfg):
                reusable.append(path)
        if reusable:
            ckpt = str(reusable[-1])
            print(f"  seed={seed}: reuse traced checkpoint {Path(ckpt).name} ...", end=" ", flush=True)
        else:
            print(f"  seed={seed}: training ...", end=" ", flush=True)
            ckpt = train(cfg)
        result = evaluate(cfg, ckpt)
        print(f"MSE={result['test/MSE']:.4f}, MAE={result['test/MAE']:.4f}")
        metrics.append(result)

    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    std = {k: float(np.std([m[k] for m in metrics], ddof=1)) for k in metrics[0]}
    return {"mean": avg, "std": std, "runs": metrics}


def run_all(
    input_len: int | None = None,
    output_root: str | None = None,
    include_hyperparams: bool = False,
    config: Config | None = None,
):
    """运行所有实验并保存汇总结果到 JSON。"""
    config = config or Config()
    if input_len is not None:
        config = replace(config, input_len=input_len)
    if output_root is not None:
        config = replace(config, output_root=output_root)
    input_len = config.input_len
    output_root = config.output_root
    results_dir = os.path.join(output_root, "results")
    os.makedirs(results_dir, exist_ok=True)

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
    ]

    summary = {}

    # ---- 1. 统计基线 ----
    print("=" * 60)
    print("Statistical baseline (seasonal naive)")
    print("=" * 60)
    for horizon in [90, 365]:
        result = seasonal_naive_baseline(config.data_path, input_len, horizon)
        summary[f"seasonal_naive_h{horizon}"] = {"mean": result, "std": {"MSE": 0.0, "MAE": 0.0}}
        print(f"  horizon={horizon}: MSE={result['MSE']:.4f}, MAE={result['MAE']:.4f}")
        ridge = ridge_direct_baseline(config.data_path, input_len, horizon)
        summary[f"ridge_direct_h{horizon}"] = {
            "mean": ridge,
            "std": {"MSE": 0.0, "MAE": 0.0},
        }
        print(f"  ridge horizon={horizon}: MSE={ridge['MSE']:.4f}, MAE={ridge['MAE']:.4f}")

    # ---- 2. 主实验（LSTM / Transformer / HCMRF） ----
    for model_name, horizon in main_experiments:
        print("=" * 60)
        print(f"Model: {model_name}, horizon={horizon}")
        print("=" * 60)
        summary[f"{model_name}_h{horizon}"] = _run_experiment(
            model_name, input_len, horizon, SEEDS, output_root, config
        )
        avg = summary[f"{model_name}_h{horizon}"]["mean"]
        std = summary[f"{model_name}_h{horizon}"]["std"]
        print(f"  → mean MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # ---- 3. 消融实验（仅 365d） ----
    ablation_summary = {}
    for model_name in ablation_models:
        print("=" * 60)
        print(f"Ablation: {model_name}, horizon=365")
        print("=" * 60)
        result = _run_experiment(model_name, input_len, 365, SEEDS, output_root, config)
        ablation_summary[f"{model_name}_h365"] = result
        # 同时写入主 summary
        summary[f"{model_name}_h365"] = result
        avg = result["mean"]
        std = result["std"]
        print(f"  → mean MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # ---- 4. 超参数消融（可选，仅 365d） ----
    hyperparam_summary = {}

    if not include_hyperparams:
        with open(os.path.join(results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(results_dir, "ablation_summary.json"), "w") as f:
            ablation_with_full = {"hcmrf_h365": summary["hcmrf_h365"]}
            ablation_with_full.update(ablation_summary)
            json.dump(ablation_with_full, f, indent=2)
        print(f"\nResults saved to {results_dir}")
        return summary

    # 4a. 压缩因子消融
    print("\n" + "=" * 60)
    print("Hyperparameter ablation: compress_factor")
    print("=" * 60)
    for cf in [2, 3, 4]:
        label = f"hcmrf_cf{cf}"
        print(f"  compress_factor={cf}:")
        metrics = []
        for seed in SEEDS:
            cfg = replace(config, model_name="hcmrf", input_len=input_len, horizon=365, seed=seed,
                          hcmrf_hcm_compress_factor=cf, ckpt_prefix=label, output_root=output_root)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics], ddof=1)) for k in metrics[0]}
        hyperparam_summary[f"compress_factor_{cf}"] = {"mean": avg, "std": std, "runs": metrics}
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
            cfg = replace(config, model_name="hcmrf", input_len=input_len, horizon=365, seed=seed,
                          hcmrf_drd_refine_kernel=rk, ckpt_prefix=label, output_root=output_root)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics], ddof=1)) for k in metrics[0]}
        hyperparam_summary[f"refine_kernel_{rk}"] = {"mean": avg, "std": std, "runs": metrics}
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
            cfg = replace(config, model_name="hcmrf", input_len=input_len, horizon=365, seed=seed,
                          hcmrf_drd_coarse_weeks=cw, ckpt_prefix=label, output_root=output_root)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics], ddof=1)) for k in metrics[0]}
        hyperparam_summary[f"coarse_weeks_{cw}"] = {"mean": avg, "std": std, "runs": metrics}
        print(f"    → MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # ---- 保存结果 ----
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMain results saved to {results_dir}/summary.json")

    with open(os.path.join(results_dir, "ablation_summary.json"), "w") as f:
        # 加入完整模型作为基线
        ablation_with_full = {"hcmrf_h365": summary["hcmrf_h365"]}
        ablation_with_full.update(ablation_summary)
        json.dump(ablation_with_full, f, indent=2)
    print(f"Ablation results saved to outputs/results/ablation_summary.json")

    with open(os.path.join(results_dir, "hyperparam_ablation.json"), "w") as f:
        json.dump(hyperparam_summary, f, indent=2)
    print(f"Hyperparameter ablation saved to outputs/results/hyperparam_ablation.json")


if __name__ == "__main__":
    run_all()
