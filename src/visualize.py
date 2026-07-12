"""可视化模块 — 绘制预测曲线和消融实验柱状图（日用电量 kWh 量纲）。

提供以下函数：
  - plot_model_comparison: 多种模型预测曲线 vs 真实值对比图（90d/365d分开）
  - plot_ablation: 消融变体 MSE/MAE 柱状图（仅 365d，带误差棒）
  - plot_hyperparam_ablation: 超参数消融柱状图（仅 365d）
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import torch
import joblib

from .config import Config
from .system import ForecastSystem
from .train import build_model
from .features import add_features
from .windows import PreparedWindows, load_daily_frame, prepare_windows


def _get_eval_data(config: Config) -> PreparedWindows:
    return prepare_windows(config.data_path, config.input_len, config.horizon, config.step_size)


def _predict_one_sample(config: Config, ckpt_path: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """加载 checkpoint，对评估样本推理，返回 kWh 量纲的(预测, 真实)数组。"""
    model = build_model(config)
    system = ForecastSystem.load_from_checkpoint(
        ckpt_path, model=model, model_name=config.model_name, horizon=config.horizon
    )
    system.eval()
    device = next(system.parameters()).device

    artifact_path = Path(ckpt_path).with_suffix(".artifacts.joblib")
    if not artifact_path.exists():
        raise FileNotFoundError(f"缺少实验产物：{artifact_path}")
    artifact = joblib.load(artifact_path)
    split = artifact["split"]
    scaler = artifact["scaler"]
    frame = add_features(load_daily_frame(config.data_path))
    values = scaler.transform(frame.iloc[:, 1:])
    start = split.test_starts[0]
    x = values[start : start + split.input_len]
    y = values[start + split.input_len : start + split.input_len + split.horizon, 0]

    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        y_pred = system(x_t).squeeze(0).cpu().numpy()

    n_features = values.shape[1]
    def _inverse_target(arr):
        dummy = np.zeros((len(arr), n_features))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred_orig = _inverse_target(y_pred)
    y_orig = _inverse_target(y)
    target_dates = pd.DatetimeIndex(
        pd.to_datetime(frame["Date"]).iloc[
            start + split.input_len : start + split.input_len + split.horizon
        ]
    )
    return y_pred_orig, y_orig, target_dates


def plot_model_comparison(ckpt_paths: dict[str, str], horizon: int, save_path: str):
    """绘制多种模型预测曲线 vs 真实值对比图。"""
    fig, ax = plt.subplots(figsize=(14, 5))

    gt_shown = False
    colors = {"LSTM": "tab:blue", "Transformer": "tab:orange", "多分辨率模型": "tab:green"}

    for name, ckpt in ckpt_paths.items():
        model_key = "hcmrf" if name == "多分辨率模型" else name.lower()
        cfg = Config(model_name=model_key, horizon=horizon)
        y_pred, y_true, dates = _predict_one_sample(cfg, ckpt)

        color = colors.get(name, None)
        ax.plot(dates, y_pred, label=name, alpha=0.9, color=color, linewidth=1.5)

        if not gt_shown:
            ax.plot(dates, y_true, label="真实值", color="black", linewidth=2.0, linestyle="--")
            gt_shown = True

    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(f"电力消耗预测对比（{horizon} 天）", fontsize=14)
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("日用电量 (kWh)", fontsize=12)
    ax.grid(True, alpha=0.3)
    locator = mdates.MonthLocator(interval=1 if horizon == 90 else 2)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ablation(summary_path: str, output_dir: str):
    """绘制消融实验柱状图（仅 365d）。

    消融变体：完整 HCMRF / -w/o MultiScale / -w/o Patch / -w/o DRD / -w/o Shared
    """
    with open(summary_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    name_map = {
        "hcmrf": "完整模型",
        "hcmrf_wo_MultiScale": "去掉多尺度池化",
        "hcmrf_wo_Patch": "去掉自适应分块",
        "hcmrf_wo_DRD": "去掉动态解码器",
        "hcmrf_wo_Shared": "去掉共享编码器",
    }
    variants = ["hcmrf", "hcmrf_wo_MultiScale", "hcmrf_wo_Patch", "hcmrf_wo_DRD", "hcmrf_wo_Shared"]

    records = []
    for v in variants:
        key = f"{v}_h365"
        if key in data:
            records.append({
                "variant": name_map[v],
                "MSE": data[key]["mean"]["test/MSE"],
                "MSE_std": data[key]["std"]["test/MSE"],
                "MAE": data[key]["mean"]["test/MAE"],
                "MAE_std": data[key]["std"]["test/MAE"],
            })

    df = pd.DataFrame(records)

    # MSE 图
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar(x, df["MSE"], yerr=df["MSE_std"], capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("MSE (kWh$^2$)", fontsize=12)
    ax.set_title("消融实验——MSE（365 天）", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_mse_h365.png"), dpi=150)
    plt.close(fig)

    # MAE 图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, df["MAE"], yerr=df["MAE_std"], capsize=4, color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("MAE (kWh)", fontsize=12)
    ax.set_title("消融实验——MAE（365 天）", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_mae_h365.png"), dpi=150)
    plt.close(fig)

    print(f"Saved ablation plots to {output_dir}")


def plot_hyperparam_ablation(summary_path: str, output_dir: str):
    """绘制超参数消融柱状图（仅 365d）。"""
    with open(summary_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # 三个子图：compress_factor, refine_kernel, coarse_weeks
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 压缩因子
    cf_keys = [k for k in data if k.startswith("compress_factor_")]
    cf_keys.sort(key=lambda k: int(k.split("_")[-1]))
    if cf_keys:
        labels = [k.split("_")[-1] for k in cf_keys]
        mse_vals = [data[k]["mean"]["test/MSE"] / 1000 for k in cf_keys]
        mse_stds = [data[k]["std"]["test/MSE"] / 1000 for k in cf_keys]
        axes[0].bar(range(len(labels)), mse_vals, yerr=mse_stds, capsize=4, color="steelblue", alpha=0.8)
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels)
        axes[0].set_xlabel("压缩因子")
        axes[0].set_ylabel("MSE ($\\times 10^3$ kW$^2$)")
        axes[0].set_title("池化压缩因子")
        axes[0].grid(True, alpha=0.3, axis="y")

    # 精修 kernel
    rk_keys = [k for k in data if k.startswith("refine_kernel_")]
    rk_keys.sort(key=lambda k: int(k.split("_")[-1]))
    if rk_keys:
        labels = [k.split("_")[-1] for k in rk_keys]
        mse_vals = [data[k]["mean"]["test/MSE"] / 1000 for k in rk_keys]
        mse_stds = [data[k]["std"]["test/MSE"] / 1000 for k in rk_keys]
        axes[1].bar(range(len(labels)), mse_vals, yerr=mse_stds, capsize=4, color="coral", alpha=0.8)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels)
        axes[1].set_xlabel("卷积核大小")
        axes[1].set_ylabel("MSE ($\\times 10^3$ kW$^2$)")
        axes[1].set_title("解码器卷积核")
        axes[1].grid(True, alpha=0.3, axis="y")

    # 粗预测周数
    cw_keys = [k for k in data if k.startswith("coarse_weeks_")]
    cw_keys.sort(key=lambda k: int(k.split("_")[-1]))
    if cw_keys:
        labels = [k.split("_")[-1] for k in cw_keys]
        mse_vals = [data[k]["mean"]["test/MSE"] / 1000 for k in cw_keys]
        mse_stds = [data[k]["std"]["test/MSE"] / 1000 for k in cw_keys]
        axes[2].bar(range(len(labels)), mse_vals, yerr=mse_stds, capsize=4, color="seagreen", alpha=0.8)
        axes[2].set_xticks(range(len(labels)))
        axes[2].set_xticklabels(labels)
        axes[2].set_xlabel("粗预测周数")
        axes[2].set_ylabel("MSE ($\\times 10^3$ kW$^2$)")
        axes[2].set_title("粗预测周数")
        axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("超参数消融（365 天）", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hyperparam_ablation.png"), dpi=150)
    plt.close(fig)
    print(f"Saved hyperparameter ablation plot to {output_dir}")


def main():
    """生成所有可视化图表。"""
    import glob

    output_root = "outputs/revised"
    os.makedirs(f"{output_root}/figures", exist_ok=True)

    def find_ckpt(model_name, horizon):
        matches = glob.glob(f"{output_root}/checkpoints/{model_name}_h{horizon}_s42*.ckpt")
        return max(matches, key=os.path.getmtime) if matches else None

    # 90d 对比图
    ckpts_90 = {
        "LSTM": find_ckpt("lstm", 90),
        "Transformer": find_ckpt("transformer", 90),
        "多分辨率模型": find_ckpt("hcmrf", 90),
    }
    ckpts_90 = {k: v for k, v in ckpts_90.items() if v}
    if ckpts_90:
        plot_model_comparison(ckpts_90, 90, f"{output_root}/figures/comparison_90d.png")

    # 365d 对比图
    ckpts_365 = {
        "LSTM": find_ckpt("lstm", 365),
        "Transformer": find_ckpt("transformer", 365),
        "多分辨率模型": find_ckpt("hcmrf", 365),
    }
    ckpts_365 = {k: v for k, v in ckpts_365.items() if v}
    if ckpts_365:
        plot_model_comparison(ckpts_365, 365, f"{output_root}/figures/comparison_365d.png")

    # 消融实验图（仅 365d）
    plot_ablation(f"{output_root}/results/ablation_summary.json", f"{output_root}/figures")

    # 超参数消融图
    hyperparam_path = f"{output_root}/results/hyperparam_ablation.json"
    if os.path.exists(hyperparam_path):
        plot_hyperparam_ablation(hyperparam_path, f"{output_root}/figures")

    print("\nAll visualizations generated.")


if __name__ == "__main__":
    main()
