"""评估入口模块 — 加载 checkpoint 并计算测试集 MSE/MAE（日用电量 kWh 量纲）。

提供 evaluate() 函数：从 checkpoint 加载模型，在测试集上运行推理，
对预测值做逆归一化后计算 MSE 和 MAE 指标。

"""
import numpy as np
import torch
import joblib
from pathlib import Path

from .config import Config
from .system import ForecastSystem
from .train import build_model
from .features import add_features
from .windows import load_daily_frame


def evaluate(config: Config, ckpt_path: str) -> dict:
    """加载 checkpoint 并在最终留出区间计算 MSE/MAE（日用电量 kWh 量纲）。

    关键：对预测值和真实值都做逆归一化，返回原始量纲的指标。

    Args:
        config: 全局配置对象
        ckpt_path: 训练保存的 checkpoint 文件路径（.ckpt 格式）

    Returns:
        测试集指标字典，格式如 {"test/MSE": 12345.6, "test/MAE": 98.7}
        单位为日用电量（Global_active_power: kWh）
    """
    model = build_model(config)
    system = ForecastSystem.load_from_checkpoint(
        ckpt_path, model=model, model_name=config.model_name, horizon=config.horizon
    )
    system.eval()

    artifact_path = Path(ckpt_path).with_suffix(".artifacts.joblib")
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"缺少与 checkpoint 配套的实验产物 {artifact_path}；"
            "为避免使用错误的归一化器，拒绝评估旧检查点。"
        )
    artifact = joblib.load(artifact_path)
    scaler = artifact["scaler"]
    split = artifact["split"]
    frame = add_features(load_daily_frame(config.data_path))
    feature_values = frame.iloc[:, 1:]
    values = scaler.transform(feature_values)
    windows = [
        (
            values[start : start + split.input_len],
            values[start + split.input_len : start + split.input_len + split.horizon, 0],
        )
        for start in split.test_starts
    ]

    # 推理
    device = next(system.parameters()).device
    preds, targets = [], []
    with torch.no_grad():
        for x, y in windows:
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            y_pred = system(x_t).squeeze(0).cpu().numpy()
            preds.append(y_pred)
            targets.append(y)

    if not preds:
        return {"test/MSE": float('nan'), "test/MAE": float('nan')}

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # === 逆归一化 ===
    # scaler 是对整个特征矩阵做的，目标变量是第 0 列 (Global_active_power)
    # 构造 dummy 矩阵，仅第 0 列填目标值，其余填 0，调用 inverse_transform 后取第 0 列
    n_features = values.shape[1]

    def _inverse_target(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1)
        dummy = np.zeros((flat.size, n_features))
        dummy[:, 0] = flat
        inv = scaler.inverse_transform(dummy)[:, 0]
        return inv.reshape(arr.shape)

    preds_original = _inverse_target(preds)
    targets_original = _inverse_target(targets)

    # 计算原始量纲的 MSE 和 MAE
    mse = float(np.mean((targets_original - preds_original) ** 2))
    mae = float(np.mean(np.abs(targets_original - preds_original)))

    return {"test/MSE": mse, "test/MAE": mae}
