"""评估入口模块 — 加载 checkpoint 并计算测试集 MSE/MAE（原始kW量纲）。

提供 evaluate() 函数：从 checkpoint 加载模型，在测试集上运行推理，
对预测值做逆归一化后计算 MSE 和 MAE 指标。

对于 365d 预测任务，测试集数据不足（330天 < 90+365=455天），
采用跨边界评估：输入来自训练集最后90天，目标来自测试集真实值。
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from .config import Config
from .features import add_features
from .system import ForecastSystem
from .train import build_model


def evaluate(config: Config, ckpt_path: str) -> dict:
    """加载 checkpoint 并在测试集上计算 MSE/MAE（原始 kW 量纲）。

    关键：对预测值和真实值都做逆归一化，返回原始量纲的指标。

    Args:
        config: 全局配置对象
        ckpt_path: 训练保存的 checkpoint 文件路径（.ckpt 格式）

    Returns:
        测试集指标字典，格式如 {"test/MSE": 12345.6, "test/MAE": 98.7}
        单位为原始数据单位（Global_active_power: kW）
    """
    model = build_model(config)
    system = ForecastSystem.load_from_checkpoint(
        ckpt_path, model=model, model_name=config.model_name, horizon=config.horizon
    )
    system.eval()

    train_df = pd.read_csv(f"{config.data_path}/train.csv")
    train_df = add_features(train_df)
    test_df = pd.read_csv(f"{config.data_path}/test.csv")
    test_df = add_features(test_df)

    min_required = config.input_len + config.horizon

    # 归一化（scaler 仅在 train 上 fit，避免数据泄漏）
    scaler = MinMaxScaler()
    scaler.fit(train_df.iloc[:, 1:])  # 跳过 Date 列
    train_values = scaler.transform(train_df.iloc[:, 1:])
    test_values = scaler.transform(test_df.iloc[:, 1:])

    input_len = config.input_len
    horizon = config.horizon
    n_train = len(train_values)
    n_test = len(test_values)

    # 构建评估窗口 (x, y)，全部基于测试期数据，确保为留出集评估：
    #   - 90d: test (330天) >= input+horizon (180)，在 test 内滑动窗口
    #   - 365d: test (330天) < input+horizon (455)，采用"跨边界"评估——
    #     输入 = 训练集最后 90 天，目标 = test 真实值（2010 年）。
    #     这恰好对应模型部署场景：用最近 90 天预测未来。
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    if n_test >= input_len + horizon:
        for start in range(0, n_test - input_len - horizon + 1, config.step_size):
            x = test_values[start : start + input_len]
            y = test_values[start + input_len : start + input_len + horizon, 0]
            windows.append((x, y))
    else:
        x = train_values[-input_len:]
        y = test_values[:horizon, 0]
        windows.append((x, y))

    # 推理
    device = next(system.parameters()).device
    preds, targets = [], []
    with torch.no_grad():
        for x, y in windows:
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            y_pred = system(x_t).squeeze(0).cpu().numpy()
            n = min(len(y_pred), len(y))
            preds.append(y_pred[:n])
            targets.append(y[:n])

    if not preds:
        return {"test/MSE": float('nan'), "test/MAE": float('nan')}

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # === 逆归一化 ===
    # scaler 是对整个特征矩阵做的，目标变量是第 0 列 (Global_active_power)
    # 构造 dummy 矩阵，仅第 0 列填目标值，其余填 0，调用 inverse_transform 后取第 0 列
    n_features = scaler.n_features_in_

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
