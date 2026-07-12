"""滑动窗口数据集模块 — 将连续时序切分为 (输入, 目标) 样本对。

提供 PowerDataset 类，基于 numpy 数组的滑动窗口切分，
支持由数据模块传入显式窗口起点。
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class PowerDataset(Dataset):
    """单序列滑动窗口数据集。

    将一个连续时序数组按固定窗口长度切分为训练/测试样本。
    每个样本由 (input_len, horizon) 一对张量构成。

    输入:
        data: (total_days, n_features) 的 numpy 数组
        input_len: 输入窗口长度（天）
        horizon: 预测 horizon（天）
        starts: 窗口起点索引

    输出 (通过 __getitem__):
        x: (input_len, n_features) 的特征张量
        y: (horizon,) 的目标张量，取自 data 的第 0 列（Global_active_power）
    """

    def __init__(self, data: np.ndarray, input_len: int, horizon: int, starts: list[int]):
        super().__init__()
        self.data = data
        self.input_len = input_len
        self.horizon = horizon
        self.starts = starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取第 idx 个样本。

        Args:
            idx: 样本索引（从 0 开始）

        Returns:
            x: (input_len, n_features) float32 张量
            y: (horizon,) float32 张量（第 0 列 = Global_active_power）
        """
        i = self.starts[idx]
        x = self.data[i : i + self.input_len]
        y = self.data[i + self.input_len : i + self.input_len + self.horizon, 0]  # target is column 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
