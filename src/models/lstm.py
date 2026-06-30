"""双 LSTM 时序预测模型。

实现两层 LSTM + 线性头的网络结构。第一层提取时序特征，
最后一层时间步的输出经线性层映射到 horizon 维度的预测值。
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """两层 LSTM 模型 — 适用于短期和长期时序预测。

    输入:
        n_features: 输入特征维度（含工程特征后的总数）
        hidden_dim: LSTM 隐藏层维度，默认 128
        num_layers: LSTM 层数，默认 2
        horizon: 预测 horizon（90 或 365）

    输出:
        forward(x, horizon): (B, horizon) 预测张量
    """

    def __init__(self, n_features: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2, horizon: int = 90):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        # 线性头：隐藏层 → horizon
        self.head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, x: torch.Tensor, horizon: int = 90) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, T, n_features) 输入张量
            horizon: 预测 horizon（此模型不使用，保留为统一接口）

        Returns:
            (B, horizon) 预测张量，取 LSTM 最后一个时间步的输出
        """
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
