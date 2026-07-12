"""Transformer 编码器时序预测模型。

使用正弦/余弦位置编码 + N 层 Transformer Encoder + Global Average Pooling
来提取时序特征，最后通过线性头输出 horizon 维度的预测值。
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """固定的正弦/余弦位置编码。

    输入:
        d_model: 位置编码维度
        max_len: 最大序列长度，默认 5000

    输出:
        (B, T, d_model) 添加了位置编码的张量
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)，不参与训练

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """在输入张量上添加位置编码。

        Args:
            x: (B, T, d_model) 输入张量

        Returns:
            (B, T, d_model) 添加了位置编码的输出
        """
        return x + self.pe[: x.size(1), :]


class TransformerModel(nn.Module):
    """Transformer 编码器模型 — 用于短期和长期时序预测。

    输入:
        n_features: 输入特征维度（含工程特征后的总数）
        d_model: Transformer 嵌入维度，默认 128
        n_heads: 多头注意力头数，默认 4
        n_layers: Transformer Encoder 层数，默认 2
        horizon: 预测 horizon（90 或 365）

    输出:
        forward(x, horizon): (B, horizon) 预测张量

    架构:
        Linear Embedding → PositionalEncoding → TransformerEncoder × n_layers
        → GlobalAvgPool → Linear(64) → Linear(horizon)
    """

    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1, horizon: int = 90):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        # 线性头：d_model → 64 → horizon
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, horizon))

    def forward(self, x: torch.Tensor, horizon: int = 90) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, T, n_features) 输入张量
            horizon: 预测 horizon（此模型不使用，保留为统一接口）

        Returns:
            (B, horizon) 预测张量，对所有时间步做全局平均池化后输出
        """
        x = self.pos(self.embed(x))
        x = self.encoder(x)
        return self.head(x.mean(dim=1))
