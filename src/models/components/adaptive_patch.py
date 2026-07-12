"""Adaptive-Patch Transformer 子模块 — horizon 自适应 patch 尺寸的 Transformer 编码器。

核心功能：
  将相邻时间步拼接成 patch（patch_size × C → C*patch_size），
  通过 Linear 投影层映射回 d_model 维度，再输入 Transformer Encoder。

  90d: patch_size=1（细粒度注意力，捕捉局部波动）
  365d: patch_size=3（粗粒度注意力，减少序列长度，关注全局趋势）
"""
import math

import torch
import torch.nn as nn


class AdaptivePatchTransformer(nn.Module):
    """Adaptive-Patch Transformer — horizon 自适应 patch 尺寸。

    输入:
        d_model: Transformer 隐藏层维度
        n_heads: 多头注意力头数
        n_layers: Transformer Encoder 层数
        d_ff: 前馈网络隐藏层维度，默认 256
        dropout: dropout 率，默认 0.1

    输出:
        forward(x, patch_size): (B, T', d_model) 经过 patch + 投影 + Transformer 编码的张量
        T' = T（patch=1）或 T//patch_size（patch>1）
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        # 投影层: 将 C * patch_size 映射回 d_model
        # patch_size=3 时输入维度为 3*d_model，使用显式 Linear 避免 LazyLinear 序列化问题
        self.projection = nn.Linear(3 * d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    @staticmethod
    def _positional_encoding(length: int, d_model: int, device, dtype) -> torch.Tensor:
        """生成与 token 顺序对应的固定正弦/余弦位置编码。"""
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """前向传播 — patch 拼接 + 投影 + Transformer 编码。

        Args:
            x: (B, T, C) 输入特征张量
            patch_size: patch 尺寸（1 或 3）

        Returns:
            (B, T', d_model) 编码后的特征张量
        """
        if patch_size > 1:
            B, T, C = x.shape
            T_patched = T // patch_size
            # 将相邻 patch_size 个时间步拼接沿特征维度（C * patch_size）
            x = x[:, : T_patched * patch_size].reshape(B, T_patched, C * patch_size)
            # 投影回 d_model 维度
            x = self.projection(x)
        x = x + self._positional_encoding(x.size(1), x.size(2), x.device, x.dtype)
        # Transformer 编码
        return self.encoder(x)
