"""HCM 子模块 — 多尺度池化模块（Multi-Scale Pooling）。

核心功能：
  90d: 保持原始时序分辨率（T' = T = 90）
  365d: 使用 AdaptiveAvgPool1d 按压缩因子降低分辨率，捕捉季节趋势

设计约束：
  由于 90d 和 365d 模型分别训练（课程硬性要求），每个模型实例只看到一个 horizon，
  因此分辨率分支是硬编码的 architectural design choice。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonConditioning(nn.Module):
    """多尺度池化模块 — 根据 horizon 调整时序分辨率。

    输入:
        d_model: 特征维度（通道数），仅用于文档（本模块不使用）
        compress_factor: 365d 压缩因子，默认 3
        min_steps: 压缩后最小时间步数，默认 30

    输出:
        forward(x, horizon): (B, T', d_model) 经过分辨率调整的张量
        T' = T（90d）或 max(T // compress_factor, min_steps)（365d）
    """

    def __init__(self, d_model: int, compress_factor: int = 3, min_steps: int = 30):
        super().__init__()
        self.compress_factor = compress_factor
        self.min_steps = min_steps

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """前向传播 — 根据 horizon 做分辨率压缩。

        Args:
            x: (B, T, C) 特征张量
            horizon: 预测 horizon（90 或 365）

        Returns:
            (B, T', C) 经过分辨率调整的张量
        """
        if horizon == 90:
            # 90d: 不压缩时序分辨率
            return x
        else:
            T = x.size(1)
            out_T = max(T // self.compress_factor, self.min_steps)
            return F.adaptive_avg_pool1d(x.transpose(1, 2), out_T).transpose(1, 2)
