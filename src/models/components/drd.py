"""DRD 子模块 — 动态分辨率解码器（Dynamic Resolution Decoder）。

核心功能：
  90d: 直接输出（GlobalAvgPool → Dense(128) → Dense(90)）
  365d: 粗→精解码（周级粗预测 → 线性插值上采样 → 多层 Conv1D 精修）

  精修阶段使用 3 层 Conv1D(k=7) 堆叠，感受野 ≈ 19 个插值点，
  能有效修正线性插值引入的平滑伪影。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicResolutionDecoder(nn.Module):
    """动态分辨率解码器 — 粗→精解码策略。

    输入:
        d_model: 输入特征维度

    输出:
        forward(x, horizon): (B, horizon) 预测张量
    """

    def __init__(self, d_model: int, coarse_weeks: int = 52, refine_layers: int = 3, refine_kernel: int = 7):
        super().__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU())
        # 90d 直接输出头
        self.head_90 = nn.Linear(128, 90)
        # 365d 粗预测头（周级粒度）
        self.head_coarse = nn.Linear(128, coarse_weeks)

        # 多层 Conv1D 精修（增强感受野，修正插值伪影）
        # 可配置精修层数和核大小
        layers = []
        for i in range(refine_layers - 1):
            in_ch = 1 if i == 0 else 16
            layers.extend([
                nn.Conv1d(in_ch, 16, kernel_size=refine_kernel, padding="same"),
                nn.ReLU(),
            ])
        # 最后一层：16 → 1（无 ReLU）
        layers.append(nn.Conv1d(16, 1, kernel_size=refine_kernel, padding="same"))
        self.refine = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """前向传播 — 根据 horizon 选择解码路径。

        Args:
            x: (B, d_model) 输入特征（GlobalAvgPool 后的全局向量）
            horizon: 预测 horizon（90 或 365）

        Returns:
            (B, horizon) 预测张量
        """
        h = self.shared(x)
        if horizon == 90:
            # 90d: 直接输出
            return self.head_90(h)

        # 365d: 粗→精解码
        # Step 1: 周级粗预测（52 周）
        coarse = self.head_coarse(h)  # (B, 52)
        # Step 2: 线性插值上采样到 365 天
        coarse_365 = F.interpolate(coarse.unsqueeze(1), size=365, mode="linear").squeeze(1)
        # Step 3: 多层 Conv1D 精修（修正插值伪影）
        refined = self.refine(coarse_365.unsqueeze(1)).squeeze(1)
        return refined
