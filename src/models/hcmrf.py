"""Horizon-Specialized Multi-Resolution Forecasting (HSMRF) 完整模型。

HSMRF 是项目第三部分（开放题）提出的自改进模型，包含三个创新组件：
  1. Multi-Scale Pooling: 根据 horizon 调整时序分辨率（90d 保持原始，365d 按比例压缩）
  2. Adaptive-Patch Transformer: horizon 自适应 patch 尺寸的 Transformer 编码器
  3. DRD (Dynamic Resolution Decoder): 粗→精解码器（90d 直接输出，365d 周级→上采样→精修）

创新动机：
  短期预测 (90d) 需要细粒度信息捕捉周模式和局部波动，
  长期预测 (365d) 需要粗粒度结构捕捉季节趋势，同时对噪声不敏感。
  HSMRF 让模型根据预测距离设计专门化的多分辨率架构。

设计约束：
  由于课程要求 90d 和 365d 模型分别训练，每个模型实例只看到一个 horizon，
  因此分辨率分支是架构设计决策而非可学习的 horizon conditioning。
"""
import torch.nn as nn

from .components.adaptive_patch import AdaptivePatchTransformer
from .components.drd import DynamicResolutionDecoder
from .components.hcm import HorizonConditioning


class HCMRF(nn.Module):
    """Horizon-Specialized Multi-Resolution Forecasting — 完整模型。

    输入:
        n_features: 输入特征维度（含工程特征后的总数）
        d_model: 隐藏层维度，默认 64
        n_heads: 多头注意力头数，默认 4
        n_layers: Transformer Encoder 层数，默认 2
        dropout: dropout 率，默认 0.1

    输出:
        forward(x, horizon): (B, horizon) 预测张量

    架构流程:
        (B, T, n_features) → Conv1D 编码器 → HCM → Adaptive-Patch Transformer
        → GlobalAvgPool → DRD → (B, horizon)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
        encoder_kernel_size: int = 7,
        hcm_compress_factor: int = 3,
        hcm_min_steps: int = 30,
        drd_coarse_weeks: int = 52,
        drd_refine_layers: int = 3,
        drd_refine_kernel: int = 7,
        patch_size_90d: int = 1,
        patch_size_365d: int = 3,
    ):
        super().__init__()
        # 共享 1D-CNN 编码器：提取局部特征
        self.encoder = nn.Conv1d(n_features, d_model, kernel_size=encoder_kernel_size, padding="same")
        # HCM：按预测跨度进行分辨率压缩
        self.hcm = HorizonConditioning(d_model, compress_factor=hcm_compress_factor, min_steps=hcm_min_steps)
        # Adaptive-Patch Transformer：horizon 自适应 patch 尺寸的编码器
        self.transformer = AdaptivePatchTransformer(d_model, n_heads, n_layers, d_ff=dim_feedforward, dropout=dropout)
        # DRD：粗→精解码器
        self.decoder = DynamicResolutionDecoder(d_model, coarse_weeks=drd_coarse_weeks, refine_layers=drd_refine_layers, refine_kernel=drd_refine_kernel)
        self.patch_size_90d = patch_size_90d
        self.patch_size_365d = patch_size_365d

    def forward(self, x, horizon: int):
        """前向传播。

        Args:
            x: (B, T, n_features) 输入张量
            horizon: 预测 horizon（90 或 365），决定分辨率和 patch 策略

        Returns:
            (B, horizon) 预测张量
        """
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)
        x = self.hcm(x, horizon)
        patch_size = self.patch_size_90d if horizon == 90 else self.patch_size_365d
        x = self.transformer(x, patch_size)
        x = x.mean(dim=1)  # GlobalAvgPool over time
        return self.decoder(x, horizon)
