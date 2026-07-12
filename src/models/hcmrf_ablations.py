"""HCMRF 消融实验变体 — 用于验证各创新组件的有效性。

每个变体继承自 HCMRF，只覆写 forward() 或 __init__() 中需要修改的部分。

消融变体列表（仅对 365d 有意义，90d 路径不池化、patch=1）:
  A. HCMRF（完整基线）
  B. HCMRF_wo_MultiScale: 去掉多尺度池化（365d 不压缩，保持 90 步）
  C. HCMRF_wo_Patch:      去掉 Adaptive-Patch（365d 固定 patch=1）
  D. HCMRF_wo_DRD:        去掉 DRD（365d 用直接 Dense 输出）
  E. HCMRF_wo_Shared:     独立编码器（90d 和 365d 使用不同参数）
"""
import torch.nn as nn
import torch.nn.functional as F

from .hcmrf import HCMRF


class HCMRF_wo_MultiScale(HCMRF):
    """消融 B: 去掉多尺度池化 — 365d 不压缩，保持 90 步。

    用于验证多尺度池化（分辨率压缩）是否对预测有帮助。
    仅对 365d 有意义；90d 路径本就不池化，结果与完整模型相同。
    """

    def forward(self, x, horizon):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        # 跳过 self.hcm（不做池化压缩）
        patch_size = 1 if horizon == 90 else 3
        x = self.transformer(x, patch_size)
        x = x.mean(dim=1)
        return self.decoder(x, horizon)


class HCMRF_wo_Patch(HCMRF):
    """消融 C: 去掉 Adaptive-Patch — 保留多尺度池化，固定 patch=1。

    用于验证动态 patch 是否比单纯 pooling 压缩更好。
    仅对 365d 有意义；90d 路径本就用 patch=1。
    """

    def forward(self, x, horizon):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        x = self.hcm(x, horizon)
        x = self.transformer(x, patch_size=1)  # 固定 patch=1
        x = x.mean(dim=1)
        return self.decoder(x, horizon)


class HCMRF_wo_DRD(HCMRF):
    """消融 D: 去掉 DRD — 90d 用原 DRD，365d 用直接 Dense(365) 输出。

    用于验证粗→精解码是否优于直接多步输出。
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, dim_feedforward: int = 256, encoder_kernel_size: int = 7,
                 hcm_compress_factor: int = 3, hcm_min_steps: int = 30,
                 drd_coarse_weeks: int = 52, drd_refine_layers: int = 3, drd_refine_kernel: int = 7,
                 patch_size_90d: int = 1, patch_size_365d: int = 3):
        super().__init__(n_features, d_model, n_heads, n_layers, dropout, dim_feedforward,
                         encoder_kernel_size, hcm_compress_factor, hcm_min_steps,
                         drd_coarse_weeks, drd_refine_layers, drd_refine_kernel,
                         patch_size_90d, patch_size_365d)
        self.direct_head = nn.Linear(d_model, 365)

    def forward(self, x, horizon):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        x = self.hcm(x, horizon)
        patch_size = 1 if horizon == 90 else 3
        x = self.transformer(x, patch_size)
        x = x.mean(dim=1)
        if horizon == 90:
            return self.decoder(x, horizon)
        return self.direct_head(x)


class HCMRF_wo_Shared(HCMRF):
    """消融 E: 独立编码器 — 90d 和 365d 使用不同参数（参数翻倍）。

    用于验证共享编码器是否有效。
    """

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1, dim_feedforward: int = 256, encoder_kernel_size: int = 7,
                 hcm_compress_factor: int = 3, hcm_min_steps: int = 30,
                 drd_coarse_weeks: int = 52, drd_refine_layers: int = 3, drd_refine_kernel: int = 7,
                 patch_size_90d: int = 1, patch_size_365d: int = 3):
        super().__init__(n_features, d_model, n_heads, n_layers, dropout, dim_feedforward,
                         encoder_kernel_size, hcm_compress_factor, hcm_min_steps,
                         drd_coarse_weeks, drd_refine_layers, drd_refine_kernel,
                         patch_size_90d, patch_size_365d)
        self.encoder_365 = nn.Conv1d(n_features, d_model, kernel_size=encoder_kernel_size, padding="same")

    def forward(self, x, horizon):
        enc = self.encoder if horizon == 90 else self.encoder_365
        x = enc(x.transpose(1, 2)).transpose(1, 2)
        x = self.hcm(x, horizon)
        patch_size = 1 if horizon == 90 else 3
        x = self.transformer(x, patch_size)
        x = x.mean(dim=1)
        return self.decoder(x, horizon)
