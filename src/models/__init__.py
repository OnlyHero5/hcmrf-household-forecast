"""模型定义包。

包含以下模型：
  - lstm: LSTMModel — 两层 LSTM + 线性头
  - transformer: TransformerModel — Transformer 编码器 + 全局平均池化
  - hcmrf: HCMRF — Horizon-Specialized Multi-Resolution Forecasting
  - hcmrf_ablations: 4 个消融变体（HCMRF_wo_MultiScale/Patch/DRD/Shared）
  - components: HCMRF 子模块（HCM / AdaptivePatch / DRD）
"""
