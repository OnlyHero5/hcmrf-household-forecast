"""全局配置模块 — 统一管理所有超参数。

支持两种配置方式：
  1. dataclass 直接创建（代码中直接使用）
  2. YAML 文件加载（通过 yaml_to_config 函数）

所有超参数在 __init__ 形参中声明，支持类型注解和默认值。
"""
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class Config:
    """全局配置类 — 所有超参数集中在一个入口。

    Attributes:
        # 数据配置
        data_path: 已处理数据的目录路径，默认 "data/processed"
        input_len: 输入窗口长度（天数），默认 90 天
        short_horizon: 短期预测 horizon（天数），默认 90
        long_horizon: 长期预测 horizon（天数），默认 365
        step_size: 滑动窗口采样步长；步长>1 减少相邻样本重叠，默认 7

        # 训练配置
        batch_size: 训练批次大小，默认 32
        max_epochs: 最大训练轮数，默认 100
        patience: 早停耐心值（验证集无改善的轮数），默认 10
        learning_rate: 优化器学习率，默认 1e-3
        weight_decay: Adam 优化器的 L2 正则化系数，默认 1e-5
        seed: 随机种子（用于可复现性），默认 42

        # 模型选择
        model_name: 模型名称，可选值: lstm / transformer / hcmrf / hcmrf_wo_MultiScale / hcmrf_wo_Patch / hcmrf_wo_DRD / hcmrf_wo_Shared
        horizon: 预测 horizon，可选 90（短期）或 365（长期）

        # LSTM 超参数
        lstm_hidden_dim: LSTM 隐藏层维度，默认 128
        lstm_num_layers: LSTM 层数，默认 2
        lstm_dropout: LSTM dropout 率，默认 0.2

        # Transformer 超参数
        transformer_d_model: Transformer 嵌入维度，默认 128
        transformer_n_heads: 多头注意力头数，默认 4
        transformer_n_layers: Encoder 层数，默认 2
        transformer_dim_feedforward: FFN 隐藏层维度，默认 256
        transformer_dropout: Transformer dropout 率，默认 0.1

        # HCMRF 超参数
        hcmrf_d_model: HCMRF 隐藏层维度，默认 64
        hcmrf_n_heads: HCMRF Transformer 头数，默认 4
        hcmrf_n_layers: HCMRF Transformer 层数，默认 2
        hcmrf_dim_feedforward: HCMRF FFN 隐藏层维度，默认 256
        hcmrf_dropout: HCMRF dropout 率，默认 0.1
        hcmrf_encoder_kernel_size: Conv1D 编码器核大小，默认 7
        hcmrf_drd_coarse_weeks: 365d 粗预测周数，默认 52
        hcmrf_drd_refine_layers: DRD 精修 Conv1D 层数，默认 3
        hcmrf_drd_refine_kernel: DRD 精修 Conv1D 核大小，默认 7
        hcmrf_hcm_compress_factor: HCM 365d 压缩因子，默认 3
        hcmrf_hcm_min_steps: HCM 最小时间步数，默认 30
        hcmrf_patch_size_90d: 90d patch 尺寸，默认 1
        hcmrf_patch_size_365d: 365d patch 尺寸，默认 3
    """

    # ===== Data =====
    data_path: str = "data/processed"
    input_len: int = 90
    short_horizon: int = 90
    long_horizon: int = 365
    step_size: int = 7                  # 滑动窗口步长；7 减少重叠，1 最大化样本数

    # ===== Training =====
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42

    # ===== Model selection =====
    model_name: str = "lstm"            # lstm / transformer / hcmrf / hcmrf_wo_MultiScale / ...
    ckpt_prefix: str = ""               # checkpoint 命名前缀；空字符串时用 model_name（超参数消融用）
    horizon: int = 90                   # 90 or 365

    # ===== LSTM hyperparameters =====
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2

    # ===== Transformer hyperparameters =====
    transformer_d_model: int = 128
    transformer_n_heads: int = 4
    transformer_n_layers: int = 2
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1

    # ===== HCMRF hyperparameters =====
    hcmrf_d_model: int = 64
    hcmrf_n_heads: int = 4
    hcmrf_n_layers: int = 2
    hcmrf_dim_feedforward: int = 256
    hcmrf_dropout: float = 0.1
    hcmrf_encoder_kernel_size: int = 7
    hcmrf_drd_coarse_weeks: int = 52
    hcmrf_drd_refine_layers: int = 3
    hcmrf_drd_refine_kernel: int = 7
    hcmrf_hcm_compress_factor: int = 3
    hcmrf_hcm_min_steps: int = 30
    hcmrf_patch_size_90d: int = 1
    hcmrf_patch_size_365d: int = 3


def load_config(yaml_path: str) -> Config:
    """从 YAML 配置文件加载 Config。

    Args:
        yaml_path: YAML 配置文件路径

    Returns:
        从 YAML 文件填充的 Config 对象

    Example YAML structure:
        data:
          data_path: "data/processed"
          input_len: 90
        training:
          batch_size: 32
          learning_rate: 0.001
        lstm:
          hidden_dim: 128
        transformer:
          d_model: 128
        hcmrf:
          d_model: 64
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    # Data section
    if "data" in raw:
        data = raw["data"]
        for k in ["data_path", "input_len", "short_horizon", "long_horizon", "step_size"]:
            if k in data:
                setattr(cfg, k, data[k])

    # Training section
    if "training" in raw:
        train = raw["training"]
        for k in ["batch_size", "max_epochs", "patience", "learning_rate", "weight_decay", "seed"]:
            if k in train:
                setattr(cfg, k, train[k])

    # LSTM section
    if "lstm" in raw:
        lstm = raw["lstm"]
        for k in ["hidden_dim", "num_layers", "dropout"]:
            if k in lstm:
                setattr(cfg, f"lstm_{k}", lstm[k])

    # Transformer section
    if "transformer" in raw:
        t = raw["transformer"]
        for k in ["d_model", "n_heads", "n_layers", "dim_feedforward", "dropout"]:
            if k in t:
                setattr(cfg, f"transformer_{k}", t[k])

    # HCMRF section
    if "hcmrf" in raw:
        h = raw["hcmrf"]
        for k in ["d_model", "n_heads", "n_layers", "dim_feedforward", "dropout",
                  "encoder_kernel_size", "drd_coarse_weeks", "drd_refine_layers",
                  "drd_refine_kernel", "hcm_compress_factor", "hcm_min_steps",
                  "patch_size_90d", "patch_size_365d"]:
            if k in h:
                setattr(cfg, f"hcmrf_{k}", h[k])

    return cfg
