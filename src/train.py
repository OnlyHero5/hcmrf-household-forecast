"""训练入口模块 — 构建模型并执行单次训练。

提供 build_model() 工厂函数和 train() 入口。
build_model() 根据 Config.model_name 选择对应的模型类；
train() 使用 PyTorch Lightning 执行训练，返回最优 checkpoint 路径。
"""
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from .config import Config
from .datamodule import PowerDataModule
from .models.hcmrf import HCMRF
from .models.hcmrf_ablations import (
    HCMRF_wo_DRD,
    HCMRF_wo_MultiScale,
    HCMRF_wo_Patch,
    HCMRF_wo_Shared,
)
from .models.lstm import LSTMModel
from .models.transformer import TransformerModel
from .system import ForecastSystem


def build_model(config: Config):
    """模型工厂函数 — 根据配置创建对应的 nn.Module 实例。

    Args:
        config: 全局配置对象，包含 model_name、horizon 等参数

    Returns:
        对应的 nn.Module 实例（如 LSTMModel、TransformerModel、HCMRF 等）
    """
    model_name = config.model_name
    # 特征数：add_features() 后 13 列原始特征 + 11 列工程特征 = 24
    n_features = 24

    if model_name == "lstm":
        return LSTMModel(
            n_features,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            horizon=config.horizon,
        )
    elif model_name == "transformer":
        return TransformerModel(
            n_features,
            d_model=config.transformer_d_model,
            n_heads=config.transformer_n_heads,
            n_layers=config.transformer_n_layers,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=config.transformer_dropout,
            horizon=config.horizon,
        )
    # HCMRF 超参数公共部分
    hcmrf_kwargs = dict(
        d_model=config.hcmrf_d_model,
        n_heads=config.hcmrf_n_heads,
        n_layers=config.hcmrf_n_layers,
        dropout=config.hcmrf_dropout,
        dim_feedforward=config.hcmrf_dim_feedforward,
        encoder_kernel_size=config.hcmrf_encoder_kernel_size,
        hcm_compress_factor=config.hcmrf_hcm_compress_factor,
        hcm_min_steps=config.hcmrf_hcm_min_steps,
        drd_coarse_weeks=config.hcmrf_drd_coarse_weeks,
        drd_refine_layers=config.hcmrf_drd_refine_layers,
        drd_refine_kernel=config.hcmrf_drd_refine_kernel,
    )

    if model_name == "hcmrf":
        return HCMRF(n_features, **hcmrf_kwargs)
    elif model_name == "hcmrf_wo_MultiScale":
        return HCMRF_wo_MultiScale(n_features, **hcmrf_kwargs)
    elif model_name == "hcmrf_wo_Patch":
        return HCMRF_wo_Patch(n_features, **hcmrf_kwargs)
    elif model_name == "hcmrf_wo_DRD":
        return HCMRF_wo_DRD(n_features, **hcmrf_kwargs)
    elif model_name == "hcmrf_wo_Shared":
        return HCMRF_wo_Shared(n_features, **hcmrf_kwargs)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def train(config: Config) -> str:
    """训练单个模型并返回最优 checkpoint 路径。

    使用 PyTorch Lightning 训练，支持早停、checkpoint 和学习率监控。

    Args:
        config: 全局配置对象

    Returns:
        最优 checkpoint 的文件路径（以 val/MSE 为监控指标）
    """
    pl.seed_everything(config.seed, workers=True)

    model = build_model(config)
    dm = PowerDataModule(config.data_path, config.input_len, config.horizon, config.batch_size, config.step_size)
    system = ForecastSystem(model, config.model_name, config.horizon, config.learning_rate, config.weight_decay)

    callbacks = [
        # 早停：验证集 MSE 连续 patience 轮无改善则停止
        EarlyStopping(monitor="val/MSE", patience=config.patience, mode="min"),
        # Checkpoint：保存验证集 MSE 最优的模型权重
        ModelCheckpoint(
            dirpath="outputs/checkpoints",
            filename=f"{config.ckpt_prefix or config.model_name}_h{config.horizon}_s{config.seed}",
            monitor="val/MSE",
            mode="min",
        ),
        # 学习率监控：记录每个 epoch 的学习率到日志
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(max_epochs=config.max_epochs, callbacks=callbacks, enable_progress_bar=True)

    trainer.fit(system, dm)
    return trainer.checkpoint_callback.best_model_path
