"""训练 / 验证 / 测试 LightningModule 模块 — 编排训练逻辑。

ForecastSystem 是一个 LightningModule，接收预构建的 nn.Module 模型，
负责训练步骤、验证步骤、测试步骤和优化器配置。
它不定义模型结构，只编排训练逻辑。
"""
import lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class ForecastSystem(pl.LightningModule):
    """Lightning 训练系统 — 统一编排 train/val/test 逻辑。

    输入:
        model: 预构建的 nn.Module 实例（如 LSTMModel、TransformerModel、HCMRF）
        model_name: 模型名称字符串，用于 checkpoint 命名
        horizon: 预测 horizon（90 或 365），传递给 model.forward()
        learning_rate: 优化器学习率，默认 1e-3
        weight_decay: 优化器 L2 正则化系数，默认 1e-5

    输出:
        训练/验证/测试步骤通过 self.log() 记录 MSE 和 MAE 指标到 Lightning 日志系统。

    设计:
        - 损失函数: MSE
        - 指标: MSE + MAE（使用 torchmetrics）
        - 优化器: Adam(lr, weight_decay)
        - 所有模型通过统一的 forward(x, horizon) 接口调用
    """

    def __init__(self, model: nn.Module, model_name: str, horizon: int, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        # 保存超参数（排除 model，因为 nn.Module 不参与序列化）
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.MSELoss()
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "val/MSE": torchmetrics.MeanSquaredError(),
                "val/MAE": torchmetrics.MeanAbsoluteError(),
            }
        )
        self.test_metrics = torchmetrics.MetricCollection(
            {
                "test/MSE": torchmetrics.MeanSquaredError(),
                "test/MAE": torchmetrics.MeanAbsoluteError(),
            }
        )

    def forward(self, x):
        """前向传播 — 将输入和 horizon 传递给内部模型。

        Args:
            x: (B, input_len, n_features) 输入张量

        Returns:
            (B, horizon) 预测张量
        """
        return self.model(x, self.hparams.horizon)

    def training_step(self, batch, batch_idx):
        """训练步骤 — 计算 MSE 损失并记录到 Lightning 日志。

        Args:
            batch: (x, y) 对，x=(B,T,F), y=(B,horizon)
            batch_idx: 当前批次索引

        Returns:
            标量损失张量
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤 — 计算 MSE 和 MAE 指标并记录。

        Args:
            batch: (x, y) 对
            batch_idx: 当前批次索引
        """
        x, y = batch
        y_pred = self(x)
        self.val_metrics(y_pred, y)
        self.log_dict(self.val_metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """测试步骤 — 计算 MSE 和 MAE 指标并记录。

        Args:
            batch: (x, y) 对
            batch_idx: 当前批次索引
        """
        x, y = batch
        y_pred = self(x)
        self.test_metrics(y_pred, y)
        self.log_dict(self.test_metrics, on_epoch=True)

    def configure_optimizers(self):
        """配置优化器 — Adam，带学习率和 weight_decay。

        Returns:
            torch.optim.Adam 实例
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
