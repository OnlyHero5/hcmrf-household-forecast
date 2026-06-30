"""训练 / 验证 / 测试 LightningDataModule 模块 — 封装完整数据管线。

PowerDataModule 负责：
  1. 读取 train.csv / test.csv
  2. 特征工程（调用 features.add_features）
  3. 归一化（MinMaxScaler，仅在 train 上 fit）
  4. 切分 train / val（最后 20% 天做 val）
  5. 构建 DataLoader 供 PyTorch Lightning 使用
"""
import lightning as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from .dataset import PowerDataset
from .features import add_features


class PowerDataModule(pl.LightningDataModule):
    """Lightning 数据模块 — 管理家庭电力数据集的 train/val/test 加载器。

    输入:
        data_dir: 已处理数据目录（含 train.csv 和 test.csv）
        input_len: 输入窗口长度（天）
        horizon: 预测 horizon（天）
        batch_size: 批次大小
        step_size: 滑动窗口步长，默认 7

    输出 (通过 train/val/test_dataloader):
        每个 batch 返回 (x, y) 对:
          x: (B, input_len, n_features) float32 张量
          y: (B, horizon) float32 张量（Global_active_power 归一化值）

    注意:
        - 归一化 fit 仅在训练集上执行，避免数据泄漏
        - val 集 = 训练集的最后 20% 天数（时序不可 shuffle）
        - scaler 实例化后保存为 dm.scaler，可供 evaluate/visualize 逆变换
    """

    def __init__(self, data_dir: str, input_len: int, horizon: int, batch_size: int, step_size: int = 7):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        """加载数据 → 特征工程 → 归一化 → 构建 Dataset。

        由 Lightning Trainer 在 fit/test/predict 之前自动调用。

        验证集大小取 max(固定最小窗口, 20%)，确保长 horizon 实验有足够的验证样本。
        """
        train_df = pd.read_csv(f"{self.hparams.data_dir}/train.csv")
        test_df = pd.read_csv(f"{self.hparams.data_dir}/test.csv")

        # Step 1: 特征工程（添加 sin/cos 时间编码、滞后、滚动统计）
        train_df = add_features(train_df)
        test_df = add_features(test_df)

        # Step 2: 归一化 — MinMaxScaler，仅在训练集上 fit
        self.scaler = MinMaxScaler()
        train_values = self.scaler.fit_transform(train_df.iloc[:, 1:])  # 跳过 Date 列
        test_values = self.scaler.transform(test_df.iloc[:, 1:])

        # Step 3: 切分训练集 / 验证集（时序数据：最后 20% 天做 val）
        n_total = len(train_values)
        n_val_pct = int(n_total * 0.2)
        # 确保 val 集有足够窗口：需要 input_len + horizon 天
        n_val_needed = self.hparams.input_len + self.hparams.horizon
        n_val = max(n_val_pct, n_val_needed)
        # 但要保证训练集至少有 input_len + horizon 天
        n_train_min = n_val_needed
        n_val = min(n_val, n_total - n_train_min)

        val_values = train_values[-n_val:]
        train_values = train_values[:-n_val]

        # Step 4: 构建滑动窗口 Dataset（step_size > 1 减少相邻样本重叠）
        self.train_dataset = PowerDataset(train_values, self.hparams.input_len, self.hparams.horizon, self.hparams.step_size)
        self.val_dataset = PowerDataset(val_values, self.hparams.input_len, self.hparams.horizon, self.hparams.step_size)
        self.test_dataset = PowerDataset(test_values, self.hparams.input_len, self.hparams.horizon, self.hparams.step_size)

    def train_dataloader(self):
        """训练集 DataLoader，shuffle=True 打乱样本顺序。"""
        return DataLoader(self.train_dataset, self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        """验证集 DataLoader，时序数据不 shuffle。"""
        return DataLoader(self.val_dataset, self.hparams.batch_size)

    def test_dataloader(self):
        """测试集 DataLoader。"""
        return DataLoader(self.test_dataset, self.hparams.batch_size)
