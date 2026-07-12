"""训练 / 验证 / 测试 LightningDataModule 模块 — 封装完整数据管线。

PowerDataModule 负责：
  1. 读取完整日级序列
  2. 特征工程（调用 features.add_features）
  3. 归一化（MinMaxScaler，仅在训练开发窗口上 fit）
  4. 按 horizon 构建 train / val / test 窗口
  5. 构建 DataLoader 供 PyTorch Lightning 使用
"""
import lightning as pl
from torch.utils.data import DataLoader

from .dataset import PowerDataset
from .windows import prepare_windows


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
        - 归一化 fit 仅在训练开发窗口上执行，避免测试泄漏
        - val 集 = 测试目标期之前的开发窗口末段（时序不可 shuffle）
        - scaler 实例化后保存为 dm.scaler，可供 evaluate/visualize 逆变换
    """

    def __init__(self, data_dir: str, input_len: int, horizon: int, batch_size: int, step_size: int = 7):
        super().__init__()
        self.data_dir = data_dir
        self.input_len = input_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.step_size = step_size
        self.save_hyperparameters()

    def setup(self, stage=None):
        """加载数据 → 特征工程 → 归一化 → 构建 Dataset。

        由 Lightning Trainer 在 fit/test/predict 之前自动调用。

        输入长度由配置显式指定，测试目标期固定为完整的最后 horizon 天。
        """
        prepared = prepare_windows(self.data_dir, self.input_len, self.horizon, self.step_size)
        self.prepared = prepared
        self.scaler = prepared.scaler
        split = prepared.split
        self.split = split

        self.train_dataset = PowerDataset(prepared.values, split.input_len, split.horizon, split.train_starts)
        self.val_dataset = PowerDataset(prepared.values, split.input_len, split.horizon, split.val_starts)
        self.test_dataset = PowerDataset(prepared.values, split.input_len, split.horizon, split.test_starts)

    def train_dataloader(self):
        """训练集 DataLoader，shuffle=True 打乱样本顺序。"""
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self):
        """验证集 DataLoader，时序数据不 shuffle。"""
        return DataLoader(self.val_dataset, self.batch_size)

    def test_dataloader(self):
        """测试集 DataLoader。"""
        return DataLoader(self.test_dataset, self.batch_size)
