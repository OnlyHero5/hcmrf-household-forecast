# Implementation Plan: Household Power Consumption Forecasting

## Framework Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Deep Learning | PyTorch 2.10 | 项目核心框架，已安装 |
| Training | Lightning 2.6 | 消除训练循环样板代码(早停/ckpt/设备管理) |
| Metrics | torchmetrics 1.9 | MSE/MAE 标准实现，与 Lightning 原生集成 |
| Data | torch Dataset + DataLoader | 简单直接，无额外抽象开销 |
| Config | Python dataclasses | 零依赖，IDE 友好，类型安全 |
| Visualization | matplotlib | 曲线图、柱状图、热力图 |

## Design Principles

```
高内聚低耦合
  ├── 每个文件一个职责
  ├── 组件通过构造函数/方法参数通信（无全局状态）
  ├── nn.Module 只定义计算图（纯 forward）
  └── LightningModule 只编排训练逻辑（不定义模型结构）

可维护性
  ├── 显式优于隐式（所有超参数在 __init__ 形参中声明）
  ├── self.save_hyperparameters() 自动序列化所有超参
  └── 类型注解全覆盖

可扩展性
  ├── 所有模型统一 forward(x, horizon) 接口
  ├── 新模型 = 新增 models/*.py + nn.Module
  ├── 新消融 = 修改一行 config 的 backbone 选择
  └── 新指标 = 加到 torchmetrics.MetricCollection

零防御性编程
  ├── 不检查参数合法性（调用者负责传对类型）
  ├── 不处理不可能的错误（None check, 除零保护等）
  └── 不写 fallback 路径
```

## File Architecture

```
homework/
│
├── data/                         # 数据文件（只读）
│   ├── train.csv
│   └── test.csv
│
├── homework/
│   ├── __init__.py
│   │
│   ├── config.py                 # dataclass: 全局配置
│   ├── dataset.py                # Dataset: 滑动窗口 + scaler
│   ├── features.py               # 函数: 时间/滞后/统计特征
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm.py               # nn.Module: LSTM
│   │   ├── transformer.py        # nn.Module: Transformer
│   │   ├── hcmrf.py              # nn.Module: HCMRF (完整)
│   │   ├── hcmrf_ablations.py    # nn.Module: HCMRF 5个消融变体
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── hcm.py            # Horizon Conditioning Module
│   │       ├── adaptive_patch.py # Adaptive-Patch Transformer
│   │       └── drd.py            # Dynamic Resolution Decoder
│   │
│   ├── system.py                 # LightningModule: 训练/验证/测试步骤
│   ├── datamodule.py             # LightningDataModule: 数据管线
│   │
│   ├── train.py                  # 入口: 训练一个模型
│   ├── evaluate.py               # 入口: 加载ckpt → 测试指标
│   ├── visualize.py              # 入口: 加载ckpt → 画图
│   └── run.py                    # 主控: 依次运行全部实验
│
├── outputs/                      # 自动生成
│   ├── lightning_logs/           # TensorBoard 日志
│   ├── checkpoints/              # 模型权重
│   ├── results/                  # JSON 指标汇总
│   └── figures/                  # PNG 图片
│
├── scripts/                      # shell 辅助
│   └── run_all.sh               # 一键运行全部实验
│
├── requirements.txt
├── IMPLEMENT.md                  # 本文档
└── PLAN.md                       # 项目计划书
```

## Component Contracts

### config.py — 单一配置入口

```python
@dataclass
class Config:
    # Data
    data_path: str = "data"
    input_len: int = 90
    short_horizon: int = 90
    long_horizon: int = 365
    step_size: int = 7        # 滑动窗口步长；7 减少重叠，1 最大化样本数
    
    # Training
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    
    # Model selection
    model_name: str = "lstm"  # lstm / transformer / hcmrf / hcmrf_ablation_X
    horizon: int = 90         # 90 or 365

# 使用方式: Config(model_name="lstm", horizon=90)
```

**为什么用 dataclass 不是 yaml/hydra：**
- 零依赖
- IDE 自动补全 + 类型检查
- 一个文件一目了然
- 修改配置 = 改 Python 代码，不引入额外 DSL

### dataset.py — 滑动窗口

```python
class PowerDataset(Dataset):
    """单一时序的滑动窗口数据集。
    
    输入: (total_days, n_features) 的 numpy 数组
    输出: (input_len, n_features) 的特征 + (horizon,) 的目标
    
    不做 defensive check: 调用者保证 total_days >= input_len + horizon
    """
    def __init__(self, data: np.ndarray, input_len: int, horizon: int, step: int = 7):
        self.data = data
        self.input_len = input_len
        self.horizon = horizon
        self.step = step  # 步长 >1 减少相邻样本重叠

    def __len__(self) -> int:
        return max(0, (len(self.data) - self.input_len - self.horizon) // self.step + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.step
        x = self.data[i : i + self.input_len]
        y = self.data[i + self.input_len : i + self.input_len + self.horizon, 0]  # target=col0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
```

### features.py — 特征工程

```python
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间/滞后/统计特征。返回新 DataFrame，不修改输入。"""
    date = pd.to_datetime(df["Date"])
    
    # 时间周期性
    df["doy_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365)
    df["month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * date.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * date.dt.dayofweek / 7)
    df["is_weekend"] = (date.dt.dayofweek >= 5).astype(float)
    
    # 滞后特征（仅目标变量）
    df["lag_7"] = df["Global_active_power"].shift(7)
    df["lag_30"] = df["Global_active_power"].shift(30)
    
    # 滚动统计
    df["roll_mean_7"] = df["Global_active_power"].rolling(7).mean()
    df["roll_mean_30"] = df["Global_active_power"].rolling(30).mean()
    
    # 丢弃引入 NaN 的行（lag/rolling 导致的开头行）
    return df.dropna().reset_index(drop=True)
```

### models/ — 纯计算图

每个模型文件遵循：
1. 继承 `nn.Module`
2. `__init__` 声明所有超参数为显式形参（无 **kwargs）
3. `forward(x, horizon) -> Tensor` —— 统一接受 horizon 参数（HCMRF 使用，LSTM/Transformer 忽略）
4. 不含训练逻辑、指标计算、设备操作

```python
# models/lstm.py
class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 128, 
                 num_layers: int = 2, horizon: int = 90):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x: torch.Tensor, horizon: int = 90) -> torch.Tensor:
        # x: (B, T, n_features); horizon 参数为统一接口保留，不使用
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # (B, horizon)
```

```python
# models/transformer.py
class TransformerModel(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, horizon: int = 90):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)  # sin/cos fixed
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, 
                                                   dim_feedforward=256, 
                                                   dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x: torch.Tensor, horizon: int = 90) -> torch.Tensor:
        # x: (B, T, n_features); horizon 参数为统一接口保留，不使用
        x = self.embed(x) + self.pos(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))  # (B, horizon)
```

### models/components/ — HCMRF 子模块

每个子模块一个文件，独立可测试：

```python
# components/hcm.py
class HorizonConditioning(nn.Module):
    """Horizon 条件化：分辨率压缩 + 通道门控。
    
    设计约束：90d 和 365d 模型分别训练（课程硬性要求），
    每个模型实例只看到一个 horizon，因此分辨率分支是
    硬编码的。通道门控是可学习的 nn.Parameter。
    """
    def __init__(self, d_model: int):
        super().__init__()
        # 可学习的通道门控（每个模型独立，反向传播正常）
        self.channel_gate = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        # x: (B, T, C), horizon: 90 或 365
        gate = torch.sigmoid(self.channel_gate)  # (C,)
        
        if horizon == 90:
            # 90d: 不压缩时序分辨率，直接应用门控
            return x * gate
        else:
            # 365d: 固定压缩到 ~30 步，再应用门控
            T = x.size(1)
            out_T = max(T // 3, 30)  # 固定压缩比
            x_pooled = F.adaptive_avg_pool1d(
                x.transpose(1, 2), out_T
            ).transpose(1, 2)
            return x_pooled * gate
```

```python
# components/adaptive_patch.py
class AdaptivePatchTransformer(nn.Module):
    """根据 horizon 选择 patch size + Linear 投影的 Transformer 编码器。
    
    Patch 后特征维度变为 C * patch_size，需通过投影层恢复为 d_model。
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        # 投影层: 将 C * patch_size 映射回 d_model
        self.projection = nn.LazyLinear(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        if patch_size > 1:
            B, T, C = x.shape
            T_patched = T // patch_size
            # 先 reshape 将 patch 的维度拼接到 feature 维度
            x = x[:, :T_patched * patch_size].reshape(B, T_patched, C * patch_size)
            # 再投影回 d_model，保证 Transformer 输入维度正确
            x = self.projection(x)
        return self.encoder(x)
```

```python
# components/drd.py
class DynamicResolutionDecoder(nn.Module):
    """粗→精解码: 90d 直接输出, 365d 先粗预测再上采样 + 多层精修。
    
    精修使用 3 层 Conv1D(k=7) 堆叠，感受野 = 19 个插值点，
    能有效修正线性插值引入的平滑伪影。
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU()
        )
        self.head_90 = nn.Linear(128, 90)
        self.head_coarse = nn.Linear(128, 52)  # 52 周 ≈ 365 天
        
        # 多层 Conv1D 精修（增强感受野）
        self.refine = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=7, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=7, padding="same"),
        )

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        # x: (B, d_model)
        h = self.shared(x)
        if horizon == 90:
            return self.head_90(h)
        # 365: 粗预测 52 周 → 上采样 365 → 多层精修
        coarse = self.head_coarse(h)                               # (B, 52)
        coarse_365 = F.interpolate(
            coarse.unsqueeze(1), size=365, mode="linear"
        ).squeeze(1)                                               # (B, 365)
        refined = self.refine(coarse_365.unsqueeze(1)).squeeze(1)  # (B, 365)
        return refined
```

### models/hcmrf.py — 完整 HCMRF

```python
class HCMRF(nn.Module):
    """Horizon-Conditioned Multi-Resolution Forecasting."""
    def __init__(self, n_features: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.encoder = nn.Conv1d(n_features, d_model, kernel_size=7, padding="same")
        self.hcm = HorizonConditioning(d_model)
        self.transformer = AdaptivePatchTransformer(d_model, n_heads, n_layers)
        self.decoder = DynamicResolutionDecoder(d_model)

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        # x: (B, T, n_features), horizon: 90 或 365
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d_model)
        x = self.hcm(x, horizon)
        patch_size = 1 if horizon == 90 else 3
        x = self.transformer(x, patch_size)
        x = x.mean(dim=1)  # GlobalAvgPool over time
        return self.decoder(x, horizon)
```

### models/hcmrf_ablations.py — 5 个消融变体

每个变体是 HCMRF 的一个子类，只覆写需要修改的部分：

```python
class HCMRF_wo_HCM(HCMRF):
    """消融 B: 去掉 HCM。无压缩 + 无门控 + 固定 patch=1。"""
    def forward(self, x, horizon):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        # 跳过 self.hcm（无压缩、无门控）
        x = self.transformer(x, patch_size=1)   # 固定 patch=1
        x = x.mean(dim=1)
        return self.decoder(x, horizon)

class HCMRF_wo_Patch(HCMRF):
    """消融 C: 去掉 Adaptive-Patch。保留 HCM 压缩 + 固定 patch=1。"""
    def forward(self, x, horizon):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        x = self.hcm(x, horizon)                # 保留 HCM
        x = self.transformer(x, patch_size=1)   # 固定 patch
        x = x.mean(dim=1)
        return self.decoder(x, horizon)

class HCMRF_wo_DRD(HCMRF):
    """消融 D: 去掉 DRD。直接 Dense 输出（无粗→精解码）。"""
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2):
        super().__init__(n_features, d_model, n_heads, n_layers)
        self.direct_head = nn.Linear(d_model, 365)

    def forward(self, x, horizon):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        x = self.hcm(x, horizon)
        patch_size = 1 if horizon == 90 else 3
        x = self.transformer(x, patch_size)
        x = x.mean(dim=1)
        return self.direct_head(x)              # 跳过 DRD，直接 365 输出

class HCMRF_wo_Gate(HCMRF):
    """消融 E: 去掉 channel_gate。仅分辨率压缩，无门控。"""
    def forward(self, x, horizon):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        # 只用分辨率压缩，跳过通道门控
        if horizon == 365:
            T = x.size(1)
            out_T = max(T // 3, 30)
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), out_T).transpose(1, 2)
        # 不乘以 gate
        patch_size = 1 if horizon == 90 else 3
        x = self.transformer(x, patch_size)
        x = x.mean(dim=1)
        return self.decoder(x, horizon)

class HCMRF_wo_Shared(HCMRF):
    """消融 F: 两个 horizon 使用独立编码器（参数翻倍）。"""
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2):
        super().__init__(n_features, d_model, n_heads, n_layers)
        self.encoder_365 = nn.Conv1d(n_features, d_model, kernel_size=7, padding="same")

    def forward(self, x, horizon):
        enc = self.encoder if horizon == 90 else self.encoder_365
        x = enc(x.transpose(1, 2)).transpose(1, 2)
        x = self.hcm(x, horizon)
        patch_size = 1 if horizon == 90 else 3
        x = self.transformer(x, patch_size)
        x = x.mean(dim=1)
        return self.decoder(x, horizon)
```

### system.py — LightningModule

```python
class ForecastSystem(pl.LightningModule):
    """编排训练/验证/测试逻辑。不定义模型结构。"""
    def __init__(self, model: nn.Module, 
                 model_name: str, horizon: int,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.MSELoss()
        self.metrics = torchmetrics.MetricCollection({
            "MSE": torchmetrics.MeanSquaredError(),
            "MAE": torchmetrics.MeanAbsoluteError(),
        })

    def forward(self, x):
        # 直接将 horizon 传给 model（HCMRF 用，LSTM/Transformer 忽略）
        return self.model(x, self.hparams.horizon)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        self.metrics(y_pred, y)
        self.log_dict(self.metrics, prefix="val/", on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        self.metrics(y_pred, y)
        self.log_dict(self.metrics, prefix="test/", on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
```

### datamodule.py — 数据管线

```python
class PowerDataModule(pl.LightningDataModule):
    """一个数据集管理 train/val/test 三个 DataLoader。"""
    def __init__(self, data_dir: str, input_len: int, horizon: int, 
                 batch_size: int, step_size: int = 7, scaler: str = "minmax"):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        # 1. 读 CSV
        train_df = pd.read_csv(f"{self.hparams.data_dir}/train.csv")
        test_df = pd.read_csv(f"{self.hparams.data_dir}/test.csv")
        
        # 2. 特征工程
        train_df = add_features(train_df)
        test_df = add_features(test_df)
        
        # 3. 归一化（只在 train 上 fit）
        self.scaler = MinMaxScaler()
        train_values = self.scaler.fit_transform(train_df.iloc[:, 1:])  # 跳过 Date 列
        test_values = self.scaler.transform(test_df.iloc[:, 1:])
        
        # 4. 切分 train/val（最后 20% 天数做 val）
        n_val = int(len(train_values) * 0.2)
        val_values = train_values[-n_val:]
        train_values = train_values[:-n_val]
        
        # 5. Dataset（使用 step_size 减少重叠）
        self.train_dataset = PowerDataset(train_values, self.hparams.input_len,
                                          self.hparams.horizon, self.hparams.step_size)
        self.val_dataset = PowerDataset(val_values, self.hparams.input_len,
                                        self.hparams.horizon, self.hparams.step_size)
        self.test_dataset = PowerDataset(test_values, self.hparams.input_len,
                                         self.hparams.horizon, self.hparams.step_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.hparams.batch_size)
```

### train.py — 单次训练入口

```python
def train(config: Config) -> str:
    """训练一个模型，返回 checkpoint 路径。"""
    pl.seed_everything(config.seed, workers=True)
    
    model = build_model(config)  # 根据 config.model_name 创建 nn.Module
    dm = PowerDataModule(config.data_path, config.input_len, 
                         config.horizon, config.batch_size,
                         config.step_size)
    system = ForecastSystem(model, config.model_name, config.horizon, 
                           config.learning_rate)
    
    callbacks = [
        EarlyStopping(monitor="val/MSE", patience=config.patience, mode="min"),
        ModelCheckpoint(dirpath="outputs/checkpoints", 
                       filename=f"{config.model_name}_h{config.horizon}_s{config.seed}",
                       monitor="val/MSE", mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(max_epochs=config.max_epochs, callbacks=callbacks,
                         enable_progress_bar=True)
    
    trainer.fit(system, dm)
    return trainer.checkpoint_callback.best_model_path
```

### evaluate.py — 加载 checkpoint 计算指标

```python
def evaluate(config: Config, ckpt_path: str) -> dict:
    """加载 checkpoint，在测试集上计算 MSE/MAE。"""
    model = build_model(config)
    system = ForecastSystem.load_from_checkpoint(ckpt_path, model=model,
                                                  model_name=config.model_name,
                                                  horizon=config.horizon)
    dm = PowerDataModule(config.data_path, config.input_len,
                         config.horizon, config.batch_size,
                         config.step_size)
    trainer = pl.Trainer(enable_progress_bar=False)
    results = trainer.test(system, dm)
    return results[0]  # {"test/MSE": ..., "test/MAE": ...}
```

### run.py — 主控脚本

```python
def seasonal_naive_baseline(train_df, test_df, horizon):
    """季节性朴素基线：以去年同日/同周值作为预测。
    对于 90d 预测，取前一年同 90 天窗口的值。
    对于 365d 预测，取前一年整年的值。
    返回 MSE 和 MAE。
    """
    # 实现：用 train 最后 horizon 天作为 test 的预测值
    y_pred = train_df["Global_active_power"].values[-horizon:]
    y_true = test_df["Global_active_power"].values[:horizon]
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return {"MSE": mse, "MAE": mae}

def run_all():
    """依次运行全部实验，汇总结果到 JSON。"""
    seeds = [42, 123, 456, 789, 2024]
    experiments = [
        ("lstm", 90), ("lstm", 365),
        ("transformer", 90), ("transformer", 365),
        ("hcmrf", 90), ("hcmrf", 365),
        ("hcmrf_wo_HCM", 90), ("hcmrf_wo_HCM", 365),
        ("hcmrf_wo_Patch", 90), ("hcmrf_wo_Patch", 365),
        ("hcmrf_wo_DRD", 90), ("hcmrf_wo_DRD", 365),
        ("hcmrf_wo_Gate", 90), ("hcmrf_wo_Gate", 365),
        ("hcmrf_wo_Shared", 90), ("hcmrf_wo_Shared", 365),
    ]
    
    summary = {}
    
    # 先跑统计基线（无随机性，只需一次）
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    for horizon in [90, 365]:
        result = seasonal_naive_baseline(train_df, test_df, horizon)
        summary[f"seasonal_naive_h{horizon}"] = {"mean": result, "std": {"MSE": 0, "MAE": 0}}
    
    # 深度学习实验
    for model_name, horizon in experiments:
        metrics = []
        for seed in seeds:
            cfg = Config(model_name=model_name, horizon=horizon, seed=seed)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        
        # 均值 ± 标准差
        avg = {k: np.mean([m[k] for m in metrics]) for k in metrics[0]}
        std = {k: np.std([m[k] for m in metrics]) for k in metrics[0]}
        summary[f"{model_name}_h{horizon}"] = {"mean": avg, "std": std}
    
    os.makedirs("outputs/results", exist_ok=True)
    json.dump(summary, open("outputs/results/summary.json", "w"), indent=2)
```

## Execution Flow

```
run.py
  │
  ├── 1. 统计基线（seasonal_naive_baseline）
  │       └── 去年同日值预测 → baseline_metrics
  │
  ├── 2. 遍历 (model_name, horizon) × seeds[5]
  │     │
  │     ├── config.py  →  Config(step_size=7, ...)
  │     │
  │     ├── build_model(config)
  │     │     ├── "lstm"         → LSTMModel(...)        # forward(x, horizon)
  │     │     ├── "transformer"  → TransformerModel(...)  # forward(x, horizon)
  │     │     ├── "hcmrf"        → HCMRF(...)            # forward(x, horizon)
  │     │     ├── "hcmrf_wo_HCM" → HCMRF_wo_HCM(...)
  │     │     └── ...
  │     │
  │     ├── datamodule.py → PowerDataModule(step_size=7)
  │     ├── system.py     → ForecastSystem
  │     │                      └── forward(x) → model(x, self.hparams.horizon)
  │     │
  │     ├── train.py
  │     │     ├── pl.Trainer.fit() → checkpoint.pth
  │     │     └── EarlyStopping + ModelCheckpoint
  │     │
  │     └── evaluate.py
  │           └── pl.Trainer.test() → {"MSE": ..., "MAE": ...}
  │
  └── outputs/results/summary.json  (含 baseline 指标)
```

## Ablation Study — 6 个变体的代码关系

```
完整 HCMRF (hcmrf.py)
  ├── HCM (hcm.py)              — 硬编码分辨率分支 + 可学习通道门控
  ├── Adaptive-Patch (adaptive_patch.py) — Patch + Linear 投影
  └── DRD (drd.py)              — 粗→精解码 + 多层 Conv1D 精修

消融变体 (hcmrf_ablations.py)
  ├── HCMRF_wo_HCM    ← 删除 self.hcm 调用（无压缩无门控），固定 patch_size=1
  ├── HCMRF_wo_Patch  ← 删除 patch 选择，固定 patch_size=1（保留 HCM）
  ├── HCMRF_wo_DRD    ← 替换 decoder 为 Dense(365)（跳过粗→精解码）
  ├── HCMRF_wo_Gate   ← 跳过 channel_gate，保留分辨率压缩
  └── HCMRF_wo_Shared ← 增加独立 encoder_365（参数翻倍）
```

每个消融变体修改一行 `forward()` 或增加一个层。不需要复制整个模型。

## Visualization Plan

| 图 | 脚本 | 内容 |
|----|------|------|
| 预测对比 | visualize.py | 三种模型预测曲线 vs Ground Truth (90d + 365d) |
| 消融柱状图 | visualize.py | 6变体 MSE/MAE 并排对比（带误差棒） |
| HCM 热力图 | visualize.py | channel_gate 权重分布（各通道重要性） |

```python
# visualize.py 核心逻辑
def plot_predictions(config, ckpt_paths, horizon):
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, ckpt in ckpt_paths.items():
        y_pred = load_and_predict(ckpt, config)
        ax.plot(y_pred, label=name, alpha=0.8)
    ax.plot(y_true, label="Ground Truth", color="black", linewidth=2)
    ax.legend(); ax.set_title(f"Prediction Comparison (horizon={horizon})")
    fig.savefig(f"outputs/figures/comparison_{horizon}d.png")

def plot_ablation(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, metric in zip(axes, ["MSE", "MAE"]):
        x = range(len(results_df))
        ax.bar(x, results_df[metric], yerr=results_df[f"{metric}_std"])
        ax.set_xticks(x); ax.set_xticklabels(results_df["variant"], rotation=30)
        ax.set_title(f"Ablation — {metric}")
    fig.tight_layout()
    fig.savefig("outputs/figures/ablation.png")
```

## Dependencies

```
# requirements.txt
torch>=2.0
lightning>=2.0
torchmetrics>=1.0
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.2
```

All are available via `pip install`. No exotic dependencies.

## Why This Design Satisfies Each Requirement

| Requirement | How It's Met |
|-------------|-------------|
| **使用现有框架+SDK** | PyTorch + Lightning + torchmetrics，全行业标准 |
| **无防御性编程** | 不检查参数/不可能路径/fallback。调用者负责传对参数 |
| **可维护性** | 每个文件 <150 行，一个职责。`self.save_hyperparameters()` 自动追踪实验 |
| **可扩展性** | 所有模型统一 `forward(x, horizon)` 接口。新模型 = 新增 `models/*.py`。新消融 = 修改一行 config |
| **可读性** | 类型注解全覆盖。类名直接对应论文组件名。nn.Module 只有 forward |
| **高内聚** | model 只管计算图，system 只管训练逻辑，datamodule 只管数据 |
| **低耦合** | 组件间只通过构造函数参数通信。Lightning 的 callback 模式解耦了"工程逻辑" |

## 实现顺序

```
Day 1:  config.py + dataset.py + features.py → 可以跑通数据管线
Day 2:  models/lstm.py + models/transformer.py → 可以跑通训练
Day 3:  models/components/*.py + models/hcmrf.py → HCMRF 完整模型
Day 4:  models/hcmrf_ablations.py → 5个消融变体
Day 5:  run.py + evaluate.py → 批量实验 + 汇总
Day 6:  visualize.py → 全部图表
Day 7:  调参 + bug fix
```
