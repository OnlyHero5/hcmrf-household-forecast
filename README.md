# HCMRF

> **基于 Horizon 特化的多分辨率家庭电力消耗时序预测**
>
> *Horizon-Specialized Multi-Resolution Forecasting for Household Power Consumption*

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [目录结构](#3-目录结构)
4. [环境配置](#4-环境配置)
5. [使用方法](#5-使用方法)
6. [配置说明](#6-配置说明)
7. [核心模块详解](#7-核心模块详解)
8. [实验方案与评估](#8-实验方案与评估)
9. [依赖项](#9-依赖项)
10. [参考文献](#10-参考文献)

---

## 1. 项目概述

### 1.1 研究问题

家庭电力消耗预测是智能电网、需求侧管理和能源规划中的核心任务。本项目基于 UCI **Individual Household Electric Power Consumption** 数据集（法国一户家庭，2006/12–2010/11，分钟级采样），将数据聚合为日粒度，使用过去 **90 天**的多变量时序数据，预测未来 **90 天（短期）**和 **365 天（长期）**的有功功率（`Global_active_power`）。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        预测任务定义                                    │
│                                                                     │
│   输入窗口 (90 天)                    预测窗口                         │
│   ┌──────────────────────┐           ┌──────────────────────┐       │
│   │ 多变量日级时序特征     │  ──────►  │ Global_active_power │       │
│   │ (24 维 × 90 步)       │           │ 未来 90d 或 365d     │       │
│   └──────────────────────┘           └──────────────────────┘       │
│                                                                     │
│   短期任务: 90 → 90    长期任务: 90 → 365                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 课程任务与三大模型

| 部分 | 模型 | 性质 | 说明 |
|------|------|------|------|
| 1 | **LSTM** | 基础题 | 双层 LSTM + 全连接输出头 |
| 2 | **Transformer** | 基础题 | 位置编码 + Encoder + 全局池化 |
| 3 | **HCMRF** | 开放题 | 自改进模型，Horizon 特化多分辨率架构 |

### 1.3 核心思路

```
                    过去 90 天多变量时序
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              v              v              v
        ┌──────────┐  ┌────────────┐  ┌──────────────┐
        │   LSTM   │  │Transformer │  │    HCMRF     │
        │  序列建模 │  │  自注意力   │  │ 多分辨率特化  │
        └─────┬────┘  └─────┬──────┘  └──────┬───────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             v
                  ┌─────────────────────┐
                  │  未来 90d / 365d     │
                  │  Global_active_power │
                  └─────────────────────┘
```

**HCMRF 的设计动机：** 绝大多数时序模型对 90 天和 365 天预测使用完全相同的架构，忽略了两个基本事实：

- **短期 (90d)** 需要细粒度信息，捕捉周模式、设备开关等局部波动
- **长期 (365d)** 需要粗粒度结构，捕捉季节趋势，同时对噪声不敏感

HCMRF 通过 **Horizon 特化的多分辨率架构**（分辨率压缩 + 自适应 Patch + 粗→精解码）解决这一问题。

### 1.4 评估标准

| 指标 | 说明 |
|------|------|
| **MSE** | 均方误差，主优化目标 |
| **MAE** | 平均绝对误差，辅助评估 |
| **重复实验** | 5 个随机种子 (42, 123, 456, 789, 2024)，报告 mean ± std |
| **可视化** | 预测曲线 vs 真实值；消融实验柱状图 |

---

## 2. 系统架构

### 2.1 总体架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HCMRF 项目系统架构                              │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │   configs/       │  │   scripts/       │  │   src/            │  │
│  │                  │  │                  │  │                  │  │
│  │  default.yaml    │  │  prepare_data.py │  │  config.py       │  │
│  │                  │  │  run_all.sh      │  │  features.py     │  │
│  │                  │  │  re_evaluate.py  │  │  dataset.py      │  │
│  └────────┬─────────┘  └────────┬─────────┘  │  datamodule.py   │  │
│           │                     │             │  system.py       │  │
│           │                     │             │  train.py        │  │
│           │                     │             │  evaluate.py     │  │
│           │                     │             │  visualize.py    │  │
│           │                     │             │  run.py          │  │
│           │                     │             │  models/         │  │
│           │                     │             └────────┬─────────┘  │
│           │                     │                      │            │
│           v                     v                      v            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      数据流向                                  │   │
│  │                                                              │   │
│  │  data/raw/ ──► prepare_data.py ──► data/processed/            │   │
│  │  (分钟级 UCI)                    (train.csv / test.csv)       │   │
│  │                                        │                     │   │
│  │                                        v                     │   │
│  │                              PowerDataModule                  │   │
│  │                              (特征工程 + 归一化 + 滑动窗口)    │   │
│  │                                        │                     │   │
│  │                         ┌──────────────┼──────────────┐      │   │
│  │                         v              v              v      │   │
│  │                      LSTM         Transformer       HCMRF     │   │
│  │                         │              │              │      │   │
│  │                         └──────────────┼──────────────┘      │   │
│  │                                        v                     │   │
│  │                              outputs/ (checkpoints / results) │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 训练框架分层

```
┌─────────────────────────────────────────────────────────────────┐
│                     PyTorch Lightning 分层架构                    │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │  run.py / cli.py / scripts/run_all.sh                     │ │
│   │  ── 实验编排入口 ──                                        │ │
│   └───────────────────────────┬─────────────────────────────┘ │
│                               │                                 │
│                               v                                 │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │  ForecastSystem (LightningModule)                        │ │
│   │  ── 训练 / 验证 / 测试步骤编排 ──                          │ │
│   │  ── 早停、checkpoint、指标记录 ──                          │ │
│   └───────────────────────────┬─────────────────────────────┘ │
│                               │                                 │
│              ┌────────────────┼────────────────┐                │
│              v                v                v                │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │  LSTMModel   │  │Transformer   │  │   HCMRF      │         │
│   │  (nn.Module) │  │  Model       │  │  (nn.Module) │         │
│   │  纯计算图     │  │  纯计算图     │  │  纯计算图     │         │
│   └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │  PowerDataModule (LightningDataModule)                     │ │
│   │  ── CSV 读取 → 特征工程 → MinMaxScaler → DataLoader ──    │ │
│   └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 数据管线流程

```
原始数据 (分钟级)
    │
    │  prepare_data.py
    │  ├── 按天聚合 (功率求和, 电压/电流平均)
    │  ├── 融合 ST QUENTIN 气象站月度天气数据
    │  ├── 缺失值处理 (插值 + ffill/bfill)
    │  └── 时序切分 (2010-01-01 为界)
    │
    v
data/processed/train.csv (≈1,112 天)    data/processed/test.csv (≈330 天)
    │
    │  PowerDataModule.setup()
    │  ├── add_features()  → +11 列工程特征
    │  ├── MinMaxScaler    → fit on train only
    │  ├── 80/20 切分 val  → 训练集末尾 20% 做验证
    │  └── PowerDataset    → 滑动窗口 (step=7)
    │
    v
DataLoader batch: (x, y)
    x: (B, 90, 24)   输入特征
    y: (B, horizon)  目标 Global_active_power (归一化)
```

---

## 3. 目录结构

```
电力预测/                                  # 项目根目录
│
├── README.md                              # 项目说明文档（本文件）
├── requirements.txt                       # Python 依赖清单
├── CLAUDE.md                              # AI 辅助开发说明（英文）
│
├── configs/                               # 配置文件
│   └── default.yaml                       #   全局超参数（数据/训练/模型）
│
├── data/                                  # 数据目录（.gitignore 排除）
│   ├── raw/                               #   原始 UCI 分钟级数据
│   │   └── household_power_consumption.txt
│   ├── weather/                           #   气象站月度数据
│   └── processed/                         #   预处理输出
│       ├── train.csv                      #     训练集 (2006-12 ~ 2009-12)
│       └── test.csv                       #     测试集 (2010-01 ~ 2010-11)
│
├── scripts/                               # 脚本入口
│   ├── prepare_data.py                    #   原始数据 → 日级 CSV
│   ├── run_all.sh                         #   一键运行全部实验
│   ├── run_baselines_parallel.sh          #   并行运行基线模型
│   └── re_evaluate.py                     #   重新评估已有 checkpoint
│
├── src/                                   # 核心源码
│   ├── config.py                          #   Config dataclass + YAML 加载
│   ├── features.py                        #   特征工程 (sin/cos, lag, rolling)
│   ├── dataset.py                         #   PowerDataset 滑动窗口
│   ├── datamodule.py                      #   PowerDataModule 数据管线
│   ├── system.py                          #   ForecastSystem (LightningModule)
│   ├── train.py                           #   build_model() + train()
│   ├── evaluate.py                        #   加载 checkpoint → 测试指标
│   ├── visualize.py                       #   预测曲线 + 消融柱状图
│   ├── run.py                             #   run_all() 主实验编排
│   ├── cli.py                             #   命令行统一入口
│   └── models/
│       ├── lstm.py                        #   LSTM 基线模型
│       ├── transformer.py                 #   Transformer 基线模型
│       ├── hcmrf.py                       #   HCMRF 完整模型
│       ├── hcmrf_ablations.py             #   4 个消融变体
│       └── components/
│           ├── hcm.py                     #   Horizon Conditioning Module
│           ├── adaptive_patch.py          #   Adaptive-Patch Transformer
│           └── drd.py                     #   Dynamic Resolution Decoder
│
├── outputs/                               # 实验输出（.gitignore 排除）
│   ├── checkpoints/                       #   模型权重 .ckpt
│   ├── results/                           #   JSON 指标汇总
│   │   ├── summary.json
│   │   ├── ablation_summary.json
│   │   └── hyperparam_ablation.json
│   └── figures/                           #   PNG 可视化
│
├── paper/                                 # LaTeX 实验报告
│   ├── report_final.tex
│   └── report_final.pdf
│
└── docs/                                  # 设计文档
    ├── PLAN.md                            #   项目计划书
    └── IMPLEMENT.md                       #   实现方案
```

---

## 4. 环境配置

### 4.1 系统要求

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| Python | 3.10+ | 3.11 |
| PyTorch | 2.0+ | 2.10 |
| GPU | 可选（CPU 可运行） | CUDA GPU 加速训练 |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 2 GB（含原始数据） | 5 GB（含实验输出） |

### 4.2 安装步骤

```bash
# 1. 进入项目目录
cd 电力预测

# 2. 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备数据（需先下载 UCI 原始数据到 data/raw/）
python scripts/prepare_data.py

# 5. 验证安装
python -c "from src.config import Config; from src.train import build_model; print('Setup OK')"
```

### 4.3 数据准备

原始数据需手动下载并放置：

```
data/raw/household_power_consumption.txt   ← UCI 数据集
data/weather/*.csv.gz                      ← 气象站数据（可选，脚本会自动查找）
```

运行 `prepare_data.py` 后将生成：

| 文件 | 行数 | 时间范围 |
|------|------|---------|
| `train.csv` | ≈1,112 天 | 2006-12 ~ 2009-12 |
| `test.csv` | ≈330 天 | 2010-01 ~ 2010-11 |

---

## 5. 使用方法

### 5.1 一键运行全部实验

```bash
bash scripts/run_all.sh
```

该脚本依次执行：
1. `python -m src.run` — 全部模型 × 全部 horizon × 5 种子 + 消融 + 超参数消融
2. 可视化入口检查

### 5.2 命令行接口 (CLI)

```bash
# 训练单个模型
python -m src.cli train --model lstm --horizon 90 --seed 42

# 使用 YAML 配置
python -m src.cli train --config configs/default.yaml --model hcmrf --horizon 365

# 运行全部实验
python -m src.cli run-all

# 评估 checkpoint
python -m src.cli evaluate --model lstm --horizon 90 \
  --ckpt outputs/checkpoints/lstm_h90_s42.ckpt

# 生成可视化
python -m src.cli visualize --horizon 90
```

### 5.3 Python 交互式调用

```bash
# 训练
python -c "from src.config import Config; from src.train import train; \
  train(Config(model_name='lstm', horizon=90, seed=42))"

# 评估
python -c "from src.config import Config; from src.evaluate import evaluate; \
  print(evaluate(Config(model_name='lstm', horizon=90, seed=42), \
  'outputs/checkpoints/lstm_h90_s42.ckpt'))"

# 可视化
python -c "from src.visualize import plot_model_comparison; plot_model_comparison(90)"
```

### 5.4 实验执行流程

```
┌─────────────────────────────────────────────────────────┐
│  run_all() 实验编排                                       │
│                                                         │
│  Phase 1: 统计基线                                       │
│    └── 季节性朴素预测 (去年同日值)                         │
│                                                         │
│  Phase 2: 主实验 (× 5 seeds)                             │
│    ├── lstm × (90d, 365d)                               │
│    ├── transformer × (90d, 365d)                        │
│    └── hcmrf × (90d, 365d)                              │
│                                                         │
│  Phase 3: 消融实验 (仅 365d, × 5 seeds)                  │
│    ├── hcmrf_wo_MultiScale                                │
│    ├── hcmrf_wo_Patch                                     │
│    ├── hcmrf_wo_DRD                                       │
│    └── hcmrf_wo_Shared                                    │
│                                                         │
│  Phase 4: 超参数消融 (仅 365d)                            │
│    ├── compress_factor: 2, 3, 4                         │
│    ├── refine_kernel: 3, 5, 7                           │
│    └── coarse_weeks: 26, 52                              │
│                                                         │
│  输出: outputs/results/*.json                             │
└─────────────────────────────────────────────────────────┘
```

---

## 6. 配置说明

### 6.1 Config 数据类

所有超参数集中在 `src/config.py` 的 `Config` dataclass 中：

```python
from src.config import Config

cfg = Config(
    model_name="hcmrf",    # lstm / transformer / hcmrf / hcmrf_wo_*
    horizon=365,           # 90 或 365
    seed=42,
    batch_size=32,
    learning_rate=1e-3,
    step_size=7,           # 滑动窗口步长
)
```

### 6.2 YAML 配置文件

`configs/default.yaml` 结构：

```yaml
data:
  data_path: "data/processed"
  input_len: 90              # 输入窗口
  short_horizon: 90          # 短期预测
  long_horizon: 365          # 长期预测
  step_size: 7               # 滑动步长

training:
  batch_size: 32
  max_epochs: 100
  patience: 10               # 早停
  learning_rate: 0.001
  seed: 42

lstm:
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2

transformer:
  d_model: 128
  n_heads: 4
  n_layers: 2

hcmrf:
  d_model: 64
  hcm_compress_factor: 3     # 365d 压缩因子
  drd_coarse_weeks: 52       # 粗预测周数
  drd_refine_kernel: 7       # Conv1D 精修核大小
  patch_size_90d: 1
  patch_size_365d: 3
```

### 6.3 关键超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_len` | 90 | 输入回看窗口（天） |
| `horizon` | 90 / 365 | 预测长度（天） |
| `step_size` | 7 | 滑动窗口步长；>1 减少相邻样本重叠 |
| `batch_size` | 32 | 训练批次大小 |
| `patience` | 10 | 验证集无改善时的早停轮数 |
| `learning_rate` | 1e-3 | Adam 优化器学习率 |

---

## 7. 核心模块详解

### 7.1 特征工程 (`features.py`)

```
原始特征 (13 维)                    工程特征 (+11 维)
┌─────────────────────────┐        ┌─────────────────────────┐
│ Global_active_power     │        │ doy_sin / doy_cos       │  年中日周期
│ Global_reactive_power   │        │ month_sin / month_cos   │  月份周期
│ Voltage                 │   +    │ dow_sin / dow_cos       │  星期周期
│ Global_intensity        │        │ is_weekend              │  周末标记
│ Sub_metering_1/2/3      │        │ lag_7 / lag_30          │  滞后特征
│ Sub_metering_remainder  │        │ roll_mean_7/30          │  滚动均值
│ RR, NBJRR1/5/10, NBJBROU│        └─────────────────────────┘
└─────────────────────────┘
              │
              v
        合计 24 维特征
        目标列: Global_active_power (第 0 列)
```

**设计要点：**
- sin/cos 编码避免边界跳变（如 12 月 31 日 → 1 月 1 日）
- 滞后和滚动统计仅在目标变量上计算
- `MinMaxScaler` 仅在训练集上 fit，避免数据泄漏

### 7.2 滑动窗口 (`dataset.py`)

```
时间轴:  ────●────●────●────●────●────●────●────►
              │←90→│←horizon→│
              └─ x ─┘└─ y ──┘     样本 1 (step=7 时跳 7 天)
                    │←90→│←horizon→│
                    └─ x ─┘└─ y ──┘  样本 2

| 参数 | 短期 (90→90) | 长期 (90→365) |
|------|-------------|--------------|
| input_window | 90 | 90 |
| output_window | 90 | 365 |
| step | 7 | 7 |
| 训练样本估算 | ~134 | ~94 |
```

### 7.3 LSTM 模型

```
输入: (batch, 90, 24)
  │
  ├── LSTM(hidden=128, layers=2, return_sequences=True) + Dropout(0.2)
  ├── LSTM(hidden=64,  return_sequences=False)         + Dropout(0.2)
  │
  ├── Linear(64, ReLU)
  ├── Linear(horizon)    ← 90 或 365
  │
输出: (batch, horizon)
```

| 超参数 | 值 |
|--------|-----|
| hidden_dim | 128 → 64 |
| num_layers | 2 |
| dropout | 0.2 |
| optimizer | Adam (lr=1e-3) |
| loss | MSE |

### 7.4 Transformer 模型

```
输入: (batch, 90, 24)
  │
  ├── Linear Embedding (24 → d_model=128)
  ├── Positional Encoding (sin/cos)
  │
  ├── TransformerEncoder × 2
  │     ├── Multi-Head Self-Attention (heads=4)
  │     ├── Feed-Forward (d_ff=256)
  │     └── LayerNorm + Dropout(0.1)
  │
  ├── Global Average Pooling
  ├── Linear(64, ReLU)
  ├── Linear(horizon)
  │
输出: (batch, horizon)
```

### 7.5 HCMRF 模型 — 完整架构

HCMRF（Horizon-Specialized Multi-Resolution Forecasting）是项目的核心创新模型，包含三个组件：

```
输入 (90天 × 24特征)
         │
    ┌────▼────┐
    │ Conv1D  │  ← 共享编码器 (kernel=7, filters=64)
    │ Encoder │
    └────┬────┘
         │  特征: (batch, 90, 64)
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ ★ Horizon Conditioning Module (HCM)                 │
    │                                                       │
    │  90d 路径:                                            │
    │    不压缩时序分辨率 (T' = 90)                          │
    │    × 逐通道门控 (可学习 nn.Parameter)                  │
    │                                                       │
    │  365d 路径:                                           │
    │    AdaptiveAvgPool1d → 压缩到 ~30 步                   │
    │    × 逐通道门控 (可学习 nn.Parameter)                  │
    └────┬─────────────────────────────────────────────────┘
         │  输出: (batch, T', 64)   T'=90 或 30
         │
    ┌────▼─────────────────────────────────┐
    │ Adaptive-Patch Transformer Encoder   │
    │                                       │
    │  90d:  patch_size=1, num_patches=90  │  ← 细粒度注意力
    │  365d: patch_size=3, num_patches=30  │  ← 粗粒度注意力
    │                                       │
    │  Patch + Linear 投影:                 │
    │    (C × patch_size) → d_model         │
    │                                       │
    │  TransformerLayer × 2:               │
    │    Multi-Head Attention (heads=4)     │
    │    FeedForward (d_ff=256)             │
    └────┬─────────────────────────────────┘
         │  输出: (batch, T', d_model)
         │
    ┌────▼───────────────────────────────────────────┐
    │ ★★ Dynamic Resolution Decoder (DRD)            │
    │                                                  │
    │  90d 路径:                                       │
    │    GlobalAvgPool → Dense(128) → Dense(90)       │
    │                                                  │
    │  365d 路径:                                      │
    │    Step 1 粗预测: GlobalAvgPool → Dense(52)      │  ← 周级粒度
    │    Step 2 上采样: Linear interpolation 52→365    │
    │    Step 3 精修:   Conv1D 堆叠 (3层, k=7)         │  ← 感受野=19
    └────┬───────────────────────────────────────────┘
         │
    ┌────▼────┐
    │  预测输出 │  (batch, 90) 或 (batch, 365)
    └─────────┘
```

#### 三大创新组件

| 组件 | 作用 | 不使用的代价 |
|------|------|-------------|
| **HCM** (分辨率压缩 + 通道门控) | 365d 压缩噪声，90d 保持细粒度 | 365d 关注过多天级噪声 |
| **Adaptive-Patch** | horizon 自适应 patch 尺寸 + 维度投影 | 维度不匹配或粒度不当 |
| **DRD** (粗→精解码) | 365d 先预测 52 周再插值精修 | 直接 Dense(365) 误差累积 |

### 7.6 消融变体

```
变体 A (完整 HCMRF):
  Encoder → HCM(res+gate) → AdaptivePatch → Transformer → DRD

变体 B (-wo_MultiScale):
  Encoder → [不压缩, T=90] → [patch=3] → Transformer → DRD

变体 C (-wo_Patch):
  Encoder → HCM(res) → [固定 patch=1] → Transformer → DRD

变体 D (-wo_DRD):
  Encoder → HCM → AdaptivePatch → Transformer → Dense(365)

变体 E (-wo_Shared):
  Encoder_90d ──┐
                 ├→ HCM → ... → DRD
  Encoder_365d ─┘  (两套独立参数)
```

> **注意：** 消融实验仅对 **365d** 有意义。90d 路径不池化、patch=1，消融退化为恒等操作。

| 变体名 | 去掉的组件 | 验证的问题 |
|--------|-----------|-----------|
| `hcmrf_wo_MultiScale` | 365d 分辨率压缩 | 多尺度池化是否必要 |
| `hcmrf_wo_Patch` | Adaptive-Patch | 动态 patch 是否优于固定 patch |
| `hcmrf_wo_DRD` | 粗→精解码 | DRD 是否优于直接多步输出 |
| `hcmrf_wo_Shared` | 共享编码器 | 独立编码器是否更好 |

---

## 8. 实验方案与评估

### 8.1 实验矩阵

| 模型 | 短期 (90d) | 长期 (365d) | 训练方式 |
|------|-----------|------------|---------|
| 季节性朴素基线 | ✅ | ✅ | 无训练，去年同日值 |
| LSTM | ✅ | ✅ | 独立训练 |
| Transformer | ✅ | ✅ | 独立训练 |
| HCMRF (完整) | ✅ | ✅ | 独立训练 |
| HCMRF 消融变体 | — | ✅ | 仅 365d |

### 8.2 评估流程

```
┌─────────────────────────────────────────────────────────┐
│  单次实验流程 (× 5 seeds)                                 │
│                                                         │
│  1. Config(model, horizon, seed)                        │
│  2. train(cfg) → checkpoint                             │
│  3. evaluate(cfg, ckpt) → MSE, MAE                     │
│  4. 汇总 5 轮 → mean ± std                               │
│                                                         │
│  指标在原始 kW 量纲上计算 (inverse_transform 还原)         │
└─────────────────────────────────────────────────────────┘
```

### 8.3 输出文件

```
outputs/
├── checkpoints/
│   ├── lstm_h90_s42.ckpt
│   ├── transformer_h365_s123.ckpt
│   ├── hcmrf_h90_s42.ckpt
│   └── hcmrf_wo_DRD_h365_s456.ckpt
│
├── results/
│   ├── summary.json              # 全部实验 mean±std
│   ├── ablation_summary.json     # 消融实验汇总
│   └── hyperparam_ablation.json  # 超参数消融汇总
│
└── figures/
    ├── comparison_90d.png        # 多模型预测曲线对比
    ├── comparison_365d.png
    ├── ablation_365d.png         # 消融 MSE/MAE 柱状图
    └── hyperparam_ablation.png   # 超参数敏感性
```

### 8.4 可视化

```bash
# 生成预测曲线对比图
python -c "from src.visualize import plot_model_comparison; plot_model_comparison(90)"
python -c "from src.visualize import plot_model_comparison; plot_model_comparison(365)"

# 生成消融实验柱状图
python -c "from src.visualize import plot_ablation; plot_ablation()"
```

### 8.5 设计原则

```
高内聚低耦合
  ├── 每个文件一个职责
  ├── 组件通过构造函数/方法参数通信（无全局状态）
  ├── nn.Module 只定义计算图（纯 forward）
  └── LightningModule 只编排训练逻辑

可维护性
  ├── 显式优于隐式（所有超参数在 __init__ 形参中声明）
  ├── self.save_hyperparameters() 自动序列化
  └── 类型注解全覆盖

可扩展性
  ├── 所有模型统一 forward(x, horizon) 接口
  ├── 新模型 = 新增 models/*.py + 在 build_model() 注册
  └── 新指标 = 加到 torchmetrics.MetricCollection
```

---

## 9. 依赖项

```
torch>=2.0              # 深度学习框架
lightning>=2.0          # 训练循环编排
torchmetrics>=1.0       # MSE/MAE 指标
pandas>=2.0             # 数据处理
numpy>=1.24             # 数值计算
matplotlib>=3.7         # 可视化
scikit-learn>=1.2       # MinMaxScaler
pyyaml                  # YAML 配置加载（可选）
```

---

## 10. 参考文献

| 数据集 / 方法 | 引用 |
|--------------|------|
| UCI Household Power Consumption | Hebrail & Berndt, 2012 |
| LSTM for Time Series | Hochreiter & Schmidhuber, 1997 |
| Transformer | Vaswani et al., 2017 |
| Patch-based Time Series | Nie et al. (PatchTST), 2023 |
| Reflexion / Self-Refine | Shinn et al., 2023; Madaan et al., 2023 |

---

## 许可证

本项目为机器学习课程大作业，仅供学术研究与课程考核使用。
