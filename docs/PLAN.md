# 机器学习课程大作业 — 家庭电力消耗多变量时间序列预测

## 项目计划书

---

## 一、项目概述

### 1.1 问题定义
基于 UCI "Individual Household Electric Power Consumption" 数据集（法国一户家庭，2006/12–2010/11，分钟级粒度），使用过去 **90 天**的电力消耗数据，预测未来 **90 天（短期）**和 **365 天（长期）**的有功功率（Global_active_power）。

### 1.2 三大任务
| 部分 | 方法 | 性质 | 权重 |
|------|------|------|------|
| 1 | LSTM | 基础题 | 1/3 |
| 2 | Transformer | 基础题 | 1/3 |
| 3 | 自改进模型 | 开放题（新颖性优先） | 1/3 |

### 1.3 评估标准
- **指标**：MSE（均方误差）、MAE（平均绝对误差）
- **要求**：至少 5 轮实验，取平均值 ± 标准差
- **可视化**：预测值 vs 真实值曲线对比图

### 1.4 提交物
- 实验报告（问题介绍 → 模型描述（含伪代码）→ 结果与分析 → 讨论）
- 代码（GitHub 链接）
- 结果截图 + 对比曲线

---

## 二、数据现状

### 2.1 已完成处理
| 步骤 | 状态 | 说明 |
|------|------|------|
| 原始数据下载 | ✅ | UCI 数据集已下载解压 |
| 分钟级→日聚合 | ✅ | global_active_power 等按天求和，voltage 等按天平均 |
| 天气数据融合 | ✅ | ST QUENTIN 气象站月数据（RR, NBJRR1/5/10, NBJBROU） |
| 缺失值处理 | ✅ | 插值 + ffill/bfill 补全，无缺失 |
| 基础切分 | ✅ | train 1,112天 (2006-12→2009-12), test 330天 (2010-01→2010-11) |

### 2.2 待补充（必要）
- **时间特征**：sin/cos 编码（月、星期、一年中的第几天）
- **滞后特征**：lag_7 (上周同日)、lag_30 (上月同日)
- **滚动统计**：7天/30天移动平均

---

## 三、技术路线

### 3.1 整体工作流

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: 特征工程 + 归一化 + 滑动窗口构建               │
│  ① 添加时间周期性特征（sin/cos编码）                     │
│  ② 添加滞后特征与滚动统计                                │
│  ③ MinMaxScaler 归一化（fit on train only）              │
│  ④ 构建 sliding windows（90→90, 90→365）                │
│  ⑤ PyTorch DataLoader 封装                               │
├─────────────────────────────────────────────────────────┤
│  Phase 2: LSTM 模型（基础题 1）                          │
│  ① 设计 LSTM 网络结构                                    │
│  ② 训练与验证（早停 + 学习率衰减）                      │
│  ③ 5 轮重复实验，记录 MSE/MAE 均值±标准差               │
├─────────────────────────────────────────────────────────┤
│  Phase 3: Transformer 模型（基础题 2）                   │
│  ① 设计 Transformer 编码器结构                           │
│  ② 训练与验证（同上）                                    │
│  ③ 5 轮重复实验，记录 MSE/MAE 均值±标准差               │
├─────────────────────────────────────────────────────────┤
│  Phase 4: 自改进模型（开放题 3）                          │
│  ① 设计方案（CNN + Transformer / Autoformer / 其他）     │
│  ② 训练与验证                                            │
│  ③ 5 轮重复实验 + 新颖性分析                             │
├─────────────────────────────────────────────────────────┤
│  Phase 5: 结果分析与报告撰写                              │
│  ① 三种方法对比表（MSE/MAE/参数量/训练时间）             │
│  ② 预测 vs 真实值曲线图（90天和365天各一张）             │
│  ③ 撰写实验报告四部分                                    │
│  ④ 上传 GitHub                                           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 数据预处理细节

#### 特征工程
```python
# 时间周期性特征（sin/cos 避免边界跳变）
features_time = [
    sin(2π * dayofyear / 365), cos(2π * dayofyear / 365),
    sin(2π * month / 12),       cos(2π * month / 12),
    sin(2π * weekday / 7),      cos(2π * weekday / 7),
]

# 布尔特征
features_bool = [is_weekend, is_holiday (法国法定假日)]

# 滞后特征
features_lag = [lag_7 (global_active_power), lag_30 (global_active_power)]

# 滚动统计
features_roll = [rolling_mean_7d, rolling_mean_30d, rolling_std_7d]
```

#### 归一化
- 使用 `MinMaxScaler(feature_range=(0,1))`
- 仅在训练集上 `.fit()`，transform 训练集和测试集
- 预测后 `.inverse_transform()` 还原为原始量纲以计算指标

#### 滑动窗口（核心）

每个样本由 `(input_window + output_window)` 天的连续数据构成：

```python
def create_sequences(data, input_len=90, output_len=90, step=1):
    """
    data: (n_days, n_features)
    returns: X (n_samples, input_len, n_features)
             y (n_samples, output_len)  — 仅 global_active_power
    """
    X, y = [], []
    for i in range(0, len(data) - input_len - output_len + 1, step):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len, target_col])
    return np.array(X), np.array(y)
```

| 参数 | 短期 (90→90) | 长期 (90→365) |
|------|-------------|--------------|
| input_window | 90 | 90 |
| output_window | 90 | 365 |
| step（滑动步长） | 7（可配置，默认7） | 7（可配置，默认7） |
| 训练样本数估算 (step=7) | ~134 | ~94 |
| 注：step=7 每7天采一个窗口，减少相邻样本高度重叠导致的过拟合风险。step=1 虽可增加样本数量（~843/558），但相邻窗口共享 89/90 的数据，有效信息量远不及样本数所暗示。 | | |

---

## 四、模型架构

### 4.1 LSTM 模型

```
输入: (batch, 90, n_features)
  │
  ├── LSTM(units=128, return_sequences=True) + Dropout(0.2)
  ├── LSTM(units=64, return_sequences=False) + Dropout(0.2)
  │
  ├── Dense(64, ReLU)
  ├── Dense(output_horizon)  ← 90 或 365
  │
 输出: (batch, output_horizon)     # 未来每天的有功功率
```

**超参数：**
| 参数 | 值 | 说明 |
|------|-----|------|
| LSTM layers | 2 | 平衡表达力与训练代价 |
| hidden_units | 128 → 64 | 逐层降维 |
| dropout | 0.2 | 防过拟合 |
| learning_rate | 1e-3 → 1e-4 衰减 | Adam 优化器 |
| batch_size | 32 | |
| epochs | 100（早停 patience=10） | |
| loss | MSE | |

### 4.2 Transformer 模型

```
输入: (batch, 90, n_features)
  │
  ├── Linear Embedding (n_features → d_model=128)
  ├── Positional Encoding (sin/cos, 可学习)
  │
  ├── TransformerEncoder × 2-3 层
  │     ├── Multi-Head Self-Attention (heads=4)
  │     ├── Feed-Forward (d_ff=256)
  │     └── LayerNorm + Dropout(0.1)
  │
  ├── Global Average Pooling (over time dimension)
  │   (或取最后一个时间步)
  │
  ├── Dense(64, ReLU)
  ├── Dense(output_horizon)  ← 90 或 365
  │
 输出: (batch, output_horizon)
```

**超参数：**
| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 128 | 嵌入维度 |
| n_heads | 4 | 注意力头数 |
| n_layers | 2-3 | 编码器层数 |
| d_ff | 256 | 前馈网络隐层 |
| dropout | 0.1 | |
| learning_rate | 1e-3 | 带 warmup 调度 |

### 4.3 自改进模型：Horizon-Conditioned Multi-Resolution Forecasting (HCMRF)

#### 设计动机

> 绝大多数时序预测模型（LSTM、Transformer、Autoformer）对**预测长度不同的任务使用完全相同的架构**——相同的时序分辨率、相同的信息处理路径。这忽略了两个基本事实：
>
> - **短期预测 (90d)** 需要细粒度信息来捕捉周模式、设备开关等局部波动
> - **长期预测 (365d)** 需要粗粒度结构来捕捉季节趋势，同时对噪声不敏感
>
> **核心创新：让模型感知"要预测多远"，并据此动态调整其内部的时序分辨率。**

#### 整体架构

```
输入 (90天 × n_features)
         │
    ┌────▼────┐
    │ 共享编码器 │  ← 1D-CNN (kernel=7, filters=64)
    │ (Shared) │     两个horizon有同构的编码器（参数独立，因分别训练）
    └────┬────┘
         │  特征: (batch, 90, 64)
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ ★ Horizon Conditioning Module (HCM)                 │
    │                                                       │
    │  ⚠ 设计修正: 90d/365d 模型分别训练（课程要求），         │
    │  每个模型实例只看到一个 horizon，故:                    │
    │                                                       │
    │  90d 路径:                                            │
    │    不压缩时序分辨率 (T' = 90)                          │
    │    × 逐通道门控 (可学习 nn.Parameter)                  │
    │                                                       │
    │  365d 路径:                                           │
    │    AdaptiveAvgPool1d → 固定压缩到 ~30 步               │
    │    × 逐通道门控 (可学习 nn.Parameter)                  │
    │                                                       │
    │  (可视化: 训练后画出通道权重分布)                       │  ← 消融: 去掉HCM→退化基准
    └────┬─────────────────────────────────────────────────┘
         │  输出: (batch, T', 64)  T'=90(90d时) or 30(365d时)
         │
    ┌────▼─────────────────────────────────┐
    │ Adaptive-Patch Transformer Encoder   │
    │                                       │
    │  90d: patch_size=1, num_patches=90   │  ← 细粒度注意力
    │  365d: patch_size=3, num_patches=30  │  ← 粗粒度注意力
    │                                       │
    │  ★ Patch + Linear 投影:               │
    │     (C * patch_size) → d_model       │  ← 修复维度不匹配
    │                                       │
    │  TransformerLayer × 2:               │
    │    Multi-Head Attention (heads=4)     │
    │    FeedForward (d_ff=256)             │
    │    LayerNorm + Dropout(0.1)           │
    └────┬─────────────────────────────────┘
         │  输出: (batch, T', d_model=128)
         │
    ┌────▼───────────────────────────────────────────┐
    │ ★★ Dynamic Resolution Decoder (DRD)            │
    │                                                  │
    │  90d 路径:                                       │
    │    GlobalAvgPool → Dense(128,ReLU) → Dense(90)  │ ← 直接多步输出
    │                                                  │
    │  365d 路径:                                      │
    │    Step 1 粗预测: GlobalAvgPool → Dense(52)      │ ← 周级粒度
    │    Step 2 上采样: Linear interpolation 52→365    │
    │    Step 3 精修:   多层 Conv1D 堆叠:              │ ← 增强感受野
    │      Conv1D(1→16, k=7) → ReLU →                 │
    │      Conv1D(16→16, k=7) → ReLU →                │
    │      Conv1D(16→1,  k=7)                         │
    │                                                  │
    │  (可视化: 粗预测曲线 vs 精修后曲线)               │ ← 消融: 去掉精修→只看粗预测
    └────┬───────────────────────────────────────────┘
         │
    ┌────▼────┐
    │  预测输出  │  (batch, 90) 或 (batch, 365)
    └─────────┘
```

#### 三个创新点的必要性论证

| 创新组件 | 解决什么问题 | 不这么做的代价 |
|---------|------------|--------------|
| **HCM** (硬编码分辨率分支 + 可学习门控) | 90d/365d 模型分别训练，无法动态感知 horizon；但仍需不同的时序分辨率和通道重要性 | 365d 预测关注过多天级噪声；90d 预测过度平滑；无门控则所有通道等权处理 |
| **Adaptive-Patch + Linear 投影** | patch 后维度膨胀 (C×patch_size)，直接输入 Transformer 会维度不匹配 | 发生 RuntimeError；或被迫让 Transformer 接受错误维度 |
| **DRD** (粗→精解码 + 多层 Conv1D 精修) | 直接预测 365 点误差大；单层 k=3 精修感受野仅 3，无法有效修正插值平滑伪影 | 365d 预测后期误差累积严重；精修形同虚设 |

#### 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 共享编码器 | Conv1D(k=7, filters=64) | 提取局部特征 |
| HCM 分辨率压缩 | 90d: 不压缩; 365d: 固定 AdaptiveAvgPool1d→30步 | 硬编码分支（因分别训练） |
| HCM 通道门控 | nn.Parameter(torch.ones(64)) × Sigmoid | 可学习，每个模型独立优化 |
| Transformer d_model | 128 | |
| Transformer heads | 4 | |
| Transformer layers | 2 | |
| d_ff | 256 | |
| dropout | 0.1 | |
| Adaptive-Patch 投影 | Linear(C×patch_size, d_model) | 修复维度不匹配 |
| 粗预测 (365d DRD) | 52 周 | 52周 ≈ 364天，接近 365 |
| DRD 精修结构 | Conv1D(1→16,k=7)→ReLU→Conv1D(16→16,k=7)→ReLU→Conv1D(16→1,k=7) | 3层堆叠，感受野=19 |
| optimizer | Adam (lr=1e-3, weight_decay=1e-5) | |
| batch_size | 32 | |
| epochs | 100 (early stopping patience=10) | |
| loss | MSE | |

---

## 五、实验方案

### 5.1 实验配置

每个模型需要进行 **5 次独立实验**：

```yaml
实验流程:
  1. 初始化模型（不同随机种子: 42, 123, 456, 789, 2024）
  2. 训练（早停 + 验证集监控）
  3. 在测试集上评估:
     - MSE = mean((y_true - y_pred)²)
     - MAE = mean(|y_true - y_pred|)
  4. 记录结果
  5. 重复 5 轮

最终报告:
  - MSE_mean ± MSE_std
  - MAE_mean ± MAE_std
```

### 5.2 实验矩阵

| 模型 | 短期 (90d) | 长期 (365d) | 训练方式 |
|------|-----------|------------|---------|
| **季节性朴素基线**（对比参照） | ✅ | ✅ | 无训练，以去年同日/同周值为预测 |
| LSTM | ✅ | ✅ | 独立训练，参数不共享 |
| Transformer | ✅ | ✅ | 独立训练，参数不共享 |
| HCMRF (自改进) | ✅ | ✅ | 独立训练，同构架构 + 硬编码horizon分支 |

### 5.3 消融实验方案（核心）

为验证 HCMRF 各创新组件的有效性，设计 **5 组消融实验 + 1 组完整模型**，分别在 90d 和 365d 上各运行 5 轮：

#### 消融实验清单

| 编号 | 变体 | 去掉的组件 | 回答的问题 |
|------|------|-----------|-----------|
| **A** | **完整 HCMRF**（基线） | — | 我们的完整方法表现如何 |
| **B** | **HCMRF -w/o HCM** | 去掉 HCM（固定 resolution_factor=1.0，固定 patch_size=1） | HCM 是否有用？分辨率自适应是否必要？ |
| **C** | **HCMRF -w/o Patch** | 去掉 Adaptive-Patch（固定 patch_size=1，保留 HCM 的 pool 压缩） | 动态 patch 是否比单纯 pooling 更好？ |
| **D** | **HCMRF -w/o DRD** | 365d 路径去掉 DRD（改用直接 Dense(365)） | 粗→精解码是否优于直接多步输出？ |
| **E** | **HCMRF -w/o Gate** | 去掉 channel_gate（只保留 resolution_factor） | 通道门控是否必要？ |
| **F** | **HCMRF -w/o Shared** | 共享编码器 → 两个 horizon 独立编码器 | 共享编码器是否有限？ |

#### 消融实验示意图

```
变体 A (完整):
  SharedEncoder → HCM(res+gate) → AdaptivePatch → Transformer → DRD

变体 B (-w/o HCM):
  SharedEncoder → [固定res=1.0] → [固定patch=1] → Transformer → Dense

变体 C (-w/o Patch):
  SharedEncoder → HCM(res) → [固定patch=1] → Transformer → DRD
                    ↓
               (仅res压缩)

变体 D (-w/o DRD):
  SharedEncoder → HCM(res+gate) → AdaptivePatch → Transformer → Dense(365)
                                                             (直接多步)

变体 E (-w/o Gate):
  SharedEncoder → HCM(res_only) → AdaptivePatch → Transformer → DRD

变体 F (-w/o Shared):
  Encoder_90d ──┐
                 ├→ HCM → ... → DRD
  Encoder_365d ─┘
  (两套独立参数)
```

#### 预期结果假设与分析框架

| 变体 | 90d 预期 | 365d 预期 | 含义 |
|------|---------|----------|------|
| A (完整) | ★★★★ | ★★★★ | 完整方法兼顾两者 |
| B (无HCM) | ★★★★ (不变) | ★★★ (下降) | 365d 因无压缩而关注过多噪声 |
| C (无Patch) | ★★★★ (不变) | ★★★★ (不变或略降) | 单纯 pooling 与 patch 作用类似 |
| D (无DRD) | — | ★★★ (下降) | 直接 365 输出误差累积 |
| E (无Gate) | ★★★★ | ★★★★ | 通道门控贡献可能较小 |
| F (无Shared) | ★★★★ | ★★★★ (略升) | 独立编码器可能更好但参数量翻倍 |

#### 消融实验报告模板

每个变体报告一张表：

```
表: HCMRF 消融实验 — 短期预测 (90d) / MSE (×10³)
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│  轮次   │  A(完整) │ B(无HCM) │C(无Patch)│D(无DRD)  │E(无Gate) │F(无Shared)│
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ seed=42  │   ...    │   ...    │   ...    │   ...    │   ...    │   ...    │
│ seed=123 │   ...    │   ...    │   ...    │   ...    │   ...    │   ...    │
│ seed=456 │   ...    │   ...    │   ...    │   ...    │   ...    │   ...    │
│ seed=789 │   ...    │   ...    │   ...    │   ...    │   ...    │   ...    │
│ seed=2024│   ...    │   ...    │   ...    │   ...    │   ...    │   ...    │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ mean±std │   ...    │   ...    │   ...    │   ...    │   ...    │   ...    │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

#### 可视化消融结果

- **柱状图 1**: 90d 预测下各消融变体的 MSE/MAE（带误差棒）
- **柱状图 2**: 365d 预测下各消融变体的 MSE/MAE（带误差棒）
- **折线图 3**: 各变体的预测曲线 vs Ground Truth（选最好的轮次）
- **热力图 4**: HCM 模块学到的通道门控权重分布（每个通道的重要性）

### 5.4 可视化要求
- 每张图包含：**预测曲线 vs Ground Truth 曲线**（测试集上）
- 短期：展示连续的 90 天预测段
- 长期：展示连续的 365 天预测段
- 三种方法的曲线放在同一张图中以便比较
- **额外**：消融实验对比柱状图（6 个变体并排）
- **额外**：HCM 门控权重热力图（解释模型行为）

---

## 六、项目文件结构

```
/homework/
├── 2026-专硕机器学习课程考核.pdf      # 课程要求
├── household_power_consumption.txt     # UCI 原始数据（分钟级）
├── train.csv                           # 日聚合训练集（1,112天）
├── test.csv                            # 日聚合测试集（330天）
├── prepare_data.py                     # 数据处理脚本
├── PLAN.md                             # 本计划书
│
├── features.py                         # 特征工程
├── dataset.py                          # 滑动窗口 + DataLoader
├── normalize.py                        # 归一化工具
│
├── models/
│   ├── __init__.py
│   ├── lstm.py                         # LSTM 模型
│   ├── transformer.py                  # Transformer 模型
│   ├── improved.py                     # HCMRF 自改进模型
│   ├── components/
│   │   ├── __init__.py
│   │   ├── hcm.py                      # Horizon Conditioning Module
│   │   ├── adaptive_patch.py           # Adaptive-Patch Transformer
│   │   └── drd.py                      # Dynamic Resolution Decoder
│   └── ablations/
│       ├── __init__.py
│       ├── without_hcm.py              # 消融 B: 去掉 HCM
│       ├── without_patch.py            # 消融 C: 去掉 Adaptive-Patch
│       ├── without_drd.py              # 消融 D: 去掉 DRD
│       ├── without_gate.py             # 消融 E: 去掉 channel_gate
│       └── without_shared.py           # 消融 F: 去掉共享编码器
│
├── train.py                            # 训练脚本（通用）
├── evaluate.py                         # 评估脚本
├── visualize.py                        # 可视化脚本
├── ablation.py                         # 消融实验主控脚本
│
├── config.py                           # 全局配置
├── main.py                             # 主入口
│
├── results/                            # 实验结果输出
│   ├── lstm_results.json
│   ├── transformer_results.json
│   ├── hcmrf_results.json              # 完整 HCMRF
│   └── ablation_results/               # 消融实验结果
│       ├── ablation_90d.json
│       └── ablation_365d.json
│
└── figures/                            # 图片输出
    ├── lstm_pred_90d.png
    ├── transformer_pred_90d.png
    ├── hcmrf_pred_90d.png
    ├── comparison_90d.png
    ├── lstm_pred_365d.png
    ├── transformer_pred_365d.png
    ├── hcmrf_pred_365d.png
    ├── comparison_365d.png
    ├── ablation_90d.png               # 消融对比柱状图
    ├── ablation_365d.png
    └── hcm_gate_heatmap.png            # HCM 门控权重热力图
```

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 训练数据仅 3 年，365d 预测样本少 | 长期预测过拟合 | step=7 减少重叠；正则化+dropout；降低模型容量 |
| Transformer/HCMRF 在小数据上过拟合 | 预测能力不如 LSTM，甚至不如线性模型 | 降低 d_model 和 layers；设置强早停；在报告中引用文献分析原因（课程以新颖性优先） |
| HCM 硬编码分支限制了模型灵活性 | 两个 horizon 的架构差异是预设而非学得的 | 在报告讨论部分诚实分析此设计取舍；保留通道门控作为可学习参数 |
| 天气数据为月粒度，日级辨识度低 | 同月每天天气特征相同，预测价值有限 | 在讨论中分析，天气特征仅贡献季节性背景信号 |
| 长时间预测误差累积 | 预测后期偏差大 | DRD 粗→精解码缓解；增强型多层精修提升效果 |
| 自改进模型新颖性不足 | 开放题得分低 | 已通过 HCMRF + 消融系统确保原理创新 |
| **消融实验计算量大** | 6个变体×2个horizon×5轮=60次训练 | 先跑 A(完整)，再优先跑 B/D(最可能出差异)，C/E/F 后置 |
| **消融结果可能不明显** | B/C/D/E/F 差异很小，难以下结论 | 报告差异 ±std + 统计检验；差异小本身也有分析价值（说明组件冗余） |
| **课程误解风险：HCM 用 Embedding 学 horizon 差异** | 原设计中 horizon_id 恒定的致命缺陷 | 已修正为硬编码分支 + nn.Parameter 门控，不再依赖 Embedding |

---

## 八、时间线

| 阶段 | 内容 | 预计工作量 |
|------|------|-----------|
| Phase 1 | 特征工程 + 数据管道搭建（含 step=7 滑动窗口） | 1 天 |
| Phase 2 | LSTM 模型实现 + 调参 + 实验（5轮） | 1-2 天 |
| Phase 3 | Transformer 模型实现 + 调参 + 实验（5轮） | 1-2 天 |
| Phase 3b | **统计基线：季节性朴素预测（去年同日值）** | 0.5 天 |
| Phase 4 | HCMRF 模型实现 + 调参 + 主实验（5轮） | 2 天 |
| Phase 4b | **消融实验（6个变体×2个horizon×5轮=60次训练）** | **2-3 天** |
| Phase 5 | 结果分析 + 可视化 + 消融图表 + 报告撰写 | 2 天 |
| Phase 6 | GitHub 上传 + 最终检查 | 0.5 天 |

**总预估：11-13 天**

---

*计划制定日期：2026-06-25*
*参考项目：su-Insight/power-prediction, Fenriel/IHEPC*
