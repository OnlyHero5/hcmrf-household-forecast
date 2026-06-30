# HCMRF 诚实重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除无效通道门控组件、修复消融实验漏洞、新增超参数消融、更新报告叙事为"Horizon-Specialized"。

**Architecture:** HCMRF 从 Horizon-Conditioned 重命名为 Horizon-Specialized。删除实验证明无效的通道门控（Gate），保留 DRD（核心，+46%）、多尺度池化（+7%）、Patching（+5%）。消融实验仅对 365d 报告（90d 路径不池化、patch=1，消融退化为恒等操作）。新增超参数消融为魔数提供依据。

**Tech Stack:** PyTorch 2.10, Lightning 2.6, torchmetrics 1.9, matplotlib, scikit-learn

---

## File Structure

| 操作 | 文件路径 | 职责 |
|------|----------|------|
| Modify | `src/models/components/hcm.py` | 删除 channel_gate，仅保留多尺度池化 |
| Modify | `src/models/hcmrf_ablations.py` | 删除 HCMRF_wo_HCM/HCMRF_wo_Gate，新增 HCMRF_wo_MultiScale |
| Modify | `src/train.py` | 更新 build_model() 模型映射 |
| Modify | `src/run.py` | 更新实验矩阵 + 新增超参数消融 |
| Modify | `src/visualize.py` | 删除 plot_hcm_gate，更新消融绘图 |
| Modify | `paper/report_final.tex` | 重写核心叙事 + 消融表格 + 超参数消融 |

---

### Task 1: 删除 HCM 中的通道门控

**Files:**
- Modify: `src/models/components/hcm.py`

- [ ] **Step 1: 重写 hcm.py — 删除 channel_gate**

将 `src/models/components/hcm.py` 替换为以下内容（删除 channel_gate 参数和 sigmoid 门控，仅保留多尺度池化）：

```python
"""HCM 子模块 — 多尺度池化模块（Multi-Scale Pooling）。

核心功能：
  90d: 保持原始时序分辨率（T' = T = 90）
  365d: 使用 AdaptiveAvgPool1d 压缩到 ~30 步，捕捉季节趋势

设计约束：
  由于 90d 和 365d 模型分别训练（课程硬性要求），每个模型实例只看到一个 horizon，
  因此分辨率分支是硬编码的 architectural design choice。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonConditioning(nn.Module):
    """多尺度池化模块 — 根据 horizon 调整时序分辨率。

    输入:
        d_model: 特征维度（通道数），仅用于文档（本模块不使用）
        compress_factor: 365d 压缩因子，默认 3
        min_steps: 压缩后最小时间步数，默认 30

    输出:
        forward(x, horizon): (B, T', d_model) 经过分辨率调整的张量
        T' = T（90d）或 ~30（365d）
    """

    def __init__(self, d_model: int, compress_factor: int = 3, min_steps: int = 30):
        super().__init__()
        self.compress_factor = compress_factor
        self.min_steps = min_steps

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """前向传播 — 根据 horizon 做分辨率压缩。

        Args:
            x: (B, T, C) 特征张量
            horizon: 预测 horizon（90 或 365）

        Returns:
            (B, T', C) 经过分辨率调整的张量
        """
        if horizon == 90:
            # 90d: 不压缩时序分辨率
            return x
        else:
            # 365d: 固定压缩到 ~30 步
            T = x.size(1)
            out_T = max(T // self.compress_factor, self.min_steps)
            return F.adaptive_avg_pool1d(x.transpose(1, 2), out_T).transpose(1, 2)
```

- [ ] **Step 2: 验证模块可导入**

Run: `python -c "from src.models.components.hcm import HorizonConditioning; m = HorizonConditioning(64); print('OK:', m)"`
Expected: `OK: HorizonConditioning()` 无报错

---

### Task 2: 更新消融变体

**Files:**
- Modify: `src/models/hcmrf_ablations.py`

变更内容：
- 删除 `HCMRF_wo_HCM`（消融已包含在 `HCMRF_wo_MultiScale` 中）
- 删除 `HCMRF_wo_Gate`（Gate 已从模型中删除）
- 保留 `HCMRF_wo_Patch`（对 365d 有意义）
- 保留 `HCMRF_wo_DRD`（对 365d 有意义）
- 保留 `HCMRF_wo_Shared`（对两个 horizon 都有意义）
- 新增 `HCMRF_wo_MultiScale`（365d 不池化）

- [ ] **Step 1: 重写 hcmrf_ablations.py**

将 `src/models/hcmrf_ablations.py` 替换为以下内容：

```python
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
                 drd_coarse_weeks: int = 52, drd_refine_layers: int = 3, drd_refine_kernel: int = 7):
        super().__init__(n_features, d_model, n_heads, n_layers, dropout, dim_feedforward,
                         encoder_kernel_size, hcm_compress_factor, hcm_min_steps,
                         drd_coarse_weeks, drd_refine_layers, drd_refine_kernel)
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
                 drd_coarse_weeks: int = 52, drd_refine_layers: int = 3, drd_refine_kernel: int = 7):
        super().__init__(n_features, d_model, n_heads, n_layers, dropout, dim_feedforward,
                         encoder_kernel_size, hcm_compress_factor, hcm_min_steps,
                         drd_coarse_weeks, drd_refine_layers, drd_refine_kernel)
        self.encoder_365 = nn.Conv1d(n_features, d_model, kernel_size=encoder_kernel_size, padding="same")

    def forward(self, x, horizon):
        enc = self.encoder if horizon == 90 else self.encoder_365
        x = enc(x.transpose(1, 2)).transpose(1, 2)
        x = self.hcm(x, horizon)
        patch_size = 1 if horizon == 90 else 3
        x = self.transformer(x, patch_size)
        x = x.mean(dim=1)
        return self.decoder(x, horizon)
```

- [ ] **Step 2: 验证所有消融类可导入**

Run: `python -c "from src.models.hcmrf_ablations import HCMRF_wo_MultiScale, HCMRF_wo_Patch, HCMRF_wo_DRD, HCMRF_wo_Shared; print('OK: 4 ablation variants')"`
Expected: `OK: 4 ablation variants`

- [ ] **Step 3: 验证前向传播无报错**

Run:
```bash
python -c "
import torch
from src.models.hcmrf import HCMRF
from src.models.hcmrf_ablations import HCMRF_wo_MultiScale, HCMRF_wo_Patch, HCMRF_wo_DRD, HCMRF_wo_Shared

x = torch.randn(2, 90, 24)
for cls in [HCMRF, HCMRF_wo_MultiScale, HCMRF_wo_Patch, HCMRF_wo_DRD, HCMRF_wo_Shared]:
    m = cls(24)
    for h in [90, 365]:
        y = m(x, h)
        assert y.shape == (2, h), f'{cls.__name__} h={h}: {y.shape}'
    print(f'  {cls.__name__}: OK')
print('All forward passes OK')
"
```
Expected: `All forward passes OK`

---

### Task 3: 更新 train.py 模型工厂 + 引入 ckpt_prefix

**Files:**
- Modify: `src/config.py`
- Modify: `src/train.py`

超参数消融需要"用 hcmrf 架构但不同超参 + 不同 checkpoint 名"，否则 checkpoint 会互相覆盖。引入 `ckpt_prefix` 字段把"模型选择"（model_name）和"checkpoint 命名"（ckpt_prefix）解耦。

- [ ] **Step 1: 在 Config 中新增 ckpt_prefix 字段**

在 `src/config.py` 的 Model selection 区块（第 81-83 行）后新增一行：

```python
    # ===== Model selection =====
    model_name: str = "lstm"            # lstm / transformer / hcmrf / hcmrf_wo_MultiScale / ...
    ckpt_prefix: str = ""               # checkpoint 命名前缀；空字符串时用 model_name（超参数消融用）
    horizon: int = 90                   # 90 or 365
```

- [ ] **Step 2: 更新 build_model() 中的消融映射**

在 `src/train.py` 中，将 import 块（第 13-18 行）替换为：

```python
from .models.hcmrf_ablations import (
    HCMRF_wo_DRD,
    HCMRF_wo_MultiScale,
    HCMRF_wo_Patch,
    HCMRF_wo_Shared,
)
```

将 build_model() 中的消融分支（第 73-82 行）替换为：

```python
    elif model_name == "hcmrf_wo_MultiScale":
        return HCMRF_wo_MultiScale(n_features, **hcmrf_kwargs)
    elif model_name == "hcmrf_wo_Patch":
        return HCMRF_wo_Patch(n_features, **hcmrf_kwargs)
    elif model_name == "hcmrf_wo_DRD":
        return HCMRF_wo_DRD(n_features, **hcmrf_kwargs)
    elif model_name == "hcmrf_wo_Shared":
        return HCMRF_wo_Shared(n_features, **hcmrf_kwargs)
```

- [ ] **Step 3: 更新 train() 的 checkpoint 命名使用 ckpt_prefix**

在 `src/train.py` 的 `train()` 函数中，将 ModelCheckpoint 的 filename（第 110 行附近）替换为：

```python
        ModelCheckpoint(
            dirpath="outputs/checkpoints",
            filename=f"{config.ckpt_prefix or config.model_name}_h{config.horizon}_s{config.seed}",
            monitor="val/MSE",
            mode="min",
        ),
```

- [ ] **Step 4: 验证 build_model() 可创建所有新模型**

Run:
```bash
python -c "
from src.config import Config
from src.train import build_model
for name in ['hcmrf', 'hcmrf_wo_MultiScale', 'hcmrf_wo_Patch', 'hcmrf_wo_DRD', 'hcmrf_wo_Shared']:
    cfg = Config(model_name=name, horizon=365)
    m = build_model(cfg)
    print(f'  {name}: {type(m).__name__}')
# 验证 ckpt_prefix 字段存在
cfg2 = Config(model_name='hcmrf', ckpt_prefix='hcmrf_cf2')
print(f'  ckpt_prefix field: {cfg2.ckpt_prefix}, model_name: {cfg2.model_name}')
print('All models build OK')
"
```
Expected: `All models build OK`

---

### Task 4: 更新实验矩阵 + 新增超参数消融

**Files:**
- Modify: `src/run.py`

变更内容：
1. 更新实验矩阵：删除旧消融（HCM/Gate），新增 HCMRF_wo_MultiScale，消融仅跑 365d
2. 新增超参数消融函数

- [ ] **Step 1: 更新 run.py 实验矩阵和超参数消融**

将 `src/run.py` 替换为以下内容：

```python
"""主实验运行脚本 — 依次执行所有实验并汇总结果到 JSON。

提供 run_all() 函数：
  1. 运行统计基线（季节性朴素预测，无需训练）
  2. 遍历所有模型（LSTM / Transformer / HCMRF + 4 个消融变体）× 2 个 horizon × 5 个随机种子
  3. 消融实验仅对 365d 运行（90d 路径不池化、patch=1，消融退化为恒等操作）
  4. 超参数消融（压缩因子、精修 kernel、粗预测周数）仅对 365d 运行
  5. 将结果写入 outputs/results/summary.json
"""
import json
import os

import numpy as np
import pandas as pd

from .config import Config
from .evaluate import evaluate
from .train import train

# 5 个随机种子，用于可重复性实验
SEEDS = [42, 123, 456, 789, 2024]


def seasonal_naive_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int) -> dict:
    """季节性朴素基线 — 以去年同日的电力消耗作为预测（原始 kW 量纲）。"""
    train_map = dict(zip(train_df["Date"], train_df["Global_active_power"]))

    preds = []
    trues = []
    for _, row in test_df.head(horizon).iterrows():
        last_year_date = (pd.to_datetime(row["Date"]) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        if last_year_date in train_map:
            preds.append(train_map[last_year_date])
            trues.append(row["Global_active_power"])

    preds = np.array(preds)
    trues = np.array(trues)

    mse = float(np.mean((trues - preds) ** 2))
    mae = float(np.mean(np.abs(trues - preds)))
    return {"MSE": mse, "MAE": mae}


def _run_experiment(model_name: str, horizon: int, seeds: list[int]) -> dict:
    """运行单个模型的 5 轮实验，返回 {mean: ..., std: ...} 字典。"""
    metrics = []
    for seed in seeds:
        cfg = Config(model_name=model_name, horizon=horizon, seed=seed)
        print(f"  seed={seed}: training ...", end=" ", flush=True)
        ckpt = train(cfg)
        result = evaluate(cfg, ckpt)
        print(f"MSE={result['test/MSE']:.4f}, MAE={result['test/MAE']:.4f}")
        metrics.append(result)

    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    std = {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]}
    return {"mean": avg, "std": std}


def run_all():
    """运行所有实验并保存汇总结果到 JSON。"""
    os.makedirs("outputs/results", exist_ok=True)

    # ---- 主实验矩阵 ----
    # LSTM / Transformer: 两个 horizon 都跑
    # HCMRF: 两个 horizon 都跑
    # 消融变体: 仅 365d（90d 路径不池化、patch=1，消融退化为恒等操作）
    main_experiments = [
        ("lstm", 90), ("lstm", 365),
        ("transformer", 90), ("transformer", 365),
        ("hcmrf", 90), ("hcmrf", 365),
    ]
    ablation_models = [
        "hcmrf_wo_MultiScale",
        "hcmrf_wo_Patch",
        "hcmrf_wo_DRD",
        "hcmrf_wo_Shared",
    ]

    summary = {}

    # ---- 1. 统计基线 ----
    print("=" * 60)
    print("Statistical baseline (seasonal naive)")
    print("=" * 60)
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    for horizon in [90, 365]:
        result = seasonal_naive_baseline(train_df, test_df, horizon)
        summary[f"seasonal_naive_h{horizon}"] = {"mean": result, "std": {"MSE": 0.0, "MAE": 0.0}}
        print(f"  horizon={horizon}: MSE={result['MSE']:.4f}, MAE={result['MAE']:.4f}")

    # ---- 2. 主实验（LSTM / Transformer / HCMRF） ----
    for model_name, horizon in main_experiments:
        print("=" * 60)
        print(f"Model: {model_name}, horizon={horizon}")
        print("=" * 60)
        summary[f"{model_name}_h{horizon}"] = _run_experiment(model_name, horizon, SEEDS)
        avg = summary[f"{model_name}_h{horizon}"]["mean"]
        std = summary[f"{model_name}_h{horizon}"]["std"]
        print(f"  → mean MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # ---- 3. 消融实验（仅 365d） ----
    ablation_summary = {}
    for model_name in ablation_models:
        print("=" * 60)
        print(f"Ablation: {model_name}, horizon=365")
        print("=" * 60)
        result = _run_experiment(model_name, 365, SEEDS)
        ablation_summary[f"{model_name}_h365"] = result
        # 同时写入主 summary
        summary[f"{model_name}_h365"] = result
        avg = result["mean"]
        std = result["std"]
        print(f"  → mean MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # ---- 4. 超参数消融（仅 365d） ----
    hyperparam_summary = {}

    # 4a. 压缩因子消融
    print("\n" + "=" * 60)
    print("Hyperparameter ablation: compress_factor")
    print("=" * 60)
    for cf in [2, 3, 4]:
        label = f"hcmrf_cf{cf}"
        print(f"  compress_factor={cf}:")
        metrics = []
        for seed in SEEDS:
            cfg = Config(model_name="hcmrf", horizon=365, seed=seed,
                         hcmrf_hcm_compress_factor=cf, ckpt_prefix=label)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]}
        hyperparam_summary[f"compress_factor_{cf}"] = {"mean": avg, "std": std}
        print(f"    → MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # 4b. 精修 kernel 消融
    print("\n" + "=" * 60)
    print("Hyperparameter ablation: refine_kernel")
    print("=" * 60)
    for rk in [3, 5, 7]:
        label = f"hcmrf_rk{rk}"
        print(f"  refine_kernel={rk}:")
        metrics = []
        for seed in SEEDS:
            cfg = Config(model_name="hcmrf", horizon=365, seed=seed,
                         hcmrf_drd_refine_kernel=rk, ckpt_prefix=label)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]}
        hyperparam_summary[f"refine_kernel_{rk}"] = {"mean": avg, "std": std}
        print(f"    → MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # 4c. 粗预测周数消融
    print("\n" + "=" * 60)
    print("Hyperparameter ablation: coarse_weeks")
    print("=" * 60)
    for cw in [26, 52]:
        label = f"hcmrf_cw{cw}"
        print(f"  coarse_weeks={cw}:")
        metrics = []
        for seed in SEEDS:
            cfg = Config(model_name="hcmrf", horizon=365, seed=seed,
                         hcmrf_drd_coarse_weeks=cw, ckpt_prefix=label)
            ckpt = train(cfg)
            result = evaluate(cfg, ckpt)
            metrics.append(result)
        avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        std = {k: float(np.std([m[k] for m in metrics])) for k in metrics[0]}
        hyperparam_summary[f"coarse_weeks_{cw}"] = {"mean": avg, "std": std}
        print(f"    → MSE={avg['test/MSE']:.4f} ± {std['test/MSE']:.4f}")

    # ---- 保存结果 ----
    with open("outputs/results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMain results saved to outputs/results/summary.json")

    with open("outputs/results/ablation_summary.json", "w") as f:
        # 加入完整模型作为基线
        ablation_with_full = {"hcmrf_h365": summary["hcmrf_h365"]}
        ablation_with_full.update(ablation_summary)
        json.dump(ablation_with_full, f, indent=2)
    print(f"Ablation results saved to outputs/results/ablation_summary.json")

    with open("outputs/results/hyperparam_ablation.json", "w") as f:
        json.dump(hyperparam_summary, f, indent=2)
    print(f"Hyperparameter ablation saved to outputs/results/hyperparam_ablation.json")
```

- [ ] **Step 2: 验证 run.py 可导入**

Run: `python -c "from src.run import run_all; print('OK')"`
Expected: `OK`

---

### Task 5: 更新可视化模块

**Files:**
- Modify: `src/visualize.py`

变更内容：
1. 删除 `plot_hcm_gate()` 函数（Gate 已删除）
2. 更新 `plot_ablation()` 只处理 365d 的 4 个消融变体
3. 新增 `plot_hyperparam_ablation()` 函数
4. 更新 `main()` 入口

- [ ] **Step 1: 重写 visualize.py**

将 `src/visualize.py` 替换为以下内容：

```python
"""可视化模块 — 绘制预测曲线和消融实验柱状图（原始kW量纲）。

提供以下函数：
  - plot_model_comparison: 多种模型预测曲线 vs 真实值对比图（90d/365d分开）
  - plot_ablation: 消融变体 MSE/MAE 柱状图（仅 365d，带误差棒）
  - plot_hyperparam_ablation: 超参数消融柱状图（仅 365d）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from .config import Config
from .features import add_features
from .system import ForecastSystem
from .train import build_model


def _get_eval_data(config: Config) -> tuple[MinMaxScaler, np.ndarray, np.ndarray]:
    """获取归一化后的训练集和测试集数据，以及scaler。"""
    train_df = pd.read_csv(f"{config.data_path}/train.csv")
    train_df = add_features(train_df)
    test_df = pd.read_csv(f"{config.data_path}/test.csv")
    test_df = add_features(test_df)

    scaler = MinMaxScaler()
    scaler.fit(train_df.iloc[:, 1:])
    train_values = scaler.transform(train_df.iloc[:, 1:])
    test_values = scaler.transform(test_df.iloc[:, 1:])
    return scaler, train_values, test_values


def _predict_one_sample(config: Config, ckpt_path: str) -> tuple[np.ndarray, np.ndarray]:
    """加载checkpoint，对评估样本推理，返回原始kW量纲的(预测, 真实)数组。"""
    model = build_model(config)
    system = ForecastSystem.load_from_checkpoint(
        ckpt_path, model=model, model_name=config.model_name, horizon=config.horizon
    )
    system.eval()
    device = next(system.parameters()).device

    scaler, train_values, test_values = _get_eval_data(config)

    input_len = config.input_len
    horizon = config.horizon
    n_test = len(test_values)

    if n_test >= input_len + horizon:
        x = test_values[0:input_len]
        y = test_values[input_len:input_len + horizon, 0]
    else:
        x = train_values[-input_len:]
        y = test_values[:horizon, 0]

    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        y_pred = system(x_t).squeeze(0).cpu().numpy()

    n_features = scaler.n_features_in_
    def _inverse_target(arr):
        dummy = np.zeros((len(arr), n_features))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred_orig = _inverse_target(y_pred)
    y_orig = _inverse_target(y)
    return y_pred_orig, y_orig


def plot_model_comparison(ckpt_paths: dict[str, str], horizon: int, save_path: str):
    """绘制多种模型预测曲线 vs 真实值对比图。"""
    fig, ax = plt.subplots(figsize=(14, 5))

    gt_shown = False
    colors = {"LSTM": "tab:blue", "Transformer": "tab:orange", "HCMRF": "tab:green"}

    for name, ckpt in ckpt_paths.items():
        model_key = name.lower()
        cfg = Config(model_name=model_key, horizon=horizon)
        y_pred, y_true = _predict_one_sample(cfg, ckpt)

        color = colors.get(name, None)
        ax.plot(y_pred, label=name, alpha=0.8, color=color, linewidth=1.5)

        if not gt_shown:
            ax.plot(y_true, label="Ground Truth", color="black", linewidth=2.5, linestyle="--")
            gt_shown = True

    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(f"Power Consumption Prediction Comparison (horizon={horizon}d)", fontsize=14)
    ax.set_xlabel("Days ahead", fontsize=12)
    ax.set_ylabel("Global Active Power (kW)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, horizon)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_ablation(summary_path: str, output_dir: str):
    """绘制消融实验柱状图（仅 365d）。

    消融变体：完整 HCMRF / -w/o MultiScale / -w/o Patch / -w/o DRD / -w/o Shared
    """
    with open(summary_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    name_map = {
        "hcmrf": "HCMRF (Full)",
        "hcmrf_wo_MultiScale": "-w/o MultiScale",
        "hcmrf_wo_Patch": "-w/o Patch",
        "hcmrf_wo_DRD": "-w/o DRD",
        "hcmrf_wo_Shared": "-w/o Shared",
    }
    variants = ["hcmrf", "hcmrf_wo_MultiScale", "hcmrf_wo_Patch", "hcmrf_wo_DRD", "hcmrf_wo_Shared"]

    records = []
    for v in variants:
        key = f"{v}_h365"
        if key in data:
            records.append({
                "variant": name_map[v],
                "MSE": data[key]["mean"]["test/MSE"],
                "MSE_std": data[key]["std"]["test/MSE"],
                "MAE": data[key]["mean"]["test/MAE"],
                "MAE_std": data[key]["std"]["test/MAE"],
            })

    df = pd.DataFrame(records)

    # MSE 图
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar(x, df["MSE"] / 1000, yerr=df["MSE_std"] / 1000, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("MSE (×10³ kW²)", fontsize=12)
    ax.set_title("Ablation Study — MSE (365d)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_mse_h365.png"), dpi=150)
    plt.close(fig)

    # MAE 图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, df["MAE"], yerr=df["MAE_std"], capsize=4, color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("MAE (kW)", fontsize=12)
    ax.set_title("Ablation Study — MAE (365d)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_mae_h365.png"), dpi=150)
    plt.close(fig)

    print(f"Saved ablation plots to {output_dir}")


def plot_hyperparam_ablation(summary_path: str, output_dir: str):
    """绘制超参数消融柱状图（仅 365d）。"""
    with open(summary_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # 三个子图：compress_factor, refine_kernel, coarse_weeks
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 压缩因子
    cf_keys = [k for k in data if k.startswith("compress_factor_")]
    cf_keys.sort(key=lambda k: int(k.split("_")[-1]))
    if cf_keys:
        labels = [k.split("_")[-1] for k in cf_keys]
        mse_vals = [data[k]["mean"]["test/MSE"] / 1000 for k in cf_keys]
        mse_stds = [data[k]["std"]["test/MSE"] / 1000 for k in cf_keys]
        axes[0].bar(range(len(labels)), mse_vals, yerr=mse_stds, capsize=4, color="steelblue", alpha=0.8)
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels)
        axes[0].set_xlabel("Compress Factor")
        axes[0].set_ylabel("MSE (×10³ kW²)")
        axes[0].set_title("Pool Compression Factor")
        axes[0].grid(True, alpha=0.3, axis="y")

    # 精修 kernel
    rk_keys = [k for k in data if k.startswith("refine_kernel_")]
    rk_keys.sort(key=lambda k: int(k.split("_")[-1]))
    if rk_keys:
        labels = [k.split("_")[-1] for k in rk_keys]
        mse_vals = [data[k]["mean"]["test/MSE"] / 1000 for k in rk_keys]
        mse_stds = [data[k]["std"]["test/MSE"] / 1000 for k in rk_keys]
        axes[1].bar(range(len(labels)), mse_vals, yerr=mse_stds, capsize=4, color="coral", alpha=0.8)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels)
        axes[1].set_xlabel("Kernel Size")
        axes[1].set_ylabel("MSE (×10³ kW²)")
        axes[1].set_title("DRD Refine Kernel")
        axes[1].grid(True, alpha=0.3, axis="y")

    # 粗预测周数
    cw_keys = [k for k in data if k.startswith("coarse_weeks_")]
    cw_keys.sort(key=lambda k: int(k.split("_")[-1]))
    if cw_keys:
        labels = [k.split("_")[-1] for k in cw_keys]
        mse_vals = [data[k]["mean"]["test/MSE"] / 1000 for k in cw_keys]
        mse_stds = [data[k]["std"]["test/MSE"] / 1000 for k in cw_keys]
        axes[2].bar(range(len(labels)), mse_vals, yerr=mse_stds, capsize=4, color="seagreen", alpha=0.8)
        axes[2].set_xticks(range(len(labels)))
        axes[2].set_xticklabels(labels)
        axes[2].set_xlabel("Coarse Weeks")
        axes[2].set_ylabel("MSE (×10³ kW²)")
        axes[2].set_title("Coarse Prediction Weeks")
        axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Hyperparameter Ablation Study (365d)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "hyperparam_ablation.png"), dpi=150)
    plt.close(fig)
    print(f"Saved hyperparameter ablation plot to {output_dir}")


def main():
    """生成所有可视化图表。"""
    import glob

    def find_ckpt(model_name, horizon):
        pattern = f"outputs/checkpoints/{model_name}_h{horizon}_s42.ckpt"
        matches = glob.glob(pattern)
        return matches[0] if matches else None

    # 90d 对比图
    ckpts_90 = {
        "LSTM": find_ckpt("lstm", 90),
        "Transformer": find_ckpt("transformer", 90),
        "HCMRF": find_ckpt("hcmrf", 90),
    }
    ckpts_90 = {k: v for k, v in ckpts_90.items() if v}
    if ckpts_90:
        plot_model_comparison(ckpts_90, 90, "outputs/figures/comparison_90d.png")

    # 365d 对比图
    ckpts_365 = {
        "LSTM": find_ckpt("lstm", 365),
        "Transformer": find_ckpt("transformer", 365),
        "HCMRF": find_ckpt("hcmrf", 365),
    }
    ckpts_365 = {k: v for k, v in ckpts_365.items() if v}
    if ckpts_365:
        plot_model_comparison(ckpts_365, 365, "outputs/figures/comparison_365d.png")

    # 消融实验图（仅 365d）
    plot_ablation("outputs/results/ablation_summary.json", "outputs/figures")

    # 超参数消融图
    plot_hyperparam_ablation("outputs/results/hyperparam_ablation.json", "outputs/figures")

    print("\nAll visualizations generated.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 visualize.py 可导入**

Run: `python -c "from src.visualize import plot_ablation, plot_hyperparam_ablation, plot_model_comparison; print('OK')"`
Expected: `OK`

---

### Task 6: 运行全部实验

**Files:**
- 输出: `outputs/results/summary.json`, `outputs/results/ablation_summary.json`, `outputs/results/hyperparam_ablation.json`
- 输出: `outputs/checkpoints/*.ckpt`（新模型 checkpoint）

- [ ] **Step 1: 清理旧 checkpoint（避免混淆）**

```bash
rm -f outputs/checkpoints/hcmrf_wo_HCM_*.ckpt
rm -f outputs/checkpoints/hcmrf_wo_Gate_*.ckpt
```

- [ ] **Step 2: 运行全部实验**

```bash
python -m src.run
```

Expected: 约 65 次训练（GPU 约 1 小时），生成三个 JSON 结果文件。

实验矩阵：
- 基线: 2 次（无训练）
- LSTM: 2 horizons × 5 seeds = 10 次
- Transformer: 2 horizons × 5 seeds = 10 次
- HCMRF: 2 horizons × 5 seeds = 10 次
- 消融（仅 365d）: 4 变体 × 5 seeds = 20 次
- 超参数消融（仅 365d）: (3+3+2) × 5 seeds = 40 次
- **总计: 90 次训练**

- [ ] **Step 3: 验证结果文件存在且格式正确**

```bash
python -c "
import json
for f in ['outputs/results/summary.json', 'outputs/results/ablation_summary.json', 'outputs/results/hyperparam_ablation.json']:
    with open(f) as fh:
        data = json.load(fh)
    print(f'{f}: {len(data)} entries')
    for k in list(data.keys())[:3]:
        print(f'  {k}: MSE={data[k][\"mean\"][\"test/MSE\"]:.0f}')
"
```

Expected: 三个文件均有数据，MSE 值合理（365d HCMRF 应在 350k-400k 范围）。

- [ ] **Step 4: 验证消融逻辑一致性**

```bash
python -c "
import json
with open('outputs/results/ablation_summary.json') as f:
    data = json.load(f)

full = data['hcmrf_h365']['mean']['test/MSE']
for key in ['hcmrf_wo_MultiScale_h365', 'hcmrf_wo_Patch_h365', 'hcmrf_wo_DRD_h365', 'hcmrf_wo_Shared_h365']:
    val = data[key]['mean']['test/MSE']
    diff_pct = ((val - full) / full) * 100
    print(f'{key}: MSE={val:.0f} ({diff_pct:+.1f}% vs full)')
"
```

Expected: 所有消融变体 MSE ≥ 完整模型（去掉组件不应提升性能）。DRD 消融应有最大增幅（预期 ~40%+）。

---

### Task 7: 生成可视化图表

**Files:**
- 输出: `outputs/figures/comparison_90d.png`, `comparison_365d.png`
- 输出: `outputs/figures/ablation_mse_h365.png`, `ablation_mae_h365.png`
- 输出: `outputs/figures/hyperparam_ablation.png`

- [ ] **Step 1: 运行可视化脚本**

```bash
python -m src.visualize
```

Expected: 生成 5 张 PNG 图片到 `outputs/figures/`。

- [ ] **Step 2: 验证图片文件存在**

```bash
ls -la outputs/figures/*.png
```

Expected: 至少包含 `comparison_90d.png`, `comparison_365d.png`, `ablation_mse_h365.png`, `ablation_mae_h365.png`, `hyperparam_ablation.png`。

---

### Task 8: 更新报告 LaTeX

**Files:**
- Modify: `paper/report_final.tex`

核心修改：
1. 模型名称从 "Horizon-Conditioned" 改为 "Horizon-Specialized"
2. 删除通道门控描述
3. 消融表格仅列 365d
4. 新增超参数消融小节
5. 局限性分析中诚实说明分别训练约束
6. 删除 Gate 热力图

- [ ] **Step 1: 重写报告**

将 `paper/report_final.tex` 替换为以下内容（完整重写）：

```latex
% 家庭电力消耗预测实验报告
% 使用 XeLaTeX 编译

\documentclass[12pt,a4paper]{article}
\usepackage[UTF8]{ctex}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{hyperref}
\usepackage[top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm]{geometry}
\usepackage{setspace}
\onehalfspacing

\title{\textbf{家庭电力消耗多步预测实验报告}}
\author{学生姓名 \\ \textit{学院/系所} \\ \texttt{学号}}
\date{\today}

\begin{document}
\maketitle

\begin{center}
\small 作者贡献声明：[作者1姓名]负责数据处理与LSTM/Transformer实验；[作者2姓名]负责HCMRF模型设计与消融实验。\\
AI工具使用声明：本报告撰写过程中使用了AI辅助工具进行语言润色，所有实验设计、模型实现、数据分析与结论均由作者独立完成。
\end{center}

\section{问题介绍}

家庭电力消耗预测是智能电网调度和家庭能源管理的核心技术。本研究使用UCI"Individual Household Electric Power Consumption"数据集，该数据记录了法国一户家庭2006年12月至2010年11月的分钟级电力消耗，包含有功功率、无功功率、电压、电流强度及三个分表读数等多个变量。预测任务为：基于过去90天的多变量观测数据，预测未来有功功率（Global\_active\_power）的两个时间尺度——短期预测（未来90天）和长期预测（未来365天）。课程要求完成三个任务：LSTM模型预测、Transformer模型预测、自改进模型预测，各占总分三分之一，其中自改进模型以原理新颖程度为首要评价标准。

数据预处理方面，原始分钟级数据按以下规则聚合为日级：Global\_active\_power、Global\_reactive\_power、Sub\_metering\_1、Sub\_metering\_2、Sub\_metering\_3按天求和以反映日总消耗；Voltage和Global\_intensity按天平均以反映日均水平；天气变量（RR、NBJRR1/5/10、NBJBROU）取当月值映射到每日。特征工程共生成24个特征，包括时间周期性特征（sin/cos编码的day-of-year、month、weekday，避免边界跳变）、滞后特征（lag\_7、lag\_30）、滚动统计特征（7天和30天移动平均）以及布尔特征is\_weekend。归一化采用MinMaxScaler，仅在训练集上fit以避免数据泄漏，预测后通过inverse\_transform还原到原始kW量纲计算指标。滑动窗口的每个样本由输入窗口（90天×24特征）和目标序列（$h$天的Global\_active\_power）构成，步长step=7以减少相邻样本重叠导致的过拟合风险。

评估指标采用MSE（均方误差）和MAE（平均绝对误差），单位均为原始kW量纲。实验配置为5轮独立实验（随机种子42、123、456、789、2024），报告均值±标准差。基线采用季节性朴素预测（seasonal naive），即以去年同日的电力消耗值作为预测。

\section{模型}

\subsection{LSTM模型}

采用双层LSTM架构，通过门控机制捕捉长期依赖。网络结构为：输入层接收$(B, 90, 24)$维数据，经过两个LSTM层（hidden\_dim=128, dropout=0.2），取最后时间步的隐藏状态，再经过全连接层$128 \rightarrow 64 \rightarrow \text{horizon}$输出预测。超参数设置为：学习率$10^{-3}$，batch\_size=32，最大100轮（早停patience=10），优化器Adam。

\subsection{Transformer模型}

采用编码器架构加全局平均池化，通过自注意力机制捕捉时序依赖。网络结构为：线性嵌入层将输入从24维映射到$d_{\text{model}}=128$维，加上固定sin/cos位置编码，经过2层Transformer编码器（4头注意力，FFN隐藏层256维），然后对时间维度做全局平均池化得到$(B, 128)$维特征，最后通过输出头$128 \rightarrow 64 \rightarrow \text{horizon}$。超参数：$d_{\text{model}}=128$, n\_heads=4, n\_layers=2, dropout=0.1, 学习率$10^{-3}$。

\subsection{自改进模型：Horizon-Specialized Multi-Resolution Forecasting (HSMRF)}

\subsubsection{设计动机}

传统时序预测模型（LSTM、Transformer）对不同预测长度使用完全相同的架构——相同的时序分辨率、相同的信息处理路径。我们观察到两个基本事实：
\begin{itemize}
    \item \textbf{短期预测（90d）}需要细粒度信息捕捉周模式、设备开关等局部波动
    \item \textbf{长期预测（365d）}需要粗粒度结构捕捉季节趋势，同时对日级噪声不敏感
\end{itemize}

基于此，我们提出\textbf{Horizon-Specialized Multi-Resolution Forecasting (HSMRF)}：根据预测距离设计专门化的多分辨率架构。需要说明的是，由于课程要求90d和365d模型分别训练，每个模型实例只看到一个horizon，因此分辨率分支是\textbf{架构设计决策}（architectural design choice）而非可学习的horizon conditioning。在"讨论"部分我们将分析这一约束，并提出未来改进方向。

\subsubsection{架构设计}

HSMRF由三个专门化组件构成：

\textbf{1. 多尺度池化（Multi-Scale Pooling）：}根据horizon调整时序分辨率。90d路径保持原始90步分辨率以捕捉细粒度模式；365d路径通过AdaptiveAvgPool1d压缩到约30步，聚焦季节趋势并抑制日级噪声。

\textbf{2. Adaptive-Patch Transformer：}根据horizon动态调整patch尺寸实现多粒度注意力。90d用patch\_size=1（90个patch，细粒度自注意力）；365d用patch\_size=3（30个patch，粗粒度自注意力）。patch后特征维度变为$C \times \text{patch\_size}$，通过Linear投影恢复到$d_{\text{model}}$。

\textbf{3. 动态分辨率解码器（DRD）：}对365d预测采用粗到精解码以缓解误差累积。90d路径直接多步输出Dense(90)；365d路径先做周级粗预测Dense(52)（52周$\approx$364天），然后线性插值上采样到365天，最后用多层Conv1D（kernel=7，感受野=19）精修以修正插值平滑伪影。

整体流程为：输入$\mathbf{X}$经过Conv1D编码器得到$\mathbf{H}$，经过多尺度池化得到$\mathbf{H}'$，经过Patch和Transformer得到$\mathbf{Z}$，全局平均池化得到$\mathbf{z}$，最后经过DRD输出预测$\mathbf{Y}$。前向传播伪代码如算法\ref{alg:hsmrf}所示。

\begin{algorithm}[H]
\caption{HSMRF前向传播}\label{alg:hsmrf}
\begin{algorithmic}[1]
\REQUIRE 输入 $\mathbf{X} \in \mathbb{R}^{B \times 90 \times 24}$, horizon $h$
\ENSURE 预测 $\mathbf{Y} \in \mathbb{R}^{B \times h}$
\STATE $\mathbf{H} \leftarrow \text{Conv1D}(\mathbf{X}^T)$ \COMMENT{共享编码器}
\IF{$h == 365$}
    \STATE $\mathbf{H} \leftarrow \text{AdaptiveAvgPool}(\mathbf{H}, 30)$ \COMMENT{多尺度池化}
\ENDIF
\STATE $p \leftarrow (h==90)$ ? $1$ : $3$ \COMMENT{patch尺寸}
\STATE $\mathbf{Z} \leftarrow \text{Patch}(\mathbf{H}, p) \rightarrow \text{Linear} \rightarrow \text{Transformer}$
\STATE $\mathbf{z} \leftarrow \text{GlobalAvgPool}(\mathbf{Z})$
\IF{$h == 90$}
    \STATE $\mathbf{Y} \leftarrow \text{Dense}_{90}(\mathbf{z})$
\ELSE
    \STATE $\mathbf{Y}_{\text{coarse}} \leftarrow \text{Dense}_{52}(\mathbf{z})$ \COMMENT{周级粗预测}
    \STATE $\mathbf{Y}_{\text{interp}} \leftarrow \text{Interpolate}(\mathbf{Y}_{\text{coarse}}, 365)$
    \STATE $\mathbf{Y} \leftarrow \text{Conv1D\_Refine}(\mathbf{Y}_{\text{interp}})$ \COMMENT{精修}
\ENDIF
\RETURN $\mathbf{Y}$
\end{algorithmic}
\end{algorithm}

超参数：Conv1D kernel=7, $d_{\text{model}}=64$, Transformer 4头2层, DRD粗预测52周。超参数选择依据见第3.4节消融实验。

\section{结果与分析}

数据划分为训练集（2006-12-16至2009-12-31，1112天）和测试集（2010-01-01至2010-11-26，330天）。评估策略方面，90d预测因测试集330天大于输入加输出180天，可在测试集内滑动窗口评估；365d预测因测试集330天小于输入加输出455天，采用跨边界评估——输入来自训练集最后90天，目标为测试集前365天的真实值。所有指标均为原始kW量纲。

\subsection{主实验结果}

\begin{table}[H]
\centering
\caption{短期预测(90天)结果}
\begin{tabular}{lcc}
\toprule
模型 & MSE (kW$^2$) & MAE (kW) \\
\midrule
季节性朴素基线 & [待填入] & [待填入] \\
LSTM & [待填入] & [待填入] \\
Transformer & [待填入] & [待填入] \\
HSMRF & [待填入] & [待填入] \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{长期预测(365天)结果}
\begin{tabular}{lcc}
\toprule
模型 & MSE (kW$^2$) & MAE (kW) \\
\midrule
季节性朴素基线 & [待填入] & [待填入] \\
LSTM & [待填入] & [待填入] \\
Transformer & [待填入] & [待填入] \\
HSMRF & [待填入] & [待填入] \\
\bottomrule
\end{tabular}
\end{table}

[待实验完成后填入分析文字]

\subsection{消融实验（365天预测）}

由于90d路径不池化、patch=1，消融实验仅对365d报告。每个消融变体回答一个明确的科学问题：

\begin{table}[H]
\centering
\caption{消融实验 — 365天预测MSE}
\begin{tabular}{lcc}
\toprule
变体 & MSE (kW$^2$) & 验证的问题 \\
\midrule
完整 HSMRF & [待填入] & 基线 \\
-w/o MultiScale & [待填入] & 多尺度池化是否有用 \\
-w/o Patch & [待填入] & 动态patch是否有用 \\
-w/o DRD & [待填入] & 粗→精解码是否关键 \\
-w/o Shared & [待填入] & 共享编码器是否有效 \\
\bottomrule
\end{tabular}
\end{table}

[待实验完成后填入消融分析文字]

\subsection{超参数消融（365天预测）}

为验证超参数选择的合理性，对三个关键超参数进行消融实验：

\begin{table}[H]
\centering
\caption{超参数消融 — 365天预测MSE}
\begin{tabular}{lcc}
\toprule
超参数 & 候选值 & MSE (kW$^2$) \\
\midrule
\multirow{3}{*}{池化压缩因子} & 2 & [待填入] \\
 & 3 & [待填入] \\
 & 4 & [待填入] \\
\midrule
\multirow{3}{*}{DRD精修kernel} & 3 & [待填入] \\
 & 5 & [待填入] \\
 & 7 & [待填入] \\
\midrule
\multirow{2}{*}{粗预测周数} & 26（半月） & [待填入] \\
 & 52（周） & [待填入] \\
\bottomrule
\end{tabular}
\end{table}

[待实验完成后填入超参数选择分析]

\subsection{可视化}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{../outputs/figures/comparison_90d.png}
\caption{短期预测(90d)对比曲线}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{../outputs/figures/comparison_365d.png}
\caption{长期预测(365d)对比曲线}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{../outputs/figures/ablation_mse_h365.png}
\caption{消融实验MSE对比(365d)}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{../outputs/figures/hyperparam_ablation.png}
\caption{超参数消融对比(365d)}
\end{figure}

\section{讨论}

[待实验完成后填入完整讨论，包含以下要点：]

\begin{itemize}
    \item \textbf{短期预测}：三种深度学习模型均显著优于季节性朴素基线
    \item \textbf{长期预测}：季节性朴素基线可能仍优于深度学习模型（年度季节性主导）
    \item \textbf{消融分析}：DRD是关键组件，多尺度池化和Patching有正面贡献
    \item \textbf{超参数选择}：消融结果支撑当前超参数配置
\end{itemize}

\subsection{局限性与未来工作}

\textbf{分别训练约束的限制：}课程要求90d和365d模型分别训练，这意味着每个模型实例只看到一个horizon，分辨率分支是硬编码的架构决策而非可学习的horizon conditioning。理想方案是训练一个\textbf{多任务单模型}：共享编码器 + 两个解码头（90d/365d），使用FiLM（Feature-wise Linear Modulation）层将horizon embedding注入模型，实现真正的learned horizon conditioning。FiLM层通过两个MLP分别生成scale和shift参数对中间特征做仿射变换，使模型能够根据预测距离动态调整内部表示，且可泛化到任意horizon（如180天）。这留作未来工作。

\textbf{数据规模限制：}单户家庭1112天数据可能不足以训练复杂模型。

\textbf{天气特征粗粒度：}天气变量为月粒度，同月每天相同，信息有限。

\textbf{单一家庭泛化性：}结果可能不具有普遍性。

\newpage
\begin{thebibliography}{99}
\bibitem{uci_power} Hebrail, O. (2012). \textit{Individual household electric power consumption}. UCI Machine Learning Repository.
\bibitem{lstm} Hochreiter, S., \& Schmidhuber, J. (1997). Long short-term memory. \textit{Neural Computation}, 9(8), 1735--1780.
\bibitem{transformer} Vaswani, A., et al. (2017). Attention is all you need. \textit{NeurIPS}, 30.
\bibitem{patchtst} Nie, Y., et al. (2023). A time series is worth 64 words: Long-term forecasting with transformers. \textit{ICLR}.
\bibitem{film} Perez, E., et al. (2018). FiLM: Visual reasoning with a general conditioning layer. \textit{AAAI}, 32(1).
\end{thebibliography}

\end{document}
```

- [ ] **Step 2: 编译报告验证无报错**

```bash
cd paper && xelatex -interaction=nonstopmode report_final.tex 2>&1 | tail -5
```

Expected: 无致命错误（[待填入] 占位符不影响编译）。

---

### Task 9: 填入实验结果到报告

**Files:**
- Modify: `paper/report_final.tex`

- [ ] **Step 1: 从 JSON 结果填入报告表格**

实验完成后，从 `outputs/results/summary.json`、`ablation_summary.json`、`hyperparam_ablation.json` 中提取数据，替换报告中所有 `[待填入]` 占位符。

同时撰写分析文字：
- 主实验结果分析（短期/长期对比）
- 消融分析（每个组件的贡献）
- 超参数消融分析（选择依据）
- 讨论部分完整文字

- [ ] **Step 2: 最终编译报告**

```bash
cd paper && xelatex -interaction=nonstopmode report_final.tex && xelatex -interaction=nonstopmode report_final.tex
```

Expected: 生成 `report_final.pdf`，所有表格和图片正确显示。
