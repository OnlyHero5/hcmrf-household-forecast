# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Household electric power consumption forecasting — ML course assignment. Uses UCI "Individual Household Electric Power Consumption" dataset (French household, minute-level, 2006/12–2010/11) aggregated to daily granularity. Predicts future 90-day (short-term) and 365-day (long-term) active power (`Global_active_power`) from a 90-day lookback window.

Three models: **LSTM**, **Transformer**, and a custom **HCMRF** (Horizon-Conditioned Multi-Resolution Forecasting) with 5 ablation variants.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data (raw → daily aggregation + weather merge → train/test split)
python scripts/prepare_data.py

# Run all experiments (baselines + 3 models × 2 horizons × 5 seeds + 5 ablations)
bash scripts/run_all.sh

# Or directly via Python
python -m src.run

# Train a single model (interactive)
python -c "from src.config import Config; from src.train import train; train(Config(model_name='lstm', horizon=90, seed=42))"

# Evaluate a checkpoint
python -c "from src.config import Config; from src.evaluate import evaluate; print(evaluate(Config(model_name='lstm', horizon=90, seed=42), 'outputs/checkpoints/lstm_h90_s42.ckpt'))"

# Generate visualizations
python -c "from src import visualize"
```

## Architecture

```
src/
├── config.py              # dataclass Config — all hyperparams in one place
├── dataset.py             # PowerDataset (Dataset) — sliding window, stride > 1
├── features.py            # add_features() — sin/cos time, lag, rolling stats
│
├── models/
│   ├── lstm.py            # LSTMModel — 2-layer LSTM + linear head
│   ├── transformer.py     # TransformerModel — encoder + global avg pool + head
│   ├── hcmrf.py           # HCMRF — SharedEncoder → HCM → AdaptivePatch → Transformer → DRD
│   ├── hcmrf_ablations.py # 5 ablation variants (inherit from HCMRF)
│   └── components/
│       ├── hcm.py             # Horizon Conditioning Module (resolution compression + channel gating)
│       ├── adaptive_patch.py  # Adaptive-Patch Transformer (horizon-dependent patching)
│       └── drd.py             # Dynamic Resolution Decoder (coarse→upsample→refine for 365d)
│
├── system.py              # ForecastSystem (LightningModule) — train/val/test step orchestration
├── datamodule.py          # PowerDataModule (LightningDataModule) — data pipeline
├── train.py               # build_model() + train() — single training run
├── evaluate.py            # evaluate() — load checkpoint → test metrics
├── visualize.py           # plot_predictions(), plot_ablation()
└── run.py                 # run_all() — master script for all experiments
```

**Key design decisions:**
- **PyTorch Lightning** for training loop (early stopping, checkpointing, device management)
- **dataclass Config** for configuration — zero dependency, IDE-friendly
- `build_model(config)` in `train.py` is the model factory — add new models here
- All models implement `forward(x, horizon) → Tensor` for unified interface
- `nn.Module` files define pure computation graphs only; `ForecastSystem` handles training logic
- **No global state** — components communicate via constructor/method parameters

**HCMRF architecture flow:**
```
Input (B, 90, n_features) → Conv1D encoder → HCM (resolution + gate)
  → Adaptive-Patch Transformer (patch_size: 1 for 90d, 3 for 365d)
  → GlobalAvgPool → DRD (90d: direct; 365d: coarse 52-week → upsample → Conv1D refine)
```

**Ablation variants** (`hcmrf_ablations.py`):
| Variant | Removed Component |
|---------|------------------|
| `HCMRF_wo_HCM` | Horizon conditioning (no compression, no gating, fixed patch=1) |
| `HCMRF_wo_Patch` | Adaptive patching (fixed patch=1, keep HCM) |
| `HCMRF_wo_DRD` | Coarse-to-fine decoder (direct Dense(365)) |
| `HCMRF_wo_Gate` | Channel gating (keep resolution compression) |
| `HCMRF_wo_Shared` | Shared encoder (independent encoders per horizon) |

## Data Pipeline

1. `scripts/prepare_data.py` — raw minute-level `.txt` → daily aggregation + weather merge → `data/processed/{train,test}.csv`
2. `datamodule.py` → reads CSVs → `add_features()` → MinMaxScaler (fit on train only) → 80/20 train/val split → `PowerDataset` with configurable stride
3. Feature count after engineering: **24 features** (13 original + 11 engineered)
4. Target column is always column 0 (`Global_active_power`)

## Output Structure

```
outputs/
├── checkpoints/       # .ckpt files named {model}_h{horizon}_s{seed}.ckpt
├── results/           # summary.json (MSE/MAE mean±std per experiment)
├── figures/           # PNG comparison plots, ablation bar charts
└── lightning_logs/    # TensorBoard logs
```

## Design Principles (from IMPLEMENT.md)

- **Zero defensive programming** — no parameter validation, no None checks, no fallbacks
- **High cohesion, low coupling** — one file, one responsibility
- **nn.Module = pure forward pass**; LightningModule = training orchestration only
- **Explicit over implicit** — all hyperparams declared in `__init__` signatures, `self.save_hyperparameters()` for auto-serialization
- **Type annotations required** on all new code
