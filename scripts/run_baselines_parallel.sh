#!/bin/bash
# 并行运行 LSTM 和 Transformer 基线实验
# 使用 12GB GPU，同时运行 4 个任务

set -e

# 确保输出目录存在
mkdir -p outputs/revised/checkpoints outputs/revised/results outputs/revised/figures

# 日志目录
LOGDIR="outputs/revised/logs"
mkdir -p "$LOGDIR"

echo "======================================"
echo "并行运行 LSTM 和 Transformer 基线实验"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "可用显存: ~12GB"
echo "======================================"

# 定义实验列表
MODELS=("lstm" "transformer")
HORIZONS=(90 365)
SEEDS=(42 123 456 789 2024)

PIDS=()
LOG_FILES=()

for model in "${MODELS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "启动: $model horizon=$horizon seed=$seed"
            python -c "
from src.config import Config
from src.train import train
from src.evaluate import evaluate

cfg = Config(model_name='$model', horizon=$horizon, seed=$seed)
print(f'训练 $model h=$horizon seed=$seed')
ckpt = train(cfg)
result = evaluate(cfg, ckpt)
print(f'$model h=$horizon seed=$seed: MSE={result[\"test/MSE\"]:.4f}, MAE={result[\"test/MAE\"]:.4f}')
" > "$LOGDIR/${model}_h${horizon}_s${seed}.log" 2>&1 &
            PIDS+=($!)
            LOG_FILES+=("$LOGDIR/${model}_h${horizon}_s${seed}.log")
        done
    done
done

echo ""
echo "已启动 ${#PIDS[@]} 个并行任务"
echo "等待完成..."
echo ""

# 等待所有任务完成
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    LOG=${LOG_FILES[$i]}
    if wait "$PID"; then
        echo "✅ 完成: $(basename "$LOG" .log)"
    else
        echo "❌ 失败: $(basename "$LOG" .log) (见 $LOG)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "======================================"
echo "运行完成! 成功: $(( ${#PIDS[@]} - FAILED )), 失败: $FAILED"
echo "======================================"

# 汇总结果
if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "=== 结果汇总 ==="
    python -c "
from src.run import run_all

# 只运行 LSTM 和 Transformer 的基线
import json
import os
import numpy as np
from pathlib import Path
from src.config import Config
from src.evaluate import evaluate
from src.run import seasonal_naive_baseline

SEEDS = [42, 123, 456, 789, 2024]
experiments = [
    ('lstm', 90), ('lstm', 365),
    ('transformer', 90), ('transformer', 365),
]

def latest_checkpoint(model_name, horizon, seed):
    candidates = list(Path('outputs/revised/checkpoints').glob(f'{model_name}_h{horizon}_s{seed}*.ckpt'))
    return str(max(candidates, key=lambda path: path.stat().st_mtime))

summary = {}

for horizon in [90, 365]:
    result = seasonal_naive_baseline('data/processed', Config().input_len, horizon)
    summary[f'seasonal_naive_h{horizon}'] = {'mean': result, 'std': {'MSE': 0.0, 'MAE': 0.0}}
    print(f'seasonal_naive h={horizon}: MSE={result[\"MSE\"]:.4f}, MAE={result[\"MAE\"]:.4f}')

for model_name, horizon in experiments:
    metrics = []
    for seed in SEEDS:
        cfg = Config(model_name=model_name, horizon=horizon, seed=seed)
        ckpt = latest_checkpoint(model_name, horizon, seed)
        result = evaluate(cfg, ckpt)
        metrics.append(result)
        print(f'{model_name} h={horizon} s={seed}: MSE={result[\"test/MSE\"]:.4f}, MAE={result[\"test/MAE\"]:.4f}')

    avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
    std = {k: float(np.std([m[k] for m in metrics], ddof=1)) for k in metrics[0]}
    summary[f'{model_name}_h{horizon}'] = {'mean': avg, 'std': std, 'runs': metrics}
    print(f'  → mean MSE={avg[\"test/MSE\"]:.4f} ± {std[\"test/MSE\"]:.4f}')

os.makedirs('outputs/revised/results', exist_ok=True)
with open('outputs/revised/results/summary_baselines.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\n结果已保存到 outputs/revised/results/summary_baselines.json')
"
fi
