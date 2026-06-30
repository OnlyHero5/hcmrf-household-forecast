#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Running all experiments ==="
python -m src.run
echo ""

echo "=== Generating figures ==="
python -c "from src.visualize import *; print('Visualization entry point ready')"
echo ""

echo "=== Done ==="
echo "Results: outputs/results/summary.json"
echo "Figures: outputs/figures/"
