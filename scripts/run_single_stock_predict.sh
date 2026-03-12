#!/bin/bash
set -euo pipefail

# Scripts to run daily prediction on a single evaluated stock using its trained adapter.
# Usage: bash scripts/run_single_stock_predict.sh <symbol> <adapter_path> [horizon] [context_len]
# Example: bash scripts/run_single_stock_predict.sh 002594 data/adapters/single_stock/002594/adapter_20260312.pth 1 60

cd "$(dirname "$0")/.."

SYMBOL="${1:-}"
ADAPTER="${2:-}"

if [ -z "$SYMBOL" ] || [ -z "$ADAPTER" ]; then
  echo "Usage: $0 <symbol> <adapter_path> [horizon] [context_len]"
  exit 1
fi

HORIZON="${3:-1}"
CONTEXT_LEN="${4:-60}"

MARKET_DUCKDB="${MARKET_DUCKDB:-data/market.duckdb}"
INDEX_DUCKDB="${INDEX_DUCKDB:-data/index_market.duckdb}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TASK_DIR="data/tasks/predict_single_${SYMBOL}_${TIMESTAMP}"
OUTPUT_DIR="${TASK_DIR}/predictions"

mkdir -p "${OUTPUT_DIR}"

export PATH=/opt/anaconda3/bin:$PATH
export PYTHONPATH=src

echo "=== Running Single Stock Prediction for ${SYMBOL} ==="
python -m timesfm_cn_forecast.cli \
  --provider "duckdb" \
  --symbol "${SYMBOL}" \
  --adapter "${ADAPTER}" \
  --duckdb-path "${MARKET_DUCKDB}" \
  --index-duckdb "${INDEX_DUCKDB}" \
  --horizon "${HORIZON}" \
  --context-length "${CONTEXT_LEN}" \
  --output-dir "${OUTPUT_DIR}"

echo "Prediction complete. Check ${OUTPUT_DIR} for results."
