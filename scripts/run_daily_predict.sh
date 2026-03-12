#!/bin/bash
set -euo pipefail

# Scripts to run daily prediction for a group or a list of symbols
# Usage: bash scripts/run_daily_predict.sh <group_name> <adapter_path> [horizon] [context_len]
# Example: bash scripts/run_daily_predict.sh small_fengzhi data/research/small_fengzhi/adapter.pth 1 60

cd "$(dirname "$0")/.."

GROUP="${1:-}"
ADAPTER="${2:-}"
if [ -z "$GROUP" ] || [ -z "$ADAPTER" ]; then
  echo "Usage: $0 <group_name> <adapter_path> [horizon] [context_len]"
  exit 1
fi

HORIZON="${3:-1}"
CONTEXT_LEN="${4:-60}"

MARKET_DUCKDB="${MARKET_DUCKDB:-data/market.duckdb}"
INDEX_DUCKDB="${INDEX_DUCKDB:-data/index_market.duckdb}"

if [ -z "${OUTPUT_DIR:-}" ] || [ "${OUTPUT_DIR:-}" == "data/research/daily_picks" ]; then
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  TASK_DIR="data/tasks/daily_predict_${GROUP}_${TIMESTAMP}"
  OUTPUT_DIR="${TASK_DIR}/predictions"
fi

mkdir -p "${OUTPUT_DIR}"

export PATH=/opt/anaconda3/bin:$PATH
export PYTHONPATH=src

echo "=== Running Daily Prediction for ${GROUP} ==="
python -m timesfm_cn_forecast.cli \
  --provider "duckdb" \
  --group "${GROUP}" \
  --adapter "${ADAPTER}" \
  --duckdb-path "${MARKET_DUCKDB}" \
  --index-duckdb "${INDEX_DUCKDB}" \
  --horizon "${HORIZON}" \
  --context-length "${CONTEXT_LEN}" \
  --output-dir "${OUTPUT_DIR}"

echo "Prediction complete. Check ${OUTPUT_DIR} for results."
