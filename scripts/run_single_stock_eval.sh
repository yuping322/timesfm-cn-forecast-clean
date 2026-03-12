#!/bin/bash
set -euo pipefail

# Scripts to run evaluation on a single stock using local providers and adapter training
# Usage: bash scripts/run_single_stock_eval.sh <symbol> <train_days> [horizon] [feature_set] [test_days]
# Example: bash scripts/run_single_stock_eval.sh 002594 60

cd "$(dirname "$0")/.."

SYMBOL="${1:-}"
if [ -z "$SYMBOL" ]; then
  echo "Usage: $0 <symbol> [train_days] [horizon] [feature_set] [test_days]"
  echo "Example: $0 002594 60"
  exit 1
fi

TRAIN_DAYS="${2:-60}"

HORIZON="${3:-1}"
FEATURE_SET="${4:-full}"
TEST_DAYS="${5:-20}"

# Ensure anaconda python is in PATH and PYTHONPATH is set
export PATH=/opt/anaconda3/bin:$PATH
export PYTHONPATH=src

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TASK_DIR="data/tasks/eval_single_${SYMBOL}_${TIMESTAMP}"

# Unified Output Directories inside the task folder
ADAPTER_DIR="${TASK_DIR}/adapters"
LOG_DIR="${TASK_DIR}/logs"
DATA_DIR="${TASK_DIR}/data"

mkdir -p "${ADAPTER_DIR}" "${LOG_DIR}" "${DATA_DIR}"

HISTORY_CSV="${DATA_DIR}/history.csv"
ADAPTER_PATH="${ADAPTER_DIR}/adapter.pth"
EVAL_LOG="${LOG_DIR}/eval.log"

echo "=== Stage 1: Data Preparation ==="
# Assuming start date 2015-01-01 to give enough history for context and training
python -m timesfm_cn_forecast.providers \
  --symbol "${SYMBOL}" \
  --start "2015-01-01" \
  --output "${HISTORY_CSV}"

echo "=== Stage 2: Adapter Training ==="
python -m timesfm_cn_forecast.finetuning \
  --stock-code "${SYMBOL}" \
  --data-path "${HISTORY_CSV}" \
  --output-path "${ADAPTER_PATH}" \
  --train-days "${TRAIN_DAYS}" \
  --horizon-len "${HORIZON}" \
  --feature-set "${FEATURE_SET}"

echo "=== Stage 3: Backtest Evaluation ==="
python -m timesfm_cn_forecast.backtest \
  --symbol "${SYMBOL}" \
  --provider local \
  --input-csv "${HISTORY_CSV}" \
  --test-days "${TEST_DAYS}" \
  --horizon "${HORIZON}" \
  --adapter "${ADAPTER_PATH}" > "${EVAL_LOG}" 2>&1

echo "Done! Backtest log saved to ${EVAL_LOG}"
cat "${EVAL_LOG}"

BEST_CTX=$(grep "最优上下文长度为" "${EVAL_LOG}" | sed -E 's/.*最优上下文长度为 ([0-9]+).*/\1/')

if [ -n "$BEST_CTX" ]; then
  echo ""
  echo "================================================================="
  echo "🌟 评估完成！您可以直接复制并运行以下命令，使用最优参数进行明日预测："
  echo ""
  echo "bash scripts/run_single_stock_predict.sh ${SYMBOL} ${ADAPTER_PATH} ${HORIZON} ${BEST_CTX}"
  echo "================================================================="
fi
