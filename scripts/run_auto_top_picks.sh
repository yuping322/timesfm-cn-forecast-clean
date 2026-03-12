#!/bin/bash
set -euo pipefail

# Scripts to automatically run daily predictions for the best performing group
# Usage: bash scripts/run_auto_top_picks.sh

cd "$(dirname "$0")/.."

# Find the most recent eval_all_groups task directory
LATEST_EVAL_DIR=$(ls -td data/tasks/eval_all_groups_* 2>/dev/null | head -n 1 || true)

if [ -z "${LATEST_EVAL_DIR}" ]; then
  echo "Error: No data/tasks/eval_all_groups_* directories found."
  echo "Please run scripts/run_all_groups_eval.sh first."
  exit 1
fi

BEST_GROUP_FILE="${LATEST_EVAL_DIR}/best_group.txt"

if [ ! -f "${BEST_GROUP_FILE}" ]; then
  echo "Error: ${BEST_GROUP_FILE} not found in the latest evaluation directory."
  exit 1
fi

BEST_GROUP=$(cat "${BEST_GROUP_FILE}")
echo "=== Found Best Group: ${BEST_GROUP} ==="

ADAPTER_PATH="${LATEST_EVAL_DIR}/groups/${BEST_GROUP}/adapter.pth"
if [ ! -f "${ADAPTER_PATH}" ]; then
  echo "Error: Adapter for ${BEST_GROUP} not found at ${ADAPTER_PATH}."
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TASK_DIR="data/tasks/auto_picks_${TIMESTAMP}"
export OUTPUT_DIR="${TASK_DIR}/predictions"
mkdir -p "${OUTPUT_DIR}"

echo "=== Running Auto Top Picks for ${BEST_GROUP} ==="
bash scripts/run_daily_predict.sh "${BEST_GROUP}" "${ADAPTER_PATH}" 1 60
