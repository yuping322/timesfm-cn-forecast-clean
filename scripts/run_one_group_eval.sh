#!/bin/bash
set -euo pipefail

# Examples:
# 1) Basic run (group only):
#    bash scripts/run_one_group_eval.sh CYBZ
# 2) With explicit params:
#    bash scripts/run_one_group_eval.sh ind_消费电子 full 60 1 60 20 1000
# 3) With env overrides:
#    MARKET_DUCKDB=data/market.duckdb INDEX_DUCKDB=data/index_market.duckdb \
#    OUTPUT_DIR=data/research FEATURE_SET=full TRAIN_DAYS=60 HORIZON=1 \
#    CONTEXT_LEN=60 TEST_DAYS=20 MIN_DAYS=1000 \
#    START_DATE=2015-01-01 END_DATE=2025-12-31 CONTEXT_LENGTHS=30,60,90 \
#    bash scripts/run_one_group_eval.sh CYBZ

cd "$(dirname "$0")/.."

GROUP="${1:-}"
if [ -z "$GROUP" ]; then
  echo "Usage: $0 <group> [feature_set] [train_days] [horizon] [context_len] [test_days] [min_days]"
  exit 1
fi

FEATURE_SET="${2:-full}"
TRAIN_DAYS="${3:-60}"
HORIZON="${4:-1}"
CONTEXT_LEN="${5:-60}"
TEST_DAYS="${6:-20}"
MIN_DAYS="${7:-1000}"

MARKET_DUCKDB="${MARKET_DUCKDB:-data/market.duckdb}"
INDEX_DUCKDB="${INDEX_DUCKDB:-data/index_market.duckdb}"
OUTPUT_DIR="${OUTPUT_DIR:-data/research}"

python scripts/run_group_eval.py \
  --group "${GROUP}" \
  --market-duckdb "${MARKET_DUCKDB}" \
  --index-duckdb "${INDEX_DUCKDB}" \
  --feature-set "${FEATURE_SET}" \
  --train-days "${TRAIN_DAYS}" \
  --horizon "${HORIZON}" \
  --context-len "${CONTEXT_LEN}" \
  --test-days "${TEST_DAYS}" \
  --min-days "${MIN_DAYS}" \
  --output-dir "${OUTPUT_DIR}" \
  ${START_DATE:+--start "$START_DATE"} \
  ${END_DATE:+--end "$END_DATE"} \
  ${CONTEXT_LENGTHS:+--context-lengths "$CONTEXT_LENGTHS"}
