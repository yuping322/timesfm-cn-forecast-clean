#!/bin/bash
set -euo pipefail

# Examples:
# 1) Basic run (all groups, skip existing, auto-analyze):
#    bash scripts/run_all_groups_eval.sh
# 2) Change core params:
#    FEATURE_SET=full TRAIN_DAYS=60 HORIZON=1 CONTEXT_LEN=60 TEST_DAYS=20 MIN_DAYS=1000 \
#    bash scripts/run_all_groups_eval.sh
# 3) With date filter and output dir:
#    OUTPUT_DIR=data/research START_DATE=2015-01-01 END_DATE=2025-12-31 \
#    CONTEXT_LENGTHS=30,60,90 \
#    bash scripts/run_all_groups_eval.sh
# 4) Re-run all groups (do not skip existing) and skip analyze:
#    SKIP_EXISTING=0 ANALYZE=0 bash scripts/run_all_groups_eval.sh

cd "$(dirname "$0")/.."

MARKET_DUCKDB="${MARKET_DUCKDB:-data/market.duckdb}"
INDEX_DUCKDB="${INDEX_DUCKDB:-data/index_market.duckdb}"
OUTPUT_DIR="${OUTPUT_DIR:-data/research}"

FEATURE_SET="${FEATURE_SET:-full}"
TRAIN_DAYS="${TRAIN_DAYS:-60}"
HORIZON="${HORIZON:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-60}"
TEST_DAYS="${TEST_DAYS:-20}"
MIN_DAYS="${MIN_DAYS:-1000}"

SKIP_EXISTING="${SKIP_EXISTING:-1}"
ANALYZE="${ANALYZE:-1}"

GROUPS=$(
  INDEX_DUCKDB="${INDEX_DUCKDB}" python - <<'PY'
import os
import sys
from pathlib import Path
root = Path.cwd()
src = root / "src"
sys.path.insert(0, str(src))
from timesfm_cn_forecast.universe.storage import list_all_symbols
duckdb_path = os.environ.get("INDEX_DUCKDB") or str(root / "data" / "index_market.duckdb")
df = list_all_symbols(duckdb_path)
print(" ".join(df["index_symbol"].tolist()))
PY
)

for group in $GROUPS; do
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "${OUTPUT_DIR}/${group}/results.csv" ]; then
    echo "Skipping ${group} (results.csv exists)"
    continue
  fi
  echo "Running group: ${group}"
  MARKET_DUCKDB="${MARKET_DUCKDB}" \
  INDEX_DUCKDB="${INDEX_DUCKDB}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  START_DATE="${START_DATE:-}" \
  END_DATE="${END_DATE:-}" \
  CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-}" \
  bash scripts/run_one_group_eval.sh \
    "${group}" \
    "${FEATURE_SET}" \
    "${TRAIN_DAYS}" \
    "${HORIZON}" \
    "${CONTEXT_LEN}" \
    "${TEST_DAYS}" \
    "${MIN_DAYS}"
done

if [ "$ANALYZE" = "1" ]; then
  python scripts/analyze_group_results.py --input-dir "${OUTPUT_DIR}"
fi
