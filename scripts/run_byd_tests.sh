#!/bin/bash
set -e

# 切回项目根目录执行
cd "$(dirname "$0")/.."

echo "=== 第一阶段：数据底座准备 ==="
python -m timesfm_cn_forecast.providers --symbol 002594 --start "2015-01-01" --output data/research/history.csv
mkdir -p data/research/adapters

echo "=== 第二阶段：短窗口训练群 (Train Days: 60, Horizon: 1) ==="
python -m timesfm_cn_forecast.finetuning --stock-code 002594 --data-path data/research/history.csv --output-path data/research/adapters/byd_basic_3m_t1.pth --train-days 60 --horizon-len 1 --feature-set basic
python -m timesfm_cn_forecast.finetuning --stock-code 002594 --data-path data/research/history.csv --output-path data/research/adapters/byd_technical_3m_t1.pth --train-days 60 --horizon-len 1 --feature-set technical
python -m timesfm_cn_forecast.finetuning --stock-code 002594 --data-path data/research/history.csv --output-path data/research/adapters/byd_structural_3m_t1.pth --train-days 60 --horizon-len 1 --feature-set structural
python -m timesfm_cn_forecast.finetuning --stock-code 002594 --data-path data/research/history.csv --output-path data/research/adapters/byd_full_3m_t1.pth --train-days 60 --horizon-len 1 --feature-set full

echo "=== 第三阶段：短窗口 (T+1) 回测验证 ==="
python -m timesfm_cn_forecast.backtest --symbol 002594 --provider local --input-csv data/research/history.csv --test-days 20 --horizon 1 --adapter data/research/adapters/byd_basic_3m_t1.pth > data/research/adapters/eval_basic_3m_t1.log 2>&1
python -m timesfm_cn_forecast.backtest --symbol 002594 --provider local --input-csv data/research/history.csv --test-days 20 --horizon 1 --adapter data/research/adapters/byd_technical_3m_t1.pth > data/research/adapters/eval_technical_3m_t1.log 2>&1
python -m timesfm_cn_forecast.backtest --symbol 002594 --provider local --input-csv data/research/history.csv --test-days 20 --horizon 1 --adapter data/research/adapters/byd_structural_3m_t1.pth > data/research/adapters/eval_structural_3m_t1.log 2>&1
python -m timesfm_cn_forecast.backtest --symbol 002594 --provider local --input-csv data/research/history.csv --test-days 20 --horizon 1 --adapter data/research/adapters/byd_full_3m_t1.pth > data/research/adapters/eval_full_3m_t1.log 2>&1

echo "=== 第四阶段：中窗口训练群 (Train Days: 500, Horizon: 5) ==="
python -m timesfm_cn_forecast.finetuning --stock-code 002594 --data-path data/research/history.csv --output-path data/research/adapters/byd_basic_2y_t5.pth --train-days 500 --horizon-len 5 --feature-set basic
python -m timesfm_cn_forecast.finetuning --stock-code 002594 --data-path data/research/history.csv --output-path data/research/adapters/byd_technical_2y_t5.pth --train-days 500 --horizon-len 5 --feature-set technical
python -m timesfm_cn_forecast.finetuning --stock-code 002594 --data-path data/research/history.csv --output-path data/research/adapters/byd_structural_2y_t5.pth --train-days 500 --horizon-len 5 --feature-set structural
python -m timesfm_cn_forecast.finetuning --stock-code 002594 --data-path data/research/history.csv --output-path data/research/adapters/byd_full_2y_t5.pth --train-days 500 --horizon-len 5 --feature-set full

echo "=== 第五阶段：中窗口 (T+5) 回测验证 ==="
python -m timesfm_cn_forecast.backtest --symbol 002594 --provider local --input-csv data/research/history.csv --test-days 20 --horizon 5 --adapter data/research/adapters/byd_basic_2y_t5.pth > data/research/adapters/eval_basic_2y_t5.log 2>&1
python -m timesfm_cn_forecast.backtest --symbol 002594 --provider local --input-csv data/research/history.csv --test-days 20 --horizon 5 --adapter data/research/adapters/byd_technical_2y_t5.pth > data/research/adapters/eval_technical_2y_t5.log 2>&1
python -m timesfm_cn_forecast.backtest --symbol 002594 --provider local --input-csv data/research/history.csv --test-days 20 --horizon 5 --adapter data/research/adapters/byd_structural_2y_t5.pth > data/research/adapters/eval_structural_2y_t5.log 2>&1
python -m timesfm_cn_forecast.backtest --symbol 002594 --provider local --input-csv data/research/history.csv --test-days 20 --horizon 5 --adapter data/research/adapters/byd_full_2y_t5.pth > data/research/adapters/eval_full_2y_t5.log 2>&1

echo "ALL TESTS COMPLETED"
