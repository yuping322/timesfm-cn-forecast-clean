# 示例命令

## 1. 基础预测跑通 (不含微调)

如果你只想用纯价格数据进行预测而不训练，可以直接调用 `pipeline`：

```bash
python -m timesfm_cn_forecast.pipeline \
  --provider local \
  --input-csv /path/to/data.csv \
  --date-column date \
  --value-column close \
  --horizon 5
```

## 2. 准备数据底座

自动从 AkShare 下载包含开高低收的 K 线数据，作为大模型的弹药：

```bash
python -m timesfm_cn_forecast.providers \
  --provider akshare \
  --symbol 600519 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --output data/600519/history.csv \
  --kline
```

对于 Tushare，你只需要指定 `tushare` 并传入标准代码：

```bash
export TUSHARE_TOKEN=your_token

python -m timesfm_cn_forecast.providers \
  --provider tushare \
  --symbol 600519.SH \
  --start 2024-01-01 \
  --output data/600519_ts/history.csv
```

## 3. 微调与预测验证 (Finetuning & Backtesting)

实战中最强的玩法是利用 `finetuning` 构建股票专属特征适配器，并在 `backtest` 中验证：

### 第一步：定制训练适配器
生成 `full` 级别（27维技术指标）适配器，针对短线 T+1 博弈强化：
```bash
python -m timesfm_cn_forecast.finetuning \
  --stock-code 600519 \
  --data-path data/600519/history.csv \
  --output-path data/adapters/600519_full_adapter.pth \
  --train-days 60 \
  --horizon-len 1 \
  --feature-set full
```

### 第二步：滚动回放回测
直接在本地刚刚清洗好的全量数据上跑模拟实盘回放，检验模型的 RMSE 和胜率方向：
```bash
python -m timesfm_cn_forecast.backtest \
  --provider local \
  --symbol 600519 \
  --input-csv data/600519/history.csv \
  --adapter data/adapters/600519_full_adapter.pth \
  --test-days 20 \
  --horizon 1
```
