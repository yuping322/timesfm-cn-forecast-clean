# 示例命令

## 本地 CSV

```bash
python scripts/run_cn_forecast.py \
  --provider local \
  --input-csv /path/to/data.csv \
  --date-column date \
  --value-column close \
  --horizon 5
```

## 本地自动拉取（AkShare）

```bash
python scripts/run_cn_forecast.py \
  --provider local \
  --symbol 600519 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --horizon 5
```

## Tushare

```bash
export TUSHARE_TOKEN=your_token

python scripts/run_cn_forecast.py \
  --provider tushare \
  --symbol 600519.SH \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --tushare-field close \
  --horizon 5
```

## AkShare

```bash
python scripts/run_cn_forecast.py \
  --provider akshare \
  --symbol 600519 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --akshare-adjust qfq \
  --horizon 5
```

## K 线图 (新增)

通过 `--kline` 参数生成蜡烛图：
```bash
timesfm-cn-forecast --provider akshare --symbol 600519 --horizon 5 --kline
```
## 微调 (Finetuning)

### 1. 训练适配器
```bash
python src/timesfm_cn_forecast/advanced/finetuning.py
```

### 2. 使用适配器进行预测
```bash
python skills/timesfm-cn-forecast/scripts/run_cn_forecast.py \
  --provider local \
  --symbol 600089 \
  --input-csv data/600089.csv \
  --adapter data/tbea_adapter.pth \
  --horizon 1
```
