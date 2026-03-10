# TimesFM 中文股票预测 (精简版)

这是一个独立、精简的仓库，用于使用 TimesFM 模型预测中国股票价格。

## 安装

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **配置环境**:
   创建 `.env` 文件或导出以下变量：
   - `TUSHARE_TOKEN`: 您的 Tushare API 令牌。
   - `TIMESFM_REPO`: 原始 `timesfm` 仓库的路径（用于查找 `timesfm` 库和模型权重）。
   - `TIMESFM_MODEL_PATH` (可选): 模型权重目录的直接路径。

## 使用方法

运行预测脚本：
```bash
python scripts/run_cn_forecast.py \
  --provider akshare \
  --symbol 600519 \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --horizon 5 \
  --output-dir ./outputs/maotai
```

或者使用安装好的命令行工具（CLI）：
```bash
timesfm-cn-forecast \
  --provider local \
  --symbol 002594 \
  --start 2024-01-01 \
  --horizon 5
```

### 生成 K 线图 (新增)
通过添加 `--kline` 参数，可以同时生成历史数据的蜡烛图（K 线图）：
```bash
timesfm-cn-forecast \
  --provider akshare \
  --symbol 600519 \
  --horizon 5 \
  --kline
```
这将会在输出目录中生成 `forecast_kline_600519.png`。

## 高级用法：模型微调 (Finetuning)

您可以通过为特定股票训练线性残差适配器（linear residual adapter）来进一步提高预测精度。

1. **训练适配器**:
   ```bash
   python src/timesfm_cn_forecast/advanced/finetuning.py
   ```
   默认情况下，这会将适配器权重保存到 `data/tbea_adapter.pth`。

2. **使用适配器进行预测**:
   ```bash
   python scripts/run_cn_forecast.py \
     --provider local \
     --symbol 600089 \
     --input-csv data/600089.csv \
     --adapter data/tbea_adapter.pth \
     --horizon 1
```
