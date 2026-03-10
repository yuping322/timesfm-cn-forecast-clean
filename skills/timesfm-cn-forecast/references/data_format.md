# 数据格式

## local 模式

支持 CSV 或 Parquet。至少包含一列日期与一列数值。

### CSV 示例

字段要求：

- 日期列：默认 `date`，可用 `--date-column` 指定
- 数值列：默认 `close`，可用 `--value-column` 指定

#### 基础模式 (仅收盘价)
```csv
date,close
2024-01-01,100.2
2024-01-02,101.5
2024-01-03,99.7
```

#### K 线模式 (启用 --kline)
必须包含 `open`, `high`, `low`, `close` (或中文 `开盘`, `最高`, `最低`, `收盘`)。
```csv
date,open,high,low,close,volume
2024-01-01,100.0,105.0,98.0,100.2,10000
2024-01-02,100.2,102.0,99.0,101.5,12000
```

### Parquet 示例

字段与 CSV 一致，仅文件格式不同。

## Tushare 模式

使用 `pro.daily` 拉取日线，默认数值字段为 `close`。

必要环境变量：

- `TUSHARE_TOKEN`

`--symbol` 需使用 Tushare 代码格式，例如 `600519.SH`。

## AkShare 模式

使用 `stock_zh_a_hist` 拉取日线，默认取 `收盘`。

`--symbol` 可传 `600519` 或 `sh600519`。
 
+## 适配器权重 (Adapter Weights)
+
+微调产生的适配器保存为 `.pth` 格式的 PyTorch 模型文件。
+
+### 文件内容
+包含以下状态字典映射：
+- `adapter_coef`: 线性层权重。
+- `scaler_mean`: 特征标准化均值。
+- `scaler_std`: 特征标准化标准差。
+- `context_len`: 训练时的历史窗口长度。
+- `horizon_len`: 训练时的预测跨度。
+- `stock_code`: 对应的股票代码。
+
+可以通过 `src/timesfm_cn_forecast/advanced/finetuning.py` 中的 `load_adapter` 函数加载。
+
