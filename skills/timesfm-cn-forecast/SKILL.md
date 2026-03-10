---
name: timesfm-cn-forecast
description: 使用本项目自带的 TimesFM 模型做中文本地预测技能。适用于基于历史数据进行时间序列或股票预测，支持从本地文件、Tushare 或 AkShare 准备历史数据。
---

# 中文预测技能 (独立版)

这个 skill 是独立版本，自带 `local_timesfm_model` 模型权重。

## 适用场景

- 需要基于历史数据做本地预测。
- 数据可能来自本地文件、Tushare、AkShare。
- 需要稳定的本地脚本入口，方便后续反复调用。

## 默认入口

从仓库根目录运行：

```bash
python scripts/run_cn_forecast.py \
  --provider local \
  --input-csv /path/to/data.csv \
  --value-column close \
  --date-column date \
  --horizon 5
```

### 4. 高级功能: 微调适配器 (Advanced: Finetuning)

项目支持通过线性残差适配器进一步修正 TimesFM 的原始输出，显著提升股票预测精度。

#### 1. 训练微调模型
运行 `src/timesfm_cn_forecast/advanced/finetuning.py` 进行训练。该脚本默认会下载特变电工 (600089) 的数据并训练，权重保存至 `data/tbea_adapter.pth`。

```bash
python src/timesfm_cn_forecast/advanced/finetuning.py
```

#### 2. 使用适配器进行预测
在运行预测脚本时，通过 `--adapter` 参数指定训练好的权重路径：

```bash
python skills/timesfm-cn-forecast/scripts/run_cn_forecast.py \
    --provider local \
    --symbol 600089 \
    --input-csv data/600089.csv \
    --adapter data/tbea_adapter.pth \
    --horizon 1
```

---

## 🔑 核心参数说明 (Key Parameters)

| 参数 | 说明 | 示例 |
| :--- | :--- | :--- |
| `--provider` | 数据源类型 | `local`, `akshare`, `tushare`, `oss` |
| `--symbol` | 股票代码 | `600519`, `000001.SZ`, `sh600519` |
| `--horizon` | 预测未来跨度 | `5` (天) |
| `--kline` | 生成 K 线图 (蜡烛图) | `--kline` |
| `--adapter` | (高级) 适配器权重路径 | `models/adapter.pth` |

---

## 🛠️ 数据格式 (Data Format)

> [!TIP]
> 增强后的 `providers.py` 会自动处理不同数据源的代码前缀。
- **AkShare**: 优先使用 Sina 接口，支持 `sh600519` 格式。
- **OSS**: 默认从 `hangqing/daily_data/` 目录拉取 CSV。
- **本地**: 支持 CSV/Parquet 格式，需包含 `date` 和 `value` 列。

## 数据源

- `local`：本地 CSV / Parquet。
- `tushare`：通过环境变量中的 token 拉取日线。
- `akshare`：通过 AkShare 拉取 A 股日线。

## 关键参数

- `--symbol`：股票或序列标识；除 `local` 外通常必填。
- `--start` / `--end`：历史数据区间。
- `--context-length`：送入模型的最近历史窗口长度。
- `--model-dir`：本地模型目录，默认使用仓库根目录下的 `local_timesfm_model`。
- `--output-dir`：输出目录。

## 输出

- `history.csv`：清洗后的历史数据（含 OHLCV 列）。
- `forecast.csv`：预测结果。
- `summary.json`：摘要数据。
- `forecast.png`：收盘价预测趋势图。
- `forecast_kline_{symbol}.png`：K 线走势图（开启 --kline 时生成）。

## 环境变量

### Tushare

- `TUSHARE_TOKEN`

### AkShare

## 说明

- 主流程调用自带的 `timesfm` 能力。
- 各数据源 SDK 采用按需导入；没有安装对应依赖时会报出明确错误。
- 数据格式和示例命令见 `skills/timesfm-cn-forecast/references/data_format.md` 与 `skills/timesfm-cn-forecast/references/examples.md`。
