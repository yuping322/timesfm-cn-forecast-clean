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

## 默认核心流转：三步走 (3-Step Pipeline)

在完成代码重构后，本工程已经转为一个标准的 Python Package。建议直接使用模块调用 `python -m timesfm_cn_forecast.<module>` 的方式执行，这不仅可以彻底摆脱路径依赖，还能享用最完整的特性。

### 1. 数据底座准备 (`providers`)

从 AkShare、Tushare 或 OSS 拉取基础数据并标准化保存为 CSV。

```bash
python -m timesfm_cn_forecast.providers \
  --symbol 002594 \
  --start 2015-01-01 \
  --output data/research/history.csv \
  --kline
```

### 2. 微调与特征合成 (`finetuning`)

使用生成的数据底座，结合特征引擎（如 full 包含 27 维特征），训练针对该股票特定窗口特定步长的残差适配器。

```bash
python -m timesfm_cn_forecast.finetuning \
  --stock-code 002594 \
  --data-path data/research/history.csv \
  --output-path data/research/adapters/byd_full_3m_t1.pth \
  --train-days 60 \
  --horizon-len 1 \
  --feature-set full
```

### 3. 回测与实盘预测 (`backtest`)

使用训练好的适配器，在本地数据上滚动回放，检验胜率和误差。建议使用 `--provider local` 以免除接口限流。

```bash
python -m timesfm_cn_forecast.backtest \
  --symbol 002594 \
  --provider local \
  --input-csv data/research/history.csv \
  --test-days 20 \
  --horizon 1 \
  --adapter data/research/adapters/byd_full_3m_t1.pth
```


---

## 🔑 核心参数说明 (Key Parameters)

| 模块 | 参数 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| **providers** | `--provider` | 数据源类型 | `local`, `akshare`, `tushare`, `oss` |
| **providers** | `--symbol` | 股票代码（支持智能推断前缀）| `002594`, `600519` |
| **finetuning**| `--feature-set` | 注入大模型的特征工程体系维度 | `basic`(5), `technical`(15), `structural`(15), `full`(27) |
| **finetuning**| `--train-days` | 滚动训练的时间窗口 | `60` (短线博弈), `500` (中长线趋势) |
| **finetuning**| `--horizon-len`| 构建适配器针对的预测步长 | `1` (次日), `5` (一周) |
| **backtest**  | `--test-days` | 测试回放的天数 | `20` |
| **backtest**  | `--adapter` | (必填) 指定应用哪个适配器进行推理 | `data/xxx.pth` |

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

## 输出与产物

- **数据底座** (`history.csv`)：标准化的股票历史数据，必定包含 `open`, `high`, `low`, `close`, `volume` 五大日线元素。
- **配置权重** (`Byxxx_t1.pth`)：训练后生成的适配器文件。
- **回测日志** (`eval_xxx.log`)：包含最重要的量化指标：
    - **RMSE (Root Mean Square Error)**：预测收盘价与实际价格的离散误差（越低越好）。
    - **HitRate (胜率)**：预测涨跌方向与实际涨跌方向的匹配率（越高越好）。

## 时空配频实战指南 (经验总结)

- **做短线 T+1**：使用 `--feature-set full` + `--train-days 60`。依赖短期复杂博弈逻辑，信息越多越好。
- **做中长线 T+5**：使用 `--feature-set basic` + `--train-days 500`。屏蔽短期结构噪音，追求纯价格动量的稳定释放。

## 环境变量

### Tushare 和 OSS

- `TUSHARE_TOKEN`
- `OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET`, `OSS_ENDPOINT`, `OSS_BUCKET`

### AkShare

## 说明

- 主流程调用自带的 `timesfm` 能力。
- 各数据源 SDK 采用按需导入；没有安装对应依赖时会报出明确错误。
- 数据格式和示例命令见 `skills/timesfm-cn-forecast/references/data_format.md` 与 `skills/timesfm-cn-forecast/references/examples.md`。
