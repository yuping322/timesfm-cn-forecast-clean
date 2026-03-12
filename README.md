# TimesFM 中文股票预测 (精简版)

这是一个独立、精简的仓库，用于使用 TimesFM 模型预测中国股票价格。系统不仅支持对历史区间的数据进行回测推演评估，同时具备在生产环境下自动化的**次日收益率预测**功能。

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

## 快速使用说明 (Pipelines)

项目内部提供了一套高度封装从**科研评估**到**实盘推荐**的脚本串联体系，所有执行脚本位于 `scripts/` 目录下。

### 1. 单股快速体检表 (Single Stock Evaluation)
快速探索特定的个股在不同参数下的拟合效果：
```bash
# bash scripts/run_single_stock_eval.sh <股票代码> [train_days] [horizon] [feature_set] [test_days]
bash scripts/run_single_stock_eval.sh 002594 60 1 full 20
```

### 2. 批量组别回测跑批 (Group Evaluation)
依据特定组别或全体组别（例如：行业、概念、宽基指数），自动抓取该板块的所有成分股做群体微调适配（Adapter），并批量输出每只个股的回测打分表 `results.csv`。
```bash
# 测试特定单组
bash scripts/run_one_group_eval.sh ind_消费电子
# 测试全体数据组，并自动产出汇总表 group_summary.csv 和最佳组 best_group.txt
bash scripts/run_all_groups_eval.sh
```

### 3. 日常实盘自动选股预测 (Daily Auto Top Picks)
将之前计算出的**全网胜率最高**的板块应用至今天最新的日K线，预测明天所有股票的绝对数值和预期收益，产出具体的买卖推荐单 `daily_picks.csv` (生产预测，无数据集切分)：
```bash
bash scripts/run_auto_top_picks.sh
```

*(如果只需单独对接对某个特定板块的日常预测，调用 `bash scripts/run_daily_predict.sh <板块名> <Adapter路径> [horizon]`)*

### 📁 自动统一的输出目录规范
这套系统的所有脚本不论是回测还是预测，都会在运行起初时在 `data/tasks/` 下生成一个带时间戳的**独立专属任务包**（例如 `data/tasks/eval_single_002594_20260312_132000`）。
每次任务产生的所有中间切片特征(`data/`)、日志(`logs/`)、生成的微调权重(`adapters/`)以及给出的推款预测单(`predictions/`)均会被打包收拢在此专属目录内，杜绝多次跑批覆盖混淆的可能。

## 其他底层功能 (Low-level CLI)

核心功能依然可通过底层 `cli.py` (即 `timesfm-cn-forecast`) 进行自由调用：

```bash
# 基础调用
timesfm-cn-forecast \
  --provider local \
  --symbol 002594 \
  --start 2024-01-01 \
  --horizon 5 \
  --kline

# 带微调权重预测
timesfm-cn-forecast \
  --provider duckdb \
  --symbol 000001 \
  --adapter data/research/ind_消费电子/adapter.pth \
  --horizon 1
```
