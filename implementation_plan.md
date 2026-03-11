# 多分组批量 Pipeline 评估技术方案

## 目标
对 `index_constituents` 表中的每个分组，批量评估分组级适配器（patch）的预测能力，汇总每组内所有股票的 RMSE / HitRate 分布，横向对比各组的预测能力与稳定性。

---

## 新增模块概览

```
scripts/
  run_group_eval.py        # 批量评估入口（Python 脚本）
  analyze_group_results.py # 结果汇总 + 分组对比分析

src/timesfm_cn_forecast/
  providers.py             # [MODIFY] 新增 duckdb provider，直接从 market.duckdb 拉取 OHLCV

data/research/
  <group_name>/
    adapter.pth            # 分组级适配器（仅训练一次）
    results.csv            # 该分组下所有股票的评估结果
  group_summary.csv        # 所有分组汇总结果（最终产物）
```

---

## 实验执行方式（不改代码）

实验阶段只需**调整脚本参数**，不做任何代码修改：
1. `run_group_eval.py` 负责单个分组的训练 + 回测产出 `results.csv`
2. `analyze_group_results.py` 负责跨分组汇总

实验细节与阶段安排见：`experiment_plan.md`。

---

## Step 1：新增 DuckDB Provider

**[MODIFY] [src/timesfm_cn_forecast/providers.py](file:///Users/fengzhi/Downloads/git/timesfm-cn-forecast-clean/src/timesfm_cn_forecast/providers.py)**

新增 `load_from_duckdb(req)` 函数，其中 `DataRequest.provider = "duckdb"` 时触发。**符号归一化必须走统一函数**（扩展现有 `standardize_symbol` 或引入单一 `normalize_symbol` 入口，供 duckdb 读取与过滤同时使用）：

```python
def load_from_duckdb(req: DataRequest) -> pd.DataFrame:
    """从 market.duckdb 按 symbol 拉取 OHLCV，返回标准格式。"""
    db_symbol = normalize_symbol(req.symbol, target="db")   # 002594 -> sz002594
    con = duckdb.connect(req.duckdb_path, read_only=True)
    df = con.execute(
        "SELECT date, open, high, low, close, volume FROM daily_data "
        "WHERE symbol = ? ORDER BY date",
        [db_symbol]
    ).fetchdf()
    ...
```

CLI 新增参数 `--duckdb-path`，和 `--provider duckdb`。

---

## Step 2：批量评估脚本

**[NEW] `scripts/run_group_eval.py`**

输入：分组名称（如 `ind_消费电子`）、配置参数（feature-set、train-days、horizon）
流程：
1. 从 `index_market.duckdb` 读取该分组的股票列表
2. 交叉过滤：只保留在 `market.duckdb` 中有 ≥ 1000 天数据的股票
3. **分组级只训练一次**：将该分组所有股票历史数据合并构造训练集，训练一个分组级适配器（patch）
4. 用同一适配器对分组内每只股票运行回测（调用 `run_backtest` 函数，非 CLI）
5. 输出每只股票的最优 HitRate 和 RMSE 到 CSV（`results.csv`）

```bash
python scripts/run_group_eval.py \
  --group ind_消费电子 \
  --market-duckdb data/market.duckdb \
  --index-duckdb data/index_market.duckdb \
  --feature-set full \
  --train-days 60 \
  --horizon 1 \
  --output-dir data/research
```

---

## Step 3：结果汇总分析

**[NEW] `scripts/analyze_group_results.py`**

读取 `data/research/` 下所有 `results.csv`，提取最优上下文的 RMSE / HitRate，按分组聚合输出对比表：

| group | n_stocks | hitrate_mean | hitrate_std | hitrate_p65 | rmse_mean |
|---|---|---|---|---|---|
| ind_消费电子 | 72 | 67.3% | 9.8% | 58% | 1.85 |
| CYBZ | 88 | 59.1% | 14.2% | 31% | 2.31 |
| ... | | | | | |

- `hitrate_p65` = 组内 HitRate ≥ 65% 的股票比例（越高越好）
- `hitrate_std` = 标准差，越低组内越稳定（过拟合检测）

---

## 实验矩阵

首轮跑以下配置（固定 full + 60天 T+1）：

| 分组 | 规模 |
|---|---|
| **宽基指数** | CYBZ / ZZ500 / small |
| **行业组** | ind_消费电子 / ind_军工电子 / ind_芯片 / ind_IT服务 |
| **概念组** | con_低空经济 / con_比亚迪链 / con_信创 |

每个分组只训练一次适配器，随后对该分组内股票逐只回测。总体耗时取决于“分组数量 + 单只股票回测成本”，不再按“每只股票都训练一次”估算。

实验运行时只需变更参数（不改代码）：
1. `--group`
2. `--feature-set`
3. `--train-days`
4. `--horizon`
5. `--output-dir`

---

## 过拟合判定规则（结果后处理）

```
if hitrate_mean > 60% AND hitrate_std < 12% AND hitrate_p65 > 40%:
    -> 该分组有稳定预测能力 ✅
elif hitrate_std > 18%:
    -> 预测能力不稳定，高度依赖个股，疑似过拟合 ❌
```

---

## 实现顺序

1. `providers.py` 新增 DuckDB provider（约 50 行）
2. `run_group_eval.py` 批量脚本（约 100 行）
3. `analyze_group_results.py` 分析脚本（约 60 行）
4. 运行首轮实验（仅调整参数，不改代码），收集结果
