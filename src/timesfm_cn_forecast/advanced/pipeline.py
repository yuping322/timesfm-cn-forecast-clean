import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional
from .modeling import AdvancedStockModel
from ..providers import 批量加载历史数据

class BatchRankingPipeline:
    """
    批量股票预测与排名管线。
    集成自 real_stock_predictor.py。
    """
    def __init__(self, model: AdvancedStockModel):
        self.model = model

    def run(
        self,
        symbols: List[str],
        provider: str,
        start_date: str,
        end_date: str,
        context_len: int = 60,
        horizon_len: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        运行批量预测并按预期收益率排序。
        """
        print(f"正在为 {len(symbols)} 只股票批量加载数据 ({provider})...")
        
        # 使用增强后的 providers 逻辑加载宽表
        wide_df = 批量加载历史数据(
            symbols=symbols,
            provider=provider,
            start=start_date,
            end=end_date,
            **kwargs
        )
        
        if wide_df.empty:
            print("未能加载到任何有效的历史数据。")
            return []

        results = []
        actual_symbols = wide_df.columns
        print(f"成功加载 {len(actual_symbols)} 只股票。开始预测...")

        for symbol in actual_symbols:
            try:
                # 准备输入数据
                series = wide_df[symbol].dropna().values
                if len(series) < context_len:
                    continue
                
                model_input = series[-context_len:].astype(np.float32)
                
                # 进行预测
                # pts 形状为 (1, horizon_len)
                pts, _ = self.model.forecast(
                    inputs=[model_input],
                    horizon=horizon_len
                )
                
                predicted_price = float(pts[0, 0])
                last_price = float(series[-1])
                
                # 计算预期收益率
                expected_return = (predicted_price - last_price) / last_price * 100
                
                results.append({
                    "symbol": symbol,
                    "last_price": last_price,
                    "predicted_price": predicted_price,
                    "expected_return": expected_return,
                    "timestamp": wide_df.index[-1].strftime("%Y-%m-%d") if hasattr(wide_df.index[-1], 'strftime') else str(wide_df.index[-1])
                })
                
            except Exception as e:
                print(f"预测股票 {symbol} 时出错: {e}")
                continue

        # 按收益率降序排列
        results_sorted = sorted(results, key=lambda x: x["expected_return"], reverse=True)
        return results_sorted

def run_batch_ranking(
    model: AdvancedStockModel,
    symbols: List[str],
    provider: str,
    start_date: str,
    end_date: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """核心入口函数。"""
    pipeline = BatchRankingPipeline(model)
    return pipeline.run(symbols, provider, start_date, end_date, **kwargs)
