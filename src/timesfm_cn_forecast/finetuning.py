import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AdapterWeights:
    coef: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: List[str]
    context_len: int
    horizon_len: int
    stock_code: Optional[str] = None

class FeatureExtractor:
    @staticmethod
    def compute(context: np.ndarray, base_pred: float, ohlcv_context: Optional[np.ndarray] = None) -> np.ndarray:
        """
        提取特征向量用于线性适配器。
        
        特征包括：
        1. 基础预测值 (base_pred)
        2. 最后价格 (last_price)
        3. 均值 (mean_price)
        4. 价格变动百分比 (pct_change)
        5. 波动率 (volatility)
        6. 开盘价 (open) - 如果提供
        7. 最高价 (high) - 如果提供
        8. 最低价 (low) - 如果提供
        9. 成交量 (volume) - 如果提供
        """
        if len(context) == 0:
            return np.zeros(9, dtype=np.float32)
            
        last_price = context[-1]
        mean_price = float(np.mean(context))
        first_price = context[0]
        pct_change = float((last_price - first_price) / first_price) if first_price != 0 else 0.0
        volatility = float(np.std(context))
        
        features = [
            base_pred, 
            last_price, 
            mean_price, 
            pct_change, 
            volatility
        ]
        
        if ohlcv_context is not None and ohlcv_context.shape[1] >= 4:
            # ohlcv_context 假设为 [N, 4] -> open, high, low, volume
            last_ohlcv = ohlcv_context[-1]
            features.extend(last_ohlcv.tolist())
        else:
            features.extend([0.0] * 4)
            
        return np.array(features, dtype=np.float32)

class LinearAdapter:
    def __init__(self, weights: AdapterWeights):
        self.weights = weights
        
    def apply(self, features: np.ndarray) -> np.ndarray:
        """应用残差修正。"""
        # 缩放特征
        scaled_features = (features - self.weights.mean) / self.weights.scale
        
        # 增加偏置项
        ones = np.ones((scaled_features.shape[0], 1), dtype=np.float32)
        X_aug = np.concatenate([scaled_features, ones], axis=1)
        
        # 计算残差并返回修正后的值
        residuals = X_aug @ self.weights.coef
        return residuals

def train_linear_adapter(
    train_X: np.ndarray, 
    train_y: np.ndarray, 
    train_base: np.ndarray,
    context_len: int,
    horizon_len: int,
    stock_code: Optional[str] = None
) -> AdapterWeights:
    """
    训练线性残差适配器。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)
    residuals = train_y - train_base
    
    # 增加偏置项进行最小二乘法求解
    ones = np.ones((X_scaled.shape[0], 1), dtype=np.float32)
    X_aug = np.concatenate([X_scaled, ones], axis=1)
    
    coef, *_ = np.linalg.lstsq(X_aug, residuals, rcond=None)
    
    feature_names = ["base_pred", "last_price", "mean_price", "pct_change", "volatility", "open", "high", "low", "volume"]
    
    return AdapterWeights(
        coef=coef.astype(np.float32),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        feature_names=feature_names,
        context_len=context_len,
        horizon_len=horizon_len,
        stock_code=stock_code
    )

def save_adapter(weights: AdapterWeights, path: str):
    """保存适配器权重到 .pth 文件。"""
    data = {
        "adapter_coef": weights.coef,
        "scaler_mean": weights.mean,
        "scaler_scale": weights.scale,
        "feature_names": weights.feature_names,
        "context_len": weights.context_len,
        "horizon_len": weights.horizon_len,
        "stock_code": weights.stock_code
    }
    torch.save(data, path)

def load_adapter(path: str) -> AdapterWeights:
    """从 .pth 文件加载适配器权重。"""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return AdapterWeights(
        coef=data["adapter_coef"],
        mean=data["scaler_mean"],
        scale=data["scaler_scale"],
        feature_names=data["feature_names"],
        context_len=data["context_len"],
        horizon_len=data["horizon_len"],
        stock_code=data.get("stock_code")
    )

def main():
    import os
    import pandas as pd
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="训练线性残差适配器。")
    parser.add_argument("--stock-code", type=str, required=True, help="股票代码")
    parser.add_argument("--data-path", type=str, required=True, help="训练数据 (history.csv) 的路径")
    parser.add_argument("--output-path", type=str, required=True, help="适配器权重 (.pth) 的保存路径")
    parser.add_argument("--context-len", type=int, default=60, help="上下文长度")
    parser.add_argument("--horizon-len", type=int, default=1, help="预测步长")
    
    args = parser.parse_args()

    STOCK_CODE = args.stock_code
    DATA_PATH = Path(args.data_path)
    ADAPTER_SAVE_PATH = Path(args.output_path)
    CONTEXT_LEN = args.context_len
    HORIZON_LEN = args.horizon_len
    
    print(f"开始为股票 {STOCK_CODE} 训练线性适配器...")
    
    # 确保输出目录存在
    ADAPTER_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 2. 加载数据
    if not DATA_PATH.exists():
        print(f"数据文件 {DATA_PATH} 不存在，请先准备数据。")
    else:
        df = pd.read_csv(DATA_PATH)
        prices = df["value"].values
        # 提取 K 线数据作为特征
        ohlcv_cols = ["open", "high", "low", "volume"]
        ohlcv = df[ohlcv_cols].values if all(c in df.columns for c in ohlcv_cols) else None
        
        # 3. 准备微调数据
        n_samples = len(prices) - CONTEXT_LEN - HORIZON_LEN
        if n_samples <= 0:
            print("数据量不足以进行微调训练。")
        else:
            train_X = []
            train_y = []
            train_base = [] # 模拟基础预测
            
            print(f"生成训练样本: {n_samples} 个...")
            for i in range(n_samples):
                context = prices[i : i + CONTEXT_LEN]
                target = prices[i + CONTEXT_LEN]
                
                # 提取 K 线上下文特征
                ohlcv_context = ohlcv[i : i + CONTEXT_LEN] if ohlcv is not None else None
                
                # 注意：为了演示，我们假设基础预测值就是 context 的最后一个值（简单基准）
                # 在生产环境中，这应该是 TimesFM 给出的预测值
                base_pred = prices[i + CONTEXT_LEN - 1]
                
                feats = FeatureExtractor.compute(context, base_pred, ohlcv_context=ohlcv_context)
                train_X.append(feats)
                train_y.append(target)
                train_base.append(base_pred)
            
            train_X = np.array(train_X)
            train_y = np.array(train_y)
            train_base = np.array(train_base)
            
            # 4. 训练适配器
            weights = train_linear_adapter(
                train_X=train_X,
                train_y=train_y,
                train_base=train_base,
                context_len=CONTEXT_LEN,
                horizon_len=HORIZON_LEN,
                stock_code=STOCK_CODE
            )
            
            # 5. 保存权重
            save_adapter(weights, str(ADAPTER_SAVE_PATH))
            print(f"适配器权重已保存至: {ADAPTER_SAVE_PATH}")

if __name__ == "__main__":
    main()
