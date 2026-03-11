import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Optional

from .features import FeatureExtractor, get_feature_names

@dataclass
class AdapterWeights:
    coef: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    feature_names: List[str]
    context_len: int
    horizon_len: int
    stock_code: Optional[str] = None

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
    feature_names: List[str],
    stock_code: Optional[str] = None
) -> AdapterWeights:
    """
    训练线性残差适配器。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_X)
    residuals = train_y - train_base
    
    # 6. 训练线性回归 (使用 lstsq 增加稳定性)
    X_aug = np.concatenate([X_scaled, np.ones((X_scaled.shape[0], 1), dtype=np.float32)], axis=1)
    
    print(f"X_aug stats: min={np.min(X_aug):.2e}, max={np.max(X_aug):.2e}, has_nan={np.any(np.isnan(X_aug))}")
    
    coef, residuals_sum, rank, s = np.linalg.lstsq(X_aug, residuals, rcond=0.01)
    
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
    parser.add_argument("--feature-set", type=str, default="technical", help="使用的特征组合名称 (basic, technical, structural, full)")
    
    parser.add_argument("--train-days", type=int, default=None, help="仅使用最近 N 天的数据进行训练。")
    
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
        # 数据清洗：填充缺失值
        df = df.ffill().bfill()
        
        # 截取训练窗口
        if args.train_days:
            print(f"截取最近 {args.train_days} 天的数据进行训练...")
            # 保证有足够的 context 长度
            df = df.tail(args.train_days + CONTEXT_LEN + HORIZON_LEN)
            
        prices = df["value"].values
        # 提取 K 线数据作为特征
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        ohlcv = df[ohlcv_cols].values if all(c in df.columns for c in ohlcv_cols) else None
        
        # 获取特征组合
        feature_names = get_feature_names(args.feature_set)
        print(f"使用的特征组合 [{args.feature_set}]: 维度={len(feature_names)}")
        print(f"特征列表: {feature_names}")

        # 3. 准备微调数据
        samples = []
        targets = []
        base_preds = []
        
        n_samples = len(prices) - CONTEXT_LEN - HORIZON_LEN + 1 # Adjusted n_samples calculation
        if n_samples <= 0:
            print("数据量不足以进行微调训练。")
        else:
            print(f"生成训练样本: {n_samples} 个...")
            
            for i in range(n_samples):
                # 获取上下文
                context = prices[i : i + CONTEXT_LEN]
                # 目标是收盘价的残差
                target = prices[i + CONTEXT_LEN + HORIZON_LEN - 1] # Target is at horizon_len
                
                # 基础预测 (从模型获取或简单假设为今日收盘)
                base_pred = context[-1] # Base prediction is the last value of context
                
                # 提取 K 线上下文特征
                ohlcv_context = ohlcv[i : i + CONTEXT_LEN] if ohlcv is not None else None
                
                # 计算特征
                feats = FeatureExtractor.compute(context, base_pred, ohlcv_context=ohlcv_context, feature_names=feature_names)
                
                samples.append(feats)
                targets.append(target)
                base_preds.append(base_pred)
                
            train_X = np.array(samples, dtype=np.float32)
            train_y = np.array(targets, dtype=np.float32)
            train_base = np.array(base_preds, dtype=np.float32)
            
            # 4. 再次清洗训练矩阵，防止溢出
            train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 5. 训练适配器
            weights = train_linear_adapter(
                train_X=train_X,
                train_y=train_y,
                train_base=train_base,
                context_len=CONTEXT_LEN,
                horizon_len=HORIZON_LEN,
                feature_names=feature_names,
                stock_code=STOCK_CODE
            )
            
            # 5. 保存权重
            save_adapter(weights, str(ADAPTER_SAVE_PATH))
            print(f"适配器权重已保存至: {ADAPTER_SAVE_PATH}")

if __name__ == "__main__":
    main()
