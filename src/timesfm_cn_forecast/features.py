import numpy as np
from typing import List, Optional, Dict

# 直接在代码中定义和维护特征组合，最简单直接。
FEATURE_SETS = {
    "basic": [
        "base_pred", "last_price", "mean_price", "pct_change", "volatility"
    ],
    "technical": [
        "base_pred", "last_price", "mean_price", "pct_change", "volatility",
        "macd", "macd_signal", "macd_hist", "rsi", "boll_upper", "boll_lower",
        "open", "high", "low", "volume"
    ],
    "structural": [
        "base_pred", "last_price", "mean_price", "pct_change", "volatility",
        "body_direction", "body_ratio", "upper_ratio", "lower_ratio", 
        "close_position", "open_position", "range_ratio", "gap", 
        "body_change", "volume"
    ],
    "full": [
        "base_pred", "last_price", "mean_price", "pct_change", "volatility",
        "macd", "macd_signal", "macd_hist", "rsi", "boll_upper", "boll_lower",
        "open", "high", "low", "close", "volume",
        "body_direction", "body_ratio", "upper_ratio", "lower_ratio", 
        "close_position", "open_position", "range_ratio", "gap", 
        "body_change", "true_range", "close_norm"
    ]
}

def get_feature_names(mode: str) -> List[str]:
    """获取指定模式的特征名称列表"""
    if mode not in FEATURE_SETS:
        raise ValueError(f"未知的特征组合: {mode}。可选: {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[mode]

def generate_features_dict(context: np.ndarray, base_pred: float, ohlcv_context: Optional[np.ndarray], mode: str = "technical") -> Dict[str, float]:
    """
    计算并返回特征字典。
    这实现了完全基于字典的直接存取。
    """
    if len(context) == 0:
        return {k: 0.0 for k in get_feature_names(mode)}
        
    feats = {}
    
    last_price = float(context[-1])
    first_price = float(context[0])
    
    # --- 1. 基础特征 ---
    feats["base_pred"] = float(base_pred)
    feats["last_price"] = last_price
    feats["mean_price"] = float(np.mean(context))
    feats["pct_change"] = float((last_price - first_price) / first_price) if first_price != 0 else 0.0
    feats["volatility"] = float(np.std(context))
    feats["close_norm"] = 1.0
    
    # --- 2. 技术指标 ---
    def ema(data, window):
        alpha = 2 / (window + 1)
        res = np.zeros_like(data)
        res[0] = data[0]
        for i in range(1, len(data)):
            res[i] = alpha * data[i] + (1 - alpha) * res[i-1]
        return res
        
    if len(context) >= 30:
        ema12, ema26 = ema(context, 12), ema(context, 26)
        macd_line = ema12 - ema26
        signal_line = ema(macd_line, 9)
        macd_hist = macd_line - signal_line
        
        feats["macd"] = float(macd_line[-1])
        feats["macd_signal"] = float(signal_line[-1])
        feats["macd_hist"] = float(macd_hist[-1])
        
        delta = np.diff(context)
        gain, loss = np.where(delta > 0, delta, 0), np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:]) if len(gain)>=14 else 0
        avg_loss = np.mean(loss[-14:]) if len(loss)>=14 else 0
        feats["rsi"] = 100.0 if avg_loss == 0 else (100.0 - (100.0 / (1.0 + avg_gain / avg_loss)))
        
        ma20, std20 = np.mean(context[-20:]), np.std(context[-20:])
        feats["boll_upper"] = float(ma20 + 2*std20)
        feats["boll_lower"] = float(ma20 - 2*std20)
    else:
        for k in ["macd", "macd_signal", "macd_hist", "rsi", "boll_upper", "boll_lower"]:
            feats[k] = 0.0

    # --- 3. K线结构 ---
    if ohlcv_context is not None and len(ohlcv_context) >= 2:
        op, hi, lo, cl, vol = ohlcv_context[-1]
        p_op, p_hi, p_lo, p_cl, p_vol = ohlcv_context[-2]
        
        feats.update({"open": float(op), "high": float(hi), "low": float(lo), "close": float(cl), "volume": float(vol)})
        feats["close_norm"] = float(cl / p_cl) if p_cl != 0 else 1.0
        
        def safe_div(n, d, default=1.0): return float(n / d) if abs(d) > 0.01 else float(default)

        body, rng_raw = abs(cl - op), hi - lo
        up_sh, lo_sh = hi - max(op, cl), min(op, cl) - lo
        rng = rng_raw if rng_raw > 1e-4 else 1e-4

        feats["body_direction"] = float(np.sign(cl - op))
        feats["body_ratio"] = float(np.clip(safe_div(body, rng_raw, 0.0), 0, 1))
        feats["upper_ratio"] = float(np.clip(safe_div(up_sh, rng_raw, 0.0), 0, 1))
        feats["lower_ratio"] = float(np.clip(safe_div(lo_sh, rng_raw, 0.0), 0, 1))
        feats["close_position"] = float(np.clip(safe_div(cl - lo, rng, 0.5), 0, 1))
        feats["open_position"] = float(np.clip(safe_div(op - lo, rng, 0.5), 0, 1))
        feats["gap"] = 1.0 if op > p_hi else (1.0 if op < p_lo else 0.0)
        feats["range_ratio"] = float(safe_div(rng_raw, p_cl, 0.0))
        p_body = abs(p_cl - p_op)
        feats["body_change"] = float(np.clip(safe_div(body, p_body if p_body > 0.01 else 0.01, 1.0), 0, 5))
        feats["true_range"] = float(max(hi - lo, abs(hi - p_cl), abs(lo - p_cl)))
    else:
        for k in ["open", "high", "low", "close", "volume", "body_direction", "body_ratio", "upper_ratio", "lower_ratio", "close_position", "open_position", "range_ratio", "gap", "body_change", "true_range"]:
            feats[k] = 0.0

    # 4. 根据请求的 mode 返回对应的字典，并防溢出
    keys = get_feature_names(mode)
    out_dict = {}
    for k in keys:
        val = feats.get(k, 0.0)
        out_dict[k] = float(np.clip(val, -1e6, 1e6))
        
    return out_dict

class FeatureExtractor:
    """提供给已有代码获取 Numpy 数组的接口包装，保持向后兼容"""
    @staticmethod
    def compute(context: np.ndarray, base_pred: float, ohlcv_context: Optional[np.ndarray], feature_names: List[str]) -> np.ndarray:
        # 算所有的，但是我们实际上可以直接传一个虚拟 mode，或者直接算 full
        # 这里为了兼容灵活的 feature_names 列表参数，先获取最大字典，再提取
        full_dict = generate_features_dict(context, base_pred, ohlcv_context, mode="full")
        out = []
        for name in feature_names:
            out.append(full_dict.get(name, 0.0))
        return np.array(out, dtype=np.float32)
