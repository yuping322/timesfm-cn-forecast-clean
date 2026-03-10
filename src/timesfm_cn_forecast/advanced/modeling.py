import os
import torch
import numpy as np
from typing import List, Optional, Union
from ..modeling import _find_timesfm_src, 默认模型目录

# 尝试通过本地路径导入 timesfm
import sys
_ts_src = _find_timesfm_src()
if _ts_src:
    sys.path.insert(0, _ts_src)

try:
    from timesfm import TimesFM_2p5_200M_torch
except ImportError:
    TimesFM_2p5_200M_torch = None

from .finetuning import LinearAdapter, load_adapter, FeatureExtractor

class AdvancedStockModel:
    """
    高级股票预测模型包装类。
    支持：
    1. TimesFM 基础预测。
    2. 线性适配器 (Linear Adapter) 残差修正。
    3. 自动设备选择 (CUDA/MPS/CPU)。
    """
    def __init__(
        self, 
        base_model: Optional[TimesFM_2p5_200M_torch] = None, 
        adapter: Optional[LinearAdapter] = None
    ):
        self.base_model = base_model
        self.adapter = adapter
        self.device = self._detect_device()
        
        if self.base_model and hasattr(self.base_model, 'model'):
            self.base_model.model.to(self.device)

    def _detect_device(self):
        """强制使用 CPU 以避免与基础模型的设备冲突。"""
        return torch.device("cpu")

    def forecast(
        self, 
        inputs: List[np.ndarray], 
        horizon: int,
        ohlcv_inputs: Optional[List[np.ndarray]] = None,
        **kwargs
    ):
        """
        进行预测。如果加载了适配器，则应用残差修正。
        """
        if not self.base_model:
            raise RuntimeError("基础 TimesFM 模型未加载。")

        # 1. 基础预测
        # TimesFM 预测返回 (point_forecast, quantile_forecast)
        pts, qts = self.base_model.forecast(horizon=horizon, inputs=inputs, **kwargs)
        
        # 2. 如果没有适配器，直接返回基础预测
        if not self.adapter:
            return pts, qts

        # 3. 应用适配器修正 (仅针对点预测/中位数)
        # 注意：这里我们假设适配器是为 horizon=1 训练的，或者应用于 horizon 的第一步
        adjusted_pts = pts.copy()
        
        for i, context in enumerate(inputs):
            # 提取特征
            base_val = pts[i, 0] # 获取第一个 horizon 的预测值
            ohlcv_context = ohlcv_inputs[i] if ohlcv_inputs and len(ohlcv_inputs) > i else None
            features = FeatureExtractor.compute(context, base_val, ohlcv_context=ohlcv_context)
            
            # 计算残差修正
            # features 需要是 (1, N) 形状
            residual = self.adapter.apply(features.reshape(1, -1))[0]
            
            # 简单应用：修正第一步点预测
            adjusted_pts[i, 0] += residual
            
            # 对分位数也进行平移修正 (可选，保持简单)
            qts[i, :, :] += residual

        return adjusted_pts, qts

def load_advanced_model(
    model_dir: Optional[str] = None,
    adapter_path: Optional[str] = None,
    torch_compile: bool = False
) -> AdvancedStockModel:
    """加载高级模型。"""
    if TimesFM_2p5_200M_torch is None:
        raise ImportError("未发现 timesfm 库，请确保环境配置正确。")

    if model_dir is None:
        model_dir = 默认模型目录()

    print(f"正在加载基础模型: {model_dir}")
    base_model = TimesFM_2p5_200M_torch.from_pretrained(model_dir)
    
    from timesfm import ForecastConfig
    base_model.compile(
        ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,
            fix_quantile_crossing=True,
        )
    )
    
    adapter = None
    if adapter_path and os.path.exists(adapter_path):
        print(f"正在加载微调适配器: {adapter_path}")
        weights = load_adapter(adapter_path)
        adapter = LinearAdapter(weights)
    
    return AdvancedStockModel(base_model, adapter)
