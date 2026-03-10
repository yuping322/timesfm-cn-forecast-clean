import os
import sys
import torch
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

def _find_project_root() -> Path:
    """寻找当前项目的根目录（即包含 local_timesfm_model 的目录）。"""
    for parent in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if (parent / "local_timesfm_model").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[2]

PROJECT_ROOT = _find_project_root()

def _find_timesfm_src() -> Path | None:
    """寻找 timesfm 核心库源码路径。"""
    env_root = os.environ.get("TIMESFM_REPO")
    if env_root:
        repo_root = Path(env_root).expanduser().resolve()
        return repo_root / "src"

    # 尝试在同级或上级目录寻找原仓库
    for parent in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
        upstream = parent / "timesfm" / "src"
        if upstream.exists():
            return upstream
        # 如果当前项目本身就在原仓库内（作为技能）
        if (parent / "src" / "timesfm").exists():
            return parent / "src"
    return None

UPSTREAM_SRC = _find_timesfm_src()
if UPSTREAM_SRC and str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))

try:
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
except ImportError:
    TimesFM_2p5_200M_torch = None
    ForecastConfig = None

from .finetuning import LinearAdapter, load_adapter, FeatureExtractor

def 默认模型目录() -> str:
    env_model_path = os.environ.get("TIMESFM_MODEL_PATH")
    if env_model_path:
        return str(Path(env_model_path).expanduser().resolve())
    return str(PROJECT_ROOT / "local_timesfm_model")

def 加载模型(model_dir: str | None) -> TimesFM_2p5_200M_torch:
    实际目录 = model_dir or 默认模型目录()
    model = TimesFM_2p5_200M_torch.from_pretrained(实际目录, torch_compile=False)
    model.compile(
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
    return model

def 运行预测(
    model: TimesFM_2p5_200M_torch,
    序列: np.ndarray,
    context_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    输入 = 序列[-context_length:] if 序列.size > context_length else 序列
    点预测, 分位数预测 = model.forecast(horizon=horizon, inputs=[输入.astype(np.float32)])
    return 点预测[0], 分位数预测[0]

class AdvancedStockModel:
    """
    高级股票预测模型包装类。
    支持：
    1. TimesFM 基础预测。
    2. 线性适配器 (Linear Adapter) 残差修正。
    """
    def __init__(
        self, 
        base_model: Optional[TimesFM_2p5_200M_torch] = None, 
        adapter: Optional[LinearAdapter] = None
    ):
        self.base_model = base_model
        self.adapter = adapter
        self.device = torch.device("cpu")
        
        if self.base_model and hasattr(self.base_model, 'model'):
            self.base_model.model.to(self.device)

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
        pts, qts = self.base_model.forecast(horizon=horizon, inputs=inputs, **kwargs)
        
        # 2. 如果没有适配器，直接返回基础预测
        if not self.adapter:
            return pts, qts

        # 3. 应用适配器修正 (仅针对点预测/中位数)
        adjusted_pts = pts.copy()
        for i, context in enumerate(inputs):
            base_val = pts[i, 0]
            ohlcv_context = ohlcv_inputs[i] if ohlcv_inputs and len(ohlcv_inputs) > i else None
            features = FeatureExtractor.compute(context, base_val, ohlcv_context=ohlcv_context)
            
            # 使用适配器修正
            residual = self.adapter.apply(features.reshape(1, -1))[0]
            adjusted_pts[i, 0] += residual
            qts[i, :, :] += residual

        return adjusted_pts, qts

def load_advanced_model(
    model_dir: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> AdvancedStockModel:
    """加载高级模型。"""
    if TimesFM_2p5_200M_torch is None:
        raise ImportError("未发现 timesfm 库，请确保环境配置正确。")

    base_model = 加载模型(model_dir)
    
    adapter = None
    if adapter_path and os.path.exists(adapter_path):
        print(f"正在加载微调适配器: {adapter_path}")
        weights = load_adapter(adapter_path)
        adapter = LinearAdapter(weights)
    
    return AdvancedStockModel(base_model, adapter)
