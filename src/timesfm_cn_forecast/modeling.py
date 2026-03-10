#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型加载与预测。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
elif not UPSTREAM_SRC:
    # 如果找不到，假设已经安装在环境里，不做处理
    pass

import timesfm
from timesfm import ForecastConfig


def 默认模型目录() -> str:
    env_model_path = os.environ.get("TIMESFM_MODEL_PATH")
    if env_model_path:
        return str(Path(env_model_path).expanduser().resolve())
    return str(PROJECT_ROOT / "local_timesfm_model")


def 加载模型(model_dir: str | None) -> timesfm.TimesFM_2p5_200M_torch:
    实际目录 = model_dir or 默认模型目录()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(实际目录, torch_compile=False)
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
    model: timesfm.TimesFM_2p5_200M_torch,
    序列: np.ndarray,
    context_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    输入 = 序列[-context_length:] if 序列.size > context_length else 序列
    点预测, 分位数预测 = model.forecast(horizon=horizon, inputs=[输入.astype(np.float32)])
    return 点预测[0], 分位数预测[0]
