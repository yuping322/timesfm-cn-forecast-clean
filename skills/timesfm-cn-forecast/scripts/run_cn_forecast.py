#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文 TimesFM 预测脚本。"""

from __future__ import annotations

import sys
from pathlib import Path


# 尝试定位项目根目录，以便找到 src
_CURRENT_DIR = Path(__file__).resolve().parent
if (_CURRENT_DIR.parents[2] / "src").exists():
    REPO_ROOT = _CURRENT_DIR.parents[2]
elif (_CURRENT_DIR.parents[0] / "src").exists(): # 如果在老位置
     REPO_ROOT = _CURRENT_DIR.parents[0]
else:
     REPO_ROOT = _CURRENT_DIR.parents[2] # 兜底

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from timesfm_cn_forecast.cli import main


if __name__ == "__main__":
    main()

