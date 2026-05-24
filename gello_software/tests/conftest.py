"""Pytest fixtures + sys.path setup for the gello_software test suite.

The eval launcher imports ``molmoact`` as a top-level module (it lives at
``experiments/molmoact.py``). Production runs put ``experiments/`` on the
import path implicitly; tests must do it explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path


_GELLO = Path(__file__).resolve().parent.parent
for _p in (_GELLO, _GELLO / "experiments"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
