"""The training panel: features + labels built ONLY from data/splits/train/ (no test years).

The evaluation panel is built in eval/backtest.py, the only code allowed to open the test folder.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from council.features import build_panel
from council.splits import load_train, read_split

from council.paths import CACHE as _CACHE_DIR  # noqa: E402
CACHE = _CACHE_DIR / "train_panel.parquet"


def train_panel(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    panel = build_panel(load_train("prices"), load_train("macro"), load_train("membership"))
    assert panel.date.max() < pd.Timestamp(read_split().test_start)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE, index=False)
    return panel
