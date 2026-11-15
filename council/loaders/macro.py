"""FRED macro loader (files from scripts/fetch_macro.py).

LEAKAGE GUARD: monthly series are dated by the month they describe (CPI for March is dated
March 1) but published weeks later. We move each value to its release date, using the lags in
config/data.yaml, so a model can only see a number once it was actually public.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PATH = Path("data/raw/macro")
CONFIG = Path("config/data.yaml")


def _release_lags() -> dict[str, int]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lags = cfg["datasets"]["macro"]["release_lag"]
    return {k: int(str(v).split()[0]) for k, v in lags.items()}


def load_macro() -> pd.DataFrame:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lags = _release_lags()
    frames = []
    for sid in cfg["datasets"]["macro"]["series"]:
        df = pd.read_csv(PATH / f"{sid}.csv", parse_dates=["DATE"])
        df = df.rename(columns={"DATE": "date"})
        df["date"] = df["date"] + pd.Timedelta(days=lags.get(sid, 0))
        df["series"] = sid
        frames.append(df[["date", "series", "value"]])
    return pd.concat(frames, ignore_index=True).sort_values(["series", "date"]).reset_index(drop=True)
