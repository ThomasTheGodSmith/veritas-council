"""Point-in-time Dow membership (built by scripts/build_membership.py from Wikipedia)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PATH = Path("data/raw/membership/dow_membership.csv")


def load_membership() -> pd.DataFrame:
    """One row per membership spell. end_date is exclusive; NaT means still a member.
    `ticker` is blank (NaN) when no price history exists for that company."""
    df = pd.read_csv(PATH, parse_dates=["start_date", "end_date"])
    df = df.rename(columns={"yf_ticker": "ticker"})
    return df[["member", "company", "ticker", "start_date", "end_date"]]


def members_on(membership: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """The 30 members as of `day` (including ones with no price data)."""
    m = membership
    return m[(m.start_date <= day) & (m.end_date.isna() | (m.end_date > day))]
