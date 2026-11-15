"""Leakage tests (PLAN.md §4). Split tests live in test_splits.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from council.features import build_panel
from council.historian import Historian
from council.loaders.membership import members_on


def _toy_prices(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", periods=n)
    rows = []
    for t in ["^DJI", "AAA", "BBB", "^VIX", "^TNX", "DX-Y.NYB", "CL=F"]:
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        rows.append(pd.DataFrame({"date": days, "ticker": t, "open": c, "high": c, "low": c, "close": c,
                                  "adj_close": c, "raw_close": c, "volume": 1e6}))
    return pd.concat(rows, ignore_index=True)


def _toy_macro(days) -> pd.DataFrame:
    rows = [pd.DataFrame({"date": days, "series": s, "value": 1.0})
            for s in ["DFF", "T10Y2Y", "CPIAUCSL", "UNRATE", "DCOILWTICO"]]
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def toy():
    prices = _toy_prices()
    days = prices.date.unique()
    membership = pd.DataFrame({"member": ["AAA", "BBB"], "company": ["A", "B"], "ticker": ["AAA", "BBB"],
                               "start_date": pd.to_datetime(["2010-01-01", "2010-06-01"]),
                               "end_date": pd.to_datetime([None, None])})
    return prices, build_panel(prices, _toy_macro(pd.DatetimeIndex(days)), membership), membership


def test_features_only_use_the_past(toy):
    """Changing prices AFTER day t must not change any feature on day t."""
    prices, panel, membership = toy
    t = pd.Timestamp(sorted(prices.date.unique())[300])
    later = prices.copy()
    later.loc[later.date > t, ["open", "high", "low", "close", "adj_close", "raw_close"]] *= 3
    panel2 = build_panel(later, _toy_macro(pd.DatetimeIndex(prices.date.unique())), membership)
    feats = ["ret_20", "rsi", "macd_hist", "z20", "dist_high_250", "breadth_50"]
    a = panel[panel.date == t].set_index("ticker")[feats]
    b = panel2[panel2.date == t].set_index("ticker")[feats]
    pd.testing.assert_frame_equal(a, b)


def test_historian_point_in_time(toy):
    _, panel, _ = toy
    h = Historian(panel)
    D = pd.Timestamp(sorted(panel.date.unique())[250])
    assert (h.history(D).date < D).all()
    assert h.latest("AAA", D).date < D
    known = h.known_outcomes(D)
    assert (known.label_known < D).all()
    sims = h.similar_setups("AAA", D, ["ret_20", "rsi"], k=10)
    assert (sims.date < D).all() and len(sims) == 10


def test_point_in_time_membership(toy):
    _, panel, membership = toy
    assert not ((panel.ticker == "BBB") & (panel.date < "2010-06-01")).any()
    assert set(members_on(membership, pd.Timestamp("2010-03-01")).ticker) == {"AAA"}


def test_labels_start_the_next_day(toy):
    """Entry is the next day's open, so a row's label never includes its own day's move."""
    prices, panel, _ = toy
    row = panel[(panel.ticker == "AAA")].iloc[100]
    p = prices[prices.ticker == "AAA"].set_index("date")
    days = list(p.index)
    i = days.index(row.date)
    expected = p.adj_close.iloc[i + 20] / p.open.iloc[i + 1] - 1
    assert row.fwd_return == pytest.approx(expected)
