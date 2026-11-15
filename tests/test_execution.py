"""Realistic stop-loss fills and the tuned strategy's sizing (council/execution.py, council/strategy.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from council.execution import simulate
from council.strategy import plan


def _prices(opens, lows, closes):
    n = len(opens)
    d = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"date": d, "ticker": "X", "open": opens, "high": closes, "low": lows,
                         "close": closes, "adj_close": closes, "volume": 1.0})


def test_no_stop_sells_at_the_final_close():
    px = _prices([100] * 25, [99] * 25, [100] * 20 + [110] * 5)
    rows = pd.DataFrame({"date": [px.date[0]], "ticker": ["X"]})
    sim = simulate(rows, px, np.array([500.0]), None, h=20)
    assert sim.profit[0] == pytest.approx(500 * (110 / 100 - 1))   # bought day 1 open, sold day 20 close


def test_stop_fills_at_the_stop_price_on_an_intraday_drop():
    lows = [99] * 25
    lows[5] = 85                                  # one day trades down through a 10% stop
    px = _prices([100] * 25, lows, [100] * 25)
    rows = pd.DataFrame({"date": [px.date[0]], "ticker": ["X"]})
    sim = simulate(rows, px, np.array([500.0]), np.array([0.10]), h=20)
    assert sim.profit[0] == pytest.approx(-50.0) and sim.stopped[0] and not sim.gapped[0]


def test_a_gap_below_the_stop_sells_at_the_open():
    opens, lows = [100] * 25, [99] * 25
    opens[5], lows[5] = 80, 78                    # bad overnight news: opens 20% down
    px = _prices(opens, lows, [100] * 25)
    rows = pd.DataFrame({"date": [px.date[0]], "ticker": ["X"]})
    sim = simulate(rows, px, np.array([500.0]), np.array([0.10]), h=20)
    assert sim.profit[0] == pytest.approx(-100.0) and sim.gapped[0]   # lost S$100, not S$50


def test_strategy_leaves_room_for_a_gap():
    amount, stop = plan([0.6], [0.05], [0.02], theta=0.0, k=10, avg_win=0.03, avg_loss=0.03)
    # stop + gap = max loss / amount, so even a 1-in-20 gap through the stop stays within S$50
    assert amount[0] * (stop[0] + 0.02) == pytest.approx(50.0)


def test_strategy_skips_below_the_threshold():
    amount, _ = plan([0.52], [0.05], [0.01], theta=0.55, k=1, avg_win=0.03, avg_loss=0.03)
    assert amount[0] == 0


def test_volatility_scaled_stop_keeps_the_max_loss():
    amount, stop = plan([0.6, 0.6], [0.05, 0.30], [0.01, 0.03], theta=0.0, k=1, avg_win=0.03, avg_loss=0.03, c=1.0)
    assert stop[1] > stop[0]                  # jumpy stock: wider stop
    assert amount[1] < amount[0]              # ... and a smaller amount
    assert amount[1] * (stop[1] + 0.03) == pytest.approx(50.0)   # stop + gap still loses at most S$50
