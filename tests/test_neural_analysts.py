"""Pattern Oracle and Raw Reader: no look-ahead, plus a check that fast sampling is exact.

Each model is trained (tiny settings) on a synthetic price table. We then change every price
AFTER day t and check that the predictions for rows dated t don't change at all.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from council.analysts.base import Context
from council.analysts.pattern_oracle import PatternOracle
from council.analysts.raw_reader import RawReader


def _toy_prices(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", periods=n)
    rows = []
    for t in ["^DJI", "AAA", "BBB"]:
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        rows.append(pd.DataFrame({"date": days, "ticker": t, "open": c, "high": c, "low": c, "close": c,
                                  "adj_close": c, "raw_close": c, "volume": rng.uniform(1e6, 2e6, n)}))
    return pd.concat(rows, ignore_index=True)


def _toy_panel(prices: pd.DataFrame, h: int = 20) -> pd.DataFrame:
    out = []
    for _, g in prices.groupby("ticker"):
        g = g.sort_values("date")
        fwd = g.adj_close.shift(-h) / g.adj_close - 1
        out.append(pd.DataFrame({"date": g.date, "ticker": g.ticker, "fwd_return": fwd,
                                 "y": (fwd > 0).astype(float).where(fwd.notna())}))
    return pd.concat(out).dropna(subset=["y"]).reset_index(drop=True)


def _change_after(prices: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Scramble every price and volume dated after t."""
    p = prices.copy()
    later = p.date > t
    rng = np.random.default_rng(99)
    shock = rng.uniform(0.5, 1.5, later.sum())
    for col in ["open", "high", "low", "close", "adj_close", "raw_close", "volume"]:
        p.loc[later, col] = p.loc[later, col] * shock
    return p


@pytest.fixture(scope="module")
def toy():
    prices = _toy_prices()
    panel = _toy_panel(prices)
    t = prices.date.sort_values().unique()[250]
    rows = panel[panel.date == t].reset_index(drop=True)
    return prices, panel, pd.Timestamp(t), rows


def test_pattern_oracle_ignores_the_future(toy):
    prices, panel, t, rows = toy
    po = PatternOracle(epochs=1, embed_dim=16, layers=1, heads=2, sampled_futures=20)
    po.fit(panel[panel.date <= t - pd.Timedelta(days=40)], Context(prices=prices))
    before = po.predict_proba(rows, Context(prices=prices))
    after = po.predict_proba(rows, Context(prices=_change_after(prices, t)))
    assert np.array_equal(before, after)
    assert ((before >= 0) & (before <= 1)).all()
    # ...and the prediction does depend on the past: changing day t itself changes the context
    X1, _ = po._contexts(rows, Context(prices=prices))
    X2, _ = po._contexts(rows, Context(prices=_change_after(prices, t - pd.Timedelta(days=1))))
    assert not np.array_equal(X1, X2)
    f = po.futures("AAA", t, Context(prices=prices), n=7)
    assert f.shape == (21, 7) and np.allclose(f.iloc[0], 1.0)
    # saving keeps the weights as a state_dict and gives the same answers after loading
    assert np.array_equal(pickle.loads(pickle.dumps(po)).predict_proba(rows, Context(prices=prices)), before)


def test_pattern_oracle_fast_sampling_matches_full_forward():
    """The cached one-token-at-a-time step must give the same scores as running the whole sequence."""
    torch.manual_seed(0)
    po = PatternOracle(embed_dim=16, layers=2, heads=2)
    net = po._make_net().eval()
    x = torch.randint(0, 64, (3, 124))
    with torch.no_grad():
        full, _ = net(x)
        _, prefixes = net(x[:, :120])
        hd = po.d // po.heads
        gens = [(torch.zeros(3, 1, po.heads, 20, hd), torch.zeros(3, 1, po.heads, 20, hd)) for _ in net.blocks]
        for i in range(4):
            step = net.step(x[:, 120 + i].view(3, 1), prefixes, gens, pos=120 + i, n=i)
            assert torch.allclose(step[:, 0], full[:, 120 + i], atol=1e-5)


def test_raw_reader_ignores_the_future(toy):
    prices, panel, t, rows = toy
    rr = RawReader(epochs=1)
    rr.fit(panel[panel.date <= t - pd.Timedelta(days=40)], Context(prices=prices))
    before = rr.predict_proba(rows, Context(prices=prices))
    after = rr.predict_proba(rows, Context(prices=_change_after(prices, t)))
    assert np.array_equal(before, after)
    changed = rr.predict_proba(rows, Context(prices=_change_after(prices, t - pd.Timedelta(days=5))))
    assert not np.array_equal(before, changed)
    reasons = rr.explain(rows.iloc[0], Context(prices=prices))
    assert {k for k, _ in reasons} == {"last 20 days of the chart", "earlier 100 days of the chart"}
