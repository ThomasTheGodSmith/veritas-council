"""Leakage and sanity tests for the classic analysts (mean reversion, ML, risk, regime, macro, breadth).

The key check for each: changing prices AFTER day t must not change the prediction ON day t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from council.analysts import (dow_breadth, macro, market_regime, mean_reversion, ml_forecaster, momentum,
                              risk_officer)
from council.analysts.base import Context
from council.features import build_panel
from tests.test_no_leakage import _toy_macro, _toy_prices

N_DAYS = 900


def _membership():
    return pd.DataFrame({"member": ["AAA", "BBB"], "company": ["A", "B"], "ticker": ["AAA", "BBB"],
                         "start_date": pd.to_datetime(["2010-01-01", "2010-01-01"]),
                         "end_date": pd.to_datetime([None, None])})


def _world(prices):
    macro_df = _toy_macro(pd.DatetimeIndex(prices.date.unique()))
    macro_df["value"] = macro_df["value"] + np.random.default_rng(1).normal(0, 0.1, len(macro_df))
    panel = build_panel(prices, macro_df, _membership())
    return panel, Context(prices=prices, macro=macro_df, membership=_membership())


def _shocked_after(prices, t):
    later = prices.copy()
    sel = later.date > t
    cols = ["open", "high", "low", "close", "adj_close", "raw_close"]
    shock = np.random.default_rng(7).uniform(0.5, 2.0, sel.sum())     # scramble every later price
    later.loc[sel, cols] = later.loc[sel, cols].mul(shock, axis=0)
    return later


@pytest.fixture(scope="module")
def world():
    prices = _toy_prices(N_DAYS, seed=3)
    panel, ctx = _world(prices)
    t = pd.Timestamp(sorted(prices.date.unique())[700])
    fit_rows = panel[(panel.date <= pd.Timestamp(sorted(prices.date.unique())[500])) & panel.y.notna()]
    later_panel, later_ctx = _world(_shocked_after(prices, t))
    return dict(prices=prices, panel=panel, ctx=ctx, t=t, fit_rows=fit_rows,
                later_panel=later_panel, later_ctx=later_ctx)


def _fitted(mod, world, extra=None):
    a = mod.build()
    world["ctx"].extra = extra or {}
    a.fit(world["fit_rows"], world["ctx"])
    return a


def _p_on_t(a, panel, ctx, t):
    rows = panel[panel.date == t].sort_values("ticker")
    return a.predict_proba(rows, ctx)


@pytest.mark.parametrize("mod", [mean_reversion, ml_forecaster, macro, risk_officer, market_regime])
def test_prediction_on_t_ignores_later_prices(mod, world):
    a = _fitted(mod, world)
    p1 = _p_on_t(a, world["panel"], world["ctx"], world["t"])
    p2 = _p_on_t(a, world["later_panel"], world["later_ctx"], world["t"])
    assert np.all((p1 > 0) & (p1 < 1))
    np.testing.assert_allclose(p1, p2)


def test_garch_sigma_ignores_later_prices(world):
    a = _fitted(risk_officer, world)
    rows = world["panel"][world["panel"].date == world["t"]]
    s1 = a.sigma_for_rows(rows, world["ctx"])
    s2 = a.sigma_for_rows(world["later_panel"][world["later_panel"].date == world["t"]], world["later_ctx"])
    np.testing.assert_allclose(np.sort(s1), np.sort(s2))
    assert (s1 > 0).all()


def test_risk_report(world):
    a = _fitted(risk_officer, world)
    rep = a.risk_report("AAA", world["t"], world["ctx"], amount=1000, max_loss=100)
    assert rep["paths"].shape == (10_000, 20)
    assert rep["p05"] < rep["p50"] < rep["p95"]
    assert 0 <= rep["p_hit_stop"] <= 1
    assert rep["max_safe_amount"] == pytest.approx(100 / rep["var95_pct"])
    later = a.risk_report("AAA", world["t"], world["later_ctx"], amount=1000, max_loss=100)
    assert later["sigma_daily"] == pytest.approx(rep["sigma_daily"])
    assert later["p50"] == pytest.approx(rep["p50"])


def test_regime_on_ignores_later_prices(world):
    a = _fitted(market_regime, world)
    r1 = a.regime_on(world["t"], world["ctx"])
    r2 = a.regime_on(world["t"], world["later_ctx"])
    assert set(r1["probs"]) == {"bull", "bear", "choppy"}
    assert r1["probs"] == pytest.approx(r2["probs"])
    assert sum(r1["probs"].values()) == pytest.approx(1)


def test_dow_breadth_uses_only_same_day_rows(world):
    fast = {"momentum": momentum.build(), "mean_reversion": mean_reversion.build()}
    a = _fitted(dow_breadth, world, extra={"analysts": fast})
    p1 = _p_on_t(a, world["panel"], world["ctx"], world["t"])
    p2 = _p_on_t(a, world["later_panel"], world["later_ctx"], world["t"])
    np.testing.assert_allclose(p1, p2)
    # predicting one day on its own gives the same answer as predicting it inside a bigger panel
    big = world["panel"][world["panel"].date >= world["t"] - pd.Timedelta(days=30)]
    p_big = a.predict_proba(big, world["ctx"])[(big.date == world["t"]).to_numpy()]
    order = np.argsort(big[big.date == world["t"]].ticker.to_numpy())
    np.testing.assert_allclose(p_big[order], p1)
