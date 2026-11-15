"""News-Reaction leakage tests: headlines dated on/after day t can't change the prediction for day t.

The analyst may only read headlines dated t-1 or earlier (see council/analysts/news_reaction.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from council.analysts.base import Context
from council.analysts.news_reaction import NewsReaction
from council.features import build_panel

WORDS = ("profit loss growth lawsuit merger recall upgrade downgrade record strike deal war peace "
         "rally slump earnings beat miss dividend layoffs expansion").split()


def _toy_prices(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2010-01-01", periods=n)
    rows = []
    for t in ["^DJI", "AAA", "BBB", "^VIX", "^TNX", "DX-Y.NYB", "CL=F"]:
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        rows.append(pd.DataFrame({"date": days, "ticker": t, "open": c, "high": c, "low": c, "close": c,
                                  "adj_close": c, "raw_close": c, "volume": 1e6}))
    return pd.concat(rows, ignore_index=True)


def _toy_news(days, tickers, per_day: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [(d, t, " ".join(rng.choice(WORDS, 6)), "toy", np.nan)
            for d in days for t in tickers for _ in range(per_day)]
    return pd.DataFrame(rows, columns=["date", "ticker", "headline", "source", "score"])


@pytest.fixture(scope="module")
def fitted():
    prices = _toy_prices()
    days = pd.DatetimeIndex(sorted(prices.date.unique()))
    macro = pd.concat([pd.DataFrame({"date": days, "series": s, "value": 1.0})
                       for s in ["DFF", "T10Y2Y", "CPIAUCSL", "UNRATE", "DCOILWTICO"]], ignore_index=True)
    membership = pd.DataFrame({"member": ["AAA", "BBB"], "company": ["A", "B"], "ticker": ["AAA", "BBB"],
                               "start_date": pd.to_datetime(["2010-01-01", "2010-01-01"]),
                               "end_date": pd.to_datetime([None, None])})
    panel = build_panel(prices, macro, membership)
    ctx = Context(prices=prices, macro=macro, membership=membership,
                  market_news=_toy_news(days, ["DJIA"], 5, 1), company_news=_toy_news(days, ["AAA", "BBB"], 2, 2))
    a = NewsReaction()
    a.fit(panel[panel.y.notna() & (panel.date < days[350])], ctx)
    return a, ctx, panel, days


def _with_extra_news(ctx: Context, frm: pd.Timestamp, seed: int) -> Context:
    """Same context, but every headline dated >= frm is replaced by different, louder news."""
    days = pd.date_range(frm, frm + pd.Timedelta(days=60))
    mk = pd.concat([ctx.market_news[ctx.market_news.date < frm], _toy_news(days, ["DJIA"], 25, seed)])
    co = pd.concat([ctx.company_news[ctx.company_news.date < frm], _toy_news(days, ["AAA", "BBB"], 10, seed + 1)])
    return Context(prices=ctx.prices, macro=ctx.macro, membership=ctx.membership, market_news=mk, company_news=co)


def test_news_on_or_after_day_t_cannot_change_the_prediction(fitted):
    a, ctx, panel, days = fitted
    t = days[400]
    rows = panel[panel.date == t]
    before = a.details(rows, ctx)
    after = a.details(rows, _with_extra_news(ctx, t, seed=7))   # news dated t, t+1, ... all changed
    pd.testing.assert_frame_equal(before, after)
    assert before.has_news.eq(1).all()


def test_yesterdays_news_does_count(fitted):
    """Sanity check that the test above isn't vacuous: news dated t-1 IS read."""
    a, ctx, panel, days = fitted
    t = days[400]
    rows = panel[panel.date == t]
    before = a.details(rows, ctx)
    after = a.details(rows, _with_extra_news(ctx, t - pd.Timedelta(days=1), seed=7))
    assert not np.allclose(before.n_headlines, after.n_headlines)


def test_retrieval_only_uses_outcomes_already_known(fitted):
    a, ctx, panel, days = fitted
    t = days[360]
    for ticker in ["AAA", "^DJI"]:
        for e in a.evidence(ticker, t, ctx, n=20):
            row = panel[(panel.ticker == ticker) & (panel.date == pd.Timestamp(e["date"]))].iloc[0]
            assert row.label_known <= t


def test_no_news_falls_back_to_base_rate(fitted):
    a, ctx, panel, days = fitted
    empty = Context(prices=ctx.prices, macro=ctx.macro, membership=ctx.membership,
                    market_news=ctx.market_news.iloc[:0], company_news=ctx.company_news.iloc[:0])
    rows = panel[panel.date == days[400]]
    d = a.details(rows, empty)
    assert d.has_news.eq(0).all()
    expected = np.where(rows.ticker == "^DJI", a.base_rate["index"], a.base_rate["member"])
    np.testing.assert_allclose(d.p.to_numpy(), expected)
