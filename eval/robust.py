"""Robust test-year results: the same money test, averaged over 20 fair decision calendars.

Why 20 and not "train 12 times": every model uses a fixed random seed, so retraining reproduces
exactly the same models, and with the tuned strategy the buy decisions don't depend on them anyway.
The luck that DOES matter is the calendar: decisions are made every 5 trading days, and the S$500
compounding uses every 20th trading day. There are 5 x 4 = 20 equally fair ways to place those
days. We score all 20 and report the average, the worst and the best, so no result depends on a
lucky start date. The random baseline gets a fresh random draw in each of the 20.

Nothing here is tuned: the strategy was fixed on training years (council/strategy.py) and is only
applied. This is evaluation code, so it may open the test folder.

Run:  uv run python -m eval.robust                         (the program in VERITAS_SCENARIO, or the Dow)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from council.analysts.base import Analyst
from council.execution import simulate
from council.features import build_panel
from council.paths import RESULTS, SCENARIO
from council.splits import load_test, read_split
from council.strategy import BUDGET, MAX_LOSS, STEP, gap_for, load_strategy, plan

INDEX = "^DJI"
CHAIN = 20 // STEP          # compounding uses every 4th decision (one every 20 trading days)
METRICS = ["S500_becomes", "profit_pct", "worst_drop", "lost_over_50", "worst_loss", "total_profit",
           "profit_per_100", "buy_rate"]


def council_p(rows: pd.DataFrame, strategy: dict) -> np.ndarray:
    """The council's probability of a rise, only computed if the strategy's direction check uses it."""
    if strategy["theta"] <= 0:
        return np.ones(len(rows))
    from council.analysts.base import Context
    from council.mediator import council_probabilities
    from train import analyst_specs
    t = lambda n: load_test(n, for_evaluation=True)  # noqa: E731
    ctx = Context(prices=t("prices"), macro=t("macro"), membership=t("membership"),
                  market_news=t("market_news"), company_news=t("company_news"))
    probs = {}
    for spec in analyst_specs():
        if not spec.get("votes", True) or spec["name"] == "devils_advocate":
            continue
        try:
            probs[spec["name"]] = Analyst.load(spec["name"]).predict_proba(rows, ctx)
        except FileNotFoundError:
            continue
    p, _ = council_probabilities(rows, probs)
    return p


def chain_stats(amount: np.ndarray, profit: np.ndarray) -> tuple[float, float]:
    r = np.where(amount > 0, profit / np.where(amount > 0, amount, 1.0), 0.0) * (amount / BUDGET)
    eq = BUDGET * np.cumprod(1 + r)
    return float(eq[-1]), float((eq / np.maximum.accumulate(eq) - 1).min())


def main() -> dict:
    split = read_split()
    s = load_strategy()
    prices, macro, members = (load_test(n, for_evaluation=True) for n in ["prices", "macro", "membership"])
    panel = build_panel(prices, macro, members)
    panel = panel[(panel.date >= split.test_start) & panel.y.notna()].reset_index(drop=True)
    days = sorted(panel.loc[panel.ticker == INDEX, "date"].unique())
    risk = Analyst.load("drawdown_risk").predict_loss(panel)
    gap = panel.ticker.map(lambda t: gap_for(s["gaps_q95"], t)).to_numpy()
    p = council_p(panel, s)
    c_amount, c_stop = plan(p, risk, gap, s["theta"], s["k"], s["avg_win"], s["avg_loss"], c=s.get("c"))
    full = np.full(len(panel), BUDGET)
    # simulate every possible decision day once; each calendar then picks its own days
    sims = {
        "Always buy": (full, simulate(panel, prices, full, None).profit.to_numpy()),
        "Always buy + stop-loss": (full, simulate(panel, prices, full, np.full(len(panel), MAX_LOSS / BUDGET)).profit.to_numpy()),
        "Council": (c_amount, simulate(panel, prices, c_amount, c_stop).profit.to_numpy()),
    }
    full_profit = sims["Always buy"][1]
    is_index = (panel.ticker == INDEX).to_numpy()
    runs = {k: [] for k in ["Always buy", "Random, same buy rate", "Always buy + stop-loss", "Council"]}
    for o in range(STEP):
        on = panel.date.isin(days[o::STEP]).to_numpy()
        for j in range(CHAIN):
            rng = np.random.default_rng(o * CHAIN + j)
            rate = np.where(is_index, (c_amount[on & is_index] > 0).mean(), (c_amount[on & ~is_index] > 0).mean()
                            if (on & ~is_index).any() else 0.0)
            rand_amount = np.where(rng.random(len(panel)) < rate, BUDGET, 0.0)
            strategies = dict(sims)
            strategies["Random, same buy rate"] = (rand_amount, np.where(rand_amount > 0, full_profit, 0.0))
            for name, (amount, profit) in strategies.items():
                a, pr = amount[on], profit[on]
                idx = is_index[on]
                ca, cp = a[idx][j::CHAIN], pr[idx][j::CHAIN]
                becomes, drop = chain_stats(ca, cp)
                runs[name].append({
                    "S500_becomes": becomes, "profit_pct": becomes / BUDGET - 1, "worst_drop": drop,
                    "lost_over_50": int((pr < -MAX_LOSS - 1e-9).sum()), "worst_loss": float(-pr.min()),
                    "total_profit": float(pr.sum()),
                    "profit_per_100": float(100 * pr.sum() / a.sum()) if a.sum() else float("nan"),
                    "buy_rate": float((a > 0).mean()),
                })
    out = {"program": SCENARIO or "DOW", "calendars": STEP * CHAIN, "decisions_per_calendar": int(len(panel) / STEP),
           "test_window": [str(panel.date.min().date()), str(panel.date.max().date())],
           "strategy": {"theta": s["theta"], "k": s["k"], "c": s.get("c")}, "results": {}}
    for name, rs in runs.items():
        df = pd.DataFrame(rs)
        out["results"][name] = {m: {"mean": float(df[m].mean()), "min": float(df[m].min()), "max": float(df[m].max())}
                                for m in METRICS}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "robust.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, r in out["results"].items():
        b, d, lo = r["S500_becomes"], r["worst_drop"], r["lost_over_50"]
        print(f"{name:24s} S$500 -> S${b['mean']:,.0f} (range {b['min']:,.0f}-{b['max']:,.0f}) | worst drop "
              f"{d['mean']:.0%} | lost >S$50 {lo['mean']:.1f}")
    return out


if __name__ == "__main__":
    main()
