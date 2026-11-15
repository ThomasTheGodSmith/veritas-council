"""Profit in money for the example user, with realistic stop-loss fills (council/execution.py).

Example user: budget S$500, max loss S$50, holds 20 trading days. Every 5 trading days, for the
index and each member that day, each strategy decides how much to buy:
  Always buy               S$500 every time, no stop
  Random, same buy rate    S$500 at random, as often as the council buys
  Always buy + stop-loss   S$500 every time, stop at -S$50 (10%)
  Council                  the strategy tuned on this stock's validation years (council/strategy.py):
                           direction check, risk-budget sizing, gap-aware stop
Stops fill realistically: a day that opens below the stop sells at that open (a gap), otherwise
at the stop price. Results: total profit, profit per S$100 invested, how many purchases lost more
than S$50 when sold, the worst single loss, and S$500 compounded with one decision every 20
trading days (index rows only).

  uv run python -m eval.money validation
  uv run python -m eval.money test
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from council.execution import simulate
from council.paths import MODELS, RESULTS
from council.strategy import BUDGET, MAX_LOSS, STEP, gap_for, load_strategy, plan

INDEX = "^DJI"


def validation_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    from council.data import train_panel
    from council.splits import load_train, read_split, validation_mask
    p = train_panel()
    p = p[validation_mask(p.date, read_split()).values & p.y.notna()]
    days = sorted(p.loc[p.ticker == INDEX, "date"].unique())[::STEP]
    p = p[p.date.isin(days)]
    oof = pd.read_parquet(MODELS / "council_oof.parquet")
    risk = pd.read_parquet(MODELS / "risk_predictions" / "drawdown_risk.parquet")[["date", "ticker", "p"]]
    p = p.merge(oof, on=["date", "ticker"]).merge(risk.rename(columns={"p": "risk"}), on=["date", "ticker"])
    return p.reset_index(drop=True), load_train("prices")


def test_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    from council.analysts.base import Analyst
    from council.mediator import council_probabilities
    from council.splits import load_test
    r = pd.read_parquet(RESULTS / "test_decisions.parquet")
    skip = {"p_simple_average", "p_devils_advocate", "p_council", "p_ablation_no_devils_advocate"}
    probs = {c[2:]: r[c].to_numpy() for c in r.columns if c.startswith("p_") and c not in skip}
    r["p_council"], _ = council_probabilities(r, probs)
    r["risk"] = Analyst.load("drawdown_risk").predict_loss(r)
    return r, load_test("prices", for_evaluation=True)


def strategies(rows: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    s = load_strategy()
    gap = rows.ticker.map(lambda t: gap_for(s["gaps_q95"], t)).to_numpy()
    amount, stop = plan(rows.p_council, rows.risk, gap, s["theta"], s["k"], s["avg_win"], s["avg_loss"], c=s["c"])
    full = np.full(len(rows), BUDGET)
    idx = (rows.ticker == INDEX).to_numpy()
    rate = np.where(idx, (amount[idx] > 0).mean() if idx.any() else 0, (amount[~idx] > 0).mean() if (~idx).any() else 0)
    rand = np.random.default_rng(0).random(len(rows)) < rate
    return {
        "Always buy": (full, None),
        "Random, same buy rate": (np.where(rand, BUDGET, 0.0), None),
        "Always buy + stop-loss": (full, np.full(len(rows), MAX_LOSS / BUDGET)),
        "Council": (amount, stop),
    }


def score(rows: pd.DataFrame, prices: pd.DataFrame, amount: np.ndarray, stop) -> dict[str, float]:
    sim = simulate(rows, prices, amount, stop)
    profit = sim.profit.to_numpy()
    bought = amount > 0
    out = {
        "purchases": int(bought.sum()),
        "invested": float(amount.sum()),
        "total_profit": float(profit.sum()),
        "profit_per_100": float(100 * profit.sum() / amount.sum()) if amount.sum() else np.nan,
        "lost_over_50": int((profit < -MAX_LOSS - 1e-9).sum()),
        "worst_loss": float(-profit.min()),
        "gapped_stops": int(sim.gapped.sum()),
    }
    idx = (rows.ticker == INDEX).to_numpy()
    if idx.any():   # S$500 compounded, one decision every 20 trading days
        a, pr = amount[idx][:: 20 // STEP], profit[idx][:: 20 // STEP]
        r = np.where(a > 0, pr / np.where(a > 0, a, 1), 0.0) * (a / BUDGET)
        eq = BUDGET * np.cumprod(1 + r)
        out["S500_becomes"] = float(eq[-1])
        out["worst_drop"] = float((eq / np.maximum.accumulate(eq) - 1).min())
    return out


def main(which: str) -> pd.DataFrame:
    rows, prices = validation_rows() if which == "validation" else test_rows()
    s = load_strategy()
    tab = pd.DataFrame({k: score(rows, prices, a, st) for k, (a, st) in strategies(rows).items()}).T
    RESULTS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(RESULTS / f"money_{which}.csv")
    print(f"{which}: {len(rows):,} decisions {rows.date.min().date()} to {rows.date.max().date()} | "
          f"council strategy (tuned on validation): theta {s['theta']}, "
          + (f"volatility-scaled stop C {s['c']}" if s["c"] is not None else f"risk budget K {s['k']}"))
    pd.set_option("display.width", 220)
    print(tab.round(2).to_string())
    return tab


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "validation")
