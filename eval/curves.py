"""Growth of S$500 over the test years for each strategy, for one program (the first calendar).

Used by eval/results.ipynb for the "where it fails" chart. Applies the strategy that was fixed on
training years; nothing is tuned here. Evaluation code, so it may open the test folder.

Run:  VERITAS_SCENARIO=AMD uv run python -m eval.curves      (prints JSON)
"""
from __future__ import annotations

import json

import numpy as np

from council.analysts.base import Analyst
from council.execution import simulate
from council.features import build_panel
from council.splits import load_test, read_split
from council.strategy import BUDGET, MAX_LOSS, STEP, gap_for, load_strategy, plan

INDEX = "^DJI"


def main() -> dict:
    split = read_split()
    s = load_strategy()
    prices, macro, members = (load_test(n, for_evaluation=True) for n in ["prices", "macro", "membership"])
    panel = build_panel(prices, macro, members)
    panel = panel[(panel.date >= split.test_start) & panel.y.notna() & (panel.ticker == INDEX)]
    days = sorted(panel.date.unique())[:: STEP * 4]               # one decision every 20 trading days
    rows = panel[panel.date.isin(days)].reset_index(drop=True)
    risk = Analyst.load("drawdown_risk").predict_loss(rows)
    gap = rows.ticker.map(lambda t: gap_for(s["gaps_q95"], t)).to_numpy()
    c_amount, c_stop = plan(np.ones(len(rows)), risk, gap, s["theta"], s["k"], s["avg_win"], s["avg_loss"], c=s.get("c"))
    full = np.full(len(rows), BUDGET)
    out = {"dates": [str(d.date()) for d in rows.date]}
    for name, (amount, stop) in {"Always buy": (full, None),
                                 "Always buy + stop-loss": (full, np.full(len(rows), MAX_LOSS / BUDGET)),
                                 "Council": (c_amount, c_stop)}.items():
        profit = simulate(rows, prices, amount, stop).profit.to_numpy()
        r = np.where(amount > 0, profit / np.where(amount > 0, amount, 1.0), 0.0) * (amount / BUDGET)
        out[name] = [round(float(x), 2) for x in BUDGET * np.cumprod(1 + r)]
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main()
