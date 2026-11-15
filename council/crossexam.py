"""Round 2 cross-examination: the debate, done with data (PLAN.md §1b "Round 2").

For every pair of analysts that disagree today (one says up, p >= 0.5; the other says down),
look up every past validation row where those two disagreed THE SAME WAY, and count who was
right: the "up" analyst wins if the price rose, the "down" analyst wins if it didn't.

We use the same stock's history when it has at least MIN_SAME_TICKER such days, otherwise all
stocks pooled together (more examples, less specific). Only outcomes already known before the
decision date count (the Historian's rule).
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from council.analysts.base import Signal
from council.devils_advocate import NOT_IN_MAJORITY, wide_probs
from council.schema import Disagreement

MIN_SAME_TICKER = 30


def nice(agent: str) -> str:
    return agent.replace("_", " ").title()


def known_wide(val_preds: pd.DataFrame, decision_date) -> pd.DataFrame:
    """Validation predictions, one column per analyst, only rows whose outcome is known before D."""
    D = pd.Timestamp(decision_date)
    v = val_preds[(val_preds.date < D) & val_preds.y.notna()]
    if "label_known" in v:
        v = v[v.label_known < D]
    return wide_probs(v) if len(v) else pd.DataFrame()


def cross_examine(signals: list[Signal], val_preds: pd.DataFrame, ticker: str, decision_date,
                  wide: pd.DataFrame | None = None) -> list[Disagreement]:
    if wide is None:
        wide = known_wide(val_preds, decision_date)
    sig = {s.agent: s for s in signals if s.agent not in NOT_IN_MAJORITY}
    out: list[Disagreement] = []
    for x, y in combinations(sorted(sig), 2):
        up_x, up_y = sig[x].probability_up >= 0.5, sig[y].probability_up >= 0.5
        if up_x == up_y:
            continue
        a, b = (x, y) if up_x else (y, x)                 # `a` is the one saying up today
        if wide.empty or a not in wide or b not in wide:
            out.append(Disagreement(a=a, b=b, a_says_up=True, n_past=0, a_win_rate=0.5,
                                    summary=f"{nice(a)} (up) vs {nice(b)} (down): no past record of these two yet"))
            continue
        same = (wide[a] >= 0.5) & (wide[b] < 0.5)
        past = wide[same & (wide.ticker == ticker)]
        where = f"on {ticker}"
        if len(past) < MIN_SAME_TICKER:
            past, where = wide[same], "across all Dow stocks"
        n = len(past)
        rate = float((past.y == 1).mean()) if n else 0.5
        winner, share = (a, rate) if rate >= 0.5 else (b, 1 - rate)
        summary = (f"{nice(a)} (up) vs {nice(b)} (down): {nice(winner)} won {share:.0%} of {n:,} past "
                   f"disagreements like this {where} (2016-2019)") if n else \
                  f"{nice(a)} (up) vs {nice(b)} (down): they never disagreed this way before"
        out.append(Disagreement(a=a, b=b, a_says_up=True, n_past=n, a_win_rate=round(rate, 4), summary=summary))
    return sorted(out, key=lambda d: -d.n_past)
