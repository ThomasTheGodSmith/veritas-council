"""Checklist (PLAN.md §1 "Checklist output"): runs after the cross-examination, before the Mediator.

Part 1, the rule check, lives in council/rules.py (code, can't be overruled).
Part 2, here: positives and negatives of every analyst, from its REAL track record on the
validation years (2016-2019, predictions made by models that had not seen those years):

  overall       right X% of the time vs Y% for "always say up"          (all stocks)
  this stock    right X% of the time on this stock
  today's mood  right X% on this stock when the trend and volatility were like today's
                ("uptrend/downtrend" = 20-day average above/below the 50-day;
                 "high/low volatility" = 20-day volatility above/below the validation median)
  signal today  how strong today's call is compared with its own history (top 10% = strong)
  debate        won or lost past disagreements like today's (from crossexam.py)

evidence_strength = how many past examples back today's-mood claim: weak < 30, medium 30-100,
strong > 100. trust_score = today's-mood accuracy pulled towards 50% when there are few
examples:  (correct + 0.5 * 20) / (n + 20).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from council.analysts.base import Signal
from council.crossexam import nice
from council.devils_advocate import NOT_IN_MAJORITY
from council.schema import AnalystReview, Disagreement

PRIOR_N = 20


def strength(n: int) -> str:
    return "weak" if n < 30 else "medium" if n <= 100 else "strong"


def mood(trend: float, vol: float, vol_median: float) -> tuple[str, str]:
    t = "uptrend" if trend > 0 else "downtrend"
    v = "high volatility" if vol > vol_median else "low volatility"
    return t, v


def _acc(p: pd.Series, y: pd.Series) -> float:
    return float(((p >= 0.5) == (y == 1)).mean()) if len(p) else float("nan")


def review_analysts(signals: list[Signal], wide: pd.DataFrame, ticker: str, today: pd.Series,
                    disagreements: list[Disagreement], buy_allowed: bool) -> list[AnalystReview]:
    """wide: known validation predictions, one column per analyst, joined with trend and vol_20.
    today: today's panel row. buy_allowed: does any buying plan pass the user's rules?"""
    vol_med = float(wide["vol_20"].median()) if "vol_20" in wide and len(wide) else 0.01
    t_word, v_word = mood(today.get("trend", 1.0), today.get("vol_20", vol_med), vol_med)
    reviews = []
    for s in signals:
        if s.agent in NOT_IN_MAJORITY or s.agent not in wide:
            reviews.append(AnalystReview(agent=s.agent, positives=[], fits_user_rules=_fits(s, buy_allowed),
                                         negatives=["No out-of-sample track record (it learns from the "
                                                    "validation years itself)"] if s.agent == "devils_advocate"
                                         else ["No validation track record yet"]))
            continue
        h = wide.dropna(subset=[s.agent])
        pos, neg = [], []

        acc_all, base = _acc(h[s.agent], h.y), float((h.y == 1).mean())
        if acc_all > base:
            pos.append(f"Right {acc_all:.0%} of the time on 2016-2019 (n={len(h):,}), better than "
                       f"{base:.0%} for always saying up")
        else:
            neg.append(f"Right {acc_all:.0%} of the time on 2016-2019 (n={len(h):,}): no better than "
                       f"always saying up ({base:.0%})")

        mine = h[h.ticker == ticker]
        if len(mine):
            acc_t = _acc(mine[s.agent], mine.y)
            base_t = float((mine.y == 1).mean())
            if acc_t > base_t:
                pos.append(f"Right {acc_t:.0%} on {ticker} (n={len(mine):,}), better than {base_t:.0%} for always up")
            else:
                neg.append(f"Right {acc_t:.0%} on {ticker} (n={len(mine):,}): no better than always up ({base_t:.0%})")

        same_mood = mine[(np.sign(mine.trend) == (1 if t_word == "uptrend" else -1)) &
                         ((mine.vol_20 > vol_med) == (v_word == "high volatility"))]
        n_mood = len(same_mood)
        correct = float(((same_mood[s.agent] >= 0.5) == (same_mood.y == 1)).sum())
        trust = (correct + 0.5 * PRIOR_N) / (n_mood + PRIOR_N)
        if n_mood:
            acc_m = correct / n_mood
            line = f"Right {acc_m:.0%} on {ticker} in a {t_word} with {v_word}, like today (n={n_mood})"
            (pos if acc_m >= 0.55 else neg).append(line)

        conviction = np.abs(h[s.agent] - 0.5)
        pct = float((conviction < abs(s.probability_up - 0.5)).mean())
        if pct >= 0.9:
            pos.append(f"Strong signal today (top {max(1, round(100 * (1 - pct)))}% of its history)")
        elif pct <= 0.2:
            neg.append("Weak signal today (one of its least confident calls)")

        for d in disagreements:
            if s.agent not in (d.a, d.b) or d.n_past == 0:
                continue
            other = d.b if s.agent == d.a else d.a
            won = d.a_win_rate if s.agent == d.a else 1 - d.a_win_rate
            line = (f"{'Won' if won >= 0.5 else 'Lost'} {won:.0%} of {d.n_past:,} past disagreements "
                    f"like this with {nice(other)}")
            (pos if won >= 0.5 else neg).append(line)

        reviews.append(AnalystReview(agent=s.agent, positives=pos, negatives=neg,
                                     evidence_strength=strength(n_mood), fits_user_rules=_fits(s, buy_allowed),
                                     trust_score=round(trust, 4)))
    return reviews


def _fits(s: Signal, buy_allowed: bool) -> bool:
    """An analyst saying "buy" only fits the user's rules if some buying plan passes them."""
    return buy_allowed or s.recommended_action != "buy"
