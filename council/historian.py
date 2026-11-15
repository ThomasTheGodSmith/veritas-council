"""Historian: the point-in-time memory every analyst asks for evidence (PLAN.md §1, §4).

The one rule: on a decision date D, nothing dated on or after D exists. A past row's OUTCOME
(its 20-day return) only counts as known once its exit day is also before D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Historian:
    def __init__(self, panel: pd.DataFrame, predictions: pd.DataFrame | None = None):
        self.panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
        cols = ["date", "ticker", "agent", "p", "y"]
        self.preds = predictions if predictions is not None else pd.DataFrame(columns=cols)

    # ---------------------------------------------------------------- the point-in-time rule
    def history(self, decision_date, ticker: str | None = None) -> pd.DataFrame:
        """Every row dated strictly before the decision date (optionally one ticker)."""
        D = pd.Timestamp(decision_date)
        h = self.panel[self.panel.date < D]
        return h if ticker is None else h[h.ticker == ticker]

    def latest(self, ticker: str, decision_date) -> pd.Series:
        """The newest row before the decision date: what an analyst predicts from."""
        h = self.history(decision_date, ticker)
        if h.empty:
            raise LookupError(f"no data for {ticker} before {decision_date}")
        return h.iloc[-1]

    def known_outcomes(self, decision_date, ticker: str | None = None) -> pd.DataFrame:
        """Past rows whose 20-day outcome was already known before the decision date."""
        D = pd.Timestamp(decision_date)
        h = self.history(D, ticker)
        return h[h.label_known.notna() & (h.label_known < D) & h.y.notna()]

    # ---------------------------------------------------------------- evidence
    def similar_setups(self, ticker: str, decision_date, features: list[str], k: int = 30,
                       same_ticker: bool = False) -> pd.DataFrame:
        """The k past situations most like today's (standardised distance on `features`),
        with what actually happened next. Only outcomes known before the decision date."""
        now = self.latest(ticker, decision_date)
        past = self.known_outcomes(decision_date, ticker if same_ticker else None).dropna(subset=features)
        if past.empty:
            return past
        X = past[features].to_numpy(float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        x0 = (now[features].to_numpy(float) - mu) / sd
        d = np.sqrt((((X - mu) / sd - x0) ** 2).sum(1))
        out = past.assign(distance=d).nsmallest(k, "distance")
        return out[["date", "ticker", "distance", "fwd_return", "y", *features]]

    def record(self, predictions: pd.DataFrame) -> None:
        """Store analysts' predictions (date, ticker, agent, p, y) for track records."""
        self.preds = pd.concat([self.preds, predictions], ignore_index=True)

    def track_record(self, agent: str, decision_date, ticker: str | None = None) -> dict[str, float]:
        """How often this analyst was right, using only outcomes known before the decision date."""
        D = pd.Timestamp(decision_date)
        p = self.preds[(self.preds.agent == agent) & (self.preds.date < D) & self.preds.y.notna()]
        if "label_known" in p:
            p = p[p.label_known < D]
        if ticker is not None:
            p = p[p.ticker == ticker]
        if p.empty:
            return {}
        right = ((p.p >= 0.5) == (p.y == 1)).mean()
        return {"accuracy": round(float(right), 4), "n": float(len(p)),
                "brier": round(float(((p.p - p.y) ** 2).mean()), 4)}
