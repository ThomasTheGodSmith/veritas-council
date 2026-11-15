"""What every analyst is and returns (PLAN.md §1, "What every analyst returns").

An analyst is a model with two jobs:
  fit(panel, ctx)            learn from training rows (they all have labels: did it go up?)
  predict_proba(panel, ctx)  for each row, the probability the price is higher in h days

`panel` rows come from council/features.py. `ctx` holds the raw tables the analyst is allowed to
see right now: the training folder during training, warm-up + test years during evaluation,
and everything up to today in the app. For a row dated t, an analyst may only use data dated <= t.

Each analyst file defines `build() -> Analyst`. train.py finds them through config/agents.yaml.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from council.paths import MODELS  # noqa: E402  (models/ or models/scenarios/<stock>/)


class Signal(BaseModel):
    agent: str
    probability_up: float
    horizon_days: int = 20
    expected_return_pct: float
    recommended_action: Literal["buy", "wait", "avoid"]
    user_idea_verdict: Literal["supports", "challenges"]
    key_numbers: dict[str, float] = {}
    top_reasons: list[tuple[str, float]] = []
    track_record: dict[str, float] = {}


@dataclass
class Context:
    """Raw tables available to analysts at this moment (never more than is allowed)."""
    prices: pd.DataFrame
    macro: pd.DataFrame | None = None
    membership: pd.DataFrame | None = None
    market_news: pd.DataFrame | None = None
    company_news: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)


class Analyst:
    name: str = "analyst"
    features: list[str] = []

    # learned at fit time, used for expected return and the Judge's Kelly sizing
    avg_win: float = 0.03
    avg_loss: float = 0.03

    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        raise NotImplementedError

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        raise NotImplementedError

    def explain(self, row: pd.Series, ctx: Context) -> list[tuple[str, float]]:
        """Top reasons for this row's prediction, as (feature, weight) pairs."""
        return []

    def key_numbers(self, row: pd.Series, ctx: Context) -> dict[str, float]:
        return {f: float(row[f]) for f in self.features if f in row and pd.notna(row[f])}

    # ------------------------------------------------------------------ shared helpers
    def _learn_payoffs(self, panel: pd.DataFrame) -> None:
        r = panel["fwd_return"].dropna()
        self.avg_win = float(r[r > 0].mean())
        self.avg_loss = float(-r[r <= 0].mean())

    def expected_return(self, p: np.ndarray) -> np.ndarray:
        return p * self.avg_win - (1 - p) * self.avg_loss

    def save(self) -> Path:
        MODELS.mkdir(exist_ok=True)
        path = MODELS / f"{self.name}.pkl"
        path.write_bytes(pickle.dumps(self))
        return path

    @staticmethod
    def load(name: str) -> "Analyst":
        return pickle.loads((MODELS / f"{name}.pkl").read_bytes())


class LogisticAnalyst(Analyst):
    """Logistic regression on a few named features: p = sigmoid(w0 + sum(w_i * x_i)).
    Missing values are filled with the training median; features are standardised."""

    C: float = 1.0

    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        df = panel.dropna(subset=["y"])
        self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=self.C, max_iter=1000))
        self.model.fit(self.transform(df), df["y"].astype(int))
        self._learn_payoffs(df)

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        return panel[self.features]

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        return self.model.predict_proba(self.transform(panel))[:, 1]

    def explain(self, row: pd.Series, ctx: Context) -> list[tuple[str, float]]:
        imp, scale, lr = self.model.steps[0][1], self.model.steps[1][1], self.model.steps[2][1]
        x = scale.transform(imp.transform(self.transform(row.to_frame().T)))[0]
        contrib = x * lr.coef_[0]
        order = np.argsort(-np.abs(contrib))
        return [(self.transform(row.to_frame().T).columns[i], float(contrib[i])) for i in order]


def action_for(p: float, buy_above: float = 0.55, avoid_below: float = 0.45) -> str:
    return "buy" if p > buy_above else "avoid" if p < avoid_below else "wait"


def make_signal(analyst: Analyst, row: pd.Series, p: float, ctx: Context,
                user_thinks_up: bool = True, track_record: dict[str, float] | None = None,
                horizon_days: int = 20) -> Signal:
    supports = (p >= 0.5) == user_thinks_up
    return Signal(
        agent=analyst.name,
        probability_up=round(float(p), 4),
        horizon_days=horizon_days,
        expected_return_pct=round(float(analyst.expected_return(np.array([p]))[0]) * 100, 2),
        recommended_action=action_for(p),
        user_idea_verdict="supports" if supports else "challenges",
        key_numbers={k: round(v, 4) for k, v in analyst.key_numbers(row, ctx).items()},
        top_reasons=[(k, round(w, 4)) for k, w in analyst.explain(row, ctx)[:3]],
        track_record=track_record or {},
    )
