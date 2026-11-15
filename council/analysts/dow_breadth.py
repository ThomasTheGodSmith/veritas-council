"""Analyst 10, Dow Breadth & Weights (PLAN.md §1b-10): "what do the 30 members add up to?".

The Dow is price-weighted, so heavy (high-priced) members move it most.

1. Fast models (Momentum and Mean Reversion) give each member i on day t a probability p_i.
   They are refitted inside this analyst on the same rows it is fitted on, so its validation
   score never benefits from models that saw the validation years.
   The ML Forecaster is left out on purpose: on its own training rows LightGBM is much more
   confident than on new data, so the stage below learned to over-trust it and gave p = 0.002
   on some validation days (Brier 0.240 vs 0.237 without it). The two logistic models have
   only 2-3 weights each, so their training-row predictions are as honest as new ones.
2. Bottom-up Dow forecast, weighted by each member's Dow weight w_i that day
   (w_i = price_i / sum of members' prices, renormalised over the members we have):
       p_bottom_up = sigmoid( sum_i w_i * logit(p_i) )
   where logit(p) = ln(p / (1 - p)), and each member's logit is first averaged over the fast models.
3. Index rows:  p = sigmoid(a + b*logit(p_bottom_up) + c*(breadth_50 - 0.5) + d*divergence)
     breadth_50  share of members above their 50-day average
     divergence  1 when the index is near its 1-year high but breadth fell 10+ points in 20 days
4. Member rows: p = sigmoid(a + b*RS + c*rank + e*logit(p_bottom_up))
     RS    the stock's 60-day return minus the average 60-day return of the OTHER members that day
     rank  where its 60-day return ranks among that day's members (0 = worst, 1 = best)
   (PLAN.md compares with sector peers; we compare with all other Dow members, because the
   datasets have no sector labels.)
Only same-day information about other members is used, so nothing looks ahead.
The panel passed to predict_proba must contain that day's member rows for the bottom-up
forecast; days with no member rows get p_bottom_up = 0.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from council.analysts.base import Analyst, Context

INDEX = "^DJI"
FAST = ["momentum", "mean_reversion"]
INDEX_FEATURES = ["logit_bottom_up", "breadth_centred", "divergence"]
MEMBER_FEATURES = ["rel_strength_60", "rank_60", "logit_bottom_up"]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


class DowBreadth(Analyst):
    name = "dow_breadth"
    features = ["breadth_50", "divergence", "dow_weight", "ret_60"]

    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        available = ctx.extra.get("analysts", {})
        self.fast = {}
        for name in FAST:
            if name in available:
                model = type(available[name])()          # a fresh copy, fitted on these rows only
                model.fit(panel, ctx)
                self.fast[name] = model
        if not self.fast:
            raise RuntimeError("dow_breadth needs at least one of momentum / mean_reversion / ml_forecaster trained first")

        df = self.transform(panel.dropna(subset=["y"]), ctx)
        is_idx = df.ticker.eq(INDEX)
        self.index_model = _logistic().fit(df.loc[is_idx, INDEX_FEATURES], df.loc[is_idx, "y"].astype(int))
        self.member_model = _logistic().fit(df.loc[~is_idx, MEMBER_FEATURES], df.loc[~is_idx, "y"].astype(int))
        self._learn_payoffs(panel.dropna(subset=["y"]))

    # ------------------------------------------------------------------ features
    def transform(self, panel: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        df = panel.copy()
        is_mem = df.ticker.ne(INDEX)
        # each member's average logit over the fast models
        member_logit = np.mean([logit(m.predict_proba(df[is_mem], ctx)) for m in self.fast.values()], axis=0)
        mem = df.loc[is_mem, ["date", "dow_weight", "ret_60"]].assign(logit_p=member_logit)
        mem["w"] = mem["dow_weight"] / mem.groupby("date")["dow_weight"].transform("sum")
        bottom_up = (mem["w"] * mem["logit_p"]).groupby(mem["date"]).sum()
        df["logit_bottom_up"] = df["date"].map(bottom_up).fillna(0.0).to_numpy()

        # relative strength vs the OTHER members that day, and the rank among members
        g = mem.groupby("date")["ret_60"]
        n = g.transform("count")
        others_mean = (g.transform("sum") - mem["ret_60"]) / (n - 1).replace(0, np.nan)
        df.loc[is_mem, "rel_strength_60"] = mem["ret_60"] - others_mean
        df.loc[is_mem, "rank_60"] = g.rank(pct=True)
        df["breadth_centred"] = df["breadth_50"] - 0.5
        return df

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        df = self.transform(panel, ctx)
        is_idx = df.ticker.eq(INDEX).to_numpy()
        p = np.full(len(df), 0.5)
        if is_idx.any():
            p[is_idx] = self.index_model.predict_proba(df.loc[is_idx, INDEX_FEATURES])[:, 1]
        if (~is_idx).any():
            p[~is_idx] = self.member_model.predict_proba(df.loc[~is_idx, MEMBER_FEATURES])[:, 1]
        return p

    def explain(self, row: pd.Series, ctx: Context) -> list[tuple[str, float]]:
        """Needs the row's computed features; the council can pass a row from transform()."""
        is_idx = row["ticker"] == INDEX
        feats, model = (INDEX_FEATURES, self.index_model) if is_idx else (MEMBER_FEATURES, self.member_model)
        if not all(f in row for f in feats):
            return []
        imp, scale, lr = model.steps[0][1], model.steps[1][1], model.steps[2][1]
        x = scale.transform(imp.transform(row[feats].to_frame().T.astype(float)))[0]
        contrib = x * lr.coef_[0]
        order = np.argsort(-np.abs(contrib))
        return [(feats[i], float(contrib[i])) for i in order]


def build() -> DowBreadth:
    return DowBreadth()
