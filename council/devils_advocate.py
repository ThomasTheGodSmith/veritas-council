"""Analyst 11, Devil's Advocate (PLAN.md §1b-11): a model trained to spot when the majority is wrong.

Training data (the validation slice, 2016-2019 with the current split):
  For every (date, ticker) row, take the votes of analysts 1-10 (each says "up" if p >= 0.5).
  The majority direction is up if more than half say up (a tie is broken by the average p).
  y_wrong = 1 if the majority's direction turned out wrong.

Model: a small LightGBM classifier, p_wrong = sigmoid(sum_k f_k(z)), on
  z = vote_share_up     share of analysts saying up (how split the vote is)
      vote_spread       how far apart their probabilities are (standard deviation)
      mean_p            the average probability
      regime_p          the Market Regime analyst's probability, if it exists
      vol_20, breadth_50, dist_high_250, vix   today's market conditions

Its vote: AGAINST the majority, as strong as p_wrong.
      majority says up   -> probability_up = 0.5 - 0.5 * p_wrong
      majority says down -> probability_up = 0.5 + 0.5 * p_wrong
So it always argues the other side; how loudly depends on how often the majority has been wrong
in situations like today's. The Mediator uses p_wrong itself as an input (PLAN.md §1b).

IMPORTANT (honesty): this model is TRAINED on the validation slice, so its own score on the
validation slice is in-sample, not a fair test. train.py calls fit() with the fit-year rows;
we ignore them and use ctx.extra["validation_predictions"] instead (that is where the other
analysts' out-of-sample votes are). Its fair score is the test-years backtest. For the Mediator,
which is ALSO trained on the validation slice, we use out-of-fold p_wrong (leave-one-year-out,
`out_of_fold_p_wrong`) so the Mediator never learns from p_wrong values that saw their own answer.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from council.analysts.base import Analyst, Context

NOT_IN_MAJORITY = {"devils_advocate", "fundamentals_card"}   # analysts 1-10 only
CONTEXT_FEATURES = ["vol_20", "breadth_50", "dist_high_250", "vix"]
VOTE_FEATURES = ["vote_share_up", "vote_spread", "mean_p", "regime_p"]
FEATURES = VOTE_FEATURES + CONTEXT_FEATURES
LGB_PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=8, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)


# ------------------------------------------------------------------ shared vote helpers
def wide_probs(val_preds: pd.DataFrame) -> pd.DataFrame:
    """Validation predictions (long: date, ticker, agent, p) -> one row per (date, ticker), one
    column per analyst 1-10, plus the outcome columns."""
    v = val_preds[~val_preds.agent.isin(NOT_IN_MAJORITY)]
    wide = v.pivot_table(index=["date", "ticker"], columns="agent", values="p").reset_index()
    outcome = (v.drop_duplicates(["date", "ticker"])
                 [["date", "ticker", "y", "fwd_return", "mae", "label_known"]])
    return wide.merge(outcome, on=["date", "ticker"], how="left")


def vote_features(P: pd.DataFrame) -> pd.DataFrame:
    """P: one column per analyst (probabilities, NaN = analyst had nothing to say)."""
    votes = (P >= 0.5).astype(float).where(P.notna())
    out = pd.DataFrame(index=P.index)
    out["vote_share_up"] = votes.mean(axis=1)
    out["vote_spread"] = P.std(axis=1).fillna(0.0)
    out["mean_p"] = P.mean(axis=1)
    out["regime_p"] = P["market_regime"] if "market_regime" in P else np.nan
    return out


def majority_up(P: pd.DataFrame) -> np.ndarray:
    """True where the majority of analysts says up (ties broken by the average probability)."""
    vf = vote_features(P)
    share, mean_p = vf["vote_share_up"].fillna(0.5), vf["mean_p"].fillna(0.5)
    return ((share > 0.5) | ((share == 0.5) & (mean_p >= 0.5))).to_numpy()


def _with_context(keys: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CONTEXT_FEATURES if c in rows]
    return keys.merge(rows[["date", "ticker", *cols]], on=["date", "ticker"], how="left")


class DevilsAdvocate(Analyst):
    name = "devils_advocate"
    features = FEATURES
    refit_on_full = False     # fit() always uses the validation predictions, so refitting adds nothing

    # ---------------------------------------------------------------- training
    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        """Ignores `panel` (fit-year rows): learns from the validation predictions instead."""
        val = ctx.extra.get("validation_predictions")
        if val is None or len(val) == 0:
            raise ValueError("Devil's Advocate needs ctx.extra['validation_predictions'] (run stage 1 first)")
        features = ctx.extra.get("panel")
        if features is None:
            from council.data import train_panel       # training folder only
            features = train_panel()
        self.fit_from_validation(val, features)

    def fit_from_validation(self, val_preds: pd.DataFrame, features: pd.DataFrame) -> None:
        wide = wide_probs(val_preds).dropna(subset=["y"])
        self.agents = [c for c in wide.columns if c not in
                       {"date", "ticker", "y", "fwd_return", "mae", "label_known"}]
        X, y = self._design(wide, features)
        self.model = lgb.LGBMClassifier(**LGB_PARAMS)
        self.model.fit(X, y)
        self.base_rate_wrong = float(y.mean())
        r = wide["fwd_return"]
        self.avg_win, self.avg_loss = float(r[r > 0].mean()), float(-r[r <= 0].mean())
        self.trained_on = f"validation slice {wide.date.min().date()} to {wide.date.max().date()} (in-sample there)"

    def _design(self, wide: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        P = wide.reindex(columns=self.agents)
        X = pd.concat([vote_features(P), _with_context(wide[["date", "ticker"]], features)
                       .drop(columns=["date", "ticker"])], axis=1).reindex(columns=FEATURES)
        y_wrong = (majority_up(P) != (wide["y"].to_numpy() == 1)).astype(int)
        return X.astype(float), y_wrong

    # ---------------------------------------------------------------- prediction
    def p_wrong_from_probs(self, rows: pd.DataFrame, probs: dict[str, np.ndarray]) -> np.ndarray:
        """p_wrong for many rows, given the other analysts' probabilities for those rows."""
        P = self._prob_frame(rows, probs)
        X = pd.concat([vote_features(P), rows.reset_index(drop=True).reindex(columns=CONTEXT_FEATURES)],
                      axis=1).reindex(columns=FEATURES).astype(float)
        return self.model.predict_proba(X)[:, 1]

    def _prob_frame(self, rows: pd.DataFrame, probs: dict[str, np.ndarray]) -> pd.DataFrame:
        cols = {a: np.asarray(probs[a], float) if a in probs else np.full(len(rows), np.nan) for a in self.agents}
        return pd.DataFrame(cols)

    def against_majority(self, rows: pd.DataFrame, probs: dict[str, np.ndarray]) -> np.ndarray:
        pw = self.p_wrong_from_probs(rows, probs)
        up = majority_up(self._prob_frame(rows, probs))
        return np.where(up, 0.5 - 0.5 * pw, 0.5 + 0.5 * pw)

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        """Needs the other analysts' probabilities for these rows: from ctx.extra['analyst_probs']
        if the caller already has them, otherwise by running the saved analysts."""
        probs = ctx.extra.get("analyst_probs") or self._run_analysts(panel, ctx)
        return self.against_majority(panel, probs)

    def _run_analysts(self, panel: pd.DataFrame, ctx: Context) -> dict[str, np.ndarray]:
        fitted = ctx.extra.get("analysts", {})
        out = {}
        for a in self.agents:
            analyst = fitted.get(a)
            if analyst is None:
                try:
                    analyst = Analyst.load(a)
                except FileNotFoundError:
                    continue
            out[a] = analyst.predict_proba(panel, ctx)
        return out

    # ---------------------------------------------------------------- explanation
    def drivers(self, row: pd.DataFrame, probs: dict[str, np.ndarray], top: int = 3) -> list[tuple[str, float]]:
        """The conditions that pushed p_wrong up or down today (LightGBM's own per-feature
        contributions, in log-odds)."""
        P = self._prob_frame(row, probs)
        X = pd.concat([vote_features(P), row.reset_index(drop=True).reindex(columns=CONTEXT_FEATURES)],
                      axis=1).reindex(columns=FEATURES).astype(float)
        contrib = self.model.predict_proba(X, pred_contrib=True)[0][:-1]
        order = np.argsort(-np.abs(contrib))[:top]
        return [(FEATURES[i], round(float(contrib[i]), 4)) for i in order]

    def key_numbers(self, row, ctx):
        return {k: float(row[k]) for k in CONTEXT_FEATURES if k in row and pd.notna(row[k])}


def out_of_fold_p_wrong(val_preds: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-year-out p_wrong on the validation slice: for each year, a Devil's Advocate
    trained on the OTHER validation years predicts it. Used to train the Mediator honestly."""
    years = sorted(pd.to_datetime(val_preds.date).dt.year.unique())
    parts = []
    for yr in years:
        train = val_preds[pd.to_datetime(val_preds.date).dt.year != yr]
        hold = wide_probs(val_preds[pd.to_datetime(val_preds.date).dt.year == yr])
        da = DevilsAdvocate()
        da.fit_from_validation(train, features)
        rows = _with_context(hold[["date", "ticker"]], features)
        probs = {a: hold[a].to_numpy() for a in da.agents if a in hold}
        parts.append(hold[["date", "ticker"]].assign(p_wrong=da.p_wrong_from_probs(rows, probs)))
    return pd.concat(parts, ignore_index=True)


def build() -> DevilsAdvocate:
    return DevilsAdvocate()
