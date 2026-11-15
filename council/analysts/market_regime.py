"""Analyst 8, Market Regime (PLAN.md §1b-8): "what mood is the market in?".

A 3-state Hidden Markov Model (HMM) of the Dow index. Each day the market is assumed to be in
one hidden "mood" (state), and each mood produces days with its own typical
(daily log return, 20-day volatility). Fitted on the training years with hmmlearn.
States are named by their average return: highest = bull, lowest = bear, middle = choppy.

1. Forward algorithm ("filtering"): P(state_t | data up to t) is updated day by day:
       alpha_t(s) ∝ [sum_s' alpha_{t-1}(s') * A(s', s)] * Normal(x_t | mean_s, cov_s)
   It only ever looks backwards, unlike hmmlearn's predict_proba, which would also use
   later days (smoothing) and so look ahead.
2. Tomorrow's mood: P(state_{t+1}) = P(state_t) · A, where A is the learned transition matrix.
3. p = sum_s P(state_{t+1} = s) * P(up | s), with P(up | s) = share of training rows in mood s
   that went up over the next 20 days (weighted by how sure we were of the mood). Measured
   separately for the index and for member stocks.
Members get the index's mood ("market mood"), not a mood of their own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from council.analysts.base import Analyst, Context

INDEX = "^DJI"
SEED = 0


def index_observations(prices: pd.DataFrame) -> pd.DataFrame:
    """(daily log return, 20-day volatility) of the Dow index, one row per day."""
    c = prices.loc[prices.ticker == INDEX, ["date", "adj_close"]].sort_values("date").set_index("date")["adj_close"]
    r = np.log(c).diff()
    return pd.DataFrame({"ret": r, "vol_20": r.rolling(20).std()}).dropna()


class MarketRegime(Analyst):
    name = "market_regime"
    features = ["p_bull", "p_bear", "p_choppy"]

    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        obs = index_observations(ctx.prices)
        obs = obs[obs.index <= panel.date.max()]                 # training years only
        self.mean_, self.std_ = obs.mean(), obs.std()            # standardise both inputs
        X = ((obs - self.mean_) / self.std_).to_numpy()
        self.hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=SEED)
        self.hmm.fit(X)
        # name states by their average daily return (in original units)
        order = np.argsort(self.hmm.means_[:, 0])                # low -> high return
        self.state_names = {int(order[0]): "bear", int(order[1]): "choppy", int(order[2]): "bull"}

        # P(up | state) from the filtered state probabilities on the training rows
        df = panel.dropna(subset=["y"])
        probs = self.filtered(ctx).reindex(df.date.values).to_numpy()   # P(state_t) per row
        ok = ~np.isnan(probs).any(1)
        self.p_up = {}
        for group, sel in [("index", df.ticker.eq(INDEX).to_numpy()), ("member", df.ticker.ne(INDEX).to_numpy())]:
            w, y = probs[sel & ok], df.y.to_numpy()[sel & ok]
            self.p_up[group] = (w * y[:, None]).sum(0) / w.sum(0)  # weighted share that went up
        self._learn_payoffs(df)

    # ------------------------------------------------------------------ forward filter
    def filtered(self, ctx: Context) -> pd.DataFrame:
        """P(state_t | data up to and including day t), one row per index trading day."""
        obs = index_observations(ctx.prices)
        X = ((obs - self.mean_) / self.std_).to_numpy()
        log_b = np.column_stack([multivariate_normal(self.hmm.means_[s], self.hmm.covars_[s]).logpdf(X)
                                 for s in range(3)])            # log P(x_t | state s)
        log_A = np.log(self.hmm.transmat_)
        log_alpha = np.log(self.hmm.startprob_) + log_b[0]
        out = np.empty_like(log_b)
        for t in range(len(X)):
            if t > 0:
                log_alpha = logsumexp(log_alpha[:, None] + log_A, axis=0) + log_b[t]
            log_alpha = log_alpha - logsumexp(log_alpha)        # normalise to probabilities
            out[t] = log_alpha
        return pd.DataFrame(np.exp(out), index=obs.index, columns=range(3))

    def _rows_state_probs(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        f = self.filtered(ctx)
        return f.reindex(pd.DatetimeIndex(panel.date), method="ffill").to_numpy()

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        now = self._rows_state_probs(panel, ctx)                 # P(state_t)
        nxt = now @ self.hmm.transmat_                           # P(state_{t+1})
        p_up = np.where(panel.ticker.eq(INDEX).to_numpy()[:, None], self.p_up["index"], self.p_up["member"])
        p = (nxt * p_up).sum(1)
        return np.where(np.isnan(p), 0.5, p)

    # ------------------------------------------------------------------ for the council
    def regime_on(self, date, ctx: Context) -> dict:
        """The market's mood on `date`, using only index data up to and including that date."""
        f = self.filtered(ctx)
        f = f[f.index <= pd.Timestamp(date)]
        row = f.iloc[-1]
        probs = {self.state_names[s]: float(row[s]) for s in range(3)}
        return {"state": max(probs, key=probs.get), "probs": probs, "date": str(f.index[-1].date())}

    def explain(self, row: pd.Series, ctx: Context) -> list[tuple[str, float]]:
        """Each mood's contribution to p, minus what that mood would give with a coin flip."""
        nxt = self._rows_state_probs(row.to_frame().T.assign(date=pd.Timestamp(row["date"])), ctx)[0] @ self.hmm.transmat_
        p_up = self.p_up["index" if row["ticker"] == INDEX else "member"]
        contrib = nxt * (p_up - 0.5)
        order = np.argsort(-np.abs(contrib))
        return [(f"next_state_{self.state_names[int(s)]}", float(contrib[s])) for s in order]

    def key_numbers(self, row, ctx):
        reg = self.regime_on(row["date"], ctx)
        return {f"p_{k}": v for k, v in reg["probs"].items()}


def build() -> MarketRegime:
    return MarketRegime()
