"""Analyst 4, Risk Officer (PLAN.md §1b-4): "how much could this go wrong?".

1. GARCH(1,1) volatility, one model per ticker, fitted on the training years:
       sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}
   where eps = daily log return (x100) minus its average mu. In words: tomorrow's variance is
   a baseline, plus a bit of today's surprise squared, plus most of today's variance.
   At predict time we only run this recursion forward through the returns up to the row's
   date (starting from the model's long-run variance omega / (1 - alpha - beta)), so sigma_t
   never uses a later price.
2. 95% Value-at-Risk over h days (as a share of the amount):
       VaR% = 1.645 * sigma_t * sqrt(h) - mu * h
   plus historical VaR (5th-percentile h-day loss in the training years) and max drawdown.
3. Monte Carlo: 10,000 price paths of h days,
       P_{t+1} = P_t * exp((mu - sigma_t^2 / 2) + sigma_t * Z),   Z ~ Normal(0, 1)
   giving the 5th/50th/95th percentile prices and the chance of hitting the stop-loss.
   (sigma_t is held constant over the h days: simple, and slightly cautious after calm spells.)
4. max_safe_amount = user_max_loss / VaR%. The Checklist enforces it.

Direction vote (weak by design): p = sigmoid(w0 + w1 * sigma_t + w2 * ret_20), a logistic
regression on today's GARCH volatility and the last 20 days' drift. Its job is risk, not
direction; the Mediator learns how much to trust this vote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from council.analysts.base import Analyst, Context
from council.features import CROSS_ASSET

Z95 = 1.645
MIN_DAYS = 500          # need at least ~2 years of returns to fit a GARCH


def log_returns_pct(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Daily log returns x100 of one ticker's adjusted close, indexed by date."""
    s = prices.loc[prices.ticker == ticker, ["date", "adj_close"]].sort_values("date").set_index("date")["adj_close"]
    return (100 * np.log(s).diff()).dropna()


def long_run_var(params: dict) -> float:
    return params["omega"] / max(1 - params["alpha"] - params["beta"], 1e-3)


def garch_sigma(r: pd.Series, params: dict) -> pd.Series:
    """sigma (in %) for the NEXT day, made at each day's close using returns up to that day.
    Value on date t = forecast for t+1 = sqrt(omega + alpha*eps_t^2 + beta*sigma2_t)."""
    mu, omega, alpha, beta = params["mu"], params["omega"], params["alpha"], params["beta"]
    eps2 = (r.to_numpy() - mu) ** 2
    s2 = long_run_var(params)                      # start at the long-run variance (no data used)
    out = np.empty(len(eps2))
    for i, e2 in enumerate(eps2):
        s2 = omega + alpha * e2 + beta * s2        # sigma2 for day i+1, using day i's return
        out[i] = s2
    return pd.Series(np.sqrt(out), index=r.index)


class RiskOfficer(Analyst):
    name = "risk_officer"
    features = ["garch_sigma", "ret_20"]

    # ------------------------------------------------------------------ fitting
    def fit(self, panel: pd.DataFrame, ctx: Context) -> None:
        last_day = panel.date.max()
        prices = ctx.prices[ctx.prices.date <= last_day]       # only the years we are fitting on
        self.params: dict[str, dict] = {}
        self.train_returns: dict[str, np.ndarray] = {}         # for historical VaR / drawdown
        for t in sorted(set(prices.ticker) - set(CROSS_ASSET) - {"DIA"}):
            r = log_returns_pct(prices, t)
            if len(r) < MIN_DAYS:
                continue
            res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
            p = res.params
            self.params[t] = {"mu": float(p["mu"]), "omega": float(p["omega"]),
                              "alpha": float(p["alpha[1]"]), "beta": float(p["beta[1]"])}
            self.train_returns[t] = (r.to_numpy() / 100).astype(np.float32)
        # fallback for a ticker we never saw: the typical (median) parameters
        self.default_params = {k: float(np.median([p[k] for p in self.params.values()]))
                               for k in ["mu", "omega", "alpha", "beta"]}

        df = panel.dropna(subset=["y"])
        self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=1.0, max_iter=1000))
        self.model.fit(self.transform(df, ctx), df["y"].astype(int))
        self._learn_payoffs(df)

    def _params(self, ticker: str) -> dict:
        return self.params.get(ticker, self.default_params)

    # ------------------------------------------------------------------ per-row sigma
    def sigma_for_rows(self, panel: pd.DataFrame, ctx: Context) -> pd.Series:
        """GARCH sigma_t (daily, as a fraction) for each panel row, from returns up to its date."""
        out = pd.Series(np.nan, index=panel.index)
        for t, rows in panel.groupby("ticker"):
            r = log_returns_pct(ctx.prices, t)
            if r.empty:
                continue
            sig = garch_sigma(r, self._params(t)) / 100
            # the latest sigma made on or before each row's date
            out.loc[rows.index] = sig.reindex(pd.DatetimeIndex(rows.date), method="ffill").to_numpy()
        return out

    def transform(self, panel: pd.DataFrame, ctx: Context) -> pd.DataFrame:
        return pd.DataFrame({"garch_sigma": self.sigma_for_rows(panel, ctx), "ret_20": panel["ret_20"]},
                            index=panel.index)

    def predict_proba(self, panel: pd.DataFrame, ctx: Context) -> np.ndarray:
        return self.model.predict_proba(self.transform(panel, ctx))[:, 1]

    def explain(self, row: pd.Series, ctx: Context) -> list[tuple[str, float]]:
        frame = row.to_frame().T
        frame["date"] = pd.to_datetime(frame["date"])
        X = self.transform(frame, ctx)
        imp, scale, lr = self.model.steps[0][1], self.model.steps[1][1], self.model.steps[2][1]
        contrib = scale.transform(imp.transform(X))[0] * lr.coef_[0]
        order = np.argsort(-np.abs(contrib))
        return [(X.columns[i], float(contrib[i])) for i in order]

    def key_numbers(self, row, ctx):
        rep = self.risk_report(row["ticker"], row["date"], ctx, amount=1000, max_loss=100, n_paths=2000)
        return {k: rep[k] for k in ["sigma_daily", "var95_pct", "hist_var95_pct", "max_drawdown_pct", "p_hit_stop"]}

    # ------------------------------------------------------------------ the council's risk numbers
    def risk_report(self, ticker: str, asof_date, ctx: Context, amount: float, max_loss: float,
                    horizon: int = 20, n_paths: int = 10_000, seed: int = 0) -> dict:
        """Risk numbers for buying `amount` of `ticker` at the close of `asof_date`, using only
        prices dated on or before `asof_date`. Percentages are fractions (0.05 = 5%).
        `paths` is an (n_paths x horizon) array of price multipliers (1.02 = 2% above today)."""
        asof = pd.Timestamp(asof_date)
        p = self._params(ticker)
        known = ctx.prices[(ctx.prices.ticker == ticker) & (ctx.prices.date <= asof)]
        r = log_returns_pct(known, ticker)
        sigma = float(garch_sigma(r, p).iloc[-1] if len(r) else np.sqrt(long_run_var(p))) / 100
        mu = p["mu"] / 100
        var95 = Z95 * sigma * np.sqrt(horizon) - mu * horizon

        hist = self.train_returns.get(ticker)
        if hist is not None and len(hist) > horizon:
            h_ret = np.expm1(np.convolve(hist, np.ones(horizon), "valid"))    # every h-day return
            hist_var95 = float(-np.percentile(h_ret, 5))
            wealth = np.exp(np.cumsum(hist))
            max_dd = float((wealth / np.maximum.accumulate(wealth) - 1).min())
        else:
            hist_var95, max_dd = float("nan"), float("nan")

        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_paths, horizon))
        paths = np.exp(np.cumsum((mu - sigma ** 2 / 2) + sigma * z, axis=1))
        last_price = float(known.sort_values("date")["close"].iloc[-1]) if len(known) else float("nan")
        stop = 1 - max_loss / amount
        end = paths[:, -1] * last_price
        return {
            "ticker": ticker, "asof_date": str(asof.date()), "horizon_days": horizon,
            "last_price": last_price,
            "sigma_daily": sigma,
            "mu_daily": mu,
            "var95_pct": float(var95),
            "var95_amount": float(var95 * amount),
            "hist_var95_pct": hist_var95,
            "max_drawdown_pct": max_dd,
            "p05": float(np.percentile(end, 5)),
            "p50": float(np.percentile(end, 50)),
            "p95": float(np.percentile(end, 95)),
            "stop_multiplier": stop,
            "p_hit_stop": float((paths.min(axis=1) <= stop).mean()),
            "max_safe_amount": float(max_loss / var95) if var95 > 0 else float("inf"),
            "paths": paths,
        }


def build() -> RiskOfficer:
    return RiskOfficer()
