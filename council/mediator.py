"""Mediator: the stacking model that combines the analysts, and the plan builder (PLAN.md §1, §1b).

1. COMBINE (trained stacking model, validation slice only):
       logit(p_council) = b0 + sum_i b_i * logit(p_i) + c * regime + d * vol + e * p_wrong
   p_i     each analyst's probability (1-10; a missing one counts as 0.5, i.e. logit 0)
   regime  +1 in an uptrend (20-day average above the 50-day), -1 in a downtrend. This is a
           simple stand-in for the HMM's state so the same number exists for every past row.
   vol     20-day volatility, standardised with the validation mean and spread
   p_wrong the Devil's Advocate's "the majority is wrong" probability
   The b_i are forced to be >= 0 (an analyst can be trusted a lot, a little, or not at all,
   but we never flip its vote), and all weights get a light L2 penalty so 4 years of data
   don't get over-fitted. Fitted by minimising log-loss with scipy's L-BFGS-B (bounds = the
   >= 0 constraint).

   Honest validation: the Mediator, the Devil's Advocate and the Judge's theta all learn from the
   same 2016-2019 slice. To score them fairly there we use leave-one-year-out: each year is
   predicted by models fitted on the other three years. Those out-of-fold numbers are what
   models/council_validation.json reports and what theta is tuned on. The saved model is then
   refitted on all four years.

2. PLANS: standard ones ("buy the budget now", "wait", "don't buy"), each re-checked with the
   rule code. A failing plan is fixed (amount cut until max loss fits) and re-checked, or dropped.
3. ITS OWN PLAN: it tries split / smaller-with-tighter-stop / staged / limit-dip plans on the
   Risk Officer's 10,000 simulated price paths and proposes one only if it beats the best
   standard buy there (same-or-better expected return with lower worst-case loss, or higher
   return at the same risk).

Run:  uv run python -m council.mediator      (fits and saves models/mediator.pkl)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from council import judge as J
from council import rules as R
from council.devils_advocate import DevilsAdvocate, out_of_fold_p_wrong, wide_probs
from council.schema import MediatorResult, Plan, PlanStep, UserRequest

from council.paths import MODELS  # noqa: E402
COUNCIL_FILE = MODELS / "mediator.pkl"
REPORT_FILE = MODELS / "council_validation.json"
L2 = 0.01          # "light penalty" on every weight except the intercept
TOL = 0.001        # "the same" expected return / risk = within 0.1% of the amount


# ---------------------------------------------------------------------- stacking model
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Stacker:
    """Logistic regression with b_i >= 0 for the analyst columns (the first n_analysts columns)."""

    def __init__(self, n_analysts: int, l2: float = L2):
        self.n_analysts, self.l2 = n_analysts, l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Stacker":
        n, k = X.shape
        Xb = np.column_stack([np.ones(n), X])

        def loss(w):
            z = Xb @ w
            p = sigmoid(z)
            ll = np.mean(np.logaddexp(0, z) - y * z)          # log-loss, written stably
            grad = Xb.T @ (p - y) / n
            ll += self.l2 * np.sum(w[1:] ** 2)
            grad[1:] += 2 * self.l2 * w[1:]
            return ll, grad

        bounds = [(None, None)] + [(0, None)] * self.n_analysts + [(None, None)] * (k - self.n_analysts)
        w0 = np.zeros(k + 1)
        w0[1:1 + self.n_analysts] = 1.0 / max(self.n_analysts, 1)
        res = minimize(loss, w0, jac=True, method="L-BFGS-B", bounds=bounds)
        self.w = res.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.w[0] + X @ self.w[1:])


class CouncilModel:
    """Everything the Mediator and Judge learned from the validation slice, saved in one pickle."""

    def __init__(self, agents: list[str]):
        self.agents = agents                 # analysts 1-10 that the stacker uses, in column order
        self.extra_cols = ["regime", "vol", "p_wrong"]

    def design(self, probs: dict[str, np.ndarray], rows: pd.DataFrame, p_wrong: np.ndarray) -> np.ndarray:
        n = len(rows)
        cols = [np.nan_to_num(logit(probs[a]), nan=0.0) if a in probs else np.zeros(n) for a in self.agents]
        regime = np.nan_to_num(rows["trend"].to_numpy(float), nan=0.0)
        vol = np.nan_to_num((rows["vol_20"].to_numpy(float) - self.vol_mean) / self.vol_sd, nan=0.0)
        return np.column_stack(cols + [regime, vol, np.nan_to_num(p_wrong, nan=0.5)])

    def p_council(self, probs: dict[str, np.ndarray], rows: pd.DataFrame, p_wrong: np.ndarray) -> np.ndarray:
        return self.stacker.predict(self.design(probs, rows, p_wrong))

    def trust(self) -> dict[str, float]:
        """Each analyst's stacking weight b_i (how much the council trusts it)."""
        return {a: round(float(w), 4) for a, w in zip(self.agents, self.stacker.w[1:1 + len(self.agents)])}

    def judge_params(self) -> dict:
        return {"theta": self.theta, "avg_win": self.avg_win, "avg_loss": self.avg_loss}

    def save(self) -> Path:
        MODELS.mkdir(exist_ok=True)
        COUNCIL_FILE.write_bytes(pickle.dumps(self))
        return COUNCIL_FILE


def load_council() -> CouncilModel:
    return pickle.loads(COUNCIL_FILE.read_bytes())


def _fit_one(wide: pd.DataFrame, agents: list[str], p_wrong: np.ndarray) -> CouncilModel:
    cm = CouncilModel(agents)
    cm.vol_mean = float(wide["vol_20"].mean())
    cm.vol_sd = float(wide["vol_20"].std())
    probs = {a: wide[a].to_numpy() for a in agents}
    cm.stacker = Stacker(len(agents)).fit(cm.design(probs, wide, p_wrong), wide["y"].to_numpy(float))
    r = wide["fwd_return"]
    cm.avg_win, cm.avg_loss = float(r[r > 0].mean()), float(-r[r <= 0].mean())
    return cm


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    return {"accuracy": round(float(((p >= 0.5) == (y == 1)).mean()), 4),
            "brier": round(float(((p - y) ** 2).mean()), 4)}


def fit_council(validation_predictions: pd.DataFrame, features: pd.DataFrame | None = None,
                save: bool = True) -> CouncilModel:
    """Fit the stacking Mediator and tune the Judge's theta on the validation predictions.

    validation_predictions: long table (date, ticker, agent, p, y, fwd_return, mae, label_known)
    features: the training panel (for trend, vol_20, breadth...); loaded if not given.
    """
    if features is None:
        from council.data import train_panel
        features = train_panel()
    ctx_cols = ["date", "ticker", "trend", "vol_20", "breadth_50", "dist_high_250", "vix"]
    wide = wide_probs(validation_predictions).dropna(subset=["y"])
    wide = wide.merge(features[[c for c in ctx_cols if c in features]], on=["date", "ticker"], how="left")
    wide = wide.sort_values(["date", "ticker"]).reset_index(drop=True)
    agents = [a for a in sorted(validation_predictions.agent.unique()) if a in wide.columns]
    loss = validation_risk_loss(wide)     # Drawdown Risk model's 1-in-20 loss (None = vol formula)

    # --- out-of-fold (leave-one-year-out) probabilities for honest validation numbers + theta
    oof_pw = out_of_fold_p_wrong(validation_predictions, features)
    wide = wide.merge(oof_pw, on=["date", "ticker"], how="left")
    years = wide.date.dt.year
    oof = np.full(len(wide), np.nan)
    for yr in sorted(years.unique()):
        tr, te = years != yr, years == yr
        cm_fold = _fit_one(wide[tr].reset_index(drop=True), agents, wide.loc[tr, "p_wrong"].to_numpy())
        te_rows = wide[te].reset_index(drop=True)
        oof[te.to_numpy()] = cm_fold.p_council({a: te_rows[a].to_numpy() for a in agents}, te_rows,
                                               te_rows["p_wrong"].to_numpy())

    # --- final model on all four years (p_wrong still out-of-fold, so the stacker can't cheat)
    cm = _fit_one(wide, agents, wide["p_wrong"].to_numpy())
    cm.theta, grid = J.tune_threshold(oof, wide, cm.avg_win, cm.avg_loss, loss_pct=loss)
    cm.lam = J.LAMBDA
    cm.risk_rule = "drawdown_risk model" if loss is not None else "volatility formula"

    # --- validation report (all out-of-fold), council vs simple average vs always-buy
    y = wide["y"].to_numpy(float)
    avg = np.nanmean(np.column_stack([wide[a] for a in agents]), axis=1)
    fr, mae = wide["fwd_return"].to_numpy(float), wide["mae"].to_numpy(float)
    best = next(g for g in grid if g["theta"] == cm.theta)
    avg_theta, avg_grid = J.tune_threshold(avg, wide, cm.avg_win, cm.avg_loss, loss_pct=loss)
    is_index = (wide.ticker == "^DJI").to_numpy()
    council_buy = J.buy_rule(oof, cm.avg_win, cm.avg_loss, cm.theta, wide["vol_20"], loss_pct=loss)
    cm.validation = {
        "note": "Out-of-fold (leave-one-year-out) on the validation slice; the test years are untouched.",
        "rows": int(len(wide)), "dates": [str(wide.date.min().date()), str(wide.date.max().date())],
        "agents": agents, "trust": cm.trust(),
        "coefficients": {"intercept": round(float(cm.stacker.w[0]), 4),
                         **{k: round(float(v), 4) for k, v in zip(cm.extra_cols, cm.stacker.w[-3:])}},
        "theta": cm.theta, "lambda": cm.lam, "risk_rule": cm.risk_rule, "avg_win": round(cm.avg_win, 4), "avg_loss": round(cm.avg_loss, 4),
        "base_rate_up": round(float(y.mean()), 4),
        "stacking": _metrics(oof, y), "simple_average": _metrics(avg, y),
        "judge_at_theta": best,
        "judge_index_only": J.score(council_buy[is_index], fr[is_index], mae[is_index]),
        "simple_average_judge": {"theta": avg_theta, **next(g for g in avg_grid if g["theta"] == avg_theta)},
        "always_buy": J.score(np.ones(len(wide), bool), fr, mae),
        "never_buy": J.score(np.zeros(len(wide), bool), fr, mae),
        "tuning": grid,
    }
    if save:
        cm.save()
        # out-of-fold council probabilities on the validation slice: council/strategy.py tunes on these
        wide[["date", "ticker"]].assign(p_council=oof).to_parquet(MODELS / "council_oof.parquet", index=False)
        REPORT_FILE.write_text(json.dumps(cm.validation, indent=2), encoding="utf-8")
    return cm


# ---------------------------------------------------------------------- many rows at once
def council_probabilities(rows: pd.DataFrame, analyst_probs: dict[str, np.ndarray],
                          max_loss: float = J.EXAMPLE_MAX_LOSS,
                          use_risk_model: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Apply the saved Mediator + Judge to many rows (the backtest). rows needs the panel columns
    trend, vol_20, breadth_50, dist_high_250, vix. analyst_probs: {agent: p array aligned with rows}.
    No Monte Carlo here: the max-loss rule uses the volatility VaR (judge.vol_var_pct)."""
    rows = rows.reset_index(drop=True)
    cm = load_council()
    probs = {a: np.asarray(v, float) for a, v in analyst_probs.items() if a != "devils_advocate"}
    try:
        da = DevilsAdvocate.load("devils_advocate")
        p_wrong = da.p_wrong_from_probs(rows, probs)
    except FileNotFoundError:
        p_wrong = np.full(len(rows), 0.5)
    p = cm.p_council(probs, rows, p_wrong)
    loss = risk_model_loss(rows) if use_risk_model else None
    buy = J.buy_rule(p, cm.avg_win, cm.avg_loss, cm.theta, rows["vol_20"], max_loss, loss_pct=loss)
    return p, buy


def risk_model_loss(rows: pd.DataFrame) -> np.ndarray | None:
    """The trained Drawdown Risk model's 1-in-20 loss for each row, or None if not trained."""
    try:
        from council.analysts.base import Analyst
        return Analyst.load("drawdown_risk").predict_loss(rows)
    except FileNotFoundError:
        return None


def validation_risk_loss(wide: pd.DataFrame) -> np.ndarray | None:
    """Out-of-sample Drawdown Risk predictions on the validation rows (from train.py)."""
    path = MODELS / "risk_predictions" / "drawdown_risk.parquet"
    if not path.exists():
        return None
    r = pd.read_parquet(path)[["date", "ticker", "p"]].rename(columns={"p": "_loss"})
    return wide[["date", "ticker"]].merge(r, on=["date", "ticker"], how="left")["_loss"].to_numpy(float)


# ---------------------------------------------------------------------- risk + paths
def simple_risk_report(ticker: str, asof_date, ctx, amount: float, max_loss: float, horizon: int = 20,
                       n_paths: int = 10000, seed: int = 0) -> dict:
    """Fallback when the Risk Officer isn't available: historical volatility instead of GARCH,
    same outputs. Uses only prices dated on or before asof_date (the newest data row; the
    decision date is the next trading day). Drift mu = the long-run average daily return."""
    D = pd.Timestamp(asof_date)
    px = ctx.prices
    s = px[(px.ticker == ticker) & (px.date <= D)].sort_values("date")["adj_close"]
    r = np.log(s).diff().dropna()
    sigma = float(r.tail(60).std())
    mu = float(r.mean())
    var95 = max(J.Z95 * sigma * np.sqrt(horizon) - mu * horizon, 0.0)
    h_ret = (s / s.shift(horizon) - 1).dropna()
    hist_var = float(max(-np.quantile(h_ret, 0.05), 0.0)) if len(h_ret) > 50 else var95
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, horizon))
    paths = np.exp(np.cumsum((mu - sigma ** 2 / 2) + sigma * z, axis=1))
    price = float(px[(px.ticker == ticker) & (px.date <= D)].sort_values("date")["close"].iloc[-1])
    stop = max_loss / amount if amount > 0 else 1.0
    used = max(var95, hist_var)
    return {
        "source": "fallback: historical volatility (Risk Officer not available)",
        "sigma_daily": round(sigma, 5), "var95_pct": round(var95, 4), "hist_var95_pct": round(hist_var, 4),
        "p05": round(price * float(np.quantile(paths[:, -1], 0.05)), 2),
        "p50": round(price * float(np.quantile(paths[:, -1], 0.50)), 2),
        "p95": round(price * float(np.quantile(paths[:, -1], 0.95)), 2),
        "p_hit_stop": round(float((paths.min(axis=1) <= 1 - stop).mean()), 4),
        "max_safe_amount": round(max_loss / used, 2) if used > 0 else amount,
        "paths": paths,
    }


def get_risk(ticker: str, asof_date, ctx, amount: float, max_loss: float, horizon: int = 20,
             risk_officer=None, model_loss: float | None = None) -> dict:
    """The Risk Officer's report (GARCH + Monte Carlo) if it's trained, else the fallback above.
    var95_used_pct = the more cautious of the model VaR and the historical VaR; the max-loss
    rule uses it, and max_safe_amount = max_loss / var95_used_pct."""
    try:
        if risk_officer is None:
            from council.analysts.base import Analyst
            risk_officer = Analyst.load("risk_officer")
        rep = dict(risk_officer.risk_report(ticker, asof_date, ctx, amount, max_loss, horizon=horizon,
                                            n_paths=10000, seed=0))
        rep.setdefault("source", "risk_officer (GARCH + Monte Carlo)")
    except Exception as e:   # not written yet / not trained / failed: use the simple version
        rep = simple_risk_report(ticker, asof_date, ctx, amount, max_loss, horizon)
        rep["fallback_reason"] = f"{type(e).__name__}: {e}"[:200]
    vs = [v for v in (rep.get("var95_pct"), rep.get("hist_var95_pct")) if v is not None and np.isfinite(v)]
    used = max(vs) if vs else 0.0
    if model_loss is not None:   # the trained Drawdown Risk model sets the rule when available
        rep["var95_model_pct"] = round(float(model_loss), 4)
        used = float(model_loss)
        rep["var95_source"] = "drawdown_risk model (1-in-20 worst drop, trained)"
    rep["var95_used_pct"] = round(float(used), 4)
    rep["max_safe_amount"] = round(float(max_loss / used), 2) if used > 0 else float(amount)
    return rep


# ---------------------------------------------------------------------- plan simulation
def simulate(paths: np.ndarray, steps: list[tuple[float, int]], stop_pct: float | None,
             dip: float | None = None, dip_within: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Profit/loss (currency) of a plan on every simulated path.
    paths[:, d] = price on day d+1 divided by today's price. steps: (amount, entry day; 0 = now).
    stop_pct: sell a step if it falls this much below its entry price (filled at the stop).
    dip: only for limit plans: buy when the price first touches (1 - dip) within dip_within days.
    Returns (profit per path, whether any step's stop-loss was hit on that path)."""
    n, h = paths.shape
    full = np.column_stack([np.ones(n), paths])              # full[:, d] = price on day d
    pnl = np.zeros(n)
    any_hit = np.zeros(n, bool)
    for amount, day in steps:
        if dip is not None:
            touched = full[:, 1:dip_within + 1] <= 1 - dip
            bought = touched.any(axis=1)
            entry_day = np.where(bought, touched.argmax(axis=1) + 1, h)
            entry = np.full(n, 1 - dip)
        else:
            bought = np.ones(n, bool)
            entry_day = np.full(n, day)
            entry = full[:, day]
        ret = full[:, -1] / entry - 1
        if stop_pct:
            after = np.where(np.arange(h + 1)[None, :] >= entry_day[:, None], full, np.inf)
            hit = after.min(axis=1) <= entry * (1 - stop_pct)
            ret = np.where(hit, -stop_pct, ret)
            any_hit |= hit & bought
        pnl += np.where(bought, amount * ret, 0.0)
    return pnl, any_hit


def _score(pnl: np.ndarray) -> tuple[float, float]:
    """(expected profit, worst-case loss = minus the 5th percentile), both in currency."""
    return float(pnl.mean()), float(max(-np.quantile(pnl, 0.05), 0.0))


def build_plans(p_council: float, req: UserRequest, ticker: str, risk: dict,
                index_weights: dict[str, float] | None = None) -> MediatorResult:
    var = risk["var95_used_pct"]
    paths = risk["paths"]
    dropped: list[dict] = []
    plans: list[Plan] = []

    def checked(plan: Plan) -> Plan:
        checks = R.check_plan(plan.amount, ticker, req, var, index_weights)
        plan.failed_rules = R.failed(checks)
        plan.checklist = "all_pass" if not plan.failed_rules else "fails"
        return plan

    def with_paths(plan: Plan, steps, stop, dip=None) -> Plan:
        pnl, hit = simulate(paths, steps, stop, dip)
        er, wc = _score(pnl)
        plan.expected_return_pct = round(100 * er / plan.amount, 2) if plan.amount else 0.0
        plan.worst_case_loss = round(wc, 2)
        plan.p_hit_stop = round(float(hit.mean()), 4) if stop else None
        return plan

    # --- standard plan A: buy the whole budget now; fix it if it breaks a rule
    amount = req.amount
    stop = min(req.max_loss / amount, 0.5)
    a = checked(Plan(id="A", action="buy", amount=amount, steps=[PlanStep(amount=amount, when="now")],
                     stop_loss_pct=round(stop, 4)))
    if a.failed_rules == ["max_loss"]:
        fixed = round(R.max_amount_within_loss(req, var), 2)
        dropped.append({"plan": f"buy {R.money(amount, req.currency)} now", "failed_rule": "max_loss",
                        "fix_tried": f"reduced to {R.money(fixed, req.currency)} -> became plan A"})
        amount, stop = fixed, min(req.max_loss / fixed, 0.5) if fixed > 0 else None
        a = checked(Plan(id="A", action="buy", amount=amount, steps=[PlanStep(amount=amount, when="now")],
                         stop_loss_pct=round(stop, 4) if stop else None))
    if a.checklist == "all_pass" and a.amount > 0:
        plans.append(with_paths(a, [(a.amount, 0)], a.stop_loss_pct))
    else:
        dropped.append({"plan": f"buy {R.money(a.amount, req.currency)} now", "failed_rule": ",".join(a.failed_rules),
                        "fix_tried": "no amount fixes this rule -> dropped"})

    plans.append(checked(Plan(id="B", action="wait", amount=0.0)))
    plans.append(checked(Plan(id="C", action="avoid", amount=0.0)))

    # --- the Mediator's own plans, only tried if there is a valid buy plan to beat
    best = plans[0] if plans[0].action == "buy" else None
    if best is not None:
        own = _best_own_plan(best, req, paths, checked, with_paths)
        if own is not None:
            plans.insert(1, own)
    return MediatorResult(p_council=round(float(p_council), 4), plans=plans, dropped=dropped)


def _best_own_plan(best: Plan, req: UserRequest, paths, checked, with_paths) -> Plan | None:
    amt, stop = best.amount, best.stop_loss_pct
    m = lambda x: R.money(x, req.currency)  # noqa: E731
    half, third = round(amt / 2, 2), round(amt / 3, 2)
    small, tight = round(amt * 0.7, 2), round(max(stop * 0.5, 0.03), 4)
    candidates = [
        (Plan(id="D", source="mediator_original", action="split_buy", amount=amt, stop_loss_pct=stop,
              steps=[PlanStep(amount=half, when="now"), PlanStep(amount=amt - half, when="+10 days")]),
         [(half, 0), (amt - half, 10)], stop, None),
        (Plan(id="D", source="mediator_original", action="buy", amount=small, stop_loss_pct=tight,
              steps=[PlanStep(amount=small, when="now")]),
         [(small, 0)], tight, None),
        (Plan(id="D", source="mediator_original", action="staged_buy", amount=amt, stop_loss_pct=stop,
              steps=[PlanStep(amount=third, when="now"), PlanStep(amount=third, when="+5 days"),
                     PlanStep(amount=round(amt - 2 * third, 2), when="+10 days")]),
         [(third, 0), (third, 5), (amt - 2 * third, 10)], stop, None),
        (Plan(id="D", source="mediator_original", action="limit_buy", amount=amt, stop_loss_pct=stop,
              steps=[PlanStep(amount=amt, when="if price dips 3% within 10 days")]),
         [(amt, 0)], stop, 0.03),
    ]
    er_a = best.expected_return_pct * best.amount / 100
    wc_a = best.worst_case_loss
    tol = TOL * best.amount
    winners = []
    for plan, steps, s, dip in candidates:
        plan = with_paths(checked(plan), steps, s, dip)
        if plan.checklist != "all_pass":
            continue
        er = plan.expected_return_pct * plan.amount / 100
        wc = plan.worst_case_loss
        if (er >= er_a - tol and wc < wc_a - tol) or (er > er_a + tol and wc <= wc_a + tol):
            plan.why_better = (f"On 10,000 simulated paths: expected profit {m(er)} vs {m(er_a)} for plan A, "
                               f"worst-case loss {m(wc)} vs {m(wc_a)}")
            winners.append((er - wc, plan))
    return max(winners, key=lambda t: t[0])[1] if winners else None


def main() -> None:
    from train import validation_predictions
    preds = validation_predictions()
    if preds.empty:
        raise SystemExit("no validation predictions yet: run `uv run python train.py` first")
    preds = preds[preds.agent != "devils_advocate"]        # its validation scores are in-sample
    # import ourselves by name so the pickle says "council.mediator.CouncilModel", not "__main__"
    from council import mediator
    cm = mediator.fit_council(preds)
    v = {k: v for k, v in cm.validation.items() if k != "tuning"}
    print(json.dumps(v, indent=2))
    print(f"saved {COUNCIL_FILE} and {REPORT_FILE}")


if __name__ == "__main__":
    main()
