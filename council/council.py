"""The whole council for one request (PLAN.md §1 architecture):

  Historian -> Round 1: every saved analyst predicts -> Round 2: cross-examination +
  Devil's Advocate -> Checklist (rules + positives/negatives) -> Mediator (p_council + plans,
  its own plan tested on the Risk Officer's paths) -> Judge (buy / wait / avoid + amount)

`run_council` returns a RunResult (council/schema.py), which the report, narrator and app read.
`decline` re-runs it after the user presses Decline, with their reason applied (see apply_decline).

Point-in-time: the decision date D defaults to the next weekday after the newest data, and the
analysts predict from the newest row dated BEFORE D (Historian.latest).
"""
from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from council import judge as J
from council import rules as R
from council.analysts.base import Analyst, Context, Signal, make_signal
from council.checklist import review_analysts
from council.crossexam import cross_examine, known_wide
from council.devils_advocate import NOT_IN_MAJORITY, majority_up
from council.historian import Historian
from council.mediator import build_plans, council_probabilities, get_risk, load_council, risk_model_loss  # noqa: F401
from council.schema import RunResult, UserRequest

INDEX = "^DJI"
FALLBACK_THETA = 0.60        # only if models/mediator.pkl doesn't exist yet


def _voting_names() -> list[str]:
    from train import analyst_specs
    return [s["name"] for s in analyst_specs() if s.get("votes", True) and s["name"] not in NOT_IN_MAJORITY]


def _validation_predictions() -> pd.DataFrame:
    from train import validation_predictions
    v = validation_predictions()
    return v[v.agent != "devils_advocate"] if len(v) else v


def run_council(request: UserRequest, ctx: Context, panel: pd.DataFrame, decision_date=None) -> RunResult:
    notes: list[str] = []
    ticker = request.ticker
    val = _validation_predictions()
    hist = Historian(panel, val)
    D = pd.Timestamp(decision_date) if decision_date is not None else \
        pd.Timestamp(panel.date.max()) + pd.offsets.BDay(1)
    row = hist.latest(ticker, D)
    row_df = hist.history(D, ticker).tail(1).reset_index(drop=True)   # same row, as a 1-row table
    # every row of that day (all members + the index): some analysts compare a stock with the
    # other members that day (Dow Breadth), so they predict on the whole day and we pick ours
    day_rows = hist.history(D)
    day_rows = day_rows[day_rows.date == row.date].reset_index(drop=True)
    mine = int(np.flatnonzero(day_rows.ticker.to_numpy() == ticker)[0])

    # ---- Round 1: every saved analyst predicts on its own
    loaded: dict[str, Analyst] = {}
    for name in _voting_names():
        try:
            loaded[name] = Analyst.load(name)
        except FileNotFoundError:
            notes.append(f"{name}: not trained yet, skipped")
    pctx = Context(prices=ctx.prices, macro=ctx.macro, membership=ctx.membership, market_news=ctx.market_news,
                   company_news=ctx.company_news, extra={**ctx.extra, "analysts": loaded})
    signals: list[Signal] = []
    probs: dict[str, np.ndarray] = {}
    for name, a in loaded.items():
        try:
            p = float(a.predict_proba(day_rows, pctx)[mine])
            rec = hist.track_record(name, D)
            if rec:
                rec = {"validation_accuracy": rec["accuracy"], **rec}
            explain_row = row
            if hasattr(a, "transform") and name == "dow_breadth":   # its reasons need the whole day's members
                explain_row = a.transform(day_rows, pctx).iloc[mine]
            signals.append(make_signal(a, explain_row, p, pctx, request.thinks_up, rec, request.horizon_days))
            probs[name] = np.array([p])
        except Exception as e:      # one broken analyst must not stop the council
            notes.append(f"{name}: failed today ({type(e).__name__}: {str(e)[:120]}), skipped")

    # ---- Round 2: Devil's Advocate + cross-examination
    da_out: dict = {}
    p_wrong = np.array([0.5])
    try:
        da = Analyst.load("devils_advocate")
        p_wrong = da.p_wrong_from_probs(row_df, probs)
        p_da = float(da.against_majority(row_df, probs)[0])
        drivers = da.drivers(row_df, probs)
        maj_up = bool(majority_up(da._prob_frame(row_df, probs))[0])
        signals.append(make_signal(da, row, p_da, pctx, request.thinks_up,
                                   {"note_in_sample_on_validation": 1.0}, request.horizon_days))
        signals[-1].top_reasons = [(k, round(float(v), 4)) for k, v in drivers][:3]   # its own LightGBM drivers
        da_out = {
            "p_majority_wrong": round(float(p_wrong[0]), 4),
            "majority_says_up": maj_up,
            "base_rate_majority_wrong": round(da.base_rate_wrong, 4),
            "drivers": drivers,
            "text": (f"In situations like this, the majority (saying {'up' if maj_up else 'down'}) has been "
                     f"wrong about {p_wrong[0]:.0%} of the time (overall {da.base_rate_wrong:.0%})."),
            "trained_on": da.trained_on,
        }
    except FileNotFoundError:
        notes.append("devils_advocate: not trained yet, p_wrong = 0.5")

    wide = known_wide(val, D)
    if not wide.empty:
        wide = wide.merge(panel[["date", "ticker", "trend", "vol_20"]], on=["date", "ticker"], how="left")
    disagreements = cross_examine(signals, val, ticker, D, wide=wide)

    # ---- Risk Officer (paths, VaR, max safe amount)
    # bought at the next open after the newest data row, so the Risk Officer works "as of" that row
    model_loss = risk_model_loss(row_df)
    risk = get_risk(ticker, row.date, ctx, request.amount, request.max_loss, request.horizon_days,
                    loaded.get("risk_officer"), model_loss=None if model_loss is None else float(model_loss[0]))
    from council.strategy import load_strategy
    if load_strategy() is not None and "var95_model_pct" in risk:
        # the tuned strategy always sells with a stop placed so stop + a 1-in-20 gap <= max loss,
        # so the max-loss rule is checked on the protected loss, not the unprotected one
        risk["var95_unprotected_pct"] = risk["var95_model_pct"]
        risk["var95_used_pct"] = round(min(risk["var95_model_pct"], request.max_loss / max(request.amount, 1e-9)), 4)
        risk["var95_source"] = "drawdown_risk model, protected by the Judge's gap-aware stop-loss"
        risk["max_safe_amount"] = round(request.max_loss / risk["var95_used_pct"], 2) if risk["var95_used_pct"] > 0 else request.amount
    if risk.get("source", "").startswith("fallback"):
        notes.append("risk: Risk Officer not available, used historical volatility")
    index_weights = None
    if ticker in R.INDEX_TICKERS:
        today = panel[(panel.date == row.date) & (panel.ticker != INDEX)]
        index_weights = dict(zip(today.ticker, today.dow_weight.fillna(0.0)))

    # ---- Mediator: combine, then build and check plans
    try:
        cm = load_council()
        p_council = float(cm.p_council(probs, row_df, p_wrong)[0])
        weights, params = cm.trust(), cm.judge_params()
    except FileNotFoundError:
        notes.append("mediator: not fitted yet (run `uv run python -m council.mediator`), used a simple average")
        p_council = float(np.mean([p[0] for p in probs.values()])) if probs else 0.5
        weights = {k: round(1 / max(len(probs), 1), 4) for k in probs}
        r = hist.known_outcomes(D)["fwd_return"]
        params = {"theta": FALLBACK_THETA, "avg_win": float(r[r > 0].mean()), "avg_loss": float(-r[r <= 0].mean())}
    med = build_plans(p_council, request, ticker, risk, index_weights)
    med.weights = weights

    rule_checks = R.check_plan(request.amount, ticker, request, risk["var95_used_pct"], index_weights)
    buy_allowed = any(pl.action not in ("wait", "avoid") and pl.checklist == "all_pass" for pl in med.plans)
    reviews = review_analysts(signals, wide, ticker, row, disagreements, buy_allowed) if not wide.empty else []

    # ---- Judge
    price = float(row["close"])
    verdict = J.decide(p_council, med, request, params, price, risk)

    regime = _regime_today(row.date, ctx, loaded.get("market_regime"))
    risk_out = {k: v for k, v in risk.items() if k != "paths"}
    risk_out["paths_summary"] = {"n_paths": int(risk["paths"].shape[0]), "horizon_days": int(risk["paths"].shape[1])}
    if regime:
        risk_out["regime"] = regime
    return RunResult(
        run_id=f"{ticker.strip('^')}-{D.date()}-{uuid.uuid4().hex[:6]}",
        decision_date=str(D.date()), data_as_of=str(pd.Timestamp(row.date).date()),
        request=request, price=round(price, 2), signals=signals, crossexam=disagreements,
        devils_advocate=da_out, rules=rule_checks, reviews=reviews, mediator=med, judge=verdict,
        risk=_jsonable(risk_out), notes=notes,
    )


def _regime_today(asof, ctx, market_regime) -> dict | None:
    """The HMM's mood on the newest data day, from the Market Regime analyst (info only)."""
    try:
        return _jsonable(dict(market_regime.regime_on(asof, ctx)))
    except Exception:
        return None


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (float, np.floating)):
        return float(d) if np.isfinite(d) else None
    if isinstance(d, np.integer):
        return d.item()
    if isinstance(d, (pd.Timestamp, datetime)):
        return str(d)
    return d


# ---------------------------------------------------------------------- Decline + reason
RISKY_WORDS = ("risk", "scared", "nervous", "safe", "lose", "volatile")
AMOUNT_WORDS = ("too much", "smaller", "less money", "amount", "expensive", "cheaper")
DOWN_WORDS = ("go down", "fall", "drop", "crash", "bearish")


def apply_decline(request: UserRequest, reason: str) -> UserRequest:
    """Turn the user's Decline reason into a changed request. Simple keyword rules:
      sounds like "too risky"         -> budget AND max loss halved (a smaller, safer bet)
      sounds like "too much money"    -> budget halved
      names a sector ("no tech")      -> that sector added to avoid_sectors
      "I think it'll go down"         -> the user's idea flips to "down"
    Anything else is just recorded. The reason is always kept in decline_reasons."""
    text = reason.lower()
    new = request.model_copy(deep=True)
    new.decline_reasons = [*request.decline_reasons, reason]
    risky = any(w in text for w in RISKY_WORDS)
    if risky:
        new.max_loss = round(request.max_loss / 2, 2)
    if risky or any(w in text for w in AMOUNT_WORDS):
        new.amount = round(request.amount / 2, 2)
    for word in text.replace(",", " ").split():
        s = R.normalise_sector(word)
        if s and s not in [R.normalise_sector(a) for a in new.avoid_sectors]:
            new.avoid_sectors = [*new.avoid_sectors, s]
    if any(w in text for w in DOWN_WORDS):
        new.thinks_up, new.idea_text = False, "I think it'll go down"
    return new


def decline(request: UserRequest, reason: str, ctx: Context, panel: pd.DataFrame,
            decision_date=None) -> RunResult:
    """The user pressed Decline with a reason: change the request and run the council again."""
    return run_council(apply_decline(request, reason), ctx, panel, decision_date)


def main() -> None:
    import sys

    from council.data import train_panel
    from train import training_context
    panel = train_panel()
    ctx = training_context()
    for t in sys.argv[1:] or [INDEX, "AAPL"]:
        res = run_council(UserRequest(ticker=t), ctx, panel)
        print(res.model_dump_json(indent=2))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------- Approve button
def recheck_rules(result: RunResult, plan_id: str | None = None, ctx: Context | None = None,
                  asof=None) -> list:
    """Re-run the hard rules for the plan the Judge picked (or `plan_id`) just before ordering.
    With ctx (newest data), the Risk Officer's VaR is recomputed as of `asof` (default: newest
    price in ctx for this ticker); without it, the VaR from the original run is reused.
    Returns list[RuleCheck]; the order must be blocked if any has passed=False."""
    req = result.request
    pid = plan_id or result.judge.plan_id
    plan = next((p for p in result.mediator.plans if p.id == pid), None)
    amount = result.judge.amount if pid == result.judge.plan_id else (plan.amount if plan else 0.0)
    var = result.risk.get("var95_used_pct", 0.0)
    if ctx is not None:
        px = ctx.prices[ctx.prices.ticker == req.ticker]
        when = pd.Timestamp(asof) if asof is not None else px.date.max()
        fresh = get_risk(req.ticker, when, ctx, req.amount, req.max_loss, req.horizon_days)["var95_used_pct"]
        var = max(fresh, var)    # cautious: newer GARCH/historical VaR or the run's model VaR
    checks = R.check_plan(amount, req.ticker, req, var)
    if req.ticker in R.INDEX_TICKERS:     # sector weights of the index: keep the original run's check
        orig = {c.rule: c for c in result.rules}
        checks = [orig.get(c.rule, c) if c.rule == "avoid_sectors" else c for c in checks]
    return checks
