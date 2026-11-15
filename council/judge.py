"""Judge: the final buy / wait / avoid decision and the amount (PLAN.md §1b "Decision and position size").

Buy only if ALL of these hold:
  1. p_council > theta                      (theta tuned on the validation years, never on test)
  2. expected return > 0, where expected return = p * avg_win - (1 - p) * avg_loss
  3. every hard rule passes                 (council/rules.py; the Judge can't overrule a fail)
Otherwise: "avoid" if the council leans down (p < 0.5 or expected return <= 0), else "wait".

Why theta is tuned: the Dow went up in ~63% of 20-day windows in training, so most analysts'
probabilities sit above 0.5 and a 0.5 threshold would mean "always buy".

How theta is tuned (on out-of-fold validation probabilities, see council/mediator.py):
  for each theta on a grid, pretend to be the example user (max loss 10%, 20 days) on every
  validation row and compute
      score(theta) = average return per decision  -  LAMBDA * share of decisions that broke the
                                                      10% loss limit (bought and fell 10%+ at
                                                      some point during the 20 days)
  and keep the best theta. LAMBDA = 0.1 means one decision in ten breaking the limit costs as
  much as 1 percentage point of average return. (We tried LAMBDA = 2 first: breaches then
  dominate so completely that the best choice is "never buy", which teaches nothing. See
  `tuning` in models/council_validation.json for the whole grid.)

Size: half-Kelly, capped by risk.
  b = avg_win / avg_loss,   f = 1/2 * (p*b - (1-p)) / b   (0 if negative),
  amount = min(f * budget, max_safe_amount)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from council.schema import JudgeResult, MediatorResult, Plan, UserRequest

LAMBDA = 0.1
EXAMPLE_MAX_LOSS = 0.10       # the backtest's example user: max loss 10% of the amount
HORIZON = 20
Z95 = 1.645
THETA_GRID = np.round(np.arange(0.50, 0.851, 0.005), 3)


def expected_return(p, avg_win: float, avg_loss: float):
    return p * avg_win - (1 - p) * avg_loss


def kelly_fraction(p: float, avg_win: float, avg_loss: float) -> float:
    """Half-Kelly: f = 1/2 * (p*b - (1-p)) / b, never below 0 or above 1."""
    b = avg_win / avg_loss
    return float(np.clip(0.5 * (p * b - (1 - p)) / b, 0.0, 1.0))


def vol_var_pct(vol_20, horizon: int = HORIZON):
    """Quick 95% VaR from 20-day volatility, used when there is no Monte Carlo (backtest):
    1.645 * daily vol * sqrt(h). The drift term is left out, which is the cautious choice."""
    return Z95 * np.asarray(vol_20, float) * np.sqrt(horizon)


def buy_rule(p, avg_win: float, avg_loss: float, theta: float, vol_20,
             max_loss: float = EXAMPLE_MAX_LOSS, loss_pct=None) -> np.ndarray:
    """The Judge's rule for many rows at once (backtest and tuning): p > theta, expected return
    > 0, and the max-loss rule passes for the full amount. The 1-in-20 loss comes from the
    trained Drawdown Risk model when `loss_pct` is given, otherwise from the volatility formula."""
    p = np.asarray(p, float)
    loss = vol_var_pct(vol_20) if loss_pct is None else np.asarray(loss_pct, float)
    var_ok = np.nan_to_num(loss, nan=np.inf) <= max_loss
    return (p > theta) & (expected_return(p, avg_win, avg_loss) > 0) & var_ok


def score(buy: np.ndarray, fwd_return: np.ndarray, mae: np.ndarray, lam: float = LAMBDA) -> dict:
    r = np.where(buy, fwd_return, 0.0)
    breach = buy & (mae <= -EXAMPLE_MAX_LOSS)
    return {"avg_return": float(r.mean()), "breach_share": float(breach.mean()),
            "buy_rate": float(buy.mean()), "score": float(r.mean() - lam * breach.mean())}


def tune_threshold(p: np.ndarray, rows: pd.DataFrame, avg_win: float, avg_loss: float,
                   lam: float = LAMBDA, loss_pct=None) -> tuple[float, list[dict]]:
    """Pick theta on the validation rows. rows needs fwd_return, mae, vol_20."""
    fr, mae = rows["fwd_return"].to_numpy(float), rows["mae"].to_numpy(float)
    grid = []
    for theta in THETA_GRID:
        buy = buy_rule(p, avg_win, avg_loss, theta, rows["vol_20"], loss_pct=loss_pct)
        grid.append({"theta": float(theta), **score(buy, fr, mae, lam)})
    best = max(grid, key=lambda g: (round(g["score"], 6), -g["theta"]))
    return best["theta"], grid


# ---------------------------------------------------------------------- one live decision
def decide(p_council: float, mediator: MediatorResult, req: UserRequest, params: dict,
           price: float, risk: dict) -> JudgeResult:
    """params: theta, avg_win, avg_loss (from models/mediator.pkl). If a tuned strategy exists
    (council/strategy.py), its threshold, risk-budget sizing and gap-aware stop are used instead."""
    from council.strategy import gap_for, load_strategy
    strat = load_strategy()
    if strat is not None:
        return _decide_with_strategy(p_council, mediator, req, strat, price, risk)
    theta, aw, al = params["theta"], params["avg_win"], params["avg_loss"]
    er = float(expected_return(p_council, aw, al))
    f = kelly_fraction(p_council, aw, al)
    buy_plans = [pl for pl in mediator.plans if pl.action not in ("wait", "avoid") and pl.amount > 0
                 and pl.checklist == "all_pass"]
    # the Mediator only proposes its own plan if it beat the best standard one on the paths
    plan = next((pl for pl in buy_plans if pl.source == "mediator_original"), None) or \
        (buy_plans[0] if buy_plans else None)
    if p_council > theta and er > 0 and plan is not None and f > 0:
        amount = round(min(f * req.amount, plan.amount), 2)
        reason = (f"The council's probability of a rise is {p_council:.0%}, above the {theta:.0%} bar tuned "
                  f"on 2016-2019, the expected 20-day return is {er:+.1%}, and every rule you set passes. "
                  f"Half-Kelly sizing says {f:.0%} of your budget.")
        would = _would_change_if("buy", p_council, theta, req, price, risk, plan)
        return JudgeResult(decision="buy", plan_id=plan.id, amount=amount, confidence=round(p_council, 4),
                           threshold=theta, kelly_fraction=round(f, 4), reason=reason, would_change_if=would)

    lean_down = p_council < 0.5 or er <= 0
    if lean_down:
        decision, reason = "avoid", (f"The council leans down: {p_council:.0%} chance of a rise, expected "
                                     f"20-day return {er:+.1%}.")
    elif plan is None:
        decision, reason = "wait", "No buying plan passes all of your rules, so the Judge can't buy."
    else:
        decision, reason = "wait", (f"The council's {p_council:.0%} is below the {theta:.0%} bar tuned on "
                                    f"2016-2019 (at 50% it would buy almost always, because the Dow "
                                    f"rose in most past months).")
    wait_plan = next((pl.id for pl in mediator.plans if pl.action == decision), None)
    would = _would_change_if(decision, p_council, theta, req, price, risk, plan)
    return JudgeResult(decision=decision, plan_id=wait_plan, amount=0.0, confidence=round(p_council, 4),
                       threshold=theta, kelly_fraction=round(f, 4), reason=reason, would_change_if=would)


def _decide_with_strategy(p_council: float, mediator: MediatorResult, req: UserRequest, s: dict,
                          price: float, risk: dict) -> JudgeResult:
    """Same rule as the backtest: direction check (theta), amount = min(budget, K x max loss /
    1-in-20 risk), stop placed so the stop plus a 1-in-20 overnight gap stays within the max loss."""
    from council.strategy import gap_for, plan as size_plan
    theta, aw, al = s["theta"], s["avg_win"], s["avg_loss"]
    er = float(expected_return(p_council, aw, al))
    f = kelly_fraction(p_council, aw, al)
    q95 = float(risk.get("var95_model_pct", risk.get("var95_used_pct", 0.1)))
    gap = gap_for(s["gaps_q95"], req.ticker)
    amount, stop = size_plan([p_council], [q95], [gap], theta, s["k"], aw, al, budget=req.amount,
                             max_loss=req.max_loss, c=s.get("c"))
    amount, stop = float(amount[0]), float(stop[0])
    buy_plans = [pl for pl in mediator.plans if pl.action not in ("wait", "avoid") and pl.checklist == "all_pass"]
    chosen = buy_plans[0] if buy_plans else None
    if amount > 0 and chosen is not None:
        chosen.stop_loss_pct = round(stop, 4)
        bar = f"above the {theta:.0%} bar tuned on 2016-2019" if theta > 0 else "the direction check is switched off for this stock (tuned on 2016-2019)"
        reason = (f"The council's probability of a rise is {p_council:.0%}, {bar}. A 1-in-20 bad period could fall "
                  f"{q95:.0%}, so with the risk budget tuned on 2016-2019 the Judge buys S${amount:,.2f} with a stop-loss "
                  f"{stop:.1%} below the price, leaving room for a 1-in-20 overnight gap of {gap:.1%}, so your loss "
                  f"should stay within S${req.max_loss:,.0f}.")
        would = _would_change_if("buy", p_council, theta, req, price, risk, chosen)
        return JudgeResult(decision="buy", plan_id=chosen.id, amount=round(amount, 2), confidence=round(p_council, 4),
                           threshold=theta, kelly_fraction=round(f, 4), reason=reason, would_change_if=would)
    decision = "avoid" if (p_council < 0.5 or er <= 0) else "wait"
    reason = (f"The council's {p_council:.0%} chance of a rise is not above the {theta:.0%} bar tuned on 2016-2019."
              if chosen is not None else "No buying plan passes all of your rules, so the Judge can't buy.")
    wait_plan = next((pl.id for pl in mediator.plans if pl.action == decision), None)
    return JudgeResult(decision=decision, plan_id=wait_plan, amount=0.0, confidence=round(p_council, 4),
                       threshold=theta, kelly_fraction=round(f, 4), reason=reason,
                       would_change_if=_would_change_if(decision, p_council, theta, req, price, risk, chosen))


def _would_change_if(decision: str, p: float, theta: float, req: UserRequest, price: float, risk: dict,
                     plan: Plan | None) -> list[str]:
    from council.rules import money
    out = []
    if decision == "buy":
        out.append(f"If the council's probability fell to {theta:.0%} or below, it would switch to WAIT")
    elif p <= theta:
        out.append(f"If the council's probability rose above {theta:.0%}, it could switch to BUY")
    if decision == "buy" and plan is not None and plan.stop_loss_pct:
        stop_price = price * (1 - plan.stop_loss_pct)
        out.append(f"If the price falls below {stop_price:,.2f}, the stop-loss sells and the council "
                   f"would switch to AVOID")
    var = risk.get("var95_used_pct")
    if var:
        need = req.amount * var
        if need > req.max_loss:
            out.append(f"If your max loss were at least {money(need, req.currency)}, your whole "
                       f"{money(req.amount, req.currency)} would pass the max-loss rule")
    return out
