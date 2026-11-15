"""Hard rules (council/rules.py), the Judge's arithmetic and Decline handling. No data files needed."""
from __future__ import annotations

import numpy as np

from council import judge as J
from council import rules as R
from council.council import apply_decline
from council.mediator import build_plans, simulate
from council.schema import MediatorResult, Plan, UserRequest


def test_budget_and_max_loss():
    req = UserRequest(amount=500, max_loss=50)
    assert R.check_budget(500, req).passed and not R.check_budget(501, req).passed
    assert R.check_max_loss(500, req, 0.10).passed          # 500 * 10% = 50, exactly the limit
    bad = R.check_max_loss(500, req, 0.14)
    assert not bad.passed and "S$70" in bad.text


def test_fixed_amount_really_fits_the_max_loss():
    req = UserRequest(amount=500, max_loss=50)
    amt = R.max_amount_within_loss(req, 0.1674)              # 298.68..., must round DOWN
    assert R.check_max_loss(amt, req, 0.1674).passed
    assert R.max_amount_within_loss(req, 0.01) == 500        # capped by the budget


def test_horizon_rule():
    assert R.check_horizon(UserRequest(horizon_days=20)).passed
    assert not R.check_horizon(UserRequest(horizon_days=10)).passed


def test_sectors_for_stocks_and_the_index():
    req = UserRequest(ticker="JPM", avoid_sectors=["banks"])
    assert not R.check_sectors("JPM", req).passed
    assert R.check_sectors("AAPL", req).passed
    w = {"JPM": 0.1, "GS": 0.2, "AAPL": 0.7}                  # 30% of the "index" is financials
    assert not R.check_sectors("^DJI", req, w).passed
    assert R.check_sectors("^DJI", UserRequest(avoid_sectors=["oil"]), w).passed
    assert R.normalise_sector("Tech") == "Technology" and R.normalise_sector("pizza") is None


def test_kelly_and_expected_return():
    # p = 0.6, b = 1: full Kelly = (0.6 - 0.4) / 1 = 0.2, half = 0.1
    assert abs(J.kelly_fraction(0.6, 0.05, 0.05) - 0.1) < 1e-9
    assert J.kelly_fraction(0.4, 0.05, 0.05) == 0.0
    assert J.expected_return(0.5, 0.04, 0.02) > 0


def test_buy_rule_needs_theta_return_and_low_vol():
    p = np.array([0.7, 0.7, 0.5])
    vol = np.array([0.01, 0.05, 0.01])                       # 0.05 daily: VaR ~37% > 10%
    assert J.buy_rule(p, 0.04, 0.04, 0.6, vol).tolist() == [True, False, False]


def _flat_risk(var=0.05, drift=0.0, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    paths = np.exp(np.cumsum(drift + 0.01 * rng.standard_normal((n, 20)), axis=1))
    return {"var95_used_pct": var, "paths": paths}


def test_simulate_buy_and_hold_and_stop():
    paths = np.array([[1.0, 1.1, 1.2], [1.0, 0.8, 0.9]])
    pnl, hit = simulate(paths, [(100, 0)], stop_pct=None)
    assert np.allclose(pnl, [20, -10]) and not hit.any()
    pnl, hit = simulate(paths, [(100, 0)], stop_pct=0.1)    # path 2 touches 0.8 -> sold at -10%
    assert np.allclose(pnl, [20, -10]) and hit.tolist() == [False, True]


def test_plans_fix_max_loss_and_never_pass_a_failing_plan():
    req = UserRequest(amount=500, max_loss=50)
    med = build_plans(0.6, req, "AAPL", _flat_risk(var=0.20))
    a = next(p for p in med.plans if p.id == "A")
    assert a.amount == 250 and a.checklist == "all_pass"
    assert med.dropped and med.dropped[0]["failed_rule"] == "max_loss"
    assert all(p.checklist == "all_pass" for p in med.plans)


def test_own_plan_only_when_it_beats_plan_a():
    req = UserRequest(amount=500, max_loss=100)
    falling = build_plans(0.6, req, "AAPL", _flat_risk(drift=-0.003))   # prices drift down
    own = [p for p in falling.plans if p.source == "mediator_original"]
    assert own and own[0].why_better
    a = falling.plans[0]
    assert own[0].worst_case_loss < a.worst_case_loss


def test_judge_cannot_buy_when_no_plan_passes():
    req = UserRequest()
    med = MediatorResult(p_council=0.9, plans=[Plan(id="B", action="wait")])
    out = J.decide(0.9, med, req, {"theta": 0.55, "avg_win": 0.04, "avg_loss": 0.04}, 100.0, {})
    assert out.decision == "wait" and out.amount == 0


def test_judge_buys_with_half_kelly_capped_by_plan(monkeypatch):
    """The fallback sizing, used when no strategy has been tuned yet."""
    monkeypatch.setattr("council.strategy.load_strategy", lambda: None)
    req = UserRequest(amount=500)
    plan = Plan(id="A", action="buy", amount=300, stop_loss_pct=0.1)
    med = MediatorResult(p_council=0.7, plans=[plan, Plan(id="B", action="wait")])
    out = J.decide(0.7, med, req, {"theta": 0.55, "avg_win": 0.04, "avg_loss": 0.04}, 100.0, {})
    assert out.decision == "buy" and abs(out.amount - 100.0) < 1e-6     # half-Kelly 20% of 500
    low = J.decide(0.3, med, req, {"theta": 0.55, "avg_win": 0.04, "avg_loss": 0.04}, 100.0, {})
    assert low.decision == "avoid"


def test_decline_reasons():
    req = UserRequest(amount=500, max_loss=50)
    r1 = apply_decline(req, "too risky for me")
    assert r1.max_loss == 25 and r1.amount == 250 and r1.decline_reasons == ["too risky for me"]
    r2 = apply_decline(r1, "that's too much money, and no tech please")
    assert r2.amount == 125 and r2.max_loss == 25 and "Technology" in r2.avoid_sectors and len(r2.decline_reasons) == 2
    r3 = apply_decline(req, "I think it will go down")
    assert r3.thinks_up is False
