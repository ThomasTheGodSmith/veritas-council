"""A FAKE RunResult for building and testing the report, narrator and broker.

ALL NUMBERS HERE ARE MADE UP. They only show the shape of a real council run.
Never show this as a real prediction.
"""
from __future__ import annotations

from council.analysts.base import Signal
from council.schema import (AnalystReview, Disagreement, JudgeResult, MediatorResult, Plan,
                            PlanStep, RuleCheck, RunResult, UserRequest)


def _sig(agent, p, er, action, verdict, key, reasons, acc, n):
    return Signal(agent=agent, probability_up=p, horizon_days=20, expected_return_pct=er,
                  recommended_action=action, user_idea_verdict=verdict, key_numbers=key,
                  top_reasons=reasons, track_record={"validation_accuracy": acc, "n": n})


def fake_run_result() -> RunResult:
    request = UserRequest(ticker="^DJI", amount=500, max_loss=50, horizon_days=20,
                          thinks_up=True, idea_text="I think it'll go up")
    signals = [
        _sig("momentum", 0.58, 0.8, "buy", "supports", {"sma20_vs_sma50": 0.031, "rsi": 61.0},
             [("trend_strength", 0.42), ("rsi", 0.18), ("macd", 0.07)], 0.56, 812),
        _sig("mean_reversion", 0.46, -0.3, "wait", "challenges", {"zscore_20": 1.4, "bb_pos": 0.88},
             [("zscore_20", -0.35), ("bb_pos", -0.21)], 0.53, 812),
        _sig("ml_forecaster", 0.55, 0.5, "wait", "supports", {"vix": 16.2, "dgs10": 4.1},
             [("ret_60", 0.12), ("vix", -0.08), ("dgs10", 0.05)], 0.55, 812),
        _sig("risk_officer", 0.53, 0.3, "wait", "supports", {"var_95_pct": 7.6, "garch_vol": 0.14},
             [("garch_vol", -0.2)], 0.52, 812),
        _sig("news_reaction", 0.51, 0.1, "wait", "supports", {"similar_headlines": 20, "rose": 11},
             [("word:record", 0.09), ("word:tariff", -0.06)], 0.51, 640),
        _sig("pattern_oracle", 0.655, 1.1, "buy", "supports", {"futures_up": 131, "futures": 200},
             [("last_token", 0.3)], 0.54, 812),
        _sig("raw_reader", 0.49, -0.1, "wait", "challenges", {"window_days": 120},
             [("volume_last_10d", -0.11)], 0.52, 812),
        _sig("market_regime", 0.57, 0.7, "buy", "supports", {"p_bull": 0.71, "p_bear": 0.09},
             [("regime_bull", 0.4)], 0.55, 812),
        _sig("macro", 0.52, 0.2, "wait", "supports", {"t10y2y": 0.45, "unrate": 4.3},
             [("t10y2y", 0.1), ("cpi_yoy", -0.04)], 0.53, 812),
        _sig("dow_breadth", 0.54, 0.4, "wait", "supports", {"breadth_up": 18, "top5_weight": 0.31},
             [("breadth_up", 0.15)], 0.54, 812),
        _sig("devils_advocate", 0.46, -0.2, "wait", "challenges", {"p_majority_wrong": 0.46},
             [("disagreement", 0.22)], 0.57, 240),
    ]
    plans = [
        Plan(id="A", action="buy", amount=350, steps=[PlanStep(amount=350, when="now")],
             stop_loss_pct=0.14, expected_return_pct=0.6, worst_case_loss=49.0, p_hit_stop=0.06),
        Plan(id="B", action="wait", amount=0, expected_return_pct=0.0, worst_case_loss=0.0, p_hit_stop=0.0),
        Plan(id="C", source="mediator_original", action="split_buy", amount=350,
             steps=[PlanStep(amount=200, when="now"), PlanStep(amount=150, when="+10 days")],
             stop_loss_pct=0.14, expected_return_pct=0.6, worst_case_loss=38.0, p_hit_stop=0.05,
             why_better="Same expected return as A on 10,000 simulated paths, worst-case loss S$38 vs S$49"),
    ]
    return RunResult(
        run_id="FAKE_2026-11-15_1402_DIA",
        decision_date="2026-11-16",
        data_as_of="2026-11-13",
        request=request,
        price=523.15,
        signals=signals,
        fundamentals={"note": "index fund, no P/E card", "dividend_yield_pct": 1.6},
        crossexam=[Disagreement(a="momentum", b="mean_reversion", a_says_up=True, n_past=340,
                                a_win_rate=0.42,
                                summary="Mean Reversion won 58% of 340 past disagreements like this")],
        devils_advocate={"p_majority_wrong": 0.46, "drivers": ["high disagreement", "low volatility"]},
        rules=[RuleCheck(rule="budget", passed=True, text="Budget ≤ S$500"),
               RuleCheck(rule="max_loss", passed=True,
                         text="Max loss S$50: a 1-in-20 bad month could lose S$38 on plan C"),
               RuleCheck(rule="horizon", passed=True, text="Horizon 20 days"),
               RuleCheck(rule="sectors", passed=True, text="No avoided sectors")],
        reviews=[AnalystReview(agent="momentum", positives=["Right 61% of the time in bull regimes (validation)"],
                               negatives=["Disagrees with Mean Reversion, which won 58% of past disagreements like this"],
                               evidence_strength="strong", trust_score=0.64),
                 AnalystReview(agent="mean_reversion", positives=["Won 58% of 340 past disagreements with Momentum"],
                               negatives=["Only 51% right in bull regimes"], evidence_strength="strong",
                               trust_score=0.55)],
        mediator=MediatorResult(p_council=0.57, weights={"momentum": 0.22, "pattern_oracle": 0.18},
                                plans=plans,
                                dropped=[{"plan": "buy 500 now", "failed_rule": "max_loss",
                                          "fix_tried": "reduced to 350, became plan A"}]),
        judge=JudgeResult(decision="buy", plan_id="C", amount=350, confidence=0.62, threshold=0.55,
                          kelly_fraction=0.07, reason="p_council 0.57 is above the threshold 0.55 and every rule passes",
                          would_change_if=["If the price falls below $498, the council would switch to AVOID"]),
        risk={"var_95": 38.0, "max_safe_amount": 360.0, "price_low": 498.0, "price_high": 548.0},
    )
