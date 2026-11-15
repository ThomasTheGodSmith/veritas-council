"""The shapes passed between the council steps and to the report/app (PLAN.md §1, §1c).

council/council.py produces a RunResult; report.py, narrator.py and app.py only read it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from council.analysts.base import Signal


class UserRequest(BaseModel):
    ticker: str = "^DJI"                 # "^DJI" means the Dow itself, bought via DIA
    amount: float = 500.0                # in the user's currency (S$)
    currency: str = "SGD"
    max_loss: float = 50.0               # the most the user will accept losing, same currency
    horizon_days: int = 20
    avoid_sectors: list[str] = []
    thinks_up: bool = True               # the user's own idea: "I think it'll go up"
    idea_text: str = "I think it'll go up"
    decline_reasons: list[str] = []      # filled when the user presses Decline and re-runs


class Disagreement(BaseModel):
    a: str
    b: str
    a_says_up: bool
    n_past: int
    a_win_rate: float                    # share of past disagreements like this that `a` won
    summary: str


class RuleCheck(BaseModel):
    rule: str                            # e.g. "max_loss"
    passed: bool
    text: str                            # e.g. "Max loss S$50: a 1-in-20 bad month could lose S$70"


class AnalystReview(BaseModel):
    agent: str
    positives: list[str] = []
    negatives: list[str] = []
    evidence_strength: Literal["strong", "medium", "weak"] = "weak"
    fits_user_rules: bool = True
    trust_score: float = 0.5


class PlanStep(BaseModel):
    amount: float
    when: str                            # "now", "+10 days", "if price dips 3% within 10 days"


class Plan(BaseModel):
    id: str
    source: Literal["standard", "mediator_original"] = "standard"
    action: Literal["buy", "wait", "avoid", "split_buy", "staged_buy", "limit_buy"]
    amount: float = 0.0
    steps: list[PlanStep] = []
    stop_loss_pct: float | None = None
    expected_return_pct: float | None = None
    worst_case_loss: float | None = None  # 5th percentile loss on the simulated paths, currency
    p_hit_stop: float | None = None
    why_better: str | None = None
    checklist: Literal["all_pass", "fails"] = "all_pass"
    failed_rules: list[str] = []


class MediatorResult(BaseModel):
    p_council: float
    weights: dict[str, float] = {}       # how much the stacking model trusted each analyst today
    plans: list[Plan] = []
    dropped: list[dict] = []             # {"plan": ..., "failed_rule": ..., "fix_tried": ...}


class JudgeResult(BaseModel):
    decision: Literal["buy", "wait", "avoid"]
    plan_id: str | None = None
    amount: float = 0.0
    confidence: float
    threshold: float
    kelly_fraction: float = 0.0
    reason: str
    would_change_if: list[str] = []      # "If the price falls below $X, the council would switch to AVOID"


class RunResult(BaseModel):
    run_id: str
    decision_date: str                   # the trading day the plan would be acted on
    data_as_of: str                      # newest data used (always before decision_date)
    request: UserRequest
    price: float                         # latest close of the thing being bought
    signals: list[Signal]
    fundamentals: dict[str, float | str] = {}   # info card only, no vote
    crossexam: list[Disagreement] = []
    devils_advocate: dict = Field(default_factory=dict)   # {"p_majority_wrong": .., "drivers": [...]}
    rules: list[RuleCheck] = []
    reviews: list[AnalystReview] = []
    mediator: MediatorResult
    judge: JudgeResult
    risk: dict = Field(default_factory=dict)   # Risk Officer's VaR, price range, max_safe_amount, paths summary
    notes: list[str] = []                # e.g. analysts skipped today, fallbacks used (optional)
