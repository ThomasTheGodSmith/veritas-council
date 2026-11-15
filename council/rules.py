"""Hard user requirements, enforced in code (PLAN.md §1 "Hard rules are enforced in code").

Four rules, each one pass/fail. Nothing later in the pipeline can overrule a failed rule:
the Mediator must fix or drop a failing plan, and the Judge only sees plans that pass.

  budget        the plan spends no more than the user's budget
  max_loss      a 1-in-20 bad outcome (the Risk Officer's 95% Value-at-Risk) loses no more
                than the user's max loss:  amount * VaR% <= max_loss
  horizon       our models predict 20 trading days ahead, so the plan needs a horizon of at
                least that many days
  avoid_sectors the stock is not in a sector the user wants to avoid. For the Dow index
                itself we look at how much of the index (by Dow price weight) sits in those
                sectors and fail if it is more than 25%.
"""
from __future__ import annotations

import math

from council.schema import RuleCheck, UserRequest

MODEL_HORIZON = 20             # every analyst is trained to predict 20 trading days ahead
INDEX_SECTOR_LIMIT = 0.25      # the Dow fails avoid_sectors if > 25% of its weight is avoided

# Sector of every company that has been in the Dow in our data (current and past members),
# using the GICS sector names. "^DJI" / "DIA" are the index, handled separately.
SECTORS: dict[str, str] = {
    "AAPL": "Technology", "CSCO": "Technology", "CRM": "Technology", "HPQ": "Technology",
    "IBM": "Technology", "INTC": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AIG": "Financials", "AXP": "Financials", "BAC": "Financials", "C": "Financials",
    "GS": "Financials", "JPM": "Financials", "TRV": "Financials", "V": "Financials",
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials",
    "MMM": "Industrials", "RTX": "Industrials",
    "AMGN": "Health Care", "JNJ": "Health Care", "MRK": "Health Care", "PFE": "Health Care",
    "UNH": "Health Care",
    "KO": "Consumer Staples", "MDLZ": "Consumer Staples", "MO": "Consumer Staples",
    "PG": "Consumer Staples", "WBA": "Consumer Staples", "WMT": "Consumer Staples",
    "AMZN": "Consumer Discretionary", "FL": "Consumer Discretionary", "GT": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "DIS": "Communication Services", "GOOGL": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services",
    "CVX": "Energy", "XOM": "Energy",
    "DD": "Materials", "DOW": "Materials", "HWM": "Materials", "IP": "Materials", "SHW": "Materials",
}

# Everyday words a user might type, mapped to the sector names above.
SECTOR_ALIASES: dict[str, str] = {
    "tech": "Technology", "technology": "Technology",
    "finance": "Financials", "financial": "Financials", "financials": "Financials", "banks": "Financials",
    "banking": "Financials",
    "industrial": "Industrials", "industrials": "Industrials", "aerospace": "Industrials",
    "defense": "Industrials", "defence": "Industrials",
    "health": "Health Care", "healthcare": "Health Care", "health care": "Health Care",
    "pharma": "Health Care",
    "staples": "Consumer Staples", "consumer staples": "Consumer Staples", "tobacco": "Consumer Staples",
    "food": "Consumer Staples",
    "retail": "Consumer Discretionary", "consumer discretionary": "Consumer Discretionary",
    "media": "Communication Services", "telecom": "Communication Services",
    "communication services": "Communication Services",
    "energy": "Energy", "oil": "Energy", "fossil fuels": "Energy",
    "materials": "Materials", "chemicals": "Materials",
}

INDEX_TICKERS = {"^DJI", "DIA"}


def normalise_sector(name: str) -> str | None:
    """'tech' -> 'Technology'. Returns None if we don't recognise the word."""
    key = name.strip().lower()
    if key in SECTOR_ALIASES:
        return SECTOR_ALIASES[key]
    for s in set(SECTORS.values()):
        if s.lower() == key:
            return s
    return None


def sector_of(ticker: str) -> str:
    if ticker in INDEX_TICKERS:
        return "Index (all sectors)"
    return SECTORS.get(ticker, "Unknown")


def money(x: float, currency: str = "SGD") -> str:
    symbol = "S$" if currency == "SGD" else "$" if currency == "USD" else f"{currency} "
    return f"{symbol}{x:,.0f}"


def check_budget(amount: float, req: UserRequest) -> RuleCheck:
    ok = amount <= req.amount + 1e-9
    return RuleCheck(rule="budget", passed=ok,
                     text=f"Budget {money(req.amount, req.currency)}: plan spends {money(amount, req.currency)}")


def check_max_loss(amount: float, req: UserRequest, var95_pct: float) -> RuleCheck:
    """var95_pct: the loss (as a fraction, e.g. 0.07) a 1-in-20 bad period would cause."""
    bad = amount * var95_pct
    ok = bad <= req.max_loss + 1e-9
    return RuleCheck(rule="max_loss", passed=ok,
                     text=f"Max loss {money(req.max_loss, req.currency)}: a 1-in-20 bad "
                          f"{req.horizon_days}-day period could lose {money(bad, req.currency)}")


def check_horizon(req: UserRequest, plan_days: int = MODEL_HORIZON) -> RuleCheck:
    ok = req.horizon_days >= plan_days
    return RuleCheck(rule="horizon", passed=ok,
                     text=f"Horizon {req.horizon_days} trading days: the plan holds for {plan_days} days"
                          + ("" if ok else " (our models only predict 20 days ahead)"))


def check_sectors(ticker: str, req: UserRequest, index_weights: dict[str, float] | None = None) -> RuleCheck:
    avoid = {s for s in (normalise_sector(a) for a in req.avoid_sectors) if s}
    if not avoid:
        return RuleCheck(rule="avoid_sectors", passed=True, text="No sectors to avoid")
    names = ", ".join(sorted(avoid))
    if ticker in INDEX_TICKERS:
        w = index_weights or {}
        share = sum(v for t, v in w.items() if SECTORS.get(t) in avoid)
        ok = share <= INDEX_SECTOR_LIMIT
        return RuleCheck(rule="avoid_sectors", passed=ok,
                         text=f"Avoid {names}: {share:.0%} of the Dow is in these sectors "
                              f"(limit {INDEX_SECTOR_LIMIT:.0%})")
    sector = sector_of(ticker)
    ok = sector not in avoid
    return RuleCheck(rule="avoid_sectors", passed=ok, text=f"Avoid {names}: {ticker} is in {sector}")


def check_plan(amount: float, ticker: str, req: UserRequest, var95_pct: float,
               index_weights: dict[str, float] | None = None, plan_days: int = MODEL_HORIZON) -> list[RuleCheck]:
    """All four rules for one plan. A plan with amount 0 (wait / avoid) can't break money rules."""
    return [
        check_budget(amount, req),
        check_max_loss(amount, req, var95_pct),
        check_horizon(req, plan_days),
        check_sectors(ticker, req, index_weights),
    ]


def failed(checks: list[RuleCheck]) -> list[str]:
    return [c.rule for c in checks if not c.passed]


def max_amount_within_loss(req: UserRequest, var95_pct: float) -> float:
    """The biggest amount whose 1-in-20 bad outcome still fits the max loss (and the budget)."""
    if var95_pct <= 0:
        return req.amount
    cents = math.floor(100 * req.max_loss / var95_pct) / 100     # round DOWN so it really fits
    return min(req.amount, cents)
