"""The buying strategy, tuned per stock on its own validation years (2016-2019).

For an example user (budget S$500, max loss S$50, 20 trading days), every purchase is:
  1. Direction check: buy only if the council's probability of a rise is above THETA and the
     expected 20-day return is positive. THETA = 0 means "ignore the direction models".
  2. Size: amount = min(budget, K x max_loss / risk), where risk = the Drawdown Risk model's
     1-in-20 worst drop. K is the RISK BUDGET: K = 1 means a 1-in-20 bad period would lose
     about the max loss without a stop; bigger K buys more, relying on the stop-loss.
  3. Stop-loss: placed so that the stop PLUS a 1-in-20 overnight gap (council/execution.py)
     loses at most the max loss: stop_pct = max_loss / amount - gap. If that leaves no room,
     the amount is cut until it fits.
  Alternative sizing (tried in the same grid): a VOLATILITY-SCALED stop. The stop sits C x the
     1-in-20 drop below the price (wider for jumpy stocks, so normal swings don't sell it), and the
     amount is max_loss / (stop + gap), so hitting the stop still loses at most the max loss.
Choices are scored on realistic stop fills by
    average profit per decision - LAMBDA x average loss BEYOND the max loss,
with LAMBDA = 2 fixed in advance (a pound lost beyond the limit counts three times). The test
years are never used. How the choices are made robustly (to avoid picking a setting that just got
lucky in a few years) is explained in fit_strategy().

Run after `python -m council.mediator`:  uv run python -m council.strategy
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from council import judge as J
from council.execution import overnight_gap_q95, simulate
from council.paths import MODELS

BUDGET, MAX_LOSS = 500.0, 50.0
LAMBDA = 2.0
THETA_GRID = [0.0, 0.52, 0.54, 0.56, 0.58, 0.60]   # small on purpose: fewer chances to get lucky
K_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, float("inf")]
C_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]          # volatility-scaled stop: stop = C x 1-in-20 drop
MIN_STOP = 0.01
DEFAULT_OPTION = ("k", float("inf"))   # full budget with the gap-aware stop: the simple default
SWITCH_SHARE = 0.6                     # another sizing must beat the default in >= 60% of years
GATE_SHARE = 0.75                      # the direction check must beat "off" in >= 3 of 4 validation years
WF_MIN_HISTORY = 10                    # walk-forward years start once there are 10 years to learn from
WF_MAX_YEARS = 16                      # ... and use at most the 16 training years before the test years
STRATEGY_FILE = MODELS / "strategy.json"
STEP = 5


def gap_for(gaps: dict, ticker: str) -> float:
    return float(gaps.get(ticker, gaps.get("_default", 0.01)))


def plan(p, risk, gap, theta: float, k: float, avg_win: float, avg_loss: float,
         budget: float = BUDGET, max_loss: float = MAX_LOSS, c: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(amount, stop_pct) for each row. amount 0 = don't buy. c given = volatility-scaled stop."""
    p, risk, gap = (np.asarray(x, float) for x in (p, risk, gap))
    go = (p > theta) & (J.expected_return(p, avg_win, avg_loss) > 0) if theta > 0 else np.ones(len(p), bool)
    if c is not None:
        stop = np.maximum(c * np.nan_to_num(risk, nan=0.2), MIN_STOP)
        amount = np.where(go, np.minimum(budget, max_loss / (stop + gap)), 0.0)
        return amount, np.where(amount > 0, stop, np.inf)
    with np.errstate(divide="ignore"):
        amount = np.minimum(budget, k * max_loss / np.maximum(np.nan_to_num(risk, nan=1.0), 1e-6))
    amount = np.minimum(amount, max_loss / (gap + MIN_STOP))      # leave room for the stop + a gap
    amount = np.where(go, amount, 0.0)
    stop = np.where(amount > 0, max_loss / np.maximum(amount, 1e-9) - gap, np.inf)
    return amount, stop


def objective(profit: np.ndarray) -> float:
    beyond = np.clip(-profit - MAX_LOSS, 0, None)
    return float(profit.mean() - LAMBDA * beyond.mean())


def walk_forward_risk(panel: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Out-of-sample 1-in-20 risk for many TRAINING years: for each year, a fresh Drawdown Risk
    model learns only from rows whose outcome was known before that year starts, then predicts
    that year. This gives the sizing choice ~4x more out-of-sample decisions than 2016-2019 alone.
    (The test years are never in `panel`: it is built from the training folder.)"""
    from council.analysts.drawdown_risk import DrawdownRisk
    lab = panel[panel.y.notna()]
    days = sorted(lab.loc[lab.ticker == "^DJI", "date"].unique())[::STEP]
    rows = lab[lab.date.isin(days)]
    out = []
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        fit = lab[lab.label_known < start]
        target = rows[rows.date.dt.year == yr]
        if len(fit) < 500 or target.empty:
            continue
        m = DrawdownRisk()
        m.fit(fit)
        out.append(target[["date", "ticker", "fwd_return"]].assign(risk=m.predict_loss(target)))
    return pd.concat(out, ignore_index=True)


def yearly_scores(rows: pd.DataFrame, prices: pd.DataFrame, amount: np.ndarray, stop: np.ndarray) -> pd.Series:
    sim = simulate(rows, prices, amount, stop)
    return pd.Series(sim.profit.to_numpy()).groupby(rows.date.dt.year.to_numpy()).apply(lambda s: objective(s.to_numpy()))


def fit_strategy(save: bool = True) -> dict:
    """Two choices, each made robustly on training years only:
      1. SIZING, on a walk-forward check over many training years with the direction check off.
         Keep the simple default (full budget, gap-aware stop) unless another option beats it in
         at least SWITCH_SHARE of those years: one lucky stretch can't flip the choice.
      2. DIRECTION CHECK, on the validation slice (the only years with out-of-sample council
         probabilities). Switched on only if a threshold beats "off" in at least GATE_SHARE of the
         validation years."""
    from council.data import train_panel
    from council.mediator import load_council
    from council.splits import load_train, read_split

    split = read_split()
    panel = train_panel()
    prices = load_train("prices")
    cm = load_council()
    test_year = pd.Timestamp(split.test_start).year
    first = int(panel.loc[panel.y.notna(), "date"].dt.year.min())
    wf_years = list(range(max(first + WF_MIN_HISTORY, test_year - WF_MAX_YEARS), test_year))
    wf = walk_forward_risk(panel, wf_years)
    gaps = overnight_gap_q95(prices, wf.ticker.unique())
    gap = wf.ticker.map(lambda t: gap_for(gaps, t)).to_numpy()
    p_one = np.ones(len(wf))

    # ---- 1. sizing: per-year scores for every option, direction check off
    options = [("k", k) for k in K_GRID] + [("c", c) for c in C_GRID]
    per_year = {}
    for kind, v in options:
        a, s = plan(p_one, wf.risk, gap, 0.0, v if kind == "k" else 1.0, cm.avg_win, cm.avg_loss,
                    c=v if kind == "c" else None)
        per_year[(kind, v)] = yearly_scores(wf, prices, a, s)
    table = pd.DataFrame(per_year)
    default = DEFAULT_OPTION
    means = table.mean()
    best = means.idxmax()
    wins = float((table[best] > table[default]).mean())
    chosen = best if best != default and wins >= SWITCH_SHARE else default

    # ---- 2. direction check on the validation slice, with the chosen sizing
    oof = pd.read_parquet(MODELS / "council_oof.parquet")
    val = wf[wf.date >= pd.Timestamp(split.val_start)].merge(oof, on=["date", "ticker"])
    vgap = val.ticker.map(lambda t: gap_for(gaps, t)).to_numpy()
    kind, v = chosen
    def sized(theta):
        return plan(val.p_council, val.risk, vgap, theta, v if kind == "k" else 1.0, cm.avg_win, cm.avg_loss,
                    c=v if kind == "c" else None)
    gate = {th: yearly_scores(val, prices, *sized(th)) for th in THETA_GRID}
    gtab = pd.DataFrame(gate)
    theta = 0.0
    better = [th for th in THETA_GRID if th > 0 and (gtab[th] > gtab[0.0]).mean() >= GATE_SHARE
              and gtab[th].mean() > gtab[0.0].mean()]
    if better:
        theta = max(better, key=lambda th: gtab[th].mean())

    out = {"note": "Chosen on training years only: walk-forward years for sizing, the validation slice for "
                   "the direction check. Realistic stop fills. The test years are never used.",
           "budget": BUDGET, "max_loss": MAX_LOSS, "lambda": LAMBDA,
           "theta": theta, "k": v if kind == "k" else None, "c": v if kind == "c" else None,
           "walk_forward_years": [int(y) for y in table.index], "walk_forward_decisions": int(len(wf)),
           "sizing_mean_score": {f"{k}={x}": round(float(m), 4) for (k, x), m in means.items()},
           "best_option": f"{best[0]}={best[1]}", "best_beats_default_in_years": round(wins, 3),
           "default_option": f"{default[0]}={default[1]}",
           "direction_check_mean_score": {str(th): round(float(m), 4) for th, m in gtab.mean().items()},
           "gaps_q95": gaps, "avg_win": cm.avg_win, "avg_loss": cm.avg_loss}
    if save:
        STRATEGY_FILE.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def load_strategy() -> dict | None:
    if not STRATEGY_FILE.exists():
        return None
    s = json.loads(STRATEGY_FILE.read_text(encoding="utf-8"))
    s["k"] = float(s["k"]) if s.get("k") is not None else 1.0
    s["c"] = float(s["c"]) if s.get("c") is not None else None
    return s


if __name__ == "__main__":
    s = fit_strategy()
    print(json.dumps({k: s[k] for k in ["theta", "k", "c", "best_option", "best_beats_default_in_years",
                                        "walk_forward_years", "walk_forward_decisions", "sizing_mean_score",
                                        "direction_check_mean_score"]}, indent=2, default=str))
