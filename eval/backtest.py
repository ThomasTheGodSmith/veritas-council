"""Test-window evaluation (PLAN.md §2). The ONLY code allowed to open data/splits/test/.

Decision task: every 5 trading days in the test years, for the Dow index and for every company
that was a Dow member that day, a fixed example user ("wants to buy, max loss 10%, 20 trading
days") gets buy or don't-buy. Buying means buying at the next day's open and selling 20 trading
days later. Not buying earns 0.

Every strategy is scored the same way, next to the simple baselines:
  accuracy          did it call the direction right (buy and it rose, or skip and it fell)
  brier             are its probabilities honest (lower is better; only for probability models)
  buy_rate          share of decisions that were "buy"
  avg_buy_return    average 20-day return of the decisions it bought
  avg_return        average return per decision (0 when it stayed out)
  sharpe            avg_return / std, annualised (252/20 periods a year)
  loss_breaches     % of ALL decisions where it bought and the position fell 10%+ at some point
  breaches_per_buy  of the decisions it bought, the % that fell 10%+ (is it picking safer moments?)
  max_drawdown      index only: worst peak-to-trough of chaining one decision every 20 days

Run:  uv run python -m eval.backtest
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from council.analysts.base import Analyst, Context
from council.features import build_panel
from council.splits import load_test, read_split
from train import analyst_specs

from council.paths import MODELS, RESULTS  # noqa: E402
INDEX = "^DJI"
STEP = 5
MAX_LOSS = 0.10


def eval_context() -> Context:
    t = lambda n: load_test(n, for_evaluation=True)  # noqa: E731
    return Context(prices=t("prices"), macro=t("macro"), membership=t("membership"),
                   market_news=t("market_news"), company_news=t("company_news"))


def decision_rows(ctx: Context) -> pd.DataFrame:
    split = read_split()
    panel = build_panel(ctx.prices, ctx.macro, ctx.membership)
    panel = panel[(panel.date >= split.test_start) & panel.y.notna()]
    days = sorted(panel.loc[panel.ticker == INDEX, "date"].unique())
    return panel[panel.date.isin(days[::STEP])].reset_index(drop=True)


def strategies(rows: pd.DataFrame, ctx: Context) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Returns (buy decisions, probabilities) per strategy."""
    buys = {
        "always_buy": np.ones(len(rows), bool),
        "previous_direction": (rows.ret_20 > 0).to_numpy(),
        "sma50_rule": (rows.dist_sma50 > 0).to_numpy(),
    }
    probs: dict[str, np.ndarray] = {}
    for spec in analyst_specs():
        if not spec.get("votes", True) or spec["name"] == "devils_advocate":
            continue
        try:
            a = Analyst.load(spec["name"])
        except FileNotFoundError:
            continue
        print(f"  predicting {spec['name']} ...")
        probs[spec["name"]] = a.predict_proba(rows, ctx)
        buys[spec["name"]] = probs[spec["name"]] >= 0.5
    analysts = dict(probs)                       # analysts 1-10 only
    if len(analysts) >= 2:
        P = np.column_stack(list(analysts.values()))
        buys["majority_vote"] = (P >= 0.5).mean(1) > 0.5
        probs["simple_average"] = P.mean(1)
        buys["simple_average"] = probs["simple_average"] >= 0.5
    try:
        da = Analyst.load("devils_advocate")
        ctx.extra["analyst_probs"] = analysts
        probs["devils_advocate"] = da.predict_proba(rows, ctx)
        buys["devils_advocate"] = probs["devils_advocate"] >= 0.5
    except FileNotFoundError:
        pass
    try:   # the full council: stacking Mediator + Judge threshold, both tuned on validation only
        from council.mediator import council_probabilities
        probs["council"], buys["council"] = council_probabilities(rows, analysts)
        ablations(rows, analysts, probs, buys)
    except FileNotFoundError:
        print("  no saved mediator yet: run `uv run python -m council.mediator`")
    if "council" in buys:   # fairness check: buying LESS lowers breaches on its own, so compare
        rng = np.random.default_rng(0)   # with buying at random at the council's own buy rate
        group = (rows.ticker == INDEX).to_numpy()          # (separately for the index and members)
        rate = np.where(group, buys["council"][group].mean(), buys["council"][~group].mean())
        buys["random_at_council_rate"] = rng.random(len(rows)) < rate
    return buys, probs


def ablations(rows: pd.DataFrame, analysts: dict, probs: dict, buys: dict) -> None:
    """Our own questions (PLAN.md §2): which part of the council does the work?
    Every setting here was fixed on the validation years; nothing is re-tuned on test."""
    from council import judge as J
    from council.mediator import load_council

    from council.mediator import risk_model_loss

    cm = load_council()
    vol = rows["vol_20"]
    loss = risk_model_loss(rows)             # trained Drawdown Risk model (what the council uses)
    ones = np.ones(len(rows))
    # 1. The max-loss rule alone, no direction models: with the trained risk model, and with
    #    the plain volatility formula (1.645 x 20-day vol x sqrt(20)).
    buys["ablation_risk_model_only"] = J.buy_rule(ones, cm.avg_win, cm.avg_loss, 0.0, vol, loss_pct=loss)
    buys["ablation_vol_formula_only"] = J.buy_rule(ones, cm.avg_win, cm.avg_loss, 0.0, vol)
    # 2. The full council but with the volatility formula instead of the trained risk model.
    buys["ablation_council_vol_formula"] = J.buy_rule(probs["council"], cm.avg_win, cm.avg_loss, cm.theta, vol)
    # 3. The council's probability and threshold, but WITHOUT any max-loss rule.
    buys["ablation_council_no_rules"] = J.buy_rule(probs["council"], cm.avg_win, cm.avg_loss, cm.theta, vol,
                                                  max_loss=np.inf)
    # 4. The council without the Devil's Advocate (its input held at a neutral 0.5).
    p_nd = cm.p_council({a: np.asarray(v, float) for a, v in analysts.items()}, rows.reset_index(drop=True),
                        np.full(len(rows), 0.5))
    probs["ablation_no_devils_advocate"] = p_nd
    buys["ablation_no_devils_advocate"] = J.buy_rule(p_nd, cm.avg_win, cm.avg_loss, cm.theta, vol, loss_pct=loss)
    # 4. A simple average instead of the stacking Mediator, with its own validation-tuned threshold.
    val = json.loads((MODELS / "council_validation.json").read_text(encoding="utf-8"))
    theta_avg = val["simple_average_judge"]["theta"]
    buys["ablation_simple_average_judge"] = J.buy_rule(probs["simple_average"], cm.avg_win, cm.avg_loss,
                                                      theta_avg, vol, loss_pct=loss)
    # 5. Calibration of each risk estimate on the test years: how often the real loss beat it (target 5%).
    real = (-rows["mae"]).clip(lower=0).to_numpy()
    print(f"  test calibration: real loss beat the 1-in-20 estimate on "
          f"{np.mean(real > loss):.1%} of rows (risk model) vs {np.mean(real > J.vol_var_pct(vol)):.1%} (vol formula)")


def score(rows: pd.DataFrame, buy: np.ndarray, p: np.ndarray | None) -> dict[str, float]:
    r = np.where(buy, rows.fwd_return, 0.0)
    up = rows.y.to_numpy() == 1
    out = {
        "n": int(len(rows)),
        "accuracy": float((buy == up).mean()),
        "brier": float(((p - up) ** 2).mean()) if p is not None else math.nan,
        "buy_rate": float(buy.mean()),
        "avg_buy_return": float(rows.fwd_return[buy].mean()) if buy.any() else math.nan,
        "avg_return": float(r.mean()),
        "sharpe": float(r.mean() / r.std() * math.sqrt(252 / 20)) if r.std() > 0 else math.nan,
        "loss_breaches": float((buy & (rows.mae <= -MAX_LOSS)).mean()),
        "breaches_per_buy": float((rows.mae[buy] <= -MAX_LOSS).mean()) if buy.any() else math.nan,
    }
    if rows.ticker.nunique() == 1:   # chain one decision every 20 trading days
        chained = r[:: 20 // STEP]
        equity = np.cumprod(1 + chained)
        out["max_drawdown"] = float((equity / np.maximum.accumulate(equity) - 1).min())
        out["total_return"] = float(equity[-1] - 1)
    return out


def table(rows: pd.DataFrame, buys, probs) -> pd.DataFrame:
    return pd.DataFrame({k: score(rows, b, probs.get(k)) for k, b in buys.items()}).T


def main() -> None:
    ctx = eval_context()
    rows = decision_rows(ctx)
    buys, probs = strategies(rows, ctx)
    is_index = (rows.ticker == INDEX).to_numpy()
    idx = table(rows[is_index].reset_index(drop=True), {k: v[is_index] for k, v in buys.items()},
                {k: v[is_index] for k, v in probs.items()})
    mem = table(rows[~is_index].reset_index(drop=True), {k: v[~is_index] for k, v in buys.items()},
                {k: v[~is_index] for k, v in probs.items()})
    RESULTS.mkdir(exist_ok=True)
    idx.to_csv(RESULTS / "backtest_index.csv")
    mem.to_csv(RESULTS / "backtest_members.csv")
    rows.assign(**{f"p_{k}": v for k, v in probs.items()}).to_parquet(RESULTS / "test_decisions.parquet", index=False)
    split = read_split()
    meta = {"test_window": [split.test_start, str(rows.date.max().date())], "decision_every_trading_days": STEP,
            "index_decisions": int(is_index.sum()), "member_decisions": int((~is_index).sum())}
    (RESULTS / "backtest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(meta)
    print("\nDow index (DIA):\n", idx.round(3).to_string())
    print("\nDow members:\n", mem.round(3).to_string())


if __name__ == "__main__":
    main()
