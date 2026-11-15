"""Train every analyst on the training years only (PLAN.md §2, §5, §7).

Steps:
  1. Build the training panel from data/splits/train/ (no test years exist in it).
  2. Fit each analyst on the FIT years (training minus the validation slice minus the embargo).
  3. Predict the validation slice (2016-2019), which those models haven't seen. These
     out-of-sample predictions become each analyst's track record, and are what the
     Mediator, Devil's Advocate and Judge learn from.
  4. Refit each analyst on all training years and save it to models/.

Analysts are listed in config/agents.yaml. Each file defines build(); files that aren't
written yet are skipped. `stage` controls order: stage 2 analysts get stage 1's fitted
analysts in ctx.extra["analysts"]; stage 3 also get the validation predictions.

Run:  uv run python train.py              (all analysts)
      uv run python train.py momentum     (just these)
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from council.analysts.base import MODELS, Analyst, Context
from council.data import train_panel
from council.splits import fit_mask, load_train, read_split, validation_mask

INDEX = "^DJI"
VAL_PREDS = MODELS / "validation_predictions"
RISK_PREDS = MODELS / "risk_predictions"


def validation_predictions() -> pd.DataFrame:
    """Every analyst's out-of-sample predictions on the validation slice (one file per analyst,
    so analysts trained in parallel don't overwrite each other)."""
    files = sorted(VAL_PREDS.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def analyst_specs() -> list[dict]:
    cfg = yaml.safe_load(Path("config/agents.yaml").read_text(encoding="utf-8"))
    return cfg["analysts"]


def build_analyst(spec: dict) -> Analyst | None:
    module = spec["file"].removesuffix(".py").replace("/", ".")
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        print(f"  skip {spec['name']}: {e}")
        return None
    if not hasattr(mod, "build"):
        print(f"  skip {spec['name']}: not written yet")
        return None
    return mod.build()


def training_context(extra: dict | None = None) -> Context:
    return Context(
        prices=load_train("prices"),
        macro=load_train("macro"),
        membership=load_train("membership"),
        market_news=load_train("market_news"),
        company_news=load_train("company_news"),
        extra=extra or {},
    )


def main(only: list[str]) -> None:
    split = read_split()
    panel = train_panel()
    labelled = panel[panel.y.notna()]
    index_days = panel.loc[panel.ticker == INDEX, "date"]
    fit_rows = labelled[fit_mask(labelled.date, index_days, split).values]
    val_rows = labelled[validation_mask(labelled.date, split).values]
    print(f"fit rows {len(fit_rows):,} ({fit_rows.date.min().date()} to {fit_rows.date.max().date()}), "
          f"validation rows {len(val_rows):,} ({val_rows.date.min().date()} to {val_rows.date.max().date()})")

    ctx = training_context()
    VAL_PREDS.mkdir(parents=True, exist_ok=True)
    preds = validation_predictions()
    fitted: dict[str, Analyst] = {}

    specs = sorted(analyst_specs(), key=lambda s: s.get("stage", 1))
    for spec in specs:
        if only and spec["name"] not in only:
            try:
                fitted[spec["name"]] = Analyst.load(spec["name"])   # reuse for later stages
            except FileNotFoundError:
                pass
            continue
        analyst = build_analyst(spec)
        if analyst is None:
            continue
        t0 = time.time()
        ctx.extra = {"analysts": dict(fitted), "validation_predictions": preds}
        print(f"[{spec['name']}] fitting on fit years ...")
        try:
            analyst.fit(fit_rows, ctx)
            p = analyst.predict_proba(val_rows, ctx)
        except Exception as e:   # e.g. Dow Breadth in a single-stock run: no member companies
            print(f"[{spec['name']}] SKIPPED: {type(e).__name__}: {str(e)[:150]}")
            continue
        vp = val_rows[["date", "ticker", "y", "fwd_return", "mae", "label_known"]].assign(agent=analyst.name, p=p)
        if getattr(analyst, "kind", "direction") == "risk":
            # risk models predict a loss, not P(up): keep them out of the Mediator's inputs
            RISK_PREDS.mkdir(parents=True, exist_ok=True)
            vp.to_parquet(RISK_PREDS / f"{analyst.name}.parquet", index=False)
            cover = (-vp.mae > vp.p).mean()
            print(f"[{spec['name']}] validation: actual loss beat the predicted 1-in-20 loss on {cover:.1%} "
                  f"of rows (target 5%); refitting on all training years ...")
        else:
            vp.to_parquet(VAL_PREDS / f"{analyst.name}.parquet", index=False)   # one file per analyst
            preds = validation_predictions()
            acc = ((vp.p >= 0.5) == (vp.y == 1)).mean()
            print(f"[{spec['name']}] validation accuracy {acc:.3f} (n={len(vp):,}); refitting on all training years ...")
        if getattr(analyst, "refit_on_full", True):
            analyst = build_analyst(spec)
            analyst.fit(labelled, ctx)
        analyst.save()
        fitted[analyst.name] = analyst
        print(f"[{spec['name']}] saved ({time.time() - t0:.0f}s)")

    preds = validation_predictions()
    records = {}
    for agent, g in preds.groupby("agent"):
        right = (g.p >= 0.5) == (g.y == 1)
        records[agent] = {
            "validation_accuracy": round(float(right.mean()), 4),
            "brier": round(float(((g.p - g.y) ** 2).mean()), 4),
            "n": int(len(g)),
            "index_accuracy": round(float(right[g.ticker == INDEX].mean()), 4),
        }
    (MODELS / "track_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
