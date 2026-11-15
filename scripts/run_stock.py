"""Market Council for ANY stock: one command builds, trains, tunes and tests the whole project for it.

    uv run python scripts/run_stock.py AMD          a single stock, in the Dow's place
    uv run python scripts/run_stock.py ^DJI         the Dow index on its own
    uv run python scripts/run_stock.py DOW          the main project (the Dow + its 30 companies)

What it does, with exactly the same programs and rules every time:
  0. data      if the stock's prices aren't in data/raw/prices/, download its full daily history and
               share splits from Yahoo Finance (yfinance). Macro and world news are shared by every
               run; company headlines are used if the stock is in the Kaggle company-news dataset.
  1. split     its own 80/20 split from its own years: the training folder has no test years at all
  2. panel     features and labels from the training folder only
  3. train     every analyst: fit 1992-2015, validate 2016-2019, refit 1992-2019
  4. mediator  Devil's Advocate inputs, stacking Mediator (validation slice only)
  5. strategy  Judge threshold + risk budget, tuned on the stock's own validation years with
               realistic stop-loss fills (council/strategy.py)
  6. backtest  accuracy against baselines on its test years
  7. money     profit for the example user (S$500, max loss S$50) on validation, then test
A single stock's files go to data/, models/ and results/scenarios/<STOCK>/, so runs never mix.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

RAW = Path("data/raw/prices")
STEPS = [
    ("split", ["-m", "council.splits"]),
    ("panel", ["-c", "from council.data import train_panel; p = train_panel(rebuild=True); print(p.shape)"]),
    ("train", ["train.py"]),
    ("mediator", ["-m", "council.mediator"]),
    ("strategy", ["-m", "council.strategy"]),
    ("backtest", ["-m", "eval.backtest"]),
    ("money validation", ["-m", "eval.money", "validation"]),
    ("money test", ["-m", "eval.money", "test"]),
]


def ensure_prices(ticker: str) -> None:
    """Download the stock's prices and splits if they aren't there yet (needs internet)."""
    if (RAW / f"{ticker}.csv").exists():
        return
    import pandas as pd
    from scripts.fetch_prices import RAW as RAW_DIR, fetch
    print(f"{ticker}: downloading prices and splits from Yahoo Finance ...")
    df = fetch(ticker)
    if df.empty:
        sys.exit(f"{ticker}: no price data found on Yahoo Finance")
    df.to_csv(RAW / f"{ticker}.csv")
    splits = pd.read_csv(RAW_DIR / "splits.csv")
    s = df.loc[df["Stock Splits"] > 0, "Stock Splits"]
    new = pd.DataFrame({"ticker": ticker, "date": [d.date().isoformat() for d in s.index], "ratio": s.values})
    pd.concat([splits[splits.ticker != ticker], new], ignore_index=True).to_csv(RAW_DIR / "splits.csv", index=False)


def main(ticker: str, start_at: str | None = None) -> None:
    env = dict(os.environ)
    if ticker.upper() == "DOW":
        env.pop("VERITAS_SCENARIO", None)
        out = Path("results")
    else:
        ensure_prices(ticker)
        env["VERITAS_SCENARIO"] = ticker
        out = Path("results/scenarios") / ticker.strip("^")
    env["PYTHONPATH"] = str(Path.cwd())
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (out / "run.log").open("w", encoding="utf-8") as log:
        names = [n for n, _ in STEPS]
        steps = STEPS[names.index(start_at):] if start_at else STEPS
        for name, step in steps:
            log.write(f"\n===== {name}: python {' '.join(step)}\n")
            log.flush()
            r = subprocess.run([sys.executable, *step], env=env, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                sys.exit(f"{ticker}: step '{name}' failed, see {out / 'run.log'}")
            print(f"{ticker}: {name} done ({time.time() - t0:.0f}s)")
    print(f"{ticker}: finished in {time.time() - t0:.0f}s, results in {out}")


if __name__ == "__main__":
    # optional: --from <step> re-runs from that step on, e.g. --from strategy
    start = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None
    main(sys.argv[1], start)
