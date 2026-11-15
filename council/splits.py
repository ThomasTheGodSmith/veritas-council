"""The train / test split (PLAN.md §2, CLAUDE.md). Training code must NEVER have the test years.

How it works:
  1. One cutoff for every dataset, set by the Dow index's years (stocks decide, because every
     decision is scored on stock moves). n_test = ceil(test_fraction * n_years) newest years.
  2. `write_splits()` reads the raw data once and writes two separate folders:
        data/splits/train/<dataset>.parquet   only rows dated before the test start
        data/splits/test/<dataset>.parquet    rows dated on or after it, plus (prices, macro) a
                                              ~500-day warm-up before it flagged warmup=True, so
                                              the first test day's 60/250-day features exist
     Training code reads only the train folder (`load_train`). `load_test` refuses to run
     unless the caller says it is the evaluation.
  3. The training files are written "as of" the last training day: prices are re-adjusted only
     for splits and dividends that had happened by then, and the membership table forgets
     anything that happened later. So the training folder is exactly what you could have
     downloaded on that day.
  4. The validation slice is the newest ~validation_fraction of the training years, still in
     the train folder. `fit_mask` drops the last `embargo_trading_days` before the validation
     start, because their 20-day labels would look into validation. Labels built from the
     train folder can't look into the test years at all: the data simply isn't there.

Run:  uv run python -m council.splits
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml

from council import loaders
from council.loaders.membership import load_membership

CONFIG = Path("config/data.yaml")
from council.paths import SPLITS as OUT, scenario_inputs  # noqa: E402
TRAIN, TEST = OUT / "train", OUT / "test"
INDEX_TICKER = "^DJI"
WARMUP_DATASETS = {"prices", "macro"}
WARMUP_DAYS = 500


@dataclass(frozen=True)
class Split:
    first_year: int
    last_year: int
    n_years: int
    n_test_years: int
    n_val_years: int
    train_start: str      # first date kept at all
    val_start: str        # validation slice starts here (inside training)
    test_start: str       # test years start here; nothing on/after this is in the train folder
    last_date: str        # newest date in the data
    embargo_trading_days: int


def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def compute_split(index_dates: pd.Series, cfg: dict | None = None) -> Split:
    """Work out the split from the Dow index's real date range (nothing is hard-coded)."""
    cfg = cfg or _cfg()
    s = cfg["split"]
    dates = pd.to_datetime(index_dates)
    dates = dates[dates >= pd.Timestamp(s["start_date"])]
    first, last = dates.min().year, dates.max().year
    n_years = last - first + 1
    n_test = math.ceil(s["test_fraction"] * n_years)
    n_train = n_years - n_test
    n_val = max(1, round(s["validation_fraction"] * n_train))
    return Split(
        first_year=first,
        last_year=last,
        n_years=n_years,
        n_test_years=n_test,
        n_val_years=n_val,
        train_start=str(pd.Timestamp(s["start_date"]).date()),
        val_start=f"{last - n_test - n_val + 1}-01-01",
        test_start=f"{last - n_test + 1}-01-01",
        last_date=str(dates.max().date()),
        embargo_trading_days=int(s["embargo_trading_days"]),
    )


# ---------------------------------------------------------------- writing the split files

def _prices_as_of(train: pd.DataFrame, splits: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Re-adjust training prices as they would have looked on the last training day.

    Yahoo's close is adjusted for ALL splits up to today, so old prices quietly encode future
    splits. Here close/open/high/low are adjusted only for splits on or before the cutoff, and
    adj_close only for dividends paid on or before it. raw_close (the real traded price) is unchanged.
    """
    out = []
    for t, g in train.groupby("ticker", sort=False):
        g = g.copy()
        later = splits[(splits.ticker == t) & (splits.date >= cutoff)]["ratio"].prod()
        for c in ["open", "high", "low", "close"]:
            g[c] = g[c] * later
        g["volume"] = g["volume"] / later
        div_factor = g["adj_close"] / (g["close"] / later)      # Yahoo's dividend factor, to today
        g["adj_close"] = g["close"] * div_factor / div_factor.iloc[-1]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _membership_as_of(m: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    known = m[m.start_date < cutoff].copy()
    known.loc[known.end_date >= cutoff, "end_date"] = pd.NaT   # the future is unknown
    return known


def write_splits() -> Split:
    cfg = _cfg()
    start = pd.Timestamp(cfg["split"]["start_date"])
    prices = loaders.load_prices()
    m_all, company = load_membership(), loaders.load_company_news()
    prices, m_all, company = scenario_inputs(prices, m_all, company)   # single-stock run: stock = index
    prices = prices[prices.date >= start]
    split = compute_split(prices.loc[prices.ticker == INDEX_TICKER, "date"], cfg)
    cut = pd.Timestamp(split.test_start)
    splits = loaders.load_splits()
    from council.paths import SCENARIO
    if SCENARIO and SCENARIO != INDEX_TICKER:   # the stock's share splits follow it into the index's place
        splits = pd.concat([splits[~splits.ticker.isin([INDEX_TICKER, SCENARIO])],
                            splits[splits.ticker == SCENARIO].assign(ticker=INDEX_TICKER)], ignore_index=True)

    datasets = {
        "prices": prices,
        "macro": loaders.load_macro(),
        "market_news": loaders.load_market_news(),
        "company_news": company,
    }
    TRAIN.mkdir(parents=True, exist_ok=True)
    TEST.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        df = df[df.date >= start]
        train = df[df.date < cut]
        if name in WARMUP_DATASETS:
            # Evaluation needs ~1.5 years of history to compute the first test day's features.
            # Those rows are training-era data (not secret), flagged warmup=True, never scored.
            test = df[df.date >= cut - pd.Timedelta(days=WARMUP_DAYS)].copy()
            test["warmup"] = test.date < cut
        else:
            test = df[df.date >= cut]
        if name == "prices":
            train = _prices_as_of(train, splits, cut)
        train.to_parquet(TRAIN / f"{name}.parquet", index=False)
        test.to_parquet(TEST / f"{name}.parquet", index=False)
        scored = test[~test["warmup"]] if "warmup" in test else test
        print(f"{name:13s} train {len(train):>9,} rows to {train.date.max().date()}   "
              f"test {len(scored):>9,} rows from {scored.date.min().date()}")

    m = m_all
    _membership_as_of(m, cut).to_parquet(TRAIN / "membership.parquet", index=False)
    m.to_parquet(TEST / "membership.parquet", index=False)   # evaluation sees the full history
    splits[splits.date < cut].to_parquet(TRAIN / "splits.parquet", index=False)
    splits[splits.date >= cut].to_parquet(TEST / "splits.parquet", index=False)

    (OUT / "split.json").write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")
    return split


# ---------------------------------------------------------------- reading the split files

def read_split() -> Split:
    return Split(**json.loads((OUT / "split.json").read_text(encoding="utf-8")))


def load_train(name: str) -> pd.DataFrame:
    """The only way training code gets data: the train folder, which has no test years."""
    return pd.read_parquet(TRAIN / f"{name}.parquet")


def load_test(name: str, *, for_evaluation: bool = False) -> pd.DataFrame:
    """Test years. Only eval/backtest.py may call this, after every model is trained and saved."""
    if not for_evaluation:
        raise PermissionError("Test data is only for evaluation (eval/backtest.py), never for training.")
    return pd.read_parquet(TEST / f"{name}.parquet")


def fit_mask(dates: pd.Series, trading_days: pd.Series, split: Split) -> pd.Series:
    """True for training rows a model may FIT on: before the validation slice, minus the embargo.

    The last `embargo_trading_days` trading days before val_start are dropped, because a
    20-day-ahead label from those days would be computed from validation-period prices.
    """
    val_start = pd.Timestamp(split.val_start)
    days = pd.Series(sorted(pd.to_datetime(trading_days).unique()))
    before = days[days < val_start]
    last_fit_day = before.iloc[-split.embargo_trading_days - 1]
    return pd.to_datetime(dates) <= last_fit_day


def validation_mask(dates: pd.Series, split: Split) -> pd.Series:
    """True for rows in the validation slice (tuning the Mediator, Devil's Advocate and Judge)."""
    return pd.to_datetime(dates) >= pd.Timestamp(split.val_start)


if __name__ == "__main__":
    print(json.dumps(asdict(write_splits()), indent=2))
