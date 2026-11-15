"""Proof that training code can't see the test years (CLAUDE.md, PLAN.md §2 and §4).

These run against the real split files. Build them first with `uv run python -m council.splits`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from council import splits as S

pytestmark = pytest.mark.skipif(not (S.OUT / "split.json").exists(), reason="run `python -m council.splits` first")
DATASETS = ["prices", "macro", "market_news", "company_news", "splits"]


@pytest.fixture(scope="module")
def split() -> S.Split:
    return S.read_split()


def test_split_is_80_20_by_year(split):
    assert split.n_test_years == -(-split.n_years * 2 // 10)        # ceil(20%)
    assert split.test_start == f"{split.last_year - split.n_test_years + 1}-01-01"
    assert split.val_start < split.test_start


@pytest.mark.parametrize("name", DATASETS)
def test_train_folder_has_no_test_years(name, split):
    df = S.load_train(name)
    assert (pd.to_datetime(df["date"]) < pd.Timestamp(split.test_start)).all()


@pytest.mark.parametrize("name", DATASETS)
def test_test_folder_has_only_test_years(name, split):
    df = S.load_test(name, for_evaluation=True)
    scored = df[~df["warmup"]] if "warmup" in df else df
    assert (pd.to_datetime(scored["date"]) >= pd.Timestamp(split.test_start)).all()
    if "warmup" in df:   # warm-up rows are history before the cutoff, never scored
        assert (pd.to_datetime(df.loc[df["warmup"], "date"]) < pd.Timestamp(split.test_start)).all()


def test_train_membership_forgets_the_future(split):
    m = S.load_train("membership")
    cut = pd.Timestamp(split.test_start)
    assert (m.start_date < cut).all()
    assert (m.end_date.isna() | (m.end_date < cut)).all()
    assert not m.ticker.isin(["AMZN", "NVDA", "SHW", "GOOGL"]).any()   # joined after 2019


def test_train_prices_are_as_of_the_cutoff(split):
    p = S.load_train("prices")
    aapl = p[p.ticker == "AAPL"].sort_values("date")
    # Apple split 4:1 in Aug 2020 (a test year). In the train folder its last close must be the
    # real price from that day, not Yahoo's today-adjusted number (which is 4x smaller).
    last = aapl.iloc[-1]
    assert last.close == pytest.approx(last.raw_close, rel=1e-9)
    assert last.close > 250


def test_load_test_refuses_without_permission():
    with pytest.raises(PermissionError):
        S.load_test("prices")


def test_only_evaluation_code_opens_the_test_folder():
    root = Path(__file__).resolve().parents[1]
    allowed = {root / "council" / "splits.py"}
    offenders = []
    for path in [root / "train.py", root / "app.py", *(root / "council").rglob("*.py")]:
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"load_test|splits[/\\]test|for_evaluation", text):
            offenders.append(path.relative_to(root))
    assert not offenders, f"training/app code touches the test folder: {offenders}"


def test_fit_mask_applies_the_embargo(split):
    p = S.load_train("prices")
    days = p.loc[p.ticker == "^DJI", "date"]
    mask = S.fit_mask(days, days, split)
    fit_days = days[mask]
    val_start = pd.Timestamp(split.val_start)
    gap = ((days > fit_days.max()) & (days < val_start)).sum()
    assert gap == split.embargo_trading_days
