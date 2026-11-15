# Data card

Everything in `data/raw/` is downloaded or rebuilt by the scripts in `scripts/` (gitignored, about 770 MB).
Fetched 15 Nov 2026; prices run to the close of **Fri 13 Nov 2026**.

```bash
uv run python scripts/build_membership.py   # who was in the Dow, and when
uv run python scripts/fetch_prices.py       # prices, dividends, splits (needs membership first)
uv run python scripts/fetch_macro.py        # FRED macro series
uv run python scripts/sync_news.py          # news, copied from D:/code/veritas_hackathon (re-run to pick up backfill)
```

## What we have

| Dataset | File(s) | Coverage | Source |
|---|---|---|---|
| Dow index | `prices/^DJI.csv` | 1992-01-02 → 2026-11-13, 8,780 days | Yahoo Finance via yfinance |
| Dow fund (paper orders) | `prices/DIA.csv` | 1998-01-20 → 2026-11-13 | Yahoo Finance |
| Member prices | `prices/<ticker>.csv` (49 tickers) | from 1990 or listing → 2026-11-13 | Yahoo Finance |
| Splits | `splits.csv` | all splits in the member files | Yahoo Finance |
| Dow membership | `membership/dow_membership.csv` | 60 membership spells, 1991-05-06 → today | Wikipedia, hand-mapped |
| Market news | `news/stocknews/Combined_News_DJIA.csv` | 2008-08-08 → 2016-07-01 | Kaggle (aaron7sun) |
| Market news | `news/reddit_days/*.json` | 2021-04-22 → 2026-11-13 (backfill still running) | Arctic Shift Reddit archive |
| Company news | `news/company/*.csv` | 2009-02-14 → 2020-06-11 | Kaggle (miguelaenlle) |
| Macro | `macro/<series>.csv` | fed funds, 10y, 2y, 10y−2y, 10y−3m, VIX, WTI oil, CPI, unemployment from before 1992; broad dollar from 2006 | FRED |
| Cross-asset | `prices/^VIX.csv`, `^TNX`, `DX-Y.NYB`, `CL=F` | 1990 → today (oil futures from 2000) | Yahoo Finance |

## Checks we ran

- **Membership:** the code checks that every change keeps exactly 30 members and that the final set matches
  Wikipedia's 29 Jun 2026 snapshot.
- **Real (split-corrected) prices:** `raw_close = Close × product of later split ratios`. On the 1,882 days when
  all 30 members have prices, `Σ raw_close / ^DJI` (the implied Dow divisor) is **0.168** today, and moves by a
  median of 0.0009% a day. It only jumps on real events: Apple joining (2015-03-19), Nike's 2:1 split
  (2015-12-24), Amazon joining (2024-02-26), 3M's Solventum spin-off (2024-04-01), Nvidia/Sherwin-Williams
  joining (2024-11-08) and Alphabet joining (2026-06-29). That confirms both the membership table and the
  price weights.

## Gaps and caveats (report these in the write-up)

**Members with price history, by year** (out of 30):

| Years | Members with prices | Missing |
|---|---|---|
| 1992–1997 | 20 | old AT&T, Bethlehem Steel, Kodak, old GM, Sears, Texaco, Union Carbide, Westinghouse, Woolworth, Alcoa |
| 1998–2004 | 24–26 | same, as they leave the index |
| 2005–2013 | 28–29 | old GM (to 2009), Alcoa (to 2013) |
| 2014–2017 | **30** | — |
| 2018–2024 | 29 | Walgreens (WBA, taken private 2025) |
| 2025–2026 | **30** | — |

- These are survivorship gaps: companies that went bankrupt or were bought out are missing. That means
  member-level backtests before 2014 slightly favour survivors. The council's test window (2020–2026) is
  29–30 of 30.
- **Ticker lines that aren't quite the same company:** JPM before 2000 is Chase Manhattan, C (1997–98) is
  Travelers Group, and the DD line runs through the DowDuPont merger. Spin-offs (GE, MO, 3M, DD, Honeywell)
  are not split-adjusted, which is correct for real prices but shows up as one-day drops in returns.
- **Market news has a gap from 2016-07 to 2021-04** while the Reddit backfill runs. With the shared cutoff
  (train to 2019, test 2020–2026), the news test years are covered from 2021-04. News training currently ends
  2016-07, so the 2016–2019 validation slice has no news until the backfill reaches it. Re-run
  `scripts/sync_news.py` then `python -m council.splits` to pick up new days.
- **Company news ends in June 2020**, so the News-Reaction analyst can't use company headlines in the test
  window. Inside it, it falls back to market-level headlines. Alpaca's News API would fill this but needs keys.
- **Macro release lag (leakage):** CPI and unemployment are dated by the month they describe but published
  later. Loaders must shift them to their release date: CPI +46 days, unemployment +38 days from the first of
  the month.
- FRED's high-yield spread (`BAMLH0A0HYM2`) now only goes back to 2023, so it's not used.
