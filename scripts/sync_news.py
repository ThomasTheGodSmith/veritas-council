"""Copy the news datasets from the earlier project (D:/code/veritas_hackathon) into data/raw/news/.

Safe to re-run: the Reddit backfill in veritas_hackathon is still filling in older days,
so run this again later to pick up the new days. Only new or changed files are copied.

Sources (all collected by the earlier project; see CREDITS there):
  stocknews/          Kaggle "Daily News for Stock Market Prediction" (aaron7sun):
                      Combined_News_DJIA.csv, RedditNews.csv, upload_DJIA_table.csv (2008-08 to 2016-07)
  reddit_days/        one JSON per trading day, top 25 r/worldnews posts by score, rebuilt from the
                      Arctic Shift archive (filling backwards from 2026-11-13 towards 2016-07)
  company/            Kaggle "Massive Stock News Analysis DB" (miguelaenlle): ticker-tagged
                      headlines 2009-02 to 2020-06 (raw_analyst_ratings.csv, raw_partner_headlines.csv)

Run:  uv run python scripts/sync_news.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path("D:/code/veritas_hackathon/data")
OUT = Path("data/raw/news")

COPIES = {
    SRC / "stocknews" / "Combined_News_DJIA.csv": OUT / "stocknews",
    SRC / "stocknews" / "RedditNews.csv": OUT / "stocknews",
    SRC / "stocknews" / "upload_DJIA_table.csv": OUT / "stocknews",
    SRC / "massive_stock_news" / "raw_analyst_ratings.csv": OUT / "company",
    SRC / "massive_stock_news" / "raw_partner_headlines.csv": OUT / "company",
}


def newer(src: Path, dst: Path) -> bool:
    return not dst.exists() or src.stat().st_size != dst.stat().st_size or src.stat().st_mtime > dst.stat().st_mtime


def main() -> None:
    for src, folder in COPIES.items():
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / src.name
        if newer(src, dst):
            shutil.copy2(src, dst)
            print(f"copied {src.name}")
    days = OUT / "reddit_days"
    days.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in (SRC / "reddit_days").glob("*.json"):
        dst = days / src.name
        if newer(src, dst):
            shutil.copy2(src, dst)
            n += 1
    names = sorted(p.stem for p in days.glob("*.json"))
    print(f"reddit_days: {n} new/updated, {len(names)} total, {names[0]} -> {names[-1]}")


if __name__ == "__main__":
    main()
