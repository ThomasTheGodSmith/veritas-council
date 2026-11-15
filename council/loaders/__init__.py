"""One loader per dataset (PLAN.md §0b). Each returns a pandas DataFrame in a standard format.

Standard formats (see README.md in this folder):
  prices      date, ticker, open, high, low, close, adj_close, raw_close, volume
  membership  member, company, ticker, start_date, end_date
  splits      date, ticker, ratio
  macro       date, series, value          (date = when the value was PUBLISHED, not what it describes)
  news        date, ticker, headline, source, score

Only council/splits.py should call these. Everything else reads the split files it writes,
so training code never touches the test years (see CLAUDE.md).
"""
from council.loaders.macro import load_macro
from council.loaders.membership import load_membership
from council.loaders.news import load_company_news, load_market_news
from council.loaders.prices import load_prices, load_splits

__all__ = [
    "load_prices",
    "load_splits",
    "load_membership",
    "load_macro",
    "load_market_news",
    "load_company_news",
]
