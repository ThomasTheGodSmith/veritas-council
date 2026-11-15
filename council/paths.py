"""Where every program reads and writes. One place, so a whole run can be pointed at one stock.

Normally the project is about the Dow Jones. Setting the environment variable
    VERITAS_SCENARIO=AMD
runs the SAME programs (splits, train.py, mediator, backtest, money) as if the whole project
had been built for that one stock instead of the Dow, with everything kept in its own folders:
    data/scenarios/AMD/   models/scenarios/AMD/   results/scenarios/AMD/
The stock takes the Dow's place (see scenario_inputs), so every model trains on it exactly the
way it trained on the Dow. The Dow project's own files are never touched.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

SCENARIO = os.environ.get("VERITAS_SCENARIO") or None
INDEX = "^DJI"

if SCENARIO:
    _name = SCENARIO.strip("^")
    DATA = Path("data/scenarios") / _name
    MODELS = Path("models/scenarios") / _name
    RESULTS = Path("results/scenarios") / _name
else:
    DATA, MODELS, RESULTS = Path("data"), Path("models"), Path("results")

SPLITS = DATA / "splits"
CACHE = DATA / "cache"


def scenario_inputs(prices: pd.DataFrame, membership: pd.DataFrame,
                    company_news: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """For a single-stock run, put the stock in the Dow's place.

    - prices: the stock's rows take the index's name and the real Dow index is removed, so
      every model (Momentum, Market Regime, Macro, Risk Officer, ...) treats it exactly as it
      treated the Dow. Cross-asset series (VIX, 10-year yield, dollar, oil) stay, as they did
      for the Dow.
    - membership: a single stock has no member companies, so this is empty. Breadth features
      and the Dow Breadth analyst therefore have nothing to work with.
    - company news: the stock's own headlines, tagged with the index's name, so News-Reaction
      reads them the way it read the Dow's news.
    """
    if not SCENARIO:
        return prices, membership, company_news
    own = prices[prices.ticker == SCENARIO].assign(ticker=INDEX)
    prices = pd.concat([prices[~prices.ticker.isin([INDEX, SCENARIO])], own], ignore_index=True)
    prices = prices[~prices.ticker.isin(membership.ticker.dropna())]     # no member companies
    membership = membership.iloc[0:0]
    company_news = company_news[company_news.ticker == SCENARIO].assign(ticker=INDEX)
    return prices, membership, company_news
