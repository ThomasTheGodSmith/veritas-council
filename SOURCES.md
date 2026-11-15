# Sources and credits

Every dataset, library, model and AI tool used, as the Veritas AI brief requires (§2 rule 4, §5).
Keep this updated; it feeds the write-up's Sources section and the submission form.

> This project is a prototype and is not financial advice.

## Datasets
- **Yahoo Finance** daily prices, dividends and share splits for the Dow Jones Industrial Average (^DJI), the
  SPDR Dow ETF (DIA), 49 current and past Dow members, AMD and Mosaic, and VIX, the 10-year yield, the US
  dollar index and oil futures. Downloaded with [yfinance](https://pypi.org/project/yfinance/)
  (`scripts/fetch_prices.py`). https://finance.yahoo.com
- **FRED**, Federal Reserve Bank of St. Louis: Fed funds rate, 10-year and 2-year Treasury yields, the 10y−2y and
  10y−3m spreads, VIX, WTI oil, the broad dollar index, CPI and unemployment (`scripts/fetch_macro.py`).
  https://fred.stlouisfed.org
- **Wikipedia**, "Historical components of the Dow Jones Industrial Average": point-in-time Dow membership,
  hand-mapped to tickers (`scripts/build_membership.py`).
  https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average
- **Kaggle, "Daily News for Stock Market Prediction"** (aaron7sun): top 25 r/worldnews headlines per day,
  Aug 2008 – Jul 2016. https://www.kaggle.com/datasets/aaron7sun/stocknews
- **Arctic Shift Reddit archive** (API by Arthur Heitmann): r/worldnews post titles and scores, rebuilt the
  same way for 2016–2026. Titles, scores and timestamps only, no usernames.
  https://arctic-shift.photon-reddit.com
- **Kaggle, "Massive Stock News Analysis DB for NLP Backtests"** (miguelaenlle): ticker-tagged company
  headlines, 2009–2020. https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

## Pretrained models
None in any prediction or decision: every analyst, the Devil's Advocate, the Mediator, the Judge and the
Drawdown Risk model are trained from scratch on the data above.

## Libraries
pandas, NumPy, PyArrow, PyYAML, Pydantic, scikit-learn, LightGBM, statsmodels, arch (GARCH), hmmlearn
(hidden Markov model), PyTorch (Pattern Oracle transformer, Raw Reader CNN), SciPy, matplotlib, Jinja2,
Gradio (app), requests, lxml, python-dotenv, alpaca-py (Alpaca **paper** trading API), anthropic (report
writing), pytest, and [uv](https://docs.astral.sh/uv/) for the Python environment.

## AI tools
- **Claude Code** (Claude Opus 5.5, Anthropic): coding assistant. It helped plan the project, write the code,
  tests and documentation, and run the experiments. Several Claude Code sessions worked in parallel on
  separate parts. Every part must be reviewed and explained by the team.
- **Claude Opus 5.5** (Anthropic API, `council/narrator.py`): turns the finished council results into a
  plain-English report. It never sees market data or news and never makes or changes a decision; code checks
  every number it writes against the models' output.

## Tutorials and references
- Veritas AI International High School Hackathon 2026, Project Briefs (§4.2 AI + Finance, §4.2.4 News
  headlines and the stock market). https://highschoolhackathon.org
