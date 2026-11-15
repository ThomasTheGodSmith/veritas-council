# Market Council

**Track:** Finance · **Team:** Thomas Smith (solo)

> This project is a prototype and is not financial advice.

**Code:** this repository · **Results notebook:** [eval/results.ipynb](eval/results.ipynb) · **Full results:** [RESULTS.md](RESULTS.md)

## The problem
A teenager with S$500 wants to buy "the Dow", or a company like Apple, and has decided they can't afford to lose
more than S$50. Apps and social media promise that AI can tell them what to do. I wanted to build a tool that
protects that person's money: something that stress-tests their plan, stands up to them when the evidence
disagrees, and never lets them break their own loss limit.

## What we built
**Market Council** is a website where you enter what you want to buy, how much, the most you'll lose, how long you'll
hold it and your own idea ("I think it'll rise").

A council of **11 AI analysts, each trained from scratch**, gives its view:
- trend and mean-reversion models
- a gradient-boosted forecaster
- a GARCH risk model
- a text model of world headlines
- a tiny transformer that writes possible future prices
- a CNN that reads the chart
- a market-mood model (HMM)
- macro and Dow-breadth models
- a Devil's Advocate that predicts when the majority is wrong

The analysts debate. Your rules are checked in code. A **Judge** then decides how much to buy, with a stop-loss
placed so that even a bad overnight jump stays inside your limit. You approve, which places a paper trade only, or
decline with a reason, which re-runs the council. A readable report shows every analyst's reasons.

## What we added
We started from suggested project 4.2.4 (news headlines and the stock market) and went well beyond it:
- **Our own user:** a first-time teen investor with self-set rules.
- **Our own question:** does a council of self-trained models make *safer* decisions than simple rules?
- **Our own data handling:** the Dow as it actually was on each date (so the model can't "know" today's winners),
  and real share prices for Dow weights.
- **A general program** that rebuilds the whole project for any stock, which we tested on 13 separately trained
  councils.

## How it works
**Data:**
- Yahoo Finance daily prices for the Dow, its 30 companies and past members (1992–2026)
- FRED macro data
- Wikipedia's history of Dow members
- about 85,000 world-news headlines (Kaggle 2008–2016, plus our own rebuild for 2021–2026 from the Arctic Shift
  Reddit archive)
- 3 million company headlines (Kaggle)

**Leak-proof split:** we trained on 1992–2019 only, using 2016–2019 to tune settings. **2020–2026 is the test
set.** It sits in a separate folder that the training code physically can't open, and 75 automated tests check this
and other leaks.

**Models:** every prediction model was trained from scratch, with no pre-trained models. Settings were chosen
with robust rules over 16 walk-forward training years, so one lucky year can't decide them.

## Results
These are test years, averaged over 20 fair decision calendars, for a S$500 budget and S$50 limit. For the main
council (the Dow and its companies):

| Strategy | S$500 became | Worst drop | Purchases that lost over S$50 |
|---|---|---|---|
| Always buy (baseline) | S$890 (+78%) | −25% | 673 |
| Always buy + stop-loss (baseline) | S$915 (+83%) | −20% | 339 |
| **Market Council** | **S$915 (+83%)** | **−20%** | **184** |

- **Safety:** across all 13 councils, ours had the **fewest purchases breaking the S$50 limit in 12 of 13**.
- **Direction:** no model predicted direction better than "always up" (62.3% on validation). So the system honestly
  switches its direction check off, and its useful AI is in **judging risk**.
- **Where it fails:**
  - **Big rallies:** AMD rose about 10× in 2020–2026. The council's stop-loss sold during dips that recovered, so
    S$500 became S$1,743 against S$3,551 for buy-and-hold.
  - **Calendar luck:** for Mosaic, the result ranged from S$124 to S$989 depending only on which days were decision
    days.
  - **No fees or slippage** are included in the backtest.

## What we'd do next
1. Test it on more years and markets, including a long falling market, and add trading costs.
2. Predict each trade's overnight-gap risk with a trained model, to place stops more precisely.

## Sources
**Data:**
- Yahoo Finance via yfinance
- FRED (Federal Reserve Bank of St. Louis)
- Wikipedia: Historical components of the DJIA
- Kaggle: Daily News for Stock Market Prediction (aaron7sun)
- Kaggle: Massive Stock News Analysis DB (miguelaenlle)
- Arctic Shift Reddit archive

**Libraries:** pandas, NumPy, scikit-learn, LightGBM, PyTorch, statsmodels, arch, hmmlearn, SciPy, matplotlib,
FastAPI, Gradio, Jinja2, alpaca-py (paper trading only).

**AI tools:**
- **Claude Code (Claude Opus 5.5, Anthropic):** coding assistant. It helped plan, write and test the code, and
  draft this write-up.
- **Claude Opus 5.5 via the Anthropic API:** turns finished results into the plain-English report. It never
  predicts or decides.

Full list with links: [SOURCES.md](SOURCES.md).
