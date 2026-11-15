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

**A real example** (a website run using data up to 13 Nov 2026):
- **Input:** buy the Dow, S$500, lose at most S$50, hold for 20 trading days, "I think it'll go up".
- **Output:** **buy S$500 with a stop-loss 9.9% below the price.**
  - All 4 of the user's rules pass.
  - 9 analysts supported the idea and 2 challenged it. The Devil's Advocate gave only a 38% chance of a rise.
  - The risk model's 1-in-20 bad case was a 9% fall, a loss of S$47, inside the S$50 limit.
  - For Apple, the same request gets a tighter stop (8.2%), because Apple's overnight jumps are bigger.

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
- Yahoo Finance daily prices for the Dow, 49 current and past members, AMD and Mosaic (1992–2026)
- FRED macro data
- Wikipedia's history of Dow members: 60 membership spells, checked to always total 30
- about 85,000 world-news headlines (Kaggle 2008–2016, plus our own rebuild for 2021–2026 from the Arctic Shift
  Reddit archive)
- 3 million company headlines (Kaggle)

**Cleaning:**
- Prices are adjusted for splits and dividends only up to the training cutoff.
- CPI and unemployment are shifted to the date they were published, not the month they describe.
- Each date uses the companies in the Dow on that date.
- A 20-trading-day gap separates training from validation, so labels can't overlap.

**Leak-proof split:** we trained on 1992–2019 only, using 2016–2019 to tune settings. **2020–2026 is the test
set:**
- the Dow itself: 342 decisions, one every 5 trading days
- the companies in the Dow at the time: 10,051 decisions

The test set sits in a separate folder that the training code physically can't open, and 75 automated tests
check this and other leaks.

**Models:** every prediction model was trained from scratch, with no pre-trained models. Settings were chosen
with robust rules over 16 walk-forward training years, so one lucky year can't decide them.

## Results
These are test years, averaged over 20 fair decision calendars, for a S$500 budget and S$50 limit. For the main
council (the Dow and its companies):

| Strategy | S$500 became | Worst drop | Purchases that lost over S$50 |
|---|---|---|---|
| Always buy (baseline) | S$890 (+78%) | −25% | 673 |
| Random, same buy rate (baseline) | S$890 (+78%) | −25% | 673 |
| Always buy + stop-loss (baseline) | S$915 (+83%) | −20% | 339 |
| **Market Council** | **S$915 (+83%)** | **−20%** | **184** |

The random baseline matches "always buy" because the tuned council buys at every decision. It changes how much
it buys and where it puts the stop, not whether it buys.

- **Safety:** across all 13 councils, ours had the **fewest purchases breaking the S$50 limit in 12 of 13**.
- **Direction (test years, the Dow on its own, 342 decisions):**

  | Predictor | Accuracy |
  |---|---|
  | "Always up" (baseline) | 65.2% |
  | "Same as the last 20 days" (baseline) | 51.5% |
  | Best analyst (Macro) | 63.7% |
  | Pattern Oracle | 62.6% |
  | ML Forecaster | 59.9% |

  The other analysts only learned to say "up", scoring exactly 65.2%. No model beat "always up" on direction,
  on the test years or on the validation years (62.3%). So the system honestly switches its direction check off,
  and its useful AI is in **judging risk**.
- **Where it fails:**
  - **Big rallies:** AMD rose about 10× in 2020–2026. The council's stop-loss sold during dips that recovered, so
    S$500 became S$1,743 against S$3,551 for buy-and-hold.
  - **Calendar luck:** for Mosaic, the result ranged from S$124 to S$989 depending only on which days were decision
    days.
  - **No fees or slippage** are included in the backtest.
- **Honesty notes:**
  - We looked at test-year results during development, and some design ideas came after seeing them. Every
    setting was still chosen on training years only.
  - AMD, Johnson & Johnson and Mosaic were picked after looking at their full histories, so treat them as stress
    tests. The 8 extra companies, run with settings frozen beforehand, are the freshest check.

## What we'd do next
1. Test it on more years and markets, including a long falling market, and add trading costs.
2. Predict each trade's overnight-gap risk with a trained model, to place stops more precisely.

## Sources
**Data:**
- Yahoo Finance via yfinance: https://finance.yahoo.com, https://pypi.org/project/yfinance/
- FRED (Federal Reserve Bank of St. Louis): https://fred.stlouisfed.org
- Wikipedia, Historical components of the DJIA:
  https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average
- Kaggle, Daily News for Stock Market Prediction (aaron7sun): https://www.kaggle.com/datasets/aaron7sun/stocknews
- Kaggle, Massive Stock News Analysis DB (miguelaenlle):
  https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
- Arctic Shift Reddit archive (post titles, scores and dates only, no usernames): https://arctic-shift.photon-reddit.com

**Pretrained models:** none in any prediction or decision.

**Libraries:** pandas, NumPy, scikit-learn, LightGBM, PyTorch, statsmodels, arch, hmmlearn, SciPy, matplotlib,
FastAPI, Gradio, Jinja2, alpaca-py (paper trading only), uv.

**AI tools:**
- **Claude Code (Claude Opus 5.5, Anthropic):** coding assistant. It helped plan, write and test the code, and
  draft this write-up. https://claude.com/claude-code
- **Claude Opus 5.5 via the Anthropic API:** turns finished results into the plain-English report. It never
  predicts or decides. https://www.anthropic.com/api

**Tutorials and references:** Veritas AI Hackathon 2026 Project Briefs (§4.2.4). https://highschoolhackathon.org

Full list: [SOURCES.md](SOURCES.md).
