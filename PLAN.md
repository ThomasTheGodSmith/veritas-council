# Market Council: build plan (Veritas AI Hackathon 2026, AI + Finance track)

> This project is a prototype and is not financial advice.

**Deadline:** Mon Nov 16, **12:59 PM Singapore time** (= Sun Nov 15, 11:59 PM ET). Aim to submit a first version by **Mon 9:00 AM SGT**, then resubmit as you improve it.

---

## 0. The pitch

The finance track asks for *"an AI tool that helps a person … make a **safer or fairer** money decision, and show that it **beats a simple rule**."* Judges reward honest results and penalise anything that looks too good to be true.

> **Market Council** helps a first-time investor (for example a 17-year-old with S$500 who wants to buy the Dow, or a Dow company like Apple) stress-test a planned purchase. A council of **AI models we trained ourselves**, each using a different prediction method and each knowing **only the data we gave it**, predicts what happens next *under the investor's own rules* (max loss, time horizon, budget, sectors to avoid). The council challenges the investor when the evidence says their idea is wrong. A Judge makes the final call, which can be **"don't invest"**. The investor approves or declines, and a decline sends the plan back to the council with the investor's reason.

The question you're testing, and the one that makes the project your own:
> **Does a council of self-trained models make safer decisions than (a) any single model, (b) a simple majority vote, and (c) simple rules like buy-and-hold?**

The honest answer may be "not much." That still scores well if you measure it properly (brief §6).

### Decisions already made
| Decision | Choice | Why |
|---|---|---|
| **Where the AI comes from** | **Every analyst, the Mediator, the Devil's Advocate and the Judge is a model trained from scratch on the datasets provided.** No Claude, no pre-trained finance models (no FinBERT, no Chronos). | A model trained only on the training years **can't know what happened in the test years**, so the test results are clean. It's also the clearest way to show "AI is central." |
| Claude's role | **Claude Opus 5.5 writes the readable report** from the finished results (§1c), and **Claude Code (Opus 5.5) helps you write the code.** Claude never makes or changes a prediction or decision. | Readable explanations for the user, while every decision still comes only from our own models. Cite both uses in the write-up. |
| Investing | A green **✅ Approve & Invest** button **places the order automatically** in an **Alpaca paper-trading account**, or a built-in simulator if there are no keys. A red **❌ Decline** button asks for a reason and re-runs the council. | Real brokerages need you to be 18+, and an untested prototype risking real money is the opposite of a "safer decision". Paper orders fill at real prices, so the demo looks the same. |
| How many analysts | **11 voting analysts** + 1 Fundamentals info card (no vote), plus Historian, Checklist, Mediator and Judge | Build in priority order (§7). The council works with whichever are finished. |
| Model size | **Small models.** About 25–35 years of daily data is small for deep learning, so the neural models stay tiny and are trained on all Dow members together. | Big models trained from scratch on small data just memorise noise. |
| UI | **Gradio** (the brief recommends it), plus an HTML/PDF report | Fast to build and easy to show in the demo video |

---

## 0b. Scope: the Dow Jones Industrial Average (DJIA)

**What the user can invest in:** the Dow itself, or any of its 30 current companies. You can't buy an index directly, so "buy the Dow" means buying **DIA**, the fund that tracks it. The ticker box is a dropdown: `DJIA (DIA)` plus the 30 current members.

**How the Dow works (this matters for the analysts):** the Dow is **price-weighted**:
- `DJIA = Σ P_i / divisor`, where the divisor is about 0.16.
- Each company's **position (weight)** is `w_i = P_i / Σ P_j`. A $300 stock moves the Dow 10× more than a $30 stock, whatever the company size.
- A $1 move in any member moves the Dow by `1 / divisor` ≈ 6 points.
- The app shows a live **weights table**: company, price, weight %, and "points contributed today".

### Data: the datasets provided (final choice still open)

**The project uses whichever datasets are provided.** Which ones isn't decided yet, so the code must not assume any particular source. Every dataset is listed in `config/data.yaml` with its file path or loader, its column mapping and its date range. The date ranges below are the **intended** coverage. The real ranges are read from the datasets provided, and **the train/test split is calculated from them automatically** (§2).

| Data needed | Intended range | Dataset | Examples of datasets that could be provided |
|---|---|---|---|
| Dow index prices | 1992 → today | the dataset provided | yfinance `^DJI` |
| Tradable Dow fund (for paper orders) | 1998 → today | the dataset provided | yfinance `DIA` |
| Member stock prices | 1992 → today (or from listing) | the dataset provided | yfinance, one ticker per member |
| **Who was in the Dow, and when** | 1992 → today | the dataset provided | Wikipedia "Historical components of the DJIA". Recent changes: Nvidia and Sherwin-Williams in, Intel and Dow Inc. out (Nov 2024); Alphabet in, Verizon out (Jun 29, 2026) |
| Stock splits (for true weights) | 1992 → today | the dataset provided | yfinance `Ticker.splits` |
| **Dow-level / market news** | 2008 → today | the dataset(s) provided | Kaggle `Combined_News_DJIA.csv` (2008–2016), Alpaca News API (2015 → today) |
| **Company news (every member)** | 2008 → today | the dataset(s) provided | FNSPID on Hugging Face (to 2023), Alpaca News API (2015 → today) |
| Macro and cross-asset | 1992 → today | the dataset(s) provided | FRED (Fed funds, 10y−2y, CPI); VIX, 10-year yield, dollar index, oil |

**Rules for any dataset provided:**
- Each dataset gets a small **loader** that maps it to a standard format (`date, ticker, open, high, low, close, volume` for prices; `date, ticker or "DJIA", headline, source` for news). Swapping datasets means changing the loader or config, not the analysts.
- If a dataset doesn't cover a needed range (for example, news only to 2016), the analysts that need it **skip the missing dates** and the report shows the coverage %.
- Because news sources change over time, every news feature is **standardised within its own source**, and `source` is kept as a column.
- Cite every dataset you end up using in the write-up.

### Three traps specific to the Dow (handled in code)
1. **Survivorship bias.** Today's 30 members are the winners. Training on "Nvidia in 2010" pretends you knew it would join the Dow. **Fix:** every component-level calculation uses `dow_membership.csv` as of the decision date. Old members missing from the price dataset (for example delisted companies) are left out and the remaining weights re-normalised. The coverage % is reported.
2. **Price-weight trap.** Many price datasets (yfinance included) are split-adjusted, but the Dow weights use the **real price on that day**. Apple's 4-for-1 split in 2020 cut its Dow weight by about 75% overnight. **Fix:** `raw_price = adjusted_price × (product of all split ratios after that date)`.
3. **Index changes are events.** Joining or leaving the Dow is news, and the News-Reaction analyst treats each change as an event.

---

## 1. Architecture

```
 User: ticker + amount + REQUIREMENTS (hard rules) + their own idea ("I think it'll go up")
                │
                ▼
 ┌──────────── HISTORIAN (storage) ────────────────┐
 │ point-in-time data store + every model's past    │
 │ predictions + "find similar past situations"     │
 └──────────────┬───────────────────────────────────┘
                │ evidence on request
   ┌────────────┼──────────────────────────────────────────────┐
   ▼            ▼            ▼            ▼                    ▼
 Momentum   Mean-revert   ML (LightGBM)   Risk (GARCH, VaR, Monte Carlo)
 News-Reaction (own text model)   Pattern Oracle (own mini-transformer)   Raw Reader (own CNN)
 Market Regime (HMM)   Macro   Dow Breadth & Weights        [Fundamentals: info card, no vote]
   │  Round 1: every analyst predicts independently
   │  Round 2: cross-examination: for every disagreement, the Historian checks
   │           "when these two disagreed like this before, who was right?"
   │           + Devil's Advocate model: "how often is the majority wrong in situations like this?"
   ▼
 CHECKLIST ── rule checks (code) + positives/negatives of each analyst from its real track record
   ▼
 MEDIATOR (trained stacking model) ── combines everything, builds 1–3 plans, may add ITS OWN plan,
                                       re-checks every plan with the Checklist ◄──┐
   │                                                      fails a rule → fix or drop │
   └─────────────────────────────────────────────────────────────────────────────────┘
   ▼
 JUDGE ── trained decision rule: buy / wait / avoid + amount (never overrides a failed rule)
   ▼
 READABLE REPORT (written by Claude from the finished results) + AI FOLDERS (each AI's own
                 decision and reasons, untouched; see §1c)
   ▼
 USER: ✅ Approve & Invest → order placed automatically (paper)  |  ❌ Decline + reason → loop back
```

### Agents and what each one does

Every agent below is **trained from scratch on the training years only**. The equations are in §1b.

| # | Agent | What it is | What it tells you |
|---|---|---|---|
| 1 | **Momentum** | Logistic regression on trend features (moving averages, RSI, MACD) | "The trend is up/down, and trends like this usually continue X% of the time" |
| 2 | **Mean Reversion** | Logistic regression on how stretched the price is (z-score, Bollinger bands) | "The price is unusually high/low and tends to snap back" |
| 3 | **ML Forecaster** | LightGBM (hundreds of small decision trees) on price, volume and cross-asset features (VIX, 10-year yield, dollar, oil) | Probability plus the features that mattered most |
| 4 | **Risk Officer** | GARCH volatility model + Value-at-Risk + Monte Carlo price paths | Worst-case loss, likely price range, and the **maximum safe amount** under your loss limit |
| 5 | **News-Reaction** | **Our own text model**: learns from the training headlines which words were followed by rises or falls, then finds the most similar past headlines and what the stock actually did next | "The last 20 headlines most like today's were followed by +X% over 5 days, 14 times out of 20" |
| 6 | **Pattern Oracle** *(pure "next word" prediction)* | **Our own mini language model for prices**: daily moves become tokens, and a small transformer trained from scratch predicts the next token, like an LLM predicts the next word. It writes 200 possible futures. | "131 of 200 imagined futures ended higher" |
| 7 | **Raw Reader** *(pure pattern-spotting)* | **Our own small neural network (1D CNN)** that sees only 120 days of raw price and volume, with no indicators, and learns to guess the direction | Its gut read of the chart, as a probability |
| 8 | **Market Regime** | Hidden Markov Model trained to sort the market into bull / bear / choppy | Current mood + which analysts have been right in this mood |
| 9 | **Macro** | Logistic regression on rates, yield curve, inflation and VIX | "The economy is a headwind/tailwind" |
| 10 | **Dow Breadth & Weights** | Combines the fast models' predictions for **all 30 members** by their Dow price weight, plus breadth (how many members are rising). For a single member, also compares it with its sector peers inside the Dow. | "The 5 heaviest stocks are weakening even though the index looks fine" |
| 11 | **Devil's Advocate** | **A model trained to predict when the majority of the other analysts is wrong**, using the validation years | "In situations like this, the majority has been wrong 46% of the time, so be careful" |
| ℹ️ | **Fundamentals** *(info card, no vote)* | P/E, PEG, debt/equity, growth, shown as facts | Whether the company looks cheap or expensive. Live only, never in the vote. |
| – | **Historian** (storage) | Point-in-time store of all data and every model's past predictions + similar-situation search | Supplies evidence. Never votes, and isn't shown in the report. |
| – | **Checklist** | **Code** for your hard rules + track-record statistics for each analyst | ✅/❌ for each of your rules, and positives and negatives for every analyst (below) |
| – | **Mediator** | **Trained stacking model** (learns how much to trust each analyst, and when) + plan builder | One combined probability and 1–3 plans, possibly including its own (below) |
| – | **Judge** | Decision rule with thresholds **tuned on the validation years** + half-Kelly sizing | Final buy / wait / avoid, amount, confidence |

**Hard rules are enforced in code.** If the Checklist says a plan breaks your max-loss rule, the Judge cannot approve it. Show this in the demo: it's the "safer decision" guarantee.

**Every analyst must say whether it supports or challenges your idea.** If you said "I think it'll go up" and an analyst's probability is below 50%, its card says **"challenges your idea"** with the reason. Track how often challenging you turned out to be right.

### Checklist output
It runs after the Round 2 cross-examination and before the Mediator. It has two parts:

1. **Rule check (code, cannot be overruled):** one line per requirement, for example `✅ Budget ≤ S$500` or `❌ Max loss S$50: a 1-in-20 bad month could lose S$70`.
2. **Positives and negatives of every analyst**, calculated from its **real track record** on the validation years and its situation **right now**:

```json
{
  "agent": "momentum",
  "positives": ["Right 61% of the time in bull regimes (validation)", "Strong signal today (top 10% of its history)"],
  "negatives": ["Only 48% right when volatility is high, and volatility is high today", "Disagrees with Mean Reversion, which won 58% of past disagreements like this"],
  "evidence_strength": "strong|medium|weak",
  "fits_user_rules": true,
  "trust_score": 0.64
}
```

`evidence_strength` comes from **how many past examples** support the claim: weak under 30, medium 30–100, strong over 100. The report shows a table: **👍 Positives | 👎 Negatives | Evidence | Fits your rules?**

### Mediator output
1. **Combine:** the trained stacking model turns all analysts' probabilities, plus the regime and volatility, into one `p_council` (§1b).
2. **Build plans** from the result, for example "Buy S$350 now", "Wait", or "Don't buy".
3. **Create its own plan (optional):** it also tests plan shapes nobody proposed, using the Risk Officer's simulated price paths, and adds one if it scores better:
   - **Split entry:** S$200 now and S$150 in 10 days
   - **Smaller amount** with a tighter stop-loss
   - **Staged buys:** S$100 a week for 3 weeks
   - **Wait for a price level:** buy only if the price dips X% within 10 days

   Its own plan is marked `"source": "mediator_original"` and must beat the best standard plan **on the simulated paths**: the same or better expected return with a lower worst-case loss, or a higher return at the same risk. If it can't, it isn't proposed.
4. **Check every plan with the Checklist rule code.** A failing plan is **fixed** (for example, the amount reduced until max loss fits) and re-checked, or **dropped**. Only plans with all ✅ reach the Judge.

```json
{
  "p_council": 0.57,
  "plans": [
    {"id": "A", "source": "standard", "action": "buy", "amount": 350, "checklist": "all_pass"},
    {"id": "B", "source": "standard", "action": "wait", "checklist": "all_pass"},
    {"id": "C", "source": "mediator_original", "action": "split_buy",
     "steps": [{"amount": 200, "when": "now"}, {"amount": 150, "when": "+10 days"}],
     "why_better": "Same expected return as A on 10,000 simulated paths, worst-case loss S$38 vs S$49",
     "checklist": "all_pass"}
  ],
  "dropped": [{"plan": "buy 500 now", "failed_rule": "max_loss", "fix_tried": "reduced to 350 → became plan A"}]
}
```

### What every analyst returns
```json
{
  "agent": "momentum",
  "probability_up": 0.58,
  "horizon_days": 20,
  "expected_return_pct": 1.9,
  "recommended_action": "buy|wait|avoid",
  "user_idea_verdict": "supports|challenges",
  "key_numbers": {"sma20_vs_sma50": 0.031, "rsi": 61},
  "top_reasons": [["trend_strength", 0.42], ["rsi", 0.18]],
  "track_record": {"validation_accuracy": 0.56, "n": 812}
}
```
Validated with Pydantic. `top_reasons` comes from each model's own explanation (coefficients, LightGBM feature importance or SHAP, attention or occlusion for the neural models), so every sentence in the report traces back to a number.

---

## 1b. Prediction methods and their equations

Notation: `P_t` = price on day t, `r_t = ln(P_t / P_{t-1})` = daily log return, `h` = horizon (default 20 trading days). Every analyst ends with **p = probability the price is higher after h days**. **All parameters, scalers and token bins are fitted on the training years only.**

**1. Momentum**
- `SMA_n = (1/n) · Σ_{i=0}^{n-1} P_{t-i}`. Trend is `+1` if `SMA_20 > SMA_50`, else `−1`.
- `RSI = 100 − 100 / (1 + RS)`, where `RS = avg gain / avg loss` over 14 days.
- `MACD = EMA_12 − EMA_26`, with signal line `EMA_9(MACD)`, where `EMA_t = α·P_t + (1−α)·EMA_{t-1}` and `α = 2/(n+1)`.
- Trained model: `p = σ(w₀ + w₁·trend + w₂·(RSI−50)/50 + w₃·MACD_hist/σ_P)`, where `σ(x) = 1/(1+e^−x)`.

**2. Mean Reversion**
- `z = (P_t − SMA_20) / σ_20`. Bollinger bands are `SMA_20 ± 2σ_20`.
- Trained model: `p = σ(w₀ + w₁·z + w₂·z²)`. It learns from the data whether stretched prices snap back.

**3. ML Forecaster (gradient boosting)**
- Features `x_t` = returns over the last 1, 5, 20 and 60 days, 20-day volatility, volume change, RSI, distance from SMA_50, plus VIX, 10-year yield change, dollar-index change and oil change.
- Target `y = 1` if `P_{t+h} > P_t`.
- LightGBM: `p = σ(Σ_k f_k(x_t))`, where each `f_k` is a small decision tree trained to fix the errors of the previous ones. Trained on all members + the index (pooled) to get more examples.

> **Changed 15 Nov:** the ML Forecaster dropped the macro LEVEL features (Fed funds, CPI, unemployment, yield curve, VIX level). With them, the trees learned which era it was and scored 0.438 on validation, worse than always-buy. It now uses the price features plus 20-day changes in VIX, the 10-year yield, the dollar and oil (validation 0.606). Chosen on the validation slice only; macro levels stay in the Macro analyst.

**4. Risk Officer**
- GARCH(1,1) volatility: `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}`, fitted on the training years.
- 95% Value-at-Risk over h days: `VaR = amount · (1.645·σ_t·√h − μ·h)`. Also historical VaR (5th percentile of training-period h-day returns) and max drawdown.
- **Monte Carlo price paths:** `P_{t+1} = P_t · exp((μ − σ_t²/2) + σ_t·Z)`, with `Z ~ N(0,1)`. Simulate 10,000 paths of h days; report the 5th–95th percentile prices and the chance of hitting the stop-loss. The Mediator reuses these paths to score plans.
- Outputs `max_safe_amount = user_max_loss / VaR%`, which the Checklist enforces. Its direction vote is weak by design, and the Mediator learns how much to trust it.

**5. News-Reaction (our own text model)**
- **Market-adjusted move:** `AR_t = r_t − (α + β·r_{market,t})`, with α and β from a 250-day regression. Over the next 5 days: `CAR = Σ_{d=0}^{5} AR_{t+d}`.
- **Learned word weights:** turn each day's headlines into TF-IDF word vectors (vocabulary built from training headlines only), then train a logistic regression: `p_text = σ(w · tfidf(headlines))`. The model learns its **own** "good news" and "bad news" words from what actually happened, with no pre-trained sentiment model.
- **Similar past headlines:** cosine similarity on the same TF-IDF vectors, `sim = a·b / (|a||b|)`. Take the 20 most similar past training-period headlines; `retrieved = Σ sim_i · CAR_i / Σ sim_i`.
- Combined once (news is never counted twice): `p = σ(a + b·logit(p_text) + c·retrieved + d·news_volume_z)`.
- For the index: Dow-level news plus members' news weighted by Dow price weight. Membership changes count as events.

**6. Pattern Oracle (our own mini language model for prices)**
- **Tokeniser:** split training-period daily returns into **64 quantile bins**; each bin is a "token". The bin edges come from the training years only.
- **Model:** a small decoder-only transformer (GPT-style): about 4 layers, 4 attention heads, 128-dimensional embeddings, roughly 1M parameters. Context = the last 120 tokens.
- **Training:** next-token prediction, exactly like an LLM: minimise `−Σ log P(token_{t+1} | token_{1..t})` over all members' and the index's training-year sequences.
- **Prediction:** sample 200 futures of h tokens each, convert tokens back to returns, and `p` = share of futures ending above today's price. It also gives median and percentile forecasts.
- It knows **no indicators and no finance theory**, only the patterns it learned from our data.

> **Changed 15 Nov:** the Pattern Oracle uses 64-dimensional embeddings (217k parameters), not 128 (828k). There are only ~240k training tokens, and sampling 200 futures per row was about twice as slow at 128. It learned volatility clustering (validation next-token loss 4.047 against 4.159 for a uniform guess) but not direction.

**7. Raw Reader (our own pattern-spotting network)**
- Input: 120 days of raw price and volume, each window scaled to start at 1 (no indicators).
- Model: a small 1D convolutional network (3 conv layers → pooling → dense), trained from scratch as a classifier: `p = σ(CNN(window))`, with cross-entropy loss.
- Difference from #6: the Pattern Oracle **writes possible futures**; the Raw Reader **looks at the chart shape and answers up or down directly**.

**8. Market Regime (Hidden Markov Model)**
- Fit a 3-state Gaussian HMM on `(r_t, σ_20)` over the training years. The states usually come out as bull, bear and choppy.
- The forward algorithm gives `P(state_t | data)`, and the transition matrix `A` gives `P(state_{t+1}) = P(state_t) · A`.
- `p = Σ_s P(state_{t+1} = s) · P(up | s)`, with `P(up | s)` measured on the training years. It also tells the Mediator and Checklist which analysts did best in this regime.

**9. Macro**
- Inputs from the macro dataset provided: Fed funds rate, the 10-year minus 2-year yield spread, CPI year-on-year, and VIX.
- `p_market = σ(a + Σ b_i·macro_i)`, trained on the Dow's h-day direction.
- For a member stock: `E[r_stock] = β·E[r_DJIA]`, where `β = cov(r_stock, r_DJIA) / var(r_DJIA)`.

**10. Dow Breadth & Weights**
- Price weights on the decision date (members as of that date, raw prices): `w_i = P_i / Σ_j P_j`.
- Run the fast models (Momentum, Mean Reversion, ML) on each member to get `p_i` and `E[r_i]`.
- Bottom-up Dow forecast: `E[r_DJIA] ≈ Σ_i w_i · E[r_i]`, and `p_bottom-up = σ(Σ_i w_i · logit(p_i))`.
- Breadth: `B = (# members above their 50-day SMA) / N`; advance-decline line `AD_t = AD_{t-1} + (#up − #down)`.
- **Divergence warning:** the index makes new highs while `B` falls, meaning a narrow rally.
- Index: `p = σ(a + b·logit(p_bottom-up) + c·(B − 0.5) + d·divergence)`, trained.
- Single member: `RS = R_stock(60d) − R_sector peers in the Dow(60d)` and its rank among members, giving `p = σ(a + b·RS + c·rank + e·logit(p_bottom-up))`. It also reports the stock's Dow weight.

**11. Devil's Advocate (trained to find when the majority is wrong)**
- On the **validation years**, record the majority vote of analysts 1–10 and whether it was wrong: `y_wrong = 1` if the majority's direction was wrong.
- Train a small LightGBM: `p_wrong = σ(Σ_k f_k(z))`, where `z` = how split the vote was, regime, volatility, breadth, news volume, distance from recent highs.
- Its vote is **against the majority**, scaled by `p_wrong`. It names the conditions that drove it, for example "high volatility + narrow breadth: the majority was wrong 46% of the time here."

### Round 2 cross-examination (the debate, done with data)
For every pair of analysts that disagree today, the Historian looks up **all past days where those two disagreed in the same direction** and reports who was right more often, for example "Momentum vs Mean Reversion: Mean Reversion won 58% of 340 past disagreements." These results feed the Checklist's negatives and the Mediator. The report shows it as the "debate" between analysts.

### How the council combines them (Mediator)
- **Stacking model**, trained on the validation years: `logit(p_council) = b₀ + Σ_i b_i·logit(p_i) + c·regime + d·σ_t + e·p_wrong`.
- Coefficients are constrained to be non-negative for the analysts, with a light penalty, so it doesn't overfit the short validation slice.
- Compare it in the evaluation with a simple average and a majority vote.

> **Changed 15 Nov (chosen on validation only):** the Judge's penalty λ is 0.1, not 2. At λ ≥ 0.5 the best threshold (θ ≈ 0.69) means never buying, so the council would just sit in cash. Tuned θ = 0.55; the full grid is in `models/council_validation.json`. The Mediator's `regime` input is each row's trend (+1/−1) rather than the HMM state, so every past row has a value; the HMM's mood is still shown in the report. The Devil's Advocate is trained on the validation slice itself, so the Mediator uses its leave-one-year-out predictions.

### Decision and position size (Judge)
- **Buy** only if `p_council > θ` **and** expected return > 0 **and** every hard rule passes; otherwise **wait** or **avoid**.
- `θ` is **chosen on the validation years** to maximise a safety-first score: average return − λ·(losses beyond the user's limit). θ is never tuned on test years.
- Size with half-Kelly, capped by risk: `f = ½ · (p·b − (1−p)) / b`, where `b` = average win ÷ average loss. Then `amount = min(f · budget, max_safe_amount)`.

### What happens when you press ✅ Approve & Invest
1. The Checklist re-runs the hard rules on the latest price. If anything changed, the order is blocked and the reason is shown.
2. `paper_broker.py` sends a **bracket order** to the Alpaca paper API: a market buy for `amount`, a **stop-loss** at the user's max-loss price, and an optional take-profit. Split or staged plans schedule their later steps, and each step re-checks the rules first.
3. The fill price, order ID and full council output are saved to SQLite, and the Portfolio tab shows the position.
4. Without Alpaca keys, the local simulator fills the order at the latest price in the dataset provided.

---

## 1c. The readable report

The report is what the user reads before pressing ✅ or ❌. It's the "easy to read file" from your original idea: one page that explains what every AI thought and why.

**What it contains, top to bottom:**
1. **Disclaimer:** "This project is a prototype and is not financial advice."
2. **The verdict box:** the Judge's decision in big text ("WAIT", "BUY S$350", "DON'T BUY"), the council's probability, the confidence, and one sentence on why.
3. **Your rules:** the Checklist's ✅/❌ list.
4. **Your idea vs the council:** how many analysts supported or challenged your idea, and the strongest reason against it.
5. **One card per analyst:** its prediction, a small chart, 2–3 plain-English sentences, its track record ("right 56% of the time in testing"), and whether it supports or challenges you.
6. **The debate:** the Round 2 cross-examination results and the Devil's Advocate's warning.
7. **Positives and negatives table** from the Checklist.
8. **The plans:** the Mediator's 1–3 plans side by side, with expected return, worst-case loss and the chance of hitting your stop-loss, its own plan marked as such, and any dropped plans with the rule they broke.
9. **What would change the decision:** for example "If the price falls below $X or volatility rises above Y, the council would switch to AVOID."
10. **📁 AI folders** (see below): one folder per AI holding its own decision and reasons.
11. **Buttons:** ✅ Approve & Invest and ❌ Decline (with a reason box).

### Who writes it: Claude Opus 5.5 (`council/narrator.py`)
Sections 2–9 are **written by Claude Opus 5.5** in plain, friendly English for a first-time investor.
- **What Claude receives:** only the **finished outputs** of this run: every AI's JSON (predictions, reasons, track records), the Checklist table, the Mediator's plans and the Judge's decision. It **never sees raw market data or news**, so it can't add its own market knowledge.
- **What Claude is told:** explain these results; don't add predictions, opinions or facts that aren't in the input; keep every number exactly as given; say clearly when an AI challenges the user's idea.
- **What Claude can't do:** make or change any prediction or decision. It runs **after** the Judge has decided, so the decision is already fixed.
- **Number check (code):** after Claude writes, `narrator.py` extracts every number from its text and checks it against the input. If any number is missing from the input or changed, that section is replaced with the **template version** (a sentence filled straight from the numbers, for example `"{agent} thinks there is a {p:.0%} chance the price is higher in {h} days, mainly because {top_reason}."`) and a small note says so.
- **Fallback:** no API key, an API error or a timeout → the whole report uses the templates, so the app always works.
- **Evaluation:** the narrator only writes the explanation, so it's never called during backtests and can't affect the results.
- Model: `claude-opus-5-5`, set in `config/agents.yaml`. One call per report (about 1 per decision the user actually views).

### 📁 AI folders: each AI's own decision and reasons
Below Claude's written report is a **folders section**: one collapsible folder per AI. Click a folder to open it. Everything inside comes **directly from that AI**, not from Claude, so the user can check the written report against the source.

```
📁 AI folders (run 2026-11-15 14:02 SGT · DJIA (DIA))
 ├─ 📁 1 · Momentum              ▸ BUY · 58% up · supports your idea
 ├─ 📁 2 · Mean Reversion        ▸ WAIT · 46% up · challenges your idea
 ├─ 📁 3 · ML Forecaster
 ├─ 📁 4 · Risk Officer
 ├─ …
 ├─ 📁 11 · Devil's Advocate
 ├─ 📁 ℹ️ Fundamentals (info card)
 ├─ 📁 Round 2 · Cross-examination
 ├─ 📁 Checklist
 ├─ 📁 Mediator
 └─ 📁 Judge
```

**Inside each analyst's folder:**
- **Decision:** buy / wait / avoid, `probability_up`, expected return, horizon, and supports/challenges your idea
- **Reasons, by the AI itself:** its `top_reasons` with their weights (coefficients, feature importance or the neural models' attributions), shown as a small bar chart
- **Key numbers:** the inputs it used today (for example RSI, z-score, VaR, number of similar headlines found)
- **Chart:** its own view (trend lines, simulated price paths, Pattern Oracle's 200 futures, regime probabilities and so on)
- **Track record:** validation accuracy, accuracy in today's regime, and number of past examples
- **Evidence it requested from the Historian:** for example "similar past setups: 30, rose 18 times"
- **Raw output:** the exact JSON it produced, collapsible, for anyone who wants every number

**Inside the Checklist, Mediator and Judge folders:** the rule results, the positives/negatives table, all candidate plans including dropped ones and their failed rules, the stacking weights the Mediator gave each analyst today, the Judge's threshold and sizing calculation.

**The same folders are saved to disk** for every run, so the user can keep or share them:
```
runs/2026-11-15_1402_DIA/
  report.html  report.pdf  summary.md          ← Claude's written report
  analysts/01_momentum/decision.json  reasons.md  chart.png
  analysts/02_mean_reversion/…
  council/checklist.json  mediator.json  judge.json  crossexam.json
```
A **"Download all folders (.zip)"** button in the app zips that run's folder. `reasons.md` in each folder is generated from the AI's own output with templates, not by Claude.

The report is HTML (Jinja2 template, folders as `<details>` sections so they open and close without extra code) with a "Download PDF" button. In the PDF, all folders are printed open.

---

## 2. The evaluation the judges actually score on (don't skip this)

This is **Technical quality**: a baseline, unseen test data, and failure cases.

**Data:** the datasets provided (§0b). The intended coverage is prices from 1992 and news from 2008, to today, for the Dow and its members.

> **Superseded 15 Nov (decision by the team):** one shared cutoff for every dataset, set by the Dow index's years, instead of a separate split per dataset and an overlap window. Reason: the news model's labels are stock moves, so a later news cutoff would train it on the stock models' test years, and the Mediator needs one validation period that every analyst was blind to. Result: train 1992–2019 (validation 2016–2019), test 2020–2026 for everything. Training and test data live in separate folders (`data/splits/train`, `data/splits/test`); see `council/splits.py`. The per-dataset text and worked example below are kept for the record.

**Split rule (by date only, never random): the newest 20% of years is the test set, and the rest is training.**
- For each dataset: `n_years = last_year − first_year + 1`, `n_test = ceil(0.2 × n_years)`. The newest `n_test` years are **test**; everything before is **training**.
- `test_fraction: 0.2` lives in `config/data.yaml`, so you can change it, and the years are always recalculated from the datasets actually provided. Nothing is hard-coded.
- **Inside the training years**, the newest ~15% are held back as a **validation** slice for the Mediator, the Devil's Advocate and the Judge's threshold. This is still training data, never test data.
- **Embargo:** the labels look 20 days ahead, so the last 20 trading days before each boundary are dropped. Otherwise a training label could peek into validation or test.
- **Council-level test window:** price and news datasets cover different years, so the whole council is scored on the years that are in the test portion of **every** dataset it uses (the overlap). Each analyst is also scored on its own dataset's test years.

**Worked example with the intended ranges** (the real numbers come from the datasets provided):

| Dataset | Years | Test (newest 20%) | Training (rest) | Validation slice (inside training) |
|---|---|---|---|---|
| Prices 1992–2026 | 35 | 7 years: **2020–2026** | 1992–2019 | ≈ 2016–2019 |
| News 2008–2026 | 19 | 4 years (3.8 rounded up): **2023–2026** | 2008–2022 | ≈ 2020–2022 |
| **Whole council** | — | **2023–2026** (overlap) | — | — |

Because every model is trained from scratch, **the whole test window is clean**: no model has seen any of it.

**Decision task:** every 5 trading days in the test window, for **the Dow (DIA)** and for **each member that was in the Dow on that date**, given a fixed synthetic user ("wants to buy, max loss 10%, horizon 20 trading days"), choose buy / wait / avoid and score the next 20-day return. There's no API cost, so **every** decision is evaluated, not a sample. The headline results are for the Dow; member results are a second table.

**Expect a strong "always buy" baseline.** The Dow has risen over most 20-day windows since 1992, so always buying is right well over half the time. Beating it on accuracy is hard. The council's realistic win is **safety**: smaller drawdowns, fewer broken loss limits, and sitting out some crashes. Report both.

**Baselines (report all of them next to the council):**
1. Always buy (buy-and-hold)
2. Previous-direction rule (same direction as the last 20 days)
3. SMA-50 rule (buy only if price > SMA-50)
4. Best single analyst alone
5. **Simple majority vote of all analysts, with no Mediator, Devil's Advocate or cross-examination.** This is the key ablation.

**Ablations (your own question):** council with vs without the Devil's Advocate; stacking Mediator vs simple average; Pattern Oracle vs Raw Reader (does "writing futures" beat "reading the chart"?).

**Metrics:** directional accuracy, Brier score (are the probabilities honest?), mean return of "buy" decisions, Sharpe ratio, max drawdown, **% of decisions that broke the user's loss limit** (the safety metric), and how often challenging the user's idea was right.

**Failure analysis:** show at least one case where the council was confidently wrong, and why (for example, a sudden news shock no model could see).

---

## 3. Your own additions (brief §3 requires at least one; you have several)
1. **Your own user:** a first-time teen investor with self-set hard rules.
2. **Your own question:** does a council beat a simple vote? Is it *safer* even when it isn't more accurate? Can a model learn when the majority is wrong?
3. **Your own models:** every AI trained from scratch on our data, including a mini language model for prices.
4. **Your own data handling:** point-in-time Dow membership and true price weights.

---

## 4. Leakage: the trap that sinks finance projects

Training from scratch removes the biggest risk (a pre-trained model remembering the future), but these still matter:
- **Point-in-time Historian:** only rows dated *before* the decision date. Enforce with a hard `date < decision_date` filter in code, plus a unit test.
- **Everything fitted on training years only:** model weights, scalers, TF-IDF vocabulary, token bins, HMM, GARCH, and the Judge's threshold (validation slice).
- **Embargo** of 20 trading days at each split boundary (§2).
- **Point-in-time Dow membership and split-corrected weights** (§0b).
- **No pre-trained models** (no FinBERT, Chronos, pre-trained embeddings or LLMs) in any prediction.
- **Fundamentals datasets often only hold today's numbers**, so Fundamentals is an info card with no vote and is never used in backtests.
- If test results look far better than the baselines, assume leakage first (brief §4.2.4 and §6 both say so).

---

## 5. Repo layout
```
veritas-council/
  CLAUDE.md              # rules every Claude Code agent follows
  PLAN.md
  config/agents.yaml     # analysts list, horizons, model sizes; add analysts here
  config/data.yaml       # datasets provided (paths, column maps), test_fraction: 0.2
  data/                  # parquet cache (gitignored)
  models/                # trained model files (gitignored, rebuilt by train.py)
  council/
    loaders/             # one loader per dataset provided → standard format
    splits.py            # 80/20 year split, validation slice, embargo
    historian.py         # point-in-time store + past predictions + similar-setup search
    analysts/            # one file per analyst → returns a Signal
    crossexam.py         # Round 2: who won past disagreements
    devils_advocate.py
    mediator.py          # stacking model + plan builder (incl. its own plans)
    rules.py             # hard-requirement checks
    checklist.py         # rules + positives/negatives from track records
    judge.py             # threshold + half-Kelly
    narrator.py          # Claude Opus 5.5 writes the report + number check + template fallback
    report.py            # HTML/PDF report with collapsible AI folders (§1c); saves runs/<run_id>/
    paper_broker.py      # Alpaca paper API + local simulator
  train.py               # trains every model on the training years, in order
  eval/
    backtest.py          # test-window evaluation, baselines, ablations, metrics
    results.ipynb        # charts + tables for the write-up (save with outputs!)
  app.py                 # Gradio UI: inputs → council → report → Approve/Decline
  tests/
```

---

## 6. Running it in herdr with Claude Code (Opus 5.5)

[herdr](https://github.com/ogulcancelik/herdr) runs several Claude Code sessions side by side and alerts you when one is blocked. **Claude Code writes the code; the product itself uses only your trained models.**

1. `git init` the repo and commit `CLAUDE.md` and `PLAN.md` first.
2. Install herdr's optional Claude hook so blocked panes are flagged.
3. Give **each pane its own git worktree and branch** so agents never edit the same files.
4. Start every pane with `claude --model claude-opus-5-5` and have it plan before writing code.

### Pane layout and starting prompts

| Pane | Branch | Paste this prompt |
|---|---|---|
| **1 · Data + splits + Historian** | `feat/data` | "Read PLAN.md §0b, §2, §4 and §5. Build `council/loaders/` (one loader per dataset in `config/data.yaml`, mapping to the standard format), `council/splits.py` (newest 20% of years = test, ~15% validation slice inside training, 20-day embargo) and `council/historian.py` (point-in-time query, past predictions, similar-setup search). Include point-in-time Dow membership and split-corrected price weights. Write tests proving no row ≥ decision_date is returned and no test year is in training. Plan first." |
| **2 · Classic analysts** | `feat/classic` | "Read PLAN.md §1 and §1b. Implement analysts 1, 2, 3, 4, 8, 9 and 10 and the Fundamentals info card in `council/analysts/`, exactly following the §1b equations, trained on training years only, each returning a `Signal`. Add their training steps to `train.py`. Use a fixture until pane 1 merges." |
| **3 · Neural + news analysts** | `feat/neural` | "Read PLAN.md §1 and §1b. Implement analysts 5 (TF-IDF text model + similar-headline retrieval), 6 (from-scratch mini transformer on 64 return tokens, ~1M params, PyTorch) and 7 (1D CNN on raw 120-day windows). No pre-trained weights anywhere. Train on all members + the index, training years only. Keep training under ~30 min on CPU (Colab GPU optional). Use a fixture until pane 1 merges." |
| **4 · Council logic** | `feat/council` | "Read PLAN.md §1 and §1b. Build `crossexam.py`, `devils_advocate.py`, `mediator.py` (stacking model + plan builder that tests its own plans on the Risk Officer's simulated paths), `rules.py`, `checklist.py` (positives/negatives from track records with evidence strength) and `judge.py` (θ tuned on validation, half-Kelly), plus the decline-with-reason loop. Mock the analysts' signals." |
| **5 · UI + report + broker** | `feat/ui` | "Read PLAN.md §1, §1b and §1c. Build `app.py` (Gradio), `narrator.py` (Claude Opus 5.5 writes the report from finished results only, number-matching check, template fallback with no API key), `report.py` (HTML report exactly as §1c with collapsible per-AI folders, PDF download with folders open, runs saved to `runs/<run_id>/`, download-all-folders zip button) and `paper_broker.py` (Alpaca paper bracket orders + local simulator, scheduled steps for split plans). Green ✅ Approve & Invest auto-places the order after re-checking rules; red ❌ Decline takes a reason and re-runs the council. Add a Portfolio tab and the Dow weights table. Disclaimer on every page." |
| **6 · Evaluation** | `feat/eval` | "Read PLAN.md §2 and §4. Build `eval/backtest.py`: every 5 trading days in the council test window, all 5 baselines, the ablations and all §2 metrics, plus `results.ipynb` with tables, charts and at least one failure case. Run the cheap baselines first." |

You, in the main pane, are the **integrator**. Merge in the order 1 → 2 → 3 → 4 → 5 → 6, run `pytest` after each merge, and use `/code-review` before each merge.

**Useful Claude Code features here:**
- **Plan mode** (Shift+Tab) for every pane's first step, so you approve the design before code is written.
- **`/code-review`** before merges. The things to catch are leakage bugs and date off-by-one errors.
- A **Stop hook** that runs `pytest tests/test_no_leakage.py` so no agent finishes with leakage tests failing.

**Compute:** every model here trains on a normal laptop CPU. The transformer and CNN are the slowest (aim for under 30 minutes each); a free Colab GPU makes them faster if needed. There's no per-decision API cost.

---

## 7. Timeline (Singapore time)

| By | Milestone | Check |
|---|---|---|
| **Sun afternoon** | Repo, CLAUDE.md, herdr panes up; datasets loading through loaders; splits computed | Parquet files exist; split test passes |
| **Sun evening** | Classic analysts trained; majority-vote baseline + "always buy" scored on the test window | First results table |
| **Sun night** | Council logic + report + app running end-to-end on the Dow | One input in, report out, ✅ places a paper order |
| **Mon early morning** | Neural + news analysts trained and plugged in; full evaluation + ablations; failure case picked | Final results table |
| **Mon 9:00 AM** | Write-up, video recorded, **first submission sent** | Links open in an incognito window |
| **Mon → 12:59 PM** | Polish, then resubmit | — |

**Analyst build order (highest priority first):** Momentum → Risk Officer → ML Forecaster → Dow Breadth & Weights → News-Reaction → Pattern Oracle → Raw Reader → Mean Reversion → Market Regime → Devil's Advocate → Macro → Fundamentals info card.

**Cut list if you fall behind (cut from the top):** the last analysts in the build order → Alpaca connection (keep the local simulator; the button still auto-invests) → Mediator's own plans (keep standard plans) → PDF export (keep HTML). **Never cut** baselines, date splits, leakage tests, failure examples, or the disclaimer.

---

## 8. Submission checklist (brief §5)
- [ ] Public GitHub repo (or Colab with outputs saved), README containing the write-up template
- [ ] Write-up: problem → what we built → **what we added** → how it works → results (**baseline X vs council Y**) → where it fails → next steps → sources
- [ ] The line *"This project is a prototype and is not financial advice."*
- [ ] Credits: **every dataset you end up using** (with links), LightGBM, scikit-learn, PyTorch, statsmodels, arch, hmmlearn, Alpaca paper API, Gradio, Claude Opus 5.5 (writes the readable report from our models' results) and Claude Code (coding assistant), herdr, and any Claude-council projects you took ideas from
- [ ] A 2-minute video: problem (15s) → live demo with a decline-and-retry loop (60s) → results vs baselines (30s) → limitation (15s)
- [ ] Every teammate can explain each model. Hold a 20-minute walkthrough before submitting.
- [ ] All links checked in an incognito window; first submission sent by Mon 9:00 AM SGT

---

## 9. Necessity and accuracy review

Rule of thumb: an analyst only helps accuracy if it uses **different information or a different way of learning**. Analysts that see the same data in the same way make the same mistakes, and the vote counts them twice.

| Analyst | Verdict | Why |
|---|---|---|
| Momentum, Mean Reversion | **Keep** | Simple, explainable, and often disagree with each other, which is useful |
| ML Forecaster | **Keep** | Learns how to combine many features, including cross-asset ones |
| Risk Officer | **Keep** | Drives the safety metrics, the project's most likely real win. Absorbed the old Statistical analyst. |
| News-Reaction | **Keep** | The only analyst using text. Absorbed the old News Sentiment analyst so news isn't counted twice. |
| Pattern Oracle | **Keep, as your "next word" experiment** | Small data makes from-scratch transformers hard. Pooling all members helps, and the Mediator shrinks it if it's weak. |
| Raw Reader | **Keep, as a comparison** | Same raw data as the Pattern Oracle but a different way of learning, so the ablation is a real finding |
| Market Regime | **Keep** | Improves the Mediator and Checklist, not just the vote |
| Macro | **Keep** | Matters more for an index than for one stock |
| Dow Breadth & Weights | **Keep** | Uses the price-weighting and the 30 members, information nothing else sees. Absorbed Sector Strength. |
| Devil's Advocate | **Keep** | Now a trained model, so it adds measurable information: when the majority tends to fail |
| Fundamentals | **Info card only** | No free point-in-time history, so it can't be tested |

**What will actually move results** (in order):
1. Point-in-time membership, split-corrected weights and the embargo: without these, the test results are quietly wrong.
2. The trained Mediator: weak analysts automatically count for less.
3. Pooling all Dow members when training: much more data for every model.
4. Predicting **risk** (drawdown, volatility) rather than only direction. Volatility is much more predictable than direction, which is where the council can clearly beat "always buy".
