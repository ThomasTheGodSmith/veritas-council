# Market Council: results

> This project is a prototype and is not financial advice.

Veritas AI International High School Hackathon 2026, AI + Finance track.

## How to read this

- **The question:** can a council of AI models we trained ourselves help a first-time investor make *safer* decisions than simple rules?
- **The example user:** S$500 to invest, a maximum loss of S$50, holding each purchase for 20 trading days.
- **Training data:** 1992–2019 only. The last 4 of those years (2016–2019) are a *validation* slice used to tune settings. **Test years: 2020–2026.** No model or setting ever saw them; the training code physically can't open them, and automated tests check this.
- **A decision every 5 trading days** (10,374 decisions per calendar for the main project, 342 for one stock). Each purchase is bought at the next day's open.
- **"S$500 became"** puts S$500 to work once every 20 trading days, compounding.
- **Averaged over 20 calendars.** Results depend a lot on which days happen to be decision days, so every number is the average over all 20 fair ways to place them (5 decision offsets × 4 compounding offsets), with the range. Retraining with the same seeds gives identical models, so repeating training would add nothing.
- **Stop-losses are realistic:** if a price opens below the stop after overnight news, it sells at that lower open.
- **The strategies:**
  - *Always buy*: S$500 every time.
  - *Random*: buys at random as often as the council does.
  - *Always buy + stop-loss*: S$500 with a stop at −S$50.
  - *Council*: our system, with settings tuned only on training years.

## Headline

- **Safety:** across 13 separate programs, the council had the fewest purchases losing more than the S$50 limit in 12 of 13.
- **Profit:** in the 2020–2026 rising market it usually earned less than simply buying and holding. It came out ahead only where its stop-loss avoided big falls.
- **Direction:** none of our 11 trained models predicted direction better than "always up" on unseen years. Tuned honestly, the system switches its direction check off. Its useful AI is in judging risk, not in predicting the market.
- **This is an honest, modest result.** The brief says judges value this more than an unexplained high score.

## Main project: the Dow + its companies

Settings chosen on training years: full budget with a gap-aware stop; direction check off.

| Strategy | S$500 became (average) | Profit | Range over 20 calendars | Worst drop from peak | Purchases that lost >S$50 | Worst single loss |
|---|---|---|---|---|---|---|
| Always buy | S$890 | S$390 (+78%) | S$804 to S$998 | -25% | 673.0 | S$341 |
| Random, same buy rate | S$890 | S$390 (+78%) | S$804 to S$998 | -25% | 673.0 | S$341 |
| Always buy + stop-loss | S$915 | S$415 (+83%) | S$790 to S$1,175 | -20% | 338.6 | S$147 |
| **Council** | S$915 | S$415 (+83%) | S$792 to S$1,175 | -20% | 184.2 | S$147 |

Validation accuracy of each analyst (2016–2019): momentum 62.3%, mean reversion 62.3%, ml forecaster 60.6%, risk officer 62.3%, news reaction 62.3%, pattern oracle 60.4%, raw reader 62.3%, market regime 62.3%, macro 62.3%, dow breadth 62.3%. Always-up scores 62.3%.

## AMD (volatile stock)

Settings chosen on training years: full budget with a gap-aware stop; direction check off.

| Strategy | S$500 became (average) | Profit | Range over 20 calendars | Worst drop from peak | Purchases that lost >S$50 | Worst single loss |
|---|---|---|---|---|---|---|
| Always buy | S$4,793 | S$4,293 (+859%) | S$3,430 to S$7,468 | -62% | 62.4 | S$160 |
| Random, same buy rate | S$4,793 | S$4,293 (+859%) | S$3,430 to S$7,468 | -62% | 62.4 | S$160 |
| Always buy + stop-loss | S$2,994 | S$2,494 (+499%) | S$869 to S$5,585 | -59% | 26.0 | S$80 |
| **Council** | S$2,708 | S$2,208 (+442%) | S$551 to S$5,961 | -54% | 5.2 | S$66 |

Validation accuracy of each analyst (2016–2019): momentum 52.9%, mean reversion 54.0%, ml forecaster 45.0%, risk officer 52.6%, news reaction 37.8%, pattern oracle 37.3%, raw reader 41.2%, market regime 37.3%, macro 37.7%. Always-up scores 62.7%.

## Johnson & Johnson (steady stock)

Settings chosen on training years: full budget with a gap-aware stop; direction check off.

| Strategy | S$500 became (average) | Profit | Range over 20 calendars | Worst drop from peak | Purchases that lost >S$50 | Worst single loss |
|---|---|---|---|---|---|---|
| Always buy | S$1,108 | S$608 (+122%) | S$985 to S$1,360 | -17% | 1.8 | S$92 |
| Random, same buy rate | S$1,108 | S$608 (+122%) | S$985 to S$1,360 | -17% | 1.8 | S$92 |
| Always buy + stop-loss | S$935 | S$435 (+87%) | S$695 to S$1,214 | -20% | 5.2 | S$64 |
| **Council** | S$949 | S$449 (+90%) | S$687 to S$1,254 | -19% | 2.2 | S$56 |

Validation accuracy of each analyst (2016–2019): momentum 62.0%, mean reversion 62.0%, ml forecaster 55.1%, risk officer 62.0%, news reaction 62.0%, pattern oracle 61.2%, raw reader 62.0%, market regime 62.0%, macro 62.0%. Always-up scores 62.0%.

## Mosaic (declining stock)

Settings chosen on training years: full budget with a gap-aware stop; direction check off.

| Strategy | S$500 became (average) | Profit | Range over 20 calendars | Worst drop from peak | Purchases that lost >S$50 | Worst single loss |
|---|---|---|---|---|---|---|
| Always buy | S$542 | S$42 (+8%) | S$391 to S$793 | -70% | 60.8 | S$266 |
| Random, same buy rate | S$542 | S$42 (+8%) | S$391 to S$793 | -70% | 60.8 | S$266 |
| Always buy + stop-loss | S$436 | S$-64 (-13%) | S$179 to S$844 | -73% | 20.6 | S$72 |
| **Council** | S$435 | S$-65 (-13%) | S$124 to S$989 | -71% | 8.6 | S$72 |

Validation accuracy of each analyst (2016–2019): momentum 49.6%, mean reversion 51.7%, ml forecaster 49.9%, risk officer 56.6%, news reaction 50.7%, pattern oracle 45.1%, raw reader 55.0%, market regime 48.6%, macro 55.0%. Always-up scores 55.0%.

## The Dow index on its own

Settings chosen on training years: full budget with a gap-aware stop; direction check off.

| Strategy | S$500 became (average) | Profit | Range over 20 calendars | Worst drop from peak | Purchases that lost >S$50 | Worst single loss |
|---|---|---|---|---|---|---|
| Always buy | S$890 | S$390 (+78%) | S$804 to S$998 | -25% | 5.4 | S$157 |
| Random, same buy rate | S$890 | S$390 (+78%) | S$804 to S$998 | -25% | 5.4 | S$157 |
| Always buy + stop-loss | S$915 | S$415 (+83%) | S$790 to S$1,175 | -20% | 2.2 | S$57 |
| **Council** | S$915 | S$415 (+83%) | S$792 to S$1,175 | -20% | 2.0 | S$57 |

Validation accuracy of each analyst (2016–2019): momentum 68.8%, mean reversion 68.8%, ml forecaster 59.4%, risk officer 68.8%, news reaction 68.8%, pattern oracle 68.3%, raw reader 68.8%, market regime 68.8%, macro 68.8%. Always-up scores 68.8%.

## The same program on 8 more Dow companies

These were run with the general program (`scripts/run_stock.py`) and settings frozen in advance; nothing was adjusted after seeing them.

| Stock | Always buy | Always buy + stop-loss | **Council** | Purchases lost >S$50: always buy / stop-loss / council |
|---|---|---|---|---|
| Coca-Cola | S$979 (+96%), -25% | S$1,028 (+106%), -22% | S$1,043 (+109%), -21% | 6.6 / 2.0 / 0.6 |
| Caterpillar | S$2,432 (+386%), -35% | S$2,130 (+326%), -34% | S$2,268 (+354%), -33% | 30.2 / 18.0 / 9.0 |
| Microsoft | S$1,660 (+232%), -32% | S$1,359 (+172%), -41% | S$1,272 (+154%), -43% | 17.2 / 9.0 / 4.0 |
| Goldman Sachs | S$1,737 (+247%), -28% | S$1,429 (+186%), -32% | S$1,319 (+164%), -34% | 16.2 / 9.8 / 4.8 |
| Home Depot | S$872 (+74%), -33% | S$798 (+60%), -36% | S$834 (+67%), -36% | 24.8 / 7.4 / 2.8 |
| Merck | S$1,029 (+106%), -40% | S$948 (+90%), -39% | S$946 (+89%), -39% | 13.0 / 5.6 / 2.2 |
| Boeing | S$261 (-48%), -64% | S$291 (-42%), -61% | S$295 (-41%), -60% | 51.8 / 21.4 / 11.4 |
| IBM | S$1,271 (+154%), -31% | S$1,315 (+163%), -29% | S$1,262 (+152%), -30% | 23.6 / 13.4 / 9.6 |

Each cell shows the average of what S$500 became (profit %), then the worst drop from a peak.

## Where it fails

- **Big rallies:** on AMD, which rose about 10× in 2020–2026, the council's stop-loss sold during dips that recovered, so it kept far less of the rise than buy-and-hold.
- **Calendar luck:** for Mosaic, the council's S$500 ended anywhere from about S$124 to S$989 depending only on which days were decision days. Single backtests can mislead.
- **Gap risk:** some losses still exceed S$50 when a stock opens far below the stop after bad overnight news.
- **Direction:** every direction model, including a small transformer and a CNN trained from scratch, learned little beyond "markets usually rise".

## Honesty notes

- **Every setting was chosen on training years only:** walk-forward years for sizing, 2016–2019 for the direction check.
- **The test years were looked at several times during development.** Some design ideas (sizing instead of skipping, realistic stops, robust tuning) came after seeing earlier test results. The 8 extra companies are the freshest check.
- **AMD, Johnson & Johnson and Mosaic were picked by looking at their full histories,** so they are stress tests, not a random sample.
- **Not included:** trading fees and slippage. The app uses a fixed exchange rate of S$1.30 per US$.
