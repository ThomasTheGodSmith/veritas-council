# Market Council: rules for every Codex agent

Read PLAN.md before starting any task. This is a hackathon prototype due **Mon Nov 16, 12:59 PM Singapore time** (Sun Nov 15, 11:59 PM ET). Prefer working and simple over clever.

## Non-negotiable
- **Every prediction model is trained from scratch on the datasets provided.** No pre-trained weights in any analyst, the Mediator, the Devil's Advocate or the Judge: no LLM calls, no FinBERT, no Chronos, no pre-trained embeddings. The product's decisions must only ever know the data we gave it.
- **No look-ahead leakage.** Any function that takes a `decision_date` must only use data with `date < decision_date`. Splits are by date only, with **one cutoff for every dataset** set by the Dow index's years: the **newest 20% of years (rounded up) is test, the rest is training** (currently train 1992–2019, validation slice 2016–2019, test 2020–2026). `council/splits.py` computes it from the real date range and writes **separate folders**: training code reads only `data/splits/train/` via `load_train()`, which contains no test years and prices adjusted only as of the cutoff. Only `eval/backtest.py` may call `load_test(..., for_evaluation=True)`; `tests/test_splits.py` fails if anything else does. Use `fit_mask()` for the 20-trading-day embargo before the validation slice. Never hard-code years. Never use a random split.
- **Fit everything on training years only:** model weights, scalers, TF-IDF vocabulary, token bins, HMM, GARCH. Tune the Mediator, Devil's Advocate and Judge threshold on the validation slice only, never on test years.
- **Datasets are chosen and downloaded** (15 Nov). `config/data.yaml` lists them and `data/README.md` has coverage, gaps and checks. Read data only through one loader per dataset in `council/loaders/`; never hard-code a data source inside analysts. Shift CPI and unemployment to their release dates (`release_lag` in `config/data.yaml`) before any model sees them.
- **Dow membership is point-in-time.** Component-level code must use `dow_membership.csv` as of the decision date, never today's 30 members. Price weights use split-corrected **raw** prices (PLAN.md §0b).
- **Hard user requirements are checked in code** (`council/rules.py`). The Judge cannot override a failed rule.
- **Paper trading only.** The ✅ Approve & Invest button auto-places orders through `paper_broker.py`, which talks only to the Alpaca *paper* endpoint (`https://paper-api.alpaca.markets`) or the local simulator. Never add a live-trading endpoint. API keys live in `.env`, never in code or git.
- **Equations come from PLAN.md §1b.** Don't swap a method for a different one without noting it in PLAN.md.
- **11 voting analysts + the Fundamentals info card** (PLAN.md §1). Fundamentals has no vote and never runs in backtests.
- **Codex Opus 5.5 writes the readable report** (`council/narrator.py`, PLAN.md §1c). It is the only place Codex is called. It receives only the finished outputs (never raw market data or news), runs after the Judge decides, and every number it writes must match the input or that section falls back to the template. Never call it during evaluation, and the app must still work without an API key.
- **Each AI's folder shows that AI's own output**, untouched by Codex. `decision.json` and `reasons.md` come straight from the model; `reasons.md` uses templates.
- Every UI page and report starts with: "This project is a prototype and is not financial advice."
- Report results honestly, including results worse than the baselines.

## Conventions
- Python 3.11 via `uv` (`uv sync`, `uv run pytest`, `uv run python train.py`); type hints, Pydantic models for every analyst's output.
- Keep models small enough to train on a laptop CPU (the transformer and CNN under ~30 minutes each). Set random seeds so results are reproducible.
- `train.py` rebuilds every model from the data; trained files go in `models/` (gitignored).
- Stay inside your own module/branch. Don't edit other panes' files; leave a TODO and tell the integrator instead.
- Run `pytest` before saying you're done.

## Hackathon rules (Veritas AI brief, Drive file `15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-`)
- Teams are 1–3 high school students and it's their project. AI coding assistants are allowed, but the team must understand every part and explain it live to AI researchers at the 21 Nov finals. Explain choices as you go and keep code simple enough to defend.
- Cite everything (datasets, libraries, models, tutorials, AI tools including Codex) in the write-up's Sources section.
- All work must happen inside the window (14 Nov 00:00 ET to 15 Nov 23:59 ET). Report honest results, including weak ones.
- Submit through the Airtable form: code (GitHub, viewable by anyone with the link, notebook outputs saved), a one-page write-up from the brief's template, team details and credits. A 2-minute video is optional.
