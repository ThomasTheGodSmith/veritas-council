# Market Council

> This project is a prototype and is not financial advice.

A council of AI models, each **trained from scratch on the datasets provided**, that stress-tests a planned
investment in the Dow Jones Industrial Average (or one of its 30 companies) under the user's own rules.
Claude Opus 5.5 writes the readable report from the finished results; every prediction and decision comes
from our own models.

- Full build plan: [PLAN.md](PLAN.md)
- Rules for Claude Code agents: [CLAUDE.md](CLAUDE.md)

## Folder structure
```
config/        settings: datasets, split, analysts
data/          cached data (gitignored)
models/        trained model files (gitignored, rebuilt by train.py)
council/       all the AI and decision code
  loaders/     one loader per dataset provided
  analysts/    the 11 voting analysts + Fundamentals info card
  templates/   report HTML template
eval/          evaluation against baselines (results.ipynb)
runs/          one folder per council run (each AI's decision + reasons)
tests/         leakage and rule tests
train.py       trains every model on the training years
app.py         Gradio app
```

## Quick start
```bash
uv sync
cp .env.example .env      # add keys (optional)
uv run python train.py
uv run python web/server.py         # the website: http://127.0.0.1:8000
```

The website lets you pick which trained council to ask: **Main** (the Dow and its companies), **AMD**,
**Johnson & Johnson**, **Mosaic** or **Dow alone**. The older Gradio app (`uv run python app.py`) still works.

## Results
See **[RESULTS.md](RESULTS.md)**: every council's results on the unseen test years 2020-2026, averaged over 20
decision calendars. Rebuild it with `uv run python scripts/write_results.py`.

## Run the whole project for any stock
```bash
uv run python scripts/run_stock.py AMD      # downloads prices if needed, trains, tunes, tests
uv run python scripts/run_stock.py DOW      # the main Dow project
```

## Write-up
Use the template in the Veritas brief §5. Include the line: "This project is a prototype and is not financial advice."
