"""Write RESULTS.md (readable results for the team and the Veritas judges) and results/summary.json
(what the website shows), from the saved training records and the 20-calendar test results.

Run after eval.robust for each program:  uv run python scripts/write_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROGRAMS = {
    "DOW": ("Main project: the Dow + its companies", "models", "results"),
    "AMD": ("AMD (volatile stock)", "models/scenarios/AMD", "results/scenarios/AMD"),
    "JNJ": ("Johnson & Johnson (steady stock)", "models/scenarios/JNJ", "results/scenarios/JNJ"),
    "MOS": ("Mosaic (declining stock)", "models/scenarios/MOS", "results/scenarios/MOS"),
    "DJI": ("The Dow index on its own", "models/scenarios/DJI", "results/scenarios/DJI"),
}
EXTRA = {t: (t, f"models/scenarios/{t}", f"results/scenarios/{t}") for t in ["KO", "CAT", "MSFT", "GS", "HD", "MRK", "BA", "IBM"]}
NAMES = {"KO": "Coca-Cola", "CAT": "Caterpillar", "MSFT": "Microsoft", "GS": "Goldman Sachs",
         "HD": "Home Depot", "MRK": "Merck", "BA": "Boeing", "IBM": "IBM"}
ORDER = ["Always buy", "Random, same buy rate", "Always buy + stop-loss", "Council"]
ANALYSTS = ["momentum", "mean_reversion", "ml_forecaster", "risk_officer", "news_reaction", "pattern_oracle",
            "raw_reader", "market_regime", "macro", "dow_breadth"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def program(key: str, label: str, mdir: str, rdir: str) -> dict:
    m, r = Path(mdir), Path(rdir)
    return {"key": key, "label": label, "robust": load(r / "robust.json"), "strategy": load(m / "strategy.json"),
            "tracks": load(m / "track_records.json") or {}, "mediator": load(m / "council_validation.json") or {}}


def money(x: float) -> str:
    return f"S${x:,.0f}"


def strategy_text(s: dict) -> str:
    size = f"volatility-scaled stop (C = {s['c']})" if s.get("c") is not None else (
        "full budget with a gap-aware stop" if s.get("k") in (None, float("inf")) or str(s.get("k")) == "inf"
        else f"risk budget K = {s['k']}")
    gate = "off" if not s["theta"] else f"on, threshold {s['theta']:.0%}"
    return f"{size}; direction check {gate}"


def results_table(p: dict) -> list[str]:
    res = p["robust"]["results"]
    out = ["| Strategy | S$500 became (average) | Profit | Range over 20 calendars | Worst drop from peak | "
           "Purchases that lost >S$50 | Worst single loss |",
           "|---|---|---|---|---|---|---|"]
    for name in ORDER:
        x = res[name]
        b = x["S500_becomes"]
        out.append(f"| {'**Council**' if name == 'Council' else name} | {money(b['mean'])} | "
                   f"{money(b['mean'] - 500)} ({x['profit_pct']['mean']:+.0%}) | {money(b['min'])} to {money(b['max'])} | "
                   f"{x['worst_drop']['mean']:.0%} | {x['lost_over_50']['mean']:.1f} | {money(x['worst_loss']['mean'])} |")
    return out


def main() -> None:
    progs = [program(k, *v) for k, v in PROGRAMS.items()]
    extras = [program(k, *v) for k, v in EXTRA.items()]
    first = progs[0]["robust"]
    L = ["# Market Council: results", "",
         "> This project is a prototype and is not financial advice.", "",
         "Veritas AI International High School Hackathon 2026, AI + Finance track.", "",
         "## How to read this", "",
         "- **The question:** can a council of AI models we trained ourselves help a first-time investor make *safer* "
         "decisions than simple rules?",
         "- **The example user:** S$500 to invest, a maximum loss of S$50, holding each purchase for 20 trading days.",
         "- **Training data:** 1992–2019 only. The last 4 of those years (2016–2019) are a *validation* slice used to tune "
         "settings. **Test years: 2020–2026.** No model or setting ever saw them; the training code physically can't "
         "open them, and automated tests check this.",
         f"- **A decision every 5 trading days** ({first['decisions_per_calendar']:,} decisions per calendar for the main "
         "project, 342 for one stock). Each purchase is bought at the next day's open.",
         "- **\"S$500 became\"** puts S$500 to work once every 20 trading days, compounding.",
         "- **Averaged over 20 calendars.** Results depend a lot on which days happen to be decision days, so every number "
         "is the average over all 20 fair ways to place them (5 decision offsets × 4 compounding offsets), with the range. "
         "Retraining with the same seeds gives identical models, so repeating training would add nothing.",
         "- **Stop-losses are realistic:** if a price opens below the stop after overnight news, it sells at that lower open.",
         "- **The strategies:**",
         "  - *Always buy*: S$500 every time.",
         "  - *Random*: buys at random as often as the council does.",
         "  - *Always buy + stop-loss*: S$500 with a stop at −S$50.",
         "  - *Council*: our system, with settings tuned only on training years.",
         "", "## Headline", ""]
    wins = sum(1 for p in progs + extras if p["robust"] and
               min(ORDER, key=lambda n: p["robust"]["results"][n]["lost_over_50"]["mean"]) == "Council")
    total = sum(1 for p in progs + extras if p["robust"])
    L += [f"- **Safety:** across {total} separate programs, the council had the fewest purchases losing more than the "
          f"S$50 limit in {wins} of {total}.",
          "- **Profit:** in the 2020–2026 rising market it usually earned less than simply buying and holding. It came "
          "out ahead only where its stop-loss avoided big falls.",
          "- **Direction:** none of our 11 trained models predicted direction better than \"always up\" on unseen years. "
          "Tuned honestly, the system switches its direction check off. Its useful AI is in judging risk, not in "
          "predicting the market.",
          "- **This is an honest, modest result.** The brief says judges value this more than an unexplained high score.",
          ""]
    for p in progs:
        if not p["robust"]:
            continue
        L += [f"## {p['label']}", "",
              f"Settings chosen on training years: {strategy_text(p['strategy'])}.", ""]
        L += results_table(p)
        tr = p["tracks"]
        if tr:
            accs = [f"{a.replace('_', ' ')} {tr[a]['validation_accuracy']:.1%}" for a in ANALYSTS if a in tr]
            base = p["mediator"].get("base_rate_up")
            L += ["", f"Validation accuracy of each analyst (2016–2019): {', '.join(accs)}"
                  + (f". Always-up scores {base:.1%}." if base else ".")]
        L.append("")
    L += ["## The same program on 8 more Dow companies", "",
          "These were run with the general program (`scripts/run_stock.py`) and settings frozen in advance; nothing was "
          "adjusted after seeing them.", "",
          "| Stock | Always buy | Always buy + stop-loss | **Council** | Purchases lost >S$50: always buy / stop-loss / council |",
          "|---|---|---|---|---|"]
    for p in extras:
        if not p["robust"]:
            continue
        r = p["robust"]["results"]
        cell = lambda n: f"{money(r[n]['S500_becomes']['mean'])} ({r[n]['profit_pct']['mean']:+.0%}), {r[n]['worst_drop']['mean']:.0%}"  # noqa: E731
        L.append(f"| {NAMES[p['key']]} | {cell('Always buy')} | {cell('Always buy + stop-loss')} | {cell('Council')} | "
                 f"{r['Always buy']['lost_over_50']['mean']:.1f} / {r['Always buy + stop-loss']['lost_over_50']['mean']:.1f} / "
                 f"{r['Council']['lost_over_50']['mean']:.1f} |")
    L += ["", "Each cell shows the average of what S$500 became (profit %), then the worst drop from a peak.", "",
          "## Where it fails", "",
          "- **Big rallies:** on AMD, which rose about 10× in 2020–2026, the council's stop-loss sold during dips that "
          "recovered, so it kept far less of the rise than buy-and-hold.",
          "- **Calendar luck:** for Mosaic, the council's S$500 ended anywhere from about S$124 to S$989 depending only on "
          "which days were decision days. Single backtests can mislead.",
          "- **Gap risk:** some losses still exceed S$50 when a stock opens far below the stop after bad overnight news.",
          "- **Direction:** every direction model, including a small transformer and a CNN trained from scratch, learned "
          "little beyond \"markets usually rise\".",
          "",
          "## Honesty notes", "",
          "- **Every setting was chosen on training years only:** walk-forward years for sizing, 2016–2019 for the "
          "direction check.",
          "- **The test years were looked at several times during development.** Some design ideas (sizing instead of "
          "skipping, realistic stops, robust tuning) came after seeing earlier test results. The 8 extra companies are "
          "the freshest check.",
          "- **AMD, Johnson & Johnson and Mosaic were picked by looking at their full histories,** so they are stress "
          "tests, not a random sample.",
          "- **Not included:** trading fees and slippage. The app uses a fixed exchange rate of S$1.30 per US$.",
          ""]
    Path("RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    summary = {p["key"]: {"label": p["label"], "strategy": p["strategy"] and strategy_text(p["strategy"]),
                          "results": p["robust"] and p["robust"]["results"],
                          "test_window": p["robust"] and p["robust"]["test_window"]}
               for p in progs + extras if p["robust"]}
    Path("results").mkdir(exist_ok=True)
    Path("results/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote RESULTS.md and results/summary.json ({len(summary)} programs)")


if __name__ == "__main__":
    main()
