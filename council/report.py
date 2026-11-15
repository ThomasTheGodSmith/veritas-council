"""Readable report + collapsible AI folders (PLAN.md §1c).

render_html(result, sections, info) -> one self-contained HTML page (charts embedded as images)
save_run(result, sections, info)    -> runs/<run_id>/ with report.html, summary.md and one folder per AI
zip_run(run_dir)                    -> runs/<run_id>.zip for the "Download all folders" button

The written parts (sections 2-9) come from narrator.py. Everything in the AI folders comes
straight from each AI's own output; `reasons.md` is filled in from templates, never by Claude.
"""
from __future__ import annotations

import base64
import io
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # draw charts without a screen
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from council.analysts.base import Signal  # noqa: E402
from council.narrator import NICE_NAMES, _nice  # noqa: E402
from council.schema import RunResult  # noqa: E402

DISCLAIMER = "This project is a prototype and is not financial advice."
RUNS = Path("runs")
TEMPLATES = Path(__file__).parent / "templates"
CONFIG = Path("config/agents.yaml")


LABELS = {
    "trend": "Trend (20 vs 50-day average)", "rsi_centred": "RSI, centred", "rsi": "RSI (14-day)",
    "macd_hist": "MACD histogram", "sma20_sma50": "20-day vs 50-day average", "z20": "Stretch from 20-day average",
    "z20_sq": "Stretch, squared", "bb_pct": "Position in Bollinger bands", "dist_sma50": "Distance from 50-day average",
    "dist_high_250": "Distance from 1-year high", "ret_1": "1-day return", "ret_5": "5-day return",
    "ret_20": "20-day return", "ret_60": "60-day return", "vol_20": "20-day volatility", "vol_60": "60-day volatility",
    "vol_ratio": "Volatility, recent vs longer", "volume_z": "Unusual trading volume", "vix": "VIX (market fear)",
    "vix_chg_20": "VIX change, 20 days", "tnx_chg_20": "10-year yield change", "dxy_chg_20": "US dollar change",
    "oil_chg_20": "Oil price change", "dff": "Fed funds rate", "t10y2y": "Yield curve (10y minus 2y)",
    "cpi_yoy": "Inflation (CPI, yearly)", "unrate": "Unemployment rate", "unrate_chg_3m": "Unemployment change, 3 months",
    "breadth_50": "Share of Dow stocks above 50-day average", "breadth_centred": "Market breadth",
    "breadth_chg_20": "Breadth change, 20 days", "divergence": "Narrow-rally warning", "logit_bottom_up": "Dow members' combined view",
    "rel_strength_60": "Strength vs other Dow stocks", "rank_60": "Rank among Dow stocks", "vote_spread": "How split the vote is",
    "vote_share_up": "Share of analysts saying up", "mean_p": "Average analyst view", "regime_p": "Market mood model",
    "sigma": "Forecast volatility (GARCH)", "p_bull": "Chance of a bull market", "p_bear": "Chance of a bear market",
    "p_choppy": "Chance of a choppy market", "validation_accuracy": "Right on past years", "accuracy": "Accuracy",
    "brier": "Brier score (lower is better)", "n": "Past examples",
}


def label(name) -> str:
    """A readable name for a model feature (falls back to tidying the raw name)."""
    s = str(name)
    return LABELS.get(s, s.replace("_", " ").strip().capitalize())


def rosette_svg(decision: str) -> str:
    """The banknote-style guilloché behind the verdict: a few closed hypotrochoid curves."""
    seed = {"buy": 1, "wait": 2, "avoid": 3}.get(decision, 1)
    R, paths = 96, []
    for k in range(5):
        r, d, n = 12 + 3 * k + seed, 34 + 7 * k, 420
        pts = []
        for i in range(n + 1):
            t = i / n * math.pi * 2 * r
            x = (R - r) * math.cos(t) + d * math.cos((R - r) / r * t)
            y = (R - r) * math.sin(t) - d * math.sin((R - r) / r * t)
            pts.append(f"{'L' if i else 'M'}{215 + x * 1.42:.1f} {215 + y * 1.42:.1f}")
        paths.append(f'<path d="{"".join(pts)}"/>')
    return f'<svg class="rosette" viewBox="0 0 430 430" aria-hidden="true">{"".join(paths)}</svg>'


def display_ticker(ticker: str) -> str:
    return "DJIA (DIA)" if ticker in ("^DJI", "DIA") else ticker


def agent_ids() -> dict[str, int]:
    """name -> folder number from config/agents.yaml (momentum -> 1, ..., fundamentals_card -> 0)."""
    try:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        return {a["name"]: int(a["id"]) for a in cfg.get("analysts", [])}
    except (OSError, yaml.YAMLError, KeyError):
        return {}


# ------------------------------------------------------------------------------ charts
def reasons_chart(sig: Signal) -> bytes:
    """A small horizontal bar chart of the AI's own top reasons (green pushes up, orange pushes down)."""
    reasons = list(sig.top_reasons)[:5]
    ink, soft = "#173a33", "#4d6660"
    fig, ax = plt.subplots(figsize=(4.6, 0.55 + 0.42 * max(len(reasons), 2)), dpi=130)
    fig.patch.set_facecolor("#ffffff")
    if reasons:
        names = [label(n) for n, _ in reasons][::-1]
        vals = [float(w) for _, w in reasons][::-1]
        bars = ax.barh(names, vals, color=["#1e6e61" if v >= 0 else "#a9443d" for v in vals], height=0.62)
        ax.axvline(0, color=soft, lw=0.8)
        span = max(abs(v) for v in vals) or 1.0
        ax.set_xlim(-span * 1.35, span * 1.35)
        for b, v in zip(bars, vals):
            ax.text(v + (span * 0.04 if v >= 0 else -span * 0.04), b.get_y() + b.get_height() / 2, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7, color=soft)
        ax.set_xticks([])
        ax.set_xlabel("pushes towards a fall  ←   →  pushes towards a rise", fontsize=7, color=soft)
    else:   # every analyst explains itself now; this is only a safety net
        ax.text(0.5, 0.5, "No reasons were given", ha="center", va="center", fontsize=8, color=soft)
        ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(labelsize=7.5, colors=ink, length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _b64(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


# ------------------------------------------------------------------------------ reasons.md (template, not Claude)
def reasons_md(sig: Signal) -> str:
    lines = [f"# {_nice(sig.agent)}", "",
             f"> {DISCLAIMER}", "",
             f"**Decision:** {sig.recommended_action.upper()} · {sig.probability_up:.0%} chance the price is "
             f"higher in {sig.horizon_days} days · expected return {sig.expected_return_pct:+.2f}% · "
             f"it **{sig.user_idea_verdict}** your idea.", "",
             "## Reasons, by the AI itself (weight)", ""]
    lines += [f"- {name}: {w:+.4f}" for name, w in sig.top_reasons] or ["- (none given)"]
    lines += ["", "## Key numbers it used", ""]
    lines += [f"- {k}: {v}" for k, v in sig.key_numbers.items()] or ["- (none)"]
    lines += ["", "## Track record", ""]
    lines += [f"- {k}: {v}" for k, v in sig.track_record.items()] or ["- (none recorded)"]
    lines += ["", "_Generated from this AI's own output with a template, not by Claude._", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ HTML
def _env() -> Environment:
    return Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "jinja"]))


def _folders(r: RunResult, charts: dict[str, bytes]) -> list[dict]:
    ids = agent_ids()
    out = []
    for sg in sorted(r.signals, key=lambda s: ids.get(s.agent, 99)):
        out.append({
            "title": f"{ids.get(sg.agent, '?')}. {_nice(sg.agent)}",
            "summary": f"leans {sg.recommended_action}, {sg.probability_up:.0%} chance of a rise, {sg.user_idea_verdict} your idea",
            "signal": sg, "chart": _b64(charts[sg.agent]),
            "raw": json.dumps(sg.model_dump(mode="json"), indent=2),
        })
    return out


def _council_folders(r: RunResult) -> list[dict]:
    d = r.model_dump(mode="json")
    folders = []
    if r.fundamentals:
        folders.append({"title": "Fundamentals (information only, no vote)", "raw": json.dumps(d["fundamentals"], indent=2)})
    folders.append({"title": "Round 2: cross-examination",
                    "raw": json.dumps({"crossexam": d["crossexam"], "devils_advocate": d["devils_advocate"]}, indent=2)})
    folders.append({"title": "Checklist", "raw": json.dumps({"rules": d["rules"], "reviews": d["reviews"]}, indent=2)})
    folders.append({"title": "Mediator", "raw": json.dumps(d["mediator"], indent=2)})
    folders.append({"title": "Judge", "raw": json.dumps(d["judge"], indent=2)})
    if r.risk:
        folders.append({"title": "Risk Officer's simulation summary", "raw": json.dumps(d["risk"], indent=2)})
    return folders


def verdict_word(r: RunResult) -> str:
    j = r.judge
    if j.decision == "buy":
        return f"BUY S${j.amount:,.0f}"
    return {"wait": "WAIT", "avoid": "DON'T BUY"}[j.decision]


def render_html(r: RunResult, sections: dict[str, str], info: dict | None = None,
                charts: dict[str, bytes] | None = None) -> str:
    info = dict(info or {"source": "templates", "notes": [], "fallback": []})
    info["notes"] = [*info.get("notes", []), *getattr(r, "notes", [])]   # council notes: skipped analysts etc.
    charts = charts or {sg.agent: reasons_chart(sg) for sg in r.signals}
    reviews = {v.agent: v for v in r.reviews}
    plan = next((p for p in r.mediator.plans if p.id == r.judge.plan_id), None)
    stop = plan.stop_loss_pct if (plan and r.judge.decision == "buy" and plan.stop_loss_pct) else None
    bad = (r.risk or {}).get("var95_unprotected_pct", (r.risk or {}).get("var95_used_pct"))
    j = r.judge
    title = f"Buy S${j.amount:,.2f}" if j.decision == "buy" else {"wait": "Wait", "avoid": "Don't buy"}[j.decision]
    return _env().get_template("report.html.jinja").render(
        rosette=rosette_svg(j.decision), verdict_title=title, stop_pct=stop, bad_case=bad, label=label,
        disclaimer=DISCLAIMER, r=r, s=sections, info=info, fallback=set(info.get("fallback", [])),
        ticker=display_ticker(r.request.ticker), verdict=verdict_word(r),
        n_support=sum(sg.user_idea_verdict == "supports" for sg in r.signals),
        n_challenge=sum(sg.user_idea_verdict == "challenges" for sg in r.signals),
        folders=_folders(r, charts), council_folders=_council_folders(r), reviews=reviews,
        nice=_nice, nice_names=NICE_NAMES,
    )


# ------------------------------------------------------------------------------ summary.md
def summary_md(r: RunResult, sections: dict[str, str], info: dict) -> str:
    order = [("Verdict", "verdict"), ("Your rules", "rules"), ("Your idea vs the council", "idea_vs_council")]
    lines = [f"> {DISCLAIMER}", "", f"# Council report: {display_ticker(r.request.ticker)} — {r.run_id}", "",
             f"**{verdict_word(r)}** · data as of {r.data_as_of} · decision date {r.decision_date}", ""]
    for title, key in order:
        lines += [f"## {title}", "", sections.get(key, ""), ""]
    lines += ["## The analysts", ""]
    for sg in r.signals:
        lines += [f"- **{_nice(sg.agent)}:** {sections.get('analyst:' + sg.agent, '')}"]
    for title, key in [("The debate", "debate"), ("Positives and negatives", "positives_negatives"),
                       ("The plans", "plans"), ("What would change the decision", "what_would_change")]:
        lines += ["", f"## {title}", "", sections.get(key, "")]
    lines += ["", "---", *[f"_{n}_" for n in info.get("notes", [])], ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ save + zip
def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_run(r: RunResult, sections: dict[str, str], info: dict, root: Path = RUNS) -> Path:
    """Write runs/<run_id>/ exactly as PLAN.md §1c shows. Returns the folder."""
    run_dir = Path(root) / r.run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    charts = {sg.agent: reasons_chart(sg) for sg in r.signals}
    ids = agent_ids()
    for sg in r.signals:
        folder = run_dir / "analysts" / f"{ids.get(sg.agent, 99):02d}_{sg.agent}"
        _write_json(folder / "decision.json", sg.model_dump(mode="json"))
        (folder / "reasons.md").write_text(reasons_md(sg), encoding="utf-8")
        (folder / "chart.png").write_bytes(charts[sg.agent])
    if r.fundamentals:
        _write_json(run_dir / "analysts" / "00_fundamentals_card" / "info.json",
                    {"agent": "fundamentals_card", "votes": False, **r.fundamentals})
    d = r.model_dump(mode="json")
    _write_json(run_dir / "council" / "checklist.json", {"rules": d["rules"], "reviews": d["reviews"]})
    _write_json(run_dir / "council" / "mediator.json", d["mediator"])
    _write_json(run_dir / "council" / "judge.json", d["judge"])
    _write_json(run_dir / "council" / "crossexam.json",
                {"crossexam": d["crossexam"], "devils_advocate": d["devils_advocate"]})
    _write_json(run_dir / "run.json", d)                       # the full council output, for the record
    (run_dir / "report.html").write_text(render_html(r, sections, info, charts), encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_md(r, sections, info), encoding="utf-8")
    return run_dir


def zip_run(run_dir: Path) -> Path:
    """Zip one run folder -> runs/<run_id>.zip."""
    run_dir = Path(run_dir)
    return Path(shutil.make_archive(str(run_dir), "zip", root_dir=run_dir.parent, base_dir=run_dir.name))
