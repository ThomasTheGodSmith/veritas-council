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
    fig, ax = plt.subplots(figsize=(4.2, 1.9), dpi=110)
    if reasons:
        names = [str(n) for n, _ in reasons][::-1]
        vals = [float(w) for _, w in reasons][::-1]
        ax.barh(names, vals, color=["#2e8b57" if v >= 0 else "#d9822b" for v in vals])
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_xlabel("weight (+ pushes up, − pushes down)", fontsize=7)
    else:
        ax.text(0.5, 0.5, "This AI gave no reasons", ha="center", va="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{_nice(sig.agent)}: {sig.probability_up:.0%} chance up", fontsize=9)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
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
            "title": f"{ids.get(sg.agent, '?')} · {_nice(sg.agent)}",
            "summary": f"{sg.recommended_action.upper()} · {sg.probability_up:.0%} up · {sg.user_idea_verdict} your idea",
            "signal": sg, "chart": _b64(charts[sg.agent]),
            "raw": json.dumps(sg.model_dump(mode="json"), indent=2),
        })
    return out


def _council_folders(r: RunResult) -> list[dict]:
    d = r.model_dump(mode="json")
    folders = []
    if r.fundamentals:
        folders.append({"title": "ℹ️ Fundamentals (info card, no vote)", "raw": json.dumps(d["fundamentals"], indent=2)})
    folders.append({"title": "Round 2 · Cross-examination",
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
    return _env().get_template("report.html.jinja").render(
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
