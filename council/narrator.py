"""Claude Opus 5.5 writes the readable report (PLAN.md §1c, sections 2-9).

How it works, in four steps:
1. Every section first gets a TEMPLATE sentence filled straight from the numbers. These always work.
2. If an API key exists, Claude gets ONLY the finished council outputs (the RunResult JSON: every
   AI's prediction, the Checklist, the plans and the Judge's decision). It never sees raw market
   data or news, and it runs after the Judge has decided, so it cannot change any decision.
3. NUMBER CHECK (code): every number Claude writes must appear in the input (0.57 may be written
   as 57%). If one doesn't, that section goes back to its template and a small note says so.
4. No key, an error, a refusal or a timeout -> the whole report uses the templates.

This is the only place in the project that calls Claude. Evaluation never calls it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

from council.schema import RunResult

CONFIG = Path("config/agents.yaml")

# The fixed sections Claude writes (PLAN.md §1c). Analyst cards are added as "analyst:<name>".
SECTIONS = ["verdict", "rules", "idea_vs_council", "debate", "positives_negatives", "plans",
            "what_would_change"]

NICE_NAMES = {
    "momentum": "Momentum", "mean_reversion": "Mean Reversion", "ml_forecaster": "ML Forecaster",
    "risk_officer": "Risk Officer", "news_reaction": "News-Reaction", "pattern_oracle": "Pattern Oracle",
    "raw_reader": "Raw Reader", "market_regime": "Market Regime", "macro": "Macro",
    "dow_breadth": "Dow Breadth & Weights", "devils_advocate": "Devil's Advocate",
    "fundamentals_card": "Fundamentals",
}

SYSTEM_PROMPT = """You explain the results of "Market Council", a school prototype where several \
small AI models each predicted whether a stock or the Dow index will be higher in a few days, and \
a Judge model then made a final decision. The reader is a teenager making their first investment.

Rules you must follow:
- Explain ONLY what is in the JSON you are given. Do not add predictions, opinions, market facts, \
news or advice that are not in it. The decision is already made; do not question or change it.
- Keep every number exactly as given. You may write a probability like 0.57 as 57%, and round \
money to whole dollars, but never invent, combine or calculate new numbers (no sums, no differences).
- When an analyst challenges the user's idea, say so clearly.
- Plain, friendly English. Short sentences. Explain any jargon in a few words.
- Each section is 1-4 sentences. Each analyst card is 2-3 sentences.
- Write "S$" for the user's money and "$" for the share price."""


def _nice(agent: str) -> str:
    return NICE_NAMES.get(agent, agent.replace("_", " ").title())


def _pct(p: float) -> str:
    return f"{p:.0%}"


def _money(x: float) -> str:
    return f"S${x:,.0f}"


# ------------------------------------------------------------------------------ templates
def template_sections(r: RunResult) -> dict[str, str]:
    """One plain sentence (or a few) per section, filled straight from the numbers."""
    j, m, req = r.judge, r.mediator, r.request
    word = {"buy": "BUY", "wait": "WAIT", "avoid": "DON'T BUY"}[j.decision]
    if j.decision == "buy":
        word += f" {_money(j.amount)}"
    s: dict[str, str] = {}
    s["verdict"] = (f"The council's answer is {word}. It thinks there is a {_pct(m.p_council)} chance "
                    f"the price is higher in {req.horizon_days} days, with {_pct(j.confidence)} "
                    f"confidence. Why: {j.reason}.")

    passed = [c for c in r.rules if c.passed]
    failed = [c for c in r.rules if not c.passed]
    if not r.rules:
        s["rules"] = "No rule results were recorded for this run."
    elif failed:
        s["rules"] = (f"{len(failed)} of your rules failed: " + "; ".join(c.text for c in failed)
                      + ". The Judge is not allowed to approve a plan that breaks one of your rules.")
    else:
        s["rules"] = f"All {len(passed)} of your rules pass: " + "; ".join(c.text for c in passed) + "."

    voting = [sg for sg in r.signals]
    sup = [sg for sg in voting if sg.user_idea_verdict == "supports"]
    cha = [sg for sg in voting if sg.user_idea_verdict == "challenges"]
    idea = f"You said: \"{req.idea_text}\". {len(sup)} analysts support your idea and {len(cha)} challenge it."
    if cha:
        strongest = max(cha, key=lambda sg: abs(sg.probability_up - 0.5))
        idea += (f" The strongest challenge comes from {_nice(strongest.agent)}, which gives a "
                 f"{_pct(strongest.probability_up)} chance of the price being higher.")
    s["idea_vs_council"] = idea

    for sg in r.signals:
        line = (f"{_nice(sg.agent)} thinks there is a {_pct(sg.probability_up)} chance the price is "
                f"higher in {sg.horizon_days} days and suggests {sg.recommended_action.upper()}")
        if sg.top_reasons:
            line += f", mainly because of {sg.top_reasons[0][0]}"
        line += f". It {sg.user_idea_verdict} your idea."
        acc = sg.track_record.get("validation_accuracy")
        if acc is not None:
            line += f" In testing on past years it was right {_pct(acc)} of the time."
        s[f"analyst:{sg.agent}"] = line

    debate = [d.summary + "." for d in r.crossexam]
    pw = r.devils_advocate.get("p_majority_wrong")
    if pw is not None:
        debate.append(f"The Devil's Advocate warns that in situations like this the majority of "
                      f"analysts has been wrong {_pct(pw)} of the time.")
    s["debate"] = " ".join(debate) or "The analysts did not disagree in a way the Historian could check."

    if r.reviews:
        best = max(r.reviews, key=lambda v: v.trust_score)
        s["positives_negatives"] = (f"The Checklist trusts {_nice(best.agent)} most today "
                                    f"(trust score {best.trust_score:.2f}). The table below lists "
                                    "each analyst's strengths and weaknesses from its real track record.")
    else:
        s["positives_negatives"] = "No analyst reviews were recorded for this run."

    plan_bits = []
    for p in m.plans:
        bit = f"Plan {p.id} ({p.action.replace('_', ' ')}"
        if p.amount:
            bit += f" {_money(p.amount)}"
        bit += ")"
        if p.worst_case_loss is not None:
            bit += f" has a worst-case loss of {_money(p.worst_case_loss)}"
        if p.source == "mediator_original":
            bit += " and is the Mediator's own idea"
        plan_bits.append(bit)
    s["plans"] = ("The Mediator built " + str(len(m.plans)) + " plans. " + ". ".join(plan_bits) + "."
                  if m.plans else "The Mediator found no plan that passes all your rules.")
    if j.plan_id:
        s["plans"] += f" The Judge picked plan {j.plan_id}."

    s["what_would_change"] = (" ".join(x if x.endswith(".") else x + "." for x in j.would_change_if)
                              or "The council did not record what would change its decision.")
    return s


# ------------------------------------------------------------------------------ number check
_NUM = re.compile(r"(?<![A-Za-z_\d.])-?\d[\d,]*(?:\.\d+)?")


def numbers_in_text(text: str) -> list[tuple[float, int]]:
    """Every number written in the text, with how many decimals it was written with."""
    out = []
    for m in _NUM.finditer(text):
        s = m.group().replace(",", "")
        if s in ("-", ""):
            continue
        dec = len(s.split(".")[1]) if "." in s else 0
        out.append((abs(float(s)), dec))
    return out


def _walk_numbers(obj, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
    elif isinstance(obj, str):
        for v, _ in numbers_in_text(obj):
            out.add(v)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk_numbers(k, out)
            _walk_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, out)


def allowed_numbers(payload) -> set[float]:
    """Every number in the input, plus the same number as a percentage (0.57 -> 57)."""
    base: set[float] = set()
    _walk_numbers(payload, base)
    return base | {v * 100 for v in base}


def number_ok(value: float, decimals: int, allowed: set[float]) -> bool:
    """A written number passes if some input number rounds to it at the precision it was written."""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(value - a) <= tol for a in allowed)


def check_section(text: str, allowed: set[float]) -> list[float]:
    """The numbers in `text` that are NOT in the input (empty list = the section passes)."""
    return [v for v, d in numbers_in_text(text) if not number_ok(v, d, allowed)]


def narrator_payload(r: RunResult) -> dict:
    """What Claude sees: the finished council outputs, plus a few counts it may quote."""
    data = r.model_dump(mode="json")
    data["counts"] = {
        "n_analysts": len(r.signals),
        "n_support": sum(s.user_idea_verdict == "supports" for s in r.signals),
        "n_challenge": sum(s.user_idea_verdict == "challenges" for s in r.signals),
        "n_rules": len(r.rules),
        "n_rules_failed": sum(not c.passed for c in r.rules),
        "n_plans": len(r.mediator.plans),
    }
    return data


# ------------------------------------------------------------------------------ Claude call
def _settings() -> dict:
    try:
        return yaml.safe_load(CONFIG.read_text(encoding="utf-8")).get("narrator", {})
    except (OSError, yaml.YAMLError):
        return {}


def _schema(agents: list[str]) -> dict:
    props = {k: {"type": "string"} for k in SECTIONS}
    props["analysts"] = {
        "type": "array",
        "items": {"type": "object", "additionalProperties": False, "required": ["agent", "text"],
                  "properties": {"agent": {"type": "string", "enum": agents}, "text": {"type": "string"}}},
    }
    return {"type": "object", "additionalProperties": False, "required": [*SECTIONS, "analysts"],
            "properties": props}


def ask_claude(payload: dict, agents: list[str], model: str, timeout: float = 90.0) -> dict[str, str]:
    """One Claude call -> {section: text}. Raises on any problem (the caller falls back)."""
    import anthropic

    client = anthropic.Anthropic(timeout=timeout, max_retries=1)
    prompt = ("Here are the finished results of one council run as JSON. Write the report sections.\n"
              "verdict: the Judge's decision, the council's probability, the confidence and why.\n"
              "rules: the user's rules and whether each passed.\n"
              "idea_vs_council: how many analysts support or challenge the user's idea and the strongest reason against it.\n"
              "analysts: 2-3 sentences for every analyst in `signals`.\n"
              "debate: the cross-examination results and the Devil's Advocate warning.\n"
              "positives_negatives: one short intro to the table of each analyst's strengths and weaknesses.\n"
              "plans: the Mediator's plans side by side, its own plan, any dropped plans and the one the Judge picked.\n"
              "what_would_change: what would change the decision.\n\n"
              + json.dumps(payload, ensure_ascii=False))
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                 # if a safety check declines, the API retries on another model
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _schema(agents)}},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"Claude stopped early: {resp.stop_reason}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    out = {k: data[k] for k in SECTIONS}
    for item in data.get("analysts", []):
        out[f"analyst:{item['agent']}"] = item["text"]
    return out


def has_api_key() -> bool:
    load_dotenv()
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def write_report(r: RunResult, use_claude: bool = True, claude_fn=None) -> tuple[dict[str, str], dict]:
    """Return (sections, info). info = {"source": "claude"|"templates", "notes": [...], "fallback": [...]}

    `claude_fn(payload, agents) -> dict` can be passed in tests to stand in for the real API.
    """
    templates = template_sections(r)
    info = {"source": "templates", "notes": [], "fallback": []}
    cfg = _settings()
    if claude_fn is None:
        if not (use_claude and cfg.get("enabled", True) and has_api_key()):
            info["notes"].append("Written from templates (no Claude API key set).")
            return templates, info
        model = cfg.get("model", "claude-opus-5-5")
        claude_fn = lambda payload, agents: ask_claude(payload, agents, model)  # noqa: E731

    payload = narrator_payload(r)
    try:
        written = claude_fn(payload, [s.agent for s in r.signals])
    except Exception as e:  # any API error, timeout, refusal or bad JSON -> templates
        info["notes"].append(f"Claude was unavailable ({type(e).__name__}), so the report uses templates.")
        return templates, info

    allowed = allowed_numbers(payload)
    sections = {}
    for key, tmpl in templates.items():
        text = written.get(key)
        if not text:
            sections[key] = tmpl
            info["fallback"].append(key)
            continue
        bad = check_section(text, allowed)
        if bad:
            sections[key] = tmpl
            info["fallback"].append(key)
        else:
            sections[key] = text
    info["source"] = "claude"
    info["notes"].append("Written by Claude Opus 5.5 from the AIs' finished results; every number was checked.")
    if info["fallback"]:
        info["notes"].append(f"{len(info['fallback'])} section(s) used the template version because a "
                             "number didn't match the AIs' output.")
    return sections, info
