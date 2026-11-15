"""Run one live council decision for the website, with one trained council (program).

The website starts this as a separate process with VERITAS_SCENARIO set (or unset for the main Dow
council), so every program loads its own trained models from its own folder (council/paths.py).
It reads a JSON request on stdin and prints one JSON line with the result.

  {"request": {...UserRequest...}, "decline_reason": null}

Live use, not evaluation: it reads the full raw data up to today through council/loaders (never
the test folder) and the models that were trained on 1992-2019 only. For a single-stock council the
stock takes the Dow's place, exactly as in training (council/paths.py: scenario_inputs).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from council.paths import CACHE, INDEX, SCENARIO, scenario_inputs

RAW = Path("data/raw")


def live_inputs():
    """ctx + panel for today. Cached (rebuilt when any raw file is newer than the cache)."""
    from council.analysts.base import Context
    from council.features import build_panel
    from council.loaders import load_company_news, load_macro, load_market_news, load_prices
    from council.loaders.membership import load_membership

    cache = CACHE / "live_inputs.pkl"
    newest = max(p.stat().st_mtime for p in RAW.rglob("*") if p.is_file())
    if cache.exists() and cache.stat().st_mtime > newest:
        return pickle.loads(cache.read_bytes())
    prices, membership, company = scenario_inputs(load_prices(), load_membership(), load_company_news())
    macro = load_macro()
    ctx = Context(prices=prices, macro=macro, membership=membership,
                  market_news=load_market_news(), company_news=company)
    out = (ctx, build_panel(prices, macro, membership))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def main() -> None:
    from council import narrator, report
    from council.council import apply_decline, run_council
    from council.schema import UserRequest

    payload = json.loads(sys.stdin.read())
    req = UserRequest(**payload["request"])
    shown = req.ticker
    if SCENARIO:                       # a single-stock council trained the stock in the Dow's place
        req = req.model_copy(update={"ticker": INDEX})
    if payload.get("decline_reason"):
        req = apply_decline(req, payload["decline_reason"])
    ctx, panel = live_inputs()
    result = run_council(req, ctx, panel)
    if SCENARIO:                       # show and trade the real stock, not the stand-in name
        result.request.ticker = shown
    sections, info = narrator.write_report(result)
    run_dir = report.save_run(result, sections, info)
    print(json.dumps({"result": json.loads(result.model_dump_json()), "run_dir": str(run_dir).replace("\\", "/"),
                      "narrator": info.get("source", "templates")}))


if __name__ == "__main__":
    main()
