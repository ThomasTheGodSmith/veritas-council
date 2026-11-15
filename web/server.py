"""Market Council website: a local web app that runs the council you pick.

    uv run python web/server.py          this computer only: http://127.0.0.1:8000
    uv run python web/server.py --lan    also other devices on your home network: http://<this PC's IP>:8000

Councils (each trained separately, see scripts/run_stock.py):
  main  the Dow + its 30 companies     AMD  volatile stock     JNJ  steady stock
  MOS   declining stock                DJI  the Dow index on its own
Each run starts scripts/live_run.py in its own process with that council's models. Orders go to the
paper broker only (local simulator, or Alpaca paper if keys are in .env). Never real money.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from council import paper_broker  # noqa: E402
from council.report import DISCLAIMER  # noqa: E402
from council.schema import RunResult, UserRequest  # noqa: E402

PROGRAMS = {
    "main": {"label": "Main", "detail": "The Dow and its 30 companies", "scenario": None,
             "models": "models", "summary": "DOW"},
    "AMD": {"label": "AMD", "detail": "Volatile stock", "scenario": "AMD", "models": "models/scenarios/AMD", "summary": "AMD"},
    "JNJ": {"label": "Johnson & Johnson", "detail": "Steady stock", "scenario": "JNJ", "models": "models/scenarios/JNJ", "summary": "JNJ"},
    "MOS": {"label": "Mosaic", "detail": "Declining stock", "scenario": "MOS", "models": "models/scenarios/MOS", "summary": "MOS"},
    "DJI": {"label": "Dow alone", "detail": "The Dow index on its own", "scenario": "^DJI",
            "models": "models/scenarios/DJI", "summary": "DJI"},
}
NAMES = {"AMD": "AMD", "JNJ": "Johnson & Johnson", "MOS": "Mosaic"}

app = FastAPI(title="Market Council")
app.mount("/static", StaticFiles(directory=ROOT / "web" / "static"), name="static")
(ROOT / "runs").mkdir(exist_ok=True)
app.mount("/runs", StaticFiles(directory=ROOT / "runs"), name="runs")


def current_members() -> list[dict]:
    out = []
    with (ROOT / "data/raw/membership/dow_membership.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["end_date"] and row["yf_ticker"]:
                out.append({"ticker": row["yf_ticker"], "name": row["company"].split(" / ")[-1]})
    return sorted(out, key=lambda r: r["name"])


def tickers_for(key: str) -> list[dict]:
    if key == "main":
        return [{"ticker": "^DJI", "name": "The Dow (bought as DIA)"}] + current_members()
    if key == "DJI":
        return [{"ticker": "^DJI", "name": "The Dow (bought as DIA)"}]
    return [{"ticker": key, "name": NAMES.get(key, key)}]


def summary() -> dict:
    p = ROOT / "results/summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "static" / "index.html")


@app.get("/api/programs")
def programs():
    s = summary()
    out = []
    for key, p in PROGRAMS.items():
        out.append({"key": key, "label": p["label"], "detail": p["detail"],
                    "ready": (ROOT / p["models"] / "strategy.json").exists(),
                    "tickers": tickers_for(key), "strategy": (s.get(p["summary"]) or {}).get("strategy")})
    return {"disclaimer": DISCLAIMER, "programs": out}


@app.get("/api/results")
def results():
    return summary()


class RunBody(BaseModel):
    program: str
    request: UserRequest
    decline_reason: str | None = None


@app.post("/api/run")
def run(body: RunBody):
    p = PROGRAMS.get(body.program)
    if p is None:
        raise HTTPException(404, f"Unknown council '{body.program}'.")
    if not (ROOT / p["models"] / "strategy.json").exists():
        raise HTTPException(409, f"The {p['label']} council isn't trained yet. Run scripts/run_stock.py first.")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("VERITAS_SCENARIO", None)
    if p["scenario"]:
        env["VERITAS_SCENARIO"] = p["scenario"]
    payload = json.dumps({"request": body.request.model_dump(), "decline_reason": body.decline_reason})
    r = subprocess.run([sys.executable, "scripts/live_run.py"], input=payload, capture_output=True,
                       text=True, env=env, cwd=ROOT, timeout=600)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.startswith("{")]
    if r.returncode != 0 or not lines:
        raise HTTPException(500, "The council couldn't run: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1][:300])
    out = json.loads(lines[-1])
    out["report_url"] = "/" + out["run_dir"] + "/report.html"
    return out


class ApproveBody(BaseModel):
    result: dict


@app.post("/api/approve")
def approve(body: ApproveBody):
    return paper_broker.place_from_run(RunResult(**body.result))


@app.get("/api/portfolio")
def portfolio():
    return {"positions": paper_broker.list_positions(), "orders": paper_broker.list_orders(),
            "pending": paper_broker.list_pending()}


def lan_ip() -> str:
    """This computer's address on the home network (the one other devices use)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))     # no packet is sent; this just picks the LAN interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    lan = "--lan" in sys.argv
    host = "0.0.0.0" if lan else "127.0.0.1"
    print(f"Market Council: http://127.0.0.1:{port}" + (f"  and on your network: http://{lan_ip()}:{port}" if lan else ""))
    uvicorn.run(app, host=host, port=port)
