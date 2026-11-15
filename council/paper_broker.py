"""Paper trading only: the Alpaca PAPER API, or a local simulator (PLAN.md §1b, "Approve & Invest").

What happens when the user presses ✅ Approve & Invest:
1. The hard rules are checked again on the latest price. If any fails, nothing is bought.
2. A bracket order is placed: a market buy for the "now" part of the plan, a stop-loss at the
   price where the user would lose their max-loss amount, and an optional take-profit.
3. Later steps of a split/staged plan are saved as PENDING. Each one re-checks the rules when run.
4. Every order, its fill and the full council output are saved in data/portfolio.sqlite.

Which broker:
- APCA_API_KEY_ID + APCA_API_SECRET_KEY in .env -> Alpaca's paper endpoint
  (https://paper-api.alpaca.markets). TradingClient(paper=True) only ever talks to that.
- no keys -> the local simulator fills at the latest close in our dataset.
There is deliberately NO live-trading endpoint anywhere in this file.

Money: the user's amounts are in S$; US shares are priced in US$. We convert with one fixed,
approximate rate (SGD_PER_USD below, or the SGD_PER_USD environment variable).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from council.schema import Plan, RuleCheck, RunResult

DB = Path("data/portfolio.sqlite")
PAPER_URL = "https://paper-api.alpaca.markets"    # the only Alpaca endpoint this project may use
SGD_PER_USD = 1.30                                 # approximate; override with the SGD_PER_USD env var

# rules_check(result, plan, price) -> list of RuleCheck (or anything with a .passed attribute)
RulesCheck = Callable[[RunResult, Plan | None, float], list]


def trade_symbol(ticker: str) -> str:
    """The Dow itself can't be bought, so "the Dow" means the DIA fund that tracks it."""
    return "DIA" if ticker in ("^DJI", "DJIA", "DIA") else ticker


def fx_rate() -> float:
    load_dotenv()
    try:
        return float(os.environ.get("SGD_PER_USD", SGD_PER_USD))
    except ValueError:
        return SGD_PER_USD


def latest_close(symbol: str) -> tuple[float, str]:
    """Latest close in the dataset (live use: the full raw data, never the test folder)."""
    from council.loaders import load_prices

    p = load_prices([symbol])
    if p.empty:
        raise LookupError(f"no price data for {symbol}")
    last = p.sort_values("date").iloc[-1]
    return float(last["close"]), str(last["date"].date())


def stop_price_for(price: float, amount: float, max_loss: float, plan_stop_pct: float | None) -> float | None:
    """Price at which this position has lost the user's max-loss amount.
    If the plan has its own tighter stop, use that. None if no stop makes sense."""
    if amount <= 0:
        return None
    pct = max_loss / amount
    if plan_stop_pct:
        pct = min(pct, plan_stop_pct)
    if pct >= 1:
        return None
    return round(price * (1 - pct), 2)


# ------------------------------------------------------------------------------ storage
def _db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY, ts TEXT, run_id TEXT, plan_id TEXT, broker TEXT, broker_order_id TEXT,
        symbol TEXT, side TEXT, qty REAL, amount_sgd REAL, amount_usd REAL, fill_price REAL,
        stop_price REAL, take_profit REAL, status TEXT, note TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS pending_steps (
        id INTEGER PRIMARY KEY, created TEXT, run_id TEXT, plan_id TEXT, symbol TEXT,
        amount_sgd REAL, step_when TEXT, max_loss REAL, stop_pct REAL, status TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS council_runs (run_id TEXT PRIMARY KEY, ts TEXT, json TEXT)")
    return con


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------------------ brokers
class SimBroker:
    """Local simulator: fills immediately at the latest close in the dataset."""
    name = "simulator"

    def buy(self, symbol: str, amount_usd: float, price: float, stop_price: float | None,
            take_profit: float | None) -> dict:
        qty = round(amount_usd / price, 6)                   # fractional shares are fine here
        return {"broker_order_id": f"SIM-{datetime.now():%Y%m%d%H%M%S%f}", "qty": qty,
                "fill_price": price, "status": "filled",
                "note": "Simulated fill at the latest close in the dataset."}

    def positions(self) -> list[dict]:
        con = _db()
        rows = con.execute("""SELECT symbol, SUM(qty), SUM(qty*fill_price), MIN(stop_price) FROM orders
                              WHERE broker='simulator' AND status='filled' AND side='buy'
                              GROUP BY symbol""").fetchall()
        con.close()
        out = []
        for sym, qty, cost, stop in rows:
            try:
                px, _ = latest_close(sym)
            except Exception:
                px = cost / qty if qty else 0.0
            out.append({"symbol": sym, "qty": round(qty, 6), "avg_price": round(cost / qty, 2) if qty else 0,
                        "last_price": round(px, 2), "market_value_usd": round(qty * px, 2),
                        "unrealised_pl_usd": round(qty * px - cost, 2), "stop_price": stop})
        return out


class AlpacaPaperBroker:
    """Alpaca's PAPER account (fake money, real market prices)."""
    name = "alpaca_paper"

    def __init__(self, key: str, secret: str):
        from alpaca.trading.client import TradingClient
        self.client = TradingClient(key, secret, paper=True)    # paper=True -> paper-api.alpaca.markets only

    def buy(self, symbol: str, amount_usd: float, price: float, stop_price: float | None,
            take_profit: float | None) -> dict:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        whole = int(amount_usd // price)
        if whole >= 1 and stop_price:
            # Alpaca bracket orders need whole shares: buy + stop-loss (+ take-profit) as one order.
            req = MarketOrderRequest(
                symbol=symbol, qty=whole, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET, stop_loss=StopLossRequest(stop_price=stop_price),
                take_profit=TakeProfitRequest(limit_price=take_profit or round(price * 1.5, 2)))
            note = f"Bracket order for {whole} whole share(s) (Alpaca brackets need whole shares)."
        else:
            # Too small for one share: a fractional (notional) market buy. Alpaca can't attach a
            # stop to fractional orders, so the stop-loss price is recorded here for the user.
            req = MarketOrderRequest(symbol=symbol, notional=round(amount_usd, 2), side=OrderSide.BUY,
                                     time_in_force=TimeInForce.DAY)
            note = "Fractional buy; stop-loss is recorded locally because Alpaca can't attach it to fractional orders."
        order = self.client.submit_order(req)
        fill = float(order.filled_avg_price) if order.filled_avg_price else price
        qty = float(order.qty) if order.qty else round(amount_usd / price, 6)
        return {"broker_order_id": str(order.id), "qty": qty, "fill_price": fill,
                "status": str(order.status.value if hasattr(order.status, "value") else order.status),
                "note": note}

    def positions(self) -> list[dict]:
        out = []
        for p in self.client.get_all_positions():
            out.append({"symbol": p.symbol, "qty": float(p.qty), "avg_price": float(p.avg_entry_price),
                        "last_price": float(p.current_price or 0), "market_value_usd": float(p.market_value or 0),
                        "unrealised_pl_usd": float(p.unrealized_pl or 0), "stop_price": None})
        return out


def get_broker():
    load_dotenv()
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if key and secret:
        try:
            return AlpacaPaperBroker(key, secret)
        except Exception:
            pass
    return SimBroker()


# ------------------------------------------------------------------------------ rules
MAX_PRICE_MOVE = 0.03     # if the price moved more than 3% since the council ran, ask for a re-run


def _var95_pct(result: RunResult) -> float | None:
    """The Risk Officer's 1-in-20 loss as a fraction (0.07 = 7%)."""
    for key in ("var95_used_pct", "var95_pct", "var_95_pct"):
        if key in result.risk:
            v = float(result.risk[key])
            return v / 100 if v > 1 else v
    for sg in result.signals:
        if sg.agent == "risk_officer":
            for key in ("var95_pct", "var_95_pct"):
                if key in sg.key_numbers:
                    v = float(sg.key_numbers[key])
                    return v / 100 if v > 1 else v
    return None


def default_rules_check(result: RunResult, plan: Plan | None, price: float, ctx=None) -> list:
    """Re-run the four hard rules (council.council.recheck_rules: with ctx it recomputes the
    Risk Officer's VaR on the newest data) plus a "has the price moved?" check."""
    checks = []
    if result.price:
        # result.price is the latest close of what the council looked at (^DJI for the Dow,
        # while we buy DIA), so compare like with like.
        sym = trade_symbol(result.request.ticker)
        now = price if sym == result.request.ticker else latest_close(result.request.ticker)[0]
        move = now / result.price - 1
        checks.append(RuleCheck(rule="price_unchanged", passed=abs(move) <= MAX_PRICE_MOVE,
                                text=f"Price moved {move:+.1%} since the council ran (limit ±{MAX_PRICE_MOVE:.0%})"))
    try:
        from council.council import recheck_rules
        return checks + list(recheck_rules(result, plan.id if plan else None, ctx=ctx))
    except ImportError:
        pass
    var = _var95_pct(result)
    try:
        from council.rules import check_plan
    except ImportError:
        check_plan = None
    if check_plan is None or var is None:
        return checks + list(result.rules)          # can't recompute: keep the run's own rule results
    return checks + check_plan(result.judge.amount, result.request.ticker, result.request, var,
                               index_weights=result.risk.get("index_weights"))


def _failed(checks) -> list[str]:
    return [getattr(c, "text", str(c)) for c in (checks or []) if not getattr(c, "passed", True)]


# ------------------------------------------------------------------------------ main entry
def _find_plan(result: RunResult) -> Plan | None:
    return next((p for p in result.mediator.plans if p.id == result.judge.plan_id), None)


def place_from_run(result: RunResult, rules_check: RulesCheck | None = default_rules_check,
                   broker=None, take_profit_pct: float | None = None) -> dict:
    """✅ Approve & Invest. Returns a dict the app shows: status, message, order, pending steps."""
    j = result.judge
    if j.decision != "buy" or j.amount <= 0:
        return {"status": "not_placed",
                "message": f"The council said {j.decision.upper()}, so there is nothing to buy."}
    plan = _find_plan(result)
    symbol = trade_symbol(result.request.ticker)
    price, as_of = latest_close(symbol)

    # 1. rules again, on the latest price. A failed rule blocks the order. No exceptions.
    checks = rules_check(result, plan, price) if rules_check else None
    failed = _failed(checks)
    if failed:
        return {"status": "blocked", "message": "Order blocked, a rule failed on the latest price: "
                + "; ".join(failed), "price": price}
    rules_note = "Rules re-checked on the latest price." if checks is not None else \
        "Note: council/rules.py is not available yet, so the rules from the council run were used."
    if checks is None and any(not c.passed for c in result.rules):
        return {"status": "blocked", "message": "Order blocked: a rule failed in the council run."}

    # 2. split the plan into "now" and later steps
    # The Judge may size the plan down (half-Kelly), so scale every step to the Judge's amount.
    steps = plan.steps if plan and plan.steps else []
    scale = j.amount / plan.amount if plan and plan.amount > 0 else 1.0
    steps = [s.model_copy(update={"amount": round(s.amount * scale, 2)}) for s in steps]
    now_sgd = sum(s.amount for s in steps if s.when.strip().lower() == "now") if steps else j.amount
    later = [s for s in steps if s.when.strip().lower() != "now"]
    stop_pct = plan.stop_loss_pct if plan else None
    stop = stop_price_for(price, j.amount, result.request.max_loss, stop_pct)
    tp = round(price * (1 + take_profit_pct), 2) if take_profit_pct else None

    broker = broker or get_broker()
    con = _db()
    con.execute("INSERT OR REPLACE INTO council_runs VALUES (?,?,?)",
                (result.run_id, _now(), result.model_dump_json()))
    order = None
    if now_sgd > 0:
        usd = now_sgd / fx_rate()
        fill = broker.buy(symbol, usd, price, stop, tp)
        con.execute("""INSERT INTO orders (ts, run_id, plan_id, broker, broker_order_id, symbol, side, qty,
                       amount_sgd, amount_usd, fill_price, stop_price, take_profit, status, note)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (_now(), result.run_id, j.plan_id, broker.name, fill["broker_order_id"], symbol, "buy",
                     fill["qty"], now_sgd, round(usd, 2), fill["fill_price"], stop, tp, fill["status"], fill["note"]))
        order = {"symbol": symbol, "amount_sgd": now_sgd, "amount_usd": round(usd, 2), "stop_price": stop,
                 "take_profit": tp, "price_as_of": as_of, "broker": broker.name, **fill}
    for s in later:
        con.execute("""INSERT INTO pending_steps (created, run_id, plan_id, symbol, amount_sgd, step_when,
                       max_loss, stop_pct, status) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (_now(), result.run_id, j.plan_id, symbol, s.amount, s.when, result.request.max_loss,
                     stop_pct, "pending"))
    con.commit()
    con.close()
    msg = f"Paper order placed with the {broker.name}." if order else "No 'now' step; later steps saved."
    if later:
        msg += f" {len(later)} later step(s) saved as pending."
    return {"status": "placed", "message": msg + " " + rules_note, "order": order,
            "pending": [s.model_dump() for s in later]}


def run_pending_step(step_id: int, rules_check: RulesCheck | None = default_rules_check, broker=None) -> dict:
    """Run one saved later step now (the user clicks it in the Portfolio tab). Re-checks the rules first."""
    con = _db()
    row = con.execute("SELECT run_id, plan_id, symbol, amount_sgd, step_when, max_loss, stop_pct, status "
                      "FROM pending_steps WHERE id=?", (step_id,)).fetchone()
    if not row or row[7] != "pending":
        con.close()
        return {"status": "not_found", "message": f"No pending step {step_id}."}
    run_id, plan_id, symbol, amount, when, max_loss, stop_pct, _ = row
    saved = con.execute("SELECT json FROM council_runs WHERE run_id=?", (run_id,)).fetchone()
    price, _ = latest_close(symbol)
    if saved and rules_check:
        result = RunResult.model_validate_json(saved[0])
        failed = _failed(rules_check(result, _find_plan(result), price))
        if failed:
            con.close()
            return {"status": "blocked", "message": "Step blocked, a rule failed: " + "; ".join(failed)}
    broker = broker or get_broker()
    stop = stop_price_for(price, amount, max_loss, stop_pct)
    usd = amount / fx_rate()
    fill = broker.buy(symbol, usd, price, stop, None)
    con.execute("""INSERT INTO orders (ts, run_id, plan_id, broker, broker_order_id, symbol, side, qty,
                   amount_sgd, amount_usd, fill_price, stop_price, take_profit, status, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (_now(), run_id, plan_id, broker.name, fill["broker_order_id"], symbol, "buy", fill["qty"],
                 amount, round(usd, 2), fill["fill_price"], stop, None, fill["status"], f"Step '{when}'. " + fill["note"]))
    con.execute("UPDATE pending_steps SET status='done' WHERE id=?", (step_id,))
    con.commit()
    con.close()
    return {"status": "placed", "message": f"Step {step_id} ({when}) placed.", "fill": fill}


# ------------------------------------------------------------------------------ portfolio tab
def list_orders() -> list[dict]:
    con = _db()
    cur = con.execute("SELECT id, ts, run_id, plan_id, broker, symbol, side, qty, amount_sgd, amount_usd, "
                      "fill_price, stop_price, take_profit, status, note FROM orders ORDER BY id DESC")
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()
    return rows


def list_pending() -> list[dict]:
    con = _db()
    cur = con.execute("SELECT id, created, run_id, plan_id, symbol, amount_sgd, step_when, status "
                      "FROM pending_steps WHERE status='pending' ORDER BY id")
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    con.close()
    return rows


def list_positions(broker=None) -> list[dict]:
    try:
        return (broker or get_broker()).positions()
    except Exception as e:
        return [{"error": f"Could not read positions: {type(e).__name__}"}]


if __name__ == "__main__":
    print(json.dumps({"broker": get_broker().name, "orders": list_orders()[:5]}, indent=2, default=str))
