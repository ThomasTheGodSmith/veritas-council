"""Market Council: the Gradio app (PLAN.md §1, §1c).

Run it with:  uv run python app.py   then open http://127.0.0.1:7860

Tabs:
  Council      your request + idea -> Run Council -> readable report + AI folders + zip download,
               then ✅ Approve & Invest (paper order) or ❌ Decline (give a reason, the council re-runs)
  Portfolio    paper positions, orders, and pending later steps of split/staged plans
  Dow weights  how much each of the 30 companies counts in the Dow today

This is live use, so it reads the full raw data through council.loaders (never the test split).
"""
from __future__ import annotations

import traceback
from functools import lru_cache

import gradio as gr
import pandas as pd

from council import narrator, paper_broker, report
from council.loaders import load_membership, load_prices
from council.schema import RunResult, UserRequest

DISCLAIMER = report.DISCLAIMER
DOW_LABEL = "DJIA (DIA)"
SECTOR_CHOICES = ["Technology", "Financials", "Industrials", "Health Care", "Consumer Staples",
                  "Consumer Discretionary", "Communication Services", "Energy", "Materials"]


# ------------------------------------------------------------------------------ the 30 members
@lru_cache(maxsize=1)
def current_members() -> pd.DataFrame:
    """Today's 30 Dow members: rows of dow_membership.csv whose end_date is blank."""
    m = load_membership()
    return m[m.end_date.isna()].sort_values("company").reset_index(drop=True)


def ticker_choices() -> list[str]:
    try:
        rows = current_members()
        return [DOW_LABEL] + [f"{c} ({t})" for c, t in zip(rows.company, rows.ticker) if isinstance(t, str)]
    except Exception:
        return [DOW_LABEL]


def label_to_ticker(label: str) -> str:
    return "^DJI" if label == DOW_LABEL else label.rsplit("(", 1)[-1].rstrip(")")


# ------------------------------------------------------------------------------ Dow weights
def dow_weights() -> tuple[pd.DataFrame, str]:
    """Price weights on the latest day: weight = price / sum of prices; points = Δprice / divisor,
    divisor = sum of raw member prices / Dow close (the Dow is a price-weighted average)."""
    members = current_members()
    tickers = [t for t in members.ticker if isinstance(t, str)]
    p = load_prices(tickers + ["^DJI"])
    dji = p[p.ticker == "^DJI"].sort_values("date")
    day = dji.date.iloc[-1]
    px = p[p.ticker != "^DJI"].pivot(index="date", columns="ticker", values="raw_close").sort_index()
    px = px[px.index <= day]
    today, prev = px.iloc[-1], px.iloc[-2]
    divisor = today.sum() / float(dji.close.iloc[-1])
    names = dict(zip(members.ticker, members.company))
    table = pd.DataFrame({
        "Company": [names.get(t, t) for t in today.index],
        "Ticker": today.index,
        "Price ($)": today.round(2).values,
        "Weight %": (today / today.sum() * 100).round(2).values,
        "Points today": ((today - prev) / divisor).round(1).values,
    }).sort_values("Weight %", ascending=False).reset_index(drop=True)
    note = (f"Data as of {day.date()}. Divisor = {today.sum():,.2f} / {float(dji.close.iloc[-1]):,.2f} "
            f"= {divisor:.4f}. A $1 move in any member moves the Dow by about {1 / divisor:.2f} points. "
            f"Sum of points today: {table['Points today'].sum():+.1f}.")
    if len(today.dropna()) < 30:
        note += f" Only {len(today.dropna())} members have prices."
    return table, note


# ------------------------------------------------------------------------------ council
class CouncilNotReady(Exception):
    pass


@lru_cache(maxsize=1)
def _live_inputs():
    """ctx + panel for today, built from the full raw data (live use, not evaluation)."""
    import council.council as cc
    if hasattr(cc, "build_live_inputs"):
        return cc.build_live_inputs()
    from council.analysts.base import Context
    from council.features import build_panel
    from council.loaders import load_company_news, load_macro, load_market_news
    prices, macro, membership = load_prices(), load_macro(), load_membership()
    ctx = Context(prices=prices, macro=macro, membership=membership,
                  market_news=load_market_news(), company_news=load_company_news())
    return ctx, build_panel(prices, macro, membership)


def run_council_for(req: UserRequest) -> RunResult:
    try:
        from council.council import run_council
    except ImportError as e:
        raise CouncilNotReady("The council isn't built yet (council/council.py is missing).") from e
    try:
        ctx, panel = _live_inputs()
        return run_council(req, ctx, panel)
    except FileNotFoundError as e:
        raise CouncilNotReady(f"Some models aren't trained yet. Run `uv run python train.py` first. ({e})") from e


def decline_for(req: UserRequest, reason: str) -> RunResult:
    """❌ Decline: council.council turns the reason into a changed request, then we re-run."""
    from council.council import apply_decline
    return run_council_for(apply_decline(req, reason))


def _friendly(msg: str) -> str:
    return (f"<div style='padding:16px;border:1px solid #e8c766;background:#fff4d6;border-radius:8px;color:#1d1d1f'>"
            f"<b>{DISCLAIMER}</b><p>{msg}</p></div>")


def _publish(result: RunResult):
    """Narrate, render, save and zip one council result -> outputs for the UI."""
    sections, info = narrator.write_report(result)
    run_dir = report.save_run(result, sections, info)
    html = report.render_html(result, sections, info)
    zip_path = report.zip_run(run_dir)
    can_buy = result.judge.decision == "buy" and result.judge.amount > 0
    status = (f"Council says **{report.verdict_word(result)}**. "
              + ("Press ✅ Approve & Invest to place the paper order, or ❌ Decline with a reason."
                 if can_buy else "There's nothing to buy. You can ❌ Decline with a reason to re-run."))
    return html, str(zip_path), result.model_dump_json(), status


def on_run(label, amount, max_loss, horizon, avoid, idea, direction):
    req = UserRequest(ticker=label_to_ticker(label), amount=float(amount), currency="SGD",
                      max_loss=float(max_loss), horizon_days=int(horizon), avoid_sectors=list(avoid or []),
                      thinks_up=(direction == "Up"), idea_text=idea or ("I think it'll go up" if direction == "Up"
                                                                       else "I think it'll go down"))
    try:
        return _publish(run_council_for(req))
    except CouncilNotReady as e:
        return _friendly(str(e)), None, "", "The council isn't ready yet."
    except Exception as e:
        traceback.print_exc()
        return _friendly(f"Something went wrong running the council: {type(e).__name__}: {e}"), None, "", "Error."


def on_decline(state_json, reason):
    if not state_json:
        return gr.update(), None, state_json, "Run the council first."
    if not (reason or "").strip():
        return gr.update(), gr.update(), state_json, "Please type a reason for declining, then press ❌ Decline again."
    old = RunResult.model_validate_json(state_json)
    try:
        return _publish(decline_for(old.request, reason.strip()))
    except CouncilNotReady as e:
        return _friendly(str(e)), None, state_json, "The council isn't ready yet."
    except Exception as e:
        traceback.print_exc()
        return gr.update(), gr.update(), state_json, f"Re-run failed: {type(e).__name__}: {e}"


def on_approve(state_json):
    if not state_json:
        return "Run the council first."
    result = RunResult.model_validate_json(state_json)
    try:
        ctx, _ = _live_inputs()          # newest data, so the Risk Officer's VaR is recomputed
        check = lambda res, plan, price: paper_broker.default_rules_check(res, plan, price, ctx=ctx)  # noqa: E731
        out = paper_broker.place_from_run(result, rules_check=check)
    except Exception as e:
        traceback.print_exc()
        return f"❌ Order failed: {type(e).__name__}: {e}"
    lines = [f"**{out['status'].upper()}** — {out['message']}"]
    o = out.get("order")
    if o:
        lines.append(f"- {o['symbol']}: {o['qty']} shares filled at ${o['fill_price']:.2f} "
                     f"(S${o['amount_sgd']:,.0f} ≈ US${o['amount_usd']:,.2f}) · broker: {o['broker']} · "
                     f"order id `{o['broker_order_id']}`")
        if o.get("stop_price"):
            lines.append(f"- Stop-loss at ${o['stop_price']:.2f}"
                         + (f", take-profit at ${o['take_profit']:.2f}" if o.get("take_profit") else ""))
        lines.append(f"- {o['note']}")
    for s in out.get("pending", []):
        lines.append(f"- Pending later step: S${s['amount']:,.0f} {s['when']} (see the Portfolio tab)")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ portfolio
def portfolio_tables():
    def df(rows, empty_cols):
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=empty_cols)
    broker = paper_broker.get_broker()
    return (f"Broker: **{broker.name}** (paper only). S$ → US$ at {paper_broker.fx_rate():.2f} S$ per US$ (approximate).",
            df(paper_broker.list_positions(broker), ["symbol", "qty", "avg_price", "last_price", "market_value_usd"]),
            df(paper_broker.list_orders(), ["id", "ts", "symbol", "qty", "fill_price", "stop_price", "status"]),
            df(paper_broker.list_pending(), ["id", "symbol", "amount_sgd", "step_when", "status"]))


def on_run_step(step_id):
    try:
        ctx, _ = _live_inputs()
        check = lambda res, plan, price: paper_broker.default_rules_check(res, plan, price, ctx=ctx)  # noqa: E731
        out = paper_broker.run_pending_step(int(step_id), rules_check=check)
    except Exception as e:
        out = {"message": f"{type(e).__name__}: {e}"}
    return (out["message"], *portfolio_tables()[1:])


# ------------------------------------------------------------------------------ UI
CSS = """
#approve { background: #2e8b57 !important; color: white !important; }
#decline { background: #c0392b !important; color: white !important; }
.disclaimer-bar { font-weight: 600; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Market Council") as demo:
        gr.Markdown(f"# 🏛️ Market Council\n**{DISCLAIMER}**", elem_classes="disclaimer-bar")
        with gr.Tab("Council"):
            gr.Markdown(f"**{DISCLAIMER}**")
            with gr.Row():
                with gr.Column(scale=1):
                    ticker = gr.Dropdown(ticker_choices(), value=DOW_LABEL, label="What to buy")
                    amount = gr.Number(500, label="Amount (S$)", minimum=1)
                    max_loss = gr.Number(50, label="The most you'd accept losing (S$)", minimum=0)
                    horizon = gr.Slider(5, 120, value=20, step=1, label="How long you'll hold (days)")
                    avoid = gr.CheckboxGroup(SECTOR_CHOICES, label="Sectors to avoid")
                    direction = gr.Radio(["Up", "Down"], value="Up", label="You think it will go…")
                    idea = gr.Textbox("I think it'll go up", label="Your idea, in your own words")
                    run_btn = gr.Button("Run Council", variant="primary")
                with gr.Column(scale=3):
                    status = gr.Markdown("Fill in your request and press **Run Council**.")
                    report_html = gr.HTML()
                    zip_file = gr.File(label="Download all folders (.zip)")
                    with gr.Row():
                        approve_btn = gr.Button("✅ Approve & Invest (paper)", elem_id="approve")
                        decline_btn = gr.Button("❌ Decline", elem_id="decline")
                    decline_reason = gr.Textbox(label="Why are you declining? (the council re-runs with your reason)")
                    order_out = gr.Markdown()
            state = gr.State("")
            run_btn.click(on_run, [ticker, amount, max_loss, horizon, avoid, idea, direction],
                          [report_html, zip_file, state, status])
            decline_btn.click(on_decline, [state, decline_reason], [report_html, zip_file, state, status])
            approve_btn.click(on_approve, [state], [order_out])

        with gr.Tab("Portfolio"):
            gr.Markdown(f"**{DISCLAIMER}** Paper trading only: no real money is ever used.")
            broker_md = gr.Markdown()
            refresh = gr.Button("Refresh")
            gr.Markdown("### Positions")
            pos = gr.Dataframe(interactive=False)
            gr.Markdown("### Orders")
            orders = gr.Dataframe(interactive=False)
            gr.Markdown("### Pending later steps (split / staged plans)")
            pending = gr.Dataframe(interactive=False)
            with gr.Row():
                step_id = gr.Number(label="Pending step id", precision=0)
                step_btn = gr.Button("Run this step now (re-checks your rules first)")
            step_out = gr.Markdown()
            refresh.click(portfolio_tables, None, [broker_md, pos, orders, pending])
            step_btn.click(on_run_step, [step_id], [step_out, pos, orders, pending])
            demo.load(portfolio_tables, None, [broker_md, pos, orders, pending])

        with gr.Tab("Dow weights"):
            gr.Markdown(f"**{DISCLAIMER}** The Dow is price-weighted: a company with a higher share "
                        "price counts for more, whatever the company's size.")
            weights_note = gr.Markdown()
            weights = gr.Dataframe(interactive=False)

            def _weights():
                try:
                    t, n = dow_weights()
                    return n, t
                except Exception as e:
                    return f"Couldn't build the weights table: {type(e).__name__}: {e}", pd.DataFrame()
            demo.load(_weights, None, [weights_note, weights])
    return demo


if __name__ == "__main__":
    build_ui().launch(css=CSS)
