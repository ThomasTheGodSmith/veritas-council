"""Paper broker with the local simulator (never a live endpoint). Uses the FAKE fixture."""
import pytest

from council import paper_broker as PB
from council.schema import RuleCheck
from tests.fixtures.fake_run import fake_run_result


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(PB, "DB", tmp_path / "portfolio.sqlite")
    monkeypatch.setattr(PB, "latest_close", lambda symbol: (523.15, "2026-11-13"))
    monkeypatch.setattr(PB, "fx_rate", lambda: 1.30)


def test_no_live_endpoint_in_broker():
    src = open(PB.__file__, encoding="utf-8").read()
    assert "https://api.alpaca.markets" not in src and "paper=False" not in src


def test_simulator_places_split_plan_now_step_and_saves_the_rest():
    r = fake_run_result()                                 # judge picked plan C: S$200 now, S$150 later
    out = PB.place_from_run(r, broker=PB.SimBroker())
    assert out["status"] == "placed", out
    o = out["order"]
    assert o["symbol"] == "DIA" and o["amount_sgd"] == 200
    assert o["fill_price"] == 523.15
    assert o["qty"] == pytest.approx(200 / 1.30 / 523.15, rel=1e-4)
    # stop = the tighter of the max-loss price (50/350 = 14.3% down) and the plan's own 14% stop
    assert o["stop_price"] == pytest.approx(523.15 * (1 - 0.14), abs=0.01)
    assert len(PB.list_orders()) == 1
    pending = PB.list_pending()
    assert len(pending) == 1 and pending[0]["amount_sgd"] == 150
    pos = PB.SimBroker().positions()
    assert pos[0]["symbol"] == "DIA" and pos[0]["qty"] == pytest.approx(o["qty"])
    step = PB.run_pending_step(pending[0]["id"], broker=PB.SimBroker())
    assert step["status"] == "placed" and PB.list_pending() == []


def test_failed_rule_blocks_the_order():
    r = fake_run_result()
    bad = lambda result, plan, price: [RuleCheck(rule="max_loss", passed=False, text="too risky now")]  # noqa: E731
    out = PB.place_from_run(r, rules_check=bad, broker=PB.SimBroker())
    assert out["status"] == "blocked" and PB.list_orders() == []


def test_price_move_blocks_the_order(monkeypatch):
    monkeypatch.setattr(PB, "latest_close", lambda symbol: (480.0, "2026-11-14"))
    out = PB.place_from_run(fake_run_result(), broker=PB.SimBroker())
    assert out["status"] == "blocked"


def test_wait_decision_places_nothing():
    r = fake_run_result()
    r.judge.decision = "wait"
    out = PB.place_from_run(r, broker=PB.SimBroker())
    assert out["status"] == "not_placed" and PB.list_orders() == []
