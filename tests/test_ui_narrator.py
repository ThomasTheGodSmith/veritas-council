"""Narrator: template fallback and the number check (PLAN.md §1c). Uses the FAKE fixture."""
from council import narrator as N
from tests.fixtures.fake_run import fake_run_result


def test_templates_pass_their_own_number_check():
    r = fake_run_result()
    allowed = N.allowed_numbers(N.narrator_payload(r))
    for key, text in N.template_sections(r).items():
        assert N.check_section(text, allowed) == [], key


def test_percent_and_rounding_are_allowed():
    allowed = N.allowed_numbers({"p": 0.5734, "loss": 38.2})
    assert N.check_section("57% chance, 57.3% exactly, loss S$38", allowed) == []
    assert N.check_section("a 64% chance", allowed) == [64.0]


def test_section_with_invented_number_falls_back_to_template():
    r = fake_run_result()
    templates = N.template_sections(r)

    def fake_claude(payload, agents):
        out = dict(templates)
        out["verdict"] = "The council says buy. It is 57% sure."                     # all numbers real
        out["debate"] = "Mean Reversion won 91% of the time, so watch out."           # 91 is invented
        out["analyst:momentum"] = "Momentum sees a 58% chance of a rise."
        return out

    sections, info = N.write_report(r, claude_fn=fake_claude)
    assert info["source"] == "claude"
    assert sections["verdict"] == "The council says buy. It is 57% sure."
    assert sections["analyst:momentum"].startswith("Momentum sees")
    assert sections["debate"] == templates["debate"]
    assert info["fallback"] == ["debate"]


def test_api_error_means_whole_report_uses_templates():
    r = fake_run_result()

    def broken(payload, agents):
        raise TimeoutError("too slow")

    sections, info = N.write_report(r, claude_fn=broken)
    assert info["source"] == "templates"
    assert sections == N.template_sections(r)


def test_no_api_key_uses_templates(monkeypatch):
    monkeypatch.setattr(N, "has_api_key", lambda: False)
    sections, info = N.write_report(fake_run_result())
    assert info["source"] == "templates"


def test_claude_never_sees_raw_data():
    payload = N.narrator_payload(fake_run_result())
    assert set(payload) <= set(fake_run_result().model_dump()) | {"counts"}
