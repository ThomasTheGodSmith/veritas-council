"""Report rendering and the saved run folder, from the FAKE fixture."""
import zipfile

from council import narrator, report
from tests.fixtures.fake_run import fake_run_result


def test_report_html_has_every_section_and_disclaimer_first():
    r = fake_run_result()
    s, info = narrator.write_report(r, use_claude=False)
    html = report.render_html(r, s, info)
    body = html.split("<body>", 1)[1]
    assert body.index(report.DISCLAIMER) < body.index("Buy S$350")
    for heading in ["Your rules", "Your idea and the council", "The analysts", "The debate",
                    "Strengths and weaknesses of each analyst", "The plans", "What would change the decision", "AI folders"]:
        assert heading in html
    assert html.count("<details>") >= len(r.signals)
    assert "✓" in html and "data:image/png;base64," in html
    assert 'class="rosette"' in html
    assert "challenges your idea" in html


def test_save_run_writes_the_folder_layout(tmp_path):
    r = fake_run_result()
    s, info = narrator.write_report(r, use_claude=False)
    d = report.save_run(r, s, info, root=tmp_path)
    for f in ["report.html", "summary.md", "council/checklist.json", "council/mediator.json",
              "council/judge.json", "council/crossexam.json", "analysts/01_momentum/decision.json",
              "analysts/01_momentum/reasons.md", "analysts/01_momentum/chart.png",
              "analysts/11_devils_advocate/decision.json"]:
        assert (d / f).exists(), f
    assert (d / "summary.md").read_text(encoding="utf-8").startswith("> " + report.DISCLAIMER)
    z = report.zip_run(d)
    with zipfile.ZipFile(z) as zf:
        assert any(n.endswith("report.html") for n in zf.namelist())
