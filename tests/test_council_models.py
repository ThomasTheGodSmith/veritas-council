"""Devil's Advocate, stacking Mediator + Judge tuning, cross-examination and checklist, all on
synthetic validation predictions (no data files or trained models needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from council.analysts.base import Context, Signal
from council.checklist import review_analysts, strength
from council.crossexam import cross_examine, known_wide
from council.devils_advocate import DevilsAdvocate, majority_up, out_of_fold_p_wrong
from council.mediator import Stacker, fit_council


def _signal(agent: str, p: float) -> Signal:
    return Signal(agent=agent, probability_up=p, expected_return_pct=0.0,
                  recommended_action="buy" if p > 0.55 else "avoid" if p < 0.45 else "wait",
                  user_idea_verdict="supports" if p >= 0.5 else "challenges")


@pytest.fixture(scope="module")
def synthetic():
    """4 validation years, 5 tickers. 'good' knows the answer 75% of the time, 'noise' is random,
    'contrarian' is usually wrong. When volatility is high, the majority is made to be wrong."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2016-01-01", "2019-12-31")
    rows = pd.DataFrame([(d, t) for d in days for t in ["^DJI", "A", "B", "C", "D"]], columns=["date", "ticker"])
    n = len(rows)
    rows["vol_20"] = rng.uniform(0.005, 0.03, n)
    rows["breadth_50"] = rng.uniform(0, 1, n)
    rows["dist_high_250"] = -rng.uniform(0, 0.3, n)
    rows["vix"] = rng.uniform(10, 40, n)
    rows["trend"] = rng.choice([-1.0, 1.0], n)
    y = (rng.uniform(size=n) < 0.6).astype(float)
    high_vol = rows.vol_20 > 0.02
    knows = rng.uniform(size=n) < 0.75
    good = np.where(knows, y, 1 - y)
    noise = rng.uniform(size=n) < 0.5
    contrarian = np.where(rng.uniform(size=n) < 0.7, 1 - y, y)
    # high volatility: 'noise' and 'contrarian' (so the majority) say the opposite of what happens
    noise = np.where(high_vol, 1 - y, noise)
    contrarian = np.where(high_vol, 1 - y, contrarian)
    fr = np.where(y == 1, rng.uniform(0.0, 0.06, n), -rng.uniform(0.0, 0.06, n))
    mae = np.minimum(fr, 0) - rng.uniform(0, 0.05, n)
    base = rows[["date", "ticker"]].assign(y=y, fwd_return=fr, mae=mae,
                                          label_known=rows.date + pd.Timedelta(days=28))
    preds = []
    for agent, vote in [("good", good), ("noise", noise), ("contrarian", contrarian)]:
        p = np.clip(np.where(vote == 1, 0.62, 0.40) + rng.normal(0, 0.03, n), 0.01, 0.99)
        preds.append(base.assign(agent=agent, p=p))
    return pd.concat(preds, ignore_index=True), rows


def test_majority_up_with_tie_break():
    # one up, one down = a tie, broken by the average probability (0.5 counts as up)
    P = pd.DataFrame({"a": [0.6, 0.55, 0.4, 0.7], "b": [0.4, 0.40, 0.4, 0.6]})
    assert majority_up(P).tolist() == [True, False, False, True]


def test_devils_advocate_learns_when_the_majority_is_wrong(synthetic):
    preds, feats = synthetic
    da = DevilsAdvocate()
    # train.py passes fit-year rows; they must be ignored in favour of the validation predictions
    da.fit(pd.DataFrame({"junk": [1]}), Context(prices=pd.DataFrame(),
                                               extra={"validation_predictions": preds, "panel": feats}))
    test = feats.iloc[:400].reset_index(drop=True)
    probs = {"good": np.full(400, 0.62), "noise": np.full(400, 0.62), "contrarian": np.full(400, 0.40)}
    pw = da.p_wrong_from_probs(test, probs)
    hi = test.vol_20.to_numpy() > 0.02
    assert pw[hi].mean() > pw[~hi].mean() + 0.2
    # it votes against the majority (which says up here): p_up below 0.5, lower when p_wrong is high
    p_up = da.against_majority(test, probs)
    assert (p_up < 0.5).all() and p_up[hi].mean() < p_up[~hi].mean()
    assert da.drivers(test.iloc[[0]], {k: v[:1] for k, v in probs.items()})


def test_out_of_fold_p_wrong_covers_every_row(synthetic):
    preds, feats = synthetic
    oof = out_of_fold_p_wrong(preds, feats)
    assert len(oof) == preds[["date", "ticker"]].drop_duplicates().shape[0]
    assert oof.p_wrong.between(0, 1).all()


def test_stacker_keeps_analyst_weights_non_negative():
    rng = np.random.default_rng(1)
    n = 4000
    y = (rng.uniform(size=n) < 0.5).astype(float)
    good = y * 2 - 1 + rng.normal(0, 1, n)          # helpful analyst (logit scale)
    anti = -(y * 2 - 1) + rng.normal(0, 1, n)       # anti-correlated: would get a negative weight
    X = np.column_stack([good, anti, rng.normal(size=n)])
    st = Stacker(n_analysts=2).fit(X, y)
    assert st.w[1] > 0.5 and st.w[2] == pytest.approx(0.0, abs=1e-6)
    assert ((st.predict(X) >= 0.5) == (y == 1)).mean() > 0.75


def test_fit_council_tunes_theta_and_reports_out_of_fold(synthetic, tmp_path):
    preds, feats = synthetic
    cm = fit_council(preds, feats, save=False)
    assert 0.5 <= cm.theta <= 0.85
    assert all(w >= 0 for w in cm.trust().values())
    assert cm.trust()["good"] > cm.trust()["contrarian"]
    v = cm.validation
    assert {"stacking", "simple_average", "always_buy", "judge_at_theta", "tuning"} <= set(v)
    p = cm.p_council({"good": np.array([0.62]), "noise": np.array([0.62]), "contrarian": np.array([0.4])},
                     feats.iloc[[0]].reset_index(drop=True), np.array([0.3]))
    assert 0 < p[0] < 1


def test_cross_exam_counts_who_won(synthetic):
    preds, _ = synthetic
    D = pd.Timestamp("2020-02-01")
    wide = known_wide(preds, D)
    sigs = [_signal("good", 0.62), _signal("contrarian", 0.40), _signal("noise", 0.61)]
    out = cross_examine(sigs, preds, "A", D, wide=wide)
    pair = next(d for d in out if {d.a, d.b} == {"good", "contrarian"})
    assert pair.a == "good" and pair.a_says_up and pair.n_past >= 30
    same = wide[(wide.ticker == "A") & (wide.good >= 0.5) & (wide.contrarian < 0.5)]
    assert pair.a_win_rate == pytest.approx((same.y == 1).mean(), abs=1e-4)
    assert not any({d.a, d.b} == {"good", "noise"} for d in out)      # they agree today


def test_cross_exam_ignores_outcomes_not_known_yet(synthetic):
    preds, _ = synthetic
    early = pd.Timestamp("2016-03-01")
    wide = known_wide(preds, early)
    assert (wide.label_known < early).all()


def test_checklist_reviews(synthetic):
    preds, feats = synthetic
    D = pd.Timestamp("2020-02-01")
    wide = known_wide(preds, D).merge(feats[["date", "ticker", "trend", "vol_20"]], on=["date", "ticker"])
    sigs = [_signal("good", 0.62), _signal("contrarian", 0.40)]
    dis = cross_examine(sigs, preds, "A", D, wide=wide)
    today = pd.Series({"trend": 1.0, "vol_20": 0.01})
    revs = {r.agent: r for r in review_analysts(sigs, wide, "A", today, dis, buy_allowed=False)}
    assert revs["good"].evidence_strength == "strong" and revs["good"].fits_user_rules is False
    assert revs["contrarian"].fits_user_rules is True          # it says avoid, which never breaks a rule
    assert revs["good"].trust_score > revs["contrarian"].trust_score
    assert any("disagreements" in x for x in revs["contrarian"].negatives)
    assert [strength(n) for n in (10, 30, 100, 101)] == ["weak", "medium", "medium", "strong"]
