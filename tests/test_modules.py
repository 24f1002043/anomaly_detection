"""Module A, Module B, the decision layer and the metrics.

The worked examples from the technical approach are tests here, because a claim
that the arithmetic is checkable by hand is only worth making if something
actually checks it.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from burnin.config import Config, load_config
from burnin.decision import PASS, REJECT, WATCH, DecisionThresholds, combine, early_decision, terminal_decision
from burnin.features import MAD_TO_SIGMA, build_module_a_features, robust_z
from burnin.generator import BurnInGenerator
from burnin.metrics import (
    detection_metrics,
    early_rejection_metrics,
    f_beta_score,
    interval_coverage,
    select_operating_point,
)
from burnin.module_a import ModuleA, assess_lot_integrity
from burnin.module_b import ModuleB, SafetySlope, component_early_decision


@pytest.fixture(scope="module")
def cfg() -> Config:
    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["run"]["n_lots"] = 60
    raw["module_a"]["isolation_forest"]["n_estimators"] = 60
    raw["module_b"]["random_forest"]["n_estimators"] = 40
    raw["module_b"]["cv_folds"] = 3
    return Config(raw=raw, path=base.path)


@pytest.fixture(scope="module")
def df(cfg):
    return BurnInGenerator(cfg).generate().frame


# ------------------------------------------------------- Section 4 worked example
def test_robust_z_reproduces_the_forty_five_microamp_example():
    """'In a lot averaging 10 uA, a part reading 45 uA is a severe anomaly even
    though the datasheet maximum is 50 uA.'

    Built by hand so the arithmetic is explicit and independent of the generator.
    """
    lot = pd.Series(["L"] * 201)
    values = pd.Series(np.concatenate([np.full(200, 10.0), [45.0]]))
    # a lot with essentially no spread needs the noise floor to avoid a divide
    # by zero; give it a realistic 0.5 uA MAD instead
    rng = np.random.default_rng(0)
    values.iloc[:200] = 10.0 + rng.normal(0, 0.5, 200)
    z = robust_z(values, lot, min_scale=1e-6)
    med = values.iloc[:201].median()
    mad = (values - med).abs().median()
    expected = (45.0 - med) / (MAD_TO_SIGMA * mad)

    assert z.iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert z.iloc[-1] > 10.0, "45 uA against a 10 uA lot must be a severe outlier"
    assert 45.0 < 50.0, "and it passes the static datasheet screen"


def test_robust_z_is_hand_checkable(df, cfg):
    """The quoted z must reproduce from the quoted median and MAD."""
    p = cfg.param("iddq")
    col = p.col(168)
    lot = df["lot_id"].iloc[0]
    sub = df[df["lot_id"] == lot]
    med = sub[col].median()
    mad = (sub[col] - med).abs().median()
    z = robust_z(df[col], df["lot_id"], min_scale=1e-9)
    hand = (sub[col].iloc[3] - med) / (MAD_TO_SIGMA * mad)
    assert z.loc[sub.index[3]] == pytest.approx(hand, rel=1e-9)


# --------------------------------------------------------------- Section 4.2/4.3
def test_detectors_combine_as_a_union_not_an_intersection(df, cfg):
    feats, per_param = build_module_a_features(df, cfg)
    ma = ModuleA(cfg).fit(feats, per_param)
    res = ma.score(feats)
    for p in cfg.parameters:
        combined = res.per_param_severity[p.name].to_numpy()
        r = res.rule_severity[p.name].to_numpy()
        f = res.forest_severity[p.name].to_numpy()
        assert np.allclose(combined, np.maximum(r, f))
        # a union must never be below either detector alone
        assert (combined >= r - 1e-12).all() and (combined >= f - 1e-12).all()


def test_component_severity_is_the_worst_parameter_never_an_average(df, cfg):
    feats, per_param = build_module_a_features(df, cfg)
    res = ModuleA(cfg).fit(feats, per_param).score(feats)
    assert np.allclose(
        res.severity.to_numpy(), res.per_param_severity.max(axis=1).to_numpy()
    )
    mean_sev = res.per_param_severity.mean(axis=1).to_numpy()
    assert (res.severity.to_numpy() >= mean_sev - 1e-12).all()


def test_responsible_parameter_is_the_one_that_drove_the_score(df, cfg):
    feats, per_param = build_module_a_features(df, cfg)
    res = ModuleA(cfg).fit(feats, per_param).score(feats)
    for i in res.severity.index[:200]:
        param = res.responsible_param.loc[i]
        assert res.per_param_severity.loc[i, param] == pytest.approx(
            res.severity.loc[i]
        )


# --------------------------------------------------------------- Section 4.4
def test_lot_integrity_flags_contaminated_lots(cfg):
    base = BurnInGenerator(cfg)
    ds = base.generate()
    integ = assess_lot_integrity(ds.frame, cfg)
    contaminated = set(ds.meta["contaminated_lots"])
    assert contaminated, "fixture produced no contaminated lots"
    missed = contaminated - integ.alarmed_lots
    assert not missed, f"contaminated lots not alarmed: {sorted(missed)}"
    false_alarms = integ.alarmed_lots - contaminated
    assert len(false_alarms) <= 1, f"too many false lot alarms: {sorted(false_alarms)}"


def test_mad_inflation_alone_is_insufficient(cfg):
    """Documents the finding: MAD has a 50 % breakdown point, so a 15-20 %
    contaminated lot does not inflate it past any usable factor. The tail
    inflation index is what actually separates them."""
    ds = BurnInGenerator(cfg).generate()
    integ = assess_lot_integrity(ds.frame, cfg)
    assert not integ.mad_alarmed_lots, (
        "MAD inflation fired -- if this ever becomes true the documented "
        "rationale for the tail-inflation index needs revisiting"
    )
    assert integ.tail_alarmed_lots


# --------------------------------------------------------------- Section 5.3
def test_safety_slope_margin_is_independent_of_v0():
    """margin = S_safe - S_pred = (L_max - M - V_168_hat) / 168.

    V_0 cancels, which is what makes the margin a clean identity an inspector
    can check by hand.
    """
    s = SafetySlope(parameter="leak", limit=20.0, margin_M=2.0, sigma_meas=0.05,
                    horizon=168.0)
    v168 = np.array([14.2, 14.2])
    for v0 in ([1.5, 9.0], [0.1, 19.0]):
        m = s.margin(np.array(v0), v168)
        assert m[0] == pytest.approx(m[1])
        assert m[0] == pytest.approx((20.0 - 2.0 - 14.2) / 168.0)


def test_reliability_margin_rule(cfg):
    """M = max(0.10 * L_max, 3 * sigma_meas), stated explicitly per parameter."""
    p = cfg.param("leak")
    assert p.reliability_margin(sigma_meas=0.01) == pytest.approx(0.10 * p.limit)
    assert p.reliability_margin(sigma_meas=5.0) == pytest.approx(15.0)


def test_early_rejection_acts_on_the_upper_bound(cfg, df):
    """A point prediction ignores its own uncertainty. The rule must reject on
    the upper bound, so a part the model cannot confidently clear is not
    cleared."""
    mb = ModuleB(cfg, "iddq").fit(df)
    pred = mb.predict(df)
    lim = cfg.param("iddq").limit - mb.slope.margin_M
    by_upper = pred.upper > lim
    by_point = pred.point > lim
    # anything the point prediction rejects, the upper bound also rejects
    assert (by_point <= by_upper).all()
    assert by_upper.sum() >= by_point.sum()
    assert (pred.upper >= pred.point).all()
    assert (pred.lower <= pred.point).all()


def test_module_b_beats_all_three_baselines(cfg, df):
    """Section 5.2: the model has to beat all three baselines to justify itself."""
    # split by LOT, never within a lot (Section 5.2)
    lots = np.sort(df["lot_id"].unique())
    cut = set(lots[: len(lots) // 2])
    tr = df[df["lot_id"].isin(cut)]
    te = df[~df["lot_id"].isin(cut)]
    for name in ("iddq", "leak", "tpd"):
        mb = ModuleB(cfg, name).fit(tr)
        y = mb.target(te)
        rf_mae = np.mean(np.abs(mb.predict(te).point - y))
        for bname, bpred in mb.baseline_predictions(te).items():
            b_mae = np.mean(np.abs(bpred - y))
            assert rf_mae < b_mae, f"{name}: RF ({rf_mae:.4f}) lost to {bname} ({b_mae:.4f})"


# --------------------------------------------------------------- Section 6
def test_watch_is_scored_as_flagged_never_as_pass(cfg, df):
    feats, per_param = build_module_a_features(df, cfg)
    res = ModuleA(cfg).fit(feats, per_param).score(feats)
    thr = DecisionThresholds(alert=0.90, reject=0.999)
    term = terminal_decision(df, res, thr, cfg)
    early = early_decision(
        pd.DataFrame(
            {
                "early_reject": np.zeros(len(df), dtype=bool),
                "early_reason": ["none"] * len(df),
                "early_driver_param": ["iddq"] * len(df),
                "early_worst_margin": np.zeros(len(df)),
            },
            index=df.index,
        )
    )
    dec = combine(term, early)
    assert (dec.loc[dec["final_state"] == WATCH, "flagged"]).all()
    assert (dec.loc[dec["final_state"] == REJECT, "flagged"]).all()
    assert not (dec.loc[dec["final_state"] == PASS, "flagged"]).any()
    assert dec["watched"].sum() > 0, "fixture produced no WATCH parts"


def test_static_limit_violation_is_always_a_reject(cfg, df):
    feats, per_param = build_module_a_features(df, cfg)
    res = ModuleA(cfg).fit(feats, per_param).score(feats)
    term = terminal_decision(df, res, DecisionThresholds(0.9, 0.999), cfg)
    assert (term.loc[term["static_fail"], "terminal_state"] == REJECT).all()


# --------------------------------------------------------------- Section 8
def test_f_beta_weights_recall_over_precision():
    """beta = 2 must favour recall, which is the whole point under a 20:1 cost."""
    high_recall = f_beta_score(precision=0.30, recall=0.95, beta=2.0)
    high_precision = f_beta_score(precision=0.95, recall=0.30, beta=2.0)
    assert high_recall > high_precision


def test_cost_ratio_is_twenty_to_one(cfg):
    ev = cfg.evaluation
    assert ev["cost_fn"] / ev["cost_fp"] == pytest.approx(20.0)


def test_operating_point_holds_the_escape_target():
    cfg = load_config()
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.02
    score = np.clip(rng.normal(0.3, 0.2, 5000) + y * 0.5, 0, 1)
    thr, info = select_operating_point(y, score, cfg)
    assert info["feasible"]
    m = detection_metrics(y, score, thr, cfg)
    assert m.escape_rate <= cfg.evaluation["escape_rate_target"] + 1e-9


def test_escape_rate_is_one_minus_recall():
    cfg = load_config()
    rng = np.random.default_rng(1)
    y = rng.random(2000) < 0.05
    s = rng.random(2000)
    m = detection_metrics(y, s, 0.5, cfg)
    assert m.escape_rate == pytest.approx(1.0 - m.recall)


def test_chamber_hours_saved_is_144_per_correct_early_reject():
    cfg = load_config()
    y = np.array([True, True, False, False])
    r = np.array([True, False, True, False])
    m = early_rejection_metrics(y, r, cfg)
    assert m["n_correctly_early_rejected"] == 1
    assert m["chamber_hours_saved"] == pytest.approx(144.0)
    assert m["early_escape_rate"] == pytest.approx(0.5)


def test_interval_coverage_counts_what_it_says():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    lo = np.array([0.0, 1.0, 2.0, 0.0])
    hi = np.array([2.0, 3.0, 4.0, 5.0])
    c = interval_coverage(y, lo, hi)
    assert c["coverage"] == pytest.approx(0.75)
    assert c["above_upper"] == pytest.approx(0.25)
