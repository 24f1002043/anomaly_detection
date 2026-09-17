"""Section 7 and 8.3 - explainability is a scored criterion, not a courtesy.

Coverage is binary and must be total: a flag without a complete explanation
record is a defect in the system.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from burnin.config import Config, load_config
from burnin.decision import DecisionThresholds, combine, early_decision, terminal_decision
from burnin.explain import Explainer, coverage_audit, records_to_frame
from burnin.features import build_module_a_features
from burnin.generator import BurnInGenerator
from burnin.module_a import ModuleA
from burnin.module_b import ModuleB, component_early_decision


@pytest.fixture(scope="module")
def cfg() -> Config:
    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["run"]["n_lots"] = 40
    raw["module_a"]["isolation_forest"]["n_estimators"] = 50
    raw["module_b"]["random_forest"]["n_estimators"] = 40
    raw["module_b"]["cv_folds"] = 3
    return Config(raw=raw, path=base.path)


@pytest.fixture(scope="module")
def built(cfg):
    df = BurnInGenerator(cfg).generate().frame
    feats, per_param = build_module_a_features(df, cfg)
    res = ModuleA(cfg).fit(feats, per_param).score(feats)
    mbs = {p.name: ModuleB(cfg, p.name).fit(df) for p in cfg.parameters}
    preds = {k: v.predict(df) for k, v in mbs.items()}
    fb = {k: v.features(df) for k, v in mbs.items()}
    term = terminal_decision(df, res, DecisionThresholds(0.97, 0.999), cfg)
    dec = combine(term, early_decision(component_early_decision(preds, df.index)))
    return df, feats, res, dec, preds, fb, mbs


def test_every_flag_gets_a_complete_record(built, cfg):
    df, feats, res, dec, preds, fb, mbs = built
    idx = list(dec.index[dec["flagged"]])[:150]
    assert idx, "fixture produced no flags"
    recs = Explainer(cfg, mbs).build(df, feats, res, dec, preds, fb, idx)
    audit = coverage_audit(recs)
    assert audit["coverage"] == 1.0, "explanation coverage must be total"
    assert audit["passes"]


def test_lot_arithmetic_reproduces_by_hand(built, cfg):
    df, feats, res, dec, preds, fb, mbs = built
    idx = list(dec.index[dec["flagged"]])[:80]
    recs = Explainer(cfg, mbs).build(df, feats, res, dec, preds, fb, idx)
    for r in recs:
        la = r.lot_arithmetic
        assert la["reproduces"], f"{r.component_id}: quoted z does not reproduce"
        hand = (la["value"] - la["lot_median"]) / (1.4826 * la["lot_mad"])
        assert hand == pytest.approx(la["robust_z"], rel=1e-6)


def test_record_names_a_responsible_parameter_and_shows_a_margin(built, cfg):
    df, feats, res, dec, preds, fb, mbs = built
    idx = list(dec.index[dec["flagged"]])[:40]
    recs = Explainer(cfg, mbs).build(df, feats, res, dec, preds, fb, idx)
    for r in recs:
        assert r.responsible_parameter in {p.name for p in cfg.parameters}
        assert r.margin["predicted_168h"] is not None
        assert r.margin["interval_lower"] <= r.margin["predicted_168h"]
        assert r.margin["interval_upper"] >= r.margin["predicted_168h"]
        assert len(r.trajectory["values"]) == len(cfg.checkpoints)
        assert len(r.shap_contributions) >= 1


def test_margin_identity_holds_in_the_record(built, cfg):
    """margin = (L_max - M - predicted_168h) / 168, checkable by hand."""
    df, feats, res, dec, preds, fb, mbs = built
    idx = list(dec.index[dec["flagged"]])[:40]
    recs = Explainer(cfg, mbs).build(df, feats, res, dec, preds, fb, idx)
    for r in recs:
        m = r.margin
        expected = (m["datasheet_limit"] - m["reliability_margin_M"]
                    - m["predicted_168h"]) / 168.0
        assert m["margin_to_safety_slope"] == pytest.approx(expected, rel=1e-6)


def test_records_export_to_a_flat_inspector_table(built, cfg):
    df, feats, res, dec, preds, fb, mbs = built
    idx = list(dec.index[dec["flagged"]])[:30]
    recs = Explainer(cfg, mbs).build(df, feats, res, dec, preds, fb, idx)
    frame = records_to_frame(recs)
    assert len(frame) == len(recs)
    for col in ("component_id", "responsible_parameter", "robust_z",
                "predicted_168h", "explanation", "top_contributions"):
        assert col in frame.columns
    assert frame["explanation"].str.len().gt(20).all()
