"""Section 2 and 5.1 - checkpoint discipline.

Module B is restricted to Value_0h and Value_24h by the problem statement, and
that restriction is enforced in code by a feature allowlist, not by convention.
These tests are what makes that claim checkable.

A 0/24/96 h regression may be reported as a clearly labelled auxiliary
experiment. It must never be presented as the Module B result.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from burnin.config import Config, load_config
from burnin.features import build_module_b_features
from burnin.generator import BurnInGenerator
from burnin.module_b import BASELINES, ModuleB
from burnin.schema import CheckpointLeakError, module_b_allowlist


@pytest.fixture(scope="module")
def cfg() -> Config:
    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["run"]["n_lots"] = 40
    raw["module_b"]["random_forest"]["n_estimators"] = 40
    raw["module_b"]["cv_folds"] = 3
    return Config(raw=raw, path=base.path)


@pytest.fixture(scope="module")
def df(cfg):
    return BurnInGenerator(cfg).generate().frame


def test_allowlist_rejects_forbidden_checkpoints(cfg):
    al = module_b_allowlist(cfg)
    assert al.allowed_hours == (0, 24)
    assert al.forbidden_hours == (96, 168)
    for bad in ["iddq_v96h", "leak_v168h", "tpd_z_96h", "x_168h_drift"]:
        with pytest.raises(CheckpointLeakError):
            al.validate([bad])


def test_allowlist_accepts_legal_derived_features(cfg):
    al = module_b_allowlist(cfg)
    al.validate(
        ["iddq_v0h", "iddq_v24h", "iddq_d24", "iddq_reld24", "iddq_z_24h",
         "iddq_lotmed_0h", "xp_leak_zd24"]
    )


@pytest.mark.parametrize("param", ["iddq", "leak", "tpd"])
def test_module_b_feature_set_contains_no_late_checkpoint(cfg, df, param):
    feats, names = build_module_b_features(df, cfg, param)
    module_b_allowlist(cfg).validate(names)


@pytest.mark.parametrize("param", ["iddq", "leak", "tpd"])
def test_module_b_features_are_numerically_independent_of_late_checkpoints(
    cfg, df, param
):
    """The strongest form of the check: perturb the 96 h and 168 h columns to
    nonsense and confirm not one Module B feature value moves."""
    feats_before, _ = build_module_b_features(df, cfg, param)
    poisoned = df.copy()
    for p in cfg.parameters:
        for h in (96, 168):
            poisoned[p.col(h)] = poisoned[p.col(h)] * -7.0 + 1234.5
    feats_after, _ = build_module_b_features(poisoned, cfg, param)
    assert feats_before.equals(feats_after), (
        "a Module B feature changed when a forbidden checkpoint changed"
    )


def test_module_b_model_prediction_is_independent_of_late_checkpoints(cfg, df):
    """End-to-end: the fitted model must predict the same on poisoned data.

    The exact guarantee is asserted on the feature matrix, which must be
    bit-identical. Predictions are compared to a float tolerance rather than
    with ``==`` because the forest is evaluated with ``n_jobs=-1`` and the
    threaded accumulation of tree outputs is not bitwise reproducible; the
    observed disagreement is ~1e-16 relative, which is float noise and not a
    dependence on the poisoned columns.
    """
    mb = ModuleB(cfg, "iddq").fit(df)
    feats_before = mb.features(df)[mb.feature_names].to_numpy()
    before = mb.predict(df).point

    poisoned = df.copy()
    for p in cfg.parameters:
        for h in (96, 168):
            poisoned[p.col(h)] = poisoned[p.col(h)] * 0.0 - 99.0

    feats_after = mb.features(poisoned)[mb.feature_names].to_numpy()
    after = mb.predict(poisoned).point

    # the real invariant: not one feature value moved
    assert np.array_equal(feats_before, feats_after)
    assert np.allclose(before, after, rtol=1e-12, atol=1e-10)


def test_baselines_also_respect_the_allowlist(cfg, df):
    """The three reported baselines must be computable at 24 h, or they are not
    baselines for this task."""
    poisoned = df.copy()
    for p in cfg.parameters:
        for h in (96, 168):
            poisoned[p.col(h)] = 0.0
    p = cfg.param("leak")
    for name, fn in BASELINES.items():
        a = fn(df, p, cfg)
        b = fn(poisoned, p, cfg)
        assert (a == b).all(), f"baseline {name} reads a forbidden checkpoint"


def test_target_is_the_terminal_checkpoint(cfg, df):
    mb = ModuleB(cfg, "tpd")
    assert (mb.target(df) == df[cfg.param("tpd").col(168)].to_numpy()).all()
