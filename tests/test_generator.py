"""Section 3 - the generator must earn the right to be reported on.

These tests are the acceptance criterion of Section 3.3 expressed as code. If
they fail, no model result from this repository means anything.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from burnin.config import Config, load_config
from burnin.generator import (
    CLS_BENIGN_HIGH,
    CLS_DEFECT_LOWSTART,
    BurnInGenerator,
    acceptance_report,
    static_screen_mask,
)


@pytest.fixture(scope="module")
def cfg() -> Config:
    base = load_config()
    raw = copy.deepcopy(base.raw)
    raw["run"]["n_lots"] = 160  # smaller run; the structure is what is tested
    return Config(raw=raw, path=base.path)


@pytest.fixture(scope="module")
def dataset(cfg):
    return BurnInGenerator(cfg).generate()


# --------------------------------------------------------------- Section 3.1
def test_arrhenius_factors_match_specification():
    """AF = exp[(Ea/k)(1/T_use - 1/T_stress)] must give ~31 and ~9.4e2."""
    c = load_config()
    assert c.af_healthy == pytest.approx(31.0, rel=0.02)
    assert c.af_defect == pytest.approx(9.4e2, rel=0.02)


def test_defective_parts_accelerate_about_thirty_times_faster():
    """The physical reason burn-in works: a latent defect drifts roughly thirty
    times faster under stress while being unremarkable at ambient."""
    c = load_config()
    assert c.af_defect / c.af_healthy == pytest.approx(30.6, rel=0.05)


# --------------------------------------------------------------- Section 3.3
def test_static_limit_catches_only_a_clear_minority(dataset, cfg):
    """Generator acceptance criterion. If a tuned static threshold performs
    close to the models, the data is too easy and the demonstration is
    worthless."""
    rep = acceptance_report(dataset.frame, cfg)
    assert rep["static_limit_recall"] < 0.5, (
        f"static limit catches {rep['static_limit_recall']:.1%} of defects, "
        "which is not a clear minority -- the generator is too easy"
    )


def test_most_defects_finish_under_the_datasheet_limit(dataset, cfg):
    """At least 60 % of defective parts must have Value_168h below the maximum.
    These are the escapes the problem statement is about."""
    rep = acceptance_report(dataset.frame, cfg)
    target = float(cfg.difficulty["latent_under_limit_target"])
    assert rep["defects_under_limit_at_168h"] >= target


def test_benign_high_baseline_parts_are_high_but_flat(dataset, cfg):
    """A rule that flags on absolute level must be punished by this population."""
    df = dataset.frame
    p = cfg.param("iddq")
    benign = df["defect_class"] == CLS_BENIGN_HIGH
    healthy = df["defect_class"] == "healthy"
    # higher level ...
    assert df.loc[benign, p.col(0)].median() > df.loc[healthy, p.col(0)].median()
    # ... but less relative movement
    def rel_drift(m):
        return (
            (df.loc[m, p.col(168)] - df.loc[m, p.col(0)]) / df.loc[m, p.col(0)]
        ).median()

    assert rel_drift(benign) < rel_drift(healthy)


def test_low_start_defects_are_unremarkable_at_24h(dataset, cfg):
    """These are the hard cases for Module B and set the realistic ceiling on
    early rejection: depressed at 0 h, only mildly elevated at 24 h."""
    df = dataset.frame
    p = cfg.param("iddq")
    low = df["defect_class"] == CLS_DEFECT_LOWSTART
    std = df["defect_class"] == "defect_standard"
    healthy = df["defect_class"] == "healthy"

    # absolute level at 24 h sits at or below the healthy population
    assert df.loc[low, p.col(24)].median() <= df.loc[healthy, p.col(24)].median()
    # and their 24 h movement is milder than a standard defect's
    def rel24(m):
        return (
            (df.loc[m, p.col(24)] - df.loc[m, p.col(0)]) / df.loc[m, p.col(0)]
        ).median()

    assert rel24(low) < rel24(std)
    # but still above healthy: the signal is in the movement, not the level
    assert rel24(low) > rel24(healthy)


def test_contaminated_lots_exist_and_are_heavily_defective(dataset, cfg):
    df = dataset.frame
    prev = df.groupby("lot_id")["is_defective"].mean()
    contaminated = dataset.meta["contaminated_lots"]
    assert len(contaminated) >= 1
    for lot in contaminated:
        assert prev[lot] > 0.10, f"{lot} is not meaningfully contaminated"


# --------------------------------------------------------------- Section 3.2
def test_within_lot_spread_is_narrower_than_between_lot_spread(dataset, cfg):
    """Required so a part can be a severe lot-relative outlier while sitting
    mid-range against the global population."""
    df = dataset.frame[dataset.frame["defect_class"] == "healthy"]
    for p in cfg.parameters:
        col = p.col(0)
        within = df.groupby("lot_id")[col].std().median()
        between = df.groupby("lot_id")[col].median().std()
        assert within < between, f"{p.name}: within-lot spread is not narrower"


def test_one_defect_draw_per_component_not_per_parameter(dataset, cfg):
    """is_defective is a component-level property. Cross-parameter correlation
    must therefore be visible in the drift of defective parts."""
    df = dataset.frame
    y = df["is_defective"].to_numpy()
    rel = {}
    for p in cfg.parameters:
        rel[p.name] = (
            (df[p.col(168)] - df[p.col(0)]) / df[p.col(0)]
        ).to_numpy()
    names = [p.name for p in cfg.parameters]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = np.corrcoef(rel[names[i]][y], rel[names[j]][y])[0, 1]
            assert c > 0.3, (
                f"defective drift in {names[i]} and {names[j]} is uncorrelated "
                f"(r={c:.2f}); a single defect site should move several parameters"
            )


def test_chamber_common_mode_cancels_under_lot_relative_statistics(dataset, cfg):
    """A shared per-checkpoint chamber offset must survive an absolute-drift
    rule and cancel under a lot-relative one."""
    df = dataset.frame[dataset.frame["defect_class"] == "healthy"]
    p = cfg.param("iddq")
    col = p.col(168)
    # absolute level varies between chamber loads ...
    by_chamber = df.groupby("chamber_id")[col].median()
    assert by_chamber.std() > 0
    # ... but the lot-relative deviation is centred for every chamber
    med = df.groupby("lot_id")[col].transform("median")
    dev = (df[col] - med).groupby(df["chamber_id"]).median()
    assert dev.abs().max() < by_chamber.std()


def test_no_part_enters_burn_in_already_out_of_spec(dataset, cfg):
    """Incoming inspection screens out-of-spec parts before the chamber.

    The cap is applied to the part's true V_0. The 0 h burn-in reading is an
    independent measurement of that part, carrying chamber common-mode and
    measurement noise, so in the far tail a reading can still land above the
    limit -- exactly as it would on a real line, where incoming inspection and
    the 0 h checkpoint are two different measurements. What must not happen is a
    population of pre-failed parts, so the rate is bounded rather than zero.
    """
    df = dataset.frame
    for p in cfg.parameters:
        rate = float((df[p.col(0)] > p.limit).mean())
        assert rate < 0.001, f"{p.name}: {rate:.4%} of parts read over-limit at 0 h"


def test_static_screen_has_non_zero_but_small_yield_loss(dataset, cfg):
    """Marginal good parts must exist: a screen that never fails a good part is
    not modelling a real burn-in line."""
    rep = acceptance_report(dataset.frame, cfg)
    assert 0.0 < rep["static_limit_false_reject_rate"] < 0.05


def test_generator_is_reproducible(cfg):
    a = BurnInGenerator(cfg, seed=4242).generate().frame
    b = BurnInGenerator(cfg, seed=4242).generate().frame
    p = cfg.param("iddq")
    assert np.array_equal(a[p.col(168)].to_numpy(), b[p.col(168)].to_numpy())


def test_static_screen_is_monotone_in_checkpoints(dataset, cfg):
    """Screening the full grid can only reject more parts than screening 0 h."""
    df = dataset.frame
    early = static_screen_mask(df, cfg, hours=[0]).sum()
    full = static_screen_mask(df, cfg).sum()
    assert full >= early
