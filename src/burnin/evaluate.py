"""Section 8.4 - the ablation ladder, and the end-to-end evaluation driver.

    static limit
      -> lot-relative robust z-rule alone
      -> Isolation Forest alone
      -> combined detector with a cost-calibrated threshold
      -> combined detector plus Module B early rejection

Each rung is reported at the same operating point, so the contribution of each
component is visible rather than asserted. "Same operating point" is applied as
the same selection procedure on the same validation lots against the same
declared escape target, since the rungs do not share a score scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from burnin.config import Config
from burnin.generator import static_screen_mask
from burnin.metrics import (
    DetectionMetrics,
    cost_optimal_threshold,
    detection_metrics,
    select_operating_point,
)
from burnin.module_a import ModuleAResult


@dataclass
class Rung:
    name: str
    description: str
    valid_score: np.ndarray
    test_score: np.ndarray
    fixed_threshold: float | None = None   # for rules with no tunable threshold
    # A binary rule OR-ed with the threshold decision. Used by the final rung,
    # where Module B early rejection is a decision and not a ranking.
    overlay_valid: np.ndarray | None = None
    overlay_test: np.ndarray | None = None
    is_decision_rule: bool = False
    metrics: DetectionMetrics | None = None
    threshold: float = float("nan")
    calibration: dict = field(default_factory=dict)
    cost_threshold: float | None = None
    cost_metrics: DetectionMetrics | None = None


def build_ladder(
    cfg: Config,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    res_valid: ModuleAResult,
    res_test: ModuleAResult,
    early_valid: np.ndarray,
    early_test: np.ndarray,
) -> list[Rung]:
    """Assemble the five rungs. Scores are already on a common [0, 1] severity
    scale within each rung, but not across rungs, which is why each rung gets its
    own calibration against the same escape target."""
    rungs: list[Rung] = [
        Rung(
            name="static_limit_only",
            description="Absolute datasheet pass/fail limits at every checkpoint",
            valid_score=static_screen_mask(valid_df, cfg).to_numpy().astype(float),
            test_score=static_screen_mask(test_df, cfg).to_numpy().astype(float),
            fixed_threshold=0.5,
        ),
        Rung(
            name="lot_relative_z_rule",
            description="R1 alone: lot-relative robust z at any checkpoint, or z of total drift",
            valid_score=res_valid.rule_severity.max(axis=1).to_numpy(),
            test_score=res_test.rule_severity.max(axis=1).to_numpy(),
        ),
        Rung(
            name="isolation_forest",
            description="R2 alone: Isolation Forest over the full Section 4.1 feature vector",
            valid_score=res_valid.forest_severity.max(axis=1).to_numpy(),
            test_score=res_test.forest_severity.max(axis=1).to_numpy(),
        ),
        Rung(
            name="combined_detector",
            description="R1 and R2 under a maximum-severity union, cost-calibrated threshold",
            valid_score=res_valid.severity.to_numpy(),
            test_score=res_test.severity.to_numpy(),
        ),
    ]

    # Rung 5: the combined detector plus the Module B early rejection at 24 h.
    # This rung is a DECISION, not a ranking: a severity threshold OR-ed with a
    # binary early-reject rule. Folding the early rejection into the score (for
    # instance by saturating those parts to severity 1.0) would put every early
    # false positive at the top of the ranking and destroy PR-AUC, reporting a
    # collapse that is an artefact of the encoding rather than a property of the
    # system. The rung therefore carries an overlay and reports PR-AUC as NaN.
    rungs.append(
        Rung(
            name="combined_plus_module_b_early",
            description="Combined detector plus Module B early rejection at 24 h",
            valid_score=res_valid.severity.to_numpy(),
            test_score=res_test.severity.to_numpy(),
            overlay_valid=early_valid.astype(bool),
            overlay_test=early_test.astype(bool),
            is_decision_rule=True,
        )
    )
    return rungs


def evaluate_ladder(
    rungs: list[Rung],
    y_valid: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
) -> list[Rung]:
    """Calibrate each rung on validation lots, then report it on test lots."""
    for r in rungs:
        if r.fixed_threshold is not None:
            r.threshold = float(r.fixed_threshold)
            r.calibration = {
                "feasible": True,
                "note": "fixed engineering rule; no tunable threshold",
            }
        else:
            vs = r.valid_score
            if r.overlay_valid is not None:
                # calibrate on what the rung actually decides: parts the overlay
                # already rejects cannot escape, whatever the threshold is
                vs = np.where(r.overlay_valid, 1.0, vs)
            thr, info = select_operating_point(y_valid, vs, cfg)
            r.threshold, r.calibration = float(thr), info

        flags = None
        pr_auc = None
        if r.is_decision_rule:
            flags = (r.test_score >= r.threshold) | (
                r.overlay_test if r.overlay_test is not None else False
            )
            pr_auc = float("nan")  # no single continuous score behind this rung
        r.metrics = detection_metrics(
            y_test, r.test_score, r.threshold, cfg, pr_auc=pr_auc, flags=flags
        )

        if r.fixed_threshold is None and not r.is_decision_rule:
            ct, _ = cost_optimal_threshold(y_valid, r.valid_score, cfg)
            r.cost_threshold = float(ct)
            r.cost_metrics = detection_metrics(y_test, r.test_score, ct, cfg)
    return rungs


def ladder_frame(rungs: list[Rung]) -> pd.DataFrame:
    """Section 9.1 results table."""
    rows = []
    for r in rungs:
        m = r.metrics
        assert m is not None
        rows.append(
            {
                "configuration": r.name,
                "description": r.description,
                "operating_point": round(r.threshold, 6),
                "escape_rate": round(m.escape_rate, 4),
                "recall": round(m.recall, 4),
                "f_beta_2": round(m.f_beta, 4),
                "pr_auc": ("n/a (decision rule)" if r.is_decision_rule
                           else round(m.pr_auc, 4)),
                "precision": round(m.precision, 4),
                "yield_loss": round(m.false_reject_rate, 4),
                "cost": round(m.cost, 1),
                "cost_per_1k_parts": round(m.cost_per_1k, 2),
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
                "tn": m.tn,
                "calibration_feasible": bool(r.calibration.get("feasible", True)),
            }
        )
    return pd.DataFrame(rows)


def ladder_cost_frame(rungs: list[Rung]) -> pd.DataFrame:
    """The same ladder at each rung's cost-minimising threshold, so the 20:1
    cost calibration is visible rather than asserted."""
    rows = []
    for r in rungs:
        if r.cost_metrics is None:
            continue
        m = r.cost_metrics
        rows.append(
            {
                "configuration": r.name,
                "cost_optimal_threshold": round(float(r.cost_threshold or 0.0), 6),
                "escape_rate": round(m.escape_rate, 4),
                "recall": round(m.recall, 4),
                "f_beta_2": round(m.f_beta, 4),
                "yield_loss": round(m.false_reject_rate, 4),
                "cost_per_1k_parts": round(m.cost_per_1k, 2),
            }
        )
    return pd.DataFrame(rows)


def subgroup_breakdown(
    y_true: np.ndarray,
    flagged: np.ndarray,
    defect_class: pd.Series,
    cfg: Config,
) -> pd.DataFrame:
    """Recall broken out by the Section 3.3 difficulty populations.

    This is where the difficulty controls earn their keep: a headline recall that
    hides a low-start fast-drift recall of near zero would be misleading.
    """
    rows = []
    dc = defect_class.to_numpy()
    for cls in pd.unique(dc):
        m = dc == cls
        y = y_true[m]
        f = flagged[m]
        if y.any():
            rows.append(
                {
                    "population": cls,
                    "n": int(m.sum()),
                    "n_defective": int(y.sum()),
                    "recall": round(float((f & y).sum() / y.sum()), 4),
                    "escape_rate": round(float((~f & y).sum() / y.sum()), 4),
                }
            )
        else:
            rows.append(
                {
                    "population": cls,
                    "n": int(m.sum()),
                    "n_defective": 0,
                    "recall": None,
                    "false_flag_rate": round(float(f.mean()), 4),
                }
            )
    return pd.DataFrame(rows)


def latency_note(cfg: Config) -> dict:
    """Deployment arithmetic quoted alongside the results."""
    return {
        "chamber_hours_per_part_full_soak": float(cfg.terminal_hour),
        "chamber_hours_saved_per_early_reject": float(
            cfg.evaluation["chamber_hours_saved_per_part"]
        ),
        "early_decision_point_h": float(cfg.early_allowlist[-1]),
    }
