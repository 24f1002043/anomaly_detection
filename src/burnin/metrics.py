"""Section 8 - Evaluation protocol.

Built directly on the three metrics named in the problem statement, in the order
they are named: anomaly detection (false-negative weighted), drift prediction
accuracy, and explainability.

A missed defective part is catastrophic and a wrongly rejected good part is
merely expensive. The evaluation encodes that asymmetry explicitly instead of
optimising a balanced score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from burnin.config import Config


# ------------------------------------------------------- 8.1 anomaly detection
@dataclass
class DetectionMetrics:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    escape_rate: float          # FN / (TP + FN)  <- headline figure
    recall: float
    precision: float
    f_beta: float               # beta = 2, recall weighted over precision
    false_reject_rate: float    # yield loss
    cost: float                 # C_FN * FN + C_FP * FP
    cost_per_1k: float
    pr_auc: float               # PR-AUC, not ROC-AUC; prevalence is 1-3 %
    n: int
    n_positive: int
    prevalence: float

    def to_dict(self) -> dict:
        return asdict(self)


def f_beta_score(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def detection_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    cfg: Config,
    pr_auc: float | None = None,
    flags: np.ndarray | None = None,
) -> DetectionMetrics:
    """Confusion-derived metrics at one stated operating point.

    Every reported figure names its operating point. A metric quoted without one
    is not reported (Section 8.1).

    ``flags`` overrides the thresholded decision. It exists for configurations
    that are a decision rather than a ranking -- the combined detector plus
    Module B early rejection is a threshold OR-ed with a binary rule, and there
    is no single continuous score behind it. Such configurations report
    PR-AUC as NaN rather than inventing one.
    """
    y = np.asarray(y_true).astype(bool)
    flag = (
        np.asarray(score) >= threshold if flags is None
        else np.asarray(flags).astype(bool)
    )
    tp = int((flag & y).sum())
    fp = int((flag & ~y).sum())
    fn = int((~flag & y).sum())
    tn = int((~flag & ~y).sum())

    n_pos = tp + fn
    recall = tp / n_pos if n_pos else 0.0
    escape = fn / n_pos if n_pos else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    ev = cfg.evaluation
    c_fn, c_fp = float(ev["cost_fn"]), float(ev["cost_fp"])
    cost = c_fn * fn + c_fp * fp
    if pr_auc is None:
        pr_auc = (
            float(average_precision_score(y, score)) if y.any() and (~y).any() else float("nan")
        )

    return DetectionMetrics(
        threshold=float(threshold),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        escape_rate=float(escape),
        recall=float(recall),
        precision=float(precision),
        f_beta=float(f_beta_score(precision, recall, float(ev["f_beta"]))),
        false_reject_rate=float(fp / (fp + tn)) if (fp + tn) else 0.0,
        cost=float(cost),
        cost_per_1k=float(1000.0 * cost / max(len(y), 1)),
        pr_auc=float(pr_auc),
        n=int(len(y)),
        n_positive=int(n_pos),
        prevalence=float(y.mean()) if len(y) else 0.0,
    )


def select_operating_point(
    y_true: np.ndarray,
    score: np.ndarray,
    cfg: Config,
    escape_target: float | None = None,
) -> tuple[float, dict]:
    """Section 8.1, operating point selection.

    The threshold is chosen as the lowest severity that holds the escape rate at
    or below a declared target on validation lots. The procedure is a downward
    scan: start at the top of the severity scale and descend until the escape
    target is met, then stop. The stopping point is the largest threshold whose
    escape rate is still within target, so the escape constraint binds and the
    yield loss it costs is reported as a consequence rather than minimised as a
    competing objective.
    """
    ev = cfg.evaluation
    target = float(ev["escape_rate_target"]) if escape_target is None else float(escape_target)
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    if not y.any():
        return float(np.max(s)), {"feasible": False, "reason": "no positives"}

    # Candidate thresholds: the distinct scores of the positive class are the
    # only points at which the escape rate changes.
    cand = np.unique(s[y])
    cand = np.concatenate([cand, [np.nextafter(cand.max(), np.inf)]])
    n_pos = int(y.sum())
    # escape(t) = fraction of positives with score < t
    escapes = np.searchsorted(np.sort(s[y]), cand, side="left") / n_pos
    feasible = cand[escapes <= target]
    if len(feasible) == 0:
        chosen = float(cand.min())
        info = {
            "feasible": False,
            "target": target,
            "best_escape": float(escapes.min()),
            "reason": "target unreachable at any threshold",
        }
    else:
        # escape(t) is non-decreasing in t, so the feasible set is a lower
        # interval. Its maximum is where the downward scan stops.
        chosen = float(feasible.max())
        info = {"feasible": True, "target": target}
    return chosen, info


def cost_optimal_threshold(
    y_true: np.ndarray, score: np.ndarray, cfg: Config
) -> tuple[float, float]:
    """Threshold minimising C_FN*FN + C_FP*FP. Reported alongside the declared
    operating point so the cost calibration is visible rather than asserted."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype=float)
    ev = cfg.evaluation
    c_fn, c_fp = float(ev["cost_fn"]), float(ev["cost_fp"])
    order = np.argsort(-s)
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    fn = int(y.sum()) - tp
    cost = c_fn * fn + c_fp * fp
    i = int(np.argmin(cost))
    return float(ss[i]), float(cost[i])


# ------------------------------------------------- 8.2 drift prediction accuracy
@dataclass
class RegressionMetrics:
    parameter: str
    units: str
    mae: float
    normalised_mae: float       # MAE / lot interquartile range
    rmse: float
    median_ae: float
    r2: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def lot_iqr(df: pd.DataFrame, column: str) -> float:
    """Median across lots of each lot's interquartile range.

    Normalising by this makes the three parameters comparable. A single pooled
    figure across currents in uA and delays in ns is meaningless and is not
    reported (Section 8.2).
    """
    per_lot = df.groupby("lot_id")[column].apply(
        lambda s: float(s.quantile(0.75) - s.quantile(0.25))
    )
    return float(per_lot.median())


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, parameter: str, units: str, iqr: float
) -> RegressionMetrics:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return RegressionMetrics(
        parameter=parameter,
        units=units,
        mae=float(np.mean(np.abs(err))),
        normalised_mae=float(np.mean(np.abs(err)) / max(iqr, 1e-12)),
        rmse=float(np.sqrt(np.mean(err**2))),
        median_ae=float(np.median(np.abs(err))),
        r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        n=int(len(yt)),
    )


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict:
    """Empirical fraction of held-out parts falling inside the nominal interval,
    plus the mean interval width (Section 8.2)."""
    yt = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    inside = (yt >= lo) & (yt <= hi)
    return {
        "coverage": float(inside.mean()),
        "mean_width": float(np.mean(hi - lo)),
        "median_width": float(np.median(hi - lo)),
        "below_lower": float((yt < lo).mean()),
        "above_upper": float((yt > hi).mean()),
        "n": int(len(yt)),
    }


def early_rejection_metrics(
    y_true: np.ndarray, early_reject: np.ndarray, cfg: Config
) -> dict:
    """Section 8.2 - early-rejection escape rate at 24 h and chamber-hours saved.

    Chamber-hours saved is computed as 144 h multiplied by the number of
    correctly early-rejected parts: a part rejected at 24 h frees the remaining
    144 h of the 168 h soak.
    """
    y = np.asarray(y_true).astype(bool)
    r = np.asarray(early_reject).astype(bool)
    tp = int((r & y).sum())
    fp = int((r & ~y).sum())
    fn = int((~r & y).sum())
    tn = int((~r & ~y).sum())
    n_pos = tp + fn
    hours = float(cfg.evaluation["chamber_hours_saved_per_part"])
    return {
        "early_reject_rate": float(r.mean()),
        "early_recall": float(tp / n_pos) if n_pos else 0.0,
        "early_escape_rate": float(fn / n_pos) if n_pos else 0.0,
        "early_precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "early_false_reject_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "n_correctly_early_rejected": tp,
        "n_wrongly_early_rejected": fp,
        "chamber_hours_saved": float(hours * tp),
        "chamber_hours_lost_to_false_rejects": float(hours * fp),
        "net_chamber_hours_saved": float(hours * (tp + fp)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
