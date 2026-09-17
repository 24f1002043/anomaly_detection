"""Section 6 - Decision layer.

Two decision paths run at different times. They are kept separate, because
collapsing them destroys the value of early rejection.

    Path      Runs at   Evidence                              Output
    --------  --------  ------------------------------------  ---------------------
    Early     24 h      Module B only, from Value_0h and      CONTINUE or
                        Value_24h                             EARLY-REJECT
    Terminal  168 h     Static limits plus Module A over the   PASS, WATCH or
                        full checkpoint grid                  REJECT

The early path carries no persistence requirement and no confirmation from later
checkpoints, because none exist yet. This is the path that saves 144 h of chamber
occupancy per rejected part.

How WATCH is scored
    The official anomaly metric is binary. WATCH maps to flagged (positive) for
    scoring, never to pass. An intermediate state that quietly absorbed
    defective parts would manufacture false negatives, which is precisely the
    error the evaluation punishes hardest. The cost of this choice is yield loss,
    and it is reported honestly alongside the escape rate it buys.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from burnin.config import Config
from burnin.generator import static_screen_mask
from burnin.module_a import ModuleAResult

PASS, WATCH, REJECT = "PASS", "WATCH", "REJECT"
CONTINUE, EARLY_REJECT = "CONTINUE", "EARLY-REJECT"


@dataclass
class DecisionThresholds:
    """Both edges of the severity scale, stated explicitly.

    ``alert`` is the calibrated operating point of Section 8.1: at or above it a
    part stops being PASS. ``reject`` splits the flagged population into WATCH
    and REJECT. The binary detection metric keys off ``alert``, because WATCH
    counts as flagged.
    """

    alert: float
    reject: float

    def to_dict(self) -> dict:
        return {"alert": self.alert, "reject": self.reject}


def make_thresholds(
    operating_point: float,
    train_severity: np.ndarray,
    cfg: Config,
    calibration_feasible: bool = True,
) -> DecisionThresholds:
    """Derive both edges from the calibrated operating point.

    The alert edge IS the operating point of Section 8.1. It is never raised
    above it: raising the threshold flags fewer parts and therefore increases
    the escape rate, which is the one quantity the calibration exists to hold.
    ``severity_alert_quantile`` is only a fallback for the degenerate case where
    no threshold met the escape target at all.
    """
    ma = cfg.module_a
    alert = (
        float(operating_point)
        if calibration_feasible
        else float(np.quantile(train_severity, float(ma["severity_alert_quantile"])))
    )
    reject = float(
        max(alert, np.quantile(train_severity, float(ma["severity_reject_quantile"])))
    )
    return DecisionThresholds(alert=alert, reject=reject)


def terminal_decision(
    df: pd.DataFrame,
    result: ModuleAResult,
    thresholds: DecisionThresholds,
    cfg: Config,
) -> pd.DataFrame:
    """Terminal path at 168 h: static limits plus Module A over the full grid."""
    sev = result.severity.to_numpy()
    static_fail = static_screen_mask(df, cfg).to_numpy()
    lot_alarm = result.lot_alarm.to_numpy().astype(bool)
    n_fired = result.rule_fired.sum(axis=1).to_numpy()

    above_reject = sev >= thresholds.reject
    above_alert = sev >= thresholds.alert
    # "a single-parameter exceedance" means one parameter exceeding is already
    # enough to warrant WATCH -- not that exactly one must exceed. A part with
    # two or three parameters out is more suspect, not less.
    single_param_exceedance = n_fired >= 1

    state = np.full(len(df), PASS, dtype=object)
    # WATCH: severity in the alert band, or a single-parameter exceedance, or
    # membership of a lot under an integrity alarm.
    watch = above_alert | single_param_exceedance | lot_alarm
    state[watch] = WATCH
    # REJECT: static limit violated, or severity above the reject threshold.
    state[above_reject | static_fail] = REJECT

    reason = np.where(
        static_fail,
        "static_limit_violation",
        np.where(
            above_reject,
            "severity_above_reject_threshold",
            np.where(
                above_alert,
                "severity_in_alert_band",
                np.where(
                    single_param_exceedance,
                    "single_parameter_z_exceedance",
                    np.where(lot_alarm, "lot_integrity_alarm", "within_limits"),
                ),
            ),
        ),
    )
    return pd.DataFrame(
        {
            "terminal_state": state,
            "terminal_reason": reason,
            "severity": sev,
            "responsible_param": result.responsible_param.to_numpy(),
            "static_fail": static_fail,
            "lot_integrity_alarm": lot_alarm,
            "n_params_rule_fired": n_fired,
        },
        index=df.index,
    )


def early_decision(early: pd.DataFrame) -> pd.DataFrame:
    """Early path at 24 h: Module B only. CONTINUE or EARLY-REJECT."""
    state = np.where(early["early_reject"].to_numpy(), EARLY_REJECT, CONTINUE)
    return pd.DataFrame(
        {
            "early_state": state,
            "early_reason": early["early_reason"].to_numpy(),
            "early_driver_param": early["early_driver_param"].to_numpy(),
            "early_worst_margin": early["early_worst_margin"].to_numpy(),
        },
        index=early.index,
    )


def combine(terminal: pd.DataFrame, early: pd.DataFrame) -> pd.DataFrame:
    """Final disposition of a component across both paths.

    A part early-rejected at 24 h never reaches 168 h, so its final disposition
    is REJECT regardless of what the terminal path would have said. The terminal
    columns are retained for the counterfactual reported in the ablation ladder.
    """
    out = terminal.join(early)
    early_rej = out["early_state"].to_numpy() == EARLY_REJECT
    final = out["terminal_state"].to_numpy().copy()
    final[early_rej] = REJECT
    out["final_state"] = final
    out["final_reason"] = np.where(
        early_rej, "module_b_early_rejection_at_24h", out["terminal_reason"].to_numpy()
    )
    # Binary label used by the official metric: WATCH counts as flagged.
    out["flagged"] = out["final_state"] != PASS
    out["rejected"] = out["final_state"] == REJECT
    out["watched"] = out["final_state"] == WATCH
    return out


def disposition_counts(decisions: pd.DataFrame, labels: pd.Series) -> dict:
    """Disposition breakdown against ground truth, for the results tables."""
    y = labels.to_numpy().astype(bool)
    out: dict[str, dict] = {}
    for state in (PASS, WATCH, REJECT):
        m = (decisions["final_state"] == state).to_numpy()
        out[state] = {
            "n": int(m.sum()),
            "share": float(m.mean()),
            "defective": int((m & y).sum()),
            "healthy": int((m & ~y).sum()),
        }
    out["flagged_is_positive"] = True
    return out
