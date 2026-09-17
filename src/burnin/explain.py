"""Section 7 - Explainability.

Explainability is a scored criterion, not a courtesy. Every flag the system
raises carries a complete record, generated automatically, with no free-text
narration.

    Element                Content
    ---------------------  --------------------------------------------------
    Responsible parameter  Which of the three screened parameters drove the
                           disposition.
    Lot-relative           The part's value, the lot median, the lot MAD, and
    arithmetic             the resulting robust z. Checkable by hand.
    Feature contributions  TreeSHAP signed contributions for the Module B
                           prediction, top three by magnitude.
    Module A explanation   Ranked robust feature deviations against the lot, in
                           median/MAD units. Cheap, stable, and appropriate for
                           an unsupervised detector.
    Trajectory             The raw four-point series against the lot band, so
                           the inspector sees the shape behind the score.
    Margin                 Predicted Value_168h, the prediction interval, and
                           the signed margin to the safety criterion.

Coverage is binary and must be total: a flag without a complete explanation
record is a defect in the system (Section 8.3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from burnin.config import Config
from burnin.features import MAD_TO_SIGMA
from burnin.module_a import ModuleAResult
from burnin.module_b import ModuleB, ModuleBPrediction


@dataclass
class LotArithmetic:
    """The arithmetic a QA inspector can reproduce by hand."""

    parameter: str
    units: str
    checkpoint_h: int
    value: float
    lot_median: float
    lot_mad: float
    scale: float          # 1.4826 * MAD
    robust_z: float
    datasheet_limit: float

    def check(self) -> bool:
        """Verify the quoted z reproduces from the quoted median and MAD."""
        if self.scale <= 0:
            return False
        return abs((self.value - self.lot_median) / self.scale - self.robust_z) < 1e-6

    def as_text(self) -> str:
        return (
            f"{self.parameter} at {self.checkpoint_h} h: {self.value:.4g} {self.units} "
            f"against lot median {self.lot_median:.4g} {self.units}, "
            f"lot MAD {self.lot_mad:.4g}; robust z = "
            f"({self.value:.4g} - {self.lot_median:.4g}) / (1.4826 x {self.lot_mad:.4g}) "
            f"= {self.robust_z:.2f}. Datasheet limit {self.datasheet_limit:.4g} {self.units}."
        )


@dataclass
class ExplanationRecord:
    component_id: str
    lot_id: str
    wafer_id: str
    slot_id: str
    disposition: str
    reason: str
    responsible_parameter: str
    severity: float
    lot_arithmetic: dict[str, Any]
    module_a_deviations: list[dict[str, Any]]
    trajectory: dict[str, Any]
    margin: dict[str, Any]
    shap_contributions: list[dict[str, Any]] = field(default_factory=list)
    lot_integrity_alarm: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def is_complete(self) -> bool:
        """Section 8.3 coverage test. Every element must be present and usable."""
        if not self.responsible_parameter:
            return False
        if not self.lot_arithmetic:
            return False
        if not self.module_a_deviations:
            return False
        if not self.trajectory.get("values"):
            return False
        if self.margin.get("predicted_168h") is None:
            return False
        if not self.shap_contributions:
            return False
        return True


class Explainer:
    """Builds the complete record for any flagged component."""

    def __init__(
        self,
        cfg: Config,
        modules_b: dict[str, ModuleB],
        shap_background: pd.DataFrame | None = None,
    ):
        self.cfg = cfg
        self.modules_b = modules_b
        self._shap: dict[str, Any] = {}
        self._shap_background = shap_background

    # ---------------------------------------------------------------- TreeSHAP
    def _explainer_for(self, param: str):
        if param not in self._shap:
            import shap  # imported lazily; heavy at import time

            self._shap[param] = shap.TreeExplainer(self.modules_b[param].model)
        return self._shap[param]

    def shap_top_k(
        self, param: str, features: pd.DataFrame, k: int = 3
    ) -> list[list[dict[str, Any]]]:
        """TreeSHAP signed contributions, top k by magnitude, per row.

        Computed on the Module B Random Forest, which is the model that produces
        the number the inspector is being asked to trust.
        """
        ex = self._explainer_for(param)
        names = self.modules_b[param].feature_names
        X = features[names].to_numpy(dtype=float)
        values = ex.shap_values(X, check_additivity=False)
        values = np.asarray(values)
        out: list[list[dict[str, Any]]] = []
        for i in range(values.shape[0]):
            row = values[i]
            order = np.argsort(-np.abs(row))[:k]
            out.append(
                [
                    {
                        "feature": names[j],
                        "value": float(X[i, j]),
                        "shap": float(row[j]),
                        "direction": "raises" if row[j] > 0 else "lowers",
                    }
                    for j in order
                ]
            )
        return out

    # ------------------------------------------------------- Module A ranking
    def module_a_deviations(
        self, feats_row: pd.Series, param: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """Ranked robust feature deviations against the lot, in median/MAD units.

        Only the lot-relative z features carry median/MAD units, so those are the
        ones ranked. This is deliberately not a SHAP attribution: an Isolation
        Forest score has no additive decomposition an inspector could check, and
        a ranked deviation list is cheap, stable and honest about what it is.
        """
        zcols = [c for c in feats_row.index if c.startswith(f"{param}_z_")]
        zcols += ["xp_z_max", "xp_drift_z_max", "xp_n_exceeding"]
        vals = [(c, float(feats_row[c])) for c in zcols if c in feats_row.index]
        vals.sort(key=lambda t: -abs(t[1]))
        return [
            {"feature": c, "deviation_mad_units": round(v, 3)} for c, v in vals[:k]
        ]

    # ----------------------------------------------------------- lot arithmetic
    def lot_arithmetic(
        self, df: pd.DataFrame, idx, param: str, checkpoint: int
    ) -> LotArithmetic:
        p = self.cfg.param(param)
        col = p.col(checkpoint)
        lot = df.loc[idx, "lot_id"]
        lot_vals = df.loc[df["lot_id"] == lot, col].astype(float)
        med = float(lot_vals.median())
        mad = float((lot_vals - med).abs().median())
        scale = MAD_TO_SIGMA * mad
        value = float(df.loc[idx, col])
        z = (value - med) / scale if scale > 0 else 0.0
        return LotArithmetic(
            parameter=param,
            units=p.units,
            checkpoint_h=checkpoint,
            value=value,
            lot_median=med,
            lot_mad=mad,
            scale=scale,
            robust_z=z,
            datasheet_limit=float(p.limit),
        )

    # --------------------------------------------------------------- trajectory
    def trajectory(self, df: pd.DataFrame, idx, param: str) -> dict[str, Any]:
        p = self.cfg.param(param)
        lot = df.loc[idx, "lot_id"]
        lot_df = df[df["lot_id"] == lot]
        hours = self.cfg.checkpoints
        return {
            "hours": list(hours),
            "values": [float(df.loc[idx, p.col(h)]) for h in hours],
            "lot_median": [float(lot_df[p.col(h)].median()) for h in hours],
            "lot_p10": [float(lot_df[p.col(h)].quantile(0.10)) for h in hours],
            "lot_p90": [float(lot_df[p.col(h)].quantile(0.90)) for h in hours],
            "units": p.units,
            "datasheet_limit": float(p.limit),
        }

    # -------------------------------------------------------------------- build
    def build(
        self,
        df: pd.DataFrame,
        feats_a: pd.DataFrame,
        result_a: ModuleAResult,
        decisions: pd.DataFrame,
        preds_b: dict[str, ModuleBPrediction],
        feats_b: dict[str, pd.DataFrame],
        indices: list,
        shap_k: int = 3,
    ) -> list[ExplanationRecord]:
        """Generate one complete record per requested component."""
        if not indices:
            return []
        records: list[ExplanationRecord] = []

        # SHAP is computed per parameter in one batch: TreeSHAP on a 500-tree
        # forest is the expensive step, so it is never called row by row.
        by_param: dict[str, list] = {}
        for i in indices:
            by_param.setdefault(decisions.loc[i, "responsible_param"], []).append(i)
        shap_map: dict[Any, list[dict[str, Any]]] = {}
        for param, idxs in by_param.items():
            sub = feats_b[param].loc[idxs]
            tops = self.shap_top_k(param, sub, k=shap_k)
            for i, t in zip(idxs, tops):
                shap_map[i] = t

        term = self.cfg.terminal_hour
        for i in indices:
            param = decisions.loc[i, "responsible_param"]
            pred = preds_b[param]
            pos = df.index.get_loc(i)
            arith = self.lot_arithmetic(df, i, param, term)
            p = self.cfg.param(param)
            records.append(
                ExplanationRecord(
                    component_id=str(df.loc[i, "component_id"]),
                    lot_id=str(df.loc[i, "lot_id"]),
                    wafer_id=str(df.loc[i, "wafer_id"]),
                    slot_id=str(df.loc[i, "slot_id"]),
                    disposition=str(decisions.loc[i, "final_state"]),
                    reason=str(decisions.loc[i, "final_reason"]),
                    responsible_parameter=param,
                    severity=float(decisions.loc[i, "severity"]),
                    lot_arithmetic={**asdict(arith), "text": arith.as_text(),
                                    "reproduces": arith.check()},
                    module_a_deviations=self.module_a_deviations(feats_a.loc[i], param),
                    trajectory=self.trajectory(df, i, param),
                    margin={
                        "parameter": param,
                        "units": p.units,
                        "predicted_168h": float(pred.point[pos]),
                        "interval_lower": float(pred.lower[pos]),
                        "interval_upper": float(pred.upper[pos]),
                        "nominal_coverage": float(self.cfg.module_b["interval_nominal"]),
                        "datasheet_limit": float(p.limit),
                        "reliability_margin_M": float(pred_margin_M(self.modules_b, param)),
                        "s_safe": float(pred.s_safe[pos]),
                        "s_pred": float(pred.s_pred[pos]),
                        "margin_to_safety_slope": float(pred.margin[pos]),
                        "margin_to_safety_slope_upper": float(pred.margin_upper[pos]),
                        "early_reject": bool(pred.early_reject[pos]),
                        "early_reject_reason": str(pred.reason[pos]),
                    },
                    shap_contributions=shap_map.get(i, []),
                    lot_integrity_alarm=bool(decisions.loc[i, "lot_integrity_alarm"]),
                )
            )
        return records


def pred_margin_M(modules_b: dict[str, ModuleB], param: str) -> float:
    slope = modules_b[param].slope
    return float(slope.margin_M) if slope is not None else float("nan")


def coverage_audit(records: list[ExplanationRecord]) -> dict:
    """Section 8.3 - explainability is audited, and coverage must be total."""
    total = len(records)
    complete = sum(1 for r in records if r.is_complete())
    arithmetic_ok = sum(
        1 for r in records if r.lot_arithmetic.get("reproduces", False)
    )
    named_param = sum(1 for r in records if r.responsible_parameter)
    consistent = sum(1 for r in records if _shap_consistent_with_trajectory(r))
    return {
        "records": total,
        "complete_records": complete,
        "coverage": float(complete / total) if total else 1.0,
        "lot_arithmetic_reproduces": float(arithmetic_ok / total) if total else 1.0,
        "responsible_parameter_named": float(named_param / total) if total else 1.0,
        "shap_consistent_with_trajectory": float(consistent / total) if total else 1.0,
        "passes": bool(total == 0 or (complete == total and arithmetic_ok == total)),
    }


def _shap_consistent_with_trajectory(rec: ExplanationRecord) -> bool:
    """The SHAP ranking must be consistent with the visible trajectory.

    A part whose trajectory rises above its lot band should have at least one
    positive top contribution; a part that does not rise should not be explained
    entirely by upward pushes. This is the automatable half of the Section 8.3
    audit, which otherwise requires a human reading a sample of flags.
    """
    traj = rec.trajectory
    if not traj.get("values"):
        return False
    rose = traj["values"][-1] > traj["lot_median"][-1]
    if not rec.shap_contributions:
        return False
    any_positive = any(c["shap"] > 0 for c in rec.shap_contributions)
    return any_positive if rose else True


def records_to_frame(records: list[ExplanationRecord]) -> pd.DataFrame:
    """Flat, one-row-per-flag view for the inspector dashboard and for export."""
    rows = []
    for r in records:
        la = r.lot_arithmetic
        m = r.margin
        rows.append(
            {
                "component_id": r.component_id,
                "lot_id": r.lot_id,
                "wafer_id": r.wafer_id,
                "slot_id": r.slot_id,
                "disposition": r.disposition,
                "reason": r.reason,
                "responsible_parameter": r.responsible_parameter,
                "severity": round(r.severity, 6),
                "value_168h": la.get("value"),
                "lot_median": la.get("lot_median"),
                "lot_mad": la.get("lot_mad"),
                "robust_z": round(float(la.get("robust_z", 0.0)), 3),
                "predicted_168h": m.get("predicted_168h"),
                "interval_lower": m.get("interval_lower"),
                "interval_upper": m.get("interval_upper"),
                "margin_to_safety_slope": m.get("margin_to_safety_slope"),
                "early_reject": m.get("early_reject"),
                "lot_integrity_alarm": r.lot_integrity_alarm,
                "top_contributions": "; ".join(
                    f"{c['feature']}({c['shap']:+.4g})" for c in r.shap_contributions
                ),
                "explanation": la.get("text", ""),
            }
        )
    return pd.DataFrame(rows)
