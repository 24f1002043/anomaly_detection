"""Section 4 - Module A, dynamic lot-relative outlier detection.

    four checkpoints per part -> interval-rate and lot-relative features
    -> robust z-rule + Isolation Forest -> severity score

Two detectors run in parallel (Section 4.2):

  R1  a deterministic robust z-rule. Per parameter, flag when the lot-relative
      robust z at any checkpoint, or the robust z of total drift, exceeds a
      calibrated critical value. Fully auditable, runs in microseconds,
      reproduces the 45 uA worked example exactly, and gives the QA inspector
      arithmetic they can check by hand.

  R2  an Isolation Forest fitted on the full feature vector, unsupervised, with
      contamination set to roughly two to three times the assumed defect
      prevalence so the model leans toward recall. This catches multivariate
      shapes no single z-score expresses.

They are combined under a MAXIMUM-SEVERITY UNION, not an intersection. Under an
asymmetric cost where a miss is catastrophic, requiring both detectors to agree
is the wrong operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from burnin.config import Config
from burnin.features import MAD_TO_SIGMA, _min_scale, _rate_floor


# --------------------------------------------------------------- lot integrity
@dataclass
class LotIntegrity:
    """Section 4.4 - lot contamination fallback.

    Median and MAD degrade as a lot becomes contaminated: a lot that is heavily
    defective redefines its own normal, and every bad part looks central. The
    pipeline therefore monitors lot-level dispersion against a historical band.
    If a lot's MAD exceeds a stated factor, lot-relative statistics are replaced
    by a pooled cross-lot baseline for that lot and a lot integrity alarm is
    raised. The alarm is itself a useful output: a whole suspect lot is a more
    important finding than a single suspect part.

    Implementation note, and a finding worth stating plainly. The specified
    trigger -- lot MAD exceeding a stated factor of a historical band -- does
    NOT fire at realistic contamination levels. MAD has a 50 % breakdown point,
    so a lot that is 15-20 % defective inflates its MAD by only 1.1-1.5 x, while
    clean lots already reach 1.18 x at their 99th percentile. There is no
    separating factor. The very robustness that makes MAD the right denominator
    for a lot-relative z-score makes it blind as a contamination detector.

    A tail-inflation index does separate them cleanly:

        TI = ( Q_q(rel drift) - median(rel drift) ) / ( 1.4826 * MAD )

    measured at q = 0.90. A clean lot sits near 1.4 (the normal-theory value);
    a 20 % contaminated lot reaches 8-60, because the defective sub-population
    lifts the upper decile far beyond anything the MAD predicts. Both triggers
    are computed, both are reported, and the alarm is their union.
    """

    alarmed_lots: set[str]
    lot_mad: pd.DataFrame            # lot x parameter, MAD of relative total drift
    tail_inflation: pd.DataFrame     # lot x parameter, (Q90 - Q50) / (1.4826*MAD)
    historical_band: dict[str, float]
    factor: float
    tail_factor: float
    mad_alarmed_lots: set[str] = field(default_factory=set)
    tail_alarmed_lots: set[str] = field(default_factory=set)
    pooled_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        out = self.lot_mad.add_prefix("mad_").join(
            self.tail_inflation.add_prefix("tail_")
        )
        out["mad_alarm"] = out.index.isin(self.mad_alarmed_lots)
        out["tail_alarm"] = out.index.isin(self.tail_alarmed_lots)
        out["lot_integrity_alarm"] = out.index.isin(self.alarmed_lots)
        return out.reset_index()


def assess_lot_integrity(
    df: pd.DataFrame, cfg: Config, historical_band: dict[str, float] | None = None
) -> LotIntegrity:
    """Flag lots whose internal dispersion is inflated beyond a stated band."""
    ma = cfg.module_a["lot_integrity"]
    factor = float(ma["mad_inflation_factor"])
    tail_factor = float(ma["tail_inflation_factor"])
    tail_q = float(ma["tail_quantile"])
    min_parts = int(ma["min_parts_for_lot_stats"])
    term = cfg.terminal_hour
    h0 = cfg.checkpoints[0]

    rows: dict[str, pd.Series] = {}
    tails: dict[str, pd.Series] = {}
    for p in cfg.parameters:
        v0 = df[p.col(h0)].astype(float)
        rel = (df[p.col(term)].astype(float) - v0) / v0.clip(lower=1e-9)
        grp = rel.groupby(df["lot_id"])
        med = grp.transform("median")
        mad = (rel - med).abs().groupby(df["lot_id"]).median()
        rows[p.name] = mad
        scale = (MAD_TO_SIGMA * mad).clip(lower=1e-9)
        tails[p.name] = (grp.quantile(tail_q) - grp.quantile(0.50)) / scale
    lot_mad = pd.DataFrame(rows)
    lot_mad.index.name = "lot_id"
    tail_inflation = pd.DataFrame(tails)
    tail_inflation.index.name = "lot_id"

    counts = df.groupby("lot_id").size()
    eligible = counts[counts >= min_parts].index

    band = (
        historical_band
        if historical_band is not None
        else {c: float(lot_mad.loc[eligible, c].median()) for c in lot_mad.columns}
    )

    mad_alarm = pd.Series(False, index=lot_mad.index)
    for c in lot_mad.columns:
        b = max(band[c], 1e-9)
        mad_alarm = mad_alarm | (lot_mad[c] > factor * b)
    tail_alarm = pd.Series(False, index=tail_inflation.index)
    for c in tail_inflation.columns:
        tail_alarm = tail_alarm | (tail_inflation[c] > tail_factor)

    mad_alarm.loc[~mad_alarm.index.isin(eligible)] = False
    tail_alarm.loc[~tail_alarm.index.isin(eligible)] = False
    alarm = mad_alarm | tail_alarm

    # pooled cross-lot baseline, computed from lots that are NOT alarmed
    clean = df[~df["lot_id"].isin(set(alarm[alarm].index))]
    pooled: dict[str, dict[str, float]] = {}
    for p in cfg.parameters:
        typical = float(df[p.col(h0)].median())
        mscale = _min_scale(p, typical)
        entry: dict[str, float] = {}
        for h in cfg.checkpoints:
            s = clean[p.col(h)].astype(float)
            m = float(s.median())
            entry[f"med_{h}h"] = m
            entry[f"scale_{h}h"] = max(
                MAD_TO_SIGMA * float((s - m).abs().median()), mscale
            )
        tot = clean[p.col(term)].astype(float) - clean[p.col(h0)].astype(float)
        m = float(tot.median())
        entry["med_total"] = m
        entry["scale_total"] = max(
            MAD_TO_SIGMA * float((tot - m).abs().median()), mscale
        )
        relt = tot / clean[p.col(h0)].astype(float).clip(lower=1e-9)
        m = float(relt.median())
        entry["med_total_rel"] = m
        entry["scale_total_rel"] = max(
            MAD_TO_SIGMA * float((relt - m).abs().median()), 1e-4
        )
        pooled[p.name] = entry

    return LotIntegrity(
        alarmed_lots=set(alarm[alarm].index),
        lot_mad=lot_mad,
        tail_inflation=tail_inflation,
        historical_band=band,
        factor=factor,
        tail_factor=tail_factor,
        mad_alarmed_lots=set(mad_alarm[mad_alarm].index),
        tail_alarmed_lots=set(tail_alarm[tail_alarm].index),
        pooled_stats=pooled,
    )


def apply_pooled_fallback(
    feats: pd.DataFrame, df: pd.DataFrame, cfg: Config, integrity: LotIntegrity
) -> pd.DataFrame:
    """Rewrite lot-relative features for alarmed lots using the pooled baseline."""
    if not integrity.alarmed_lots:
        return feats
    feats = feats.copy()
    mask = df["lot_id"].isin(integrity.alarmed_lots).to_numpy()
    if not mask.any():
        return feats
    term = cfg.terminal_hour
    h0 = cfg.checkpoints[0]
    for p in cfg.parameters:
        st = integrity.pooled_stats[p.name]
        for h in cfg.checkpoints:
            col = f"{p.name}_z_{h}h"
            v = df.loc[mask, p.col(h)].astype(float)
            feats.loc[mask, col] = (v - st[f"med_{h}h"]) / st[f"scale_{h}h"]
        tot = df.loc[mask, p.col(term)].astype(float) - df.loc[mask, p.col(h0)].astype(float)
        feats.loc[mask, f"{p.name}_z_total_drift"] = (
            tot - st["med_total"]
        ) / st["scale_total"]
        relt = tot / df.loc[mask, p.col(h0)].astype(float).clip(lower=1e-9)
        feats.loc[mask, f"{p.name}_z_total_drift_rel"] = (
            relt - st["med_total_rel"]
        ) / st["scale_total_rel"]
        zcols = [f"{p.name}_z_{h}h" for h in cfg.checkpoints]
        feats.loc[mask, f"{p.name}_z_level_max"] = feats.loc[mask, zcols].max(axis=1)

    # cross-parameter family must be rebuilt from the corrected z-scores
    thr = float(cfg.module_a["cross_param_z_threshold"])
    prim = pd.concat(
        [
            feats.loc[mask, [f"{p.name}_z_level_max", f"{p.name}_z_total_drift"]].max(axis=1)
            for p in cfg.parameters
        ],
        axis=1,
    )
    dmat = feats.loc[mask, [f"{p.name}_z_total_drift" for p in cfg.parameters]]
    feats.loc[mask, "xp_z_max"] = prim.max(axis=1)
    feats.loc[mask, "xp_z_mean"] = prim.mean(axis=1)
    feats.loc[mask, "xp_z_min"] = prim.min(axis=1)
    feats.loc[mask, "xp_drift_z_max"] = dmat.max(axis=1)
    feats.loc[mask, "xp_drift_z_mean"] = dmat.mean(axis=1)
    feats.loc[mask, "xp_n_exceeding"] = (prim > thr).sum(axis=1).astype(float)
    feats.loc[mask, "xp_n_drift_exceeding"] = (dmat > thr).sum(axis=1).astype(float)
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ------------------------------------------------------------------ severity map
class SeverityCalibrator:
    """Maps a raw detector score onto a common [0, 1] severity scale.

    Calibrated on the training population only, using the empirical CDF. Both
    detectors must live on the same scale before a maximum-severity union means
    anything.
    """

    def __init__(self, n_grid: int = 2048):
        self.n_grid = n_grid
        self.knots_: np.ndarray | None = None

    def fit(self, scores: np.ndarray) -> "SeverityCalibrator":
        q = np.linspace(0.0, 1.0, self.n_grid)
        self.knots_ = np.quantile(np.asarray(scores, dtype=float), q)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        assert self.knots_ is not None, "calibrator not fitted"
        pos = np.searchsorted(self.knots_, np.asarray(scores, dtype=float), side="right")
        return pos / float(len(self.knots_))


# ------------------------------------------------------------------- the module
@dataclass
class ModuleAResult:
    severity: pd.Series                   # component severity, [0, 1]
    responsible_param: pd.Series          # which parameter drove the disposition
    per_param_severity: pd.DataFrame
    rule_severity: pd.DataFrame           # R1, per parameter, [0, 1]
    forest_severity: pd.DataFrame         # R2, per parameter, [0, 1]
    primary_z: pd.DataFrame               # raw z units, for the audit record
    rule_fired: pd.DataFrame              # R1 boolean at the calibrated z_crit
    z_critical: float
    lot_alarm: pd.Series


class ModuleA:
    """Fit on unlabelled training parts; score any population."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.forests: dict[str, IsolationForest] = {}
        self.forest_cal: dict[str, SeverityCalibrator] = {}
        self.rule_cal: dict[str, SeverityCalibrator] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.z_critical: float = float(cfg.module_a["z_critical_grid"][2])

    # -- R1 ------------------------------------------------------------------
    @staticmethod
    def primary_z(feats: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        """Per-parameter R1 statistic: the worst of the lot-relative level z at
        any checkpoint and the robust z of total drift."""
        cols = {}
        for p in cfg.parameters:
            cols[p.name] = feats[
                [f"{p.name}_z_level_max", f"{p.name}_z_total_drift"]
            ].max(axis=1)
        return pd.DataFrame(cols, index=feats.index)

    # -- fit -----------------------------------------------------------------
    def fit(self, feats: pd.DataFrame, per_param: dict[str, list[str]]) -> "ModuleA":
        cfg = self.cfg
        isf = cfg.module_a["isolation_forest"]
        contamination = min(
            0.5,
            float(isf["contamination_multiplier"])
            * float(cfg.population["defect_prevalence"]),
        )
        pz = self.primary_z(feats, cfg)
        for p in cfg.parameters:
            names = per_param[p.name]
            self.feature_names[p.name] = names

            model = IsolationForest(
                n_estimators=int(isf["n_estimators"]),
                max_samples=int(isf["max_samples"]),
                contamination=contamination,
                random_state=int(isf["random_state"]),
                n_jobs=-1,
            )
            model.fit(feats[names].to_numpy())
            self.forests[p.name] = model
            raw = -model.score_samples(feats[names].to_numpy())
            self.forest_cal[p.name] = SeverityCalibrator().fit(raw)
            self.rule_cal[p.name] = SeverityCalibrator().fit(pz[p.name].to_numpy())
        return self

    # -- R1 critical value ----------------------------------------------------
    def calibrate_z_critical(
        self, feats: pd.DataFrame, y_true: np.ndarray
    ) -> dict:
        """Calibrate the R1 critical value on labelled validation lots.

        Section 4.2 calls for a *calibrated* critical value, not a textbook 3
        sigma. The grid in the config is swept and the value minimising
        C_FN * FN + C_FP * FP is kept, which is the same 20:1 asymmetry the rest
        of the evaluation uses. This matters beyond R1 itself: the WATCH state of
        Section 6 fires on a single-parameter exceedance, so an uncalibrated
        critical value would flood the queue with healthy parts.
        """
        cfg = self.cfg
        ev = cfg.evaluation
        c_fn, c_fp = float(ev["cost_fn"]), float(ev["cost_fp"])
        y = np.asarray(y_true).astype(bool)
        pz = self.primary_z(feats, cfg)
        worst = pz.max(axis=1).to_numpy()

        trace = []
        best = (np.inf, float(cfg.module_a["z_critical_grid"][0]))
        for z in cfg.module_a["z_critical_grid"]:
            flag = worst > float(z)
            fn = int((~flag & y).sum())
            fp = int((flag & ~y).sum())
            cost = c_fn * fn + c_fp * fp
            n_pos = int(y.sum())
            trace.append({
                "z_critical": float(z),
                "escape_rate": float(fn / n_pos) if n_pos else 0.0,
                "false_reject_rate": float(fp / max(int((~y).sum()), 1)),
                "cost": float(cost),
            })
            if cost < best[0]:
                best = (cost, float(z))
        self.z_critical = best[1]
        return {"z_critical": self.z_critical, "grid": trace}

    # -- score ---------------------------------------------------------------
    def score(
        self,
        feats: pd.DataFrame,
        lot_alarm: pd.Series | None = None,
        z_critical: float | None = None,
    ) -> ModuleAResult:
        cfg = self.cfg
        zc = self.z_critical if z_critical is None else float(z_critical)
        pz = self.primary_z(feats, cfg)

        rule_sev, forest_sev, comb = {}, {}, {}
        fired = {}
        for p in cfg.parameters:
            names = self.feature_names[p.name]
            raw = -self.forests[p.name].score_samples(feats[names].to_numpy())
            fs = self.forest_cal[p.name].transform(raw)
            rs = self.rule_cal[p.name].transform(pz[p.name].to_numpy())
            forest_sev[p.name] = fs
            rule_sev[p.name] = rs
            # MAXIMUM-SEVERITY UNION, not an intersection (Section 4.2)
            comb[p.name] = np.maximum(rs, fs)
            fired[p.name] = pz[p.name].to_numpy() > zc

        per_param = pd.DataFrame(comb, index=feats.index)
        # Section 4.3: the component disposition is the WORST severity across
        # the three parameters. A component is never averaged into a pass.
        severity = per_param.max(axis=1)
        responsible = per_param.idxmax(axis=1)

        if lot_alarm is None:
            lot_alarm = pd.Series(False, index=feats.index)

        return ModuleAResult(
            severity=severity,
            responsible_param=responsible,
            per_param_severity=per_param,
            rule_severity=pd.DataFrame(rule_sev, index=feats.index),
            forest_severity=pd.DataFrame(forest_sev, index=feats.index),
            primary_z=pz,
            rule_fired=pd.DataFrame(fired, index=feats.index),
            z_critical=zc,
            lot_alarm=lot_alarm,
        )
