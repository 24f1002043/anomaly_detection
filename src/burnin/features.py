"""Feature construction.

Section 4.1  Module A  - four checkpoints per part per parameter. Features are
                        built for that reality: a short, unevenly spaced series.
Section 5.1  Module B  - restricted to 0 h and 24 h by an explicit allowlist.

Features that are deliberately EXCLUDED, and why (Section 4.1):
    RMS, kurtosis, crest factor, rolling-window mean and standard deviation,
    polynomial curvature fits, and persistence rules requiring consecutive
    windows. All of these belong to dense vibration or telemetry streams. With
    four unevenly spaced samples per part they are either undefined or
    noise-dominated. Carrying them into a burn-in design is a sign that the
    feature set was copied from a different sensing problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from burnin.config import Config, ParameterSpec

MAD_TO_SIGMA = 1.4826  # scale factor making MAD a consistent sigma estimator


# ----------------------------------------------------------------- lot statistics
def lot_median_mad(
    values: pd.Series, lot: pd.Series, min_scale: float
) -> tuple[pd.Series, pd.Series]:
    """Per-lot median and MAD of a column, broadcast back to every row.

    ``min_scale`` floors the MAD so that a lot whose readings collapse onto a
    single quantisation step cannot produce an infinite robust z.
    """
    med = values.groupby(lot).transform("median")
    abs_dev = (values - med).abs()
    mad = abs_dev.groupby(lot).transform("median")
    scale = (MAD_TO_SIGMA * mad).clip(lower=min_scale)
    return med, scale


def robust_z(
    values: pd.Series, lot: pd.Series, min_scale: float
) -> pd.Series:
    """robust_z(V_t) = ( V_t - median_lot(V_t) ) / ( 1.4826 * MAD_lot(V_t) ).

    This is the arithmetic a QA inspector can check by hand, and it is what
    reproduces the worked example of Section 4: in a lot averaging 10 uA, a part
    reading 45 uA is a severe anomaly even though the datasheet maximum is 50 uA.
    """
    med, scale = lot_median_mad(values, lot, min_scale)
    return (values - med) / scale


def percentile_rank(values: pd.Series, lot: pd.Series) -> pd.Series:
    """Rank of each part within its own lot, in [0, 1]."""
    return values.groupby(lot).rank(pct=True)


def _min_scale(p: ParameterSpec, typical: float) -> float:
    """Noise-aware floor for the robust-z denominator, in parameter units."""
    return max(
        0.5 * float(p.quantisation),
        0.25 * float(p.noise_proportional) * typical + 0.25 * float(p.noise_floor),
    )


def _rate_floor(p: ParameterSpec, typical: float, span_h: float) -> float:
    """A 'one measurement sigma spread over this interval' rate, used to guard
    the rate-acceleration logs against division by zero (Section 4.1)."""
    sigma = float(p.noise_proportional) * typical + float(p.noise_floor)
    return max(sigma / span_h, 1e-12)


# ------------------------------------------------------------ Module A features
def build_module_a_features(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Section 4.1 feature families, for every parameter.

    Returns the feature frame and a mapping parameter -> its own feature names.
    Cross-parameter features are appended to every parameter's list, because a
    real defect site moves several parameters together while noise usually moves
    one.
    """
    lot = df["lot_id"]
    hours = cfg.checkpoints
    out: dict[str, pd.Series] = {}
    per_param: dict[str, list[str]] = {}
    primary_z: dict[str, pd.Series] = {}
    drift_z: dict[str, pd.Series] = {}

    for p in cfg.parameters:
        names: list[str] = []
        typical = float(df[p.col(hours[0])].median())
        mscale = _min_scale(p, typical)
        v = {h: df[p.col(h)].astype(float) for h in hours}

        # --- Level: raw position, retained for the static baseline and for
        #     explanation.
        for h in hours:
            k = f"{p.name}_v{h}h"
            out[k] = v[h]
            names.append(k)

        # --- Interval delta: movement per segment of the soak.
        d1 = v[24] - v[0]
        d2 = v[96] - v[24]
        d3 = v[168] - v[96]
        for k, s in (("d1", d1), ("d2", d2), ("d3", d3)):
            key = f"{p.name}_{k}"
            out[key] = s
            names.append(key)

        # --- Interval rate: intervals are unequal, so rates make the three
        #     segments comparable.
        r1 = d1 / 24.0
        r2 = d2 / 72.0
        r3 = d3 / 72.0
        for k, s in (("r1", r1), ("r2", r2), ("r3", r3)):
            key = f"{p.name}_{k}"
            out[key] = s
            names.append(key)

        # --- Rate acceleration: separates the super-linear defective exponent
        #     from the saturating healthy one. Guarded against division by zero:
        #     physical drift is non-negative, so a rate is floored at one
        #     measurement sigma spread over its own interval before the log.
        f1 = _rate_floor(p, typical, 24.0)
        f2 = _rate_floor(p, typical, 72.0)
        a21 = np.log(r2.clip(lower=f2) / r1.clip(lower=f1))
        a32 = np.log(r3.clip(lower=f2) / r2.clip(lower=f2))
        out[f"{p.name}_accel_r2_r1"] = a21
        out[f"{p.name}_accel_r3_r2"] = a32
        names += [f"{p.name}_accel_r2_r1", f"{p.name}_accel_r3_r2"]

        # --- Total drift: summary magnitude, absolute and relative.
        total = v[168] - v[0]
        rel_total = total / v[0].clip(lower=max(mscale, 1e-9))
        out[f"{p.name}_total_drift"] = total
        out[f"{p.name}_total_drift_rel"] = rel_total
        names += [f"{p.name}_total_drift", f"{p.name}_total_drift_rel"]

        # --- Lot-relative level: robust z at each checkpoint + percentile rank.
        zs = []
        for h in hours:
            z = robust_z(v[h], lot, mscale)
            key = f"{p.name}_z_{h}h"
            out[key] = z
            names.append(key)
            zs.append(z)
            pk = f"{p.name}_pct_{h}h"
            out[pk] = percentile_rank(v[h], lot)
            names.append(pk)

        # --- Lot-relative drift: catches a part that is normal in level but
        #     abnormal in movement.
        z_total = robust_z(total, lot, mscale)
        out[f"{p.name}_z_total_drift"] = z_total
        names.append(f"{p.name}_z_total_drift")
        for k, s, fl in (("r1", r1, f1), ("r2", r2, f2), ("r3", r3, f2)):
            key = f"{p.name}_z_{k}"
            out[key] = robust_z(s, lot, fl)
            names.append(key)
        out[f"{p.name}_z_total_drift_rel"] = robust_z(rel_total, lot, 1e-4)
        names.append(f"{p.name}_z_total_drift_rel")

        # per-parameter summary z used by the cross-parameter family and by the
        # deterministic z-rule R1 (Section 4.2)
        level_z_max = pd.concat(zs, axis=1).max(axis=1)
        primary_z[p.name] = pd.concat([level_z_max, z_total], axis=1).max(axis=1)
        drift_z[p.name] = z_total
        out[f"{p.name}_z_level_max"] = level_z_max
        names.append(f"{p.name}_z_level_max")

        per_param[p.name] = names

    # --- Cross-parameter family (shared by all parameters) --------------------
    zmat = pd.concat([primary_z[p.name] for p in cfg.parameters], axis=1)
    dmat = pd.concat([drift_z[p.name] for p in cfg.parameters], axis=1)
    thr = float(cfg.module_a["cross_param_z_threshold"])
    cross = {
        "xp_z_max": zmat.max(axis=1),
        "xp_z_mean": zmat.mean(axis=1),
        "xp_z_min": zmat.min(axis=1),
        "xp_drift_z_max": dmat.max(axis=1),
        "xp_drift_z_mean": dmat.mean(axis=1),
        "xp_n_exceeding": (zmat > thr).sum(axis=1).astype(float),
        "xp_n_drift_exceeding": (dmat > thr).sum(axis=1).astype(float),
    }
    out.update(cross)
    for p in cfg.parameters:
        per_param[p.name] = per_param[p.name] + list(cross.keys())

    feats = pd.DataFrame(out, index=df.index)
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats, per_param


# ------------------------------------------------------------ Module B features
def build_module_b_features(
    df: pd.DataFrame, cfg: Config, target_param: str
) -> tuple[pd.DataFrame, list[str]]:
    """Section 5.1 - inputs restricted to the 0 h and 24 h checkpoints.

    Derived features are permitted as long as they are functions of those two
    checkpoints and of lot statistics computed at those two checkpoints only.
    The restriction is enforced by :func:`burnin.schema.module_b_allowlist`,
    which is applied by the caller before the model ever sees the frame.
    """
    lot = df["lot_id"]
    allowed = cfg.early_allowlist  # [0, 24]
    h0, h24 = allowed[0], allowed[1]
    out: dict[str, pd.Series] = {}

    tp = cfg.param(target_param)
    for p in cfg.parameters:
        is_target = p.name == target_param
        typical = float(df[p.col(h0)].median())
        mscale = _min_scale(p, typical)
        v0 = df[p.col(h0)].astype(float)
        v24 = df[p.col(h24)].astype(float)
        d24 = v24 - v0

        if is_target:
            # --- Required raw (target parameter) --------------------------
            out[f"{p.name}_v{h0}h"] = v0
            out[f"{p.name}_v{h24}h"] = v24
            # --- Early drift -----------------------------------------------
            out[f"{p.name}_d24"] = d24
            out[f"{p.name}_reld24"] = d24 / v0.clip(lower=max(mscale, 1e-9))
            out[f"{p.name}_rate24"] = d24 / 24.0
            # --- Lot context at 0 h and 24 h --------------------------------
            med0, sc0 = lot_median_mad(v0, lot, mscale)
            med24, sc24 = lot_median_mad(v24, lot, mscale)
            out[f"{p.name}_lotmed_{h0}h"] = med0
            out[f"{p.name}_lotmed_{h24}h"] = med24
            out[f"{p.name}_lotmad_{h0}h"] = sc0 / MAD_TO_SIGMA
            out[f"{p.name}_lotmad_{h24}h"] = sc24 / MAD_TO_SIGMA
            out[f"{p.name}_z_{h0}h"] = (v0 - med0) / sc0
            out[f"{p.name}_z_{h24}h"] = (v24 - med24) / sc24
            out[f"{p.name}_zd24"] = robust_z(d24, lot, _rate_floor(p, typical, 1.0))
            out[f"{p.name}_pct_{h24}h"] = percentile_rank(v24, lot)
            # lot-median early trajectory, the third baseline of Section 5.2
            out[f"{p.name}_lot_reldrift"] = (med24 - med0) / med0.clip(
                lower=max(mscale, 1e-9)
            )
            # the part's own value relative to its lot, in lot-MAD units,
            # is the quantity the safety slope is eventually judged against
            out[f"{p.name}_headroom"] = (float(tp.limit) - v24) / max(
                float(tp.limit), 1e-9
            )
        else:
            # --- Cross-parameter early: d24 and its robust z for the other
            #     two parameters (Section 5.1).
            out[f"xp_{p.name}_d24"] = d24
            out[f"xp_{p.name}_reld24"] = d24 / v0.clip(lower=max(mscale, 1e-9))
            out[f"xp_{p.name}_zd24"] = robust_z(
                d24, lot, _rate_floor(p, typical, 1.0)
            )
            out[f"xp_{p.name}_z_{h24}h"] = robust_z(v24, lot, mscale)

    feats = pd.DataFrame(out, index=df.index)
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats, list(feats.columns)
