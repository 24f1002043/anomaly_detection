"""Section 5 - Module B, the 0 h + 24 h drift predictor.

    Value_0h + Value_24h -> early feature vector -> Random Forest
    -> predicted Value_168h + interval -> safety-slope decision at 24 h

Inputs are restricted to the 0 h and 24 h checkpoints by an explicit allowlist
(Section 5.1), enforced in :mod:`burnin.schema` rather than by convention.

The model is one Random Forest regressor per parameter, since the units and the
dynamics differ. Splitting is by ``lot_id`` using grouped cross-validation:
entire lots are held out, never individual parts within a lot.

Three baselines are reported alongside the model, and the model has to beat all
three to justify itself (Section 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from burnin.config import Config, ParameterSpec
from burnin.features import build_module_b_features
from burnin.schema import module_b_allowlist
from burnin.splits import grouped_folds


# ------------------------------------------------------------------- baselines
def baseline_last_value(df: pd.DataFrame, p: ParameterSpec, cfg: Config) -> np.ndarray:
    """Baseline 1. Predict Value_168h = Value_24h."""
    return df[p.col(cfg.early_allowlist[1])].to_numpy(dtype=float)


def baseline_linear_extrapolation(
    df: pd.DataFrame, p: ParameterSpec, cfg: Config
) -> np.ndarray:
    """Baseline 2. Value_24h + (d_24 / 24) * 144."""
    h0, h24 = cfg.early_allowlist
    v0 = df[p.col(h0)].to_numpy(dtype=float)
    v24 = df[p.col(h24)].to_numpy(dtype=float)
    return v24 + ((v24 - v0) / 24.0) * (float(cfg.terminal_hour) - h24)


def baseline_lot_median_trajectory(
    df: pd.DataFrame, p: ParameterSpec, cfg: Config
) -> np.ndarray:
    """Baseline 3. Apply the lot's median relative drift to the part's 24 h value.

    The lot's median relative drift is only observable at 24 h here, so it is
    projected to 168 h under the healthy sub-linear exponent. This keeps the
    baseline inside the 0 h + 24 h allowlist.
    """
    h0, h24 = cfg.early_allowlist
    v0 = df[p.col(h0)].astype(float)
    v24 = df[p.col(h24)].astype(float)
    rel = (v24 - v0) / v0.clip(lower=1e-9)
    lot_rel = rel.groupby(df["lot_id"]).transform("median").to_numpy()
    n = float(cfg.physics["exponent_healthy"])
    scale = (float(cfg.terminal_hour) / float(h24)) ** n
    return v24.to_numpy() * (1.0 + lot_rel * (scale - 1.0))


BASELINES = {
    "last_value": baseline_last_value,
    "linear_extrapolation": baseline_linear_extrapolation,
    "lot_median_trajectory": baseline_lot_median_trajectory,
}


# ----------------------------------------------------------------- safety slope
@dataclass
class SafetySlope:
    """Section 5.3 - the safety slope is calculated, not assumed.

        S_safe  = ( L_max - V_0 - M ) / 168      [ units per hour ]
        S_pred  = ( V_168_hat - V_0 ) / 168
        margin  = S_safe - S_pred
        M       = max( 0.10 * L_max, 3 * sigma_meas )

    The margin, not the binary verdict, is the primary output shown to the
    inspector. A part at margin -0.001 and a part at margin -0.8 are both
    rejections, and the inspector needs to see which is which.

    Note that V_0 cancels in the margin, which makes it a clean identity an
    inspector can check by hand:  margin = ( L_max - M - V_168_hat ) / 168.
    """

    parameter: str
    limit: float
    margin_M: float
    sigma_meas: float
    horizon: float

    def s_safe(self, v0: np.ndarray) -> np.ndarray:
        return (self.limit - np.asarray(v0, dtype=float) - self.margin_M) / self.horizon

    def s_pred(self, v0: np.ndarray, v168_hat: np.ndarray) -> np.ndarray:
        return (np.asarray(v168_hat, dtype=float) - np.asarray(v0, dtype=float)) / self.horizon

    def margin(self, v0: np.ndarray, v168_hat: np.ndarray) -> np.ndarray:
        return self.s_safe(v0) - self.s_pred(v0, v168_hat)


def estimate_sigma_meas(df: pd.DataFrame, p: ParameterSpec, cfg: Config) -> float:
    """Measurement sigma in parameter units, estimated from the 0 h population.

    Uses the configured proportional and floor noise terms at the observed
    typical level, so that M is stated explicitly per parameter as Section 5.3
    requires rather than being inherited from a global constant.
    """
    typical = float(df[p.col(cfg.checkpoints[0])].median())
    return float(
        np.sqrt((float(p.noise_proportional) * typical) ** 2 + float(p.noise_floor) ** 2)
    )


def build_safety_slope(df: pd.DataFrame, p: ParameterSpec, cfg: Config) -> SafetySlope:
    sigma = estimate_sigma_meas(df, p, cfg)
    return SafetySlope(
        parameter=p.name,
        limit=float(p.limit),
        margin_M=float(p.reliability_margin(sigma)),
        sigma_meas=sigma,
        horizon=float(cfg.terminal_hour),
    )


# --------------------------------------------------------------- the predictor
@dataclass
class ModuleBPrediction:
    parameter: str
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    v0: np.ndarray
    s_safe: np.ndarray
    s_pred: np.ndarray
    s_pred_upper: np.ndarray
    margin: np.ndarray
    margin_upper: np.ndarray
    early_reject: np.ndarray
    reason: np.ndarray


class ModuleB:
    """One Random Forest regressor per parameter, with a split-conformal
    prediction interval calibrated on held-out LOTS.

    A point prediction ignores its own uncertainty, which is exactly the wrong
    behaviour when a miss is catastrophic. The rejection rule therefore acts on
    the upper bound (Section 5.4), which makes the uncertainty itself a reason to
    reject: a part the model cannot confidently clear is not cleared.
    """

    def __init__(self, cfg: Config, parameter: str):
        self.cfg = cfg
        self.p = cfg.param(parameter)
        self.model: RandomForestRegressor | None = None
        self.feature_names: list[str] = []
        self.q_hi: float = float("nan")   # conformal upper residual quantile
        self.q_lo: float = float("nan")
        self.spread_floor: float = float("nan")
        self.slope: SafetySlope | None = None
        self.oof_: pd.DataFrame | None = None

    # -- features -------------------------------------------------------------
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats, names = build_module_b_features(df, self.cfg, self.p.name)
        # Hard enforcement of the checkpoint allowlist before the model sees it.
        module_b_allowlist(self.cfg).validate(names)
        return feats

    def target(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.p.col(self.cfg.terminal_hour)].to_numpy(dtype=float)

    # -- local difficulty estimate -------------------------------------------
    @staticmethod
    def _tree_spread(model: RandomForestRegressor, X: np.ndarray) -> np.ndarray:
        """Standard deviation across the forest's trees, per row.

        Cheap, needs no extra model, and is the ensemble's own statement about
        how well it understands a part: where the trees disagree, the part sits
        somewhere the training lots did not cover well.
        """
        preds = np.stack([t.predict(X) for t in model.estimators_], axis=0)
        return preds.std(axis=0)

    def _scale_from_spread(self, spread: np.ndarray) -> np.ndarray:
        """Turn a raw tree spread into a strictly positive interval scale."""
        floor = self.spread_floor
        if floor is None or not np.isfinite(floor):
            floor = float(np.quantile(spread, 0.05))
        return np.maximum(spread, max(float(floor), 1e-9))

    def _new_model(self) -> RandomForestRegressor:
        rf = self.cfg.module_b["random_forest"]
        return RandomForestRegressor(
            n_estimators=int(rf["n_estimators"]),
            min_samples_leaf=int(rf["min_samples_leaf"]),
            max_features=float(rf["max_features"]),
            n_jobs=int(rf["n_jobs"]),
            random_state=int(rf["random_state"]),
        )

    # -- fit ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "ModuleB":
        """Fit on training lots.

        Grouped cross-validation over ``lot_id`` produces out-of-fold
        predictions, which serve two purposes: an honest in-training estimate of
        accuracy, and the conformity scores for the split-conformal interval.
        Calibrating the interval on held-out lots rather than on held-out parts
        is what keeps the 90 % coverage claim honest, because parts within a lot
        share lot median and lot MAD.
        """
        X = self.features(train_df)
        self.feature_names = list(X.columns)
        y = self.target(train_df)
        groups = train_df["lot_id"].reset_index(drop=True)
        folds = grouped_folds(
            groups, int(self.cfg.module_b["cv_folds"]), self.cfg.seed + 7
        )

        Xv = X.to_numpy(dtype=float)
        oof = np.full(len(y), np.nan)
        oof_spread = np.full(len(y), np.nan)
        for tr, te in folds:
            m = self._new_model()
            m.fit(Xv[tr], y[tr])
            oof[te] = m.predict(Xv[te])
            oof_spread[te] = self._tree_spread(m, Xv[te])

        # Normalised (locally adaptive) split conformal, calibrated on the
        # grouped out-of-fold predictions -- Section 5.4.
        #
        # A constant-width band is calibrated on average and wrong in the place
        # that matters: an extreme part gets the same interval as a typical one,
        # so the rule that is supposed to refuse to clear a part the model
        # cannot confidently clear never widens for the parts it understands
        # least. Scores are therefore normalised by a per-part difficulty
        # estimate -- the spread of the forest's own trees -- so the interval
        # grows exactly where the ensemble disagrees with itself.
        nominal = float(self.cfg.module_b["interval_nominal"])
        alpha = 1.0 - nominal
        resid = y - oof
        # The floor keeps the interval from collapsing where the trees happen to
        # agree perfectly. It is fixed from the out-of-fold spread distribution
        # BEFORE the quantiles are taken, so calibration and inference use the
        # identical scale definition.
        self.spread_floor = float(np.quantile(oof_spread, 0.05))
        scale = self._scale_from_spread(oof_spread)
        self.q_hi = float(np.quantile(resid / scale, 1.0 - alpha / 2.0))
        self.q_lo = float(np.quantile(resid / scale, alpha / 2.0))

        self.oof_ = pd.DataFrame(
            {
                "component_id": train_df["component_id"].to_numpy(),
                "lot_id": train_df["lot_id"].to_numpy(),
                "y_true": y,
                "y_oof": oof,
            }
        )

        self.model = self._new_model()
        self.model.fit(Xv, y)
        self.slope = build_safety_slope(train_df, self.p, self.cfg)
        return self

    # -- predict --------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> ModuleBPrediction:
        assert self.model is not None and self.slope is not None, "ModuleB not fitted"
        X = self.features(df)[self.feature_names].to_numpy(dtype=float)
        point = self.model.predict(X)
        # interval width scales with the forest's own disagreement on this part
        scale = self._scale_from_spread(self._tree_spread(self.model, X))
        lower = point + self.q_lo * scale
        upper = point + self.q_hi * scale

        v0 = df[self.p.col(self.cfg.early_allowlist[0])].to_numpy(dtype=float)
        s_safe = self.slope.s_safe(v0)
        s_pred = self.slope.s_pred(v0, point)
        s_pred_up = self.slope.s_pred(v0, upper)
        margin = s_safe - s_pred
        margin_up = s_safe - s_pred_up

        # Section 5.4, asymmetric rejection rule:
        #   EARLY-REJECT if  V_168_hat(upper, 90%) > L_max - M   or  S_pred > S_safe
        lim = float(self.p.limit) - float(self.slope.margin_M)
        by_interval = upper > lim
        by_slope = s_pred > s_safe
        early = by_interval | by_slope

        reason = np.where(
            by_interval & by_slope,
            "upper_bound_and_slope",
            np.where(by_interval, "upper_bound_exceeds_limit_minus_M",
                     np.where(by_slope, "predicted_slope_exceeds_safe_slope", "none")),
        )

        return ModuleBPrediction(
            parameter=self.p.name,
            point=point,
            lower=lower,
            upper=upper,
            v0=v0,
            s_safe=s_safe,
            s_pred=s_pred,
            s_pred_upper=s_pred_up,
            margin=margin,
            margin_upper=margin_up,
            early_reject=early,
            reason=reason,
        )

    # -- baselines ------------------------------------------------------------
    def baseline_predictions(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {k: fn(df, self.p, self.cfg) for k, fn in BASELINES.items()}


def fit_all(cfg: Config, train_df: pd.DataFrame) -> dict[str, ModuleB]:
    return {p.name: ModuleB(cfg, p.name).fit(train_df) for p in cfg.parameters}


def component_early_decision(
    preds: dict[str, ModuleBPrediction], index: pd.Index
) -> pd.DataFrame:
    """Aggregate per-parameter early decisions to a component decision.

    As in Section 4.3, the component takes the WORST case across parameters: a
    single parameter that cannot be confidently cleared rejects the component,
    and the record names which parameter drove it.
    """
    rej = pd.DataFrame({k: v.early_reject for k, v in preds.items()}, index=index)
    margins = pd.DataFrame({k: v.margin_upper for k, v in preds.items()}, index=index)
    any_rej = rej.any(axis=1)
    driver = margins.idxmin(axis=1)  # most negative margin drives the decision
    reasons = pd.DataFrame({k: v.reason for k, v in preds.items()}, index=index)
    driver_reason = pd.Series(
        [reasons.loc[i, driver.loc[i]] for i in index], index=index
    )
    return pd.DataFrame(
        {
            "early_reject": any_rej,
            "early_driver_param": driver,
            "early_reason": driver_reason.where(any_rej, "none"),
            "early_worst_margin": margins.min(axis=1),
        }
    )
