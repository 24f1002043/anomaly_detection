"""Section 3 - Synthetic burn-in generator.

No public dataset reproduces parametric burn-in telemetry with hidden 168 h
terminal values and defect labels, so the data is generated. The realism of the
generator is therefore the load-bearing element of the submission: a generator
that makes defects obvious proves nothing, because the models can only find what
was deliberately embedded.

Physics (3.1)
    AF   = exp[ (Ea / k) * (1/T_use - 1/T_stress) ]
    V(t) = V_0 * [ 1 + beta_p * AF * (t / 168)^n ] + eps(t)
    n = 0.5 healthy   (sub-linear, saturating wear-in)
    n = 1.2 defective (super-linear, accelerating runaway)

Population structure (3.2)
    lot hierarchy, ONE is_defective draw per component, cross-parameter
    correlation rho = 0.7, proportional measurement noise + quantisation,
    chamber common-mode offset shared across a chamber load.

Difficulty controls (3.3)
    latent defects finishing under the limit, benign high-baseline healthy
    parts, low-start fast-drift defects, and contaminated lots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from burnin.config import Config
from burnin.schema import LABEL_COLUMN

# defect_class values, carried for diagnostics only -- never a model input.
CLS_HEALTHY = "healthy"
CLS_BENIGN_HIGH = "benign_high_baseline"
CLS_DEFECT_STD = "defect_standard"
CLS_DEFECT_LOWSTART = "defect_low_start_fast_drift"


@dataclass
class GeneratedDataset:
    frame: pd.DataFrame
    truth: pd.DataFrame  # noise-free trajectories, for auditing only
    meta: dict


def _equicorrelated_normals(
    rng: np.random.Generator, n_rows: int, n_dim: int, rho: float
) -> np.ndarray:
    """n_rows x n_dim standard normals with equicorrelation rho.

    Built from a single shared factor plus independent residuals, which is the
    right structural story: one physical defect site drives all three
    parameters, so the shared component is the defect and the residual is
    per-parameter measurement-level variation.
    """
    shared = rng.standard_normal((n_rows, 1))
    idio = rng.standard_normal((n_rows, n_dim))
    return np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * idio


class BurnInGenerator:
    """Generates one wide-format burn-in run."""

    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.seed = cfg.seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ physics
    def _acceleration_factors(self, n: int, defective: np.ndarray) -> np.ndarray:
        """Per-part AF from a per-part activation energy (3.1)."""
        ph = self.cfg.physics
        ea = np.where(
            defective,
            float(ph["ea_defect_ev"])
            + self.rng.normal(0.0, float(ph["ea_defect_jitter_ev"]), n),
            float(ph["ea_healthy_ev"])
            + self.rng.normal(0.0, float(ph["ea_healthy_jitter_ev"]), n),
        )
        ea = np.clip(ea, 0.15, 1.10)
        k = float(ph["boltzmann_ev_per_k"])
        inv_t = 1.0 / float(ph["t_use_k"]) - 1.0 / float(ph["t_stress_k"])
        return np.exp((ea / k) * inv_t)

    def _exponents(
        self, n: int, defective: np.ndarray, low_start: np.ndarray
    ) -> np.ndarray:
        """n in the trajectory law. Low-start defects are back-loaded further so
        that they stay 'only mildly elevated at 24 h' (3.3)."""
        ph = self.cfg.physics
        base = np.where(
            defective, float(ph["exponent_defect"]), float(ph["exponent_healthy"])
        )
        base = base + self.rng.normal(0.0, float(ph["exponent_jitter"]), n)
        base = base + np.where(low_start, 0.45, 0.0)
        return np.clip(base, 0.25, 2.2)

    # ------------------------------------------------------------- lot structure
    def _build_lots(self) -> pd.DataFrame:
        cfg = self.cfg
        n_lots = int(cfg.run["n_lots"])
        load_size = int(cfg.population["chamber_load_size_lots"])

        lot_ids = [f"LOT-{i:04d}" for i in range(n_lots)]
        chamber_ids = [f"CH-{i // load_size:03d}" for i in range(n_lots)]

        # One contaminated lot in N (3.3): 15-20 % defective inside a single lot.
        n_contam = int(cfg.difficulty["contaminated_lots"])
        contam_idx = self.rng.choice(n_lots, size=n_contam, replace=False)
        contaminated = np.zeros(n_lots, dtype=bool)
        contaminated[contam_idx] = True

        lots = pd.DataFrame(
            {"lot_id": lot_ids, "chamber_id": chamber_ids, "contaminated": contaminated}
        )
        # Lot-level mean for each parameter. Between-lot spread is deliberately
        # WIDER than within-lot spread (3.2), so a part can be a severe
        # lot-relative outlier while sitting mid-range globally.
        for p in cfg.parameters:
            lots[f"lotmean_{p.name}"] = float(p.lot_mean_center) * np.exp(
                self.rng.normal(0.0, float(p.lot_mean_spread), n_lots)
            )
        return lots

    # ------------------------------------------------------------------ generate
    def generate(self) -> GeneratedDataset:
        cfg = self.cfg
        rng = self.rng
        per_lot = int(cfg.run["components_per_lot"])
        lots = self._build_lots()
        n_lots = len(lots)
        n = n_lots * per_lot
        hours = np.asarray(cfg.checkpoints, dtype=float)
        terminal = float(cfg.terminal_hour)

        lot_index = np.repeat(np.arange(n_lots), per_lot)
        within = np.tile(np.arange(per_lot), n_lots)
        lot_id_arr = lots["lot_id"].to_numpy()

        # --- identity and genealogy (Section 2) ------------------------------
        component_id = np.array(
            [f"C{li:04d}{wi:04d}" for li, wi in zip(lot_index, within)], dtype=object
        )
        lot_id = lot_id_arr[lot_index]
        chamber_id = lots["chamber_id"].to_numpy()[lot_index]
        wafer_id = np.array(
            [f"{lid}-W{(wi // 50) + 1}" for lid, wi in zip(lot_id, within)], dtype=object
        )
        slot_id = np.array([f"S{(wi % 50) + 1:02d}" for wi in within], dtype=object)

        # --- ONE is_defective draw per component (3.2) ------------------------
        base_prev = float(cfg.population["defect_prevalence"])
        contam_prev = float(cfg.difficulty["contaminated_prevalence"])
        prev = np.where(lots["contaminated"].to_numpy()[lot_index], contam_prev, base_prev)
        defective = rng.random(n) < prev

        # --- difficulty sub-populations (3.3) ---------------------------------
        low_start = defective & (
            rng.random(n) < float(cfg.difficulty["low_start_fast_drift_frac"])
        )
        benign_high = (~defective) & (
            rng.random(n) < float(cfg.difficulty["benign_high_baseline_frac"])
        )

        defect_class = np.full(n, CLS_HEALTHY, dtype=object)
        defect_class[benign_high] = CLS_BENIGN_HIGH
        defect_class[defective] = CLS_DEFECT_STD
        defect_class[low_start] = CLS_DEFECT_LOWSTART

        af = self._acceleration_factors(n, defective)
        n_exp = self._exponents(n, defective, low_start)

        # --- cross-parameter correlation (3.2) --------------------------------
        # A genuinely defective part shows elevated drift across all three
        # parameters, correlated at rho = 0.7 (one physical defect site).
        # Healthy parts get independent draws: noise usually moves one parameter.
        rho = float(cfg.population["cross_parameter_rho"])
        n_par = len(cfg.parameters)
        z_defect = _equicorrelated_normals(rng, n, n_par, rho)
        z_healthy = rng.standard_normal((n, n_par))
        z_drift = np.where(defective[:, None], z_defect, z_healthy)

        # --- chamber common-mode (3.2) ----------------------------------------
        # One shared offset per (chamber load, checkpoint, parameter). Cancels
        # under lot-relative statistics; survives naive absolute-drift rules.
        chamber_codes, chamber_pos = np.unique(chamber_id, return_inverse=True)
        common_mode = {
            p.name: rng.normal(
                0.0, float(p.common_mode_sigma), (len(chamber_codes), len(hours))
            )
            for p in cfg.parameters
        }

        frame: dict[str, np.ndarray] = {
            "component_id": component_id,
            "lot_id": lot_id,
            "wafer_id": wafer_id,
            "slot_id": slot_id,
        }
        truth: dict[str, np.ndarray] = {"component_id": component_id}

        for j, p in enumerate(cfg.parameters):
            # ---- initial value V_0 : lot mean, then narrow within-lot spread
            lot_mean = lots[f"lotmean_{p.name}"].to_numpy()[lot_index]
            v0 = lot_mean * np.exp(rng.normal(0.0, float(p.within_lot_spread), n))

            # Benign high-baseline healthy parts: high V_0, flat profile.
            # The boost is expressed in WITHIN-LOT sigma units, not as a fixed
            # multiplier. The purpose of this population (3.3) is to punish rules
            # that flag on absolute level, and "high" for that purpose means high
            # relative to the part's own lot. A fixed multiplier would instead
            # push parts on a tight-limit parameter up against the datasheet
            # maximum, where they are caught by the reliability-margin policy of
            # Section 5.3 rather than by any level rule -- which tests nothing.
            k_sigma = rng.uniform(
                float(cfg.difficulty["benign_high_sigma_low"]),
                float(cfg.difficulty["benign_high_sigma_high"]),
                n,
            )
            boost = np.where(
                benign_high, np.exp(k_sigma * float(p.within_lot_spread)), 1.0
            )
            # Low-start fast-drift defects: depressed V_0, unremarkable at 0 h.
            depress = np.where(
                low_start, float(cfg.difficulty["low_start_depression"]), 1.0
            )
            v0 = v0 * boost * depress

            # ---- incoming inspection ----------------------------------------
            # A part that is already outside its datasheet limit at t = 0 fails
            # incoming inspection and never enters the chamber. Rather than
            # dropping rows (which would make lot sizes ragged), such parts are
            # redrawn into the top in-spec band. They stay high-baseline traps
            # without being pre-failed by the static screen at 0 h.
            cap = float(cfg.population["incoming_inspection_cap_frac"]) * float(p.limit)
            over = v0 > cap
            if over.any():
                v0[over] = rng.uniform(0.86 * float(p.limit), cap, int(over.sum()))

            # ---- drift coefficient beta_p ----------------------------------
            beta_center = np.where(
                defective, float(p.beta_defect), float(p.beta_healthy)
            )
            beta = beta_center * np.exp(float(p.beta_spread) * z_drift[:, j])
            # Benign high-baseline parts are FLAT: high level, no movement.
            beta = beta * np.where(benign_high, 0.35, 1.0)
            beta = np.maximum(beta, 0.0)

            # ---- trajectory ------------------------------------------------
            # V(t) = V_0 * [ 1 + beta_p * AF * (t/168)^n ]
            frac = (hours[None, :] / terminal) ** n_exp[:, None]
            v_true = v0[:, None] * (1.0 + (beta * af)[:, None] * frac)

            # ---- chamber common-mode, measurement noise, quantisation -------
            cm = common_mode[p.name][chamber_pos, :]
            v_obs = v_true * (1.0 + cm)
            sigma = np.sqrt(
                (float(p.noise_proportional) * v_obs) ** 2 + float(p.noise_floor) ** 2
            )
            v_obs = v_obs + rng.normal(0.0, 1.0, v_obs.shape) * sigma
            q = float(p.quantisation)
            v_obs = np.maximum(np.round(v_obs / q) * q, 0.0)

            for hi, h in enumerate(cfg.checkpoints):
                frame[p.col(h)] = v_obs[:, hi]
                truth[f"true_{p.name}_{h}h"] = v_true[:, hi]
            truth[f"beta_{p.name}"] = beta
            truth[f"sigma_meas_{p.name}"] = sigma[:, -1]

        for p in cfg.parameters:
            frame[p.limit_col] = np.full(n, float(p.limit))

        frame["chamber_id"] = chamber_id
        frame["defect_class"] = defect_class
        frame[LABEL_COLUMN] = defective

        truth["af"] = af
        truth["n_exponent"] = n_exp
        truth["is_defective"] = defective
        truth["defect_class"] = defect_class

        df = pd.DataFrame(frame)
        truth_df = pd.DataFrame(truth)

        meta = {
            "seed": self.seed,
            "n_components": int(n),
            "n_lots": int(n_lots),
            "components_per_lot": per_lot,
            "af_healthy_nominal": float(self.cfg.af_healthy),
            "af_defect_nominal": float(self.cfg.af_defect),
            "prevalence_realised": float(defective.mean()),
            "contaminated_lots": lots.loc[lots["contaminated"], "lot_id"].tolist(),
            "n_low_start_defects": int(low_start.sum()),
            "n_benign_high_baseline": int(benign_high.sum()),
            "checkpoints": list(cfg.checkpoints),
        }
        return GeneratedDataset(frame=df, truth=truth_df, meta=meta)


# ---------------------------------------------------------------- static screen
def static_screen_mask(
    df: pd.DataFrame, cfg: Config, hours: list[int] | None = None
) -> pd.Series:
    """The engineering screen that stays in the loop: absolute pass/fail limits
    applied at each checkpoint (Section 1). A part fails if any parameter
    exceeds its datasheet maximum at any checkpoint.
    """
    hrs = cfg.checkpoints if hours is None else hours
    fail = pd.Series(False, index=df.index)
    for p in cfg.parameters:
        for h in hrs:
            fail = fail | (df[p.col(h)] > df[p.limit_col])
    return fail


def acceptance_report(df: pd.DataFrame, cfg: Config) -> dict:
    """Section 3.3 generator acceptance criterion.

    The generator is only accepted if the best achievable static-limit baseline
    catches a clear minority of the defective population. If a tuned static
    threshold performs close to the models, the data is too easy and the
    demonstration is worthless.
    """
    y = df[LABEL_COLUMN].to_numpy()
    n_def = int(y.sum())
    static_fail = static_screen_mask(df, cfg).to_numpy()

    caught = int((static_fail & y).sum())
    static_recall = caught / max(n_def, 1)

    under_limit = np.ones(len(df), dtype=bool)
    per_param_under = {}
    for p in cfg.parameters:
        ok = df[p.col(cfg.terminal_hour)].to_numpy() <= float(p.limit)
        per_param_under[p.name] = float(ok[y].mean())
        under_limit &= ok

    return {
        "n_components": int(len(df)),
        "n_defective": n_def,
        "prevalence": float(y.mean()),
        "static_limit_recall": float(static_recall),
        "static_limit_escape_rate": float(1.0 - static_recall),
        "static_limit_false_reject_rate": float(
            (static_fail & ~y).sum() / max((~y).sum(), 1)
        ),
        "defects_under_limit_at_168h": float(under_limit[y].mean()),
        "defects_under_limit_per_param": per_param_under,
        "accepted_minority_rule": bool(static_recall < 0.5),
        "accepted_latent_target": bool(
            under_limit[y].mean() >= float(cfg.difficulty["latent_under_limit_target"])
        ),
    }
