"""Configuration loading. Section 10: a single config file covering prevalence,
activation energies, noise levels, cost ratio and operating-point targets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class ParameterSpec:
    """One screened parameter. Units are never pooled across parameters."""

    name: str
    column: str
    units: str
    label: str
    limit: float
    lot_mean_center: float
    lot_mean_spread: float
    within_lot_spread: float
    beta_healthy: float
    beta_defect: float
    beta_spread: float
    noise_proportional: float
    noise_floor: float
    quantisation: float
    common_mode_sigma: float
    reliability_margin_frac: float

    def col(self, hour: int) -> str:
        """Column name for this parameter at a checkpoint hour."""
        return self.column.format(t=hour)

    @property
    def limit_col(self) -> str:
        return f"limit_{self.name}"

    def reliability_margin(self, sigma_meas: float) -> float:
        """M = max(0.10 * Lmax, 3 * sigma_meas)  -- Section 5.3.

        Stated explicitly per parameter rather than assumed globally.
        """
        return max(self.reliability_margin_frac * self.limit, 3.0 * sigma_meas)


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    parameters: list[ParameterSpec] = field(init=False)

    def __post_init__(self) -> None:
        self.parameters = [ParameterSpec(**p) for p in self.raw["parameters"]]

    # -- section accessors -----------------------------------------------------
    @property
    def run(self) -> dict[str, Any]:
        return self.raw["run"]

    @property
    def checkpoints(self) -> list[int]:
        return list(self.raw["checkpoints"]["hours"])

    @property
    def early_allowlist(self) -> list[int]:
        """Checkpoints Module B is permitted to see. Section 5.1."""
        return list(self.raw["checkpoints"]["early_allowlist"])

    @property
    def terminal_hour(self) -> int:
        return int(self.raw["checkpoints"]["terminal"])

    @property
    def physics(self) -> dict[str, Any]:
        return self.raw["physics"]

    @property
    def population(self) -> dict[str, Any]:
        return self.raw["population"]

    @property
    def difficulty(self) -> dict[str, Any]:
        return self.raw["difficulty"]

    @property
    def module_a(self) -> dict[str, Any]:
        return self.raw["module_a"]

    @property
    def module_b(self) -> dict[str, Any]:
        return self.raw["module_b"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def seed(self) -> int:
        return int(self.run["seed"])

    def param(self, name: str) -> ParameterSpec:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"unknown parameter {name!r}")

    # -- derived physics -------------------------------------------------------
    def acceleration_factor(self, ea_ev: float) -> float:
        """AF = exp[ (Ea / k) * (1/T_use - 1/T_stress) ]  -- Section 3.1."""
        ph = self.physics
        k = float(ph["boltzmann_ev_per_k"])
        return math.exp((ea_ev / k) * (1.0 / float(ph["t_use_k"]) - 1.0 / float(ph["t_stress_k"])))

    @property
    def af_healthy(self) -> float:
        return self.acceleration_factor(float(self.physics["ea_healthy_ev"]))

    @property
    def af_defect(self) -> float:
        return self.acceleration_factor(float(self.physics["ea_defect_ev"]))

    def data_dir(self) -> Path:
        d = REPO_ROOT / self.run["output_dir"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def artifact_dir(self, *parts: str) -> Path:
        d = REPO_ROOT / self.run["artifact_dir"]
        for p in parts:
            d = d / p
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=p)
