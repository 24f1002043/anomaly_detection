"""Data model (Section 2) and the Module B checkpoint allowlist (Section 5.1).

The allowlist is enforced here, in code, by a feature-name validator -- not by
convention. Section 2 ("Checkpoint discipline") requires that Module B never
sees the 96 h or 168 h checkpoints, directly or through a derived feature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from burnin.config import Config

ID_COLUMNS = ["component_id", "lot_id", "wafer_id", "slot_id"]
DIAGNOSTIC_COLUMNS = ["chamber_id", "defect_class"]
LABEL_COLUMN = "is_defective"

# A feature name mentions a checkpoint through one of these tokens.
_HOUR_TOKEN = re.compile(r"(?:^|_)(?:v|value_)?(\d+)h(?:_|$)")


class CheckpointLeakError(AssertionError):
    """Raised when a Module B feature depends on a forbidden checkpoint."""


@dataclass(frozen=True)
class FeatureAllowlist:
    """Explicit allowlist of checkpoints a feature set may depend on."""

    allowed_hours: tuple[int, ...]
    all_hours: tuple[int, ...]

    @property
    def forbidden_hours(self) -> tuple[int, ...]:
        return tuple(h for h in self.all_hours if h not in self.allowed_hours)

    def hours_mentioned(self, feature_name: str) -> set[int]:
        return {int(m) for m in _HOUR_TOKEN.findall(feature_name.lower())}

    def validate(self, feature_names: list[str]) -> None:
        """Hard-fail if any feature names a checkpoint outside the allowlist."""
        offenders: dict[str, set[int]] = {}
        for name in feature_names:
            bad = self.hours_mentioned(name) & set(self.forbidden_hours)
            if bad:
                offenders[name] = bad
        if offenders:
            detail = ", ".join(f"{k} -> {sorted(v)}h" for k, v in sorted(offenders.items()))
            raise CheckpointLeakError(
                f"Module B feature set depends on forbidden checkpoints "
                f"(allowed: {list(self.allowed_hours)}h): {detail}"
            )


def module_b_allowlist(cfg: Config) -> FeatureAllowlist:
    return FeatureAllowlist(
        allowed_hours=tuple(cfg.early_allowlist),
        all_hours=tuple(cfg.checkpoints),
    )


def wide_columns(cfg: Config) -> list[str]:
    """Full wide-format column list. One row per component (Section 2)."""
    cols = list(ID_COLUMNS)
    for p in cfg.parameters:
        for h in cfg.checkpoints:
            cols.append(p.col(h))
    for p in cfg.parameters:
        cols.append(p.limit_col)
    cols.extend(DIAGNOSTIC_COLUMNS)
    cols.append(LABEL_COLUMN)
    return cols
