"""Lot-level splitting (Section 5.2).

Splitting is by ``lot_id``. Entire lots are held out, never individual parts
within a lot. Splitting within a lot leaks, because lot median and lot MAD are
computed from the lot's own members and would carry test information into
training.

The split is written to disk (Section 10, Reproducibility) so results can be
re-derived exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from burnin.config import Config

TRAIN, VALID, TEST = "train", "valid", "test"


def make_lot_split(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assign every lot to train / valid / test.

    ``valid`` is carved out of the training lots and is what the operating point
    is chosen on (Section 8.1). ``test`` is touched once, for the reported
    held-out numbers.
    """
    ev = cfg.evaluation
    rng = np.random.default_rng(cfg.seed + 1)
    lots = np.sort(df["lot_id"].unique())
    order = rng.permutation(len(lots))
    lots = lots[order]

    n = len(lots)
    n_test = int(round(float(ev["test_lot_fraction"]) * n))
    n_valid = int(round(float(ev["validation_lot_fraction"]) * n))

    assign = np.array([TRAIN] * n, dtype=object)
    assign[:n_test] = TEST
    assign[n_test : n_test + n_valid] = VALID
    return pd.DataFrame({"lot_id": lots, "split": assign}).sort_values("lot_id")


def attach_split(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    return df.merge(split, on="lot_id", how="left", validate="many_to_one")


def save_split(split: pd.DataFrame, path: Path) -> None:
    split.to_csv(path, index=False)


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_summary(df: pd.DataFrame) -> dict:
    g = df.groupby("split")
    return {
        str(k): {
            "lots": int(v["lot_id"].nunique()),
            "components": int(len(v)),
            "defective": int(v["is_defective"].sum()),
            "prevalence": float(v["is_defective"].mean()),
        }
        for k, v in g
    }


def grouped_folds(
    groups: pd.Series, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold over ``lot_id`` with a deterministic shuffle.

    scikit-learn's GroupKFold is not shuffled; shuffling the group order first
    keeps folds reproducible from the master seed while avoiding any ordering
    artefact from lot numbering.
    """
    uniq = np.sort(groups.unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of_group = {g: int(perm[i] % n_splits) for i, g in enumerate(uniq)}
    codes = groups.map(fold_of_group).to_numpy()
    idx = np.arange(len(groups))
    folds = []
    for f in range(n_splits):
        test_mask = codes == f
        folds.append((idx[~test_mask], idx[test_mask]))
    return folds


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
