"""Run every stage end to end. This is the single command that reproduces
every number reported in docs/RESULTS.md."""
import argparse
import time

import _bootstrap  # noqa: F401
from burnin.config import load_config
from burnin.pipeline import (
    stage_evaluate,
    stage_generate,
    stage_train_module_a,
    stage_train_module_b,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-explain", type=int, default=400)
    a = ap.parse_args()
    cfg = load_config(a.config)
    t0 = time.time()
    print("=" * 78)
    print("SIH26170 - AI-Driven Anomaly Detection in Component Burn-In & Screening")
    print(f"config: {cfg.path}   seed: {cfg.seed}")
    print("=" * 78)
    stage_generate(cfg)
    stage_train_module_a(cfg)
    stage_train_module_b(cfg)
    stage_evaluate(cfg, n_explain=a.n_explain)
    print("=" * 78)
    print(f"complete in {time.time() - t0:.1f}s -> artifacts/results/")
