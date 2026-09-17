"""Stage 2 - fit Module A (robust z-rule calibration + Isolation Forests)."""
import argparse
import _bootstrap  # noqa: F401
from burnin.config import load_config
from burnin.pipeline import stage_train_module_a

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    stage_train_module_a(load_config(a.config))
