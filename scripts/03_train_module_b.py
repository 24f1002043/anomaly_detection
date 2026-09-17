"""Stage 3 - fit Module B, one Random Forest per parameter, with grouped CV
over lot_id and a split-conformal interval calibrated on held-out lots."""
import argparse
import _bootstrap  # noqa: F401
from burnin.config import load_config
from burnin.pipeline import stage_train_module_b

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    stage_train_module_b(load_config(a.config))
