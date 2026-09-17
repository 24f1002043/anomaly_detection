"""Stage 1 - generate the synthetic burn-in run and check the Section 3.3
generator acceptance criterion. Exits non-zero if the data is too easy."""
import argparse
import _bootstrap  # noqa: F401
from burnin.config import load_config
from burnin.pipeline import stage_generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    stage_generate(load_config(a.config))
