"""Stage 4 - held-out evaluation: ablation ladder, decision layer, explanation
records and the explainability audit."""
import argparse
import _bootstrap  # noqa: F401
from burnin.config import load_config
from burnin.pipeline import stage_evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-explain", type=int, default=400)
    a = ap.parse_args()
    stage_evaluate(load_config(a.config), n_explain=a.n_explain)
