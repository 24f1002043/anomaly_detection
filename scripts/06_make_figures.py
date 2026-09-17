"""Stage 6 - presentation figures, written to artifacts/figures/.

Each figure answers one question a reviewer will actually ask. Nothing here is
decoration: if a plot does not change what someone concludes, it is not drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

import _bootstrap  # noqa: F401
from burnin.config import REPO_ROOT, load_config  # noqa: E402
from burnin.features import build_module_a_features  # noqa: E402
from burnin.module_a import apply_pooled_fallback, assess_lot_integrity  # noqa: E402
from burnin.pipeline import load_data, split_frames  # noqa: E402

FIG = REPO_ROOT / "artifacts" / "figures"
RES = REPO_ROOT / "artifacts" / "results"

INK = "#1f2933"
GRID = "#d8dee9"
HEALTHY = "#6b7a8f"
BENIGN = "#2f80ed"
DEFECT = "#c0392b"
LOWSTART = "#e67e22"
ACCENT = "#6b46c1"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})

CLASS_STYLE = {
    "healthy": ("Healthy", HEALTHY),
    "benign_high_baseline": ("Benign high-baseline (trap)", BENIGN),
    "defect_standard": ("Latent defect", DEFECT),
    "defect_low_start_fast_drift": ("Low-start fast-drift defect (trap)", LOWSTART),
}


def fig_trajectories(df: pd.DataFrame, cfg) -> None:
    """The central picture: why a static limit cannot see these parts."""
    hours = cfg.checkpoints
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, p in zip(axes, cfg.parameters):
        for cls, (label, colour) in CLASS_STYLE.items():
            m = df["defect_class"] == cls
            if not m.any():
                continue
            med = [df.loc[m, p.col(h)].median() for h in hours]
            lo = [df.loc[m, p.col(h)].quantile(0.25) for h in hours]
            hi = [df.loc[m, p.col(h)].quantile(0.75) for h in hours]
            ax.plot(hours, med, "-o", color=colour, label=label, lw=2, ms=5)
            ax.fill_between(hours, lo, hi, color=colour, alpha=0.13, lw=0)
        ax.axhline(p.limit, color=DEFECT, ls=":", lw=1.6)
        ax.annotate(f"datasheet limit {p.limit:g} {p.units}",
                    xy=(hours[0], p.limit), xytext=(2, 4),
                    textcoords="offset points", color=DEFECT, fontsize=8)
        ax.axvspan(0, 24, color=ACCENT, alpha=0.06, lw=0)
        ax.set_title(f"{p.label}  ({p.units})", fontsize=11)
        ax.set_xlabel("burn-in hours")
        ax.set_xticks(hours)
    axes[0].set_ylabel("parameter value (median, IQR band)")
    axes[0].annotate("Module B\nsees only\nthis window", xy=(12, axes[0].get_ylim()[1]),
                     ha="center", va="top", fontsize=7.5, color=ACCENT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.09))
    fig.suptitle(
        "Latent defects drift ~30x faster under stress, yet most finish inside the "
        "datasheet limit", fontsize=12, y=1.02)
    fig.savefig(FIG / "01_trajectories_by_population.png")
    plt.close(fig)


def fig_lot_relative(df: pd.DataFrame, cfg) -> None:
    """Why the lot is the right reference frame."""
    p = cfg.param("iddq")
    col = p.col(cfg.terminal_hour)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    lots = df["lot_id"].drop_duplicates().iloc[:14].tolist()
    sub = df[df["lot_id"].isin(lots)]
    for i, lot in enumerate(lots):
        s = sub[sub["lot_id"] == lot]
        h = s[~s["is_defective"]]
        d = s[s["is_defective"]]
        ax1.scatter(np.full(len(h), i) + np.random.uniform(-.3, .3, len(h)),
                    h[col], s=3, color=HEALTHY, alpha=0.4, lw=0)
        if len(d):
            ax1.scatter(np.full(len(d), i) + np.random.uniform(-.3, .3, len(d)),
                        d[col], s=22, color=DEFECT, alpha=0.95, lw=0, zorder=3)
        ax1.hlines(s[col].median(), i - .38, i + .38, color=INK, lw=1.6, zorder=4)
    ax1.axhline(p.limit, color=DEFECT, ls=":", lw=1.6)
    ax1.set_xticks(range(len(lots)))
    ax1.set_xticklabels([f"{i}" for i in range(len(lots))], fontsize=7)
    ax1.set_xlabel("lot index")
    ax1.set_ylabel(f"{p.label} at 168 h ({p.units})")
    ax1.set_title("Absolute value: lot means differ, so one global\n"
                  "threshold is the wrong instrument", fontsize=10.5)

    feats, _ = build_module_a_features(df, cfg)
    z = feats["iddq_z_total_drift"]
    y = df["is_defective"].to_numpy()
    bins = np.linspace(-6, 40, 90)
    ax2.hist(np.clip(z[~y], -6, 40), bins=bins, color=HEALTHY, alpha=0.75,
             label="healthy", log=True)
    ax2.hist(np.clip(z[y], -6, 40), bins=bins, color=DEFECT, alpha=0.8,
             label="defective", log=True)
    ax2.set_xlabel("robust z of total drift, within lot")
    ax2.set_ylabel("components (log scale)")
    ax2.set_title("Lot-relative drift: the same parts separate\n"
                  "cleanly once the lot is the reference", fontsize=10.5)
    ax2.legend(frameon=False)
    fig.savefig(FIG / "02_lot_relative_separation.png")
    plt.close(fig)


def fig_ladder() -> None:
    """The ablation ladder: what each component actually contributes."""
    lad = pd.read_csv(RES / "ablation_ladder.csv")
    names = {
        "static_limit_only": "Static limit\nonly",
        "lot_relative_z_rule": "+ lot-relative\nz-rule",
        "isolation_forest": "Isolation Forest\nalone",
        "combined_detector": "Combined\ndetector",
        "combined_plus_module_b_early": "Combined +\nModule B early",
    }
    lad["label"] = lad["configuration"].map(names)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    x = np.arange(len(lad))

    cols = [DEFECT if v > 0.30 else ACCENT if v > 0.12 else "#1a7f37"
            for v in lad["escape_rate"]]
    ax1.bar(x, lad["escape_rate"], color=cols, width=0.62)
    for i, v in enumerate(lad["escape_rate"]):
        ax1.text(i, v + 0.012, f"{v:.1%}", ha="center", fontsize=9, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(lad["label"], fontsize=8.5)
    ax1.set_ylabel("escape rate  (FN / positives)")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylim(0, max(lad["escape_rate"]) * 1.22)
    ax1.set_title("Escape rate — lower is better (headline metric)", fontsize=10.5)

    ax2.plot(x, lad["yield_loss"], "-o", color=ACCENT, lw=2, ms=6,
             label="yield loss (false rejects)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(lad["label"], fontsize=8.5)
    ax2.set_ylabel("yield loss")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2b = ax2.twinx()
    ax2b.plot(x, lad["f_beta_2"], "-s", color=DEFECT, lw=2, ms=6, label="F2")
    ax2b.set_ylabel("F₂  (recall-weighted)")
    ax2b.grid(False)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="center right", fontsize=9)
    ax2.set_title("What the escape-rate reduction costs", fontsize=10.5)
    fig.suptitle("Ablation ladder, every rung at the same operating point",
                 fontsize=12, y=1.03)
    fig.savefig(FIG / "03_ablation_ladder.png")
    plt.close(fig)


def fig_module_b() -> None:
    """Module B against its three baselines, plus interval calibration."""
    cfg = load_config()
    reg = pd.read_csv(RES / "module_b_accuracy.csv")
    cov = pd.read_csv(RES / "module_b_interval_coverage.csv")
    order = ["last_value", "linear_extrapolation", "lot_median_trajectory",
             "random_forest"]
    labels = ["Last value", "Linear extrap.", "Lot median traj.", "Random Forest"]
    colours = [HEALTHY, HEALTHY, HEALTHY, ACCENT]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, p in zip(axes[:3], cfg.parameters):
        sub = reg[reg["parameter"] == p.name].set_index("predictor")
        vals = [sub.loc[o, "mae"] for o in order]
        ax.barh(range(4), vals, color=colours, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v * 1.02, i, f"{v:.4g}", va="center", fontsize=8.5)
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f"MAE ({p.units})")
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_title(f"{p.label}", fontsize=10.5)

    ax = axes[3]
    nominal = float(cfg.module_b["interval_nominal"])
    ax.bar(cov["parameter"], cov["coverage"], color=ACCENT, width=0.55)
    ax.axhline(nominal, color=DEFECT, ls="--", lw=1.6)
    ax.text(2.45, nominal + 0.004, f"nominal {nominal:.0%}", color=DEFECT,
            ha="right", fontsize=8.5)
    for i, v in enumerate(cov["coverage"]):
        ax.text(i, v - 0.035, f"{v:.1%}", ha="center", fontsize=9,
                color="white", weight="bold")
    ax.set_ylim(0.7, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Prediction-interval coverage\n(split-conformal, held-out lots)",
                 fontsize=10.5)
    fig.suptitle("Module B: Value_0h + Value_24h → Value_168h, against all "
                 "three baselines", fontsize=12, y=1.04)
    fig.savefig(FIG / "04_module_b_accuracy.png")
    plt.close(fig)


def fig_pr_curve(cfg) -> None:
    """PR curve, not ROC: prevalence is 1-3 %."""
    dec = pd.read_parquet(RES / "decisions_test.parquet")
    ev = json.loads((RES / "04_evaluation.json").read_text(encoding="utf-8"))
    y = dec["is_defective"].to_numpy().astype(bool)
    s = dec["severity"].to_numpy()
    prec, rec, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(rec, prec, color=ACCENT, lw=2.2, label="Module A severity")
    ax.axhline(y.mean(), color=HEALTHY, ls="--", lw=1.4,
               label=f"prevalence baseline ({y.mean():.2%})")
    f = ev["final_system"]
    ax.scatter([f["recall"]], [f["precision"]], s=110, color=DEFECT, zorder=5,
               marker="*", label="operating point (full system)")
    ax.annotate(f"escape {f['escape_rate']:.1%}\nyield loss {f['false_reject_rate']:.2%}",
                xy=(f["recall"], f["precision"]), xytext=(-12, -42),
                textcoords="offset points", fontsize=8.5, color=DEFECT,
                ha="right",
                arrowprops=dict(arrowstyle="-", color=DEFECT, lw=0.8))
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Precision-recall (PR-AUC = "
                 f"{ev['ablation_ladder'][3]['pr_auc']:.3f})\n"
                 "ROC-AUC would flatter this at 2 % prevalence", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.savefig(FIG / "05_precision_recall.png")
    plt.close(fig)


def fig_lot_integrity(df: pd.DataFrame, cfg) -> None:
    """The MAD-vs-tail-inflation finding, in one picture."""
    gen = json.loads((RES / "01_generator.json").read_text(encoding="utf-8"))
    contaminated = set(gen["contaminated_lots"])
    integ = assess_lot_integrity(df, cfg)
    mad = integ.lot_mad
    tail = integ.tail_inflation
    band = integ.historical_band
    ratio = pd.DataFrame({c: mad[c] / band[c] for c in mad.columns})
    is_c = ratio.index.isin(contaminated)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.3))
    ax1.scatter(ratio.loc[~is_c].max(axis=1), np.random.uniform(0, 1, (~is_c).sum()),
                s=14, color=HEALTHY, alpha=0.55, lw=0, label="clean lot")
    ax1.scatter(ratio.loc[is_c].max(axis=1), np.random.uniform(0, 1, is_c.sum()),
                s=70, color=DEFECT, lw=0, label="contaminated lot", zorder=3)
    ax1.axvline(integ.factor, color=INK, ls="--", lw=1.6)
    ax1.text(integ.factor, 1.04, f" trigger = {integ.factor:g}x", fontsize=8.5)
    ax1.set_xlabel("max over parameters of  lot MAD / historical band")
    ax1.set_yticks([])
    ax1.set_title("MAD inflation: no separation, trigger never fires\n"
                  "(MAD has a 50 % breakdown point)", fontsize=10.5)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax2.scatter(tail.loc[~is_c].max(axis=1), np.random.uniform(0, 1, (~is_c).sum()),
                s=14, color=HEALTHY, alpha=0.55, lw=0, label="clean lot")
    ax2.scatter(tail.loc[is_c].max(axis=1), np.random.uniform(0, 1, is_c.sum()),
                s=70, color=DEFECT, lw=0, label="contaminated lot", zorder=3)
    ax2.axvline(integ.tail_factor, color=INK, ls="--", lw=1.6)
    ax2.text(integ.tail_factor, 1.04, f" trigger = {integ.tail_factor:g}", fontsize=8.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("max over parameters of  (Q90 − median) / (1.4826 × MAD)")
    ax2.set_yticks([])
    ax2.set_title("Tail inflation: clean ~1.4, contaminated 8–60\n"
                  "separates with a 4x margin on both sides", fontsize=10.5)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Lot-contamination fallback: the specified trigger does not work, "
                 "and the run says so", fontsize=12, y=1.03)
    fig.savefig(FIG / "06_lot_integrity.png")
    plt.close(fig)


def fig_example_flag(cfg) -> None:
    """One flagged part, end to end, as the inspector sees it."""
    ex = json.loads((RES / "explanation_example.json").read_text(encoding="utf-8"))
    if not ex:
        return
    tj, mg, la = ex["trajectory"], ex["margin"], ex["lot_arithmetic"]
    p = cfg.param(ex["responsible_parameter"])

    fig = plt.figure(figsize=(13.4, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.05, 1],
                          hspace=0.55, wspace=0.22)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    h = tj["hours"]
    ax1.fill_between(h, tj["lot_p10"], tj["lot_p90"], color=HEALTHY, alpha=0.2,
                     lw=0, label="lot p10–p90")
    ax1.plot(h, tj["lot_median"], "--", color=HEALTHY, lw=1.8, label="lot median")
    ax1.plot(h, tj["values"], "-o", color=DEFECT, lw=2.4, ms=7,
             label=f"{ex['component_id']}")
    ax1.errorbar([168], [mg["predicted_168h"]],
                 yerr=[[mg["predicted_168h"] - mg["interval_lower"]],
                       [mg["interval_upper"] - mg["predicted_168h"]]],
                 fmt="D", color=ACCENT, ms=8, capsize=5,
                 label="Module B prediction from 0 h + 24 h only")
    ax1.axhline(mg["datasheet_limit"], color=DEFECT, ls=":", lw=1.6)
    ax1.axvspan(0, 24, color=ACCENT, alpha=0.07, lw=0)
    ax1.set_xticks(h)
    ax1.set_xlabel("burn-in hours")
    ax1.set_ylabel(f"{p.label} ({p.units})")
    # headroom above the limit line so the annotation and legend cannot collide
    top = max(mg["datasheet_limit"], max(tj["values"]), mg["interval_upper"])
    ax1.set_ylim(top=top * 1.30)
    ax1.text(h[-1], mg["datasheet_limit"], f"datasheet limit {mg['datasheet_limit']:g} ",
             va="bottom", ha="right", fontsize=8.5, color=DEFECT)
    ax1.text(12, ax1.get_ylim()[1], "Module B window ", ha="center", va="top",
             fontsize=8, color=ACCENT)
    ax1.set_title(f"{ex['component_id']} · {ex['lot_id']} · "
                  f"{ex['disposition']}", fontsize=11)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left",
               bbox_to_anchor=(0.02, 0.88))

    sh = ex["shap_contributions"]
    names = [c["feature"] for c in sh][::-1]
    vals = [c["shap"] for c in sh][::-1]
    ax2.barh(range(len(vals)), vals,
             color=[DEFECT if v > 0 else BENIGN for v in vals], height=0.6)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=8.5)
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_xlabel("TreeSHAP contribution to predicted Value_168h", fontsize=8.5)
    ax2.tick_params(labelsize=8.5)
    ax2.set_title("Why the model predicts what it predicts", fontsize=10)

    ax3.axis("off")
    lines = [
        f"Responsible parameter: {ex['responsible_parameter']}",
        "",
        _wrap(la["text"], 66),
        "",
        f"predicted Value_168h   = {mg['predicted_168h']:.4g} {mg['units']}",
        f"90 % interval          = [{mg['interval_lower']:.4g}, "
        f"{mg['interval_upper']:.4g}]",
        f"reliability margin M   = {mg['reliability_margin_M']:.4g} {mg['units']}",
        f"margin to safety slope = {mg['margin_to_safety_slope']:+.5f} "
        f"{mg['units']}/h",
        "",
        f"Disposition: {ex['disposition']}  ({ex['reason']})",
    ]
    txt = "\n".join(lines)
    ax3.text(0, 1, txt, fontsize=8, family="monospace", va="top", ha="left",
             transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.7", facecolor="#f6f8fa",
                       edgecolor=GRID))
    fig.suptitle("Every flag carries a complete record, generated automatically",
                 fontsize=12, y=0.99)
    fig.savefig(FIG / "07_example_flag.png")
    plt.close(fig)


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    df = load_data(cfg)
    _, _, te = split_frames(df)

    fig_trajectories(df, cfg); print("  01_trajectories_by_population.png")
    fig_lot_relative(te, cfg); print("  02_lot_relative_separation.png")
    fig_ladder(); print("  03_ablation_ladder.png")
    fig_module_b(); print("  04_module_b_accuracy.png")
    fig_pr_curve(cfg); print("  05_precision_recall.png")
    fig_lot_integrity(df, cfg); print("  06_lot_integrity.png")
    fig_example_flag(cfg); print("  07_example_flag.png")
    print(f"\nwrote 7 figures to {FIG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
