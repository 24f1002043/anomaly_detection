"""End-to-end pipeline stages.

Each stage writes its artefacts to disk so the run is resumable and so every
reported figure can be traced back to the file it came from (Section 10,
Reproducibility).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from burnin.config import Config
from burnin.decision import (
    combine,
    disposition_counts,
    early_decision,
    make_thresholds,
    terminal_decision,
)
from burnin.evaluate import (
    build_ladder,
    evaluate_ladder,
    ladder_cost_frame,
    ladder_frame,
    latency_note,
    subgroup_breakdown,
)
from burnin.explain import Explainer, coverage_audit, records_to_frame
from burnin.features import build_module_a_features
from burnin.generator import BurnInGenerator, acceptance_report
from burnin.metrics import (
    detection_metrics,
    early_rejection_metrics,
    interval_coverage,
    lot_iqr,
    regression_metrics,
    select_operating_point,
)
from burnin.module_a import ModuleA, apply_pooled_fallback, assess_lot_integrity
from burnin.module_b import ModuleB, component_early_decision
from burnin.splits import attach_split, make_lot_split, split_summary

DATA = "burnin.parquet"
TRUTH = "truth.parquet"
SPLIT = "split.csv"


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set,)):
        return sorted(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ------------------------------------------------------------------- stage 1
def stage_generate(cfg: Config, verbose: bool = True) -> dict:
    t0 = time.time()
    gen = BurnInGenerator(cfg)
    ds = gen.generate()
    df = ds.frame
    split = make_lot_split(df, cfg)
    df = attach_split(df, split)

    data_dir = cfg.data_dir()
    df.to_parquet(data_dir / DATA, index=False)
    ds.truth.to_parquet(data_dir / TRUTH, index=False)
    split.to_csv(data_dir / SPLIT, index=False)

    acc = acceptance_report(df, cfg)
    meta = {
        **ds.meta,
        "acceptance": acc,
        "split": split_summary(df),
        "elapsed_s": round(time.time() - t0, 2),
    }
    _save_json(cfg.artifact_dir("results") / "01_generator.json", meta)

    if verbose:
        print(f"[1] generated {len(df):,} components in {meta['elapsed_s']}s")
        print(f"    prevalence                {acc['prevalence']:.4f}")
        print(f"    static-limit recall       {acc['static_limit_recall']:.4f}"
              f"   (accepted if < 0.50: {acc['accepted_minority_rule']})")
        print(f"    defects under limit @168h {acc['defects_under_limit_at_168h']:.4f}"
              f"   (accepted if >= {cfg.difficulty['latent_under_limit_target']}: "
              f"{acc['accepted_latent_target']})")
        if not (acc["accepted_minority_rule"] and acc["accepted_latent_target"]):
            raise SystemExit(
                "GENERATOR REJECTED by the Section 3.3 acceptance criterion. "
                "The data is too easy; retune before reporting any model result."
            )
    return meta


def load_data(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.data_dir() / DATA)


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        df[df["split"] == "train"].copy(),
        df[df["split"] == "valid"].copy(),
        df[df["split"] == "test"].copy(),
    )


# ------------------------------------------------------------------- stage 2
@dataclass
class ScoredSplit:
    frame: pd.DataFrame
    feats: pd.DataFrame
    result: Any
    integrity: Any


def _score_split(
    cfg: Config, model: ModuleA, d: pd.DataFrame, band: dict[str, float] | None
) -> ScoredSplit:
    ig = assess_lot_integrity(d, cfg, historical_band=band)
    feats, _ = build_module_a_features(d, cfg)
    feats = apply_pooled_fallback(feats, d, cfg, ig)
    alarm = d["lot_id"].isin(ig.alarmed_lots)
    alarm.index = d.index
    res = model.score(feats, lot_alarm=alarm)
    return ScoredSplit(frame=d, feats=feats, result=res, integrity=ig)


def stage_train_module_a(cfg: Config, verbose: bool = True) -> dict:
    t0 = time.time()
    df = load_data(cfg)
    tr, va, _ = split_frames(df)

    integ = assess_lot_integrity(tr, cfg)
    feats, per_param = build_module_a_features(tr, cfg)
    feats = apply_pooled_fallback(feats, tr, cfg, integ)
    model = ModuleA(cfg).fit(feats, per_param)

    # The R1 critical value is CALIBRATED (Section 4.2), on validation lots,
    # against the same 20:1 cost asymmetry as everything else. The WATCH state
    # fires on a single-parameter exceedance of this value, so leaving it at a
    # textbook 3 sigma would flood the inspector queue with healthy parts.
    ig_va = assess_lot_integrity(va, cfg, historical_band=integ.historical_band)
    feats_va, _ = build_module_a_features(va, cfg)
    feats_va = apply_pooled_fallback(feats_va, va, cfg, ig_va)
    zcal = model.calibrate_z_critical(feats_va, va["is_defective"].to_numpy())

    mdir = cfg.artifact_dir("models")
    joblib.dump(
        {"model": model, "historical_band": integ.historical_band}, mdir / "module_a.joblib"
    )

    # lot-integrity behaviour is a reported result in its own right
    lot_tbl = integ.as_frame()
    lot_tbl.to_csv(cfg.artifact_dir("results") / "lot_integrity_train.csv", index=False)
    out = {
        "n_train": int(len(tr)),
        "n_features_per_parameter": {k: len(v) for k, v in per_param.items()},
        "z_critical_calibration": zcal,
        "isolation_forest_contamination": float(
            min(0.5, cfg.module_a["isolation_forest"]["contamination_multiplier"]
                * cfg.population["defect_prevalence"])
        ),
        "lot_integrity": {
            "mad_triggered": sorted(integ.mad_alarmed_lots),
            "tail_triggered": sorted(integ.tail_alarmed_lots),
            "alarmed": sorted(integ.alarmed_lots),
            "historical_band": integ.historical_band,
            "mad_factor": integ.factor,
            "tail_factor": integ.tail_factor,
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    _save_json(cfg.artifact_dir("results") / "02_module_a.json", out)
    if verbose:
        print(f"[2] Module A fitted on {len(tr):,} parts in {out['elapsed_s']}s")
        print(f"    features/parameter {out['n_features_per_parameter']}")
        print(f"    lot integrity: MAD trigger {len(integ.mad_alarmed_lots)} lots, "
              f"tail trigger {len(integ.tail_alarmed_lots)} lots")
        print(f"    R1 critical value calibrated to z = {zcal['z_critical']:g}")
    return out


# ------------------------------------------------------------------- stage 3
def stage_train_module_b(cfg: Config, verbose: bool = True) -> dict:
    t0 = time.time()
    df = load_data(cfg)
    tr, _, _ = split_frames(df)
    mdir = cfg.artifact_dir("models")
    out: dict[str, Any] = {"parameters": {}}

    for p in cfg.parameters:
        ts = time.time()
        mb = ModuleB(cfg, p.name).fit(tr)
        joblib.dump(mb, mdir / f"module_b_{p.name}.joblib")
        oof = mb.oof_
        assert oof is not None
        iqr = lot_iqr(tr, p.col(cfg.terminal_hour))
        rm = regression_metrics(
            oof["y_true"].to_numpy(), oof["y_oof"].to_numpy(), p.name, p.units, iqr
        )
        assert mb.slope is not None
        out["parameters"][p.name] = {
            "oof_metrics": rm.to_dict(),
            "conformal_q_lo": mb.q_lo,
            "conformal_q_hi": mb.q_hi,
            "safety_slope": {
                "datasheet_limit": mb.slope.limit,
                "sigma_meas": mb.slope.sigma_meas,
                "reliability_margin_M": mb.slope.margin_M,
                "M_rule": "max(0.10 * L_max, 3 * sigma_meas)",
                "limit_minus_M": mb.slope.limit - mb.slope.margin_M,
            },
            "n_features": len(mb.feature_names),
            "elapsed_s": round(time.time() - ts, 2),
        }
        if verbose:
            print(f"[3] Module B [{p.name}] OOF MAE={rm.mae:.4f} {p.units} "
                  f"nMAE={rm.normalised_mae:.3f} R2={rm.r2:.4f} "
                  f"M={mb.slope.margin_M:.3f} ({out['parameters'][p.name]['elapsed_s']}s)")
    out["elapsed_s"] = round(time.time() - t0, 2)
    _save_json(cfg.artifact_dir("results") / "03_module_b.json", out)
    return out


# ------------------------------------------------------------------- stage 4
def stage_evaluate(cfg: Config, verbose: bool = True, n_explain: int = 400) -> dict:
    t0 = time.time()
    df = load_data(cfg)
    tr, va, te = split_frames(df)
    mdir = cfg.artifact_dir("models")
    rdir = cfg.artifact_dir("results")

    bundle = joblib.load(mdir / "module_a.joblib")
    model_a: ModuleA = bundle["model"]
    band = bundle["historical_band"]
    modules_b = {
        p.name: joblib.load(mdir / f"module_b_{p.name}.joblib") for p in cfg.parameters
    }

    sc_tr = _score_split(cfg, model_a, tr, band)
    sc_va = _score_split(cfg, model_a, va, band)
    sc_te = _score_split(cfg, model_a, te, band)

    # --- Module B predictions on every split -------------------------------
    preds_va, preds_te, feats_b_te = {}, {}, {}
    reg_rows, cov_rows = [], []
    for p in cfg.parameters:
        mb = modules_b[p.name]
        preds_va[p.name] = mb.predict(va)
        pr = mb.predict(te)
        preds_te[p.name] = pr
        feats_b_te[p.name] = mb.features(te)

        y = mb.target(te)
        iqr = lot_iqr(te, p.col(cfg.terminal_hour))
        rm = regression_metrics(y, pr.point, p.name, p.units, iqr)
        row = {"predictor": "random_forest", **rm.to_dict(), "lot_iqr": round(iqr, 5)}
        reg_rows.append(row)
        for bname, bpred in mb.baseline_predictions(te).items():
            bm = regression_metrics(y, bpred, p.name, p.units, iqr)
            reg_rows.append({"predictor": bname, **bm.to_dict(), "lot_iqr": round(iqr, 5)})
        cov = interval_coverage(y, pr.lower, pr.upper)
        cov_rows.append({"parameter": p.name, "units": p.units, **cov})

    reg = pd.DataFrame(reg_rows)
    reg.to_csv(rdir / "module_b_accuracy.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(rdir / "module_b_interval_coverage.csv", index=False)

    early_va = component_early_decision(preds_va, va.index)
    early_te = component_early_decision(preds_te, te.index)

    y_va = va["is_defective"].to_numpy()
    y_te = te["is_defective"].to_numpy()

    # --- Section 8.4 ablation ladder ---------------------------------------
    rungs = build_ladder(
        cfg, va, te, sc_va.result, sc_te.result,
        early_va["early_reject"].to_numpy(), early_te["early_reject"].to_numpy(),
    )
    rungs = evaluate_ladder(rungs, y_va, y_te, cfg)
    ladder = ladder_frame(rungs)
    ladder.to_csv(rdir / "ablation_ladder.csv", index=False)
    ladder_cost_frame(rungs).to_csv(rdir / "ablation_ladder_cost_optimal.csv", index=False)

    # --- Section 6 decision layer at the calibrated operating point ---------
    op, info = select_operating_point(y_va, sc_va.result.severity.to_numpy(), cfg)
    thresholds = make_thresholds(
        op,
        sc_tr.result.severity.to_numpy(),
        cfg,
        calibration_feasible=bool(info.get("feasible", True)),
    )
    term = terminal_decision(te, sc_te.result, thresholds, cfg)
    earl = early_decision(early_te)
    decisions = combine(term, earl)
    decisions.insert(0, "component_id", te["component_id"].to_numpy())
    decisions.insert(1, "lot_id", te["lot_id"].to_numpy())
    decisions["is_defective"] = y_te
    decisions.to_parquet(rdir / "decisions_test.parquet", index=False)

    flagged = decisions["flagged"].to_numpy()
    # PR-AUC for the deployed system is the PR-AUC of the severity ranking that
    # drives it (the combined detector). The final disposition additionally
    # OR-s in binary rules -- early rejection, static limits, lot alarms -- which
    # have no ranking of their own, so a PR-AUC computed on the binary flag
    # would be meaningless.
    final_metrics = detection_metrics(
        y_te, sc_te.result.severity.to_numpy(), float("nan"), cfg,
        flags=flagged,
        pr_auc=float(ladder.loc[ladder.configuration == "combined_detector",
                                "pr_auc"].iloc[0]),
    )
    early_metrics = early_rejection_metrics(
        y_te, early_te["early_reject"].to_numpy(), cfg
    )
    subgroups = subgroup_breakdown(y_te, flagged, te["defect_class"], cfg)
    subgroups.to_csv(rdir / "subgroup_recall.csv", index=False)

    # --- Section 7 explanation records for every flagged part ---------------
    flagged_idx = list(decisions.index[flagged][:n_explain])
    explainer = Explainer(cfg, modules_b)
    records = explainer.build(
        te, sc_te.feats, sc_te.result, decisions, preds_te, feats_b_te, flagged_idx
    )
    audit = coverage_audit(records)
    rf = records_to_frame(records)
    rf.to_csv(rdir / "explanations_sample.csv", index=False)
    _save_json(rdir / "explanation_example.json",
               records[0].to_dict() if records else {})

    lot_te = sc_te.integrity.as_frame()
    lot_te.to_csv(rdir / "lot_integrity_test.csv", index=False)
    contaminated = set(
        json.loads((rdir / "01_generator.json").read_text(encoding="utf-8"))[
            "contaminated_lots"
        ]
    )
    test_lots = set(te["lot_id"].unique())
    contam_in_test = contaminated & test_lots

    out = {
        "operating_point": {"value": float(op), **info,
                            "thresholds": thresholds.to_dict()},
        "final_system": final_metrics.to_dict(),
        "early_rejection": early_metrics,
        "dispositions": disposition_counts(decisions, te["is_defective"]),
        "ablation_ladder": ladder.to_dict(orient="records"),
        "module_b_accuracy": reg.to_dict(orient="records"),
        "interval_coverage": cov_rows,
        "subgroup_recall": subgroups.to_dict(orient="records"),
        "explainability_audit": audit,
        "lot_integrity_test": {
            "contaminated_lots_in_test": sorted(contam_in_test),
            "alarmed_lots": sorted(sc_te.integrity.alarmed_lots),
            "mad_triggered": sorted(sc_te.integrity.mad_alarmed_lots),
            "tail_triggered": sorted(sc_te.integrity.tail_alarmed_lots),
            "true_positive": len(sc_te.integrity.alarmed_lots & contam_in_test),
            "false_positive": len(sc_te.integrity.alarmed_lots - contam_in_test),
            "false_negative": len(contam_in_test - sc_te.integrity.alarmed_lots),
        },
        "latency": latency_note(cfg),
        "n_test": int(len(te)),
        "n_test_defective": int(y_te.sum()),
        "elapsed_s": round(time.time() - t0, 2),
    }
    _save_json(rdir / "04_evaluation.json", out)

    if verbose:
        print(f"[4] evaluation complete in {out['elapsed_s']}s")
        print(f"    operating point {op:.6f}  (alert {thresholds.alert:.6f} / "
              f"reject {thresholds.reject:.6f})")
        print(f"    FINAL escape={final_metrics.escape_rate:.4f} "
              f"recall={final_metrics.recall:.4f} F2={final_metrics.f_beta:.4f} "
              f"PR-AUC={final_metrics.pr_auc:.4f} yield_loss={final_metrics.false_reject_rate:.4f}")
        print(f"    early rejection: recall={early_metrics['early_recall']:.4f} "
              f"chamber-hours saved={early_metrics['chamber_hours_saved']:,.0f}")
        print(f"    explainability coverage={audit['coverage']:.4f} "
              f"arithmetic reproduces={audit['lot_arithmetic_reproduces']:.4f}")
    return out
