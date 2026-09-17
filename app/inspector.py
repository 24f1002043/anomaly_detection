"""Section 10 - the inspector dashboard.

One row per component: part ID, lot, parameter, current value, lot median and
MAD, robust z, severity, predicted Value_168h with interval, margin to the
safety slope, disposition, and explanation. The inspector can drill from any
disposition into the four-point trajectory against the lot band and the ranked
feature contributions.

Lot integrity alarms surface at the top of the queue, above individual part
flags: a whole suspect lot is a more important finding than a single suspect
part.

Run:  streamlit run app/inspector.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402

from burnin.config import load_config  # noqa: E402
from burnin.decision import combine, early_decision, make_thresholds, terminal_decision  # noqa: E402
from burnin.explain import Explainer  # noqa: E402
from burnin.features import build_module_a_features  # noqa: E402
from burnin.metrics import select_operating_point  # noqa: E402
from burnin.module_a import apply_pooled_fallback, assess_lot_integrity  # noqa: E402
from burnin.module_b import component_early_decision  # noqa: E402
from burnin.pipeline import load_data, split_frames  # noqa: E402

st.set_page_config(
    page_title="Burn-In Screening Inspector | SIH26170",
    page_icon="◉",
    layout="wide",
)

STATE_COLOUR = {"PASS": "#1a7f37", "WATCH": "#b7791f", "REJECT": "#b42318"}


# --------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading models and scoring the held-out lots...")
def load_everything():
    cfg = load_config()
    mdir = ROOT / "artifacts" / "models"
    if not (mdir / "module_a.joblib").exists():
        return None

    bundle = joblib.load(mdir / "module_a.joblib")
    model_a, band = bundle["model"], bundle["historical_band"]
    modules_b = {
        p.name: joblib.load(mdir / f"module_b_{p.name}.joblib") for p in cfg.parameters
    }

    df = load_data(cfg)
    tr, va, te = split_frames(df)

    def score(d):
        ig = assess_lot_integrity(d, cfg, historical_band=band)
        f, _ = build_module_a_features(d, cfg)
        f = apply_pooled_fallback(f, d, cfg, ig)
        alarm = d["lot_id"].isin(ig.alarmed_lots)
        alarm.index = d.index
        return f, model_a.score(f, lot_alarm=alarm), ig

    f_tr, r_tr, _ = score(tr)
    f_va, r_va, _ = score(va)
    f_te, r_te, ig_te = score(te)

    preds_te = {k: v.predict(te) for k, v in modules_b.items()}
    feats_b_te = {k: v.features(te) for k, v in modules_b.items()}
    early_te = component_early_decision(preds_te, te.index)

    op, info = select_operating_point(
        va["is_defective"].to_numpy(), r_va.severity.to_numpy(), cfg
    )
    thr = make_thresholds(op, r_tr.severity.to_numpy(), cfg,
                          calibration_feasible=bool(info.get("feasible", True)))
    term = terminal_decision(te, r_te, thr, cfg)
    dec = combine(term, early_decision(early_te))

    return dict(cfg=cfg, te=te, feats_a=f_te, res=r_te, dec=dec, preds=preds_te,
                feats_b=feats_b_te, modules_b=modules_b, integrity=ig_te,
                thresholds=thr, explainer=Explainer(cfg, modules_b))


S = load_everything()
if S is None:
    st.error(
        "No trained models found. Run `python scripts/run_all.py` first, "
        "then reload this page."
    )
    st.stop()

cfg, te, dec = S["cfg"], S["te"], S["dec"]
res, integrity, thr = S["res"], S["integrity"], S["thresholds"]

# --------------------------------------------------------------------- header
st.title("Component Burn-In Screening — Inspector Queue")
st.caption(
    "SIH26170 · AI-Driven Anomaly Detection in Component Burn-In & Screening · "
    f"held-out lots only ({te['lot_id'].nunique()} lots, {len(te):,} components)"
)

c = st.columns(6)
c[0].metric("Components", f"{len(te):,}")
c[1].metric("REJECT", f"{int((dec['final_state'] == 'REJECT').sum()):,}")
c[2].metric("WATCH", f"{int((dec['final_state'] == 'WATCH').sum()):,}")
c[3].metric("PASS", f"{int((dec['final_state'] == 'PASS').sum()):,}")
c[4].metric("Early-rejected @24 h", f"{int((dec['early_state'] == 'EARLY-REJECT').sum()):,}")
# Gross chamber time freed. Part of it is bought by scrapping parts that would
# have passed, so the honest split -- hours saved on true defects versus hours
# lost to false rejects -- is the one reported in docs/RESULTS.md.
c[5].metric(
    "Chamber-hours freed @24 h",
    f"{int((dec['early_state'] == 'EARLY-REJECT').sum()) * cfg.evaluation['chamber_hours_saved_per_part']:,.0f}",
    help="Gross hours released by early rejection, before separating true "
         "defects from wrongly rejected good parts. See docs/RESULTS.md 9.2 "
         "for the split.",
)

# ------------------------------------------------- lot integrity: top of queue
alarmed = sorted(integrity.alarmed_lots)
if alarmed:
    st.error(
        f"**Lot integrity alarm — {len(alarmed)} lot(s) under engineering review.** "
        "A whole suspect lot is a more important finding than a single suspect part. "
        "Lot-relative statistics for these lots have been replaced by a pooled "
        "cross-lot baseline."
    )
    lot_tbl = integrity.as_frame()
    lot_tbl = lot_tbl[lot_tbl["lot_integrity_alarm"]].copy()
    counts = te.groupby("lot_id").size()
    lot_tbl["components"] = lot_tbl["lot_id"].map(counts)
    lot_tbl["flagged_in_lot"] = lot_tbl["lot_id"].map(
        dec.assign(lot=te["lot_id"].to_numpy()).groupby("lot")["flagged"].sum()
    )
    lot_tbl["trigger"] = np.where(
        lot_tbl["lot_id"].isin(integrity.mad_alarmed_lots),
        np.where(lot_tbl["lot_id"].isin(integrity.tail_alarmed_lots),
                 "MAD + tail inflation", "MAD inflation"),
        "tail inflation",
    )
    show = ["lot_id", "trigger", "components", "flagged_in_lot"] + [
        c for c in lot_tbl.columns if c.startswith("tail_")
    ]
    st.dataframe(lot_tbl[show].round(2), use_container_width=True, hide_index=True)
else:
    st.success("No lot integrity alarms on the held-out lots.")

# ------------------------------------------------------------------ the queue
st.subheader("Component queue")
f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 2])
states = f1.multiselect("Disposition", ["REJECT", "WATCH", "PASS"], ["REJECT", "WATCH"])
params = f2.multiselect(
    "Responsible parameter", [p.name for p in cfg.parameters],
    [p.name for p in cfg.parameters],
)
only_early = f3.checkbox("Early-rejected only", value=False)
lot_filter = f4.selectbox("Lot", ["(all)"] + sorted(te["lot_id"].unique().tolist()))

view = dec.copy()
view["component_id"] = te["component_id"].to_numpy()
view["lot_id"] = te["lot_id"].to_numpy()
view["wafer_id"] = te["wafer_id"].to_numpy()
view["slot_id"] = te["slot_id"].to_numpy()

mask = view["final_state"].isin(states) & view["responsible_param"].isin(params)
if only_early:
    mask &= view["early_state"] == "EARLY-REJECT"
if lot_filter != "(all)":
    mask &= view["lot_id"] == lot_filter
view = view[mask].sort_values("severity", ascending=False)

# enrich with the per-parameter numbers the inspector needs in the row itself
rows = []
term_h = cfg.terminal_hour
for i in view.index[:500]:
    param = view.loc[i, "responsible_param"]
    p = cfg.param(param)
    pred = S["preds"][param]
    pos = te.index.get_loc(i)
    lot = view.loc[i, "lot_id"]
    lot_vals = te.loc[te["lot_id"] == lot, p.col(term_h)]
    med = float(lot_vals.median())
    mad = float((lot_vals - med).abs().median())
    rows.append(
        {
            "component_id": view.loc[i, "component_id"],
            "lot_id": lot,
            "wafer/slot": f"{view.loc[i, 'wafer_id']}/{view.loc[i, 'slot_id']}",
            "disposition": view.loc[i, "final_state"],
            "parameter": param,
            f"value@{term_h}h ({p.units})": round(float(te.loc[i, p.col(term_h)]), 4),
            "lot median": round(med, 4),
            "lot MAD": round(mad, 4),
            "robust z": round(float(res.primary_z.loc[i, param]), 2),
            "severity": round(float(view.loc[i, "severity"]), 4),
            f"pred 168h ({p.units})": round(float(pred.point[pos]), 4),
            "interval": f"[{pred.lower[pos]:.3g}, {pred.upper[pos]:.3g}]",
            "margin (u/h)": round(float(pred.margin[pos]), 5),
            "early": view.loc[i, "early_state"],
            "reason": view.loc[i, "final_reason"],
        }
    )
queue = pd.DataFrame(rows)
st.caption(
    f"{int(mask.sum()):,} components match; showing the {len(queue):,} most severe. "
    f"Operating point: alert ≥ {thr.alert:.4f}, reject ≥ {thr.reject:.4f}."
)
st.dataframe(queue, use_container_width=True, hide_index=True, height=340)

# --------------------------------------------------------------- the drill-down
st.subheader("Drill-down")
if len(queue) == 0:
    st.info("No components match the current filters.")
    st.stop()

pick = st.selectbox("Component", queue["component_id"].tolist())
idx = view.index[view["component_id"] == pick][0]
param = view.loc[idx, "responsible_param"]
p = cfg.param(param)
pos = te.index.get_loc(idx)
pred = S["preds"][param]

rec = S["explainer"].build(
    te, S["feats_a"], res, view, S["preds"], S["feats_b"], [idx]
)[0]

left, right = st.columns([3, 2])

with left:
    st.markdown(f"#### Trajectory — {p.label} ({p.units})")
    tj = rec.trajectory
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tj["hours"] + tj["hours"][::-1],
        y=tj["lot_p90"] + tj["lot_p10"][::-1],
        fill="toself", fillcolor="rgba(120,140,170,0.18)",
        line=dict(width=0), name="lot p10–p90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tj["hours"], y=tj["lot_median"],
                             mode="lines+markers", name="lot median",
                             line=dict(color="#5a6b85", dash="dash")))
    fig.add_trace(go.Scatter(x=tj["hours"], y=tj["values"],
                             mode="lines+markers", name=pick,
                             line=dict(color=STATE_COLOUR[rec.disposition], width=3),
                             marker=dict(size=9)))
    fig.add_trace(go.Scatter(
        x=[cfg.terminal_hour], y=[float(pred.point[pos])], mode="markers",
        name="Module B prediction from 0 h + 24 h",
        marker=dict(symbol="diamond", size=13, color="#6b46c1"),
        error_y=dict(type="data", symmetric=False,
                     array=[float(pred.upper[pos] - pred.point[pos])],
                     arrayminus=[float(pred.point[pos] - pred.lower[pos])])))
    fig.add_hline(y=p.limit, line_dash="dot", line_color="#b42318",
                  annotation_text=f"datasheet limit {p.limit} {p.units}")
    fig.add_vrect(x0=0, x1=24, fillcolor="rgba(107,70,193,0.07)", line_width=0,
                  annotation_text="Module B sees only this window", annotation_position="top left")
    fig.update_layout(height=430, xaxis_title="burn-in hours",
                      yaxis_title=f"{p.label} ({p.units})",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown(f"#### Disposition: :{'red' if rec.disposition=='REJECT' else 'orange' if rec.disposition=='WATCH' else 'green'}[{rec.disposition}]")
    st.markdown(f"**Reason:** `{rec.reason}`")
    st.markdown(f"**Responsible parameter:** `{rec.responsible_parameter}` ({p.label})")
    if rec.lot_integrity_alarm:
        st.warning("This part belongs to a lot under an integrity alarm.")

    st.markdown("**Lot-relative arithmetic** (checkable by hand)")
    st.info(rec.lot_arithmetic["text"])
    st.caption(
        "Reproduces from the quoted median and MAD: "
        f"{'yes' if rec.lot_arithmetic['reproduces'] else 'NO'}"
    )

    m = rec.margin
    st.markdown("**Margin to the safety slope**")
    st.code(
        f"predicted Value_168h = {m['predicted_168h']:.4g} {p.units}\n"
        f"90% interval         = [{m['interval_lower']:.4g}, {m['interval_upper']:.4g}]\n"
        f"datasheet limit Lmax = {m['datasheet_limit']:.4g} {p.units}\n"
        f"reliability margin M = {m['reliability_margin_M']:.4g} {p.units}\n"
        f"S_safe               = {m['s_safe']:.6f} {p.units}/h\n"
        f"S_pred               = {m['s_pred']:.6f} {p.units}/h\n"
        f"margin               = {m['margin_to_safety_slope']:+.6f} {p.units}/h\n"
        f"early rejection      = {m['early_reject']}  ({m['early_reject_reason']})",
        language="text",
    )

st.markdown("#### Why this part was flagged")
ca, cb = st.columns(2)
with ca:
    st.markdown("**Module A** — ranked robust deviations against the lot (median/MAD units)")
    st.dataframe(pd.DataFrame(rec.module_a_deviations), hide_index=True,
                 use_container_width=True)
with cb:
    st.markdown("**Module B** — TreeSHAP contributions to the predicted Value_168h")
    sh = pd.DataFrame(rec.shap_contributions)
    if len(sh):
        sh["shap"] = sh["shap"].round(5)
        sh["value"] = sh["value"].round(5)
    st.dataframe(sh, hide_index=True, use_container_width=True)

with st.expander("Complete explanation record (JSON, as exported for audit)"):
    st.json(rec.to_dict())

st.caption(
    "Every flag carries this record, generated automatically, with no free-text "
    "narration. Coverage is binary and must be total — a flag without a complete "
    "record is a defect in the system (Section 8.3)."
)
