"""Appendix A - external datasets.

No public dataset reproduces parametric burn-in telemetry with hidden 168 h
terminal values and defect labels. The datasets below were reviewed and are NOT
part of the evaluation of either module; none of them scores against the three
official metrics. They are recorded here only to document what was considered
and why it was set aside.

This script fetches what is small and openly served -- principally the SMAP/MSL
anomaly label index -- so that the review is substantiated from the data itself
rather than asserted. Bulk archives are not downloaded by default: they run to
gigabytes, and nothing in this submission is scored on them. Pass --full to
attempt them anyway.

Run:  python scripts/download_reference_datasets.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

import _bootstrap  # noqa: F401
from burnin.config import REPO_ROOT

OUT = REPO_ROOT / "reference_datasets"
TIMEOUT = 60

CATALOGUE = [
    {
        "key": "nasa_smap_msl",
        "name": "NASA SMAP / MSL",
        "what_it_is": (
            "Spacecraft telemetry, 82 channels, 105 labelled anomaly sequences of "
            "point and contextual type."
        ),
        "why_not_used": (
            "Anonymised time axis and pre-scaled values. No 168 h terminal target "
            "and no component identity, so neither module can be scored on it."
        ),
        "metadata_url": (
            "https://raw.githubusercontent.com/khundman/telemanom/master/"
            "labeled_anomalies.csv"
        ),
        "bulk_url": "https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
        "source": "https://github.com/khundman/telemanom",
        "licence": "Apache-2.0 (code); NASA telemetry released for research use",
    },
    {
        "key": "nasa_ims",
        "name": "NASA IMS bearing",
        "what_it_is": (
            "Run-to-failure bearing vibration, roughly 164 h of 10-minute files."
        ),
        "why_not_used": (
            "Vibration, not parametric electrical data. The degradation shape is "
            "instructive but the features do not transfer."
        ),
        "metadata_url": None,
        "bulk_url": None,
        "source": (
            "https://www.nasa.gov/intelligent-systems-division/"
            "discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
        ),
        "licence": "NASA open data, attribution requested",
    },
    {
        "key": "cwru_bearing",
        "name": "CWRU bearing",
        "what_it_is": "Seeded bearing faults at steady state across several loads.",
        "why_not_used": (
            "Classification data, not run-to-failure. Provides no drift trajectory "
            "and no remaining-life signal."
        ),
        "metadata_url": None,
        "bulk_url": None,
        "source": "https://engineering.case.edu/bearingdatacenter",
        "licence": "Case Western Reserve University, free for research use",
    },
]


def fetch(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return {"ok": True, "bytes": len(r.content), "path": str(dest.relative_to(REPO_ROOT))}
    except Exception as e:  # network conditions are part of the report
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}


def summarise_smap_labels(path: Path) -> dict:
    """Read the label index and confirm, from the data, why it cannot be scored."""
    import pandas as pd

    import re

    df = pd.read_csv(path)
    # The 'class' column holds bare words inside brackets -- "[point, contextual]"
    # -- which is not JSON, so it is parsed with a word split rather than a
    # JSON loader. 'anomaly_sequences' is genuine JSON.
    classes = [w for s in df["class"] for w in re.findall(r"[A-Za-z]+", str(s))]
    return {
        "rows": int(len(df)),
        "channels": int(df["chan_id"].nunique()),
        "spacecraft": sorted(df["spacecraft"].unique().tolist()),
        "anomaly_sequences_total": int(
            df["anomaly_sequences"].map(lambda s: len(json.loads(s))).sum()
        ),
        "anomaly_class_counts": {c: classes.count(c) for c in sorted(set(classes))},
        "total_telemetry_values": int(df["num_values"].sum()),
        "columns": df.columns.tolist(),
        "confirms_exclusion": {
            "has_component_identity": False,
            "has_168h_terminal_target": False,
            "values_pre_scaled": True,
            "note": (
                "Rows are per-channel telemetry segments. There is no part, lot or "
                "wafer identity and no terminal parametric value, so neither the "
                "Module A lot-relative statistics nor the Module B 0 h + 24 h -> "
                "168 h target can be constructed from it."
            ),
        },
    }


def main(full: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for entry in CATALOGUE:
        rec = {k: entry[k] for k in
               ("key", "name", "what_it_is", "why_not_used", "source", "licence")}
        rec["used_for_scoring"] = False

        if entry["metadata_url"]:
            dest = OUT / entry["key"] / Path(entry["metadata_url"]).name
            rec["metadata"] = fetch(entry["metadata_url"], dest)
            if rec["metadata"].get("ok") and dest.suffix == ".csv":
                try:
                    rec["metadata"]["summary"] = summarise_smap_labels(dest)
                except Exception as e:
                    rec["metadata"]["summary_error"] = str(e)[:200]
        else:
            rec["metadata"] = {
                "ok": False,
                "error": "no machine-readable index published; landing page only",
            }

        if full and entry["bulk_url"]:
            dest = OUT / entry["key"] / Path(entry["bulk_url"]).name
            rec["bulk"] = fetch(entry["bulk_url"], dest)
        elif entry["bulk_url"]:
            rec["bulk"] = {"skipped": True, "reason": "not used for scoring; pass --full"}
        else:
            rec["bulk"] = {
                "skipped": True,
                "reason": "requires interactive download from the landing page",
            }
        report.append(rec)
        status = "ok" if rec["metadata"].get("ok") else "metadata unavailable"
        print(f"  {entry['name']:20s} {status}")

    path = OUT / "catalogue.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")
    print("None of these datasets is used for scoring. See docs/EXTERNAL_DATASETS.md.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also attempt the multi-gigabyte bulk archives")
    main(**vars(ap.parse_args()))
