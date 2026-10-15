# Appendix A: External datasets

No public dataset reproduces parametric burn-in telemetry with hidden 168 h
terminal values and defect labels. The datasets below were reviewed and are
**not part of the evaluation of either module**; none of them scores against the
three official metrics. They are recorded here to document what was considered
and why it was set aside.

Run `python scripts/download_reference_datasets.py` to refresh the catalogue.
It writes `reference_datasets/catalogue.json`.

| Dataset | What it actually is | Why it is not used for scoring |
|---|---|---|
| **NASA SMAP / MSL** | Spacecraft telemetry, 82 channels, 105 labelled anomaly sequences of point and contextual type. | Anonymised time axis and pre-scaled values. No 168 h terminal target and no component identity, so neither module can be scored on it. |
| **NASA IMS** | Run-to-failure bearing vibration, roughly 164 h of 10-minute files. | Vibration, not parametric electrical data. The degradation shape is instructive but the features do not transfer. |
| **CWRU** | Seeded bearing faults at steady state across several loads. | Classification data, not run-to-failure. Provides no drift trajectory and no remaining-life signal. |

## What the review actually checked

The SMAP/MSL label index is small and openly served, so it was downloaded and
parsed rather than described from memory. Reading
`reference_datasets/nasa_smap_msl/labeled_anomalies.csv` confirms the
characterisation above directly from the data:

| Property | Measured |
|---|---|
| Rows (channel segments) | 82 |
| Distinct channel IDs | 81 |
| Spacecraft | MSL, SMAP |
| Labelled anomaly sequences | **105** |
| - of type `point` | 62 |
| - of type `contextual` | 43 |
| Total telemetry values | 517,764 |
| Columns | `chan_id`, `spacecraft`, `anomaly_sequences`, `class`, `num_values` |

The exclusion follows from the schema, not from an opinion about the data. A row
is a per-channel telemetry segment identified by `chan_id`. There is no part, lot
or wafer identity, so the lot-relative statistics that Module A is built on
cannot be formed. There is no terminal parametric value at a fixed horizon, so
the Module B target `Value_0h + Value_24h → Value_168h` cannot be constructed.
Values are pre-scaled and the time axis is anonymised, so even the units the
safety slope is expressed in are unavailable.

The IMS and CWRU sets publish no machine-readable index; both are served from
landing pages that require an interactive download. Their bulk archives run to
gigabytes and nothing here is scored on them, so the script records the source
and licence without fetching them. The SMAP/MSL bulk archive
(`s3-us-west-2.amazonaws.com/telemanom/data.zip`) now returns HTTP 403 and is
likewise not required; pass `--full` to attempt bulk downloads anyway.

## The consequence

Because no external dataset can be scored, the evaluation runs entirely on the
synthetic generator of Section 3. That places the whole weight of the submission
on the realism of the generator, which is why it carries an explicit acceptance
criterion (Section 3.3) that runs before any model is fitted:

> The generator is only accepted if the best achievable static-limit baseline
> catches a clear minority of the defective population on a held-out run.

`scripts/01_generate_data.py` exits non-zero if that gate fails, and
`tests/test_generator.py` asserts it. The measured value is reported in
[RESULTS.md](RESULTS.md) §9.0 alongside the model results, so a reader can see
what the models were actually up against.

## Claims to avoid

- That SMAP, MSL or CWRU data predicts 168 h burn-in failure.
- That CWRU data gives remaining useful life.
- That these synthetic results establish ISRO field escape rates.
- That the system is qualified or validated on flight hardware.

Deployment requires recalibration on real rig data with real terminal labels.

## Source

SIH 2026 problem statement SIH26170, Indian Space Research Organisation,
Department of Space, Software, Smart Automation. Official listing:
`sih.gov.in/sih2026PS`.
