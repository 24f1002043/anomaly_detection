# Section 9: Results

> Every figure below is produced by a held-out run on the formulation in
> the technical approach. Module B uses `Value_0h` and `Value_24h` only,
> enforced by the feature allowlist in `src/burnin/schema.py` and tested in
> `tests/test_checkpoint_discipline.py`. Nothing here is typed by hand:
> `scripts/05_build_report.py` reads `artifacts/results/`.

**Run.** seed `26170` · 100,000 components · 400 lots · realised prevalence 2.46 % · split by `lot_id` (200 train / 80 validation / 120 test lots).

## 9.0 Generator acceptance (Section 3.3)

The generator is only accepted if the best achievable static-limit baseline
catches a clear minority of the defective population. If a tuned static
threshold performed close to the models, the data would be too easy and the
demonstration worthless. This gate runs before any model is fitted and
`scripts/01_generate_data.py` exits non-zero if it fails.

| Acceptance check | Measured | Required | Verdict |
|---|---|---|---|
| Static-limit recall on defects | 32.87 % | < 50 % | PASS |
| Defects finishing under the datasheet maximum at 168 h | 67.13 % | ≥ 60 % | PASS |
| Static-limit yield loss on healthy parts | 0.002 % | small but non-zero | PASS |

Per-parameter share of defects still inside the datasheet maximum at 168 h: `iddq` 68.1 %, `leak` 90.2 %, `tpd` 82.2 %. Physics: AF(healthy, Ea=0.35 eV) = 30.6, AF(defect, Ea=0.70 eV) = 938, a ratio of 31×.

## 9.1 Module A, at operating point

Operating point selected on validation lots as the lowest severity holding
the escape rate at or below the declared target of 10 %; the yield loss this costs is
reported as a consequence, not minimised as a competing objective.
Threshold = `0.980957` (alert ≥ `0.980957`, reject ≥ `0.996094`). Cost ratio C_FN : C_FP = 20 : 1. WATCH counts as flagged.

| Configuration | Escape rate | F₂ | PR-AUC | Yield loss | Cost / 1k parts |
|---|---|---|---|---|---|
| Static limit only | 68.27 % | 0.3672 | 0.3333 | 0.01 % | 368.7 |
| Lot-relative robust z-rule | 9.51 % | 0.8957 | 0.9513 | 0.41 % | 55.3 |
| Isolation Forest | 10.99 % | 0.9092 | 0.9844 | 0.01 % | 59.5 |
| Combined, cost-calibrated | 11.48 % | 0.8918 | 0.9712 | 0.22 % | 64.1 |
| Combined + Module B early rejection | 12.59 % | 0.8493 | n/a (decision rule) | 0.75 % | 75.3 |

Held-out set: 30,000 components, 810 defective. PR-AUC is reported rather than ROC-AUC because prevalence is 2.46 %. Cost is C_FN×FN + C_FP×FP at the stated operating point. The final rung is a decision rule -- a severity threshold OR-ed with a binary early-rejection rule -- so it has no single continuous score and no PR-AUC; inventing one by saturating early-rejected parts to the top of the severity scale would report an artefact of the encoding.

**Reading the ladder honestly.** The first rung is the whole argument: the
static screen escapes 68.27 % of defects, and every dynamic rung takes that below 11.48 %.
Beyond that first step the rungs are close. Each is calibrated separately
against the same escape target, so they land at similar escape rates by
construction and the honest comparison is on what that escape rate costs.
There the Isolation Forest is much the more efficient detector: 0.01 % yield loss against 0.41 % for the z-rule at a comparable escape rate.

The maximum-severity union does **not** beat its own best component here (64.1 versus 55.3 cost per 1k parts). That is a real
negative result and it is reported rather than buried: on this generator the
two detectors largely agree, so the union mostly inherits the z-rule's false
positives without adding recall. It is kept for the reason it was chosen --
under a 20:1 cost a miss is catastrophic, and a union cannot be defeated by a
failure mode only one detector can see -- but that robustness argument is not
something this run demonstrates, and it should not be claimed as if it were.

The gain that *is* demonstrated comes from the decision layer, not the
ensemble: the deployed system in 9.3 reaches 47.7 cost per
1k parts against 64.1 for the combined detector alone, because the
WATCH triggers of Section 6 catch parts the severity threshold alone misses.

### Recall by difficulty population (Section 3.3)

A headline recall that hid a near-zero recall on the hard sub-populations
would be misleading, so the difficulty controls are reported separately.

| Population | n | defective | Recall | Escape rate |
|---|---|---|---|---|
| `healthy` | 26,292 | - | - | flag rate 2.77 % |
| `benign_high_baseline` | 2,898 | - | - | flag rate 13.25 % |
| `defect_standard` | 558 | 558 | 98.39 % | 1.61 % |
| `defect_low_start_fast_drift` | 252 | 252 | 97.22 % | 2.78 % |

## 9.2 Module B, Value_0h + Value_24h → Value_168h

One Random Forest regressor per parameter; units are never pooled. The model
has to beat all three baselines to justify itself.

| Predictor | MAE, iddq (uA) | MAE, leak (uA) | MAE, tpd (ns) | Normalised MAE (mean) |
|---|---|---|---|---|
| Last value | 1.5302 | 0.2589 | 0.2407 | 0.815 |
| Linear extrapolation | 2.6252 | 0.4351 | 0.8335 | 1.667 |
| Lot median trajectory | 1.5621 | 0.2695 | 0.2544 | 0.845 |
| Random Forest | 0.6306 | 0.1099 | 0.1377 | 0.366 |

Normalised MAE is MAE divided by the lot interquartile range of the target,
which is what makes currents in µA and delays in ns comparable. A single
pooled MAE across the three parameters is meaningless and is not reported.

Improvement over the best baseline per parameter: `iddq` 2.43×, `leak` 2.35×, `tpd` 1.75×.

### Interval coverage and early rejection

The interval is locally adaptive: conformity scores are normalised by the
spread of the forest's own trees, so width grows where the ensemble
disagrees with itself. A constant-width band would be calibrated on average
and wrong where it matters, handing an extreme part the same interval as a
typical one. Coverage below is the empirical fraction of held-out parts
inside the nominal band.

| Parameter | Coverage at nominal 90 % | Mean interval width | Reliability margin M | L_max − M |
|---|---|---|---|---|
| `iddq` | 89.79 % | 3.1446 uA | 5.000 uA | 45.000 uA |
| `leak` | 89.74 % | 0.5963 uA | 2.000 uA | 18.000 uA |
| `tpd` | 89.50 % | 0.6700 ns | 1.200 ns | 10.800 ns |

- Early-rejection recall at 24 h: **60.49 %** (escape rate 39.51 %).
- Early-rejection yield loss: **0.61 %** (177 healthy parts).
- Correctly early-rejected: **490** parts.
- **Chamber-hours saved: 70,560 h** (144 h × correctly early-rejected parts), against 25,488 h spent on parts that would have passed.

The early-rejection ceiling is set by the low-start fast-drift population of
Section 3.3, which is unremarkable at 0 h and only mildly elevated at 24 h by
construction. The terminal path is what catches those, which is exactly why
Section 6 keeps the two paths separate.

## 9.3 Final system and decision layer

| Disposition | n | Share | Defective | Healthy |
|---|---|---|---|---|
| PASS | 28,094 | 93.65 % | 16 | 28,078 |
| WATCH | 1,239 | 4.13 % | 304 | 935 |
| REJECT | 667 | 2.22 % | 490 | 177 |

Headline, at the stated operating point: escape rate **1.98 %**, recall 98.02 %, F₂ 0.7715, PR-AUC 0.9712, yield loss 3.81 % (1,112 healthy parts flagged of 29,190).

For comparison the static screen alone, on the same held-out lots, escapes 67.13 % of defects. The dynamic layer takes the escape rate from 67.13 % to 1.98 %.

## 9.4 Lot integrity (Section 4.4)

| Lot-integrity check | Value |
|---|---|
| Contaminated lots present in the held-out set | 4 |
| Lots alarmed | 4 |
| True positives / false positives / false negatives | 4 / 0 / 0 |
| Triggered by MAD inflation | 0 |
| Triggered by tail inflation | 4 |

**Finding.** The MAD-inflation trigger named in the approach does not fire at
realistic contamination levels, and the run above shows it firing on
0 lot(s). MAD has a 50 % breakdown point, so a lot that is 15–20 %
defective inflates its MAD by only 1.1–1.5× while clean lots already reach
1.18× at their 99th percentile; there is no separating factor. The very
robustness that makes MAD the right denominator for a lot-relative z-score
makes it blind as a contamination detector. A tail-inflation index,
`(Q90 − median) / (1.4826 × MAD)` on relative total drift, separates cleanly:
clean lots sit near the normal-theory value of 1.4, contaminated lots reach
8–60. Both triggers are computed and both are reported; the alarm is their
union. See `src/burnin/module_a.py` and `tests/test_modules.py`.

## 9.5 Explainability audit (Section 8.3)

Explainability is a scored criterion, not a courtesy. Coverage is binary and
must be total: a flag without a complete explanation record is a defect in
the system.

| Explainability check | Value |
|---|---|
| Flags audited | 400 |
| Complete records | 400 |
| Coverage | 100.00 % |
| Quoted lot arithmetic reproduces by hand | 100.00 % |
| Responsible parameter named | 100.00 % |
| SHAP ranking consistent with the visible trajectory | 98.25 % |
| Audit verdict | PASS |

### Worked form of the output

> Component **C00000005**, lot `LOT-0000`, wafer `LOT-0000-W1` slot `S06`. Responsible parameter **tpd**. tpd at 168 h: 10.54 ns against lot median 9.104 ns, lot MAD 0.3145; robust z = (10.54 - 9.104) / (1.4826 x 0.3145) = 3.07. Datasheet limit 12 ns. Predicted Value_168h = 10.4 ns, 90 % interval [9.855, 10.95], reliability margin M = 1.2 ns, margin to safety slope +0.0024 ns/h. Dominant contributors: tpd_v24h (+1.095), tpd_headroom (+1.036), tpd_v0h (+0.4557). Disposition: **REJECT** (module_b_early_rejection_at_24h).

The margin obeys `margin = (L_max − M − predicted_168h) / 168`; V_0
cancels, which makes it an identity an inspector can check by hand.

## 9.6 Claims and non-claims

**Kept.** The detector flags trajectory and lot-relative outliers that pass
static limits. The predictor uses `Value_0h` and `Value_24h` only to forecast
`Value_168h`. The evaluation weights false negatives explicitly and reports
the yield loss that buys. Every flag carries a named parameter, checkable lot
arithmetic and a numeric margin. Results are produced on a physics-based
synthetic generator whose construction is documented in Section 3 and gated
by the acceptance criterion in 9.0.

**Avoided.** That SMAP, MSL or CWRU data predicts 168 h burn-in failure. That
CWRU data gives remaining useful life. That these synthetic results establish
ISRO field escape rates. That the system is qualified or validated on flight
hardware. **Deployment requires recalibration on real rig data with real
terminal labels.**

---

Generated by `scripts/05_build_report.py` from `artifacts/results/`. Total pipeline time 541 s on CPU only.
