# AI-Driven Anomaly Detection in Component Burn-In & Screening

**SIH26170** · Indian Space Research Organisation / Department of Space · Software · Smart Automation

High-reliability electronic components are burned in at 125 °C for an extended
soak. Screening today applies static parametric pass/fail limits at each
checkpoint. Latent defects survive that screen: parts that stay inside the
absolute limit for the whole soak while drifting in a way a healthy part does
not. Those parts reach the payload and fail in orbit.

This repository implements two modules against one checkpoint grid
(0 h, 24 h, 96 h, 168 h), plus the decision layer, explainability record and
evaluation protocol that go with them.

| | |
|---|---|
| **Module A** | Dynamic lot-relative outlier detection over the completed soak. Interval-rate and lot-relative features → a deterministic robust z-rule **and** an Isolation Forest → a severity score. |
| **Module B** | `Value_0h` + `Value_24h` → Random Forest → predicted `Value_168h` + interval → a safety-slope **early-rejection decision at 24 h**, freeing 144 h of chamber time per rejected part. |

The design is *additive*. Existing engineering limits stay in the loop and
continue to reject hard failures. The machine-learning layer operates on the
population that **survives** the static checks and asks the two questions the
static checks cannot: is this part behaving unlike its own lot, and will this
part still be safe at 168 h?

---

## Results

See **[docs/RESULTS.md](docs/RESULTS.md)** for the complete held-out numbers,
generated directly from `artifacts/results/` — no figure in that document is
typed by hand.

A presentation deck built from the same run is at
[docs/deck.html](docs/deck.html) — open it in a browser; arrow keys move between
slides.

Headline, on held-out lots:

<!-- RESULTS_SUMMARY_START -->
|  | Held-out result | Context |
|---|---|---|
| Escape rate (headline) | **1.98 %** | static screen alone: 67.13 % |
| Recall / F₂ / PR-AUC | 98.02 % / 0.771 / 0.971 | PR-AUC, not ROC-AUC — prevalence is 2.46 % |
| Yield loss at that operating point | 3.81 % | 1,112 healthy parts flagged; WATCH counts as flagged |
| Module B accuracy | 1.7–2.4× better than the best baseline | on all three parameters, from 0 h + 24 h only |
| Early rejection at 24 h | **70,560 chamber-hours freed** | 490 parts × 144 h, at 0.61 % yield loss |
| Lot-integrity alarms | 4/4 caught, 0 false | all via tail inflation; the specified MAD trigger fired 0 times |
| Explanation coverage | 100.00 % | quoted lot arithmetic reproduces by hand: 100.00 % |
| Generator acceptance gate | PASS | static limits catch only 32.87 % of defects |

100,000 components, 400 lots, seed `26170`; 120 test lots held out at lot level. Full detail in [docs/RESULTS.md](docs/RESULTS.md).
<!-- RESULTS_SUMMARY_END -->

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
python scripts/run_all.py                           # ~8 min, CPU only
python scripts/05_build_report.py                   # fills docs/RESULTS.md
streamlit run app/inspector.py                      # inspector dashboard
```

Everything is CPU-only — tree-based models throughout, no accelerator required
on the rig. One command reproduces every reported number from the seed in
`config/default.yaml`.

```bash
pytest
```

---

## Why the data is synthetic, and why that is stated first

No public dataset reproduces parametric burn-in telemetry with hidden 168 h
terminal values and defect labels. NASA SMAP/MSL, NASA IMS and CWRU were
reviewed and set aside — see [docs/EXTERNAL_DATASETS.md](docs/EXTERNAL_DATASETS.md),
which checks the SMAP/MSL label index against its own schema rather than
describing it from memory.

So the data is generated, and the realism of the generator is the load-bearing
element of the whole submission. **A generator that makes defects obvious proves
nothing, because the models can only find what was deliberately embedded.** The
generator therefore carries an acceptance criterion that runs *before* any model
is fitted:

> The generator is only accepted if the best achievable static-limit baseline
> catches a clear minority of the defective population on a held-out run.

`scripts/01_generate_data.py` exits non-zero if that gate fails.
`tests/test_generator.py` asserts it. The measured value is reported in
RESULTS.md §9.0 next to the model results, so a reader can see what the models
were up against.

### The physics

Parameter drift under thermal stress uses an Arrhenius acceleration factor:

```
AF = exp[ (Ea / k) x (1/T_use - 1/T_stress) ],   k = 8.617e-5 eV/K
Ea = 0.35 eV (healthy baseline mechanism)  -> AF ~=  31
Ea = 0.70 eV (latent defect mechanism)     -> AF ~= 9.4e2

V(t) = V_0 x [ 1 + beta_p * AF * (t / 168)^n ] + eps(t)
n = 0.5  healthy    (sub-linear, saturating wear-in)
n = 1.2  defective  (super-linear, accelerating runaway)
```

A latent-defective part drifts roughly **thirty times faster** under stress while
remaining unremarkable at ambient. That is the physical reason burn-in works, and
it is what the detector must exploit.

### The traps

Four populations are injected specifically to defeat trivial solutions:

| Population | Purpose |
|---|---|
| Latent defects finishing under the limit | The escapes the problem statement is about. A static limit cannot catch any of them. |
| Benign high-baseline healthy parts | High `V_0`, flat profile. Punishes any rule that flags on absolute level rather than on drift. |
| Low-start fast-drift defects | Unremarkable at 0 h, only mildly elevated at 24 h. The hard cases for Module B; they set the realistic ceiling on early rejection. |
| Contaminated lots | 15–20 % defective inside a single lot, which corrupts that lot's own median and MAD. Exercises the robustness fallback. |

Plus a chamber common-mode offset shared across each chamber load, which cancels
under lot-relative statistics and survives naive absolute-drift rules.

---

## Architecture

```
config/default.yaml          one file: prevalence, activation energies, noise,
                             cost ratio, operating-point targets

src/burnin/
  config.py                  typed config + derived Arrhenius factors
  schema.py                  data model, and the Module B checkpoint ALLOWLIST
  generator.py               Section 3 physics generator + acceptance gate
  splits.py                  lot-level split; GroupKFold over lot_id
  features.py                Section 4.1 / 5.1 feature construction
  module_a.py                robust z-rule + Isolation Forest + lot integrity
  module_b.py                RF regressor, baselines, conformal interval,
                             safety slope, asymmetric rejection
  decision.py                Section 6 PASS / WATCH / REJECT + early path
  explain.py                 Section 7 explanation records + coverage audit
  metrics.py                 Section 8 metrics + operating-point selection
  evaluate.py                Section 8.4 ablation ladder
  pipeline.py                the four stages, each writing to artifacts/

scripts/
  01_generate_data.py        generate + run the Section 3.3 acceptance gate
  02_train_module_a.py       fit detectors, calibrate the R1 critical value
  03_train_module_b.py       fit one RF per parameter, grouped CV by lot
  04_evaluate.py             ablation ladder, decisions, explanation audit
  05_build_report.py         writes docs/RESULTS.md and this README's table
  06_make_figures.py         the seven presentation figures
  07_build_presentation.py   writes docs/PRESENTATION.md
  run_all.py                 stages 1-4 end to end
  download_reference_datasets.py

app/inspector.py             Streamlit inspector dashboard
tests/                       the specification, expressed as tests
artifacts/figures/           seven figures, regenerated from the run
```

### Module A — dynamic lot-relative outlier detection

Four checkpoints per part per parameter is a short, unevenly spaced series, and
the features are built for that reality: level, interval delta, interval rate,
rate acceleration, total drift, lot-relative level and drift, and
cross-parameter agreement. The lot-relative statistic is

```
robust_z(V_t) = ( V_t - median_lot(V_t) ) / ( 1.4826 x MAD_lot(V_t) )
```

which reproduces the canonical example exactly: in a lot averaging 10 µA, a part
reading 45 µA is a severe anomaly even though the datasheet maximum is 50 µA.
That arithmetic is what a QA inspector can check by hand.

**Features deliberately excluded**, and why: RMS, kurtosis, crest factor,
rolling-window statistics, polynomial curvature fits, and persistence rules
requiring consecutive windows. All of these belong to dense vibration or
telemetry streams. With four unevenly spaced samples per part they are either
undefined or noise-dominated. Carrying them into a burn-in design is a sign the
feature set was copied from a different sensing problem.

Two detectors run in parallel and are combined under a **maximum-severity
union, not an intersection** — under an asymmetric cost where a miss is
catastrophic, requiring both detectors to agree is the wrong operator. Each of
the three parameters produces its own severity; the component takes the worst,
and the flag records which parameter drove it. A component is never averaged
into a pass.

### Module B — the 0 h + 24 h drift predictor

Inputs are restricted to the 0 h and 24 h checkpoints **by an explicit allowlist
enforced in code**, not by convention (`src/burnin/schema.py`). The strongest
form of the check is in `tests/test_checkpoint_discipline.py`: the 96 h and
168 h columns are overwritten with nonsense and every Module B feature value and
every model prediction must be bit-identical.

Splitting is by `lot_id` with entire lots held out. Splitting *within* a lot
leaks, because lot median and lot MAD are computed from the lot's own members and
would carry test information into training.

A point prediction ignores its own uncertainty, which is exactly the wrong
behaviour when a miss is catastrophic. So the rejection rule acts on the upper
bound of a split-conformal interval calibrated on held-out lots:

```
S_safe = ( L_max - V_0 - M ) / 168        M = max( 0.10 x L_max, 3 x sigma_meas )
S_pred = ( V_168_hat - V_0 ) / 168
margin = S_safe - S_pred

EARLY-REJECT if  V_168_hat(upper, 90%) > L_max - M   or   S_pred > S_safe
```

This makes the uncertainty itself a reason to reject: a part the model cannot
confidently clear is not cleared. Note that `V_0` cancels in the margin, giving
the identity `margin = (L_max − M − V_168_hat) / 168` — an inspector can verify
any quoted margin with a calculator.

The interval is **locally adaptive**, not a constant-width band. A fixed band is
calibrated on average and wrong exactly where it matters: it hands an extreme
part the same interval as a typical one, so a rule meant to refuse to clear
what the model does not understand never widens for the parts it understands
least. Conformity scores are therefore normalised by the spread of the forest's
own trees, so the interval grows where the ensemble disagrees with itself.
Measured on held-out lots: coverage holds at nominal, while interval width
spans a ~450× range and tracks the actual error — narrow for easy parts, wide
for the ones the training lots did not cover.

### Decision layer

| Path | Runs at | Evidence | Output |
|---|---|---|---|
| Early | 24 h | Module B only | CONTINUE or EARLY-REJECT |
| Terminal | 168 h | Static limits + Module A over the full grid | PASS, WATCH or REJECT |

The two paths are kept separate, because collapsing them destroys the value of
early rejection. **WATCH maps to flagged (positive) for scoring, never to pass.**
An intermediate state that quietly absorbed defective parts would manufacture
false negatives, which is precisely the error the evaluation punishes hardest.
The cost of that choice is yield loss, and it is reported honestly alongside the
escape rate it buys.

### Explainability

Explainability is a scored criterion, not a courtesy. Every flag carries a
complete record, generated automatically, with no free-text narration: the
responsible parameter, the lot-relative arithmetic, TreeSHAP contributions for
the Module B prediction, ranked robust deviations for Module A, the raw
four-point trajectory against the lot band, and the numeric margin.
**Coverage is binary and must be total** — a flag without a complete record is a
defect in the system, and `coverage_audit()` fails the run if so.

### Evaluation

Built on the three metrics named in the problem statement, in the order they are
named. A missed defective part is catastrophic; a wrongly rejected good part is
merely expensive. That asymmetry is encoded explicitly rather than optimised away
into a balanced score:

```
Cost = C_FN x FN + C_FP x FP,    C_FN : C_FP = 20 : 1
Escape rate = FN / (TP + FN)          <- headline figure
F_beta, beta = 2                      <- recall weighted over precision
PR-AUC, not ROC-AUC                   <- prevalence is 1-3 %
False-reject rate (yield loss)        <- reported at the same operating point
```

The operating point is the lowest severity holding the escape rate at or below a
declared target on validation lots. The yield loss that costs is reported as a
consequence, not minimised as a competing objective. **Every reported figure
names its operating point; a metric quoted without one is not reported.**

The **ablation ladder** (§8.4) reports each rung at the same operating point, so
the contribution of each component is visible rather than asserted:

```
static limit -> lot-relative z-rule alone -> Isolation Forest alone
             -> combined detector -> combined + Module B early rejection
```

The ladder is there to be believed when it is unflattering, and on this run it
is. The first rung carries the argument — the static screen escapes most
defects and every dynamic rung takes that to single digits. Beyond it, **the
maximum-severity union does not beat its own best component**: the two detectors
largely agree here, so the union inherits the z-rule's false positives without
adding recall. It is kept for the reason it was chosen (under a 20:1 cost a
union cannot be defeated by a failure mode only one detector sees), but that
robustness is an argument, not something this run demonstrates, and RESULTS.md
says so. The gain that *is* demonstrated comes from the decision layer.

---

## An engineering finding worth stating

The approach document specifies the lot-contamination fallback as: *if a lot's
MAD exceeds a stated factor of a historical band, raise a lot integrity alarm.*

**That trigger does not fire.** MAD has a 50 % breakdown point, so a lot that is
15–20 % defective inflates its MAD by only 1.1–1.5×, while clean lots already
reach 1.18× at their 99th percentile. There is no separating factor. The very
robustness that makes MAD the right denominator for a lot-relative z-score makes
it blind as a contamination detector.

A tail-inflation index separates them cleanly:

```
TI = ( Q90(rel drift) - median(rel drift) ) / ( 1.4826 x MAD )
```

Clean lots sit near the normal-theory value of ~1.4; contaminated lots reach
8–60. Both triggers are implemented, both are reported, and the alarm is their
union. `tests/test_modules.py::test_mad_inflation_alone_is_insufficient` pins
the finding so it cannot silently regress.

---

## Reproducibility

Seeded generator, pinned dependency versions, and configuration held in a single
file covering prevalence, activation energies, noise levels, cost ratio and
operating-point targets. The train/test lot split is written to disk
(`data/split.csv`) so results can be re-derived exactly.

`data/` and `artifacts/models/` are gitignored: both are build outputs, the
fitted forests are ~300 MB each, and `python scripts/run_all.py` regenerates
every one of them from the seed.

---

## Claims to keep, and claims to avoid

**Keep.** The detector flags trajectory and lot-relative outliers that pass
static limits. The predictor uses `Value_0h` and `Value_24h` only to forecast
`Value_168h`. The evaluation weights false negatives explicitly and reports the
yield loss that buys. Every flag carries a named parameter, checkable lot
arithmetic and a numeric margin. Results are produced on a physics-based
synthetic generator whose construction is documented and gated.

**Avoid.** That SMAP, MSL or CWRU data predicts 168 h burn-in failure. That CWRU
data gives remaining useful life. That these synthetic results establish ISRO
field escape rates. That the system is qualified or validated on flight
hardware. **Deployment requires recalibration on real rig data with real terminal
labels, and this submission says so.**

---

## Source

SIH 2026 problem statement SIH26170, Indian Space Research Organisation,
Department of Space, Software, Smart Automation. Official listing:
`sih.gov.in/sih2026PS`.
