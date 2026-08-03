# F7 — Measurement Suite

**Purpose:** the scientific core. Everything else exists to make this measurable.

**Serves:** E1–E5.

---

## 1. Principle

Every measurement in this suite must be reported against a **control**. A probe R² with no baseline
is uninterpretable, because some information is trivially present in the input window regardless of
what training accomplished.

Three controls apply throughout, and all three are mandatory:

| Control | Question it answers |
|---------|--------------------|
| **Random-init encoder** | How much is architecture + windowing alone, with no learning? |
| **Raw input window** | How much is linearly present in the input already? |
| **Shuffled labels** | What R² does this probe produce on pure noise? (chance floor) |

If the trained encoder does not beat the random-init encoder, **there is no result**, and the
honest report says so. This is the single most likely way this project produces a spurious positive,
and it is cheap to guard against.

## 1b. The linear-decodability audit — a hard gate on E1 targets

**This is the safeguard against the project's most likely failure mode: a result that looks strong
and means nothing.**

### The trap

With a *history window* as model input, every time-derivative of the state is a **linear functional
of that window**. Velocity is a finite difference of positions; acceleration is a second difference;
jerk a third. So a linear probe fit directly on the raw input window recovers `v`, `a`, and `j` to
near-machine precision, with no model involved at all.

This invalidates the mitigation originally proposed in `F4-dataset.md` §2 ("feed position only, probe
for the derivatives"). Restricting the observation does **not** help: the derivatives remain linear
in the window. A result reported that way would be vacuous while appearing to have been controlled
for.

### The gate

Before any quantity may serve as a **headline** E1 target, it must pass this audit:

1. Fit a ridge probe from the **raw normalized input window** to the candidate target.
2. Record R²_raw on the test split.
3. If R²_raw exceeds a threshold (default 0.9), the target is **disqualified as a headline target**.
   It may still be reported as a sanity check, explicitly labeled as linearly trivial.

The audit runs automatically as part of the suite and its table is published alongside results, so
readers can see which targets were trivial and which were not. No target is promoted by hand.

### Expected outcome

| Target | Linear in window? | Role |
|--------|-------------------|------|
| `p`, `v`, `a`, `j` | **Yes** — finite differences | Sanity check only. Expect R² ≈ 1.0 *and say so* |
| `‖a + g·e₃‖` (tension) | No — norm | Eligible |
| cable direction `p̂` | No — normalization | Eligible |
| quadrotor attitude `R_flat` | No — cross products, SO(3) | **Preferred headline target** |
| `p_quad` | No — nonlinear via `b₃` | Eligible |

This reframes E1 into a question that is actually non-trivial:

> Does the latent linearly expose the **nonlinear consequences** of the flat map — the normalizations
> and rotations that constitute its real content — rather than merely retaining the inputs it was
> given?

Note this makes the random-init control genuinely informative rather than a formality: a random
encoder is close to a random projection, which preserves *linear* information but does not
manufacture nonlinear functionals. If a random-init encoder also recovers attitude, that is a
finding about window geometry and must be reported, not hidden.

## 2. E1 — Latent recovery

Fit a **linear** probe (ridge, α selected on validation) from frozen latent to F2 flat coordinates.

- Report per-coordinate R², not a single aggregate — recovering position but not acceleration is a
  qualitatively different finding from uniform partial recovery.
- Linear is deliberate: it tests linear accessibility, a stronger and more interpretable claim than
  MLP-extractability. Report an MLP probe as a secondary number to quantify the gap.
- Probe is fit on the training split, reported on test, with the environment-level split of F4.

See the design tension in F4 §2: if the model's input already contains the probe target, E1 is
tautological. The headline configuration must use a partial observation, and quadrotor attitude and
cable direction — not directly in the input — are the physically meaningful targets.

## 3. E2 — Intrinsic dimensionality

Estimate the effective dimensionality of the latent:

- PCA spectrum and cumulative explained variance
- participation ratio: `(Σλᵢ)² / Σλᵢ²`
- a nonlinear estimator (e.g. TwoNN) as a cross-check, since PCA-based measures assume linear
  structure

**Prediction to test:** effective dimension ≈ **9**, the minimal sufficient state under
triple-integrator dynamics (F2 §3).

Sweep allocated latent width and plot allocated vs. effective. If effective dimension tracks the
theoretical value across a range of allocated widths, that is a clean, quantitative result.

## 4. E3 — SIGReg ablation *(headline)*

Repeat E1 and E2 across `λ_sig ∈ {0, ...}` spanning several orders of magnitude.

Report jointly:

| Axis | Metric |
|------|--------|
| Prediction quality | latent prediction loss, multi-step |
| Representation quality | E1 recovery R² |
| Geometry | E2 effective dimension |
| Collapse | latent variance, effective rank |

The interesting outcome is a **dissociation** — e.g. λ that minimizes prediction loss differs from λ
that maximizes true-state recovery. That would show the standard JEPA objective and faithful state
representation are not the same goal, which is precisely the kind of claim this instrument exists to
make and which downstream-task evaluation cannot isolate.

Report the λ=0 collapse explicitly: it establishes the regularizer is load-bearing in this setting.

## 5. E4 — Hybrid mode probe

Linear probe for taut vs. near-slack (F3) on frozen latents.

Mandatory reporting, per F3 §4–5:

- balanced accuracy, AUROC, average precision — **not** raw accuracy
- base rate at every threshold in the `τ` sweep
- the trivial baseline: same probe on raw analytic acceleration
- latent geometry at the boundary (clustering), not only linear readout

Per F3 §5, if the latent offers no organizational advantage over the raw quantity it derives from,
the conclusion is that no discrete mode structure was discovered — reported as a negative result,
not omitted.

## 6. E5 — Residual monitor

Per F6: the zero-residual calibration check first, then correlation of residual magnitude against
assumption-violation signals. Report correlation and, more usefully, whether residual magnitude
*ranks* violation severity — a monitor needs ordering, not just correlation.

## 7. Statistical hygiene

Not optional, and cheap to build in from the start:

- **Multiple seeds** (≥3, preferably 5) for every configuration; report mean and spread, never a
  single run
- Confidence intervals on probe metrics
- Fixed test split, touched only for final numbers
- Every reported number traceable to a config hash and commit

A seed-to-seed spread larger than the effect being claimed means there is no effect. Given that
representation metrics are known to be seed-sensitive, this is a live risk rather than boilerplate.

## 8. Acceptance criteria

- [ ] All three controls implemented and reported alongside every probe result
- [ ] Per-coordinate reporting, not aggregate-only
- [ ] Probes fit on train, reported on test, environment-level split respected
- [ ] Multi-seed with spread reported by default
- [ ] Class-imbalance-aware metrics for E4
- [ ] Every result traceable to config hash + commit
- [ ] Suite runnable as a single command against a checkpoint
