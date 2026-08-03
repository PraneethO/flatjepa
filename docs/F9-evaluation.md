# F9 — Evaluation and Figures

**Purpose:** turn F7 measurements into the figures and tables that constitute the result.

**Serves:** reporting for E1–E5.

---

## 1. Principle

Figures are generated from stored measurement outputs by a script, never assembled by hand. Every
figure regenerates from a checkpoint plus a config. A figure that cannot be regenerated cannot be
trusted after the third revision.

## 2. Planned figures

| ID | Figure | Shows |
|----|--------|-------|
| **Fig 1** | Recovery R² per flat coordinate, trained vs. random-init vs. raw-input | E1 headline. The controls are *in the figure*, not in an appendix |
| **Fig 2** | Allocated latent width vs. effective dimensionality, with the theoretical value marked | E2 |
| **Fig 3** | λ_sig sweep: prediction loss and recovery R² on shared x-axis | E3 headline; makes any dissociation immediately visible |
| **Fig 4** | Latent PCA projection colored by tension margin | E4, qualitative |
| **Fig 5** | ROC for the taut/slack probe vs. raw-acceleration baseline | E4, quantitative |
| **Fig 6** | Residual magnitude vs. tension margin / jerk magnitude | E5-b |
| **Fig 7** | Per-step prediction error across the rollout horizon | Error compounding — the property latent prediction is claimed to improve |

## 3. Tables

- Main results table: every configuration × metric, mean ± spread across seeds
- Zero-residual calibration (E5-a) as an explicit pass/fail row, since it gates everything else
- Dataset statistics: trajectory count, window count, tension-margin distribution, near-slack base
  rate

## 4. Presentation requirements

- Error bars or CI bands on every quantitative figure; a point estimate from one seed is not a result
- Controls plotted alongside results, never in a separate figure
- Consistent color mapping across figures
- Readable when printed greyscale (do not encode meaning in color alone)
- Axis units labeled; physical quantities with SI units

## 5. Negative-result handling

F3 §5 and F6 §4 both identify plausible ways this project yields negative results (no discrete mode
structure; no violation events to correlate against). The evaluation layer must present those as
first-class outcomes with the same rigor as positive ones.

Practically: a figure showing "latent probe ≈ raw-acceleration baseline" is a real finding about
what latent-predictive models do *not* discover, and should be produced and included rather than
dropped.

## 6. Acceptance criteria

- [ ] Every figure regenerable by one command from checkpoint + config
- [ ] Controls appear in the same figure as the result they contextualize
- [ ] Error bars on all quantitative figures
- [ ] Figures write to a versioned output directory tagged with commit
- [ ] Negative results rendered, not silently dropped
