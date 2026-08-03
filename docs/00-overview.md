# 00 — Research Overview

Status: **specification**. This document is the spine; every feature document (F1–F10) exists to
serve one or more of the experiments below.

---

## 1. Hypothesis

> A differentially flat system provides analytic ground truth for the minimal sufficient latent of
> its own dynamics. Therefore, on such a system, JEPA representation quality can be measured
> directly rather than inferred from downstream task performance.

The claim being tested is not "JEPA is good" or "JEPA beats an RNN." It is that **flatness gives us
a measuring instrument** that the field currently lacks, and that using it reveals something
non-obvious about what latent-predictive models learn.

## 2. Why this is answerable here and not elsewhere

For a system to serve as this instrument it needs:

1. an exact, closed-form map from a low-dimensional set of flat outputs to the full state,
2. enough dynamical richness that learning the representation is non-trivial,
3. a regime boundary where the analytic model *provably* stops being valid, giving a control
   condition.

The quadrotor with cable-suspended payload satisfies all three. Condition (3) is the taut/slack
cable transition, which is what makes this system more interesting than a bare quadrotor: there is a
known-valid regime and a known-invalid regime, and we can ask what the latent does at the boundary.

## 3. The flat map

For the cable-suspended payload system, the flat outputs are the payload position and its
derivatives. Given payload position, velocity, acceleration, and jerk, the following are determined
in closed form:

- quadrotor position, velocity, acceleration
- quadrotor attitude (rotation matrix / quaternion)
- cable direction
- payload attitude relative to the cable

This map is implemented in the upstream PolyFly planner (`Planner.differential_flatness`) and is
re-implemented independently in this repository (F2) so that the ground truth is not silently
coupled to a specific upstream version.

**Consequence:** the theoretical minimal latent dimension is known. Any latent capacity beyond it is
either redundancy or wasted, and any recovery shortfall below it is measurable information loss.

**Important caveat.** The flat map holds under the assumptions the upstream planner makes: a rigid,
always-taut cable, and yaw pinned to zero. Both assumptions are load-bearing for this project — F3
exists to detect where the first one fails, and E5 exists to test whether the model notices.

## 4. Experiments

### E1 — Latent recovery

Train the JEPA (F5) on trajectory data. Freeze it. Fit a **linear** probe from the frozen latent to
the analytic flat coordinates (F2). Report R² per coordinate.

- A linear probe is deliberate: it tests whether the information is present *and linearly
  accessible*, which is a stronger and more interpretable claim than "an MLP can dig it out."
- Baselines: probe on random-init (untrained) encoder, and probe on raw input windows. The trained
  encoder must beat both, or the result is trivial.

### E2 — Intrinsic dimensionality

Measure the effective dimensionality of the latent (PCA spectrum, participation ratio) and compare
against the known minimal dimension implied by the flat map.

Three qualitative outcomes, all informative:
- matches → the model found the right compression
- much lower → collapse or information loss
- much higher → the latent is carrying nuisance structure it did not need

### E3 — SIGReg ablation *(the distinctive one)*

Repeat E1 and E2 with the anti-collapse regularizer ablated: off, and swept across λ.

Anti-collapse regularizers are normally justified circularly — "without it the representation
collapsed, and with it downstream accuracy improved." Because we have ground truth, we can ask a
sharper question: **does SIGReg improve recovery of the true state, or does enforcing isotropy
actively distort a latent geometry that is intrinsically non-isotropic?** The flat coordinates have
very different natural scales and physical units; there is no a priori reason their faithful
embedding should be an isotropic Gaussian. This is a real tension in the method and we can measure
it.

### E4 — Hybrid mode discovery — **BLOCKED, see F3**

> The tension-margin histogram over the generated forest corpus contains **zero** near-slack
> timesteps at every threshold swept (min margin 0.84; free fall would be 0.0). The cause is a
> generation config — payload z is bounded to a 0.75 m corridor while x/y span ~18 × 15 m, so a
> minimum-time horizontal objective never builds vertical speed. Running E4 on this data would
> report "no modes discovered" from data containing no mode transitions.
>
> Two probe trajectories then established that E4 **is recoverable with a purpose-built corpus**: a
> 20 m descent with ±20 m/s velocity bounds reaches margin 0.15, with 8.4% of timesteps below
> τ=0.2 — a workable base rate. A 6 m descent is not enough (0.60). The binding constraint is the
> jerk bound, not acceleration.
>
> So E4 needs a decision: generate a *separate* descent corpus (not mixed into the forest data,
> which would confound E1–E3), or drop E4. **E1, E2, and E3 are unaffected.** See
> `F3-taut-slack.md` for the evidence table.

Label each timestep with cable tension sign (F3). Fit a linear probe for taut-vs-slack on the frozen
latent, trained with **no mode supervision** during JEPA training.

If the latent linearly separates the regimes, a latent-predictive model discovered a discrete
contact mode from continuous data alone. This is the aerial analog of contact-mode discovery in
manipulation.

### E5 — Residual as assumption monitor

Rather than treating the physics prober's residual (F6) as an accuracy improvement, treat it as an
*instrument*: it should be near zero where the flatness assumptions hold and spike where they do not
(near-slack cable, aggressive jerk, large yaw deviation).

Includes the **zero-residual sanity check**: trained and evaluated on purely flatness-consistent
data, the learned residual must converge to approximately zero. If it does not, the prober or the
nominal physics is wrong. This is a falsifiable correctness test with an unambiguous pass/fail —
and it should be run before any other result is trusted.

## 5. Feature map

| Feature | Serves |
|---------|--------|
| F1 Data generation | all |
| F2 Flat-output extractor | E1, E2, E5 |
| F3 Taut/slack labeler | E4, E5 |
| F4 Dataset builder | all |
| F5 JEPA core | all |
| F6 Physics prober | E5 |
| F7 Measurement suite | E1–E5 |
| F8 Training harness | all |
| F9 Evaluation & figures | all |
| F10 Baselines | E1 (probe controls) |

## 6. Scope boundaries for v1

**In scope:** state-only modeling. The model sees state and action history; no perception.

**Explicitly out of scope for v1**, with reasons:

- *Depth / obstacle perception.* Would enable multimodal routing questions but adds an entire
  subsystem before any result exists. Deferred; see F11 placeholder.
- *Isaac Gym rollouts.* Ruled out on this machine — Isaac Gym requires Python 3.8 and is
  deprecated upstream.
- *Real flight logs.* None available locally. Without them there is no true sim-to-real residual,
  which is why E5 is framed around *assumption violation* rather than sim-to-real gap.
- *MPPI vs. distilled policy comparison.* Well covered in existing literature (TD-MPC2 ships both).
  Not a contribution. May appear later as a downstream demonstration only.

## 7. What would falsify this

Stating this up front so the result stays honest:

- If a probe on a **randomly initialized** encoder recovers the flat outputs about as well as the
  trained one, the recovery result is vacuous — it would mean the windowed input is already close to
  linearly sufficient, and training added nothing.
- If the zero-residual check (E5) fails, nothing downstream is trustworthy.
- If latent recovery is high but the model's multi-step prediction is poor, then "recovers the flat
  outputs" is not the same as "learned the dynamics," and the framing needs revision.

Each of these is cheap to check and each is a stopping condition.
