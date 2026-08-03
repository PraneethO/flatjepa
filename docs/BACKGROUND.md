# Background and Reasoning

Why this project is shaped the way it is. `00-overview.md` states *what* is being done; this
document records *why*, including the alternatives that were considered and rejected.

Written because the reasoning lived in a conversation rather than in the repo, and a design without
its rationale gets "simplified" back into the thing it was avoiding.

---

## 1. How the question arrived

The starting idea was ordinary: take PolyFly's trajectory data and fit a **transformer** instead of
an RNN, to see whether it predicts trajectories better.

That is a reasonable engineering question and a weak research one. Two problems:

- Architecture comparisons on a fixed dataset ("transformer vs. RNN") are well-trodden and the
  answer is usually "depends on data scale."
- More importantly, PolyFly's trajectories are **noise-free optimizer output** governed by a triple
  integrator. There is almost nothing to predict: `ṗ = v`, `v̇ = a`, `ȧ = u` is exactly integrable.
  Any sequence model fits it. A benchmark where every method scores ~100% measures nothing.

The interesting question had to be about **representation**, not architecture.

## 2. Why JEPA rather than a predictive model

A Joint Embedding Predictive Architecture predicts in **latent space** rather than observation
space. The encoder maps observations to a latent; a predictor rolls the latent forward; the loss
compares predicted latent to encoded future latent.

The advantage: the model is free to **discard what it cannot predict**. A pixel-space or
state-space predictor must model every detail including irreducible noise, and autoregressive
rollout compounds those errors. A latent predictor can represent only the predictable structure.

The cost, and the reason this project exists: **you cannot see what it kept.** There is no ground
truth for "the right latent," so the field evaluates representations *indirectly*, through
downstream task performance. That conflates three separable things — did the representation capture
the state, did the predictor learn the dynamics, is the controller any good. When the downstream
number is bad, you cannot tell which failed.

## 3. Representation collapse, and why SIGReg is load-bearing

JEPA's prediction loss has a degenerate solution that is easy to miss.

The target is produced by **the encoder being trained**, not by fixed ground truth. If the encoder
maps every input to the same constant `c`, then the target is `c`, the predictor outputs `c`, and
the loss is *exactly zero*. Not approximately — exactly. Nothing in the prediction objective alone
requires the latent to retain any information about the system.

This is why every self-predictive (non-contrastive) architecture needs an anti-collapse mechanism.
BYOL and I-JEPA use an EMA target encoder plus stop-gradient. SkyJEPA — and therefore this project —
instead uses **SIGReg** (Sketched Isotropic Gaussian Regularization, from LeJEPA by Balestriero &
LeCun): project the batch of latents onto random 1-D directions, score each projection against a
standard Gaussian with the Epps–Pulley statistic, penalize the deviation. Isotropy in expectation,
without ever forming a covariance matrix.

**This is where the project's headline experiment comes from.** SIGReg pushes the latent toward an
isotropic Gaussian. But the quantities it should encode — position, velocity, acceleration, jerk —
have different physical units, different natural scales, and a joint distribution strongly shaped by
minimum-time optimization. There is no a priori reason their faithful embedding should be isotropic.

So: *does enforcing isotropy improve recovery of the true state, or distort a latent geometry that
is intrinsically non-isotropic?* Normally unanswerable, because "the true state" is unknown. Here it
is not.

## 4. What SkyJEPA does, and what was borrowed

[arXiv:2606.23444](https://arxiv.org/abs/2606.23444). Learns a latent dynamics model for quadrotors
and uses it for real-time control.

| Component | SkyJEPA | This project |
|---|---|---|
| State encoder | TCN, channels [8,8,16] | same |
| Action encoder | TCN [4,4,8], actions = 4 motor forces | same shape, **actions = payload jerk (ℝ³)** |
| Predictor | single-layer GRU, hidden 24 | same, but **width is swept, not fixed** |
| Window | H=10 (0.5 s @ 20 Hz), T=20 | H=10, T=20 **@ 10 Hz** (see F4) |
| Anti-collapse | SIGReg, no EMA target encoder | same |
| Prober | residual on rigid-body dynamics | residual on the **triple integrator** |
| Control | MPPI, 512 samples, Orin NX @100 Hz | **not implemented** — see §6 |
| Data | 500 randomized domains × 20k trajectories | PolyFly optimal trajectories |

Two deviations matter:

- **Actions are jerk, not motor forces.** Jerk is PolyFly's actual control input. Using motor forces
  would require a dynamics layer that does not exist in this data.
- **Latent width is swept.** SkyJEPA fixed it at 24 because the model had to run at 100 Hz on
  embedded hardware. This project has **no real-time constraint**, and E2 asks whether effective
  dimensionality matches the theoretical minimum — hard-coding the width would prejudge the answer.
  Inheriting a number from a deployment constraint that does not apply here would be cargo cult.

SkyJEPA's prober absorbs the **sim-to-real gap**: unmodeled aerodynamics, motor delay, manufacturing
variation. That interpretation is unavailable here — there are no real flight logs on this machine,
and the data is generated by the very model the prober would correct. Hence the reframing in F6: the
residual becomes an **instrument** for detecting where the analytic model's assumptions break, not
an accuracy improvement.

## 5. Why differential flatness is the whole point

A differentially flat system has a closed-form map from a small set of **flat outputs** to the
entire state. For the cable-suspended payload, payload position and its derivatives determine
quadrotor position, velocity, attitude, and cable direction analytically.

That means **the minimal sufficient latent is known** — in dimension and in content. Concretely the
sufficient state for prediction-given-actions is `(p, v, a) ∈ ℝ⁹`, which is exactly the planner's own
state vector. E2 therefore has a numeric prediction (~9) rather than a vague expectation.

This converts representation quality from an inference into a **measurement**. That is the
contribution. Not "we trained a JEPA on drones."

Three properties made this system the right choice:

1. the flat map is exact and available in closed form,
2. the dynamics are rich enough that learning is non-trivial,
3. there is a regime — slack cable — where the analytic model *provably* stops being valid, giving a
   built-in control condition.

## 6. Alternatives considered and rejected

### MPPI vs. a distilled policy
Considered as the headline comparison, then rejected.

The reasoning is worth keeping because it is a natural question. A learned world model can be used
for control two ways: sample action sequences and score them against the model (MPPI), or distill a
policy that outputs actions directly (Dreamer-style).

The trade is real. A world model plus MPPI is reusable across objectives — change the cost at
runtime, add constraints, no retraining — and never averages across distinct modes, because it
samples and scores rather than regressing. A distilled policy is dramatically faster (one forward
pass vs. 512 × 15) but bakes in one cost function and can average over multimodal decisions,
producing the mean of "left of the obstacle" and "right of the obstacle."

**Rejected because it is already answered.** TD-MPC2 ships both a policy prior and sampling-based
planning; the planning-vs-policy trade-off is well characterized. A head-to-head here would
reproduce known results. It may return later as a *demonstration* that the learned model is useful —
not as a research question. (`F10-baselines.md` §4.)

### Isaac Gym rollouts
Would have provided a genuine sim-to-sim residual — execute an optimal trajectory in a
higher-fidelity simulator and learn the gap. **Ruled out on this machine:** Isaac Gym requires
Python 3.8 and is deprecated upstream; the available environments are 3.10 and 3.12.

### Real flight logs
Would provide the true sim-to-real residual. `scripts/plot_rosbag.py` shows the upstream project was
built around ROS 2 bags with odometry and motor speeds. **No bags exist on this machine**, and the
script is hard-coded to the original author's paths.

### Perception / depth
Deferred, not rejected — see `F11-deferred-perception.md`. It unlocks the multimodality questions,
but adds a whole subsystem before any result exists. Notably it would **not** require Isaac Gym:
obstacles are axis-aligned boxes, so depth can be raycast analytically in NumPy.

## 7. The failure mode this project is most exposed to

Everything above is upside. The corresponding risk is that the headline result is **vacuous**.

Because ground truth is analytically available, it is tempting to probe the latent for the state —
and if the model was *fed* that state, recovering it proves nothing.

The subtler version, which the original design walked into: with a **history window** as input,
every time-derivative is a *linear functional of that window*. Velocity is a finite difference,
acceleration a second difference. So restricting the input to position does **not** help — the
derivatives remain linearly recoverable from the raw window, and the "control" would score ~1.0 too,
making the result look controlled-for while proving nothing.

The resolution: probe for the **nonlinear** consequences of the flat map — attitude, cable
direction, tension magnitude — enforced mechanically by the linear-decodability audit in
`F7-measurement-suite.md` §1b, which disqualifies any target a linear probe on the raw window
already solves.

This is the single most important design decision in the project, and it is a *measurement*
decision, not a modeling one.

## 8. References

Full annotated list in the README. The two the design depends on directly:

- **PolyFly** — [arXiv:2510.15226](https://arxiv.org/abs/2510.15226) — trajectories, flatness map,
  optimal-planner oracle
- **SkyJEPA** — [arXiv:2606.23444](https://arxiv.org/abs/2606.23444) — the architecture adapted here

And for the cable-suspended flatness result specifically, including the hybrid taut/slack regime:
Sreenath, Michael & Kumar, *Trajectory Generation and Control of a Quadrotor with a Cable-Suspended
Load*, CDC 2013.
