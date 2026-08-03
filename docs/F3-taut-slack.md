# F3 — Cable Tension and Taut/Slack Labeling

**Purpose:** compute cable tension along every trajectory and label the regime where the
differential-flatness assumption is valid.

**Serves:** E4 (hybrid mode discovery), E5 (residual as assumption monitor).

---

## 1. Why this feature exists

The flat map in F2 assumes a **rigid, always-taut cable**. That assumption is not decoration — it is
what makes the map exist. Where it fails, the "ground truth" of F2 is not ground truth at all.

This makes F3 do double duty:

1. It provides the **label** for E4 (can the latent discover the regime boundary unsupervised?).
2. It provides the **validity mask** that keeps E1/E2 honest, by identifying timesteps where the
   analytic target is not trustworthy.

## 2. Tension in the flat model

For the payload of mass `mL` with cable unit vector `q̂` pointing payload → quadrotor, Newton gives

```
mL·a = T·q̂ − mL·g·e₃        ⟹        T·q̂ = mL·(a + g·e₃)
```

Since the flatness construction *defines* `q̂` to be parallel to `(a + g·e₃)`, the tension magnitude
reduces to a remarkably simple expression:

```
T = mL · ‖a + g·e₃‖
```

**This is computable directly from the payload acceleration columns of the CSV.** No integration, no
extra simulation. F3 is cheap.

### The important consequence

`T → 0` exactly when `a → −g·e₃`, i.e. the payload is in free fall. In the idealized model tension
can never go negative — it bottoms out at zero. So the meaningful quantity is not a sign but a
**margin**: how close the trajectory comes to the slack boundary.

### Is the boundary actually reachable?

Yes — and this is worth stating precisely, because the whole of E4 depends on it. The planner's own
bounds (`data/params/experiments/base.yaml`) are:

```
state_max: [5, 3, 0.75,  5, 5, 5,  10, 10, 10]
state_min: [-1, -3, 0,  -5,-5,-5, -10,-10,-10]
```

Acceleration is bounded to ±10 m/s² while `g = 9.81 m/s²`. **Free fall `a = (0, 0, −9.81)` lies
strictly inside the planner's feasible set.** The taut-cable assumption can therefore be violated by
trajectories the planner is free to produce — this is a real regime, not a contrived one.

> **Open empirical question, to be answered before relying on E4:** how often does the *generated*
> data actually approach the boundary? Minimum-time trajectories through cluttered environments may
> or may not push acceleration to that corner. If near-slack events turn out to be vanishingly rare,
> E4 has no signal and the honest response is to say so — or to deliberately generate aggressive
> descent scenarios to induce them. **F3's first deliverable is this histogram, not the labeler.**

## 3. Outputs

Per timestep:

| Field | Type | Meaning |
|-------|------|---------|
| `tension` | float | `mL·‖a + g·e₃‖`, in newtons |
| `tension_margin` | float | normalized: `tension / (mL·g)`; 1.0 = hover, 0.0 = free fall |
| `near_slack` | bool | `tension_margin < τ` |
| `regime` | int | 0 = taut, 1 = near-slack, for the E4 probe |

The threshold `τ` is a **reported hyperparameter**, not a hidden constant. E4 results must include a
sweep over `τ`, because a probe accuracy that swings wildly with an arbitrary threshold is not a
finding.

## 4. Class imbalance

Near-slack timesteps are expected to be rare. Accuracy is therefore a misleading metric — a probe
predicting "always taut" could score 99%.

**Requirement:** report balanced accuracy, AUROC, and average precision. Report the base rate
alongside every number. Use class-balanced sampling when fitting the probe.

## 5. Honest limitation

Within this dataset, tension is a deterministic function of payload acceleration, which is part of
the flat coordinate vector F2 already emits. So a latent that perfectly encodes acceleration
*trivially* contains the tension label.

This means **E4 as stated is a weaker result than it first appears** and must be framed carefully.
The non-trivial version of the question is whether the latent represents the regime in a way that is
*linearly separable and discretely organized* — a threshold structure — rather than merely
containing the continuous quantity it derives from.

Concretely, E4 must report:

- probe performance on the latent, versus
- probe performance on the raw analytic acceleration (the trivial upper baseline), and
- whether latent geometry shows any *clustering* at the boundary (not just linear readout)

If the latent gives no advantage in organization over the raw quantity, the honest conclusion is
that no discrete mode structure was discovered. That is a publishable negative result and should be
reported as one rather than buried.

## 6. Acceptance criteria

- [ ] Tension computed vectorized from CSV acceleration columns
- [ ] Histogram of `tension_margin` over the full generated corpus produced **first**, to establish
      whether E4 is viable at all
- [ ] Base rate of `near_slack` reported for every threshold in the sweep
- [ ] Labels aligned index-for-index with F2 outputs and F4 windows
- [ ] Unit test: hover trajectory → `tension_margin ≈ 1.0`; synthetic free fall → `≈ 0.0`
