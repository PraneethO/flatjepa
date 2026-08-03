# F5 — JEPA Core

**Purpose:** the latent world model — encoders, latent predictor, and the anti-collapse regularizer.

**Serves:** all experiments. E3 targets this component directly.

---

## 1. Architecture

Adapted from SkyJEPA ([arXiv:2606.23444](https://arxiv.org/abs/2606.23444)), with deviations noted.

| Component | Design | Output |
|-----------|--------|--------|
| State encoder `Enc_θ` | Temporal Convolutional Network, channels [8, 8, 16] | `s_t` |
| Action encoder `Enc_φ` | TCN, channels [4, 4, 8] | `z_t` (dim 8) |
| Predictor `Pred_ψ` | single-layer GRU, hidden dim 24 | `s̃_{t+1}` |

Both encoders consume the full H-step history window. The predictor is unrolled T steps, consuming
one action embedding per step:

```
s̃_{t+k} = Pred(s̃_{t+k−1}, z_{t+k−1})
```

**Deliberate deviations from SkyJEPA:**

- **Actions are payload jerk (ℝ³)**, not motor forces (ℝ⁴) — see F4 §2.
- **Latent width is a swept parameter, not fixed at 24.** E2 asks whether the effective
  dimensionality matches the theoretical minimum of ~9. Hard-coding the width would prejudge the
  answer; a width sweep is part of the experiment, and the relationship between allocated width and
  *effective* width is itself a result.

**No EMA target encoder.** SkyJEPA uses a single encoder for both the context and the prediction
target — not a BYOL/I-JEPA-style online/target pair. Targets are `Enc_θ` applied to ground-truth
future states, with gradients flowing through both branches, and collapse is prevented by SIGReg
rather than by a stop-gradient. This is a real architectural choice and worth an ablation.

## 2. Training objective

```
ℒ_total = ℒ_pred + λ_sig · ℒ_SIGReg

ℒ_pred  = (1/T) Σ_k ‖ s̃_{t+k} − Enc_θ(x_{t+k}) ‖²
```

Teacher-forced: the action sequence comes from the dataset, so the rollout is not autoregressive in
actions during training.

## 3. SIGReg

**Sketched Isotropic Gaussian Regularization**, from LeJEPA (Balestriero & LeCun). Mechanism:

1. Sample M random unit directions in latent space.
2. Project the batch of latents onto each direction, giving M univariate distributions.
3. Score each against a standard Gaussian with the Epps–Pulley statistic.
4. Penalize the aggregate deviation.

Resampling directions each step enforces isotropy and Gaussianity in expectation without ever
forming a full covariance matrix. Reference implementation:
[rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa).

### Why the collapse it prevents is total, not partial

Worth stating because it motivates E3. In `ℒ_pred`, **the target is produced by the encoder being
trained**, not by fixed ground truth. If `Enc_θ ≡ c` for all inputs, the target is `c`, the
predictor outputs `c`, and the loss is *exactly* zero. Nothing in `ℒ_pred` alone requires the latent
to retain any information about the system.

### The E3 tension

SIGReg pushes the latent toward an isotropic Gaussian. But the flat coordinates it should encode —
position, velocity, acceleration, jerk — have different physical units, different natural scales, and
a strongly non-isotropic joint distribution shaped by the dynamics and by minimum-time optimization.

There is no a priori reason the faithful embedding of that structure should look isotropic. So E3
asks a question that is genuinely open rather than rhetorical:

> Does SIGReg improve recovery of the true state, or does enforcing isotropy distort a latent
> geometry that is intrinsically non-isotropic?

Either answer is interesting. This is the experiment that most justifies the project.

**Requirement:** `λ_sig` is swept, including exactly 0. The λ=0 run is expected to collapse — and
confirming that it does, *measurably*, is itself a result, since it establishes the regularizer is
doing real work in this setting rather than being inherited cargo-cult.

## 4. Collapse diagnostics

Because collapse is the central failure mode, it must be *monitored during training*, not discovered
afterward:

- latent variance per dimension (collapse → → 0)
- rank / PCA spectrum of a batch of latents
- participation ratio
- pairwise latent distance distribution

Requirement: these are logged every epoch and a collapse alarm fires when effective rank falls below
a threshold, so an all-night run does not spend eight hours training a constant function.

## 5. Deliberate scale choice

SkyJEPA's model is ~99K parameters, sized for 100 Hz onboard inference. This project has **no
real-time constraint** — nothing is deployed to hardware in v1.

Model size should therefore be chosen for the *scientific* question (how does capacity interact with
effective dimensionality?) rather than inherited from a deployment constraint that does not apply
here. Starting at SkyJEPA's scale is reasonable for comparability; staying there for that reason
alone is not.

## 6. Acceptance criteria

- [ ] Encoders, predictor, and T-step unrolled loss implemented and shape-tested
- [ ] SIGReg implemented and unit-tested: isotropic Gaussian input → near-zero penalty;
      collapsed/constant input → large penalty
- [ ] Collapse diagnostics logged every epoch with an automatic alarm
- [ ] `λ_sig = 0` reproducibly collapses, demonstrating the regularizer is load-bearing
- [ ] Latent width configurable and swept
- [ ] Deterministic given a seed
