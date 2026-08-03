# flatjepa

**Measuring what JEPA world models actually learn, using differential flatness as analytic ground truth.**

---

## The problem this addresses

Joint Embedding Predictive Architectures (JEPAs) learn to predict in latent space rather than
observation space. This is their central advantage — the model is free to discard detail that is
unpredictable, instead of wasting capacity reconstructing it.

It is also their central evaluation problem: **you cannot directly tell what the latent learned.**

On standard benchmarks (Atari, DMControl, video) there is no analytic "true" state, so
representation quality is measured *indirectly* through downstream task performance. That
conflates three separate things:

1. whether the representation captured the system's state,
2. whether the predictor learned the dynamics,
3. whether the controller built on top is any good.

If the downstream number is bad, you do not know which one failed.

## The idea

A **differentially flat** system has a known, closed-form map from a small set of *flat outputs* to
the entire system state. If such a map exists, then the minimal sufficient latent for that system
is not a mystery — it is a known quantity, in both dimension and content.

That turns representation quality into something you can *measure directly*:

> Does the learned JEPA latent recover the flat outputs?

This repository studies that question on a quadrotor with a cable-suspended payload, using
trajectories from the [PolyFly](https://arxiv.org/abs/2510.15226) optimal planner, whose
differential-flatness map from payload jerk to full system state is exact and available in
closed form.

## Why this system in particular

The cable-suspended payload is unusually well suited to this question:

- **The flat map is exact.** Payload position and its derivatives determine quadrotor position,
  velocity, attitude, and cable direction analytically. Ground truth is free.
- **There is a genuine oracle.** PolyFly's optimizer produces globally optimal, non-conservative
  trajectories. Most world-model papers have no optimal reference to compare against.
- **The system is hybrid.** A cable is taut or slack — two dynamic regimes separated by a discrete
  event. Flatness holds only in the taut regime. This gives a built-in, physically meaningful
  boundary where the analytic model provably stops being valid, which is exactly where a learned
  residual should become visible.

## Research questions

| ID | Question | Measurement |
|----|----------|-------------|
| **E1** | Does the JEPA latent recover the flat outputs? | Linear probe latent → flat coordinates, R² against analytic ground truth |
| **E2** | Does the latent have the theoretically correct intrinsic dimensionality? | Participation ratio / PCA spectrum vs. known minimal dimension |
| **E3** | Does anti-collapse regularization (SIGReg) help or hurt recovery? | E1/E2 with SIGReg ablated — measured against ground truth, not downstream reward |
| **E4** | Does the latent discover the taut/slack hybrid mode boundary unsupervised? | Linear probe for cable tension sign on frozen latents |
| **E5** | Does the physics prober's residual localize violations of the flatness assumption? | Residual magnitude vs. analytic tension / jerk / yaw-deviation labels |

E3 is the one that is hard to ask anywhere else. Anti-collapse regularizers are normally justified
by "the representation didn't collapse and downstream accuracy went up." Here, collapse and recovery
can both be measured against a known target.

## Status

Early. The data-generation path is verified end-to-end; model and measurement layers are specified
in `docs/` and under construction. See [`docs/00-overview.md`](docs/00-overview.md) for the full
research plan and [`docs/`](docs/) for per-feature design documents.

## Repository layout

```
docs/               Design documents, one per feature (F1–F10) + research overview
src/flatjepa/
  data/             Trajectory generation, flat-output extraction, dataset assembly
  models/           JEPA encoder/predictor, physics-inspired prober
  probes/           Measurement suite (the scientific core)
  training/         Training harness, configs, checkpointing
configs/            Experiment configurations
scripts/            Entry points for data generation, training, evaluation
tests/              Correctness tests, including analytic sanity checks
```

## Requirements

- Docker (two images: a CasADi/IPOPT image for planning, a PyTorch+CUDA image for training)
- An NVIDIA GPU for training
- A local checkout of [PolyFly](https://github.com/arplaboratory/polyfly_ral) as the trajectory source

Setup instructions live in [`docs/F1-data-generation.md`](docs/F1-data-generation.md).

## References

### Directly built on

- **PolyFly: Polytopic Optimal Planning for Collision-Free Cable-Suspended Aerial Payload
  Transportation.** Sarvaiya, Li, Loianno, 2025. [arXiv:2510.15226](https://arxiv.org/abs/2510.15226)
  · [code](https://github.com/arplaboratory/polyfly_ral) — source of the trajectories, the flatness
  map, and the optimal-planner oracle.
- **SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors.**
  2026. [arXiv:2606.23444](https://arxiv.org/abs/2606.23444) ·
  [code](https://github.com/arplaboratory/SkyJEPA) — the architecture this work adapts: TCN
  encoders, GRU latent predictor, and a physics-inspired prober using differentiable kinematic
  integration.

### Differential flatness

- **Trajectory Generation and Control of a Quadrotor with a Cable-Suspended Load.** Sreenath,
  Michael, Kumar, CDC 2013 — flatness of the cable-suspended payload system, including the hybrid
  taut/slack regime. The reference for E4.
- **Minimum Snap Trajectory Generation and Control for Quadrotors.** Mellinger & Kumar, ICRA 2011 —
  the standard quadrotor flatness result.
- **Differential Flatness of Mechanical Control Systems.** Murray, Rathinam, Sluis, 1995 — general
  theory.

### JEPA and self-supervised representation learning

- **A Path Towards Autonomous Machine Intelligence.** LeCun, 2022 — the JEPA position paper.
- **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).**
  Assran et al., CVPR 2023.
- **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.** Balestriero &
  LeCun — origin of **SIGReg**, the anti-collapse regularizer used here.
  [code](https://github.com/rbalestr-lab/lejepa)
- **VICReg: Variance-Invariance-Covariance Regularization.** Bardes, Ponce, LeCun, ICLR 2022 —
  alternative anti-collapse formulation.
- **Bootstrap Your Own Latent (BYOL).** Grill et al., NeurIPS 2020 — the EMA target-encoder approach
  SkyJEPA deliberately does *not* use.

### Representation measurement

- **Understanding Intermediate Layers Using Linear Classifier Probes.** Alain & Bengio, 2016 — the
  linear-probe methodology underlying E1/E4.
- **Intrinsic Dimension of Data Representations in Deep Neural Networks.** Ansuini et al., NeurIPS
  2019 — the intrinsic-dimensionality analysis underlying E2.
- **Similarity of Neural Network Representations Revisited (CKA).** Kornblith et al., ICML 2019.

### Model-based control (context for downstream use)

- **Model Predictive Path Integral Control (MPPI).** Williams et al., 2017.
- **Dream to Control / DreamerV3.** Hafner et al. — latent world models with distilled policies.
- **TD-MPC2: Scalable, Robust World Models for Continuous Control.** Hansen et al., ICLR 2024 —
  combines a learned policy prior with sampling-based planning.

## License

MIT — see [LICENSE](LICENSE).
