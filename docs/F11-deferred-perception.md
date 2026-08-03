# F11 — Perception (Deferred)

**Status:** deferred out of v1. This document exists so the design decision is recorded rather than
forgotten, and so the work can slot in without redesign.

---

## 1. Why it is deferred

v1 is state-only (see `00-overview.md` §6). Adding perception would introduce an entire subsystem —
rendering, a visual encoder, and a much larger model — before any result exists. The
flat-latent-recovery question does not require it.

## 2. What it would enable

The questions perception unlocks are about **multimodality**. Given a depth image and current state,
which side of an obstacle the planner routes around is genuinely underdetermined by the input: it
depends on global structure outside the camera's view. That is irreducible unpredictability from
partial observation, and it is the setting where latent-predictive models are claimed to beat direct
regression — a regression model trained with an L2 loss averages the left and right routes and
produces a trajectory into the obstacle.

Relevant existing hook: the upstream PolyFly viewer policy already expects a model returning
`(predicted_traj, mode_probs)` with `M` modes, so multimodality was anticipated in that codebase.

## 3. Implementation sketch: analytic depth

**Isaac Gym is not viable on this machine** — it requires Python 3.8 and is deprecated upstream;
the available environments are Python 3.10 and 3.12.

Fortunately it is not needed. Obstacles in the PolyFly parameter YAMLs are **axis-aligned boxes**
(`x, y, z, l, b, h`). A depth image can be raycast against axis-aligned boxes analytically with the
standard slab method, vectorized in NumPy — on the order of a hundred lines, no simulator, no
renderer, no GPU.

This is a genuinely cheaper path to depth than the upstream Isaac Gym pipeline, and it is exact
rather than approximate for this obstacle class.

## 4. Prerequisites before starting

- v1 state-only results complete, including the E1 controls
- F3 histogram known, so it is clear which phenomena actually exist in the data
- A decision on whether to reuse the upstream `(traj, mode_probs)` interface or define a new one
