# F2 — Flat-Output Extractor (Ground Truth)

**Purpose:** compute, for every timestep of every trajectory, the analytic flat coordinates and the
full system state they determine. This is the ground truth that the entire project measures against.

**Serves:** E1 (latent recovery), E2 (intrinsic dimensionality), E5 (residual monitor).

---

## 1. Why this is re-implemented rather than imported

The upstream planner already contains this map (`Planner.differential_flatness`). We re-implement it
here anyway, for three reasons:

1. **Independence.** If ground truth is produced by the same code path that produced the data, a bug
   in that path is invisible — it would corrupt data and ground truth identically and consistently.
2. **Vectorization.** Upstream evaluates per-timestep inside plotting/saving paths. We need it over
   whole trajectories.
3. **Version pinning.** Upstream is an external repository that may change.

**Required test:** the re-implementation must agree with the upstream function to tight numerical
tolerance on a sample of real trajectories. This cross-check is the actual value of re-implementing;
without it, this is just duplicated code. See `tests/test_flatness_agreement.py`.

## 2. The flat map

Inputs, per timestep: payload position `p`, velocity `v`, acceleration `a`, jerk `j`. Constants:
quadrotor mass `mQ`, payload mass `mL`, cable length `L`, gravity `g = 9.81`. Yaw is pinned to zero
by upstream convention.

Define the gravity-compensated acceleration and its normalized direction:

```
a_g   = a + g·e₃
p̂     = -a_g / ‖a_g‖          (cable unit vector, payload → quad is -p̂)
```

with `ṗ̂` and `p̈̂` following from differentiating `p̂`. The net force direction is

```
F     = (mQ + mL)·a_g − mQ·L·p̈̂
b₃    = F / ‖F‖
```

which yields quadrotor attitude (with `b₁`, `b₂` from the zero-yaw convention), and

```
p_quad = p + b₃·L
v_quad = v − L·ṗ̂
a_quad = a − L·p̈̂
```

Everything on the right-hand side is a function of `(p, v, a, j)` alone. That is the flatness
property, and it is the entire basis of this project.

## 3. The flat coordinate vector

The quantity probes target in E1:

```
z_flat = [p (3), v (3), a (3), j (3)]  ∈ ℝ¹²
```

**Minimal sufficient latent dimension.** For predicting the future *given the action sequence*, the
sufficient state is `(p, v, a) ∈ ℝ⁹` — jerk is the control input, not state. This is not a
guess: it is exactly the planner's own state vector (`sol_x`, 9 columns) under triple-integrator
dynamics.

So E2 has a concrete numerical prediction rather than a vague one:

> The learned latent's effective dimensionality should be **≈ 9**.

Report both: recovery of the 9-dim state (the strong claim) and of the full 12-dim vector including
jerk (which tests whether the latent also encodes the commanded input).

## 4. Derived quantities also emitted

Beyond the flat coordinates themselves, F2 emits the analytically determined full state, so that
F6's prober has a nominal model to correct and F7 can probe for physically meaningful quantities:

| Quantity | Shape | Note |
|----------|-------|------|
| `p_quad`, `v_quad`, `a_quad` | (T, 3) each | quadrotor kinematics |
| `R_quad` | (T, 3, 3) | attitude; compare against upstream `rot_mat` columns |
| `quat_quad` | (T, 4) | note the convention issue in §5 |
| `cable_dir` | (T, 3) | unit vector |
| `tension_margin` | (T,) | passed to F3 |

## 5. Numerical hazards

These are the places this will silently produce wrong ground truth. Each needs an explicit guard and
a test.

- **Quaternion sign ambiguity.** `q` and `−q` are the same rotation. Naive per-timestep conversion
  produces sign flips that look like discontinuities and will wreck any regression target. Enforce
  hemisphere continuity along each trajectory.
- **Quaternion convention.** SciPy's `Rotation.as_quat()` returns `(x, y, z, w)`. Upstream CSV
  columns must be checked against this rather than assumed — the upstream code mixes conventions
  across files, and `utils.get_rotation_matrix_from_quat` consumes `(x,y,z,w)`.
- **Near-free-fall singularity.** `p̂` divides by `‖a_g‖`, which → 0 in free fall. This is not a
  hypothetical: see F3 §2. Guard the division, and record affected timesteps rather than silently
  emitting NaN or a garbage direction.
- **Degenerate cross product.** `b₁ ∝ b₂d × b₃` is undefined when `b₃` is parallel to
  `b₂d = (−sin ψ, cos ψ, 0)`. Guard and record.
- **Jerk source.** Prefer the planner's own `sol_u` (which *is* jerk) over numerically
  differentiating acceleration. Numerical differentiation of an interpolated signal will inject
  high-frequency noise directly into the ground truth. If both are available, cross-check them and
  report the discrepancy — a large one indicates an interpolation problem upstream.

## 6. Acceptance criteria

- [ ] Vectorized over a full trajectory; no Python-level per-timestep loop in the hot path
- [ ] Agrees with upstream `differential_flatness` within tight tolerance on sampled real
      trajectories (this is the primary test)
- [ ] Quaternion output is hemisphere-continuous along each trajectory
- [ ] All singularity guards fire on synthetic adversarial inputs (exact free fall, `b₃ ∥ b₂d`)
- [ ] Emits a per-timestep validity mask; downstream consumers must respect it
- [ ] Analytic `R_quad` reconciles with the upstream `rot_mat_*` CSV columns
