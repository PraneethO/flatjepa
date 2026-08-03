"""F2 — vectorized differential-flatness extractor (ground truth).

Given the payload flat outputs ``(p, v, a, j)`` this computes, in closed form, the full state of the
quadrotor / cable-suspended-payload system: quadrotor kinematics, quadrotor attitude, cable
direction, and the payload orientation relative to the cable.

This duplicates ``Planner.differential_flatness`` in the upstream PolyFly repository on purpose:

1. **Independence.** Ground truth produced by the same code path that produced the data would hide
   any bug in that path — data and ground truth would be wrong identically and consistently.
2. **Vectorization.** Upstream evaluates one timestep at a time inside plotting/saving paths. We
   need whole trajectories at once.
3. **Version pinning.** Upstream is an external repository that may change under us.

The value of duplicating is entirely in the cross-check: ``tests/test_flatness_agreement.py``
executes the upstream function on real trajectory data and requires agreement to tight tolerance.

Conventions match upstream: yaw is pinned to zero, quaternions are ``(x, y, z, w)``, and the
rotation matrix has columns ``(b₁, b₂, b₃)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from flatjepa.data.csv_io import Trajectory, load_trajectory_csv
from flatjepa.data.tension import cable_tension

__all__ = [
    "SystemParams",
    "GuardConfig",
    "FlatOutputs",
    "flat_outputs",
    "flat_outputs_from_csv",
    "flat_outputs_from_trajectory",
    "jerk_from_acceleration",
    "jerk_discrepancy",
]


@dataclass(frozen=True)
class SystemParams:
    """Physical constants of the quadrotor / cable / payload system.

    Defaults are the values used throughout the upstream corpus
    (``polyfly_ral/data/params/*/base.yaml``).
    """

    mass_quad: float = 0.715
    mass_load: float = 0.163
    cable_length: float = 0.567
    gravity: float = 9.81

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SystemParams":
        """Read the constants out of an upstream ``base.yaml``, ignoring every other key."""
        with open(path, "r") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        return cls(
            mass_quad=float(raw["mass_quad"]),
            mass_load=float(raw["mass_load"]),
            cable_length=float(raw["cable_length"]),
            gravity=float(raw.get("gravity", 9.81)),
        )

    @property
    def mass_total(self) -> float:
        return self.mass_quad + self.mass_load


@dataclass(frozen=True)
class GuardConfig:
    """Thresholds for the three places the flat map is numerically undefined (F2 §5).

    All are absolute thresholds on quantities that appear in a denominator.
    """

    #: ``‖a + g·e₃‖`` below this is treated as free fall. The derivative terms of the cable
    #: direction scale as ``1/‖a_g‖³``, so this is deliberately well above machine epsilon while
    #: still corresponding to a tension margin of ~1e-4 — physically unreachable in taut flight.
    min_acc_norm: float = 1e-3
    #: ``‖F‖`` below this leaves ``b₃`` undefined.
    min_force_norm: float = 1e-9
    #: ``‖b₂d × b₃‖`` below this leaves ``b₁`` undefined (``b₃`` parallel to ``b₂d``).
    min_cross_norm: float = 1e-9


@dataclass
class FlatOutputs:
    """Everything the flat map determines, for a whole trajectory.

    Every array has leading axis ``T`` and is index-aligned with the inputs, with F3's tension
    labels, and with F4's windows. Rows where ``valid`` is ``False`` carry ``NaN`` in every derived
    field: the guards never emit a plausible-looking but wrong direction. ``z_flat`` and the raw
    ``p/v/a/j`` are left untouched, since they are inputs rather than derived quantities.
    """

    # --- the flat coordinates themselves (F2 §3) ---
    z_flat: np.ndarray  # (T, 12) = [p, v, a, j]
    p: np.ndarray  # (T, 3)
    v: np.ndarray  # (T, 3)
    a: np.ndarray  # (T, 3)
    j: np.ndarray  # (T, 3)

    # --- derived full state (F2 §4) ---
    p_quad: np.ndarray  # (T, 3)
    v_quad: np.ndarray  # (T, 3)
    a_quad: np.ndarray  # (T, 3)
    R_quad: np.ndarray  # (T, 3, 3), columns (b1, b2, b3)
    quat_quad: np.ndarray  # (T, 4), (x, y, z, w), hemisphere-continuous along the trajectory
    cable_dir: np.ndarray  # (T, 3) unit vector, payload -> quadrotor
    payload_rpy: np.ndarray  # (T, 3) payload orientation relative to the cable
    tension: np.ndarray  # (T,) newtons
    tension_margin: np.ndarray  # (T,) tension / (mL·g)

    # --- flat-map intermediates, kept because F6's nominal model needs them ---
    p_hat: np.ndarray  # (T, 3) = -a_g/‖a_g‖
    p_hat_dot: np.ndarray  # (T, 3)
    p_hat_ddot: np.ndarray  # (T, 3)

    # --- validity (F2 §5, §6) ---
    # The four guard flags are mutually exclusive: a timestep is attributed to the first condition
    # that failed, so their counts partition the invalid set. See :func:`flat_outputs`.
    valid: np.ndarray  # (T,) bool; downstream consumers must respect this
    guard_free_fall: np.ndarray  # (T,) bool, ‖a + g·e₃‖ below threshold
    guard_force: np.ndarray  # (T,) bool, ‖F‖ below threshold
    guard_degenerate_cross: np.ndarray  # (T,) bool, b₃ ∥ b₂d
    guard_nonfinite_input: np.ndarray  # (T,) bool

    params: SystemParams = SystemParams()

    @property
    def n_steps(self) -> int:
        return int(self.z_flat.shape[0])

    @property
    def n_invalid(self) -> int:
        return int(np.count_nonzero(~self.valid))

    def guard_counts(self) -> dict[str, int]:
        """How many timesteps each guard fired on. Useful in run logs; cheap to call."""
        return {
            "free_fall": int(np.count_nonzero(self.guard_free_fall)),
            "force": int(np.count_nonzero(self.guard_force)),
            "degenerate_cross": int(np.count_nonzero(self.guard_degenerate_cross)),
            "nonfinite_input": int(np.count_nonzero(self.guard_nonfinite_input)),
            "invalid_total": self.n_invalid,
        }


def _as_3d(name: str, arr: np.ndarray, n: int | None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError(f"{name} must have shape (T, 3), got {out.shape}")
    if n is not None and out.shape[0] != n:
        raise ValueError(f"{name} has {out.shape[0]} timesteps, expected {n}")
    return out


def _enforce_hemisphere_continuity(quat: np.ndarray) -> np.ndarray:
    """Remove the ``q ≡ −q`` sign ambiguity along a trajectory.

    Per-timestep conversion from a rotation matrix picks the sign arbitrarily, which shows up as
    step discontinuities in an otherwise smooth signal and would wreck any regression target. We
    flip each quaternion into the hemisphere of its predecessor, and fix the global sign by giving
    the first finite quaternion a non-negative scalar part so the output is deterministic.

    Vectorized via a cumulative product of per-step flips: with ``s_t = Π_{k≤t} flip_k`` the
    corrected sequence satisfies ``⟨q'_t, q'_{t-1}⟩ = flip_t · ⟨q_t, q_{t-1}⟩ ≥ 0``.
    """
    quat = np.array(quat, dtype=np.float64, copy=True)
    if quat.shape[0] < 1:
        return quat

    dots = np.sum(quat[1:] * quat[:-1], axis=1)
    # NaN rows (guarded timesteps) break the chain; treat them as no-flip rather than propagating.
    flips = np.where(dots < 0.0, -1.0, 1.0)
    signs = np.concatenate(([1.0], np.cumprod(flips)))
    quat *= signs[:, None]

    finite = np.all(np.isfinite(quat), axis=1)
    if np.any(finite):
        first = int(np.argmax(finite))
        if quat[first, 3] < 0.0:
            quat *= -1.0
    return quat


def flat_outputs(
    p: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    j: np.ndarray,
    params: SystemParams = SystemParams(),
    yaw: float | np.ndarray = 0.0,
    guards: GuardConfig = GuardConfig(),
) -> FlatOutputs:
    """Evaluate the flat map over a whole trajectory.

    Parameters
    ----------
    p, v, a, j
        Payload position, velocity, acceleration and jerk, each ``(T, 3)``. Prefer the planner's
        own ``sol_u`` for ``j`` over numerically differentiating ``a`` (F2 §5).
    yaw
        Scalar or ``(T,)``. Upstream pins this to zero; it is exposed only so the degenerate
        ``b₃ ∥ b₂d`` case can be constructed in tests.

    Notes
    -----
    Fully vectorized: there is no Python-level per-timestep loop. The one non-elementwise step is
    the matrix→quaternion conversion, which SciPy performs in a single batched call over the valid
    rows.
    """
    p = _as_3d("p", p, None)
    n = p.shape[0]
    v = _as_3d("v", v, n)
    a = _as_3d("a", a, n)
    j = _as_3d("j", j, n)

    nonfinite = ~(
        np.all(np.isfinite(p), axis=1)
        & np.all(np.isfinite(v), axis=1)
        & np.all(np.isfinite(a), axis=1)
        & np.all(np.isfinite(j), axis=1)
    )

    g = params.gravity
    acc = a.copy()
    acc[:, 2] += g  # a_g = a + g·e₃

    # Replace non-finite rows with a benign placeholder so NaN/inf cannot spread through the
    # vectorized arithmetic or raise warnings. These rows are blanked at the end regardless, and
    # the *reported* p/v/a/j stay exactly as the caller supplied them.
    j_calc = j
    if np.any(nonfinite):
        acc = np.where(nonfinite[:, None], np.array([0.0, 0.0, g]), acc)
        j_calc = np.where(nonfinite[:, None], 0.0, j)

    acc_norm = np.linalg.norm(acc, axis=1)

    # --- Guard 1: near-free-fall singularity. p̂ divides by ‖a_g‖, which → 0 in free fall. ---
    raw_free_fall = acc_norm < guards.min_acc_norm
    safe_norm = np.where(raw_free_fall, 1.0, acc_norm)
    inv1 = 1.0 / safe_norm
    inv2 = inv1 * inv1
    inv3 = inv2 * inv1

    p_hat = -acc * inv1[:, None]

    acc_dot_jrk = np.einsum("ij,ij->i", acc, j_calc)
    jrk_sq = np.einsum("ij,ij->i", j_calc, j_calc)
    dot_acc_norm = acc_dot_jrk * inv1
    ddot_acc_norm = jrk_sq * inv1 - (acc_dot_jrk**2) * inv3

    p_hat_dot = -j_calc * inv1[:, None] + (dot_acc_norm * inv2)[:, None] * acc
    p_hat_ddot = (
        2.0 * (dot_acc_norm * inv2)[:, None] * j_calc
        + (ddot_acc_norm * inv2)[:, None] * acc
        - 2.0 * ((dot_acc_norm**2) * inv3)[:, None] * acc
    )

    force = params.mass_total * acc - params.mass_quad * params.cable_length * p_hat_ddot
    force_norm = np.linalg.norm(force, axis=1)

    # --- Guard 2: ‖F‖ → 0 leaves the thrust direction b₃ undefined. ---
    raw_force = force_norm < guards.min_force_norm
    b3 = force / np.where(raw_force, 1.0, force_norm)[:, None]

    yaw_arr = np.broadcast_to(np.asarray(yaw, dtype=np.float64), (n,))
    b2d = np.stack([-np.sin(yaw_arr), np.cos(yaw_arr), np.zeros(n)], axis=1)

    cross_b2d_b3 = np.cross(b2d, b3)
    cross_norm = np.linalg.norm(cross_b2d_b3, axis=1)

    # --- Guard 3: b₁ ∝ b₂d × b₃ is undefined when b₃ ∥ b₂d. ---
    raw_cross = cross_norm < guards.min_cross_norm
    b1 = cross_b2d_b3 / np.where(raw_cross, 1.0, cross_norm)[:, None]
    b2 = np.cross(b3, b1)

    R = np.stack([b1, b2, b3], axis=2)  # columns, matching upstream np.column_stack

    valid = ~(raw_free_fall | raw_force | raw_cross | nonfinite)

    # Failures cascade — exact free fall drives ‖F‖ to zero, which in turn makes b₃ degenerate — so
    # each timestep is attributed to the *first* condition that failed, in the order the map
    # evaluates them. The per-guard counts then partition the invalid set instead of double-counting
    # one root cause across three flags.
    guard_free_fall = raw_free_fall & ~nonfinite
    guard_force = raw_force & ~(nonfinite | guard_free_fall)
    guard_cross = raw_cross & ~(nonfinite | guard_free_fall | guard_force)

    p_quad = p + b3 * params.cable_length
    v_quad = v - params.cable_length * p_hat_dot
    a_quad = a - params.cable_length * p_hat_ddot

    # Payload orientation relative to the cable. Upstream computes the rotation taking the
    # quadrotor body z-axis (R[:, 2] = b₃) onto the cable direction, but under this flatness
    # construction the quadrotor is *placed* along b₃ (p_quad = p + b₃·L), so the two vectors are
    # identical by construction and the rotation is the identity at every timestep. We emit zeros
    # rather than recomputing an identity; the agreement test checks upstream returns zeros too.
    payload_rpy = np.zeros((n, 3), dtype=np.float64)

    quat = np.full((n, 4), np.nan, dtype=np.float64)
    if np.any(valid):
        quat[valid] = Rotation.from_matrix(R[valid]).as_quat()  # (x, y, z, w)
    quat = _enforce_hemisphere_continuity(quat)

    tension = cable_tension(a, mass_load=params.mass_load, gravity=g)
    margin = tension / (params.mass_load * g)

    # Blank every derived field on guarded rows: better an explicit NaN plus a mask than a
    # plausible-looking direction that quietly poisons the ground truth.
    invalid = ~valid
    for arr in (p_quad, v_quad, a_quad, b3, p_hat, p_hat_dot, p_hat_ddot, payload_rpy):
        arr[invalid] = np.nan
    R[invalid] = np.nan
    quat[invalid] = np.nan

    return FlatOutputs(
        z_flat=np.concatenate([p, v, a, j], axis=1),
        p=p,
        v=v,
        a=a,
        j=j,
        p_quad=p_quad,
        v_quad=v_quad,
        a_quad=a_quad,
        R_quad=R,
        quat_quad=quat,
        cable_dir=b3,
        payload_rpy=payload_rpy,
        tension=tension,
        tension_margin=margin,
        p_hat=p_hat,
        p_hat_dot=p_hat_dot,
        p_hat_ddot=p_hat_ddot,
        valid=valid,
        guard_free_fall=guard_free_fall,
        guard_force=guard_force,
        guard_degenerate_cross=guard_cross,
        guard_nonfinite_input=nonfinite,
        params=params,
    )


def flat_outputs_from_trajectory(
    traj: Trajectory,
    params: SystemParams = SystemParams(),
    guards: GuardConfig = GuardConfig(),
) -> FlatOutputs:
    """Flat outputs for a loaded trajectory, using ``sol_u`` as the jerk (F2 §5)."""
    return flat_outputs(
        traj.payload_pos,
        traj.payload_vel,
        traj.payload_acc,
        traj.payload_jerk,
        params=params,
        guards=guards,
    )


def flat_outputs_from_csv(
    path: str | Path,
    params: SystemParams = SystemParams(),
    guards: GuardConfig = GuardConfig(),
) -> FlatOutputs:
    """Load a planner CSV and evaluate the flat map over it."""
    return flat_outputs_from_trajectory(load_trajectory_csv(path), params=params, guards=guards)


def jerk_from_acceleration(a: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Numerically differentiate acceleration to get jerk (central differences, ``np.gradient``).

    Provided only as a *cross-check* against ``sol_u``. It must not be used as the ground-truth
    jerk: differentiating an interpolated signal injects high-frequency noise straight into the
    regression target.
    """
    a = _as_3d("a", a, None)
    time = np.asarray(time, dtype=np.float64)
    return np.gradient(a, time, axis=0, edge_order=2)


def jerk_discrepancy(jerk_control: np.ndarray, jerk_numeric: np.ndarray) -> dict[str, float]:
    """Compare ``sol_u`` against numerically differentiated acceleration.

    A large discrepancy points at an interpolation problem upstream rather than at this code, so it
    is worth reporting rather than asserting on. Interior statistics exclude the first and last
    samples, where ``np.gradient``'s one-sided stencils dominate the error.
    """
    jerk_control = _as_3d("jerk_control", jerk_control, None)
    jerk_numeric = _as_3d("jerk_numeric", jerk_numeric, jerk_control.shape[0])
    err = np.linalg.norm(jerk_control - jerk_numeric, axis=1)
    scale = np.linalg.norm(jerk_control, axis=1)
    interior = err[1:-1] if err.size > 2 else err
    denom = float(np.mean(scale)) if np.mean(scale) > 0 else float("nan")
    return {
        "max_abs": float(np.max(err)),
        "max_abs_interior": float(np.max(interior)),
        "mean_abs": float(np.mean(err)),
        "mean_abs_interior": float(np.mean(interior)),
        "rms_rel": float(np.sqrt(np.mean(interior**2)) / denom),
        "mean_jerk_norm": float(np.mean(scale)),
    }
