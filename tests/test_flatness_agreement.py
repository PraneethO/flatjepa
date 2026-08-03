"""The cross-check that justifies re-implementing the flat map (F2 §1, §6).

Our vectorized extractor must agree with upstream's per-timestep ``Planner.differential_flatness``
to tight numerical tolerance on real trajectory data. Without this test, ``flatness.py`` is just
duplicated code.

The upstream function is not imported (CasADi is unavailable in this environment); it is lifted out
of upstream's own source with :mod:`ast` and executed here. See ``tests/upstream_ref.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flatjepa.data.csv_io import find_trajectory_csvs, load_trajectory_csv
from flatjepa.data.flatness import (
    SystemParams,
    flat_outputs,
    flat_outputs_from_trajectory,
    jerk_discrepancy,
    jerk_from_acceleration,
)
from upstream_ref import (
    UpstreamParams,
    UpstreamUnavailable,
    load_upstream_differential_flatness,
    polyfly_dir,
)

# Machine-epsilon agreement is what the implementations actually achieve (~5e-16 max abs error over
# the corpus). 1e-12 leaves three orders of headroom for BLAS/platform variation while still being
# far tighter than any physically meaningful discrepancy.
TOL = 1e-12

#: How many timesteps per trajectory to push through the per-timestep upstream reference.
MAX_STEPS_PER_TRAJ = 300
#: How many trajectories to sample. The corpus is homogeneous; this keeps the suite fast.
MAX_TRAJECTORIES = 8

PARAMS = SystemParams()
UPSTREAM_PARAMS = UpstreamParams(
    mass_quad=PARAMS.mass_quad,
    mass_load=PARAMS.mass_load,
    cable_length=PARAMS.cable_length,
)


def _csv_paths() -> list:
    roots = [polyfly_dir() / "data" / "csvs"]
    return [p for p in find_trajectory_csvs(roots) if p.is_file()]


@pytest.fixture(scope="module")
def upstream_fn():
    try:
        return load_upstream_differential_flatness()
    except UpstreamUnavailable as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"upstream PolyFly reference unavailable: {exc}")


@pytest.fixture(scope="module")
def trajectories() -> list:
    paths = _csv_paths()
    if not paths:
        pytest.skip(f"no trajectory CSVs found under {polyfly_dir() / 'data' / 'csvs'}")
    # Spread the sample across the corpus instead of taking the first few, which would only ever
    # exercise one generation batch.
    step = max(1, len(paths) // MAX_TRAJECTORIES)
    return [load_trajectory_csv(p) for p in paths[::step][:MAX_TRAJECTORIES]]


def _sample_indices(n: int) -> np.ndarray:
    stride = max(1, n // MAX_STEPS_PER_TRAJ)
    return np.arange(0, n, stride)


def test_agrees_with_upstream_differential_flatness(upstream_fn, trajectories):
    """Primary test: identical outputs to upstream, timestep by timestep, on real data."""
    worst = {"orientation": 0.0, "pos_quad": 0.0, "vel_quad": 0.0, "acc_quad": 0.0,
             "payload_rpy": 0.0, "quat": 0.0}
    n_compared = 0

    for traj in trajectories:
        ours = flat_outputs_from_trajectory(traj, params=PARAMS)
        assert ours.valid.all(), (
            f"{traj.name}: guards fired on real data; the comparison below would be vacuous. "
            f"{ours.guard_counts()}"
        )

        for i in _sample_indices(traj.n_steps):
            ref = upstream_fn(
                traj.payload_pos[i],
                traj.payload_vel[i],
                traj.payload_acc[i],
                traj.payload_jerk[i],
                UPSTREAM_PARAMS,
            )
            mine = {
                "orientation": ours.R_quad[i],
                "pos_quad": ours.p_quad[i],
                "vel_quad": ours.v_quad[i],
                "acc_quad": ours.a_quad[i],
                "payload_rpy": ours.payload_rpy[i],
            }
            for key, value in mine.items():
                err = float(np.max(np.abs(np.asarray(ref[key]) - value)))
                worst[key] = max(worst[key], err)
                assert err < TOL, f"{traj.name}[{i}] {key}: max abs error {err:.3e} >= {TOL:.0e}"

            # q and -q are the same rotation; our output is hemisphere-normalized along the
            # trajectory, so the comparison is up to a global sign by construction.
            q_err = float(
                min(
                    np.max(np.abs(ref["quat"] - ours.quat_quad[i])),
                    np.max(np.abs(ref["quat"] + ours.quat_quad[i])),
                )
            )
            worst["quat"] = max(worst["quat"], q_err)
            assert q_err < TOL, f"{traj.name}[{i}] quat: max abs error {q_err:.3e} >= {TOL:.0e}"
            n_compared += 1

    print(
        f"\nupstream agreement over {n_compared} timesteps / {len(trajectories)} trajectories: "
        + ", ".join(f"{k}={v:.3e}" for k, v in worst.items())
    )
    assert n_compared > 0


def test_vectorized_matches_single_step_evaluation(trajectories):
    """Batch evaluation must equal evaluating each timestep alone — no cross-timestep leakage."""
    traj = trajectories[0]
    batch = flat_outputs_from_trajectory(traj, params=PARAMS)
    for i in _sample_indices(traj.n_steps)[:25]:
        single = flat_outputs(
            traj.payload_pos[i : i + 1],
            traj.payload_vel[i : i + 1],
            traj.payload_acc[i : i + 1],
            traj.payload_jerk[i : i + 1],
            params=PARAMS,
        )
        np.testing.assert_allclose(single.p_quad[0], batch.p_quad[i], atol=TOL, rtol=0)
        np.testing.assert_allclose(single.v_quad[0], batch.v_quad[i], atol=TOL, rtol=0)
        np.testing.assert_allclose(single.a_quad[0], batch.a_quad[i], atol=TOL, rtol=0)
        np.testing.assert_allclose(single.R_quad[0], batch.R_quad[i], atol=TOL, rtol=0)


def test_quad_kinematics_match_csv_columns(trajectories):
    """``sol_quad_x_*`` is written straight out of upstream's flat map, so it must match exactly."""
    for traj in trajectories:
        ours = flat_outputs_from_trajectory(traj, params=PARAMS)
        np.testing.assert_allclose(ours.p_quad, traj.quad_pos, atol=1e-9, rtol=0)
        np.testing.assert_allclose(ours.v_quad, traj.quad_vel, atol=1e-9, rtol=0)
        np.testing.assert_allclose(ours.a_quad, traj.quad_acc, atol=1e-9, rtol=0)
        np.testing.assert_allclose(ours.payload_rpy, traj.payload_rpy, atol=1e-12, rtol=0)


def test_csv_quaternion_convention_is_xyzw(trajectories):
    """F2 §5: verify the CSV quaternion convention rather than assuming it.

    ``rot_mat_*`` holds the first two columns of the same rotation the quaternion encodes. Reading
    the quaternion as ``(x, y, z, w)`` must reproduce those columns; reading it as ``(w, x, y, z)``
    must not. If both matched we would have learned nothing, so the negative check is part of it.
    """
    for traj in trajectories:
        as_xyzw = Rotation.from_quat(traj.quad_quat).as_matrix()[:, :, :2]
        np.testing.assert_allclose(as_xyzw, traj.quad_rot_cols, atol=1e-9, rtol=0)

        rolled = np.roll(traj.quad_quat, -1, axis=1)  # interpret stored order as (w, x, y, z)
        as_wxyz = Rotation.from_quat(rolled).as_matrix()[:, :, :2]
        assert np.max(np.abs(as_wxyz - traj.quad_rot_cols)) > 1e-6, (
            f"{traj.name}: both quaternion conventions reproduce rot_mat, so this trajectory "
            "cannot discriminate between them"
        )


def test_analytic_attitude_reconciles_with_csv_rot_mat(trajectories):
    """F2 §6: reconcile analytic ``R_quad`` with the upstream ``rot_mat_*`` columns.

    They are *not* equal, and the reason is not an error in either: after calling
    ``differential_flatness``, upstream post-processes the attitude with
    ``get_yaw_along_trajectory``, which composes a rate-limited, velocity-heading yaw about the
    world z-axis on top of the zero-yaw flat-map attitude
    (``planner.py`` line ~1350). The quadrotor *position/velocity/acceleration* columns are written
    before that step, which is why they match to machine precision while attitude does not.

    The reconciliation is therefore: ``R_csv = R_z(ψ) · R_flat`` exactly, for some per-timestep ψ.
    """
    max_residual = 0.0
    max_yaw = 0.0
    for traj in trajectories:
        ours = flat_outputs_from_trajectory(traj, params=PARAMS)
        c0 = traj.quad_rot_cols[:, :, 0]
        c1 = traj.quad_rot_cols[:, :, 1]
        r_csv = np.stack([c0, c1, np.cross(c0, c1)], axis=2)

        # R_rel = R_csv · R_flatᵀ must be a pure rotation about the world z-axis.
        r_rel = np.einsum("nij,nkj->nik", r_csv, ours.R_quad)
        residual = max(
            float(np.max(np.abs(r_rel[:, 2, 2] - 1.0))),
            float(np.max(np.abs(r_rel[:, :2, 2]))),
            float(np.max(np.abs(r_rel[:, 2, :2]))),
        )
        assert residual < 1e-9, (
            f"{traj.name}: R_csv·R_flatᵀ is not a rotation about z (residual {residual:.3e}); "
            "the analytic attitude does not reconcile with the CSV"
        )
        max_residual = max(max_residual, residual)
        max_yaw = max(max_yaw, float(np.max(np.abs(np.arctan2(r_rel[:, 1, 0], r_rel[:, 0, 0])))))

    print(
        f"\nR_csv = R_z(psi)·R_flat: max residual {max_residual:.3e}, "
        f"max |psi| {max_yaw:.4f} rad"
    )


def test_quaternion_is_hemisphere_continuous(trajectories):
    """F2 §5/§6: no sign flips along a trajectory."""
    for traj in trajectories:
        ours = flat_outputs_from_trajectory(traj, params=PARAMS)
        quat = ours.quat_quad[ours.valid]
        dots = np.sum(quat[1:] * quat[:-1], axis=1)
        assert np.all(dots >= 0.0), (
            f"{traj.name}: {int(np.count_nonzero(dots < 0))} quaternion sign flips remain"
        )
        # Attitude is smooth at 500 Hz, so consecutive quaternions should be nearly identical.
        # A surviving flip would show up as a dot near -1 and hence a step of order 2.
        steps = np.linalg.norm(np.diff(quat, axis=0), axis=1)
        assert float(np.max(steps)) < 0.5, f"{traj.name}: quaternion discontinuity detected"


def test_jerk_source_cross_check(trajectories):
    """F2 §5: ``sol_u`` is the jerk of record; report how far numerical differentiation lands.

    This is diagnostic, not a correctness assertion on our code — a large discrepancy would point
    at upstream's interpolation, which is why the number is printed with a loose sanity bound
    rather than pinned.
    """
    rows = []
    for traj in trajectories:
        stats = jerk_discrepancy(
            traj.payload_jerk, jerk_from_acceleration(traj.payload_acc, traj.time)
        )
        rows.append((traj.name, traj.dt, stats))
        assert np.isfinite(stats["mean_abs_interior"])

    print("\njerk cross-check (sol_u vs. d(acc)/dt):")
    for name, dt, stats in rows:
        print(
            f"  {name[:44]:44s} dt={dt:.4f}  mean|Δ|={stats['mean_abs_interior']:.3e}  "
            f"max|Δ|={stats['max_abs_interior']:.3e}  rms_rel={stats['rms_rel']:.3e}  "
            f"mean|j|={stats['mean_jerk_norm']:.3e}"
        )
