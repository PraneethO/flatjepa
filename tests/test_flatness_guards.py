"""F2 §5/§6 — the singularity guards must fire on synthetic adversarial inputs.

Each of these constructs an input that makes some denominator in the flat map vanish. The
requirement is not just "does not crash": the affected timesteps must be *recorded* in the validity
mask, and their derived outputs must be unmistakably absent (NaN) rather than a plausible-looking
direction that would quietly poison the ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from flatjepa.data.flatness import FlatOutputs, GuardConfig, SystemParams, flat_outputs

PARAMS = SystemParams()
G = PARAMS.gravity

DERIVED_VECTOR_FIELDS = (
    "p_quad",
    "v_quad",
    "a_quad",
    "cable_dir",
    "p_hat",
    "p_hat_dot",
    "p_hat_ddot",
    "payload_rpy",
)


def _assert_blanked(out: FlatOutputs, rows: np.ndarray) -> None:
    """Every derived field must be NaN on the guarded rows, and finite elsewhere."""
    rows = np.atleast_1d(rows)
    for field in DERIVED_VECTOR_FIELDS:
        arr = getattr(out, field)
        assert np.all(np.isnan(arr[rows])), f"{field} not blanked on guarded rows"
    assert np.all(np.isnan(out.R_quad[rows]))
    assert np.all(np.isnan(out.quat_quad[rows]))

    keep = np.ones(out.n_steps, dtype=bool)
    keep[rows] = False
    if keep.any():
        for field in DERIVED_VECTOR_FIELDS:
            assert np.all(np.isfinite(getattr(out, field)[keep])), f"{field} corrupted on good rows"
        assert np.all(np.isfinite(out.R_quad[keep]))
        assert np.all(np.isfinite(out.quat_quad[keep]))


def _benign(n: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(7)
    return (
        rng.normal(size=(n, 3)),
        rng.normal(size=(n, 3)),
        rng.normal(scale=1.5, size=(n, 3)),
        rng.normal(scale=3.0, size=(n, 3)),
    )


def test_exact_free_fall_fires_free_fall_guard():
    """a = (0, 0, −g) makes ‖a + g·e₃‖ vanish; p̂ divides by it."""
    p, v, a, j = _benign(9)
    a[4] = [0.0, 0.0, -G]
    out = flat_outputs(p, v, a, j, params=PARAMS)

    assert out.guard_free_fall[4]
    assert not out.valid[4]
    assert out.guard_free_fall.sum() == 1
    assert out.guard_counts()["free_fall"] == 1
    _assert_blanked(out, np.array([4]))
    # No warnings-driven infinities leaked into the arrays.
    assert not np.isinf(out.p_quad).any()


def test_near_free_fall_below_threshold_also_guarded():
    """The guard is a threshold, not an equality test: 1e-6 m/s² of residual is still singular."""
    p, v, a, j = _benign(6)
    a[2] = [0.0, 0.0, -G + 1e-6]
    out = flat_outputs(p, v, a, j, params=PARAMS)
    assert out.guard_free_fall[2]
    assert not out.valid[2]


def test_free_fall_threshold_is_configurable_and_not_over_eager():
    """A margin of 1e-3 (i.e. ‖a_g‖ ≈ 0.01) is numerically fine and must not be flagged."""
    p, v, a, j = _benign(5)
    a[1] = [0.0, 0.0, -G + 0.01]
    out = flat_outputs(p, v, a, j, params=PARAMS)
    assert not out.guard_free_fall.any()
    assert out.valid.all()

    strict = flat_outputs(p, v, a, j, params=PARAMS, guards=GuardConfig(min_acc_norm=0.1))
    assert strict.guard_free_fall[1]


def test_degenerate_cross_product_fires_when_b3_parallel_to_b2d():
    """b₁ ∝ b₂d × b₃ is undefined when b₃ ∥ b₂d = (−sin ψ, cos ψ, 0) = (0, 1, 0) at ψ = 0.

    Purely lateral +y acceleration cancelling gravity puts the whole force along +y.
    """
    p, v, a, j = _benign(7)
    a[3] = [0.0, 5.0, -G]
    j[3] = [0.0, 0.0, 0.0]  # zero jerk keeps p̈̂ (and hence the mQ·L·p̈̂ term) exactly zero
    out = flat_outputs(p, v, a, j, params=PARAMS)

    assert out.guard_degenerate_cross[3]
    assert not out.valid[3]
    assert not out.guard_free_fall[3]  # ‖a_g‖ = 5, so this is a genuinely different failure mode
    _assert_blanked(out, np.array([3]))


def test_degenerate_cross_product_fires_for_antiparallel_b3():
    p, v, a, j = _benign(4)
    a[1] = [0.0, -5.0, -G]
    j[1] = [0.0, 0.0, 0.0]
    out = flat_outputs(p, v, a, j, params=PARAMS)
    assert out.guard_degenerate_cross[1]
    assert not out.valid[1]


def test_degeneracy_follows_the_yaw_convention():
    """With yaw = π/2, b₂d = (−1, 0, 0), so the degenerate direction rotates to ±x."""
    p, v, a, j = _benign(3)
    a[1] = [5.0, 0.0, -G]
    j[1] = [0.0, 0.0, 0.0]

    at_zero_yaw = flat_outputs(p, v, a, j, params=PARAMS, yaw=0.0)
    assert not at_zero_yaw.guard_degenerate_cross.any()

    at_half_pi = flat_outputs(p, v, a, j, params=PARAMS, yaw=np.pi / 2)
    assert at_half_pi.guard_degenerate_cross[1]


def test_nonfinite_input_is_recorded_not_propagated():
    p, v, a, j = _benign(6)
    j[2, 1] = np.nan
    a[5, 0] = np.inf
    out = flat_outputs(p, v, a, j, params=PARAMS)

    assert out.guard_nonfinite_input[2] and out.guard_nonfinite_input[5]
    assert not out.valid[2] and not out.valid[5]
    _assert_blanked(out, np.array([2, 5]))


def test_all_guard_flags_are_reported_together():
    """One trajectory containing every failure mode: each is attributed to the right guard."""
    p, v, a, j = _benign(12)
    a[2] = [0.0, 0.0, -G]  # free fall
    a[5], j[5] = [0.0, 6.0, -G], [0.0, 0.0, 0.0]  # b3 ∥ b2d
    j[8, 0] = np.nan  # non-finite input
    out = flat_outputs(p, v, a, j, params=PARAMS)

    counts = out.guard_counts()
    assert counts == {
        "free_fall": 1,
        "force": 0,
        "degenerate_cross": 1,
        "nonfinite_input": 1,
        "invalid_total": 3,
    }
    np.testing.assert_array_equal(np.flatnonzero(~out.valid), [2, 5, 8])


def test_guarded_rows_do_not_corrupt_quaternion_continuity_of_the_rest():
    """A NaN row breaks the continuity chain; the segments on either side stay continuous."""
    rng = np.random.default_rng(11)
    n = 400
    t = np.linspace(0.0, 4.0, n)
    p = np.stack([np.sin(t), np.cos(t), 0.2 * t], axis=1)
    v = np.stack([np.cos(t), -np.sin(t), 0.2 * np.ones(n)], axis=1)
    a = np.stack([-np.sin(t), -np.cos(t), np.zeros(n)], axis=1)
    j = np.stack([-np.cos(t), np.sin(t), np.zeros(n)], axis=1)
    a[200] = [0.0, 0.0, -G]

    out = flat_outputs(p, v, a, j, params=PARAMS)
    assert not out.valid[200]
    for segment in (slice(0, 200), slice(201, n)):
        quat = out.quat_quad[segment]
        dots = np.sum(quat[1:] * quat[:-1], axis=1)
        assert np.all(dots >= 0.0)
        assert np.all(np.isfinite(dots))
    _ = rng  # keep the generator import honest without seeding unused randomness


def test_z_flat_is_never_blanked():
    """The flat coordinates are inputs, not derived quantities; masking them would lose data."""
    p, v, a, j = _benign(5)
    a[1] = [0.0, 0.0, -G]
    out = flat_outputs(p, v, a, j, params=PARAMS)
    assert not out.valid[1]
    assert np.all(np.isfinite(out.z_flat))
    np.testing.assert_allclose(out.z_flat, np.concatenate([p, v, a, j], axis=1))
    # Tension is well defined even in free fall — it is exactly zero there, not undefined.
    assert np.isfinite(out.tension_margin).all()
    assert out.tension_margin[1] == pytest.approx(0.0, abs=1e-12)


def test_shape_validation():
    p, v, a, j = _benign(5)
    with pytest.raises(ValueError):
        flat_outputs(p, v, a, j[:3], params=PARAMS)
    with pytest.raises(ValueError):
        flat_outputs(p[:, :2], v, a, j, params=PARAMS)
