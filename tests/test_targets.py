"""Tests for F7 probe targets (F4/F7 §1b).

The important test here is not a shape check: it is the empirical claim that motivates the whole
target taxonomy -- that derivatives are linearly recoverable from a history window and the flat
map's nonlinear outputs are not.
"""

import numpy as np
import pytest

from flatjepa.data.flatness import flat_outputs
from flatjepa.data.targets import (
    LINEAR_TRIVIAL,
    NONLINEAR,
    TARGET_NAMES,
    TARGET_SPECS,
    TOTAL_TARGET_DIM,
    pack_targets,
    target_slices,
    targets_of_kind,
)


def _flat(n: int = 400, dt: float = 0.1):
    t = np.arange(n) * dt
    p = np.stack([np.sin(t), np.cos(0.7 * t), 2.0 + 0.3 * np.sin(0.4 * t)], axis=-1)
    v = np.stack([np.cos(t), -0.7 * np.sin(0.7 * t), 0.12 * np.cos(0.4 * t)], axis=-1)
    a = np.stack([-np.sin(t), -0.49 * np.cos(0.7 * t), -0.048 * np.sin(0.4 * t)], axis=-1)
    j = np.stack([-np.cos(t), 0.343 * np.sin(0.7 * t), -0.0192 * np.cos(0.4 * t)], axis=-1)
    return flat_outputs(p, v, a, j), p, v, a, j


def test_slices_partition_the_packed_array():
    slices = target_slices()
    assert set(slices) == set(TARGET_NAMES)
    covered = np.zeros(TOTAL_TARGET_DIM, dtype=int)
    for s in slices.values():
        covered[s] += 1
    assert (covered == 1).all(), "target slices must tile the packed array exactly once"


def test_pack_shape_and_roundtrip():
    flat, p, v, a, j = _flat(50)
    idx = np.arange(10, 40)
    packed = pack_targets(flat, idx)
    assert packed.shape == (30, TOTAL_TARGET_DIM)

    sl = target_slices()
    np.testing.assert_allclose(packed[:, sl["v"]], v[idx], atol=1e-12)
    np.testing.assert_allclose(packed[:, sl["a"]], a[idx], atol=1e-12)
    np.testing.assert_allclose(packed[:, sl["j"]], j[idx], atol=1e-12)
    np.testing.assert_allclose(packed[:, sl["cable_dir"]], flat.cable_dir[idx], atol=1e-12)


def test_position_targets_are_relative_to_origin():
    flat, p, *_ = _flat(50)
    idx = np.arange(10, 40)
    origin = p[idx]
    packed = pack_targets(flat, idx, origin=origin)
    sl = target_slices()
    np.testing.assert_allclose(
        packed[:, sl["p_quad"]], flat.p_quad[idx] - origin, atol=1e-12
    )


def test_payload_position_is_not_a_target():
    """`p` at the anchor is identically zero once positions are anchor-relative, so it is a
    degenerate regression target and must stay out of the set. Caught by the audit on the first
    real build."""
    assert "p" not in TARGET_NAMES


def test_no_target_is_structurally_constant_across_a_trajectory():
    """A zero-variance target channel makes R² undefined. One known exception is documented."""
    flat, p, *_ = _flat(400)
    idx = np.arange(5, 395)
    packed = pack_targets(flat, idx, origin=p[idx])
    sl = target_slices()
    constant = {
        name: [int(i) for i, s in enumerate(packed[:, sl[name]].std(axis=0)) if s < 1e-10]
        for name in TARGET_NAMES
    }
    constant = {k: v for k, v in constant.items() if v}
    # b1_z is pinned by the flat map's zero-yaw convention; nothing else may be constant.
    assert set(constant) <= {"R_quad_cols"}, f"unexpected constant target channels: {constant}"


def test_origin_shape_validated():
    flat, *_ = _flat(50)
    with pytest.raises(ValueError, match="origin"):
        pack_targets(flat, np.arange(10), origin=np.zeros((5, 3)))


def test_rotation_columns_are_orthonormal():
    """R_quad_cols packs b1,b2 -- they must stay a valid partial rotation, or the probe target is
    not the quantity it claims to be."""
    flat, *_ = _flat(50)
    packed = pack_targets(flat, np.arange(5, 45))
    cols = packed[:, target_slices()["R_quad_cols"]].reshape(-1, 3, 2)
    b1, b2 = cols[:, :, 0], cols[:, :, 1]
    np.testing.assert_allclose(np.linalg.norm(b1, axis=-1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(b2, axis=-1), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.einsum("ij,ij->i", b1, b2), 0.0, atol=1e-9)


def _linear_r2(X: np.ndarray, Y: np.ndarray) -> float:
    """R² of an ordinary least-squares fit from X to Y, with an intercept."""
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ coef
    ss_res = float(((Y - pred) ** 2).sum())
    ss_tot = float(((Y - Y.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def test_derivatives_are_linearly_recoverable_from_a_position_window():
    """The claim that kills the naive E1 design, verified rather than asserted.

    If this ever fails, the taxonomy in targets.py needs revisiting -- but it should not, since
    finite differences are linear by construction.
    """
    flat, p, v, a, j = _flat(400)
    H = 10
    idx = np.arange(H, 380)
    # Position-only history window, exactly what "restrict the observation" would feed the model.
    window = np.stack([p[i - H + 1 : i + 1].reshape(-1) for i in idx], axis=0)

    assert _linear_r2(window, v[idx]) > 0.999, "velocity should be linearly trivial"
    assert _linear_r2(window, a[idx]) > 0.999, "acceleration should be linearly trivial"


def test_nonlinear_targets_are_not_linearly_recoverable():
    """The other half of the claim: the flat map's nonlinear outputs resist a linear probe."""
    flat, p, v, a, j = _flat(400)
    H = 10
    idx = np.arange(H, 380)
    window = np.stack([p[i - H + 1 : i + 1].reshape(-1) for i in idx], axis=0)

    margin = flat.tension_margin[idx].reshape(-1, 1)
    r2_margin = _linear_r2(window, margin)
    assert r2_margin < 0.9, f"tension margin unexpectedly linear (R²={r2_margin:.3f})"


def test_kind_tags_are_exhaustive_and_disjoint():
    linear = set(targets_of_kind(LINEAR_TRIVIAL))
    nonlinear = set(targets_of_kind(NONLINEAR))
    assert linear and nonlinear
    assert not (linear & nonlinear)
    assert linear | nonlinear == set(TARGET_NAMES)
    assert all(s.kind in {LINEAR_TRIVIAL, NONLINEAR} for s in TARGET_SPECS)
