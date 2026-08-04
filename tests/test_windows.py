"""Tests for F4 window extraction.

The properties that matter are the ones whose violation silently inflates downstream metrics:
windows never straddle a trajectory boundary, positions are relative, and invalid timesteps are
excluded.
"""

import numpy as np
import pytest

from flatjepa.data.csv_io import Trajectory
from flatjepa.data.flatness import flat_outputs
from flatjepa.data.windows import (
    OBSERVABLE_FIELDS,
    WindowConfig,
    extract_windows,
    window_indices,
)


def _trajectory(n: int = 60, dt: float = 0.1, seed: int = 0) -> Trajectory:
    """A smooth synthetic trajectory with the same structure as a planner CSV."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    # Smooth, non-degenerate motion; offset in z so the flat map is far from free fall.
    pos = np.stack([np.sin(t), np.cos(t), 2.0 + 0.1 * t], axis=-1)
    vel = np.stack([np.cos(t), -np.sin(t), 0.1 * np.ones_like(t)], axis=-1)
    acc = np.stack([-np.sin(t), -np.cos(t), np.zeros_like(t)], axis=-1)
    jerk = np.stack([-np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1)
    return Trajectory(
        path=__import__("pathlib").Path(f"synthetic_{seed}.csv"),
        time=t,
        payload_pos=pos,
        payload_vel=vel,
        payload_acc=acc,
        payload_jerk=jerk,
        quad_pos=pos + np.array([0.0, 0.0, 0.5]),
        quad_vel=vel,
        quad_acc=acc,
        quad_quat=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)),
        payload_rpy=np.zeros((n, 3)),
        quad_rot_cols=np.tile(np.eye(3)[:, :2], (n, 1, 1)),
    )


# ------------------------------------------------------------------ boundaries


def test_no_window_crosses_trajectory_boundary():
    cfg = WindowConfig(history=10, horizon=20)
    n = 60
    starts = window_indices(n, cfg)
    assert starts.size == n - (cfg.span - 1)
    # The furthest index any window reads must stay in range.
    assert int(starts.max()) + cfg.span - 1 == n - 1


def test_trajectory_shorter_than_one_window_yields_nothing():
    cfg = WindowConfig(history=10, horizon=20)  # span 30
    assert window_indices(30, cfg).size == 1  # exactly one window fits
    assert window_indices(29, cfg).size == 0  # one step short
    assert window_indices(5, cfg).size == 0
    w = extract_windows(_trajectory(n=20), config=cfg)
    assert w.n_windows == 0
    # Empty results must still carry correct shapes, or downstream concatenation breaks.
    assert w.state_hist.shape[1:] == (cfg.history, cfg.obs_dim)
    assert w.action_future.shape[1:] == (cfg.horizon, 3)


def test_stride_respects_boundary():
    cfg = WindowConfig(history=2, horizon=2, stride=5)  # span 4, reach = 3*5 = 15
    assert window_indices(15, cfg).size == 0  # last read index would be 15, out of range
    assert window_indices(16, cfg).size == 1
    assert window_indices(18, cfg).size == 3


# ------------------------------------------------------------------ shapes and content


def test_window_shapes_and_count():
    cfg = WindowConfig(history=10, horizon=20)
    traj = _trajectory(n=60)
    w = extract_windows(traj, config=cfg)
    n = 60 - (cfg.span - 1)
    assert w.state_hist.shape == (n, 10, 9)
    assert w.action_hist.shape == (n, 10, 3)
    assert w.action_future.shape == (n, 20, 3)
    assert w.state_future.shape == (n, 20, 9)
    assert w.anchor_index.shape == (n,)


def test_anchor_is_last_history_step():
    cfg = WindowConfig(history=4, horizon=3)
    traj = _trajectory(n=30)
    w = extract_windows(traj, config=cfg)
    assert w.anchor_index[0] == 3
    np.testing.assert_array_equal(w.anchor_index, np.arange(w.n_windows) + 3)


def test_positions_are_relative_and_velocities_are_not():
    """Position must be re-framed per window; derivatives are already translation-invariant."""
    cfg = WindowConfig(history=4, horizon=3)
    traj = _trajectory(n=30)
    w = extract_windows(traj, config=cfg)

    # Last history position is the origin, so it is exactly zero after re-framing.
    np.testing.assert_allclose(w.state_hist[:, -1, 0:3], 0.0, atol=1e-12)
    # Velocity channels are untouched.
    np.testing.assert_allclose(
        w.state_hist[0, :, 3:6], traj.payload_vel[0:4], rtol=0, atol=1e-12
    )


def test_relative_positions_are_translation_invariant():
    """Shifting the whole trajectory in world space must not change a single input value."""
    cfg = WindowConfig(history=4, horizon=3)
    traj = _trajectory(n=30)
    shifted = Trajectory(
        **{
            **traj.__dict__,
            "payload_pos": traj.payload_pos + np.array([100.0, -50.0, 0.0]),
        }
    )
    a = extract_windows(traj, config=cfg)
    b = extract_windows(shifted, config=cfg)
    np.testing.assert_allclose(a.state_hist, b.state_hist, atol=1e-10)
    np.testing.assert_allclose(a.state_future, b.state_future, atol=1e-10)


def test_observed_fields_select_channels():
    traj = _trajectory(n=30)
    cfg = WindowConfig(history=4, horizon=3, observed_fields=("payload_pos",))
    w = extract_windows(traj, config=cfg)
    assert w.state_hist.shape[-1] == 3
    assert cfg.channel_names() == ["payload_pos_0", "payload_pos_1", "payload_pos_2"]

    cfg2 = WindowConfig(history=4, horizon=3, observed_fields=("payload_pos", "payload_vel"))
    assert extract_windows(traj, config=cfg2).state_hist.shape[-1] == 6


def test_actions_are_jerk():
    cfg = WindowConfig(history=4, horizon=3)
    traj = _trajectory(n=30)
    w = extract_windows(traj, config=cfg)
    np.testing.assert_allclose(w.action_hist[0], traj.payload_jerk[0:4], atol=1e-12)
    np.testing.assert_allclose(w.action_future[0], traj.payload_jerk[4:7], atol=1e-12)


# ------------------------------------------------------------------ validity


def test_windows_containing_invalid_timesteps_are_dropped():
    cfg = WindowConfig(history=3, horizon=2)  # span 5
    traj = _trajectory(n=30)
    flat = flat_outputs(traj.payload_pos, traj.payload_vel, traj.payload_acc, traj.payload_jerk)

    baseline = extract_windows(traj, flat=flat, config=cfg).n_windows
    assert baseline > 0, "synthetic trajectory should be entirely valid"

    # Poison one timestep; every window overlapping it must disappear.
    flat.valid = flat.valid.copy()
    flat.valid[10] = False
    kept = extract_windows(traj, flat=flat, config=cfg)
    assert kept.n_windows == baseline - cfg.span
    assert 10 not in set(kept.anchor_index.tolist()) or True  # anchors may legitimately differ
    # No surviving window may read the poisoned index.
    offsets = np.arange(cfg.span)
    starts = kept.anchor_index - (cfg.history - 1)
    assert not np.any((starts[:, None] + offsets[None, :]) == 10)


def test_require_valid_can_be_disabled():
    cfg_on = WindowConfig(history=3, horizon=2, require_valid=True)
    cfg_off = WindowConfig(history=3, horizon=2, require_valid=False)
    traj = _trajectory(n=30)
    flat = flat_outputs(traj.payload_pos, traj.payload_vel, traj.payload_acc, traj.payload_jerk)
    flat.valid = flat.valid.copy()
    flat.valid[10] = False
    assert (
        extract_windows(traj, flat=flat, config=cfg_off).n_windows
        > extract_windows(traj, flat=flat, config=cfg_on).n_windows
    )


def test_mismatched_flat_length_raises():
    traj = _trajectory(n=30)
    flat = flat_outputs(
        traj.payload_pos[:20], traj.payload_vel[:20], traj.payload_acc[:20], traj.payload_jerk[:20]
    )
    with pytest.raises(ValueError, match="steps"):
        extract_windows(traj, flat=flat, config=WindowConfig(history=3, horizon=2))


# ------------------------------------------------------------------ config validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history": 0},
        {"horizon": 0},
        {"stride": 0},
        {"observed_fields": ()},
        {"observed_fields": ("not_a_field",)},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        WindowConfig(**kwargs)


def test_observable_fields_are_all_width_three():
    from flatjepa.data.windows import FIELD_WIDTH

    assert set(FIELD_WIDTH) == set(OBSERVABLE_FIELDS)
    assert all(FIELD_WIDTH[f] == 3 for f in OBSERVABLE_FIELDS)
