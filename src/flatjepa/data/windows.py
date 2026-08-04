"""Window extraction for F4.

Turns a :class:`~flatjepa.data.csv_io.Trajectory` plus its F2/F3 derived quantities into fixed-size
(history, future) windows.

Three properties of this corpus shape everything here:

* **The data is 10 Hz and must not be resampled.** The forest generator writes at planner knots
  (``save_result(..., dt=0.1)``), not through the 500 Hz interpolation used for the maze corpus.
  F4 §1 originally specified striding down to 20 Hz; that target is unreachable and was dropped.
  ``stride`` exists only for the 500 Hz maze corpus.
* **Positions must be relative.** Absolute world position is an artifact of environment layout and
  would let the model identify environments rather than learn dynamics.
* **The observed state is configurable.** If the model is fed ``(p, v, a)`` and probed for
  ``(p, v, a)``, E1 is tautological. See :mod:`flatjepa.data.targets` and F7 §1b.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from flatjepa.data.csv_io import Trajectory
from flatjepa.data.flatness import FlatOutputs

# Fields that may be used as model input. Order here defines channel order in the output.
OBSERVABLE_FIELDS: tuple[str, ...] = ("payload_pos", "payload_vel", "payload_acc")

# Which observed fields are position-like, and therefore need the relative-position treatment.
_POSITION_LIKE: frozenset[str] = frozenset({"payload_pos"})

FIELD_WIDTH: dict[str, int] = {
    "payload_pos": 3,
    "payload_vel": 3,
    "payload_acc": 3,
}


@dataclass(frozen=True)
class WindowConfig:
    """How to cut a trajectory into windows.

    ``observed_fields`` is the E1 lever. The default is the full planner state, which is the
    natural modelling choice but makes a probe for ``(p, v, a)`` trivial. Restricting it does *not*
    by itself fix the tautology -- every time-derivative is a linear functional of a history window,
    so ``v`` and ``a`` stay linearly recoverable from a position-only window. The real fix is
    probing for nonlinear targets; this knob exists for the ablation, not the fix.
    """

    history: int = 10
    horizon: int = 20
    stride: int = 1
    observed_fields: tuple[str, ...] = OBSERVABLE_FIELDS
    require_valid: bool = True

    def __post_init__(self) -> None:
        if self.history < 1:
            raise ValueError(f"history must be >= 1, got {self.history}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        unknown = set(self.observed_fields) - set(OBSERVABLE_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown observed_fields {sorted(unknown)}; known: {list(OBSERVABLE_FIELDS)}"
            )
        if not self.observed_fields:
            raise ValueError("observed_fields must not be empty")

    @property
    def span(self) -> int:
        """Timesteps consumed by one window, before striding."""
        return self.history + self.horizon

    @property
    def obs_dim(self) -> int:
        return sum(FIELD_WIDTH[f] for f in self.observed_fields)

    def channel_names(self) -> list[str]:
        names: list[str] = []
        for f in self.observed_fields:
            names.extend(f"{f}_{i}" for i in range(FIELD_WIDTH[f]))
        return names


@dataclass
class Windows:
    """Windows cut from a single trajectory.

    ``N`` is the number of windows. Arrays are index-aligned; ``anchor_index[i]`` is the index into
    the source trajectory of window ``i``'s final history step, which is the timestep the latent is
    understood to represent.
    """

    state_hist: np.ndarray  # (N, H, D_obs)
    action_hist: np.ndarray  # (N, H, 3)
    action_future: np.ndarray  # (N, F, 3)
    state_future: np.ndarray  # (N, F, D_obs)
    anchor_index: np.ndarray  # (N,) int
    anchor_origin: np.ndarray  # (N, 3) the world position subtracted off, kept for reversibility

    @property
    def n_windows(self) -> int:
        return int(self.state_hist.shape[0])


def _stack_observed(
    traj: Trajectory, fields: Sequence[str], idx: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Gather ``fields`` at ``idx`` into (N, K, D), subtracting ``origin`` from position-like ones.

    ``idx`` has shape (N, K); ``origin`` has shape (N, 3) and broadcasts over K.
    """
    parts = []
    for f in fields:
        arr = getattr(traj, f)[idx]  # (N, K, 3)
        if f in _POSITION_LIKE:
            arr = arr - origin[:, None, :]
        parts.append(arr)
    return np.concatenate(parts, axis=-1)


def window_indices(n_steps: int, config: WindowConfig) -> np.ndarray:
    """Start indices of every window that fits entirely inside a trajectory of ``n_steps``.

    Windows never straddle a trajectory boundary: a window starting at ``s`` reads indices
    ``s, s+stride, ..., s+(span-1)*stride``, and the last of those must be in range.
    """
    reach = (config.span - 1) * config.stride
    if n_steps <= reach:
        return np.empty(0, dtype=np.int64)
    return np.arange(n_steps - reach, dtype=np.int64)


def extract_windows(
    traj: Trajectory,
    flat: FlatOutputs | None = None,
    config: WindowConfig | None = None,
) -> Windows:
    """Cut ``traj`` into windows.

    When ``flat`` is supplied and ``config.require_valid`` is set, any window containing a timestep
    that failed an F2 guard is dropped -- a window is only as trustworthy as its worst step.
    """
    config = config or WindowConfig()
    if flat is not None and flat.n_steps != traj.n_steps:
        raise ValueError(
            f"flat outputs have {flat.n_steps} steps but trajectory has {traj.n_steps}"
        )

    starts = window_indices(traj.n_steps, config)
    if starts.size == 0:
        return _empty_windows(config)

    # (N, span) absolute indices into the trajectory.
    offsets = np.arange(config.span, dtype=np.int64) * config.stride
    idx = starts[:, None] + offsets[None, :]

    if flat is not None and config.require_valid:
        keep = flat.valid[idx].all(axis=1)
        idx = idx[keep]
        starts = starts[keep]
        if idx.shape[0] == 0:
            return _empty_windows(config)

    hist_idx = idx[:, : config.history]
    fut_idx = idx[:, config.history :]
    anchor_index = hist_idx[:, -1]

    # Positions are expressed relative to the final history frame (F4 §3).
    origin = traj.payload_pos[anchor_index]

    return Windows(
        state_hist=_stack_observed(traj, config.observed_fields, hist_idx, origin),
        action_hist=traj.payload_jerk[hist_idx],
        action_future=traj.payload_jerk[fut_idx],
        state_future=_stack_observed(traj, config.observed_fields, fut_idx, origin),
        anchor_index=anchor_index,
        anchor_origin=origin,
    )


def _empty_windows(config: WindowConfig) -> Windows:
    d = config.obs_dim
    return Windows(
        state_hist=np.empty((0, config.history, d)),
        action_hist=np.empty((0, config.history, 3)),
        action_future=np.empty((0, config.horizon, 3)),
        state_future=np.empty((0, config.horizon, d)),
        anchor_index=np.empty(0, dtype=np.int64),
        anchor_origin=np.empty((0, 3)),
    )
