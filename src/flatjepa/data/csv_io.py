"""Loading of PolyFly planner trajectory CSVs.

The upstream planner writes one CSV per solved trajectory with a fixed 35-column layout:

| Columns | Meaning |
|---------|---------|
| ``time`` | seconds, uniform sampling (500 Hz in the generated corpus) |
| ``sol_x_0..8`` | payload position (0-2), velocity (3-5), acceleration (6-8) |
| ``sol_u_0..2`` | payload **jerk** — the control input of the triple-integrator model |
| ``sol_quad_x_0..8`` | quadrotor position / velocity / acceleration |
| ``sol_quad_quat_0..3`` | quadrotor attitude quaternion, ``(x, y, z, w)`` |
| ``sol_payload_rpy_0..2`` | payload orientation relative to the cable |
| ``rot_mat_0..5`` | first two *columns* of the quadrotor rotation matrix, row-major flat |

The quaternion convention is ``(x, y, z, w)``: upstream produces these columns with
``scipy.spatial.transform.Rotation.as_quat()`` and consumes them with ``Rotation.from_quat()``.
This is verified rather than assumed — see ``tests/test_flatness_agreement.py``.

``rot_mat_*`` is ``R[:, :2]`` flattened, i.e. ``rot_mat.reshape(T, 3, 2)`` recovers the first two
columns of the (3, 3) rotation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["Trajectory", "load_trajectory_csv", "find_trajectory_csvs", "EXPECTED_COLUMNS"]


def _cols(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i}" for i in range(n)]


EXPECTED_COLUMNS: tuple[str, ...] = (
    "time",
    *_cols("sol_x", 9),
    *_cols("sol_u", 3),
    *_cols("sol_quad_x", 9),
    *_cols("sol_quad_quat", 4),
    *_cols("sol_payload_rpy", 3),
    *_cols("rot_mat", 6),
)


@dataclass(frozen=True)
class Trajectory:
    """One planner trajectory, columns already split into named arrays.

    All arrays share the leading time axis ``T`` and are index-aligned with each other and with
    everything F2/F3 derive from them.
    """

    path: Path
    time: np.ndarray  # (T,)
    payload_pos: np.ndarray  # (T, 3)
    payload_vel: np.ndarray  # (T, 3)
    payload_acc: np.ndarray  # (T, 3)
    payload_jerk: np.ndarray  # (T, 3)  -- sol_u, the planner's control input
    quad_pos: np.ndarray  # (T, 3)
    quad_vel: np.ndarray  # (T, 3)
    quad_acc: np.ndarray  # (T, 3)
    quad_quat: np.ndarray  # (T, 4)  -- (x, y, z, w)
    payload_rpy: np.ndarray  # (T, 3)
    quad_rot_cols: np.ndarray  # (T, 3, 2) -- first two columns of R_quad

    @property
    def n_steps(self) -> int:
        return int(self.time.shape[0])

    @property
    def dt(self) -> float:
        """Median sample spacing. The corpus is uniformly sampled, but do not assume exactly so."""
        if self.time.shape[0] < 2:
            return float("nan")
        return float(np.median(np.diff(self.time)))

    @property
    def name(self) -> str:
        return self.path.stem


def load_trajectory_csv(path: str | Path) -> Trajectory:
    """Read one planner CSV into a :class:`Trajectory`.

    Raises ``ValueError`` if the expected columns are missing, rather than silently producing
    misaligned arrays.
    """
    path = Path(path)
    frame = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in frame.columns]
    if missing:
        shown = f"{missing[:6]}{'...' if len(missing) > 6 else ''}"
        raise ValueError(f"{path}: missing expected columns {shown}")

    def block(names: Sequence[str]) -> np.ndarray:
        return np.ascontiguousarray(frame.loc[:, list(names)].to_numpy(dtype=np.float64))

    sol_x = block(_cols("sol_x", 9))
    quad_x = block(_cols("sol_quad_x", 9))
    rot_mat = block(_cols("rot_mat", 6))

    return Trajectory(
        path=path,
        time=frame["time"].to_numpy(dtype=np.float64),
        payload_pos=sol_x[:, 0:3],
        payload_vel=sol_x[:, 3:6],
        payload_acc=sol_x[:, 6:9],
        payload_jerk=block(_cols("sol_u", 3)),
        quad_pos=quad_x[:, 0:3],
        quad_vel=quad_x[:, 3:6],
        quad_acc=quad_x[:, 6:9],
        quad_quat=block(_cols("sol_quad_quat", 4)),
        payload_rpy=block(_cols("sol_payload_rpy", 3)),
        quad_rot_cols=rot_mat.reshape(-1, 3, 2),
    )


def find_trajectory_csvs(roots: Iterable[str | Path], pattern: str = "*.csv") -> list[Path]:
    """Collect trajectory CSVs under one or more directories (recursively), sorted and de-duped."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        root = Path(root)
        candidates = [root] if root.is_file() else sorted(root.rglob(pattern))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(candidate)
    return found
