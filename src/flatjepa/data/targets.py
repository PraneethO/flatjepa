"""Probe targets for F7, split by whether they are linearly trivial.

This module encodes the project's central methodological guard (F7 §1b).

With a history window as model input, **every time-derivative is a linear functional of that
window**: velocity is a finite difference of positions, acceleration a second difference, jerk a
third. A linear probe fit on the raw input window therefore recovers ``p``, ``v``, ``a`` and ``j``
to near-machine precision with no model involved. Reporting those as "the latent recovered the flat
outputs" would be vacuous.

The targets worth probing are the ones the flat map computes *nonlinearly* -- normalisations, cross
products, the SO(3) construction. Those are not linear functionals of the window, so recovering them
is a real claim about the representation.

Targets here are tagged :data:`LINEAR_TRIVIAL` or :data:`NONLINEAR`. F7's audit still measures
linear decodability empirically rather than trusting these tags -- the tags say what we expect, the
audit says what is true, and a disagreement is itself a finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from flatjepa.data.flatness import FlatOutputs

LINEAR_TRIVIAL = "linear_trivial"
NONLINEAR = "nonlinear"


@dataclass(frozen=True)
class TargetSpec:
    """One probe target: where it lives in the packed array and how to interpret it."""

    name: str
    width: int
    kind: str
    description: str


# Order defines the packing order in `pack_targets`.
#
# Payload position ``p`` is deliberately absent. Targets are packed at the window anchor, and
# positions are expressed relative to that anchor (F4 §3), so ``p`` at the anchor is *identically
# zero* -- a degenerate all-zero regression target that reports R² of either 0 or 1 depending on
# how a divide-by-zero is handled. This was caught by the audit in `scripts/linear_audit.py` on the
# first real build. Probing position at *future* indices is a separate, non-degenerate question and
# would need its own target packed at those indices.
TARGET_SPECS: tuple[TargetSpec, ...] = (
    # --- linearly trivial: sanity checks, never headline results ---
    TargetSpec("v", 3, LINEAR_TRIVIAL, "payload velocity"),
    TargetSpec("a", 3, LINEAR_TRIVIAL, "payload acceleration"),
    TargetSpec("j", 3, LINEAR_TRIVIAL, "payload jerk, the control input"),
    # --- nonlinear consequences of the flat map: the real E1 targets ---
    TargetSpec("cable_dir", 3, NONLINEAR, "unit cable vector; requires a normalisation"),
    # Note one of these six channels (b1_z) is structurally constant under the flat map's
    # zero-yaw convention. F7's audit scores per channel and excludes zero-variance channels, so
    # this is handled rather than hidden, but do not read the six as six independent quantities.
    TargetSpec("R_quad_cols", 6, NONLINEAR, "first two columns of R_quad; cross products, SO(3)"),
    TargetSpec("p_quad", 3, NONLINEAR, "quadrotor position (relative); nonlinear via b3"),
    TargetSpec("tension_margin", 1, NONLINEAR, "‖a + g·e₃‖ / g; requires a norm"),
)

TARGET_NAMES: tuple[str, ...] = tuple(s.name for s in TARGET_SPECS)
TOTAL_TARGET_DIM: int = sum(s.width for s in TARGET_SPECS)

# Position-like targets, which must be expressed relative to the window anchor for the same reason
# the inputs are (F4 §3): absolute world position identifies the environment.
_POSITION_LIKE: frozenset[str] = frozenset({"p_quad"})


def target_slices() -> dict[str, slice]:
    """Map target name -> slice into the packed target array."""
    out: dict[str, slice] = {}
    off = 0
    for spec in TARGET_SPECS:
        out[spec.name] = slice(off, off + spec.width)
        off += spec.width
    return out


def targets_of_kind(kind: str) -> tuple[str, ...]:
    return tuple(s.name for s in TARGET_SPECS if s.kind == kind)


def pack_targets(
    flat: "FlatOutputs", index: np.ndarray, origin: np.ndarray | None = None
) -> np.ndarray:
    """Pack every probe target at ``index`` into one (N, TOTAL_TARGET_DIM) array.

    ``origin`` (N, 3) is subtracted from position-like targets, and must be the same anchor origin
    used for the model inputs or probes will be asked to predict a differently-framed quantity than
    the one the latent saw.
    """
    index = np.asarray(index, dtype=np.int64)
    n = index.shape[0]
    if origin is None:
        origin = np.zeros((n, 3))
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (n, 3):
        raise ValueError(f"origin must have shape {(n, 3)}, got {origin.shape}")

    raw: dict[str, np.ndarray] = {
        "v": flat.v[index],
        "a": flat.a[index],
        "j": flat.j[index],
        "cable_dir": flat.cable_dir[index],
        # R_quad is (T, 3, 3) with columns (b1, b2, b3); the first two columns determine the third
        # by cross product, so storing six numbers is lossless and avoids an over-parameterised
        # regression target.
        "R_quad_cols": flat.R_quad[index][:, :, :2].reshape(n, 6),
        "p_quad": flat.p_quad[index] - origin,
        "tension_margin": flat.tension_margin[index].reshape(n, 1),
    }

    packed = np.concatenate([raw[s.name] for s in TARGET_SPECS], axis=-1)
    if packed.shape != (n, TOTAL_TARGET_DIM):
        raise AssertionError(f"packed targets have shape {packed.shape}, expected {(n, TOTAL_TARGET_DIM)}")
    return packed


def describe() -> str:
    """Human-readable table of targets, for run logs and the F7 audit report."""
    lines = [f"{'target':<16}{'dim':>4}  {'kind':<15}description"]
    for spec in TARGET_SPECS:
        lines.append(f"{spec.name:<16}{spec.width:>4}  {spec.kind:<15}{spec.description}")
    lines.append("")
    lines.append(
        "linear_trivial targets are expected to be recoverable from the raw input window and are "
        "sanity checks only; F7 §1b disqualifies them as headline results."
    )
    return "\n".join(lines)
