"""F3 — cable tension and taut/slack labeling.

For a payload of mass ``mL`` on a taut cable with unit vector ``q̂`` pointing payload → quadrotor,
Newton's second law gives

    mL·a = T·q̂ − mL·g·e₃      ⟹      T·q̂ = mL·(a + g·e₃)

The flatness construction *defines* ``q̂`` parallel to ``a + g·e₃``, so the tension magnitude
collapses to

    T = mL · ‖a + g·e₃‖

which is computable directly from the payload acceleration columns of the CSV — no integration and
no simulation. The normalized quantity actually used everywhere downstream is the margin

    margin = T / (mL·g) = ‖a + g·e₃‖ / g

so that ``margin = 1`` is hover and ``margin = 0`` is free fall. In the idealized model tension
bottoms out at zero and never goes negative, so the meaningful signal is proximity to the boundary,
not a sign.

The threshold ``τ`` separating taut from near-slack is a *reported hyperparameter*, not a hidden
constant: :func:`threshold_sweep` exists so that every reported number carries its base rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "DEFAULT_THRESHOLDS",
    "TensionLabels",
    "cable_tension",
    "tension_margin",
    "label_taut_slack",
    "labels_from_acceleration",
    "threshold_sweep",
    "margin_summary",
]

#: Thresholds swept by default when reporting near-slack base rates. Spans three decades because
#: it is not known a priori which decade the generated corpus actually reaches.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01)

# Regime codes, as specified in F3 §3.
REGIME_TAUT = 0
REGIME_NEAR_SLACK = 1


@dataclass(frozen=True)
class TensionLabels:
    """Per-timestep tension quantities and regime labels, index-aligned with F2 outputs."""

    tension: np.ndarray  # (T,) newtons
    margin: np.ndarray  # (T,) tension / (mL·g); 1 = hover, 0 = free fall
    near_slack: np.ndarray  # (T,) bool
    regime: np.ndarray  # (T,) int; 0 = taut, 1 = near-slack
    threshold: float

    @property
    def base_rate(self) -> float:
        """Fraction of timesteps labeled near-slack. Report this beside every probe number."""
        if self.near_slack.size == 0:
            return float("nan")
        return float(np.mean(self.near_slack))


def cable_tension(acc: np.ndarray, mass_load: float, gravity: float = 9.81) -> np.ndarray:
    """Cable tension in newtons: ``mL·‖a + g·e₃‖``.

    ``acc`` is payload acceleration with shape ``(..., 3)``; the result has shape
    ``acc.shape[:-1]``.
    """
    acc = np.asarray(acc, dtype=np.float64)
    if acc.shape[-1] != 3:
        raise ValueError(f"acc must have trailing dimension 3, got shape {acc.shape}")
    acc_g = acc.copy()
    acc_g[..., 2] += gravity
    return mass_load * np.linalg.norm(acc_g, axis=-1)


def tension_margin(acc: np.ndarray, gravity: float = 9.81) -> np.ndarray:
    """Normalized tension margin ``‖a + g·e₃‖ / g``.

    Mass-independent by construction: the payload mass cancels between ``T`` and ``mL·g``.
    """
    return cable_tension(acc, mass_load=1.0, gravity=gravity) / gravity


def label_taut_slack(
    margin: np.ndarray,
    threshold: float,
    tension: np.ndarray | None = None,
) -> TensionLabels:
    """Label timesteps as taut (0) or near-slack (1) given a normalized margin and threshold.

    ``tension`` (newtons) is carried through when supplied; otherwise it is reconstructed for unit
    payload mass, which keeps the margin — the quantity every threshold is expressed in — exact.
    """
    margin = np.asarray(margin, dtype=np.float64)
    near = margin < threshold
    return TensionLabels(
        tension=margin * 9.81 if tension is None else np.asarray(tension, dtype=np.float64),
        margin=margin,
        near_slack=near,
        regime=np.where(near, REGIME_NEAR_SLACK, REGIME_TAUT).astype(np.int8),
        threshold=float(threshold),
    )


def labels_from_acceleration(
    acc: np.ndarray,
    threshold: float,
    mass_load: float,
    gravity: float = 9.81,
) -> TensionLabels:
    """Convenience path from raw payload acceleration straight to labels, in physical newtons."""
    tension = cable_tension(acc, mass_load=mass_load, gravity=gravity)
    margin = tension / (mass_load * gravity)
    near = margin < threshold
    return TensionLabels(
        tension=tension,
        margin=margin,
        near_slack=near,
        regime=np.where(near, REGIME_NEAR_SLACK, REGIME_TAUT).astype(np.int8),
        threshold=float(threshold),
    )


def threshold_sweep(
    margin: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, float]]:
    """Base rate of ``near_slack`` at each threshold.

    F3 §4 requires the base rate to accompany every reported threshold, because near-slack is
    expected to be rare enough that accuracy alone is meaningless.
    """
    margin = np.asarray(margin, dtype=np.float64)
    finite = margin[np.isfinite(margin)]
    n = finite.size
    rows: list[dict[str, float]] = []
    for tau in thresholds:
        count = int(np.count_nonzero(finite < tau))
        rows.append(
            {
                "threshold": float(tau),
                "count": float(count),
                "base_rate": float(count / n) if n else float("nan"),
            }
        )
    return rows


def margin_summary(
    margin: np.ndarray,
    percentiles: Sequence[float] = (0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 95.0, 100.0),
) -> dict[str, float]:
    """Distribution summary of the tension margin over a corpus.

    This is F3's *first* deliverable: it decides whether E4 has any signal to find at all.
    """
    margin = np.asarray(margin, dtype=np.float64)
    finite = margin[np.isfinite(margin)]
    if finite.size == 0:
        raise ValueError("no finite tension-margin samples")
    out: dict[str, float] = {
        "n": float(finite.size),
        "n_nonfinite": float(margin.size - finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }
    for q, value in zip(percentiles, np.percentile(finite, percentiles)):
        out[f"p{q:g}"] = float(value)
    return out
