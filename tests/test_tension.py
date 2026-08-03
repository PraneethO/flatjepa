"""F3 §6 — tension and taut/slack labeling."""

from __future__ import annotations

import numpy as np
import pytest

from flatjepa.data.flatness import SystemParams, flat_outputs
from flatjepa.data.tension import (
    DEFAULT_THRESHOLDS,
    REGIME_NEAR_SLACK,
    REGIME_TAUT,
    cable_tension,
    label_taut_slack,
    labels_from_acceleration,
    margin_summary,
    tension_margin,
    threshold_sweep,
)

PARAMS = SystemParams()
G = PARAMS.gravity
ML = PARAMS.mass_load


def test_hover_gives_unit_margin():
    """Hover: a = 0, so T = mL·g and the margin is exactly 1."""
    acc = np.zeros((50, 3))
    np.testing.assert_allclose(cable_tension(acc, ML, G), ML * G, atol=1e-15, rtol=0)
    np.testing.assert_allclose(tension_margin(acc, G), 1.0, atol=1e-15, rtol=0)


def test_free_fall_gives_zero_margin():
    """Free fall: a = −g·e₃ cancels gravity exactly, so tension vanishes."""
    acc = np.tile([0.0, 0.0, -G], (50, 1))
    np.testing.assert_allclose(cable_tension(acc, ML, G), 0.0, atol=1e-14)
    np.testing.assert_allclose(tension_margin(acc, G), 0.0, atol=1e-15)


def test_near_free_fall_margin_is_small_but_positive():
    acc = np.tile([0.0, 0.0, -G + 0.05], (10, 1))
    margin = tension_margin(acc, G)
    assert np.all(margin > 0.0)
    np.testing.assert_allclose(margin, 0.05 / G, rtol=1e-12)


def test_tension_is_never_negative_for_arbitrary_acceleration():
    """The idealized model bottoms out at zero — the signal is a margin, not a sign (F3 §2)."""
    rng = np.random.default_rng(0)
    acc = rng.uniform(-10.0, 10.0, size=(20000, 3))  # the planner's own input bounds
    assert np.all(cable_tension(acc, ML, G) >= 0.0)


def test_margin_is_mass_independent():
    rng = np.random.default_rng(1)
    acc = rng.normal(scale=4.0, size=(500, 3))
    for mass in (0.05, 0.163, 2.0):
        np.testing.assert_allclose(
            cable_tension(acc, mass, G) / (mass * G), tension_margin(acc, G), rtol=1e-12
        )


def test_lateral_acceleration_raises_tension_above_hover():
    """Horizontal acceleration adds to ‖a + g·e₃‖: the cable pulls harder than at hover."""
    acc = np.array([[3.0, 4.0, 0.0]])
    np.testing.assert_allclose(tension_margin(acc, G), np.hypot(5.0, G) / G, rtol=1e-12)
    assert tension_margin(acc, G)[0] > 1.0


def test_labels_and_regimes_agree():
    margin = np.array([1.0, 0.5, 0.2, 0.05, 0.0])
    labels = label_taut_slack(margin, threshold=0.1)
    np.testing.assert_array_equal(labels.near_slack, [False, False, False, True, True])
    np.testing.assert_array_equal(
        labels.regime,
        [REGIME_TAUT, REGIME_TAUT, REGIME_TAUT, REGIME_NEAR_SLACK, REGIME_NEAR_SLACK],
    )
    assert labels.base_rate == pytest.approx(0.4)
    assert labels.threshold == 0.1


def test_labels_from_acceleration_are_in_newtons():
    acc = np.zeros((3, 3))
    labels = labels_from_acceleration(acc, threshold=0.1, mass_load=ML, gravity=G)
    np.testing.assert_allclose(labels.tension, ML * G, rtol=1e-12)
    np.testing.assert_allclose(labels.margin, 1.0, rtol=1e-12)
    assert not labels.near_slack.any()


def test_threshold_sweep_reports_base_rate_per_threshold():
    margin = np.linspace(0.0, 1.0, 1001)
    rows = threshold_sweep(margin, DEFAULT_THRESHOLDS)
    assert [r["threshold"] for r in rows] == list(DEFAULT_THRESHOLDS)
    for row in rows:
        # Uniform margins on [0, 1]: base rate below τ is τ, up to the grid resolution.
        assert row["base_rate"] == pytest.approx(row["threshold"], abs=2e-3)
    # Monotone in the threshold, which is the only structural guarantee worth asserting.
    rates = [r["base_rate"] for r in rows]
    assert rates == sorted(rates, reverse=True)


def test_margin_summary_ignores_nonfinite():
    margin = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
    summary = margin_summary(margin, percentiles=(0.0, 50.0, 100.0))
    assert summary["n"] == 3.0
    assert summary["n_nonfinite"] == 2.0
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(3.0)
    assert summary["p50"] == pytest.approx(2.0)


def test_tension_alignment_with_flat_outputs():
    """F3 §6: labels must be index-for-index aligned with the F2 outputs."""
    rng = np.random.default_rng(2)
    n = 256
    p = rng.normal(size=(n, 3))
    v = rng.normal(size=(n, 3))
    a = rng.normal(scale=2.0, size=(n, 3))
    j = rng.normal(scale=5.0, size=(n, 3))

    flat = flat_outputs(p, v, a, j, params=PARAMS)
    labels = labels_from_acceleration(a, threshold=0.5, mass_load=ML, gravity=G)

    assert labels.margin.shape == (n,) == flat.tension_margin.shape
    np.testing.assert_allclose(flat.tension_margin, labels.margin, rtol=1e-12)
    np.testing.assert_allclose(flat.tension, labels.tension, rtol=1e-12)


def test_bad_shape_rejected():
    with pytest.raises(ValueError):
        cable_tension(np.zeros((10, 2)), ML, G)
