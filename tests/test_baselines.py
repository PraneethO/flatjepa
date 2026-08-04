"""Tests for F10 baselines.

The one that matters is the first: the constant-jerk rollout must be *exact* on data that really is
a triple integrator. It is the reference the learned models are measured against, so an error here
would flatter or penalise everything else.
"""

import torch

from flatjepa.models.baselines import (
    DirectRegressorConfig,
    DirectStateRegressor,
    ObservationGRU,
    constant_jerk_rollout,
    zero_order_hold,
)

B, H, T, D = 4, 10, 20, 9
DT = 0.1


def test_constant_jerk_rollout_is_exact_on_a_triple_integrator():
    """Generate a trajectory by the same exact ZOH rule and check the rollout reproduces it."""
    torch.manual_seed(0)
    p = torch.randn(B, 3)
    v = torch.randn(B, 3)
    a = torch.randn(B, 3)
    jerks = torch.randn(B, T, 3)

    states = []
    pp, vv, aa = p.clone(), v.clone(), a.clone()
    for k in range(T):
        u = jerks[:, k]
        pp = pp + vv * DT + 0.5 * aa * DT**2 + u * DT**3 / 6.0
        vv = vv + aa * DT + 0.5 * u * DT**2
        aa = aa + u * DT
        states.append(torch.cat([pp, vv, aa], dim=-1))
    truth = torch.stack(states, dim=1)

    hist = torch.zeros(B, H, D)
    hist[:, -1] = torch.cat([p, v, a], dim=-1)
    out = constant_jerk_rollout(hist, jerks, DT)
    torch.testing.assert_close(out, truth, atol=1e-6, rtol=0)


def test_constant_jerk_matches_closed_form_for_zero_jerk():
    """With no jerk the motion is exactly p + vt + at²/2."""
    hist = torch.zeros(B, H, D)
    hist[:, -1, 0:3] = 1.0
    hist[:, -1, 3:6] = 2.0
    hist[:, -1, 6:9] = 3.0
    out = constant_jerk_rollout(hist, torch.zeros(B, T, 3), DT)
    k = torch.arange(1, T + 1, dtype=torch.float32) * DT
    expected_p = 1.0 + 2.0 * k + 0.5 * 3.0 * k**2
    torch.testing.assert_close(out[0, :, 0], expected_p, atol=1e-5, rtol=0)
    # Acceleration is unchanged when jerk is zero.
    torch.testing.assert_close(out[..., 6:9], torch.full((B, T, 3), 3.0), atol=1e-6, rtol=0)


def test_zero_order_hold_repeats_last_state():
    hist = torch.randn(B, H, D)
    out = zero_order_hold(hist, T)
    assert out.shape == (B, T, D)
    for k in range(T):
        torch.testing.assert_close(out[:, k], hist[:, -1, :9])


def test_direct_regressor_shapes_and_gradients():
    model = DirectStateRegressor(DirectRegressorConfig(horizon=T))
    out = model(torch.randn(B, H, D), torch.randn(B, H, 3), torch.randn(B, T, 3))
    assert out.shape == (B, T, D)
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_observation_gru_shapes_and_gradients():
    model = ObservationGRU(horizon=T)
    out = model(torch.randn(B, H, D), torch.randn(B, H, 3), torch.randn(B, T, 3))
    assert out.shape == (B, T, D)
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_baselines_are_deterministic_given_a_seed():
    x = (torch.randn(B, H, D), torch.randn(B, H, 3), torch.randn(B, T, 3))
    torch.manual_seed(7)
    a = DirectStateRegressor(DirectRegressorConfig(horizon=T))(*x)
    torch.manual_seed(7)
    b = DirectStateRegressor(DirectRegressorConfig(horizon=T))(*x)
    torch.testing.assert_close(a, b)
