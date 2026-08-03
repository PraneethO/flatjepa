"""Tests for the physics-inspired prober (F6 §5–§6).

The two that matter are:

* with the residual forced to zero, the nominal integrator must reproduce a known triple-integrator
  trajectory to numerical precision, and
* the residual head must demonstrably be *able* to fit non-zero targets — otherwise "residual ≈ 0"
  in E5-a would be equally consistent with a dead head (F6 §3).
"""

import math

import pytest
import torch

from flatjepa.models.prober import (
    PhysicalParams,
    PhysicsProber,
    ProberConfig,
    assert_timestep,
    integrate_attitude,
    orthonormalize,
    so3_exp,
    so3_hat,
)

DT = 0.05  # 20 Hz, the F4 resampled rate
PARAMS = PhysicalParams(mQ=0.85, mL=0.25, L=0.6, g=9.81)
B, T, LD = 4, 20, 24


def _prober(**kw):
    cfg = dict(latent_dim=LD, dt=DT, residual_hidden=(32, 32))
    cfg.update(kw)
    return PhysicsProber(PARAMS, ProberConfig(**cfg))


# ------------------------------------------------------------------ constants and timestep


def test_physical_params_have_no_defaults():
    """F6 §5: constants come from trajectory params; there is no silent 9.81."""
    with pytest.raises(TypeError):
        PhysicalParams()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        PhysicalParams(mQ=0.5, mL=-1.0, L=0.6, g=9.81)


def test_physical_params_from_mapping_requires_every_key():
    with pytest.raises(KeyError):
        PhysicalParams.from_mapping({"mQ": 0.85, "mL": 0.25, "L": 0.6})
    p = PhysicalParams.from_mapping({"mQ": 0.85, "mL": 0.25, "L": 0.6, "g": 9.81})
    assert p.as_dict() == {"mQ": 0.85, "mL": 0.25, "L": 0.6, "g": 9.81}


def test_prober_requires_physical_params():
    with pytest.raises(TypeError):
        PhysicsProber({"mQ": 1.0}, ProberConfig(dt=DT))  # type: ignore[arg-type]


def test_timestep_is_asserted_at_construction():
    """F6 §6: timestep consistency asserted, not assumed.  0.002 s is the raw 500 Hz CSV period —
    exactly the mismatch E5-a exists to catch."""
    assert assert_timestep(0.05) == 0.05
    with pytest.raises(ValueError, match="500 Hz"):
        assert_timestep(0.002)
    with pytest.raises(ValueError):
        assert_timestep(-0.05)
    with pytest.raises(ValueError):
        PhysicsProber(PARAMS, ProberConfig(dt=0.002))


def test_timestep_must_agree_with_resampling_provenance():
    assert assert_timestep(0.05, source_rate_hz=500.0, stride=25) == 0.05
    with pytest.raises(ValueError, match="provenance"):
        assert_timestep(0.05, source_rate_hz=500.0, stride=10)
    with pytest.raises(ValueError):
        assert_timestep(0.05, source_rate_hz=500.0)  # stride missing
    prober = PhysicsProber(PARAMS, ProberConfig(dt=0.05, source_rate_hz=500.0, stride=25))
    assert prober.dt == 0.05


def test_unusual_rate_allowed_only_with_explicit_bounds():
    prober = PhysicsProber(PARAMS, ProberConfig(dt=0.002, dt_bounds=(0.001, 0.5)))
    assert prober.dt == 0.002


# ------------------------------------------------------------------ nominal integrator


def test_nominal_integrator_matches_analytic_triple_integrator():
    """Residual forced to zero: the rollout must reproduce the closed-form solution of
    ṗ = v, v̇ = a, ȧ = u under a constant jerk, to numerical precision (F6 §6)."""
    prober = _prober(enable_residual=False, integrator="exact")
    torch.manual_seed(0)
    p0 = torch.randn(B, 3, dtype=torch.float64)
    v0 = torch.randn(B, 3, dtype=torch.float64)
    a0 = torch.randn(B, 3, dtype=torch.float64)
    u = torch.randn(B, 1, 3, dtype=torch.float64).expand(B, T, 3).contiguous()
    init = torch.cat([p0, v0, a0], dim=-1)

    states = prober.integrate(init, u)

    for k in range(T):
        t = (k + 1) * DT
        p = p0 + v0 * t + a0 * (t**2 / 2) + u[:, 0] * (t**3 / 6)
        v = v0 + a0 * t + u[:, 0] * (t**2 / 2)
        a = a0 + u[:, 0] * t
        expected = torch.cat([p, v, a], dim=-1)
        assert torch.allclose(states[:, k], expected, atol=1e-12, rtol=1e-12)


def test_nominal_integrator_matches_analytic_under_time_varying_jerk():
    """Zero-order-hold jerk: piecewise-constant between samples, exact within each step."""
    prober = _prober(enable_residual=False)
    torch.manual_seed(1)
    init = torch.randn(2, 9, dtype=torch.float64)
    u = torch.randn(2, T, 3, dtype=torch.float64)
    states = prober.integrate(init, u)

    p, v, a = init[:, 0:3], init[:, 3:6], init[:, 6:9]
    for k in range(T):
        uk = u[:, k]
        p, v, a = (
            p + v * DT + a * (DT**2 / 2) + uk * (DT**3 / 6),
            v + a * DT + uk * (DT**2 / 2),
            a + uk * DT,
        )
        assert torch.allclose(states[:, k, 0:3], p, atol=1e-13)
        assert torch.allclose(states[:, k, 3:6], v, atol=1e-13)
        assert torch.allclose(states[:, k, 6:9], a, atol=1e-13)


def test_euler_integrator_matches_the_documented_recursion():
    """F6 §1 writes explicit Euler; that form is available and must be exactly reproduced."""
    prober = _prober(enable_residual=False, integrator="euler")
    torch.manual_seed(2)
    init = torch.randn(2, 9, dtype=torch.float64)
    u = torch.randn(2, T, 3, dtype=torch.float64)
    states = prober.integrate(init, u)

    p, v, a = init[:, 0:3], init[:, 3:6], init[:, 6:9]
    for k in range(T):
        p, v, a = p + v * DT, v + a * DT, a + u[:, k] * DT
        assert torch.allclose(states[:, k], torch.cat([p, v, a], dim=-1), atol=1e-13)


def test_euler_is_measurably_worse_than_exact_at_20hz():
    """Why "exact" is the default: at Δt = 50 ms Euler's truncation error is large enough that a
    residual head would learn to absorb it, failing E5-a for a non-physical reason."""
    torch.manual_seed(3)
    init = torch.zeros(1, 9, dtype=torch.float64)
    init[0, 3:6] = 1.0  # unit velocity
    init[0, 6:9] = 1.0  # unit acceleration
    u = torch.zeros(1, T, 3, dtype=torch.float64)
    exact = _prober(enable_residual=False, integrator="exact").integrate(init, u)
    euler = _prober(enable_residual=False, integrator="euler").integrate(init, u)
    error = (exact[:, -1, 0:3] - euler[:, -1, 0:3]).abs().max()
    assert float(error) > 1e-3


def test_residual_disabled_gives_exactly_zero_residual():
    prober = _prober(enable_residual=False)
    out = prober(torch.randn(B, T, LD), torch.randn(B, T, 3), init_state=torch.randn(B, 9))
    assert torch.count_nonzero(out["residual"]) == 0
    assert torch.allclose(out["states"], out["nominal_states"], atol=1e-12)


def test_zero_initialized_residual_head_starts_at_nominal_physics():
    prober = _prober(zero_init_residual=True)
    out = prober(torch.randn(B, T, LD), torch.randn(B, T, 3), init_state=torch.randn(B, 9))
    assert float(out["residual"].abs().max()) == 0.0
    assert torch.allclose(out["states"], out["nominal_states"], atol=1e-12)


# ------------------------------------------------------------------ the residual head


def test_residual_head_can_fit_non_zero_targets():
    """F6 §3's required companion control: "residual ≈ 0" only means something if the head is not
    dead.  Fit the prober to a trajectory generated with a known non-zero jerk perturbation and
    check it recovers it."""
    torch.manual_seed(0)
    prober = _prober(residual_hidden=(64, 64), zero_init_residual=True)

    n = 64
    latents = torch.randn(n, T, LD)
    jerk = torch.randn(n, T, 3)
    init = torch.zeros(n, 9)
    # Ground truth: a perturbation that is a deterministic function of the latent, so the head has
    # something learnable to find.
    true_delta = 2.0 * torch.tanh(latents[..., :3])
    with torch.no_grad():
        target = prober.integrate(init, jerk + true_delta)

    before = float(prober(latents, jerk, init_state=init)["residual"].abs().mean())
    assert before == 0.0

    # The claim under test is "the head is not dead", so the assertions are (a) the residual becomes
    # substantially non-zero and (b) the fit improves by a large *relative* margin. An absolute loss
    # floor was tried first and is the wrong criterion: the optimisation plateaus around 3e-4 even at
    # 2000 steps, so any absolute threshold is either unreachable or arbitrary. Relative improvement
    # against the model's own starting point is the meaningful measure.
    with torch.no_grad():
        out0 = prober(latents, jerk, init_state=init, return_nominal=False)
        loss0 = float(PhysicsProber.rollout_loss(out0["states"], target))

    opt = torch.optim.Adam(prober.parameters(), lr=3e-3)
    for _ in range(2000):
        opt.zero_grad()
        out = prober(latents, jerk, init_state=init, return_nominal=False)
        loss = PhysicsProber.rollout_loss(out["states"], target)
        loss.backward()
        opt.step()

    out = prober(latents, jerk, init_state=init)
    assert float(loss) < 0.05 * loss0, f"fit barely improved: {loss0:.4g} -> {float(loss):.4g}"
    # The decisive one: a dead head would still read ~0 here.
    assert float(out["residual"].abs().mean()) > 0.1
    assert float((out["residual"] - true_delta).abs().mean()) < 0.1


def test_residual_stays_zero_on_flatness_consistent_data():
    """Miniature E5-a: when the data is generated by the nominal dynamics alone, fitting the
    prober must leave the residual near zero rather than absorbing spurious structure."""
    torch.manual_seed(0)
    prober = _prober(residual_hidden=(64, 64))

    n = 64
    latents = torch.randn(n, T, LD)
    jerk = torch.randn(n, T, 3)
    init = torch.zeros(n, 9)
    with torch.no_grad():
        target = prober.integrate(init, jerk)  # residual-free by construction

    opt = torch.optim.Adam(prober.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        out = prober(latents, jerk, init_state=init, return_nominal=False)
        PhysicsProber.rollout_loss(out["states"], target).backward()
        opt.step()

    out = prober(latents, jerk, init_state=init)
    assert float(out["residual"].abs().mean()) < 1e-3
    assert float(out["residual_displacement"]) < 1e-4


# ------------------------------------------------------------------ plumbing


def test_forward_shapes_and_separate_magnitudes():
    prober = _prober()
    out = prober(torch.randn(B, T, LD), torch.randn(B, T, 3), init_state=torch.randn(B, 9))
    assert out["states"].shape == (B, T, 9)
    assert out["residual"].shape == (B, T, 3)
    assert out["nominal_states"].shape == (B, T, 9)
    # F6 §6: residual magnitude logged separately from the nominal prediction.
    assert out["residual_magnitude"].shape == ()
    assert out["nominal_magnitude"].shape == ()


def test_initial_state_decoded_from_latent_when_not_supplied():
    prober = _prober()
    out = prober(torch.randn(B, T, LD), torch.randn(B, T, 3), init_latent=torch.randn(B, LD))
    assert out["init_state"].shape == (B, 9)
    with pytest.raises(ValueError):
        prober(torch.randn(B, T, LD), torch.randn(B, T, 3))


def test_state_feedback_variant_runs():
    prober = _prober(use_state_feedback=True, zero_init_residual=False)
    out = prober(torch.randn(B, T, LD), torch.randn(B, T, 3), init_state=torch.randn(B, 9))
    assert out["states"].shape == (B, T, 9)


def test_prober_is_fully_differentiable():
    """F6 §5: integration in PyTorch, differentiable end to end."""
    prober = _prober(zero_init_residual=False)
    latents = torch.randn(B, T, LD, requires_grad=True)
    out = prober(latents, torch.randn(B, T, 3), init_latent=torch.randn(B, LD))
    PhysicsProber.rollout_loss(out["states"], torch.randn(B, T, 9)).backward()
    assert latents.grad is not None and torch.isfinite(latents.grad).all()
    for name, p in prober.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"


def test_rollout_loss_matches_definition():
    pred = torch.randn(3, 5, 9)
    target = torch.randn(3, 5, 9)
    expected = (pred - target).pow(2).sum(dim=-1).mean()
    assert torch.allclose(PhysicsProber.rollout_loss(pred, target), expected)
    mask = torch.zeros(3, 5)
    mask[:, :2] = 1.0
    masked = PhysicsProber.rollout_loss(pred, target, mask)
    assert torch.allclose(masked, (pred - target).pow(2).sum(-1)[:, :2].mean())


def test_shape_validation():
    prober = _prober()
    with pytest.raises(ValueError):
        prober(torch.randn(B, LD), torch.randn(B, T, 3), init_state=torch.randn(B, 9))
    with pytest.raises(ValueError):
        prober(torch.randn(B, T, LD), torch.randn(B, T + 1, 3), init_state=torch.randn(B, 9))
    with pytest.raises(ValueError):
        prober(torch.randn(B, T, LD), torch.randn(B, T, 3), init_state=torch.randn(B, 6))


# ------------------------------------------------------------------ cable geometry and SO(3)


def test_cable_direction_at_rest_points_up_and_tension_is_weight():
    prober = _prober()
    q, tension = prober.cable_direction_and_tension(torch.zeros(2, 3))
    assert torch.allclose(q, torch.tensor([0.0, 0.0, 1.0]).expand(2, 3), atol=1e-6)
    assert torch.allclose(tension, torch.full((2,), PARAMS.mL * PARAMS.g), atol=1e-6)
    p = torch.zeros(2, 3)
    assert torch.allclose(prober.quad_position(p, torch.zeros(2, 3))[:, 2],
                          torch.full((2,), PARAMS.L), atol=1e-6)


def test_cable_direction_uses_params_not_constants():
    heavy = PhysicsProber(PhysicalParams(mQ=1.0, mL=2.0, L=1.5, g=3.7), ProberConfig(dt=DT))
    _, tension = heavy.cable_direction_and_tension(torch.zeros(1, 3))
    assert float(tension) == pytest.approx(2.0 * 3.7)


def test_so3_exp_is_a_rotation_and_matches_known_case():
    w = torch.tensor([[0.0, 0.0, math.pi / 2]])
    r = so3_exp(w)[0]
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(r, expected, atol=1e-6)
    assert torch.allclose(r @ r.T, torch.eye(3), atol=1e-6)
    assert float(torch.linalg.det(r)) == pytest.approx(1.0, abs=1e-6)


def test_so3_exp_is_finite_and_differentiable_at_zero():
    """The small-angle branch must not leak NaN gradients through torch.where."""
    w = torch.zeros(1, 3, requires_grad=True)
    r = so3_exp(w)
    assert torch.allclose(r[0], torch.eye(3), atol=1e-12)
    r.sum().backward()
    assert torch.isfinite(w.grad).all()


def test_so3_hat_is_skew():
    w = torch.randn(4, 3)
    k = so3_hat(w)
    assert k.shape == (4, 3, 3)
    assert torch.allclose(k, -k.transpose(-1, -2), atol=1e-7)
    v = torch.randn(4, 3)
    assert torch.allclose((k @ v.unsqueeze(-1)).squeeze(-1), torch.cross(w, v, dim=-1), atol=1e-6)


def test_orthonormalize_repairs_drift():
    torch.manual_seed(0)
    r = so3_exp(torch.randn(5, 3)) + 0.01 * torch.randn(5, 3, 3)
    fixed = orthonormalize(r)
    eye = torch.eye(3).expand(5, 3, 3)
    assert torch.allclose(fixed @ fixed.transpose(-1, -2), eye, atol=1e-6)
    assert torch.allclose(torch.linalg.det(fixed), torch.ones(5), atol=1e-6)


def test_attitude_integration_stays_on_so3_and_composes():
    """Constant body rate for T steps must equal a single exponential of the total angle — the
    check that would fail under naive componentwise integration of a rotation matrix."""
    r0 = torch.eye(3).expand(2, 3, 3).contiguous()
    omega = torch.tensor([0.0, 0.0, 1.3]).expand(2, T, 3).contiguous()
    rs = integrate_attitude(r0, omega, DT)
    assert rs.shape == (2, T, 3, 3)
    total = so3_exp(omega[:, 0] * DT * T)
    assert torch.allclose(rs[:, -1], total, atol=1e-5)
    eye = torch.eye(3).expand(2, 3, 3)
    assert torch.allclose(rs[:, -1] @ rs[:, -1].transpose(-1, -2), eye, atol=1e-6)
