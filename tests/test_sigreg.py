"""Correctness tests for SIGReg (F5 §3, §6).

The load-bearing test is the last acceptance criterion of F5 §6: isotropic Gaussian input must give
a near-zero penalty and collapsed/constant input a large one.  Everything else here guards the
formulation details that would make that test pass for the wrong reason.
"""

import math

import pytest
import torch

from flatjepa.models.sigreg import (
    EPPS_PULLEY_NULL_MEAN,
    SIGReg,
    SIGRegConfig,
    epps_pulley_statistic,
    random_unit_directions,
    sigreg_loss,
)

D = 24
N = 2048


def _gaussian(n=N, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


# ------------------------------------------------------------------ the headline criterion


def test_isotropic_gaussian_near_zero_and_collapse_large():
    torch.manual_seed(0)
    gaussian = sigreg_loss(_gaussian(), num_slices=256)
    collapsed = sigreg_loss(torch.full((N, D), 0.7), num_slices=256)

    assert float(gaussian) < 0.01, f"isotropic Gaussian penalty should be ~0, got {float(gaussian)}"
    assert float(collapsed) > 0.1, f"collapsed penalty should be large, got {float(collapsed)}"
    assert float(collapsed) > 30 * float(gaussian)


def test_gaussian_penalty_matches_analytic_null_and_vanishes_with_batch_size():
    """Under H0 the unscaled statistic has expectation ≈ 1.0594/n, so it really does go to zero as
    the batch grows — this pins the normalization convention, not just its order of magnitude."""
    torch.manual_seed(0)
    for n in (256, 1024, 4096):
        value = float(sigreg_loss(_gaussian(n=n, seed=n), num_slices=512))
        assert value == pytest.approx(EPPS_PULLEY_NULL_MEAN / n, rel=0.5)


def test_collapse_variants_all_penalized():
    torch.manual_seed(0)
    baseline = float(sigreg_loss(_gaussian(), num_slices=256))
    cases = {
        "constant zero": torch.zeros(N, D),
        "constant nonzero": torch.full((N, D), 3.0),
        "rank one": torch.randn(N, 1) @ torch.randn(1, D),
        "shrunk": _gaussian() * 0.05,
        "inflated": _gaussian() * 5.0,
    }
    for name, z in cases.items():
        value = float(sigreg_loss(z, num_slices=256))
        assert value > 20 * baseline, f"{name} should be penalized, got {value} vs {baseline}"


def test_non_gaussian_shape_detected_in_low_dimension():
    """A uniform distribution rescaled to unit variance passes any covariance-only check; the
    Epps-Pulley statistic sees it as non-Gaussian in low dimension."""
    torch.manual_seed(0)
    for d in (1, 2):
        uniform = (torch.rand(N, d) - 0.5) * math.sqrt(12.0)
        null = float(sigreg_loss(torch.randn(N, d), num_slices=256))
        assert float(sigreg_loss(uniform, num_slices=256)) > 5 * null


def test_shape_sensitivity_decays_with_latent_width():
    """Characterization, not an aspiration: random 1-D projections of a D-dimensional product
    distribution are near-Gaussian by the CLT, so power against non-Gaussian *shape* falls as D
    grows.  Recorded here because it bounds what SIGReg can be claimed to enforce in E3."""
    torch.manual_seed(0)
    # Compare against the analytic null (1.0594/n) rather than a single empirical Gaussian draw:
    # at D = 1 every "random direction" is ±1, so an empirical null estimate is one noisy sample.
    ratios = []
    for d in (1, 4, 24):
        uniform = (torch.rand(N, d) - 0.5) * math.sqrt(12.0)
        ratios.append(float(sigreg_loss(uniform, num_slices=256)) / (EPPS_PULLEY_NULL_MEAN / N))
    assert ratios[0] > ratios[1] > ratios[2]
    assert ratios[2] < 3.0  # essentially invisible at width 24


def test_penalty_is_non_negative():
    torch.manual_seed(0)
    for scale in (0.0, 0.1, 1.0, 10.0):
        assert float(sigreg_loss(_gaussian() * scale, num_slices=64)) >= 0.0


# ------------------------------------------------------------------ formulation details


def test_statistic_shape_and_batch_scaling():
    proj = torch.randn(N, 32)
    stat = epps_pulley_statistic(proj, num_points=17)
    assert stat.shape == (32,)
    scaled = epps_pulley_statistic(proj, num_points=17, scale_by_batch=True)
    assert torch.allclose(scaled, stat * N, rtol=1e-5)
    # Classical scaling: the statistic converges to an O(1) null distribution rather than to zero.
    assert 0.3 < float(scaled.mean()) < 4.0


def test_directions_are_unit_norm_and_resampled():
    g = torch.Generator().manual_seed(3)
    dirs = random_unit_directions(D, 128, generator=g)
    assert dirs.shape == (D, 128)
    assert torch.allclose(dirs.norm(dim=0), torch.ones(128), atol=1e-5)
    other = random_unit_directions(D, 128, generator=g)
    assert not torch.allclose(dirs, other)


def test_shift_is_penalized_so_projections_are_not_centered():
    """A Gaussian with the right covariance but a nonzero mean must still be penalized; if the
    projections were centered first, this would slip through and isotropy about the origin would
    not be enforced."""
    torch.manual_seed(0)
    shifted = _gaussian() + 2.0
    assert float(sigreg_loss(shifted, num_slices=256)) > 20 * float(
        sigreg_loss(_gaussian(), num_slices=256)
    )


def test_anisotropic_covariance_is_penalized():
    """Unit variance in some directions and tiny variance in others: only a direction-sketched test
    catches this."""
    torch.manual_seed(0)
    z = _gaussian().clone()
    z[:, D // 2 :] *= 0.02
    assert float(sigreg_loss(z, num_slices=256)) > 20 * float(sigreg_loss(_gaussian(), num_slices=256))


def test_accepts_rollout_shaped_latents():
    torch.manual_seed(0)
    assert sigreg_loss(torch.randn(8, 20, D), num_slices=32).shape == ()


def test_gradients_are_finite_on_a_near_collapsed_batch():
    torch.manual_seed(0)
    z = (torch.full((512, D), 0.5) + 0.01 * torch.randn(512, D)).requires_grad_(True)
    sigreg_loss(z, num_slices=256, generator=torch.Generator().manual_seed(1)).backward()
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum()) > 0.0


def test_exact_zero_collapse_is_a_stationary_point():
    """Documented limitation: a batch collapsed to exactly the origin is symmetric under x -> -x,
    so the statistic's gradient vanishes there.  Collapse to any other constant does not."""
    torch.manual_seed(0)
    zero = torch.zeros(256, D, requires_grad=True)
    sigreg_loss(zero, num_slices=128, generator=torch.Generator().manual_seed(1)).backward()
    assert float(zero.grad.abs().sum()) == 0.0

    shifted = torch.full((256, D), 0.5, requires_grad=True)
    sigreg_loss(shifted, num_slices=128, generator=torch.Generator().manual_seed(1)).backward()
    assert float(shifted.grad.abs().sum()) > 0.0


def test_optimizing_sigreg_escapes_collapse_toward_standard_normal():
    """End-to-end: descending the penalty from a nearly collapsed batch must recover unit variance
    and zero mean.  Detecting collapse is not enough; the gradient has to fix it."""
    torch.manual_seed(0)
    dirs = random_unit_directions(8, 128, generator=torch.Generator().manual_seed(2))
    z = (0.01 * torch.randn(256, 8)).requires_grad_(True)
    before = float(sigreg_loss(z, directions=dirs).detach())
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(250):
        opt.zero_grad()
        loss = sigreg_loss(z, directions=dirs)
        loss.backward()
        opt.step()
    after = float(sigreg_loss(z, directions=dirs).detach())
    assert after < 0.05 * before
    assert float(z.detach().var(dim=0).mean()) == pytest.approx(1.0, abs=0.15)
    assert abs(float(z.detach().mean())) < 0.05


def test_module_wrapper_matches_functional_and_is_seedable():
    module = SIGReg(SIGRegConfig(num_slices=64))
    z = _gaussian(n=512)
    module.set_generator(torch.Generator().manual_seed(7))
    a = float(module(z))
    module.set_generator(torch.Generator().manual_seed(7))
    b = float(module(z))
    assert a == b
    module.set_generator(torch.Generator().manual_seed(8))
    assert float(module(z)) != a  # directions really are resampled


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        sigreg_loss(torch.randn(10))
    with pytest.raises(ValueError):
        epps_pulley_statistic(torch.randn(1, 4))
    with pytest.raises(ValueError):
        epps_pulley_statistic(torch.randn(10, 4), num_points=1)
