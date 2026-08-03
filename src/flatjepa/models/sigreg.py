"""Sketched Isotropic Gaussian Regularization (F5 §3).

SIGReg comes from LeJEPA (Balestriero & LeCun, arXiv:2511.08544; reference implementation at
https://github.com/rbalestr-lab/lejepa).  The mechanism rests on the Cramér–Wold theorem: a
distribution is standard multivariate Gaussian iff *every* one-dimensional projection of it is
standard univariate Gaussian.  So instead of ever forming a ``D × D`` covariance:

1. sample ``M`` random unit directions in latent space,
2. project the batch of latents onto each, giving ``M`` univariate samples,
3. score each against ``N(0, 1)`` with the Epps–Pulley statistic,
4. penalize the average.

Directions are resampled every call, so isotropy is enforced in expectation over training rather
than along any fixed frame.

The Epps–Pulley statistic is an empirical-characteristic-function goodness-of-fit test,

.. math::  T = \\int (\\hat\\varphi_n(t) - e^{-t^2/2})^2 \\, w(t) \\, dt , \\qquad w(t) = e^{-t^2/2}

evaluated by the trapezoidal rule on a symmetric grid of knots (LeJEPA's default is 17 knots).  It
is used rather than, say, a moment penalty because it is bounded with bounded gradients, which is
what makes it safe to optimize.

Two conventions worth stating explicitly, because they are the places this implementation had to
make a call the design doc did not:

* **No centering or standardization of the projections.**  The target is ``N(0, I)`` exactly, so
  both the mean and the scale of every projection are part of what is being constrained.
  Standardizing each direction first would make any full-rank Gaussian pass and would delete the
  isotropy signal entirely.
* **Batch scaling.**  The classical statistic is multiplied by ``n``, which makes it converge to a
  fixed null distribution (mean ≈ 1.06 for this weight) instead of to zero.  As a *loss* that is
  awkward: F5 §6 asks for "near-zero penalty" on Gaussian input, and the ``n`` factor also makes
  the SIGReg gradient per sample ``O(1)`` while the prediction loss (a batch mean) contributes
  ``O(1/n)``, so ``λ_sig`` would silently change meaning with batch size.  The default here is
  therefore the *unscaled* integral (``scale_by_batch=False``), which is ``≈ 1.06/n`` under the null
  and ``≈ 0.41`` for a fully collapsed batch.  ``scale_by_batch=True`` recovers the textbook
  hypothesis-test scaling.

Known limitations, both measured rather than assumed (see ``tests/test_sigreg.py``):

* **Exact collapse to the origin is a stationary point.**  If every latent is exactly ``0`` the
  batch is symmetric under ``x → −x``, both the empirical characteristic function and the target
  are real and even, and the gradient of the statistic is *identically zero*.  Collapse to any
  other constant, or to a constant plus noise, does have a gradient and is escaped readily.  This
  is a property of any symmetric goodness-of-fit statistic, not a bug, but it means SIGReg should
  not be relied on to rescue a representation that has already collapsed to exactly zero — it is a
  preventative, and initialization noise is what keeps it in play.
* **Higher-order non-Gaussianity fades with latent width.**  Random one-dimensional projections of
  a ``D``-dimensional product distribution are themselves near-Gaussian by the central limit
  theorem, so the statistic's power against non-Gaussian *shape* drops as ``D`` grows: a uniform
  distribution rescaled to unit variance scores 40× the null at ``D = 1`` but only 1.2× at
  ``D = 24``.  Mean and covariance structure (shift, anisotropy, low rank) remain strongly
  detected at every width.  In practice, therefore, SIGReg at realistic latent widths is close to
  a whitening constraint — worth keeping in mind when interpreting E3, whose whole question is
  whether enforced isotropy distorts an intrinsically non-isotropic latent geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "SIGRegConfig",
    "SIGReg",
    "sigreg_loss",
    "epps_pulley_statistic",
    "random_unit_directions",
    "EPPS_PULLEY_NULL_MEAN",
]

# E[∫ (φ̂_n − φ)² w dt] · n under H0 with w(t) = φ(t) = exp(−t²/2):
#   ∫ (1 − e^{−t²}) e^{−t²/2} dt = √(2π) − √(2π/3) ≈ 1.0594.
# Used by the tests as an analytic reference for "near-zero on Gaussian input".
EPPS_PULLEY_NULL_MEAN = 1.0594


@dataclass
class SIGRegConfig:
    """Configuration for :class:`SIGReg`.

    Attributes
    ----------
    num_slices:
        Number of random directions ``M`` drawn per call.  LeJEPA uses ~1024; the default here is
        smaller so CPU tests stay cheap.  Memory is ``O(N · M · K)``.
    num_points:
        Number of characteristic-function knots ``K`` (LeJEPA default 17).
    t_max:
        Knots span ``[-t_max, t_max]``.  The weight ``e^{-t²/2}`` is already negligible past 5.
    scale_by_batch:
        Multiply the statistic by the batch size, recovering the classical Epps–Pulley scaling.
        See the module docstring for why this is off by default.
    """

    num_slices: int = 256
    num_points: int = 17
    t_max: float = 5.0
    scale_by_batch: bool = False


def random_unit_directions(
    dim: int,
    num_slices: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``(dim, num_slices)`` i.i.d. uniform directions on the unit sphere.

    Gaussian columns normalized to unit norm are uniform on ``S^{dim-1}``.  A ``generator`` may be
    passed so a run is reproducible from a seed even though directions are resampled every step.
    """
    if dim <= 0 or num_slices <= 0:
        raise ValueError("dim and num_slices must be positive")
    directions = torch.randn(
        dim, num_slices, device=device, dtype=dtype or torch.get_default_dtype(), generator=generator
    )
    return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


def epps_pulley_statistic(
    projections: torch.Tensor,
    num_points: int = 17,
    t_max: float = 5.0,
    scale_by_batch: bool = False,
) -> torch.Tensor:
    """Per-direction Epps–Pulley statistic against ``N(0, 1)``.

    Parameters
    ----------
    projections:
        ``(N, M)`` — ``N`` samples projected onto ``M`` directions.
    num_points:
        Number of trapezoid knots ``K`` over ``[-t_max, t_max]``.

    Returns
    -------
    ``(M,)`` non-negative statistics, one per direction.
    """
    if projections.dim() != 2:
        raise ValueError(f"expected (N, M) projections, got shape {tuple(projections.shape)}")
    if num_points < 2:
        raise ValueError("num_points must be at least 2 for trapezoidal integration")
    n = projections.shape[0]
    if n < 2:
        raise ValueError("need at least 2 samples to estimate a characteristic function")

    t = torch.linspace(
        -t_max, t_max, num_points, device=projections.device, dtype=projections.dtype
    )
    phi = torch.exp(-0.5 * t**2)  # (K,) target characteristic function == weight

    # Empirical characteristic function, real arithmetic: E[cos(t x)] and E[sin(t x)].
    arg = projections.unsqueeze(-1) * t  # (N, M, K)
    ecf_re = torch.cos(arg).mean(dim=0)  # (M, K)
    ecf_im = torch.sin(arg).mean(dim=0)  # (M, K)

    integrand = ((ecf_re - phi) ** 2 + ecf_im**2) * phi  # (M, K)
    stat = torch.trapezoid(integrand, t, dim=-1)  # (M,)
    if scale_by_batch:
        stat = stat * n
    return stat


def sigreg_loss(
    latents: torch.Tensor,
    num_slices: int = 256,
    num_points: int = 17,
    t_max: float = 5.0,
    scale_by_batch: bool = False,
    generator: torch.Generator | None = None,
    directions: torch.Tensor | None = None,
) -> torch.Tensor:
    """SIGReg penalty for a batch of latents.

    Parameters
    ----------
    latents:
        ``(N, D)``; leading axes are flattened, so ``(B, T, D)`` is accepted and treated as
        ``B·T`` samples.
    directions:
        Optional fixed ``(D, M)`` directions.  Normally left ``None`` so directions are resampled,
        which is what makes the sketch cover the sphere in expectation; supplying them is for
        tests and diagnostics.

    Returns
    -------
    Scalar penalty, mean over directions.
    """
    if latents.dim() < 2:
        raise ValueError(f"expected at least a (N, D) tensor, got shape {tuple(latents.shape)}")
    flat = latents.reshape(-1, latents.shape[-1])
    if directions is None:
        directions = random_unit_directions(
            flat.shape[-1], num_slices, device=flat.device, dtype=flat.dtype, generator=generator
        )
    elif directions.shape[0] != flat.shape[-1]:
        raise ValueError(
            f"directions have dim {directions.shape[0]} but latents have dim {flat.shape[-1]}"
        )
    projections = flat @ directions  # (N, M)
    stats = epps_pulley_statistic(
        projections, num_points=num_points, t_max=t_max, scale_by_batch=scale_by_batch
    )
    return stats.mean()


class SIGReg(nn.Module):
    """Stateless module wrapper around :func:`sigreg_loss`.

    Holds the configuration and an optional :class:`torch.Generator` so that direction resampling
    is reproducible from a seed (F5 §6 requires determinism given a seed, and the resampling is the
    only stochastic part of the training objective).
    """

    def __init__(self, config: SIGRegConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or SIGRegConfig()
        if overrides:
            cfg = SIGRegConfig(**{**cfg.__dict__, **overrides})
        self.config = cfg
        self._generator: torch.Generator | None = None

    def set_generator(self, generator: torch.Generator | None) -> None:
        """Use a dedicated RNG stream for direction sampling."""
        self._generator = generator

    def forward(self, latents: torch.Tensor, directions: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.config
        return sigreg_loss(
            latents,
            num_slices=cfg.num_slices,
            num_points=cfg.num_points,
            t_max=cfg.t_max,
            scale_by_batch=cfg.scale_by_batch,
            generator=self._generator,
            directions=directions,
        )

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"num_slices={c.num_slices}, num_points={c.num_points}, "
            f"t_max={c.t_max}, scale_by_batch={c.scale_by_batch}"
        )
