"""Collapse diagnostics for latent representations (F5 §4).

Collapse is the central failure mode of this objective — with a single encoder producing both the
prediction and its own target, ``Enc_θ ≡ c`` drives ``ℒ_pred`` to *exactly* zero — so it has to be
watched during training, not diagnosed afterwards.  F5 §4 names four quantities:

* latent variance per dimension (collapse → 0),
* PCA spectrum / rank of a batch of latents,
* participation ratio,
* pairwise latent distance distribution.

:func:`latent_diagnostics` computes all four in one pass and :class:`CollapseAlarm` turns them into
a boolean plus human-readable reasons, so an overnight run cannot spend eight hours training a
constant function.

Two notions of "effective rank" are reported because they answer slightly different questions and
E2 uses both:

* ``effective_rank`` — ``exp(H(p))`` where ``p_i = λ_i / Σλ`` (Roy & Vetterli).  Equals ``D`` for a
  flat spectrum, 1 for a rank-one spectrum.
* ``participation_ratio`` — ``(Σλ)² / Σλ²``, the estimator F7 §3 names for the E2 comparison
  against the theoretical minimal dimension of ~9.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

__all__ = [
    "LatentDiagnostics",
    "latent_diagnostics",
    "CollapseAlarm",
    "CollapseAlarmResult",
    "effective_rank",
    "participation_ratio",
]

_TINY = 1e-12


def effective_rank(eigenvalues: torch.Tensor, eps: float = _TINY) -> float:
    """Entropy-based effective rank ``exp(-Σ p log p)`` of a non-negative spectrum."""
    lam = eigenvalues.clamp_min(0.0)
    total = lam.sum()
    if float(total) <= eps:
        return 0.0
    p = lam / total
    entropy = -(p * torch.log(p.clamp_min(eps))).sum()
    return float(torch.exp(entropy))


def participation_ratio(eigenvalues: torch.Tensor, eps: float = _TINY) -> float:
    """``(Σλ)² / Σλ²``."""
    lam = eigenvalues.clamp_min(0.0)
    denom = float((lam**2).sum())
    if denom <= eps:
        return 0.0
    return float(lam.sum() ** 2 / denom)


@dataclass
class LatentDiagnostics:
    """Collapse diagnostics for one batch of latents.  Scalars are plain floats so the whole thing
    serializes straight into a JSONL log line (F8 §4)."""

    num_samples: int
    latent_dim: int
    variance_mean: float
    variance_min: float
    variance_max: float
    total_variance: float
    effective_rank: float
    participation_ratio: float
    stable_rank: float
    numerical_rank: int
    top_eigenvalue_ratio: float
    pairwise_distance_mean: float
    pairwise_distance_std: float
    pairwise_distance_min: float
    pairwise_distance_max: float
    latent_norm_mean: float
    per_dim_variance: list[float] = field(default_factory=list)
    eigenvalues: list[float] = field(default_factory=list)
    explained_variance_ratio: list[float] = field(default_factory=list)

    def as_dict(self, include_vectors: bool = False) -> dict[str, object]:
        """Flat mapping for logging; vector-valued entries are dropped unless asked for."""
        data = asdict(self)
        if not include_vectors:
            for key in ("per_dim_variance", "eigenvalues", "explained_variance_ratio"):
                data.pop(key)
        return data


@torch.no_grad()
def latent_diagnostics(
    latents: torch.Tensor,
    max_pairwise_samples: int = 512,
    generator: torch.Generator | None = None,
) -> LatentDiagnostics:
    """Compute the F5 §4 diagnostics for a batch of latents.

    Parameters
    ----------
    latents:
        ``(N, D)`` or any shape whose trailing axis is the latent; leading axes are flattened, so
        ``(B, T, D)`` rollout latents are accepted directly.
    max_pairwise_samples:
        Cap on the number of rows used for the pairwise-distance histogram, which is ``O(n²)``.

    Notes
    -----
    The spectrum is that of the *centered* covariance, i.e. PCA proper.  A batch that has collapsed
    to a single constant (even a large one) therefore reports zero variance, zero effective rank and
    zero pairwise distance, which is what the alarm keys on.
    """
    if latents.dim() < 2:
        raise ValueError(f"expected at least a (N, D) tensor, got shape {tuple(latents.shape)}")
    x = latents.detach().reshape(-1, latents.shape[-1]).to(torch.float64)
    n, d = x.shape
    if n < 2:
        raise ValueError("need at least 2 latent samples for diagnostics")

    per_dim_var = x.var(dim=0, unbiased=True)
    centered = x - x.mean(dim=0, keepdim=True)
    # Eigenvalues of the covariance via SVD of the centered matrix (numerically safer than eigh
    # on an explicitly formed covariance).
    singular = torch.linalg.svdvals(centered)
    eigenvalues = (singular**2) / max(n - 1, 1)
    total = float(eigenvalues.sum())
    explained = (eigenvalues / total).tolist() if total > _TINY else [0.0] * d
    top_ratio = float(eigenvalues[0] / total) if total > _TINY else 0.0
    stable = float(eigenvalues.sum() / eigenvalues[0]) if float(eigenvalues[0]) > _TINY else 0.0
    tol = float(singular[0]) * max(n, d) * torch.finfo(torch.float64).eps
    numerical = int((singular > tol).sum()) if float(singular[0]) > _TINY else 0

    if n > max_pairwise_samples:
        idx = torch.randperm(n, generator=generator, device=x.device)[:max_pairwise_samples]
        sample = x[idx]
    else:
        sample = x
    dists = torch.cdist(sample, sample)
    iu = torch.triu_indices(sample.shape[0], sample.shape[0], offset=1, device=x.device)
    pair = dists[iu[0], iu[1]]

    return LatentDiagnostics(
        num_samples=int(n),
        latent_dim=int(d),
        variance_mean=float(per_dim_var.mean()),
        variance_min=float(per_dim_var.min()),
        variance_max=float(per_dim_var.max()),
        total_variance=total,
        effective_rank=effective_rank(eigenvalues),
        participation_ratio=participation_ratio(eigenvalues),
        stable_rank=stable,
        numerical_rank=numerical,
        top_eigenvalue_ratio=top_ratio,
        pairwise_distance_mean=float(pair.mean()),
        pairwise_distance_std=float(pair.std()) if pair.numel() > 1 else 0.0,
        pairwise_distance_min=float(pair.min()),
        pairwise_distance_max=float(pair.max()),
        latent_norm_mean=float(x.norm(dim=-1).mean()),
        per_dim_variance=per_dim_var.tolist(),
        eigenvalues=eigenvalues.tolist(),
        explained_variance_ratio=explained,
    )


@dataclass
class CollapseAlarmResult:
    """Outcome of a collapse check."""

    triggered: bool
    reasons: list[str]
    diagnostics: LatentDiagnostics

    def __bool__(self) -> bool:
        return self.triggered

    def message(self) -> str:
        if not self.triggered:
            return "no collapse detected"
        return "collapse alarm: " + "; ".join(self.reasons)


@dataclass
class CollapseAlarm:
    """Threshold-based collapse detector, evaluated every epoch (F5 §4, F8 §4).

    Attributes
    ----------
    min_effective_rank:
        Absolute floor on the entropy-based effective rank.
    min_effective_rank_fraction:
        Optional floor expressed as a fraction of the allocated latent width, for sweeps where the
        width changes between runs and a single absolute threshold does not transfer.
    min_variance_mean:
        Floor on the mean per-dimension variance.
    min_pairwise_distance_mean:
        Floor on the mean pairwise latent distance — the check that catches a batch collapsed onto
        a single point regardless of where that point is.
    """

    min_effective_rank: float = 2.0
    min_effective_rank_fraction: float | None = None
    min_variance_mean: float = 1e-6
    min_pairwise_distance_mean: float = 1e-4

    def check(self, latents_or_diagnostics: torch.Tensor | LatentDiagnostics) -> CollapseAlarmResult:
        """Evaluate the thresholds against latents (diagnostics are computed) or existing
        diagnostics (reused)."""
        if isinstance(latents_or_diagnostics, LatentDiagnostics):
            diag = latents_or_diagnostics
        else:
            diag = latent_diagnostics(latents_or_diagnostics)

        reasons: list[str] = []
        if diag.effective_rank < self.min_effective_rank:
            reasons.append(
                f"effective rank {diag.effective_rank:.3f} < {self.min_effective_rank}"
            )
        if self.min_effective_rank_fraction is not None:
            floor = self.min_effective_rank_fraction * diag.latent_dim
            if diag.effective_rank < floor:
                reasons.append(
                    f"effective rank {diag.effective_rank:.3f} < {self.min_effective_rank_fraction}"
                    f" x latent_dim ({floor:.3f})"
                )
        if diag.variance_mean < self.min_variance_mean:
            reasons.append(
                f"mean per-dim variance {diag.variance_mean:.3e} < {self.min_variance_mean:.3e}"
            )
        if diag.pairwise_distance_mean < self.min_pairwise_distance_mean:
            reasons.append(
                f"mean pairwise distance {diag.pairwise_distance_mean:.3e} < "
                f"{self.min_pairwise_distance_mean:.3e}"
            )
        return CollapseAlarmResult(triggered=bool(reasons), reasons=reasons, diagnostics=diag)
