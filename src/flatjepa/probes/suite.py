"""Measurement suite (F7): E1 latent recovery, E2 intrinsic dimensionality, E3 SIGReg ablation.

Every probe result is reported against three controls (F7 §1). They are not optional decoration --
they are what distinguishes a finding from an artifact:

* **random-init encoder** -- same architecture, no training. If the trained encoder does not beat
  it, training accomplished nothing and E1 is vacuous.
* **raw input window** -- a linear probe on the flattened input. Anything it already solves was
  never the representation's achievement.
* **shuffled labels** -- the chance floor for this probe and dataset.

Probes are **linear** (ridge) by design: linear accessibility is a stronger and more interpretable
claim than "an MLP can dig it out". Probes are fit on train and reported on test, using the
environment-level split from F4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from flatjepa.data.targets import TARGET_SPECS

DEFAULT_ALPHAS: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


# --------------------------------------------------------------------------------------------
# Ridge probe
# --------------------------------------------------------------------------------------------


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    gram = xb.T @ xb + alpha * np.eye(xb.shape[1], dtype=x.dtype)
    gram[-1, -1] -= alpha  # unpenalised intercept
    return np.linalg.solve(gram, xb.T @ y)


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1) @ w


def r2_per_channel(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-channel R². Zero-variance channels yield NaN rather than a fabricated 0 or 1."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)
    return out


def ridge_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> dict[str, Any]:
    """Ridge probe with alpha selected on validation, scored on test.

    Returns mean and per-channel R², with zero-variance channels excluded from the mean rather than
    silently dragging it toward 0 or 1.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    x_val = np.asarray(x_val, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    best_alpha, best_score, best_w = None, -np.inf, None
    for alpha in alphas:
        w = _fit_ridge(x_train, y_train, alpha)
        score = float(np.nanmean(r2_per_channel(y_val, _predict(x_val, w))))
        if score > best_score:
            best_alpha, best_score, best_w = alpha, score, w

    per_channel = r2_per_channel(y_test, _predict(x_test, best_w))
    return {
        "r2": float(np.nanmean(per_channel)),
        "r2_per_channel": [None if np.isnan(v) else float(v) for v in per_channel],
        "alpha": best_alpha,
        "val_r2": best_score,
        "n_constant_channels": int(np.isnan(per_channel).sum()),
    }


# --------------------------------------------------------------------------------------------
# Latent extraction
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def encode_split(model, split, batch_size: int = 4096) -> np.ndarray:
    """Context latents for every window in a split, as ``(N, latent_dim)`` float64."""
    model.eval()
    chunks = []
    states = split.tensors["state_hist"]
    for start in range(0, len(split), batch_size):
        chunks.append(model.encode_context(states[start : start + batch_size]).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


# --------------------------------------------------------------------------------------------
# E2 -- intrinsic dimensionality
# --------------------------------------------------------------------------------------------


def effective_dimensionality(latents: np.ndarray) -> dict[str, Any]:
    """PCA spectrum summaries of a latent matrix.

    ``participation_ratio`` is ``(Σλ)² / Σλ²`` -- the standard continuous relaxation of "how many
    directions carry variance". E2's prediction is that it lands near 9, the dimension of the
    planner's own state ``(p, v, a)``.
    """
    x = np.asarray(latents, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    eig = np.linalg.eigvalsh(cov)[::-1]
    eig = np.clip(eig, 0.0, None)
    total = eig.sum()

    if total <= 0:
        return {
            "participation_ratio": 0.0,
            "effective_rank": 0.0,
            "n_components_90pct": 0,
            "n_components_99pct": 0,
            "allocated_dim": int(x.shape[1]),
            "eigenvalues": [],
        }

    p = eig / total
    nz = p[p > 1e-12]
    return {
        "participation_ratio": float(total**2 / (eig**2).sum()),
        "effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
        "n_components_90pct": int(np.searchsorted(np.cumsum(p), 0.90) + 1),
        "n_components_99pct": int(np.searchsorted(np.cumsum(p), 0.99) + 1),
        "allocated_dim": int(x.shape[1]),
        "eigenvalues": [float(v) for v in eig],
    }


# --------------------------------------------------------------------------------------------
# E1 -- recovery with controls
# --------------------------------------------------------------------------------------------


@dataclass
class ProbeInputs:
    """Feature matrices for one probe arm, already split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def run_recovery(
    arms: Mapping[str, ProbeInputs],
    targets: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    shuffle_seed: int = 0,
) -> dict[str, Any]:
    """Probe every target from every arm, plus the shuffled-label chance floor.

    ``arms`` maps arm name -> features; ``targets`` maps target name -> (train, val, test) labels.
    """
    rng = np.random.default_rng(shuffle_seed)
    results: dict[str, Any] = {}

    for target_name, (y_tr, y_va, y_te) in targets.items():
        per_target: dict[str, Any] = {}
        for arm_name, feats in arms.items():
            per_target[arm_name] = ridge_probe(
                feats.train, y_tr, feats.val, y_va, feats.test, y_te, alphas
            )

        # Chance floor: permute the training labels so any structure the probe finds is spurious.
        ref = next(iter(arms.values()))
        perm = rng.permutation(len(y_tr))
        per_target["shuffled_labels"] = ridge_probe(
            ref.train, y_tr[perm], ref.val, y_va, ref.test, y_te, alphas
        )
        results[target_name] = per_target

    return results


def format_recovery(results: Mapping[str, Any], arm_order: Sequence[str]) -> str:
    kinds = {s.name: s.kind for s in TARGET_SPECS}
    header = f"{'target':<16}{'kind':<15}" + "".join(f"{a:>16}" for a in arm_order)
    lines = [header, "-" * len(header)]
    for target, per_arm in results.items():
        row = f"{target:<16}{kinds.get(target, '?'):<15}"
        for arm in arm_order:
            row += f"{per_arm[arm]['r2']:>16.4f}" if arm in per_arm else f"{'--':>16}"
        lines.append(row)
    return "\n".join(lines)
