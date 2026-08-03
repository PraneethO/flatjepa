"""The assembled JEPA core (F5): encoders + latent predictor + combined objective.

::

    ℒ_total = ℒ_pred + λ_sig · ℒ_SIGReg
    ℒ_pred  = (1/T) Σ_k ‖ s̃_{t+k} − Enc_θ(x_{t+k}) ‖²

Two structural points from F5 §1 that this module implements literally:

* **One encoder, two roles.**  ``Enc_θ`` produces both the context latent and the prediction
  targets.  There is no EMA copy and no stop-gradient — gradients flow through both branches, which
  is exactly why ``ℒ_pred`` alone admits the trivial solution ``Enc_θ ≡ c`` (loss exactly zero) and
  why SIGReg is load-bearing rather than decorative.
* **Latent width is configurable.**  Nothing here hard-codes 24; ``JEPAConfig.latent_dim`` is the
  swept quantity in E2.

Target construction (a choice F5 leaves open).  ``Enc_θ`` consumes an ``H``-step window, but
``state_future`` is a sequence of individual future states.  The targets are therefore formed by
sliding the ``H``-step window forward over ``concat(state_hist, state_future)``: target ``k`` is
``Enc_θ`` applied to the window *ending* at ``t+k``.  This keeps context and target windows
statistically identical (same encoder, same window length), which a single-step target would not.
The alternative — encoding each future state alone — would make the target distribution differ from
the context distribution and confound the collapse diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .encoders import ActionEncoder, ActionEncoderConfig, StateEncoder, StateEncoderConfig
from .predictor import GRUPredictor, PredictorConfig
from .sigreg import SIGReg, SIGRegConfig

__all__ = ["JEPAConfig", "FlatJEPA"]


@dataclass
class JEPAConfig:
    """Full specification of the JEPA core.  Every field is expected to come from a YAML run
    config (F8 §2): no experiment-relevant constant should live in code.

    Attributes
    ----------
    latent_dim:
        Allocated latent width.  **Swept** — E2 compares allocated width against the *effective*
        dimensionality, whose theoretical minimum is ~9.
    lambda_sig:
        Weight on the SIGReg penalty.  Swept including exactly ``0.0``, the run F5 §3 expects to
        collapse.
    sigreg_on:
        Which latents the isotropy penalty sees: ``"encoder"`` (context + targets, the default),
        ``"targets"``, ``"context"``, or ``"all"`` (encoder outputs plus predicted latents).
    """

    state_dim: int = 9
    action_dim: int = 3
    latent_dim: int = 24
    action_embed_dim: int = 8
    history: int = 10
    horizon: int = 20
    state_channels: tuple[int, ...] = (8, 8, 16)
    action_channels: tuple[int, ...] = (4, 4, 8)
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.0
    norm: bool = True
    predictor_hidden_dim: int | None = None
    predictor_layers: int = 1
    lambda_sig: float = 1.0
    sigreg_on: str = "encoder"
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)

    def __post_init__(self) -> None:
        if self.sigreg_on not in {"encoder", "targets", "context", "all"}:
            raise ValueError(f"unknown sigreg_on={self.sigreg_on!r}")
        if isinstance(self.sigreg, dict):  # tolerate a raw YAML mapping
            self.sigreg = SIGRegConfig(**self.sigreg)
        if self.history < 1 or self.horizon < 1:
            raise ValueError("history and horizon must be >= 1")


class FlatJEPA(nn.Module):
    """State encoder + action encoder + GRU predictor + combined loss."""

    def __init__(self, config: JEPAConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or JEPAConfig()
        if overrides:
            cfg = JEPAConfig(**{**cfg.__dict__, **overrides})
        self.config = cfg

        self.state_encoder = StateEncoder(
            StateEncoderConfig(
                state_dim=cfg.state_dim,
                latent_dim=cfg.latent_dim,
                channels=tuple(cfg.state_channels),
                kernel_size=cfg.kernel_size,
                dilation_base=cfg.dilation_base,
                dropout=cfg.dropout,
                norm=cfg.norm,
            )
        )
        self.action_encoder = ActionEncoder(
            ActionEncoderConfig(
                action_dim=cfg.action_dim,
                embed_dim=cfg.action_embed_dim,
                channels=tuple(cfg.action_channels),
                kernel_size=cfg.kernel_size,
                dilation_base=cfg.dilation_base,
                dropout=cfg.dropout,
                norm=cfg.norm,
            )
        )
        self.predictor = GRUPredictor(
            PredictorConfig(
                latent_dim=cfg.latent_dim,
                action_embed_dim=cfg.action_embed_dim,
                hidden_dim=cfg.predictor_hidden_dim,
                num_layers=cfg.predictor_layers,
            )
        )
        self.sigreg = SIGReg(cfg.sigreg)

    # ------------------------------------------------------------------ encoding

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    def target_windows(self, state_hist: torch.Tensor, state_future: torch.Tensor) -> torch.Tensor:
        """Sliding ``H``-step windows ending at ``t+1 … t+T``.

        ``(B, H, D), (B, T, D) -> (B, T, H, D)``.
        """
        h = state_hist.shape[1]
        seq = torch.cat([state_hist, state_future], dim=1)  # (B, H+T, D)
        windows = seq.unfold(dimension=1, size=h, step=1)  # (B, T+1, D, H)
        return windows.permute(0, 1, 3, 2)[:, 1:].contiguous()  # drop window ending at t

    def encode_context(self, state_hist: torch.Tensor) -> torch.Tensor:
        """``(B, H, D) -> (B, latent_dim)``."""
        return self.state_encoder(state_hist)

    def encode_targets(self, state_hist: torch.Tensor, state_future: torch.Tensor) -> torch.Tensor:
        """``(B, H, D), (B, T, D) -> (B, T, latent_dim)``."""
        return self.state_encoder(self.target_windows(state_hist, state_future))

    def rollout(
        self,
        state_hist: torch.Tensor,
        action_hist: torch.Tensor,
        action_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ``T`` latents from history and the teacher-forced action sequence.

        Returns ``(context, predicted_latents)`` with shapes ``(B, L)`` and ``(B, T, L)``.
        """
        context = self.encode_context(state_hist)
        action_emb = self.action_encoder(action_hist, action_future)
        return context, self.predictor(context, action_emb)

    # ------------------------------------------------------------------ objective

    def _sigreg_input(
        self, context: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor
    ) -> torch.Tensor:
        mode = self.config.sigreg_on
        latent_dim = targets.shape[-1]
        if mode == "context":
            return context
        if mode == "targets":
            return targets.reshape(-1, latent_dim)
        parts = [context, targets.reshape(-1, latent_dim)]
        if mode == "all":
            parts.append(predictions.reshape(-1, latent_dim))
        return torch.cat(parts, dim=0)

    def forward(
        self,
        state_hist: torch.Tensor,
        action_hist: torch.Tensor,
        state_future: torch.Tensor,
        action_future: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute the full training objective.

        Parameters
        ----------
        state_hist:
            ``(B, H, state_dim)``.
        action_hist:
            ``(B, H, action_dim)``.
        state_future:
            ``(B, T, state_dim)`` ground-truth future states (targets for ``Enc_θ``).
        action_future:
            ``(B, T, action_dim)``; entry ``k`` is the jerk applied during step ``k`` of the
            rollout.
        target_mask:
            Optional ``(B, T)`` float/bool mask of valid horizon steps (F4's ``valid_mask``
            restricted to the future).  Masked steps are excluded from ``ℒ_pred``.

        Returns
        -------
        dict with ``loss``, ``loss_pred``, ``loss_sigreg``, ``per_step_loss`` ``(T,)``, and the
        ``context``/``target_latents``/``predicted_latents`` tensors for diagnostics.
        """
        self._check_shapes(state_hist, action_hist, state_future, action_future)
        context, predictions = self.rollout(state_hist, action_hist, action_future)
        targets = self.encode_targets(state_hist, state_future)

        sq_err = (predictions - targets).pow(2).sum(dim=-1)  # (B, T)
        if target_mask is None:
            per_step = sq_err.mean(dim=0)
            loss_pred = sq_err.mean()
        else:
            mask = target_mask.to(sq_err.dtype)
            denom_step = mask.sum(dim=0).clamp_min(1.0)
            per_step = (sq_err * mask).sum(dim=0) / denom_step
            loss_pred = (sq_err * mask).sum() / mask.sum().clamp_min(1.0)

        sigreg_input = self._sigreg_input(context, targets, predictions)
        lam = float(self.config.lambda_sig)
        if lam == 0.0:
            # Still logged (F8 §4 wants the penalty reported separately) but kept out of the graph:
            # the λ=0 run is a real experimental arm, not a disabled feature.
            with torch.no_grad():
                loss_sigreg = self.sigreg(sigreg_input)
            loss_total = loss_pred
        else:
            loss_sigreg = self.sigreg(sigreg_input)
            loss_total = loss_pred + lam * loss_sigreg

        return {
            "loss": loss_total,
            "loss_pred": loss_pred,
            "loss_sigreg": loss_sigreg,
            "per_step_loss": per_step,
            "context": context,
            "target_latents": targets,
            "predicted_latents": predictions,
        }

    # ------------------------------------------------------------------ helpers

    def _check_shapes(
        self,
        state_hist: torch.Tensor,
        action_hist: torch.Tensor,
        state_future: torch.Tensor,
        action_future: torch.Tensor,
    ) -> None:
        cfg = self.config
        if state_hist.shape[1:] != (cfg.history, cfg.state_dim):
            raise ValueError(
                f"state_hist should be (B, {cfg.history}, {cfg.state_dim}), got "
                f"{tuple(state_hist.shape)}"
            )
        if action_hist.shape[1:] != (cfg.history, cfg.action_dim):
            raise ValueError(
                f"action_hist should be (B, {cfg.history}, {cfg.action_dim}), got "
                f"{tuple(action_hist.shape)}"
            )
        if state_future.shape[1:] != (cfg.horizon, cfg.state_dim):
            raise ValueError(
                f"state_future should be (B, {cfg.horizon}, {cfg.state_dim}), got "
                f"{tuple(state_future.shape)}"
            )
        if action_future.shape[1:] != (cfg.horizon, cfg.action_dim):
            raise ValueError(
                f"action_future should be (B, {cfg.horizon}, {cfg.action_dim}), got "
                f"{tuple(action_future.shape)}"
            )

    def set_sigreg_generator(self, generator: torch.Generator | None) -> None:
        """Route SIGReg's direction sampling through a dedicated RNG stream."""
        self.sigreg.set_generator(generator)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Parameter count — reported because F5 §5 makes model scale a deliberate choice."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)
