"""Latent predictor for the JEPA core (F5 §1).

``Pred_ψ`` is a single-layer GRU unrolled ``T`` steps, consuming one action embedding per step::

    s̃_{t+k} = Pred(s̃_{t+k−1}, z_{t+k−1})

The GRU hidden state *is* the latent by default, which is why the hidden width follows
``latent_dim`` rather than being pinned to SkyJEPA's 24 (F5 §1: the latent width is a swept
parameter and hard-coding it would prejudge E2).  A separate ``hidden_dim`` is still allowed, in
which case linear maps carry the context into hidden space and the hidden state back out to the
latent space; this keeps capacity and latent width independently controllable for the E2 sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["PredictorConfig", "GRUPredictor"]


@dataclass
class PredictorConfig:
    """Configuration for :class:`GRUPredictor`.

    Attributes
    ----------
    latent_dim:
        Width of the latent the predictor consumes and emits (swept; see F5 §1).
    action_embed_dim:
        Width of the per-step action embedding from ``Enc_φ`` (F5 default 8).
    hidden_dim:
        GRU hidden width.  ``None`` means "equal to ``latent_dim``", the default in which the
        hidden state is literally the predicted latent and no projections are used.
    num_layers:
        GRU depth.  F5 specifies a single layer; deeper stacks are allowed for capacity ablations.
    """

    latent_dim: int = 24
    action_embed_dim: int = 8
    hidden_dim: int | None = None
    num_layers: int = 1
    dropout: float = 0.0


class GRUPredictor(nn.Module):
    """T-step unrolled GRU over latent states, teacher-forced on dataset actions.

    Because the action sequence comes from the dataset (F5 §2), the whole rollout can be evaluated
    with a single ``nn.GRU`` call: feeding the ``T`` action embeddings as a sequence with the
    context latent as ``h_0`` is exactly the recursion above, and ``output[:, k]`` is ``s̃_{t+k+1}``.
    :meth:`step` exposes the same weights one step at a time for autoregressive inference.
    """

    def __init__(self, config: PredictorConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or PredictorConfig()
        if overrides:
            cfg = PredictorConfig(**{**cfg.__dict__, **overrides})
        if cfg.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.config = cfg
        self.hidden_dim = cfg.hidden_dim if cfg.hidden_dim is not None else cfg.latent_dim
        self.gru = nn.GRU(
            input_size=cfg.action_embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        same = self.hidden_dim == cfg.latent_dim
        self.input_proj: nn.Module = nn.Identity() if same else nn.Linear(cfg.latent_dim, self.hidden_dim)
        self.output_proj: nn.Module = nn.Identity() if same else nn.Linear(self.hidden_dim, cfg.latent_dim)

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    def init_hidden(self, context: torch.Tensor) -> torch.Tensor:
        """``(B, latent_dim) -> (num_layers, B, hidden_dim)``."""
        if context.dim() != 2:
            raise ValueError(f"expected context of shape (B, latent_dim), got {tuple(context.shape)}")
        h = self.input_proj(context)
        return h.unsqueeze(0).expand(self.config.num_layers, -1, -1).contiguous()

    def forward(
        self, context: torch.Tensor, action_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Unroll ``T`` steps.

        Parameters
        ----------
        context:
            ``(B, latent_dim)`` latent at time ``t`` from ``Enc_θ``.
        action_embeddings:
            ``(B, T, action_embed_dim)``; entry ``k`` drives the transition into ``s̃_{t+k+1}``.

        Returns
        -------
        ``(B, T, latent_dim)`` predicted latents ``s̃_{t+1} … s̃_{t+T}``.
        """
        if action_embeddings.dim() != 3:
            raise ValueError(
                f"expected actions of shape (B, T, A), got {tuple(action_embeddings.shape)}"
            )
        if action_embeddings.shape[-1] != self.config.action_embed_dim:
            raise ValueError(
                f"expected action embedding dim {self.config.action_embed_dim}, "
                f"got {action_embeddings.shape[-1]}"
            )
        out, _ = self.gru(action_embeddings, self.init_hidden(context))
        return self.output_proj(out)

    def step(
        self, latent: torch.Tensor, action_embedding: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single recursion step, sharing weights with :meth:`forward`.

        Returns ``(next_latent, hidden)`` where ``hidden`` is ``(num_layers, B, hidden_dim)``.
        Pass the returned hidden state back in for multi-layer correctness; with the default single
        layer, ``hidden`` is redundant with ``next_latent``.
        """
        h = self.init_hidden(latent) if hidden is None else hidden
        out, h_next = self.gru(action_embedding.unsqueeze(1), h)
        return self.output_proj(out[:, 0]), h_next
