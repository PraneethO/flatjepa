"""Temporal convolutional encoders for the JEPA core (F5 §1).

Two encoders are defined here:

* :class:`StateEncoder` — maps an ``H``-step state history window to a single latent vector
  ``s_t`` of configurable width.  Channels default to ``(8, 8, 16)`` per F5.
* :class:`ActionEncoder` — maps an action sequence to *per-step* embeddings ``z_k`` of width 8
  (F5 default channels ``(4, 4, 8)``), one per predictor unroll step.

Design notes
------------
**Causality.** Every convolution is left-padded, so the output at time index ``i`` depends only on
inputs ``<= i``.  This matters for the action encoder, whose per-step outputs feed the predictor
one step at a time: a non-causal encoder would leak future actions into earlier embeddings and
quietly make the T-step rollout easier than it should be.  For the same reason normalization is
applied over the *channel* axis only (:class:`ChannelLayerNorm`), never over time — a plain
``GroupNorm``/``LayerNorm`` over ``(C, L)`` would mix statistics across the time axis and break
causality in a way that is invisible in shape tests.

**Latent width is a parameter, not a constant.** F5 §1 explicitly refuses to hard-code the latent
width: E2 asks whether the *effective* dimensionality lands near the theoretical minimum of ~9, and
fixing the allocated width at 24 would prejudge the sweep.  ``latent_dim`` is therefore a required
constructor argument of :class:`StateEncoder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ChannelLayerNorm",
    "CausalConv1d",
    "TCNBlock",
    "TCNEncoder",
    "StateEncoder",
    "ActionEncoder",
    "StateEncoderConfig",
    "ActionEncoderConfig",
]


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a ``(B, C, L)`` tensor.

    Normalizing over ``(C, L)`` (what ``GroupNorm(1, C)`` does) would pool statistics across time
    and destroy causality; this normalizes each time step independently.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Module):
    """1-D convolution with left padding, so output length equals input length and no output
    position depends on a future input position."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    """One dilated causal conv + channel norm + activation, with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm = ChannelLayerNorm(out_channels) if norm else nn.Identity()
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.residual: nn.Module
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(self.act(self.norm(self.conv(x))))
        return y + self.residual(x)


class TCNEncoder(nn.Module):
    """Stack of dilated causal conv blocks with a linear head.

    Parameters
    ----------
    input_dim:
        Channels of the input sequence (last axis of a ``(B, L, input_dim)`` tensor).
    channels:
        Hidden channel widths, one entry per block.  Dilation for block ``i`` is
        ``dilation_base ** i``.
    output_dim:
        Width of the linear head applied to the block stack's output.
    """

    def __init__(
        self,
        input_dim: int,
        channels: Sequence[int],
        output_dim: int,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.0,
        norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if len(channels) == 0:
            raise ValueError("channels must contain at least one block width")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.channels = tuple(int(c) for c in channels)
        self.kernel_size = int(kernel_size)
        self.dilation_base = int(dilation_base)

        blocks = []
        prev = input_dim
        for i, width in enumerate(self.channels):
            blocks.append(
                TCNBlock(
                    prev,
                    int(width),
                    kernel_size=self.kernel_size,
                    dilation=self.dilation_base**i,
                    dropout=dropout,
                    norm=norm,
                )
            )
            prev = int(width)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(prev, self.output_dim)

    @property
    def receptive_field(self) -> int:
        """Number of past time steps (inclusive) the final output position can see."""
        return 1 + sum((self.kernel_size - 1) * self.dilation_base**i for i in range(len(self.channels)))

    def forward(self, x: torch.Tensor, pool: str = "last") -> torch.Tensor:
        """Encode a batch of sequences.

        Parameters
        ----------
        x:
            ``(B, L, input_dim)``.
        pool:
            ``"last"`` returns ``(B, output_dim)`` from the final time step; ``"sequence"``
            returns ``(B, L, output_dim)``, one embedding per (causal) time step.
        """
        if x.dim() != 3:
            raise ValueError(f"expected a (B, L, D) tensor, got shape {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected last dim {self.input_dim}, got {x.shape[-1]}")
        h = x.transpose(1, 2)  # (B, C, L)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)  # (B, L, C)
        if pool == "last":
            return self.head(h[:, -1])
        if pool == "sequence":
            return self.head(h)
        raise ValueError(f"unknown pool mode {pool!r}; expected 'last' or 'sequence'")


@dataclass
class StateEncoderConfig:
    """Configuration for :class:`StateEncoder`.  ``latent_dim`` is deliberately swept (F5 §1)."""

    state_dim: int = 9
    latent_dim: int = 24
    channels: tuple[int, ...] = (8, 8, 16)
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.0
    norm: bool = True


@dataclass
class ActionEncoderConfig:
    """Configuration for :class:`ActionEncoder` (F5 §1: channels ``(4, 4, 8)``, output dim 8)."""

    action_dim: int = 3
    embed_dim: int = 8
    channels: tuple[int, ...] = (4, 4, 8)
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.0
    norm: bool = True


class StateEncoder(nn.Module):
    """``Enc_θ``: H-step state history window → latent ``s_t``.

    The same module produces both the context latent and the prediction targets — F5 §1 is explicit
    that there is *no* EMA target encoder here, and that gradients flow through both branches, with
    collapse held off by SIGReg rather than a stop-gradient.
    """

    def __init__(self, config: StateEncoderConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or StateEncoderConfig()
        if overrides:
            cfg = StateEncoderConfig(**{**cfg.__dict__, **overrides})
        self.config = cfg
        self.tcn = TCNEncoder(
            input_dim=cfg.state_dim,
            channels=cfg.channels,
            output_dim=cfg.latent_dim,
            kernel_size=cfg.kernel_size,
            dilation_base=cfg.dilation_base,
            dropout=cfg.dropout,
            norm=cfg.norm,
        )

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    @property
    def receptive_field(self) -> int:
        return self.tcn.receptive_field

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """``(B, H, state_dim) -> (B, latent_dim)``.

        A ``(B, N, H, state_dim)`` input is also accepted and returns ``(B, N, latent_dim)``, which
        is how the T target windows are encoded in one pass.
        """
        if states.dim() == 4:
            b, n = states.shape[:2]
            flat = states.reshape(b * n, states.shape[2], states.shape[3])
            return self.tcn(flat, pool="last").reshape(b, n, self.config.latent_dim)
        return self.tcn(states, pool="last")


class ActionEncoder(nn.Module):
    """``Enc_φ``: action sequence → per-step action embeddings ``z_k``.

    F5 §1 says both encoders consume the full H-step history window, while the predictor consumes
    *one* action embedding per unroll step.  Those two statements are reconciled by running the
    causal TCN over the concatenated ``[action_hist, action_future]`` sequence and reading off the
    ``T`` positions belonging to the horizon: embedding ``z_k`` then summarizes the whole action
    history up to and including the action applied at rollout step ``k``, and (by causality) nothing
    after it.

    Convention: ``action_future[:, k]`` is the jerk applied during the transition
    ``s̃_{t+k} -> s̃_{t+k+1}``, so ``T`` future actions drive exactly ``T`` predictor steps.  F4/F5
    do not pin this down; it is fixed here and asserted by the shape tests.
    """

    def __init__(self, config: ActionEncoderConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or ActionEncoderConfig()
        if overrides:
            cfg = ActionEncoderConfig(**{**cfg.__dict__, **overrides})
        self.config = cfg
        self.tcn = TCNEncoder(
            input_dim=cfg.action_dim,
            channels=cfg.channels,
            output_dim=cfg.embed_dim,
            kernel_size=cfg.kernel_size,
            dilation_base=cfg.dilation_base,
            dropout=cfg.dropout,
            norm=cfg.norm,
        )

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim

    @property
    def receptive_field(self) -> int:
        return self.tcn.receptive_field

    def forward(
        self, action_hist: torch.Tensor, action_future: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return per-step action embeddings.

        Parameters
        ----------
        action_hist:
            ``(B, H, action_dim)`` past actions.
        action_future:
            ``(B, T, action_dim)`` actions over the prediction horizon.  If ``None``, embeddings for
            all ``H`` history positions are returned instead (useful for inspection).

        Returns
        -------
        ``(B, T, embed_dim)`` when ``action_future`` is given, else ``(B, H, embed_dim)``.
        """
        if action_future is None:
            return self.tcn(action_hist, pool="sequence")
        if action_hist.shape[0] != action_future.shape[0]:
            raise ValueError("action_hist and action_future must share a batch dimension")
        horizon = action_future.shape[1]
        seq = torch.cat([action_hist, action_future], dim=1)
        emb = self.tcn(seq, pool="sequence")
        return emb[:, -horizon:]
