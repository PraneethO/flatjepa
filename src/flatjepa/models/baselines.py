"""Baselines (F10).

These exist to keep the main result honest. Three in particular:

* **Constant-jerk extrapolation** -- no learning at all. On smooth, noise-free optimizer output at
  10 Hz this is expected to be *strong* over short horizons. If it is not reported, short-horizon
  numbers look far more impressive than they are.
* **Direct state regression** -- the same encoder trained to predict future *states* rather than
  future latents. Isolates what latent-space prediction contributes, which is JEPA's core claim.
* **Observation-space GRU** -- the classical recurrent alternative.

The two learned baselines deliberately share the JEPA's encoder architecture, so a comparison
isolates the objective rather than confounding it with capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flatjepa.models.encoders import ActionEncoder, ActionEncoderConfig, StateEncoder, StateEncoderConfig


def constant_jerk_rollout(state_hist: torch.Tensor, action_future: torch.Tensor, dt: float) -> torch.Tensor:
    """Exact triple-integrator rollout using the given jerks. No learned parameters.

    ``state_hist`` (B, H, >=9) and ``action_future`` (B, T, 3) must be in **physical units**; the
    identities integrated here do not hold under per-channel normalisation.
    """
    p = state_hist[:, -1, 0:3]
    v = state_hist[:, -1, 3:6]
    a = state_hist[:, -1, 6:9]
    out = []
    for k in range(action_future.shape[1]):
        u = action_future[:, k]
        p = p + v * dt + 0.5 * a * dt**2 + u * dt**3 / 6.0
        v = v + a * dt + 0.5 * u * dt**2
        a = a + u * dt
        out.append(torch.cat([p, v, a], dim=-1))
    return torch.stack(out, dim=1)


def zero_order_hold(state_hist: torch.Tensor, horizon: int) -> torch.Tensor:
    """Repeat the last observed state. The weakest possible baseline, and a useful floor."""
    return state_hist[:, -1:, :9].expand(-1, horizon, -1).contiguous()


@dataclass
class DirectRegressorConfig:
    state_dim: int = 9
    action_dim: int = 3
    latent_dim: int = 24
    action_embed_dim: int = 8
    horizon: int = 20
    state_channels: tuple[int, ...] = (8, 8, 16)
    action_channels: tuple[int, ...] = (4, 4, 8)
    hidden: int = 128


class DirectStateRegressor(nn.Module):
    """Same encoder as the JEPA, but trained to emit future states directly.

    This is the ablation that isolates latent-space prediction: identical capacity, identical
    inputs, different objective.
    """

    def __init__(self, config: DirectRegressorConfig | None = None):
        super().__init__()
        cfg = config or DirectRegressorConfig()
        self.config = cfg
        self.state_encoder = StateEncoder(
            StateEncoderConfig(
                state_dim=cfg.state_dim, latent_dim=cfg.latent_dim, channels=tuple(cfg.state_channels)
            )
        )
        self.action_encoder = ActionEncoder(
            ActionEncoderConfig(
                action_dim=cfg.action_dim,
                embed_dim=cfg.action_embed_dim,
                channels=tuple(cfg.action_channels),
            )
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_embed_dim * cfg.horizon, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.horizon * cfg.state_dim),
        )

    def forward(
        self,
        state_hist: torch.Tensor,
        action_hist: torch.Tensor,
        action_future: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _ = action_future.shape
        ctx = self.state_encoder(state_hist)  # (B, latent)
        # Encode each future action step with the same action encoder, using a length-1 window.
        acts = torch.stack(
            [self.action_encoder(action_future[:, k : k + 1]) for k in range(t)], dim=1
        )  # (B, T, embed)
        z = torch.cat([ctx, acts.reshape(b, -1)], dim=-1)
        return self.head(z).view(b, t, self.config.state_dim)


class ObservationGRU(nn.Module):
    """Classical recurrent baseline: GRU over the observation sequence, decoding states directly."""

    def __init__(self, state_dim: int = 9, action_dim: int = 3, hidden: int = 64, horizon: int = 20):
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.encoder = nn.GRU(state_dim + action_dim, hidden, batch_first=True)
        self.cell = nn.GRUCell(action_dim, hidden)
        self.head = nn.Linear(hidden, state_dim)

    def forward(
        self,
        state_hist: torch.Tensor,
        action_hist: torch.Tensor,
        action_future: torch.Tensor,
    ) -> torch.Tensor:
        _, h = self.encoder(torch.cat([state_hist, action_hist], dim=-1))
        h = h[-1]
        out = []
        for k in range(action_future.shape[1]):
            h = self.cell(action_future[:, k], h)
            out.append(self.head(h))
        return torch.stack(out, dim=1)
