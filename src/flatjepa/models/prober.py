"""Physics-inspired prober (F6): decode latents to metric state by differentiable integration.

The prober does not regress state directly.  It propagates the *known* nominal dynamics — the
triple integrator the planner itself uses (``ṗ = v``, ``v̇ = a``, ``ȧ = u``) — and lets the network
supply only a residual jerk correction ``Δu_k`` on top:

.. code-block:: text

    p_{k+1} = p_k + v_k·Δt
    v_{k+1} = v_k + a_k·Δt
    a_{k+1} = a_k + (u_k + Δu_k)·Δt        ← Δu_k is the learned residual

Everything is torch, so the whole rollout is differentiable end to end.

Two implementation points worth stating plainly:

**Integrator.**  The equations above are explicit Euler.  Under a zero-order hold on jerk — which is
what a jerk-parameterized planner actually produces between samples — the triple integrator has a
closed-form exact discrete solution::

    p_{k+1} = p_k + v_k·Δt + a_k·Δt²/2 + u_k·Δt³/6
    v_{k+1} = v_k + a_k·Δt + u_k·Δt²/2
    a_{k+1} = a_k + u_k·Δt

This is the *same* dynamics, integrated without truncation error, and it is the default
(``integrator="exact"``).  It matters for F6 §6's first acceptance criterion: Euler cannot reproduce
a continuous-time triple-integrator trajectory "to numerical precision" at Δt = 50 ms — it is off by
``O(Δt²)`` per step, which at 20 Hz is a visible position error that the residual head would then
happily learn to absorb, silently failing the E5-a zero-residual check for a reason that has nothing
to do with physics.  ``integrator="euler"`` reproduces the doc's literal equations when wanted.

**Timestep.**  ``dt`` is the F4 *resampled* period (20 Hz → 0.05 s), never the 500 Hz raw CSV period
(0.002 s).  F6 §5 calls a silent mismatch here exactly the bug E5-a exists to catch, so it is
checked at construction rather than assumed: out-of-band values raise, and if the resampling
provenance (``source_rate_hz``, ``stride``) is supplied it must agree with ``dt``.

Physical constants (``mQ``, ``mL``, ``L``, ``g``) are required inputs read from trajectory params —
:class:`PhysicalParams` has no defaults, so a missing constant is an error, not a silent 9.81.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "PhysicalParams",
    "ProberConfig",
    "PhysicsProber",
    "assert_timestep",
    "so3_hat",
    "so3_exp",
    "orthonormalize",
    "integrate_attitude",
]


# --------------------------------------------------------------------------- physical constants


@dataclass(frozen=True)
class PhysicalParams:
    """Constants of the quadrotor + cable-suspended payload system.

    Deliberately without defaults: F6 §5 requires these to be read from the trajectory params of
    the data being probed, never hard-coded.  Construct with
    ``PhysicalParams(mQ=..., mL=..., L=..., g=...)``.
    """

    mQ: float  # quadrotor mass [kg]
    mL: float  # payload mass [kg]
    L: float  # cable length [m]
    g: float  # gravitational acceleration magnitude [m/s²]

    def __post_init__(self) -> None:
        for name in ("mQ", "mL", "L", "g"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"PhysicalParams.{name} must be a positive number, got {value!r}")

    @classmethod
    def from_mapping(cls, params: dict[str, Any]) -> "PhysicalParams":
        """Build from a trajectory-params mapping, requiring every key to be present."""
        missing = [k for k in ("mQ", "mL", "L", "g") if k not in params]
        if missing:
            raise KeyError(f"trajectory params missing physical constants: {missing}")
        return cls(mQ=float(params["mQ"]), mL=float(params["mL"]), L=float(params["L"]),
                   g=float(params["g"]))

    def as_dict(self) -> dict[str, float]:
        return {"mQ": self.mQ, "mL": self.mL, "L": self.L, "g": self.g}


def assert_timestep(
    dt: float,
    source_rate_hz: float | None = None,
    stride: int | None = None,
    bounds: tuple[float, float] = (0.01, 0.5),
    rtol: float = 1e-6,
) -> float:
    """Validate the integration timestep against the F4 resampling provenance.

    Raises ``ValueError`` when ``dt`` is outside ``bounds`` (the default band admits 2–100 Hz and
    rejects the 500 Hz raw-CSV period of 0.002 s), or when ``dt`` disagrees with
    ``stride / source_rate_hz``.
    """
    if not isinstance(dt, (int, float)) or dt <= 0:
        raise ValueError(f"dt must be a positive number, got {dt!r}")
    lo, hi = bounds
    if not (lo <= dt <= hi):
        raise ValueError(
            f"dt={dt} s is outside the expected band [{lo}, {hi}] s. The prober integrates at the "
            f"F4 *resampled* rate (default 20 Hz -> dt=0.05 s); dt=0.002 s would be the raw 500 Hz "
            f"CSV period, which is the mismatch F6 §5 warns about. Pass explicit bounds if this "
            f"rate is intended."
        )
    if source_rate_hz is not None and stride is not None:
        expected = stride / float(source_rate_hz)
        if abs(dt - expected) > rtol * max(expected, 1e-12):
            raise ValueError(
                f"dt={dt} s contradicts the resampling provenance: stride={stride} at "
                f"{source_rate_hz} Hz implies dt={expected} s."
            )
    elif (source_rate_hz is None) != (stride is None):
        raise ValueError("source_rate_hz and stride must be given together, or not at all")
    return float(dt)


# --------------------------------------------------------------------------- SO(3) utilities


def so3_hat(w: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix of a ``(..., 3)`` vector; returns ``(..., 3, 3)``."""
    if w.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) vector, got {tuple(w.shape)}")
    zero = torch.zeros_like(w[..., 0])
    row0 = torch.stack([zero, -w[..., 2], w[..., 1]], dim=-1)
    row1 = torch.stack([w[..., 2], zero, -w[..., 0]], dim=-1)
    row2 = torch.stack([-w[..., 1], w[..., 0], zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def so3_exp(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SO(3) exponential map (Rodrigues) of a ``(..., 3)`` rotation vector.

    Small angles use the Taylor series and — importantly — the large-angle branch is evaluated on a
    *sanitized* angle, so ``torch.where`` does not propagate NaN gradients through the unused
    branch at ``‖w‖ = 0``.
    """
    theta2 = (w**2).sum(dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)
    small = theta2 < eps
    theta = torch.sqrt(torch.where(small, torch.ones_like(theta2), theta2))
    sinc = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    cosc = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(eps))
    k = so3_hat(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(k)
    return eye + sinc * k + cosc * (k @ k)


def orthonormalize(r: torch.Tensor) -> torch.Tensor:
    """Re-orthonormalize ``(..., 3, 3)`` matrices by modified Gram–Schmidt.

    Gram–Schmidt rather than an SVD projection because its gradients stay well behaved; SVD
    backward is ill-conditioned when singular values are close, which they are for any matrix that
    is already nearly a rotation.
    """
    c0 = r[..., :, 0]
    c1 = r[..., :, 1]
    c0 = c0 / c0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    c1 = c1 - (c1 * c0).sum(dim=-1, keepdim=True) * c0
    c1 = c1 / c1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    c2 = torch.cross(c0, c1, dim=-1)
    return torch.stack([c0, c1, c2], dim=-1)


def integrate_attitude(
    r0: torch.Tensor, body_rates: torch.Tensor, dt: float, reorthonormalize_every: int = 1
) -> torch.Tensor:
    """Propagate attitude on SO(3) via the exponential map (F6 §5).

    ``R_{k+1} = R_k · exp(ω_k Δt)`` — never componentwise integration of a rotation matrix.

    Parameters
    ----------
    r0:
        ``(B, 3, 3)`` initial rotation.
    body_rates:
        ``(B, T, 3)`` body-frame angular velocities held constant over each step.

    Returns
    -------
    ``(B, T, 3, 3)`` rotations at steps ``1 … T``.
    """
    if r0.dim() != 3 or r0.shape[-2:] != (3, 3):
        raise ValueError(f"expected r0 of shape (B, 3, 3), got {tuple(r0.shape)}")
    if body_rates.dim() != 3 or body_rates.shape[-1] != 3:
        raise ValueError(f"expected body_rates of shape (B, T, 3), got {tuple(body_rates.shape)}")
    r = r0
    out = []
    for k in range(body_rates.shape[1]):
        r = r @ so3_exp(body_rates[:, k] * dt)
        if reorthonormalize_every > 0 and (k + 1) % reorthonormalize_every == 0:
            r = orthonormalize(r)
        out.append(r)
    return torch.stack(out, dim=1)


# --------------------------------------------------------------------------- the prober


@dataclass
class ProberConfig:
    """Configuration for :class:`PhysicsProber`.

    Attributes
    ----------
    dt:
        Integration timestep in seconds — the F4 resampled period, validated at construction.
    source_rate_hz, stride:
        Optional resampling provenance; when both are given, ``dt`` must equal ``stride /
        source_rate_hz``.
    integrator:
        ``"exact"`` (zero-order-hold-exact triple integrator, default) or ``"euler"``.
    residual_hidden:
        Widths of the residual MLP's hidden layers.
    residual_scale:
        Fixed multiplier on the residual head output; keeps the correction small at initialization
        without constraining what it can eventually represent.
    enable_residual:
        When ``False`` the prober is pure nominal physics — the configuration used for the
        integrator correctness test.
    use_state_feedback:
        Feed the current integrated ``(p, v, a)`` into the residual head as well as the latent.
    """

    latent_dim: int = 24
    action_dim: int = 3
    dt: float = 0.05
    source_rate_hz: float | None = None
    stride: int | None = None
    dt_bounds: tuple[float, float] = (0.01, 0.5)
    integrator: str = "exact"
    residual_hidden: tuple[int, ...] = (64, 64)
    residual_scale: float = 1.0
    enable_residual: bool = True
    use_state_feedback: bool = False
    zero_init_residual: bool = True
    init_state_hidden: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.integrator not in {"exact", "euler"}:
            raise ValueError(f"unknown integrator {self.integrator!r}; expected 'exact' or 'euler'")


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers += [nn.Linear(prev, int(width)), nn.GELU()]
        prev = int(width)
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PhysicsProber(nn.Module):
    """Latent → metric payload state by nominal integration plus a learned residual jerk.

    Stage-two module (F8 §1): encoder and predictor are frozen and only this trains, on the
    supervised rollout loss ``ℒ_prober = (1/T) Σ_k ‖x̃_{t+k} − x_{t+k}‖²``.

    The metric state is the flat state ``x = (p, v, a) ∈ ℝ⁹`` of the payload, in the same order the
    dataset's ``sol_x`` uses.
    """

    STATE_DIM = 9

    def __init__(
        self,
        params: PhysicalParams,
        config: ProberConfig | None = None,
        **overrides,
    ) -> None:
        super().__init__()
        if not isinstance(params, PhysicalParams):
            raise TypeError(
                "params must be a PhysicalParams built from the trajectory params (F6 §5: "
                "physical constants are never hard-coded)"
            )
        cfg = config or ProberConfig()
        if overrides:
            cfg = ProberConfig(**{**cfg.__dict__, **overrides})
        self.params = params
        self.config = cfg
        self.dt = assert_timestep(cfg.dt, cfg.source_rate_hz, cfg.stride, cfg.dt_bounds)

        residual_in = cfg.latent_dim + cfg.action_dim + (self.STATE_DIM if cfg.use_state_feedback else 0)
        self.residual_head = _mlp(residual_in, tuple(cfg.residual_hidden), cfg.action_dim)
        self.init_state_head = _mlp(cfg.latent_dim, tuple(cfg.init_state_hidden), self.STATE_DIM)
        if cfg.zero_init_residual:
            last = self.residual_head[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    # ------------------------------------------------------------------ physics

    def _step(
        self,
        p: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One triple-integrator step under jerk ``u`` held over ``dt``."""
        dt = self.dt
        if self.config.integrator == "euler":
            return p + v * dt, v + a * dt, a + u * dt
        p_next = p + v * dt + a * (0.5 * dt**2) + u * (dt**3 / 6.0)
        v_next = v + a * dt + u * (0.5 * dt**2)
        return p_next, v_next, a + u * dt

    def integrate(
        self, init_state: torch.Tensor, jerk: torch.Tensor
    ) -> torch.Tensor:
        """Pure nominal rollout with no network involvement.

        ``(B, 9), (B, T, 3) -> (B, T, 9)``; step ``k`` of the output is the state at ``t+k+1``.
        """
        p, v, a = init_state[..., 0:3], init_state[..., 3:6], init_state[..., 6:9]
        states = []
        for k in range(jerk.shape[1]):
            p, v, a = self._step(p, v, a, jerk[:, k])
            states.append(torch.cat([p, v, a], dim=-1))
        return torch.stack(states, dim=1)

    def cable_direction_and_tension(
        self, payload_accel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cable unit vector (payload → quadrotor) and cable tension, from payload acceleration.

        Under the always-taut rigid-cable assumption, ``m_L(a + g e₃) = T q``, so the direction is
        that vector normalized and the tension is ``m_L‖a + g e₃‖``.  Uses ``mL`` and ``g`` from
        :class:`PhysicalParams`.  The *full* flat map (quadrotor attitude, payload attitude) is
        F2's responsibility; this is only the cable geometry the prober needs to express its own
        output in quadrotor coordinates.
        """
        gvec = torch.zeros_like(payload_accel)
        gvec[..., 2] = self.params.g
        f = self.params.mL * (payload_accel + gvec)
        tension = f.norm(dim=-1, keepdim=True)
        q = f / tension.clamp_min(1e-12)
        return q, tension.squeeze(-1)

    def quad_position(self, payload_position: torch.Tensor, payload_accel: torch.Tensor) -> torch.Tensor:
        """Quadrotor position implied by the payload state, ``p_Q = p_L + L·q``.  Uses ``L``."""
        q, _ = self.cable_direction_and_tension(payload_accel)
        return payload_position + self.params.L * q

    # ------------------------------------------------------------------ forward

    def residual(
        self, latents: torch.Tensor, jerk: torch.Tensor, state: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Residual jerk correction ``Δu`` for a batch of steps.

        ``(B, T, latent_dim), (B, T, 3) -> (B, T, 3)``.  Returns exact zeros when the residual is
        disabled, which is the configuration used to test the nominal integrator alone.
        """
        if not self.config.enable_residual:
            return torch.zeros_like(jerk)
        features = [latents, jerk]
        if self.config.use_state_feedback:
            if state is None:
                raise ValueError("use_state_feedback=True requires the current state")
            features.append(state)
        return self.config.residual_scale * self.residual_head(torch.cat(features, dim=-1))

    def forward(
        self,
        latents: torch.Tensor,
        jerk: torch.Tensor,
        init_state: torch.Tensor | None = None,
        init_latent: torch.Tensor | None = None,
        return_nominal: bool = True,
    ) -> dict[str, Any]:
        """Roll the prober out over the horizon.

        Parameters
        ----------
        latents:
            ``(B, T, latent_dim)`` frozen latents, one per horizon step (typically the JEPA's
            predicted latents ``s̃_{t+1..t+T}``).
        jerk:
            ``(B, T, 3)`` nominal control input ``u_k`` over the horizon.
        init_state:
            ``(B, 9)`` metric initial state ``(p, v, a)`` at time ``t``.  If omitted it is decoded
            from ``init_latent``.
        init_latent:
            ``(B, latent_dim)`` context latent used to decode the initial state when ``init_state``
            is not supplied.
        return_nominal:
            Also roll out the residual-free trajectory, so residual and nominal magnitudes can be
            logged separately (F6 §5, §6).

        Returns
        -------
        dict with ``states`` ``(B, T, 9)``, ``residual`` ``(B, T, 3)``, scalar
        ``residual_magnitude``/``nominal_magnitude`` (mean L2 norm of ``Δu`` and of ``u``), and,
        when requested, ``nominal_states`` and ``residual_displacement`` (‖states − nominal‖).
        """
        if latents.dim() != 3:
            raise ValueError(f"expected latents of shape (B, T, L), got {tuple(latents.shape)}")
        if jerk.shape[:2] != latents.shape[:2]:
            raise ValueError(
                f"latents {tuple(latents.shape)} and jerk {tuple(jerk.shape)} must agree on (B, T)"
            )
        if init_state is None:
            if init_latent is None:
                raise ValueError("provide either init_state or init_latent")
            init_state = self.init_state_head(init_latent)
        if init_state.shape[-1] != self.STATE_DIM:
            raise ValueError(
                f"init_state should have {self.STATE_DIM} channels (p, v, a), got "
                f"{init_state.shape[-1]}"
            )

        horizon = latents.shape[1]
        p, v, a = init_state[..., 0:3], init_state[..., 3:6], init_state[..., 6:9]
        states, residuals = [], []
        for k in range(horizon):
            state_k = torch.cat([p, v, a], dim=-1)
            delta_u = self.residual(
                latents[:, k : k + 1],
                jerk[:, k : k + 1],
                state_k.unsqueeze(1) if self.config.use_state_feedback else None,
            )[:, 0]
            residuals.append(delta_u)
            p, v, a = self._step(p, v, a, jerk[:, k] + delta_u)
            states.append(torch.cat([p, v, a], dim=-1))

        out: dict[str, Any] = {
            "states": torch.stack(states, dim=1),
            "residual": torch.stack(residuals, dim=1),
            "init_state": init_state,
        }
        out["residual_magnitude"] = out["residual"].norm(dim=-1).mean()
        out["nominal_magnitude"] = jerk.norm(dim=-1).mean()
        if return_nominal:
            nominal = self.integrate(init_state, jerk)
            out["nominal_states"] = nominal
            out["residual_displacement"] = (out["states"] - nominal).norm(dim=-1).mean()
        return out

    @staticmethod
    def rollout_loss(
        predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``ℒ_prober = (1/T) Σ_k ‖x̃_{t+k} − x_{t+k}‖²`` (F6 §1), optionally masked over steps."""
        sq = (predicted - target).pow(2).sum(dim=-1)  # (B, T)
        if mask is None:
            return sq.mean()
        m = mask.to(sq.dtype)
        return (sq * m).sum() / m.sum().clamp_min(1.0)
