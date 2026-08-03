"""flatjepa.models — JEPA core (F5) and the physics-inspired prober (F6)."""

from .diagnostics import (
    CollapseAlarm,
    CollapseAlarmResult,
    LatentDiagnostics,
    effective_rank,
    latent_diagnostics,
    participation_ratio,
)
from .encoders import (
    ActionEncoder,
    ActionEncoderConfig,
    StateEncoder,
    StateEncoderConfig,
    TCNEncoder,
)
from .jepa import FlatJEPA, JEPAConfig
from .predictor import GRUPredictor, PredictorConfig
from .prober import (
    PhysicalParams,
    PhysicsProber,
    ProberConfig,
    assert_timestep,
    integrate_attitude,
    orthonormalize,
    so3_exp,
    so3_hat,
)
from .sigreg import SIGReg, SIGRegConfig, epps_pulley_statistic, sigreg_loss

__all__ = [
    "ActionEncoder",
    "ActionEncoderConfig",
    "CollapseAlarm",
    "CollapseAlarmResult",
    "FlatJEPA",
    "GRUPredictor",
    "JEPAConfig",
    "LatentDiagnostics",
    "PhysicalParams",
    "PhysicsProber",
    "PredictorConfig",
    "ProberConfig",
    "SIGReg",
    "SIGRegConfig",
    "StateEncoder",
    "StateEncoderConfig",
    "TCNEncoder",
    "assert_timestep",
    "effective_rank",
    "epps_pulley_statistic",
    "integrate_attitude",
    "latent_diagnostics",
    "orthonormalize",
    "participation_ratio",
    "sigreg_loss",
    "so3_exp",
    "so3_hat",
]
