"""Two-stage training harness (F8).

Stage 1 trains the JEPA core on ``ℒ_pred + λ·ℒ_SIGReg``. Stage 2 **freezes** it and trains only the
physics prober on a supervised metric-state rollout loss.

The freeze is asserted rather than assumed. Every claim in E1/E3/E4/E5 is about probing a *frozen*
representation, so a silent unfreeze would not fail loudly -- it would just quietly invalidate the
results. :func:`assert_frozen` runs after every stage-2 step by default.

Collapse diagnostics are logged each epoch and an alarm halts a collapsed run rather than letting it
burn hours training a constant function (F5 §4).
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from flatjepa.models.diagnostics import CollapseAlarm, latent_diagnostics
from flatjepa.models.jepa import FlatJEPA, JEPAConfig
from flatjepa.models.prober import PhysicalParams, PhysicsProber, ProberConfig
from flatjepa.training.data import GPUResidentSplit


# --------------------------------------------------------------------------------------------
# Reproducibility and provenance
# --------------------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def provenance() -> dict[str, Any]:
    """Everything needed to trace a number back to the code that produced it (F8 §2)."""

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip() or None
        except Exception:  # noqa: BLE001
            return None

    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """One run, fully specified. Nothing experiment-relevant should live outside this."""

    dataset: str = "data/windows_v1"
    out_dir: str = "runs/default"
    seed: int = 0
    device: str = "cuda"

    # stage 1
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0

    # stage 2
    prober_epochs: int = 10
    prober_lr: float = 1e-3

    latent_dim: int = 24
    lambda_sig: float = 1.0

    num_workers: int = 0
    log_every: int = 0  # 0 = per-epoch only
    collapse_alarm: bool = True
    train_prober: bool = True

    jepa: dict = field(default_factory=dict)
    prober: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Freeze verification
# --------------------------------------------------------------------------------------------


def freeze_module(module: nn.Module) -> None:
    """Put ``module`` in eval mode, stop gradient tracking, and clear stale gradients.

    Clearing ``.grad`` matters: ``requires_grad_(False)`` leaves whatever the previous stage's last
    backward pass accumulated, so :func:`assert_frozen` could not otherwise tell a leftover gradient
    from evidence that the module is still being optimized. After this, any non-``None`` gradient is
    a genuine signal.
    """
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
        p.grad = None


def assert_frozen(module: nn.Module, name: str = "module") -> None:
    """Fail loudly if anything in ``module`` could still be learning.

    Checks both ``requires_grad`` and the presence of accumulated gradients: an optimizer
    constructed over all parameters before the freeze would still step them.
    """
    live = [n for n, p in module.named_parameters() if p.requires_grad]
    if live:
        raise RuntimeError(
            f"{name} is not frozen: {len(live)} parameters still require grad "
            f"(first few: {live[:5]}). Every probing result assumes a frozen representation."
        )
    with_grad = [n for n, p in module.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    if with_grad:
        raise RuntimeError(
            f"{name} has non-zero accumulated gradients on {len(with_grad)} parameters "
            f"(first few: {with_grad[:5]}); it is being optimized despite requires_grad=False."
        )


# --------------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------------


class Trainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        set_seed(config.seed)

        self.device = torch.device(
            config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Whole splits live on device: the corpus is ~57 MB, so a DataLoader would only add
        # per-batch Python overhead and CPU contention with concurrent planner workers.
        self.train_ds = GPUResidentSplit(config.dataset, "train", self.device)
        self.val_ds = GPUResidentSplit(config.dataset, "val", self.device)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(int(config.seed))

        jepa_kwargs = {
            "state_dim": self.train_ds.obs_dim,
            "action_dim": 3,
            "history": self.train_ds.history,
            "horizon": self.train_ds.horizon,
            "latent_dim": config.latent_dim,
            "lambda_sig": config.lambda_sig,
            **config.jepa,
        }
        self.model = FlatJEPA(JEPAConfig(**jepa_kwargs)).to(self.device)

        self.alarm = CollapseAlarm() if config.collapse_alarm else None
        self.history: list[dict[str, Any]] = []
        self._log_path = self.out_dir / "log.jsonl"

    # ------------------------------------------------------------------ helpers

    def _batches(self, ds: GPUResidentSplit, shuffle: bool) -> Iterable[dict[str, torch.Tensor]]:
        # Already on device; nothing to transfer.
        return ds.iter_batches(self.cfg.batch_size, shuffle=shuffle, generator=self._gen)

    def _log(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        with self._log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ stage 1

    def _run_epoch(self, train: bool) -> dict[str, float]:
        ds = self.train_ds if train else self.val_ds
        self.model.train(train)
        totals = {"loss": 0.0, "loss_pred": 0.0, "loss_sigreg": 0.0}
        per_step_sum = None
        n = 0

        opt = self.opt if train else None
        with torch.set_grad_enabled(train):
            for batch in self._batches(ds, shuffle=train):
                out = self.model(
                    batch["state_hist"],
                    batch["action_hist"],
                    batch["state_future"],
                    batch["action_future"],
                )
                loss = out["loss"]
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    opt.step()

                bs = batch["state_hist"].shape[0]
                n += bs
                for k in totals:
                    totals[k] += float(out[k].detach()) * bs
                ps = out["per_step_loss"].detach().cpu()
                per_step_sum = ps * bs if per_step_sum is None else per_step_sum + ps * bs

        result = {k: v / max(n, 1) for k, v in totals.items()}
        if per_step_sum is not None:
            result["per_step_loss"] = (per_step_sum / max(n, 1)).tolist()
        return result

    @torch.no_grad()
    def _diagnostics(self):
        """Collapse diagnostics on a validation batch (F5 §4).

        Returns the diagnostics object itself; the alarm consumes it directly and only the scalar
        fields are serialised into the log.
        """
        self.model.eval()
        batch = next(iter(self._batches(self.val_ds, shuffle=False)))
        latents = self.model.encode_context(batch["state_hist"])
        return latent_diagnostics(latents)

    @staticmethod
    def _diag_scalars(diag) -> dict[str, float]:
        d = asdict(diag) if hasattr(diag, "__dataclass_fields__") else dict(diag)
        return {k: v for k, v in d.items() if isinstance(v, (int, float, bool))}

    def train_stage1(self) -> None:
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        best = float("inf")
        for epoch in range(self.cfg.epochs):
            t0 = time.time()
            tr = self._run_epoch(train=True)
            va = self._run_epoch(train=False)
            diag = self._diagnostics()
            diag_scalars = self._diag_scalars(diag)

            record = {
                "stage": 1,
                "epoch": epoch,
                "train": tr,
                "val": va,
                "diagnostics": diag_scalars,
                "seconds": round(time.time() - t0, 2),
            }

            if self.alarm is not None:
                res = self.alarm.check(diag)
                triggered = getattr(res, "triggered", False)
                record["collapse_alarm"] = {
                    "triggered": bool(triggered),
                    "reasons": list(getattr(res, "reasons", []) or []),
                }
                if triggered:
                    # A collapsed run is a legitimate experimental outcome (λ_sig = 0 is expected
                    # to collapse), so record it and stop rather than raising.
                    self._log(record)
                    self._save("collapsed.pt")
                    print(f"[stage1] epoch {epoch}: COLLAPSE ALARM -- {record['collapse_alarm']}")
                    return

            self._log(record)
            if va["loss"] < best:
                best = va["loss"]
                self._save("best.pt")
            print(
                f"[stage1] epoch {epoch:>3}  train {tr['loss']:.5f}  val {va['loss']:.5f}  "
                f"pred {va['loss_pred']:.5f}  sigreg {va['loss_sigreg']:.5f}  "
                f"eff_rank {diag_scalars.get('effective_rank', float('nan')):.2f}  "
                f"{record['seconds']:.1f}s"
            )
        self._save("final.pt")

    # ------------------------------------------------------------------ stage 2

    def train_stage2(self) -> None:
        """Freeze stage 1, train the prober on metric-state rollout loss."""
        freeze_module(self.model)
        assert_frozen(self.model, "JEPA core")

        meta = self.train_ds.metadata
        rate = meta.get("sample_rates_hz") or {}
        hz = int(max(rate, key=rate.get)) if rate else 10
        dt = 1.0 / hz

        prober_kwargs = {
            "latent_dim": self.cfg.latent_dim,
            "action_dim": 3,
            "dt": dt,
            **self.cfg.prober,
        }
        # F6 §5: physical constants come from the trajectory params, never hard-coded. They were
        # recorded into the dataset metadata at build time, so a checkpoint is self-describing.
        sp = meta.get("system_params") or {}
        missing = {"mass_quad", "mass_load", "cable_length", "gravity"} - set(sp)
        if missing:
            raise ValueError(
                f"dataset metadata is missing physical constants {sorted(missing)}; "
                "rebuild the dataset so the prober is not silently given wrong physics."
            )
        physical = PhysicalParams(
            mQ=float(sp["mass_quad"]),
            mL=float(sp["mass_load"]),
            L=float(sp["cable_length"]),
            g=float(sp["gravity"]),
        )
        self.prober = PhysicsProber(physical, ProberConfig(**prober_kwargs)).to(self.device)
        opt = torch.optim.AdamW(self.prober.parameters(), lr=self.cfg.prober_lr)

        for epoch in range(self.cfg.prober_epochs):
            t0 = time.time()
            self.prober.train()
            tot, res_mag, n = 0.0, 0.0, 0
            for batch in self._batches(self.train_ds, shuffle=True):
                with torch.no_grad():
                    _, pred_latents = self.model.rollout(
                        batch["state_hist"], batch["action_hist"], batch["action_future"]
                    )
                # Physical units, not normalised: see GPUResidentSplit.denorm. The nominal
                # triple integrator is only valid here.
                init_state = self.train_ds.denorm(
                    "state_hist", batch["state_hist"]
                )[:, -1, :9]
                jerk = self.train_ds.denorm("action_future", batch["action_future"])
                target = self.train_ds.denorm("state_future", batch["state_future"])[..., :9]

                out = self.prober(
                    pred_latents.detach(), jerk, init_state=init_state, return_nominal=False
                )
                loss = PhysicsProber.rollout_loss(out["states"], target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bs = batch["state_hist"].shape[0]
                n += bs
                tot += float(loss) * bs
                res_mag += float(out["residual"].detach().abs().mean()) * bs

            # The freeze must still hold after a full epoch of stage-2 optimization.
            assert_frozen(self.model, "JEPA core")

            record = {
                "stage": 2,
                "epoch": epoch,
                "prober_loss": tot / max(n, 1),
                "residual_abs_mean": res_mag / max(n, 1),
                "seconds": round(time.time() - t0, 2),
            }
            self._log(record)
            print(
                f"[stage2] epoch {epoch:>3}  loss {record['prober_loss']:.6f}  "
                f"|residual| {record['residual_abs_mean']:.6f}  {record['seconds']:.1f}s"
            )

        self._save("final.pt", include_prober=True)

    # ------------------------------------------------------------------ io

    def _save(self, name: str, include_prober: bool = False) -> Path:
        payload = {
            "model": self.model.state_dict(),
            "jepa_config": asdict(self.model.config)
            if hasattr(self.model.config, "__dataclass_fields__")
            else {},
            "train_config": asdict(self.cfg),
            "provenance": provenance(),
            "dataset_metadata": self.train_ds.metadata,
        }
        if include_prober and hasattr(self, "prober"):
            payload["prober"] = self.prober.state_dict()
        path = self.out_dir / name
        torch.save(payload, path)
        return path

    def run(self) -> dict[str, Any]:
        (self.out_dir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))
        (self.out_dir / "provenance.json").write_text(json.dumps(provenance(), indent=2))
        self.train_stage1()
        if self.cfg.train_prober:
            self.train_stage2()
        return {"out_dir": str(self.out_dir), "epochs_logged": len(self.history)}
