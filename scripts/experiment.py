#!/usr/bin/env python
"""Run the E1/E2/E3 experiment end to end.

Trains a sweep over ``lambda_sig`` (E3) and latent width (E2), across seeds, then probes every
resulting checkpoint against all three F7 controls.

Runs are executed concurrently: the model is ~100k parameters and one run leaves an RTX 3090 almost
entirely idle, so the sweep is throughput-bound on kernel launches rather than on compute.

    python scripts/experiment.py --out outputs/e1e3 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from flatjepa.data.targets import TARGET_SPECS, targets_of_kind
from flatjepa.models.jepa import FlatJEPA, JEPAConfig
from flatjepa.probes.suite import (
    ProbeInputs,
    effective_dimensionality,
    encode_split,
    format_recovery,
    run_recovery,
)
from flatjepa.training.data import GPUResidentSplit
from flatjepa.training.trainer import TrainConfig, Trainer, set_seed


def train_one(cfg: TrainConfig) -> str:
    trainer = Trainer(cfg)
    trainer.run()
    return cfg.out_dir


def load_model(run_dir: Path, device: torch.device) -> FlatJEPA:
    for name in ("final.pt", "best.pt", "collapsed.pt"):
        path = run_dir / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = FlatJEPA(JEPAConfig(**payload["jepa_config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def probe_run(run_dir: Path, splits: dict[str, GPUResidentSplit], device, seed: int) -> dict:
    """Probe one checkpoint: E1 recovery with controls, plus E2 dimensionality."""
    model = load_model(run_dir, device)

    # Control arm: identical architecture, never trained.
    set_seed(seed + 10_000)
    random_model = FlatJEPA(model.config).to(device).eval()

    latents = {k: encode_split(model, s) for k, s in splits.items()}
    random_latents = {k: encode_split(random_model, s) for k, s in splits.items()}
    raw = {k: s.flat_inputs().astype(np.float64) for k, s in splits.items()}

    arms = {
        "trained": ProbeInputs(latents["train"], latents["val"], latents["test"]),
        "random_init": ProbeInputs(
            random_latents["train"], random_latents["val"], random_latents["test"]
        ),
        "raw_window": ProbeInputs(raw["train"], raw["val"], raw["test"]),
    }

    targets = {
        spec.name: tuple(splits[s].target(spec.name).astype(np.float64) for s in ("train", "val", "test"))
        for spec in TARGET_SPECS
    }

    recovery = run_recovery(arms, targets, shuffle_seed=seed)
    dims = {
        "trained": effective_dimensionality(latents["test"]),
        "random_init": effective_dimensionality(random_latents["test"]),
    }

    log = []
    log_path = run_dir / "log.jsonl"
    if log_path.exists():
        log = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    stage1 = [r for r in log if r.get("stage") == 1]
    stage2 = [r for r in log if r.get("stage") == 2]

    return {
        "run_dir": str(run_dir),
        "recovery": recovery,
        "dimensionality": dims,
        "final_val_loss": stage1[-1]["val"]["loss"] if stage1 else None,
        "final_val_pred": stage1[-1]["val"]["loss_pred"] if stage1 else None,
        "final_sigreg": stage1[-1]["val"]["loss_sigreg"] if stage1 else None,
        "collapsed": bool(stage1 and stage1[-1].get("collapse_alarm", {}).get("triggered")),
        "epochs_ran": len(stage1),
        "prober_residual": stage2[-1]["residual_abs_mean"] if stage2 else None,
        "prober_loss": stage2[-1]["prober_loss"] if stage2 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/windows_v1")
    ap.add_argument("--out", default="outputs/experiment")
    ap.add_argument("--runs-root", default="runs/experiment")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16, 24, 48, 96])
    ap.add_argument("--base-width", type=int, default=24)
    ap.add_argument("--base-lambda", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--prober-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--jobs", type=int, default=4, help="concurrent training runs")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Build the sweep: lambda arm (E3) at base width, width arm (E2) at base lambda.
    configs: list[TrainConfig] = []
    for seed in args.seeds:
        for lam in args.lambdas:
            configs.append(
                TrainConfig(
                    dataset=args.dataset,
                    out_dir=f"{args.runs_root}/lam{lam}_w{args.base_width}_s{seed}",
                    seed=seed,
                    epochs=args.epochs,
                    prober_epochs=args.prober_epochs,
                    batch_size=args.batch_size,
                    latent_dim=args.base_width,
                    lambda_sig=lam,
                    collapse_alarm=False,  # the lambda=0 arm is expected to collapse; keep it running
                )
            )
        for width in args.widths:
            if width == args.base_width:
                continue  # already covered by the lambda arm
            configs.append(
                TrainConfig(
                    dataset=args.dataset,
                    out_dir=f"{args.runs_root}/lam{args.base_lambda}_w{width}_s{seed}",
                    seed=seed,
                    epochs=args.epochs,
                    prober_epochs=args.prober_epochs,
                    batch_size=args.batch_size,
                    latent_dim=width,
                    lambda_sig=args.base_lambda,
                    collapse_alarm=False,
                )
            )

    print(f"{len(configs)} runs on {device}, {args.jobs} concurrent\n")
    torch.set_float32_matmul_precision("high")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for i, done in enumerate(pool.map(train_one, configs), 1):
            print(f"  [{i}/{len(configs)}] {done}")

    # --- probe every checkpoint ---
    splits = {s: GPUResidentSplit(args.dataset, s, device) for s in ("train", "val", "test")}
    results = []
    for cfg in configs:
        r = probe_run(Path(cfg.out_dir), splits, device, cfg.seed)
        r["lambda_sig"] = cfg.lambda_sig
        r["latent_dim"] = cfg.latent_dim
        r["seed"] = cfg.seed
        results.append(r)
        print(f"probed {cfg.out_dir}")

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out / 'results.json'}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
