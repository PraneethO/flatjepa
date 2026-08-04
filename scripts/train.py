#!/usr/bin/env python
"""Train a FlatJEPA run (F8).

    python scripts/train.py --out-dir runs/base --epochs 30
    python scripts/train.py --lambda-sig 0.0 --out-dir runs/lambda0    # the collapse arm

Sweeps over seeds/lambda are driven by scripts/sweep.py, which runs configurations concurrently on
one GPU -- the model is ~100k parameters, so a single run leaves the device almost entirely idle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from flatjepa.training.trainer import TrainConfig, Trainer


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/windows_v1")
    ap.add_argument("--out-dir", default="runs/default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent-dim", type=int, default=24)
    ap.add_argument("--lambda-sig", type=float, default=1.0)
    ap.add_argument("--prober-epochs", type=int, default=10)
    ap.add_argument("--no-prober", action="store_true")
    ap.add_argument("--no-alarm", action="store_true", help="do not halt on collapse")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg = TrainConfig(
        dataset=args.dataset,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        lambda_sig=args.lambda_sig,
        prober_epochs=args.prober_epochs,
        train_prober=not args.no_prober,
        collapse_alarm=not args.no_alarm,
    )
    print(json.dumps(asdict(cfg), indent=2))
    result = Trainer(cfg).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
