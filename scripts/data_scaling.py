#!/usr/bin/env python
"""Does more data help? (decides how long trajectory generation should run)

Trains identical models on increasing fractions of the training split and reports both prediction
quality and representation quality. If the curves are flat, generating more trajectories is wasted
compute and the bottleneck is elsewhere.

Subsampling is done at the **trajectory-block level** rather than uniformly over windows: windows
overlap heavily, so a uniform 25% sample would still touch nearly every trajectory and would
overstate how much a genuinely smaller corpus achieves.

    python scripts/data_scaling.py --fractions 0.1 0.25 0.5 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flatjepa.data.targets import targets_of_kind
from flatjepa.models.jepa import FlatJEPA, JEPAConfig
from flatjepa.probes.suite import ProbeInputs, effective_dimensionality, ridge_probe
from flatjepa.training.data import GPUResidentSplit
from flatjepa.training.trainer import set_seed


def encode(model, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model.encode_context(x[i : i + batch]).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train(split, idx, cfg: JEPAConfig, device, epochs: int, bs: int, seed: int) -> FlatJEPA:
    set_seed(seed)
    model = FlatJEPA(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    tens = {k: v[idx] for k, v in split.tensors.items()}
    n = len(idx)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, device=device, generator=gen)
        for s in range(0, n, bs):
            b = order[s : s + bs]
            out = model(
                tens["state_hist"][b],
                tens["action_hist"][b],
                tens["state_future"][b],
                tens["action_future"][b],
            )
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/windows_v1")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--latent-dim", type=int, default=24)
    ap.add_argument("--lambda-sig", type=float, default=0.0)
    ap.add_argument("--out", default="outputs/data_scaling.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = GPUResidentSplit(args.dataset, "train", device)
    va = GPUResidentSplit(args.dataset, "val", device)
    te = GPUResidentSplit(args.dataset, "test", device)

    cfg = JEPAConfig(
        state_dim=tr.obs_dim,
        action_dim=3,
        history=tr.history,
        horizon=tr.horizon,
        latent_dim=args.latent_dim,
        lambda_sig=args.lambda_sig,
    )
    nonlinear = targets_of_kind("nonlinear")
    n_total = len(tr)
    rows = []

    for frac in args.fractions:
        for seed in args.seeds:
            # Contiguous block: windows are ordered by trajectory, so a block is a genuine subset
            # of trajectories rather than a thin slice through all of them.
            k = max(args.batch_size, int(n_total * frac))
            idx = torch.arange(k, device=device)

            model = train(tr, idx, cfg, device, args.epochs, args.batch_size, seed)

            with torch.no_grad():
                vout = model(
                    va.tensors["state_hist"],
                    va.tensors["action_hist"],
                    va.tensors["state_future"],
                    va.tensors["action_future"],
                )
                val_pred = float(vout["loss_pred"])

            set_seed(seed + 10_000)
            rand = FlatJEPA(cfg).to(device).eval()

            z = {n: encode(model, s.tensors["state_hist"]) for n, s in
                 (("train", tr), ("val", va), ("test", te))}
            zr = {n: encode(rand, s.tensors["state_hist"]) for n, s in
                  (("train", tr), ("val", va), ("test", te))}

            def score(feats, split_of):
                vals = []
                for t in nonlinear:
                    r = ridge_probe(
                        feats["train"], tr.target(t).astype(np.float64),
                        feats["val"], va.target(t).astype(np.float64),
                        feats["test"], te.target(t).astype(np.float64),
                    )
                    vals.append(r["r2"])
                return float(np.mean(vals))

            trained_r2 = score(z, None)
            random_r2 = score(zr, None)
            pr = effective_dimensionality(z["test"])["participation_ratio"]

            rows.append({
                "fraction": frac, "n_windows": k, "seed": seed,
                "val_pred": val_pred, "trained_r2": trained_r2,
                "random_r2": random_r2, "delta": trained_r2 - random_r2,
                "participation_ratio": pr,
            })
            print(f"frac={frac:<5} n={k:<6} seed={seed}  val_pred={val_pred:.6f}  "
                  f"R²={trained_r2:.4f}  random={random_r2:.4f}  Δ={trained_r2-random_r2:+.4f}  "
                  f"PR={pr:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))

    print(f"\n{'frac':<8}{'n':<9}{'val_pred':>12}{'R² trained':>13}{'Δ vs random':>14}")
    print("-" * 56)
    for frac in args.fractions:
        sel = [r for r in rows if r["fraction"] == frac]
        print(f"{frac:<8}{sel[0]['n_windows']:<9}"
              f"{np.mean([r['val_pred'] for r in sel]):>12.6f}"
              f"{np.mean([r['trained_r2'] for r in sel]):>13.4f}"
              f"{np.mean([r['delta'] for r in sel]):>+14.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
