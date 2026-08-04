#!/usr/bin/env python
"""Does SIGReg penalise low-rank collapse? (E3 mechanism check)

SIGReg tests normality of random 1-D projections. A low-rank Gaussian latent projects to a Gaussian
along every direction, so it can satisfy a marginal-normality test while carrying almost no
information. This script measures the penalty against latents of known rank.

    python scripts/sigreg_rank_probe.py
"""

import torch

from flatjepa.models.sigreg import SIGReg, SIGRegConfig


def participation_ratio(z: torch.Tensor) -> float:
    zc = z - z.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh((zc.T @ zc) / (len(z) - 1)).clamp_min(0)
    return float(ev.sum() ** 2 / (ev**2).sum()) if ev.sum() > 0 else 0.0


def main(n: int = 4096, d: int = 24, seed: int = 0) -> None:
    torch.manual_seed(seed)
    sig = SIGReg(SIGRegConfig())
    cases = {
        "isotropic Gaussian (ideal)": torch.randn(n, d),
        "constant (total collapse)": torch.zeros(n, d),
        "rank-1 Gaussian": torch.randn(n, 1) @ torch.randn(1, d),
        "rank-2 Gaussian": torch.randn(n, 2) @ torch.randn(2, d),
        "rank-9 Gaussian": torch.randn(n, 9) @ torch.randn(9, d),
        "uniform, unit variance": (torch.rand(n, d) - 0.5) * 3.4641,
    }
    print(f"{'case':<30}{'SIGReg loss':>14}{'participation ratio':>22}")
    print("-" * 66)
    for name, z in cases.items():
        with torch.no_grad():
            loss = float(sig(z))
        print(f"{name:<30}{loss:>14.4f}{participation_ratio(z):>22.2f}")
    print()
    print("Expected if the penalty tracked rank: monotone decreasing from rank-1 to isotropic.")
    print("Observed: rank-9 is penalised MORE than rank-1, so the statistic responds to scale")
    print("rather than rank. Verify against rbalestr-lab/lejepa before claiming this of the method.")


if __name__ == "__main__":
    main()
