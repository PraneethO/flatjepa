#!/usr/bin/env python
"""Turn experiment results into the E1/E2/E3 tables and figures (F9).

Every figure regenerates from ``results.json``; nothing is assembled by hand. Controls are plotted
alongside the results they contextualise rather than relegated to an appendix, and every
quantitative figure carries seed spread.

    python scripts/report.py --results outputs/e1e3/results.json --out outputs/e1e3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flatjepa.data.targets import TARGET_SPECS, targets_of_kind

ARMS = ("trained", "random_init", "raw_window", "shuffled_labels")
ARM_LABEL = {
    "trained": "trained encoder",
    "random_init": "random-init encoder",
    "raw_window": "raw input window",
    "shuffled_labels": "shuffled labels",
}
THEORETICAL_DIM = 9  # (p, v, a): the planner's own state under triple-integrator dynamics


def load(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


# --------------------------------------------------------------------------------------------


def e1_table(results: list[dict], base_lambda: float, base_width: int) -> tuple[str, dict]:
    """Recovery R² per target per arm, at the base configuration, mean ± std over seeds."""
    sel = [r for r in results if r["lambda_sig"] == base_lambda and r["latent_dim"] == base_width]
    agg: dict[str, dict[str, tuple[float, float]]] = {}
    for spec in TARGET_SPECS:
        agg[spec.name] = {
            arm: mean_std([r["recovery"][spec.name][arm]["r2"] for r in sel]) for arm in ARMS
        }

    lines = [
        f"### E1 — latent recovery (λ={base_lambda}, width={base_width}, {len(sel)} seeds)",
        "",
        "| target | kind | " + " | ".join(ARM_LABEL[a] for a in ARMS) + " | trained − random |",
        "|---|---|" + "---|" * (len(ARMS) + 1),
    ]
    for spec in TARGET_SPECS:
        row = f"| `{spec.name}` | {spec.kind} |"
        for arm in ARMS:
            m, s = agg[spec.name][arm]
            row += f" {m:.3f} ± {s:.3f} |"
        delta = agg[spec.name]["trained"][0] - agg[spec.name]["random_init"][0]
        row += f" **{delta:+.3f}** |"
        lines.append(row)
    return "\n".join(lines), agg


def e2_table(results: list[dict], base_lambda: float) -> tuple[str, dict]:
    """Allocated width vs measured effective dimensionality."""
    by_width = defaultdict(list)
    for r in results:
        if r["lambda_sig"] == base_lambda:
            by_width[r["latent_dim"]].append(r)

    lines = [
        f"### E2 — intrinsic dimensionality (λ={base_lambda})",
        "",
        "| allocated | participation ratio | effective rank | 90% var | 99% var | random-init PR |",
        "|---|---|---|---|---|---|",
    ]
    agg = {}
    for width in sorted(by_width):
        runs = by_width[width]
        pr = mean_std([r["dimensionality"]["trained"]["participation_ratio"] for r in runs])
        er = mean_std([r["dimensionality"]["trained"]["effective_rank"] for r in runs])
        c90 = mean_std([r["dimensionality"]["trained"]["n_components_90pct"] for r in runs])
        c99 = mean_std([r["dimensionality"]["trained"]["n_components_99pct"] for r in runs])
        rpr = mean_std([r["dimensionality"]["random_init"]["participation_ratio"] for r in runs])
        agg[width] = {"pr": pr, "er": er, "c90": c90, "c99": c99, "random_pr": rpr}
        lines.append(
            f"| {width} | {pr[0]:.2f} ± {pr[1]:.2f} | {er[0]:.2f} ± {er[1]:.2f} | "
            f"{c90[0]:.1f} | {c99[0]:.1f} | {rpr[0]:.2f} |"
        )
    lines.append("")
    lines.append(f"Theoretical minimal state dimension: **{THEORETICAL_DIM}**.")
    return "\n".join(lines), agg


def e3_table(results: list[dict], base_width: int) -> tuple[str, dict]:
    """The headline: does SIGReg help or hurt recovery, and does it dissociate from prediction?"""
    by_lambda = defaultdict(list)
    for r in results:
        if r["latent_dim"] == base_width:
            by_lambda[r["lambda_sig"]].append(r)

    nonlinear = targets_of_kind("nonlinear")
    lines = [
        f"### E3 — SIGReg ablation (width={base_width})",
        "",
        "| λ_sig | val pred loss | participation ratio | mean R² (nonlinear targets) | collapsed |",
        "|---|---|---|---|---|",
    ]
    agg = {}
    for lam in sorted(by_lambda):
        runs = by_lambda[lam]
        pred = mean_std([r["final_val_pred"] for r in runs])
        pr = mean_std([r["dimensionality"]["trained"]["participation_ratio"] for r in runs])
        rec = mean_std(
            [float(np.mean([r["recovery"][t]["trained"]["r2"] for t in nonlinear])) for r in runs]
        )
        n_collapsed = sum(bool(r["collapsed"]) for r in runs)
        agg[lam] = {"pred": pred, "pr": pr, "recovery": rec, "collapsed": n_collapsed}
        lines.append(
            f"| {lam} | {pred[0]:.4f} ± {pred[1]:.4f} | {pr[0]:.2f} ± {pr[1]:.2f} | "
            f"{rec[0]:.3f} ± {rec[1]:.3f} | {n_collapsed}/{len(runs)} |"
        )

    # The interesting outcome is a dissociation: best prediction at a different λ than best recovery.
    if agg:
        best_pred = min(agg, key=lambda k: agg[k]["pred"][0])
        best_rec = max(agg, key=lambda k: agg[k]["recovery"][0])
        lines.append("")
        lines.append(f"λ minimising prediction loss: **{best_pred}**")
        lines.append(f"λ maximising nonlinear recovery: **{best_rec}**")
        lines.append(
            "**Dissociation observed**: the objective and faithful state representation are "
            "optimised at different λ."
            if best_pred != best_rec
            else "No dissociation: the same λ optimises both."
        )
    return "\n".join(lines), agg


# --------------------------------------------------------------------------------------------


def fig_e1(agg: dict, path: Path) -> None:
    names = [s.name for s in TARGET_SPECS]
    kinds = {s.name: s.kind for s in TARGET_SPECS}
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, arm in enumerate(ARMS):
        means = [agg[n][arm][0] for n in names]
        errs = [agg[n][arm][1] for n in names]
        ax.bar(x + (i - 1.5) * width, means, width, yerr=errs, capsize=3, label=ARM_LABEL[arm])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{n}\n({'lin' if kinds[n] == 'linear_trivial' else 'nonlin'})" for n in names]
    )
    ax.set_ylabel("test R²")
    ax.set_title("E1: latent recovery vs. controls (error bars = seed std)")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_e2(agg: dict, path: Path) -> None:
    widths = sorted(agg)
    pr = [agg[w]["pr"][0] for w in widths]
    err = [agg[w]["pr"][1] for w in widths]
    rnd = [agg[w]["random_pr"][0] for w in widths]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(widths, pr, yerr=err, marker="o", capsize=3, label="trained")
    ax.plot(widths, rnd, marker="s", ls="--", label="random-init")
    ax.plot(widths, widths, ls=":", color="gray", label="allocated = effective")
    ax.axhline(THEORETICAL_DIM, color="crimson", ls="-.", label=f"theoretical ({THEORETICAL_DIM})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("allocated latent width")
    ax.set_ylabel("participation ratio (effective dim)")
    ax.set_title("E2: allocated vs. effective dimensionality")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_e3(agg: dict, path: Path) -> None:
    lams = sorted(agg)
    xs = [l if l > 0 else 1e-4 for l in lams]  # 0 plotted at the left edge of a log axis
    pred = [agg[l]["pred"][0] for l in lams]
    rec = [agg[l]["recovery"][0] for l in lams]
    rec_err = [agg[l]["recovery"][1] for l in lams]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, pred, marker="o", color="tab:blue", label="val prediction loss")
    ax1.set_xscale("log")
    ax1.set_xlabel("λ_sig  (leftmost point is λ=0)")
    ax1.set_ylabel("val prediction loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.errorbar(xs, rec, yerr=rec_err, marker="s", color="tab:red", capsize=3,
                 label="mean R² (nonlinear targets)")
    ax2.set_ylabel("recovery R² (nonlinear targets)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("E3: does SIGReg trade prediction against representation?")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="outputs/e1e3/results.json")
    ap.add_argument("--out", default="outputs/e1e3")
    ap.add_argument("--base-lambda", type=float, default=1.0)
    ap.add_argument("--base-width", type=int, default=24)
    args = ap.parse_args()

    results = load(Path(args.results))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t1, a1 = e1_table(results, args.base_lambda, args.base_width)
    t2, a2 = e2_table(results, args.base_lambda)
    t3, a3 = e3_table(results, args.base_width)

    fig_e1(a1, out / "fig1_e1_recovery.png")
    fig_e2(a2, out / "fig2_e2_dimensionality.png")
    fig_e3(a3, out / "fig3_e3_sigreg.png")

    seeds = sorted({r["seed"] for r in results})
    report = "\n\n".join(
        [
            "# Results",
            f"{len(results)} runs · seeds {seeds} · dataset `data/windows_v1`",
            t1,
            t2,
            t3,
            "### Figures",
            "![E1](fig1_e1_recovery.png)\n\n![E2](fig2_e2_dimensionality.png)\n\n"
            "![E3](fig3_e3_sigreg.png)",
        ]
    )
    (out / "RESULTS.md").write_text(report + "\n")
    print(report)
    print(f"\nwrote {out / 'RESULTS.md'} and 3 figures")


if __name__ == "__main__":
    main()
