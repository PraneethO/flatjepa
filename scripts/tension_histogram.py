"""F3's first deliverable: the tension-margin histogram over the generated corpus.

Per F3 §2, this runs *before* the labeler is trusted for anything, because it answers the question
E4 depends on: does the generated data ever approach the slack boundary at all? The planner's own
bounds permit free fall (``a_z = −9.81`` lies strictly inside ``state_min/state_max``), but whether
minimum-time trajectories through clutter actually visit that corner is an empirical matter.

If near-slack timesteps turn out to be vanishingly rare, E4 has no signal and the honest response is
to say so — or to deliberately generate aggressive descent scenarios to induce them.

Usage
-----
    python scripts/tension_histogram.py
    python scripts/tension_histogram.py --csv-root /path/to/csvs --out-dir outputs/f3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flatjepa.data.csv_io import find_trajectory_csvs, load_trajectory_csv  # noqa: E402
from flatjepa.data.flatness import SystemParams  # noqa: E402
from flatjepa.data.tension import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    margin_summary,
    tension_margin,
    threshold_sweep,
)

DEFAULT_POLYFLY_DIR = Path("/home/praneetho/Desktop/polyfly_ral")


def default_roots() -> list[Path]:
    root = Path(os.environ.get("POLYFLY_DIR", DEFAULT_POLYFLY_DIR))
    return [root / "data" / "csvs"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--csv-root",
        type=Path,
        action="append",
        dest="csv_roots",
        help="Directory to search recursively for trajectory CSVs (repeatable).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/f3"))
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Thresholds tau for the near-slack base-rate sweep.",
    )
    parser.add_argument("--params", type=Path, default=None, help="Upstream base.yaml to read.")
    parser.add_argument(
        "--worst", type=int, default=15, help="How many closest-to-slack trajectories to list."
    )
    return parser.parse_args(argv)


def collect(paths: list[Path], gravity: float) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Compute the tension margin (and payload acceleration) for every timestep."""
    chunks: list[np.ndarray] = []
    acc_chunks: list[np.ndarray] = []
    per_traj: list[dict] = []
    for path in paths:
        try:
            traj = load_trajectory_csv(path)
        except (ValueError, OSError) as exc:
            print(f"  skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if traj.n_steps == 0:
            continue
        margin = tension_margin(traj.payload_acc, gravity=gravity)
        chunks.append(margin)
        acc_chunks.append(traj.payload_acc)
        per_traj.append(
            {
                "name": traj.name,
                "group": path.parent.name,
                "n": traj.n_steps,
                "dt": traj.dt,
                "min": float(np.min(margin)),
                "p1": float(np.percentile(margin, 1.0)),
                "median": float(np.median(margin)),
                "max": float(np.max(margin)),
                "min_az": float(np.min(traj.payload_acc[:, 2])),
            }
        )
    if not chunks:
        raise SystemExit("no usable trajectory CSVs found")
    return np.concatenate(chunks), np.concatenate(acc_chunks), per_traj


def report_acceleration(acc: np.ndarray, gravity: float) -> None:
    """Why the margin sits where it does: the margin can only reach 0 if a → −g·e₃.

    Free fall lies strictly inside the planner's own input/state bounds (±10 m/s² vs. g = 9.81),
    so if the corpus never gets near the boundary the reason is the planner's *behaviour*, not its
    feasible set. These numbers say how far from that corner the trajectories stay.
    """
    az = acc[:, 2]
    lateral = np.linalg.norm(acc[:, :2], axis=1)
    print("\npayload acceleration (why the margin sits where it does)")
    print(f"  a_z    min / p1 / median / max : {az.min():.3f} / {np.percentile(az, 1):.3f} / "
          f"{np.median(az):.3f} / {az.max():.3f}   (free fall needs a_z = {-gravity:.2f})")
    print(f"  |a_xy| median / p99 / max      : {np.median(lateral):.3f} / "
          f"{np.percentile(lateral, 99):.3f} / {lateral.max():.3f}")
    closest = float(np.min(np.linalg.norm(acc + np.array([0.0, 0.0, gravity]), axis=1)))
    print(f"  closest approach to -g*e3      : {closest:.3f} m/s^2")


def report(margin: np.ndarray, per_traj: list[dict], thresholds: list[float], worst: int) -> dict:
    summary = margin_summary(margin)
    sweep = threshold_sweep(margin, thresholds)

    print("\n" + "=" * 78)
    print("F3 — tension margin  T/(mL*g)   [1.0 = hover, 0.0 = free fall / slack boundary]")
    print("=" * 78)
    print(f"trajectories : {len(per_traj)}")
    print(f"timesteps    : {int(summary['n']):,}  (non-finite: {int(summary['n_nonfinite'])})")
    print(f"min / max    : {summary['min']:.6f} / {summary['max']:.6f}")
    print(f"mean +- std  : {summary['mean']:.6f} +- {summary['std']:.6f}")

    print("\npercentiles of the margin")
    for key in ("p0", "p0.01", "p0.1", "p1", "p5", "p10", "p25", "p50", "p75", "p95", "p100"):
        if key in summary:
            print(f"  {key:>7s}  {summary[key]:.6f}")

    print("\nnear-slack base rate by threshold tau  (near_slack := margin < tau)")
    print(f"  {'tau':>6s}  {'count':>10s}  {'base rate':>12s}")
    for row in sweep:
        print(f"  {row['threshold']:6.3f}  {int(row['count']):10,d}  {row['base_rate']:12.3e}")

    groups = sorted({t["group"] for t in per_traj})
    if len(groups) > 1:
        print("\nby corpus subdirectory (sampling rate differs between them)")
        print(f"  {'group':<14s} {'files':>6s} {'steps':>10s} {'dt':>8s} {'min margin':>12s}")
        for group in groups:
            rows = [t for t in per_traj if t["group"] == group]
            print(
                f"  {group:<14s} {len(rows):6d} {sum(r['n'] for r in rows):10,d} "
                f"{np.median([r['dt'] for r in rows]):8.4f} {min(r['min'] for r in rows):12.6f}"
            )

    print(f"\n{worst} trajectories closest to the slack boundary")
    print(f"  {'min margin':>11s}  {'p1':>9s}  {'median':>9s}  {'min a_z':>8s}  trajectory")
    for row in sorted(per_traj, key=lambda r: r["min"])[:worst]:
        print(
            f"  {row['min']:11.6f}  {row['p1']:9.6f}  {row['median']:9.6f}  "
            f"{row['min_az']:8.3f}  {row['name']}"
        )

    return {"summary": summary, "sweep": sweep}


def make_figure(margin: np.ndarray, per_traj: list[dict], thresholds: list[float],
                bins: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.hist(margin, bins=bins, color="#3b6ea5", edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(1.0, color="#888888", lw=1.0, ls="--")
    ax.text(1.0, ax.get_ylim()[1], " hover", va="top", ha="left", fontsize=8, color="#555555")
    ax.set_xlabel(r"tension margin  $T/(m_L g)$")
    ax.set_ylabel("timesteps (log)")
    ax.set_title("Full distribution")

    ax = axes[1]
    low = margin[margin < 0.6]
    if low.size:
        ax.hist(low, bins=max(20, bins // 2), range=(0.0, 0.6), color="#a53b3b", edgecolor="none")
    else:
        ax.text(0.5, 0.5, "no timesteps below 0.6", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#a53b3b")
    ax.set_yscale("log")
    for tau in thresholds:
        if tau <= 0.6:
            ax.axvline(tau, color="#555555", lw=0.8, ls=":")
    ax.set_xlim(0.0, 0.6)
    ax.set_xlabel(r"tension margin  $T/(m_L g)$")
    ax.set_ylabel("timesteps (log)")
    ax.set_title("Low tail — the E4 regime (slack at 0)")

    ax = axes[2]
    mins = np.array([t["min"] for t in per_traj])
    ax.hist(mins, bins=min(40, max(5, len(mins) // 2)), color="#3b8f5a", edgecolor="none")
    ax.set_xlabel("per-trajectory minimum margin")
    ax.set_ylabel("trajectories")
    ax.set_title(f"Closest approach per trajectory (n={len(mins)})")

    fig.suptitle(
        "F3 — cable tension margin over the generated corpus  "
        r"($T = m_L\,\|a + g e_3\|$;  0 = slack boundary)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"\nfigure written to {out_path}")


def write_per_trajectory_csv(per_traj: list[dict], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_traj[0].keys()))
        writer.writeheader()
        writer.writerows(per_traj)
    print(f"per-trajectory statistics written to {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    roots = args.csv_roots or default_roots()
    params = SystemParams.from_yaml(args.params) if args.params else SystemParams()

    paths = find_trajectory_csvs(roots)
    if not paths:
        raise SystemExit(f"no CSVs found under {[str(r) for r in roots]}")
    print(f"scanning {len(paths)} CSVs under {[str(r) for r in roots]}")

    margin, acc, per_traj = collect(paths, gravity=params.gravity)
    report(margin, per_traj, args.thresholds, args.worst)
    report_acceleration(acc, params.gravity)
    make_figure(margin, per_traj, args.thresholds, args.bins,
                args.out_dir / "tension_margin_histogram.png")
    write_per_trajectory_csv(per_traj, args.out_dir / "tension_margin_per_trajectory.csv")

    lowest = float(np.min(margin))
    tau_max = max(args.thresholds)
    n_below = int(np.count_nonzero(margin < tau_max))
    print("\n" + "-" * 78)
    if n_below == 0:
        print(
            f"E4 VIABILITY: NOT SUPPORTED BY THIS CORPUS.\n"
            f"  The global minimum tension margin is {lowest:.4f}; not one timestep falls\n"
            f"  below the loosest threshold swept (tau = {tau_max:g}). The near-slack class is\n"
            f"  empty, so a taut-vs-slack probe has nothing to separate. Per F3 section 2 the\n"
            f"  honest options are (a) report this as a negative result and drop E4, or\n"
            f"  (b) deliberately generate aggressive descent scenarios that push a_z toward -g."
        )
    else:
        print(
            f"E4 VIABILITY: near-slack timesteps exist.\n"
            f"  Global minimum margin {lowest:.4f}; {n_below:,} timesteps "
            f"({n_below / margin.size:.3e} of the corpus) fall below tau = {tau_max:g}.\n"
            f"  Report the base rate alongside every probe number (F3 section 4)."
        )
    print("-" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
