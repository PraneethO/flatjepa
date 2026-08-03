#!/usr/bin/env python3
"""CLI entry point for F1 trajectory generation and manifest maintenance.

Subcommands
-----------
``check``     Validate the docker/shim/repo plumbing without solving anything.
``smoke``     Generate 2 trajectories end to end (CI gate, F1 §8).
``generate``  Batch-generate forest environments in parallel.
``manifest``  (Re)build ``manifest.jsonl`` from the CSVs already on disk. Read-only w.r.t. the
              PolyFly checkout, so it is safe to run while a generation job is in flight.
``stats``     Summarise an existing manifest.

Examples
--------
    python scripts/generate_data.py manifest --polyfly-repo ~/Desktop/polyfly_ral
    python scripts/generate_data.py smoke --polyfly-repo ~/Desktop/polyfly_ral
    python scripts/generate_data.py generate --n-environments 200 --workers 14 --base-seed 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from flatjepa.data import manifest as manifest_mod  # noqa: E402
from flatjepa.data import generate as gen  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.jsonl"
DEFAULT_SHIM = REPO_ROOT / "scripts" / "shim"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--polyfly-repo",
        type=Path,
        default=None,
        help="Path to the polyfly_ral checkout (default: $POLYFLY_REPO or $POLYFLY_DIR).",
    )
    parser.add_argument("--shim-dir", type=Path, default=DEFAULT_SHIM,
                        help="Directory containing the headless-matplotlib sitecustomize.py.")
    parser.add_argument("--image", default=gen.DEFAULT_IMAGE, help="Planner docker image.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Manifest JSONL path.")


def _add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-seed", type=int, default=0,
                        help="Seed for deterministic environment-level split assignment.")
    parser.add_argument("--split-method", choices=("hash", "shuffle"), default="hash",
                        help="'hash' is stable as the corpus grows; 'shuffle' gives exact ratios.")
    parser.add_argument("--split-fractions", type=float, nargs=3,
                        default=list(manifest_mod.DEFAULT_SPLIT_FRACTIONS),
                        metavar=("TRAIN", "VAL", "TEST"), help="Train/val/test fractions.")


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--history", "-H", type=int, default=10, help="Window history length H.")
    parser.add_argument("--horizon", "-T", type=int, default=20, help="Window future length T.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Resampling stride used when checking the H+T length requirement.")
    parser.add_argument("--require-status", action="store_true",
                        help="Reject trajectories with no positively recorded solver status.")


def _filter_kwargs(args: argparse.Namespace) -> dict:
    return {
        "h": args.history,
        "t": args.horizon,
        "stride": args.stride,
        "allow_unknown_status": not args.require_status,
    }


def _config(args: argparse.Namespace) -> gen.DockerConfig:
    return gen.DockerConfig.from_env(
        polyfly_repo=args.polyfly_repo, shim_dir=args.shim_dir, image=args.image
    )


def _solve_info_paths(args: argparse.Namespace) -> list[Path]:
    explicit = getattr(args, "solve_info", None)
    if explicit:
        return [Path(p) for p in explicit]
    return [Path(args.manifest).parent / "solve_info.jsonl"]


def cmd_check(args: argparse.Namespace) -> int:
    config = _config(args)
    print(f"polyfly repo : {config.polyfly_repo}")
    print(f"shim dir     : {config.shim_dir}")
    print(f"image        : {config.image}")
    print(f"LIBHSL_DIR   : {config.libhsl_dir or '<unset> (default IPOPT solver, ~3x slower)'}")
    problems = config.validate()
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
        return 1
    print("static checks passed")
    print("\nexample invocation:")
    example = config.command(gen.PLANNER_MODULE, ("--yaml", "experiments/maze_1.yaml"))
    print("  " + " ".join(example))
    if args.container:
        import subprocess

        probe = (
            "import poly_fly.optimal_planner.planner as p, matplotlib;"
            "print('OK backend=' + matplotlib.get_backend())"
        )
        cmd = config.raw_command(["-c", probe])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(f"\ncontainer import probe exit={result.returncode}")
        print((result.stdout or result.stderr).strip()[-2000:])
        return 0 if result.returncode == 0 else 1
    return 0


def _report_and_summarize(args: argparse.Namespace, report, records) -> int:
    print()
    print(report.format())
    manifest_mod.apply_splits(
        records,
        seed=args.split_seed,
        fractions=tuple(args.split_fractions),
        method=args.split_method,
    )
    manifest_mod.write_manifest(records, args.manifest)
    print()
    print(manifest_mod.format_summary(manifest_mod.summarize(records, **_filter_kwargs(args))))
    print(f"\nmanifest written to {args.manifest}")
    return 0 if not report.failed_jobs else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    config = _config(args)
    problems = config.validate()
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
        return 1
    jobs = gen.smoke_jobs(args.yamls or None)
    print(f"smoke: {len(jobs)} trajectories -> {[j.expected_stems[0] for j in jobs]}")
    report, records = gen.generate(
        jobs,
        config,
        args.manifest,
        workers=len(jobs),
        timeout_s=args.timeout,
        force=True,  # a smoke test must actually exercise the solve
        split_seed=args.split_seed,
        split_method=args.split_method,
    )
    return _report_and_summarize(args, report, records)


def cmd_generate(args: argparse.Namespace) -> int:
    config = _config(args)
    problems = config.validate()
    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}")
        return 1

    jobs = gen.forest_jobs(
        base_seed=args.base_seed,
        n_environments=args.n_environments,
        forest_type=args.forest_type,
        subdir=args.subdir,
    )
    total = len(jobs) * gen.TRAJECTORIES_PER_SEED
    print(f"{len(jobs)} environments x {gen.TRAJECTORIES_PER_SEED} trajectories = {total} solves")
    if args.dry_run:
        for job in jobs[: args.dry_run_limit]:
            print(f"\n[{job.job_id}] expects {len(job.expected_stems)} stems, "
                  f"first={job.expected_stems[0]}")
            print("  " + " ".join(config.command(job.module, job.args)))
        return 0

    done = 0

    def progress(result) -> None:
        nonlocal done
        done += 1
        status = "ok" if result.ok else f"FAIL(exit={result.returncode})"
        print(f"[{done}/{len(jobs)}] {result.job.job_id} {status} "
              f"{len(result.produced)}/{len(result.job.expected_stems)} csv "
              f"{result.duration_s:.1f}s", flush=True)

    report, records = gen.generate(
        jobs,
        config,
        args.manifest,
        workers=args.workers,
        timeout_s=args.timeout,
        force=args.force,
        split_seed=args.split_seed,
        split_method=args.split_method,
        on_result=progress,
    )
    return _report_and_summarize(args, report, records)


def cmd_manifest(args: argparse.Namespace) -> int:
    config = _config(args)
    records = gen.rebuild_manifest(
        config.polyfly_repo,
        args.manifest,
        subdirs=args.subdirs or None,
        split_seed=args.split_seed,
        split_fractions=tuple(args.split_fractions),
        split_method=args.split_method,
        solve_info_paths=_solve_info_paths(args),
    )
    summary = manifest_mod.summarize(records, **_filter_kwargs(args))
    print(manifest_mod.format_summary(summary))
    print(f"\nmanifest written to {args.manifest} ({len(records)} records)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"summary written to {args.json}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    records = manifest_mod.read_manifest(args.manifest)
    summary = manifest_mod.summarize(records, **_filter_kwargs(args))
    print(manifest_mod.format_summary(summary))
    if args.list_rejected:
        _, rejected = manifest_mod.filter_records(records, **_filter_kwargs(args))
        for stem, reasons in sorted(rejected.items()):
            print(f"  REJECT {stem}: {','.join(reasons)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="validate docker/shim plumbing")
    _add_common(p_check)
    p_check.add_argument("--container", action="store_true",
                         help="also start a container and import the planner (slow, ~30s).")
    p_check.set_defaults(func=cmd_check)

    p_smoke = sub.add_parser("smoke", help="generate 2 trajectories end to end")
    _add_common(p_smoke)
    _add_split_args(p_smoke)
    _add_filter_args(p_smoke)
    p_smoke.add_argument("--yamls", nargs="*", default=None,
                         help="Relative params YAMLs to solve (default: two maze environments).")
    p_smoke.add_argument("--timeout", type=float, default=600.0, help="Per-job timeout (s).")
    p_smoke.set_defaults(func=cmd_smoke)

    p_gen = sub.add_parser("generate", help="batch-generate forest trajectories")
    _add_common(p_gen)
    _add_split_args(p_gen)
    _add_filter_args(p_gen)
    p_gen.add_argument("--n-environments", type=int, required=True,
                       help=f"Environments to generate ({gen.TRAJECTORIES_PER_SEED} solves each).")
    p_gen.add_argument("--base-seed", type=int, default=0,
                       help="First environment base seed; seeds are base_seed..base_seed+N-1.")
    p_gen.add_argument("--forest-type", type=int, default=0, choices=(0, 2),
                       help="0 = small obstacles, 2 = large obstacles.")
    p_gen.add_argument("--subdir", default="forests", help="CSV/params subdirectory.")
    p_gen.add_argument("--workers", type=int, default=8, help="Concurrent containers.")
    p_gen.add_argument("--timeout", type=float, default=1800.0, help="Per-job timeout (s).")
    p_gen.add_argument("--force", action="store_true",
                       help="Re-solve environments whose outputs already exist.")
    p_gen.add_argument("--dry-run", action="store_true", help="Print commands and exit.")
    p_gen.add_argument("--dry-run-limit", type=int, default=3, help="Commands to print.")
    p_gen.set_defaults(func=cmd_generate)

    p_man = sub.add_parser("manifest", help="(re)build the manifest from CSVs on disk")
    _add_common(p_man)
    _add_split_args(p_man)
    _add_filter_args(p_man)
    p_man.add_argument("--subdirs", nargs="*", default=None,
                       help="CSV subdirectories to scan (default: all).")
    p_man.add_argument("--solve-info", nargs="*", default=None,
                       help="solve_info.jsonl sidecars to merge (default: next to the manifest).")
    p_man.add_argument("--json", type=Path, default=None, help="Also write the summary as JSON.")
    p_man.set_defaults(func=cmd_manifest)

    p_stats = sub.add_parser("stats", help="summarise an existing manifest")
    _add_common(p_stats)
    _add_filter_args(p_stats)
    p_stats.add_argument("--list-rejected", action="store_true",
                         help="Print every rejected trajectory and why.")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
