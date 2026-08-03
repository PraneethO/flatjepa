"""Batch driver for PolyFly trajectory generation (F1).

Wraps the verified containerised invocation from F1 §3, runs many of them in parallel, checks the
exit code of every one, and records what actually landed on disk.

Two upstream blockers are handled here because F1 §2 says any batch driver must handle them:

* **Headless matplotlib** — ``global_planner.py`` calls ``matplotlib.use('TkAgg')`` at import time.
  A ``sitecustomize.py`` shim (``scripts/shim/``) is mounted first on ``PYTHONPATH`` so the
  interactive backend selection becomes a no-op. The upstream checkout stays pristine.
* **Container UID mismatch** — the image ends with ``USER mambauser``, which cannot write to a
  bind-mounted host directory. ``--user $(id -u):$(id -g)`` plus ``HOME=/tmp`` fixes it. Critically,
  **this failure is silent in the planner's own summary output**: the solve reports success and only
  the process exit code reveals that nothing was written. Every job's return code is therefore
  checked, and every job's expected outputs are verified to exist afterwards.

Design notes worth knowing before changing this file:

* The unit of parallel work is *one base seed*, run as ``generate_forest -n 1 --sequential``. We do
  our own fan-out rather than using upstream's ``--mp`` so that each unit has its own container and
  therefore its own exit code. Upstream's internal multiprocessing would hide per-trajectory
  failures behind a single process return code.
* Upstream writes a CSV **only** when the solve succeeded, and never persists the IPOPT return
  status, iteration count or solve time anywhere on disk. This driver parses them out of the
  planner's stdout and writes a ``solve_info.jsonl`` sidecar, so a later manifest rebuild can
  recover provenance that would otherwise be lost forever.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from flatjepa.data import manifest as manifest_mod
from flatjepa.data.manifest import TrajectoryRecord

__all__ = [
    "DEFAULT_IMAGE",
    "DEFAULT_PYTHON",
    "FOREST_GOALS",
    "DockerConfig",
    "Job",
    "SolveInfo",
    "JobResult",
    "GenerationReport",
    "forest_env_seed",
    "expected_forest_stems",
    "forest_jobs",
    "planner_jobs",
    "smoke_jobs",
    "parse_planner_log",
    "run_jobs",
    "generate",
    "rebuild_manifest",
]

DEFAULT_IMAGE = "poly-fly:latest"
DEFAULT_PYTHON = "/opt/conda/envs/poly_fly/bin/python"
FOREST_MODULE = "poly_fly.forest_planner.generate_forest"
PLANNER_MODULE = "poly_fly.optimal_planner.planner"

#: Goal set that upstream ``generate_forest.main_mp`` enumerates, in its exact order.
FOREST_GOALS: tuple[tuple[float, float, float], ...] = ((15, 0, 0), (15, -3, 0), (15, 3, 0))
#: ``(margins, initial velocities)`` blocks, appended in this order by ``append_job_lists``.
FOREST_BLOCKS: tuple[tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]], ...] = (
    ((0.5,), ((0, 0, 0), (1, 0, 0))),
    ((0.6,), ((0, 0, 0),)),
)
#: Trajectories a single ``-n 1`` forest run attempts.
TRAJECTORIES_PER_SEED = sum(len(FOREST_GOALS) * len(m) * len(v) for m, v in FOREST_BLOCKS)


# --------------------------------------------------------------------------------------------
# Container invocation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DockerConfig:
    """Everything needed to reproduce the verified F1 §3 ``docker run`` line."""

    polyfly_repo: Path
    shim_dir: Path
    image: str = DEFAULT_IMAGE
    python_bin: str = DEFAULT_PYTHON
    libhsl_dir: str | None = None
    docker_bin: str = "docker"
    extra_env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        polyfly_repo: Path | str | None = None,
        shim_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> "DockerConfig":
        """Build a config from arguments, falling back to ``POLYFLY_REPO`` / ``LIBHSL_DIR``.

        ``LIBHSL_DIR`` is read here and passed through to the container when set. Acquiring an HSL
        academic licence later is then a zero-code-change ~3x solver speedup (F1 §5).
        """
        repo = polyfly_repo or os.environ.get("POLYFLY_REPO") or os.environ.get("POLYFLY_DIR")
        if not repo:
            raise ValueError(
                "polyfly repo not given and neither POLYFLY_REPO nor POLYFLY_DIR is set"
            )
        shim = shim_dir or Path(__file__).resolve().parents[3] / "scripts" / "shim"
        kwargs.setdefault("libhsl_dir", os.environ.get("LIBHSL_DIR") or None)
        return cls(polyfly_repo=Path(repo).resolve(), shim_dir=Path(shim).resolve(), **kwargs)

    def command(self, module: str, args: Sequence[str]) -> list[str]:
        """Return the full ``docker run`` argv for one planner invocation."""
        return self.raw_command(["-m", module, *args])

    def raw_command(self, python_args: Sequence[str]) -> list[str]:
        """Return the ``docker run`` argv for an arbitrary in-container python invocation."""
        uid, gid = os.getuid(), os.getgid()
        cmd = [
            self.docker_bin, "run", "--rm",
            "--user", f"{uid}:{gid}",
            "-v", f"{self.polyfly_repo}:/workspace:rw",
            "-v", f"{self.shim_dir}:/shim:ro",
            "-e", "POLYFLY_DIR=/workspace",
            "-e", "PYTHONPATH=/shim:/workspace/src",
            "-e", "MPLBACKEND=Agg",
            "-e", "HOME=/tmp",
            # Each container solves one problem at a time; keep BLAS from oversubscribing cores.
            "-e", "OMP_NUM_THREADS=1",
            "-e", "MKL_NUM_THREADS=1",
            "-e", "OPENBLAS_NUM_THREADS=1",
        ]
        if self.libhsl_dir:
            # Mount the host HSL tree at the same path inside the container so the value of
            # LIBHSL_DIR is meaningful on both sides; upstream looks for <dir>/lib/libcoinhsl.so.
            host_path = str(Path(self.libhsl_dir).resolve())
            cmd += ["-v", f"{host_path}:{host_path}:ro", "-e", f"LIBHSL_DIR={host_path}"]
        for key, value in self.extra_env.items():
            cmd += ["-e", f"{key}={value}"]
        cmd += ["--workdir", "/workspace", self.image, self.python_bin, *python_args]
        return cmd

    def validate(self) -> list[str]:
        """Return a list of problems that would make generation fail; empty means good to go."""
        problems: list[str] = []
        if shutil.which(self.docker_bin) is None:
            problems.append(f"docker binary {self.docker_bin!r} not found on PATH")
        if not self.polyfly_repo.is_dir():
            problems.append(f"polyfly repo not a directory: {self.polyfly_repo}")
        elif not (self.polyfly_repo / "src" / "poly_fly").is_dir():
            problems.append(f"no src/poly_fly under {self.polyfly_repo}")
        if not (self.shim_dir / "sitecustomize.py").is_file():
            problems.append(f"no sitecustomize.py in shim dir {self.shim_dir}")
        if self.libhsl_dir and not (Path(self.libhsl_dir) / "lib" / "libcoinhsl.so").is_file():
            problems.append(
                f"LIBHSL_DIR={self.libhsl_dir} set but lib/libcoinhsl.so missing; upstream will "
                "warn and silently fall back to the default IPOPT solver"
            )
        return problems


# --------------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One container invocation and the trajectory stems it is expected to produce."""

    job_id: str
    module: str
    args: tuple[str, ...]
    subdir: str
    expected_stems: tuple[str, ...]

    def csv_paths(self, csv_dir: Path) -> list[Path]:
        return [Path(csv_dir) / self.subdir / f"{stem}.csv" for stem in self.expected_stems]


@dataclass
class SolveInfo:
    """Per-trajectory solver provenance scraped from planner stdout."""

    stem: str
    solver_status: str | None = None
    iterations: int | None = None
    solve_time_s: float | None = None
    total_time_s: float | None = None
    succeeded: bool | None = None
    status_source: str = "solver_log"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "solver_status": self.solver_status,
            "iterations": self.iterations,
            "solve_time_s": self.solve_time_s,
            "total_time_s": self.total_time_s,
            "succeeded": self.succeeded,
            "status_source": self.status_source,
        }


@dataclass
class JobResult:
    job: Job
    returncode: int
    duration_s: float
    timed_out: bool
    stdout: str
    solves: list[SolveInfo]
    produced: list[str]
    missing: list[str]

    @property
    def ok(self) -> bool:
        """A job is only OK when the container exited 0 *and* wrote at least one CSV.

        The second half of that condition is the guard against the silent UID-mismatch failure
        described in F1 §2.2.
        """
        return self.returncode == 0 and not self.timed_out and bool(self.produced)


@dataclass
class GenerationReport:
    """Outcome of a batch. Failures are counted, never quietly dropped."""

    results: list[JobResult] = field(default_factory=list)
    skipped: list[Job] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def n_jobs(self) -> int:
        return len(self.results) + len(self.skipped)

    @property
    def failed_jobs(self) -> list[JobResult]:
        return [r for r in self.results if not r.ok]

    @property
    def produced_stems(self) -> list[str]:
        return sorted({stem for r in self.results for stem in r.produced})

    @property
    def missing_stems(self) -> list[str]:
        return sorted({stem for r in self.results for stem in r.missing})

    def format(self) -> str:
        lines = [
            f"jobs                 : {self.n_jobs} "
            f"({len(self.results)} run, {len(self.skipped)} skipped as already complete)",
            f"jobs failed          : {len(self.failed_jobs)}",
            f"trajectories written : {len(self.produced_stems)}",
            f"trajectories missing : {len(self.missing_stems)} "
            f"(solve failed or was not attempted)",
            f"wall time            : {self.wall_time_s:.1f}s",
        ]
        for result in self.failed_jobs:
            reason = "timeout" if result.timed_out else f"exit={result.returncode}"
            if result.returncode == 0 and not result.timed_out:
                reason = "exit=0 but no CSV written (check container UID / bind mount)"
            lines.append(f"  FAILED {result.job.job_id}: {reason}")
        return "\n".join(lines)


def forest_env_seed(base_seed: int) -> int:
    """The environment seed upstream derives from a base seed.

    ``generate_forest.main_mp`` does ``random.seed(base_seed)`` then ``random.getrandbits(32)``
    once per requested iteration. With ``-n 1`` that is a single draw, which lets us predict the
    output filenames (they embed ``_s<env_seed>``) before running anything -- which in turn is what
    makes idempotent re-runs possible. Verified against on-disk output: base seed 7 -> 1390851128.
    """
    import random as _random

    rng = _random.Random()
    rng.seed(base_seed)
    return rng.getrandbits(32)


def _tag(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "n")


def expected_forest_stems(base_seed: int, forest_type: int) -> tuple[str, ...]:
    """Reproduce the filenames a ``-n 1 --seed <base_seed>`` forest run will attempt.

    Mirrors ``run_one_seed``'s ``fname`` construction and ``append_job_lists``' enumeration order.
    Not every stem will exist afterwards: upstream skips ``save_result`` when the solve fails.
    """
    env_seed = forest_env_seed(base_seed)
    stems: list[str] = []
    idx = 0
    for margins, init_vels in FOREST_BLOCKS:
        for goal in FOREST_GOALS:
            for margin in margins:
                for vel in init_vels:
                    goal_tag = "g" + "_".join(_tag(v) for v in goal)
                    margin_tag = _tag(f"m{margin}")
                    vel_tag = "v" + "_".join(_tag(v) for v in vel)
                    stems.append(
                        f"forest_{idx:03d}_f{forest_type}_{goal_tag}_{margin_tag}_{vel_tag}"
                        f"_s{env_seed}"
                    )
                    idx += 1
    return tuple(stems)


def forest_jobs(
    base_seed: int,
    n_environments: int,
    forest_type: int = 0,
    subdir: str = "forests",
) -> list[Job]:
    """One job per environment, each producing up to :data:`TRAJECTORIES_PER_SEED` trajectories."""
    jobs: list[Job] = []
    for offset in range(n_environments):
        seed = base_seed + offset
        jobs.append(
            Job(
                job_id=f"forest_f{forest_type}_seed{seed}",
                module=FOREST_MODULE,
                args=("-n", "1", "--sequential", "--seed", str(seed),
                      "--forest-type", str(forest_type)),
                subdir=subdir,
                expected_stems=expected_forest_stems(seed, forest_type),
            )
        )
    return jobs


def planner_jobs(relative_yamls: Iterable[str]) -> list[Job]:
    """One job per existing params YAML, solved with the single-trajectory planner entry point."""
    jobs: list[Job] = []
    for rel in relative_yamls:
        rel_path = Path(rel)
        stem = rel_path.stem
        subdir = rel_path.parts[0] if len(rel_path.parts) > 1 else ""
        jobs.append(
            Job(
                job_id=f"yaml_{subdir}_{stem}" if subdir else f"yaml_{stem}",
                module=PLANNER_MODULE,
                args=("--yaml", str(rel_path)),
                subdir=subdir,
                expected_stems=(stem,),
            )
        )
    return jobs


def smoke_jobs(relative_yamls: Sequence[str] | None = None) -> list[Job]:
    """Two trajectories, end to end, for CI (F1 §8).

    Uses two of the fixed maze environments rather than a forest seed: a forest run always attempts
    nine solves and cannot be trimmed, whereas two maze YAMLs are exactly two solves at roughly
    6.5 s each, comfortably inside the two-minute budget even with container startup.
    """
    yamls = relative_yamls or ("experiments/maze_2.yaml", "experiments/maze_3.yaml")
    return planner_jobs(yamls)


# --------------------------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------------------------

_RE_OPT_TIME = re.compile(r"Optimization completed in ([\d.eE+-]+) seconds")
_RE_STATUS = re.compile(r"Solver Return Status:\s*(\S+)")
_RE_ITERS = re.compile(r"Number of iterations:\s*(\d+)")
_RE_SOLVER_FAILED = re.compile(r"^Solver failed:")
_RE_TOTAL_TIME = re.compile(r"Total time = ([\d.eE+-]+)")
_RE_SAVED = re.compile(r"^file_dir:\s*(\S+)")
_RE_OPT_FAILED = re.compile(r"^optimization failed for (\S+)")


def parse_planner_log(stdout: str) -> list[SolveInfo]:
    """Extract per-trajectory solver provenance from planner / forest-generator stdout.

    Attribution is anchored on the two lines that name a trajectory: ``file_dir: <subdir>/<stem>``
    printed by ``save_result``, and ``optimization failed for <stem>`` printed by the forest
    generator. The solver lines that precede such an anchor belong to it. Anchoring beats zipping
    solver blocks against a job list, because a solve can abort before IPOPT ever runs
    (``OpenSetEmptyException``) and the two sequences would silently desynchronise.

    ``succeeded`` is derived from the recorded IPOPT status, **not** from which anchor was hit.
    ``save_result`` prints ``file_dir:`` on entry rather than on completion, and the single-YAML
    entry point calls it unconditionally -- so a ``file_dir:`` line means "a save was attempted",
    which is a weaker claim than "a usable trajectory exists". Observed in practice: a maze solve
    that failed with ``Solver_Failed`` still wrote a full-length, entirely plausible-looking CSV
    from the solver's debug values. Nothing about the CSV's shape reveals that; only this status
    does, which is the whole reason the manifest exists.
    """
    solves: list[SolveInfo] = []
    pending: dict[str, Any] = {}

    def flush(stem: str, anchor: str) -> None:
        status = pending.get("status")
        if anchor == "failed":
            succeeded: bool | None = False
        elif status is None:
            succeeded = None
        else:
            succeeded = status in manifest_mod.GOOD_SOLVER_STATUSES
        info = SolveInfo(
            stem=stem,
            solver_status=status,
            iterations=pending.get("iterations"),
            solve_time_s=pending.get("solve_time"),
            total_time_s=pending.get("total_time"),
            succeeded=succeeded,
        )
        solves.append(info)
        pending.clear()

    for raw in stdout.splitlines():
        line = raw.strip()

        m = _RE_OPT_TIME.search(line)
        if m:
            pending["solve_time"] = float(m.group(1))
            continue
        m = _RE_STATUS.search(line)
        if m:
            pending["status"] = m.group(1)
            continue
        m = _RE_ITERS.search(line)
        if m:
            pending["iterations"] = int(m.group(1))
            continue
        if _RE_SOLVER_FAILED.match(line):
            # planner.optimize's RuntimeError branch: sentinels, no usable solution.
            pending["status"] = "Solver_Failed"
            pending["iterations"] = manifest_mod.FAILURE_ITERATION_SENTINEL
            continue
        m = _RE_TOTAL_TIME.search(line)
        if m:
            pending["total_time"] = float(m.group(1))
            continue
        m = _RE_SAVED.match(line)
        if m:
            flush(Path(m.group(1)).stem, anchor="saved")
            continue
        m = _RE_OPT_FAILED.match(line)
        if m:
            flush(Path(m.group(1)).stem, anchor="failed")
            continue

    return solves


# --------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------


def _run_one(
    job: Job,
    config: DockerConfig,
    csv_dir: Path,
    timeout_s: float,
) -> JobResult:
    cmd = config.command(job.module, job.args)
    start = time.time()
    timed_out = False
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        returncode = completed.returncode
        stdout = (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
    duration = time.time() - start

    produced = [
        stem
        for stem, path in zip(job.expected_stems, job.csv_paths(csv_dir))
        if path.is_file()
    ]
    missing = [stem for stem in job.expected_stems if stem not in produced]
    return JobResult(
        job=job,
        returncode=returncode,
        duration_s=duration,
        timed_out=timed_out,
        stdout=stdout,
        solves=parse_planner_log(stdout),
        produced=produced,
        missing=missing,
    )


def _is_complete(job: Job, csv_dir: Path) -> bool:
    """Has this job already produced everything it can?

    Idempotency is per job (one environment seed), because upstream offers no way to re-run a
    single trajectory within a seed. A job counts as complete when every stem it can produce is on
    disk as a non-empty CSV.
    """
    paths = job.csv_paths(csv_dir)
    return bool(paths) and all(p.is_file() and p.stat().st_size > 0 for p in paths)


def run_jobs(
    jobs: Sequence[Job],
    config: DockerConfig,
    *,
    workers: int = 8,
    timeout_s: float = 1800.0,
    force: bool = False,
    log_dir: Path | None = None,
    solve_info_path: Path | None = None,
    on_result: Callable[[JobResult], None] | None = None,
) -> GenerationReport:
    """Run ``jobs`` across ``workers`` containers, checking every exit code.

    Skips jobs whose outputs already exist unless ``force``. Appends solver provenance to
    ``solve_info_path`` and per-job stdout to ``log_dir``.
    """
    csv_dir = config.polyfly_repo / "data" / "csvs"
    report = GenerationReport()

    pending: list[Job] = []
    for job in jobs:
        if not force and _is_complete(job, csv_dir):
            report.skipped.append(job)
        else:
            pending.append(job)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    if solve_info_path is not None:
        Path(solve_info_path).parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    start = time.time()

    def record(result: JobResult) -> None:
        with write_lock:
            if log_dir is not None:
                (Path(log_dir) / f"{result.job.job_id}.log").write_text(
                    result.stdout, encoding="utf-8"
                )
            if solve_info_path is not None:
                with Path(solve_info_path).open("a", encoding="utf-8") as handle:
                    for info in result.solves:
                        handle.write(json.dumps(info.to_dict(), sort_keys=True) + "\n")
            report.results.append(result)
        if on_result is not None:
            on_result(result)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(_run_one, job, config, csv_dir, timeout_s): job for job in pending
            }
            for future in as_completed(futures):
                record(future.result())

    report.wall_time_s = time.time() - start
    return report


# --------------------------------------------------------------------------------------------
# Manifest integration
# --------------------------------------------------------------------------------------------


def rebuild_manifest(
    polyfly_repo: Path,
    manifest_path: Path,
    *,
    subdirs: Sequence[str] | None = None,
    split_seed: int = 0,
    split_fractions: Sequence[float] = manifest_mod.DEFAULT_SPLIT_FRACTIONS,
    split_method: str = "hash",
    solve_info_paths: Sequence[Path] = (),
) -> list[TrajectoryRecord]:
    """Rebuild ``manifest.jsonl`` from the CSVs currently on disk.

    Safe to call at any time, including while generation is still running: it only reads.
    """
    polyfly_repo = Path(polyfly_repo)
    solve_info = manifest_mod.load_solve_info(solve_info_paths)
    records = manifest_mod.build_records(
        csv_dir=polyfly_repo / "data" / "csvs",
        params_dir=polyfly_repo / "data" / "params",
        subdirs=subdirs,
        solve_info=solve_info,
    )
    manifest_mod.apply_splits(
        records, seed=split_seed, fractions=split_fractions, method=split_method
    )
    manifest_mod.write_manifest(records, manifest_path)
    return records


def generate(
    jobs: Sequence[Job],
    config: DockerConfig,
    manifest_path: Path,
    *,
    workers: int = 8,
    timeout_s: float = 1800.0,
    force: bool = False,
    out_dir: Path | None = None,
    split_seed: int = 0,
    split_method: str = "hash",
    subdirs: Sequence[str] | None = None,
    on_result: Callable[[JobResult], None] | None = None,
) -> tuple[GenerationReport, list[TrajectoryRecord]]:
    """Run a batch and rebuild the manifest from whatever it produced."""
    out_dir = Path(out_dir) if out_dir is not None else Path(manifest_path).parent
    solve_info_path = out_dir / "solve_info.jsonl"
    report = run_jobs(
        jobs,
        config,
        workers=workers,
        timeout_s=timeout_s,
        force=force,
        log_dir=out_dir / "logs",
        solve_info_path=solve_info_path,
        on_result=on_result,
    )
    if subdirs is None:
        subdirs = sorted({job.subdir for job in jobs if job.subdir}) or None
    records = rebuild_manifest(
        config.polyfly_repo,
        manifest_path,
        subdirs=subdirs,
        split_seed=split_seed,
        split_method=split_method,
        solve_info_paths=[solve_info_path],
    )
    return report, records
