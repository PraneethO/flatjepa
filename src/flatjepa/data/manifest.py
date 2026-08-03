"""Trajectory manifest: the authoritative index of generated PolyFly trajectories.

Per F1 §6, downstream features consume ``manifest.jsonl`` rather than a directory listing, so
failed or degenerate solves are excluded explicitly rather than by accident. One JSON object per
line, one line per trajectory.

The module owns three things:

1. :class:`TrajectoryRecord` — the record schema and its JSONL round-trip.
2. Quality-filter predicates (F1 §7) — solver status, degenerate paths, and trajectories too short
   to yield a single ``H + T`` training window.
3. Environment-level split assignment (F1 §4, F4 §4) — deterministic given a seed, and assigned at
   the *environment* level, never the window level.

Why the split granularity matters. The upstream forest generator emits nine trajectories per
environment seed (three goals x margin/initial-velocity combinations) that all share the same
obstacle field. Splitting per trajectory would therefore put near-identical obstacle layouts on
both sides of the train/test boundary, and splitting per window would be worse still because
windows within a trajectory overlap by construction. Both leak, and both inflate every probe
metric in F7. Splits here are keyed on the *environment id* parsed from the stem, so all nine
sibling trajectories always land in the same split.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "MANIFEST_VERSION",
    "GOOD_SOLVER_STATUSES",
    "FAILURE_TIME_SENTINEL",
    "FAILURE_PATH_LENGTH_SENTINEL",
    "FAILURE_ITERATION_SENTINEL",
    "MIN_PATH_LENGTH_M",
    "DEFAULT_SPLIT_NAMES",
    "DEFAULT_SPLIT_FRACTIONS",
    "HELDOUT_SPLIT",
    "TrajectoryRecord",
    "parse_stem",
    "resampled_length",
    "quality_issues",
    "passes_quality",
    "filter_records",
    "split_for_env",
    "assign_splits",
    "apply_splits",
    "record_from_csv",
    "build_records",
    "write_manifest",
    "read_manifest",
    "iter_manifest",
    "load_solve_info",
    "summarize",
    "format_summary",
]

MANIFEST_VERSION = 1

#: Solver return statuses IPOPT reports for a usable solution (F1 §7).
GOOD_SOLVER_STATUSES = frozenset({"Solve_Succeeded", "Solved_To_Acceptable_Level"})

#: Upstream ``planner.run`` sets these on failure instead of raising; see planner.py:1447-1457
#: (``total_time = 1000``, ``path_length = 999``) and planner.py:977-978 (``iterations = 99999``,
#: ``opt_time = 999``). They must never be mistaken for real measurements.
FAILURE_TIME_SENTINEL = 1000.0
FAILURE_PATH_LENGTH_SENTINEL = 999.0
FAILURE_ITERATION_SENTINEL = 99999

#: Below this, the payload never meaningfully moved and the trajectory carries no dynamics.
MIN_PATH_LENGTH_M = 1e-2

DEFAULT_SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
DEFAULT_SPLIT_FRACTIONS: tuple[float, float, float] = (0.8, 0.1, 0.1)

#: F4 §4 reserves the nine hand-designed maze environments entirely for qualitative evaluation,
#: so they are never dealt into train/val/test.
HELDOUT_SPLIT = "heldout"

_POSITION_COLUMNS = ("sol_x_0", "sol_x_1", "sol_x_2")

# forest_003_f0_g15_n3_0_m0p5_v1_0_0_s1390851128
#   idx 003, forest type 0, goal (15, -3, 0), margin 0.5, init vel (1, 0, 0), env seed 1390851128
_FOREST_STEM_RE = re.compile(r"^forest_(?P<idx>\d+)_f(?P<ftype>\d+)_.*_s(?P<seed>\d+)$")
_MAZE_STEM_RE = re.compile(r"^(?P<env>maze_\d+)")


@dataclass
class TrajectoryRecord:
    """One generated trajectory.

    Fields required by F1 §6 are ``stem``, ``yaml_path``, ``csv_path``, ``solver_status``,
    ``iterations``, ``solve_time_s``, ``path_length_m``, ``total_time_s`` and ``split``. The rest
    are derived bookkeeping that downstream (F4) needs anyway.

    ``solver_status`` is ``None`` when it could not be recovered. That is the normal case when a
    manifest is rebuilt from CSVs alone: upstream ``generate_forest`` prints the IPOPT return
    status to stdout and *never writes it to disk*, and only writes a CSV at all when the solve
    succeeded (generate_forest.py:533-539). ``status_source`` records which situation applies so
    the quality filter can be explicit rather than guess.
    """

    stem: str
    yaml_path: str | None
    csv_path: str | None
    subdir: str
    source: str  # "forest" | "maze" | "other"
    env_id: str

    solver_status: str | None = None
    status_source: str = "unknown"  # "solver_log" | "inferred_from_output" | "unknown"
    iterations: int | None = None
    solve_time_s: float | None = None

    path_length_m: float | None = None
    total_time_s: float | None = None
    n_timesteps: int | None = None
    dt_s: float | None = None
    sample_rate_hz: float | None = None

    split: str | None = None
    forest_type: int | None = None
    env_seed: int | None = None
    manifest_version: int = MANIFEST_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrajectoryRecord":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in payload.items() if k in known}
        unknown = {k: v for k, v in payload.items() if k not in known}
        if unknown:
            kwargs.setdefault("extra", {})
            kwargs["extra"] = {**unknown, **(kwargs.get("extra") or {})}
        return cls(**kwargs)


# --------------------------------------------------------------------------------------------
# Stem parsing / environment identity
# --------------------------------------------------------------------------------------------


def parse_stem(stem: str) -> dict[str, Any]:
    """Derive environment identity from a trajectory stem.

    All trajectories sharing an ``env_id`` were solved in the *same* obstacle field and must
    therefore share a split.
    """
    m = _FOREST_STEM_RE.match(stem)
    if m:
        ftype = int(m.group("ftype"))
        seed = int(m.group("seed"))
        return {
            "source": "forest",
            "env_id": f"forest_f{ftype}_s{seed}",
            "forest_type": ftype,
            "env_seed": seed,
        }
    m = _MAZE_STEM_RE.match(stem)
    if m:
        return {
            "source": "maze",
            "env_id": m.group("env"),
            "forest_type": None,
            "env_seed": None,
        }
    return {"source": "other", "env_id": stem, "forest_type": None, "env_seed": None}


# --------------------------------------------------------------------------------------------
# Quality filtering (F1 §7)
# --------------------------------------------------------------------------------------------


def resampled_length(n_timesteps: int, stride: int = 1) -> int:
    """Number of samples left after ``arr[::stride]`` (F4 §1 resampling)."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return (int(n_timesteps) + stride - 1) // stride


def quality_issues(
    record: TrajectoryRecord,
    *,
    h: int = 10,
    t: int = 20,
    stride: int = 1,
    allow_unknown_status: bool = True,
    min_path_length_m: float = MIN_PATH_LENGTH_M,
) -> list[str]:
    """Return the reasons ``record`` is unusable; empty list means it passes.

    Reasons are returned rather than a bare bool so the driver can report *why* a corpus shrank.

    ``allow_unknown_status`` defaults to True because upstream only writes a CSV when the solve
    succeeded, so an existing CSV with no recorded status is far more likely to be a successful
    solve whose stdout we never saw than a failure. Set it False to require a positively recorded
    status and exclude anything generated outside this driver.
    """
    issues: list[str] = []

    if record.csv_path is None or record.n_timesteps is None:
        issues.append("missing_csv")

    status = record.solver_status
    if status is None:
        if not allow_unknown_status:
            issues.append("unknown_solver_status")
    elif status not in GOOD_SOLVER_STATUSES:
        issues.append("solver_status")

    if record.iterations is not None and record.iterations >= FAILURE_ITERATION_SENTINEL:
        issues.append("solver_iteration_sentinel")

    length = record.path_length_m
    if length is None:
        if record.n_timesteps is not None:
            issues.append("missing_path_length")
    elif math.isclose(length, FAILURE_PATH_LENGTH_SENTINEL, rel_tol=0.0, abs_tol=1e-9):
        issues.append("path_length_sentinel")
    elif length < min_path_length_m:
        issues.append("degenerate_path")

    total = record.total_time_s
    if total is None:
        if record.n_timesteps is not None:
            issues.append("missing_total_time")
    elif total >= FAILURE_TIME_SENTINEL:
        issues.append("total_time_sentinel")
    elif total <= 0.0:
        issues.append("degenerate_duration")

    if record.n_timesteps is not None:
        if resampled_length(record.n_timesteps, stride) < h + t:
            issues.append("too_short_for_window")

    return issues


def passes_quality(record: TrajectoryRecord, **kwargs: Any) -> bool:
    """True when ``record`` survives every F1 §7 predicate."""
    return not quality_issues(record, **kwargs)


def filter_records(
    records: Iterable[TrajectoryRecord], **kwargs: Any
) -> tuple[list[TrajectoryRecord], dict[str, list[str]]]:
    """Split ``records`` into (kept, {stem: reasons}) for the rejected ones."""
    kept: list[TrajectoryRecord] = []
    rejected: dict[str, list[str]] = {}
    for rec in records:
        issues = quality_issues(rec, **kwargs)
        if issues:
            rejected[rec.stem] = issues
        else:
            kept.append(rec)
    return kept, rejected


# --------------------------------------------------------------------------------------------
# Environment-level splits (F1 §4, F4 §4)
# --------------------------------------------------------------------------------------------


def _validate_fractions(fractions: Sequence[float], names: Sequence[str]) -> None:
    if len(fractions) != len(names):
        raise ValueError(f"got {len(fractions)} fractions for {len(names)} split names")
    if any(f < 0 for f in fractions):
        raise ValueError(f"split fractions must be non-negative, got {fractions}")
    total = sum(fractions)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")


def _unit_hash(seed: int, env_id: str) -> float:
    """Stable float in [0, 1) from (seed, env_id). Independent of Python's hash randomization."""
    digest = hashlib.blake2b(f"{seed}|{env_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2.0**64


def split_for_env(
    env_id: str,
    seed: int,
    fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    names: Sequence[str] = DEFAULT_SPLIT_NAMES,
) -> str:
    """Deterministic split for a single environment, independent of the rest of the corpus.

    This is the default because generation is incremental: hashing each environment on its own
    means new environments can be appended without reshuffling the ones already assigned, which
    would otherwise silently move previously-tested environments into training.
    """
    _validate_fractions(fractions, names)
    u = _unit_hash(seed, env_id)
    cumulative = 0.0
    for name, frac in zip(names, fractions):
        cumulative += frac
        if u < cumulative:
            return name
    return names[-1]


def assign_splits(
    env_ids: Iterable[str],
    seed: int,
    fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    names: Sequence[str] = DEFAULT_SPLIT_NAMES,
    method: str = "hash",
) -> dict[str, str]:
    """Map every environment id to exactly one split name.

    ``method="hash"`` assigns each environment independently (stable under corpus growth, exact
    proportions only in the limit). ``method="shuffle"`` seeds an RNG, shuffles the sorted unique
    environment ids and slices them, giving exact proportions for a *fixed* corpus at the cost of
    reshuffling whenever the corpus changes.
    """
    _validate_fractions(fractions, names)
    unique = sorted(set(env_ids))

    if method == "hash":
        return {env: split_for_env(env, seed, fractions, names) for env in unique}

    if method == "shuffle":
        order = list(unique)
        random.Random(seed).shuffle(order)
        n = len(order)
        counts = [int(n * f) for f in fractions]
        # Hand the rounding remainder to the largest fractions first.
        remainder = n - sum(counts)
        for i in sorted(range(len(fractions)), key=lambda i: -fractions[i])[:remainder]:
            counts[i] += 1
        assignment: dict[str, str] = {}
        start = 0
        for name, count in zip(names, counts):
            for env in order[start : start + count]:
                assignment[env] = name
            start += count
        return assignment

    raise ValueError(f"unknown split method {method!r}; expected 'hash' or 'shuffle'")


def apply_splits(
    records: Sequence[TrajectoryRecord],
    seed: int,
    fractions: Sequence[float] = DEFAULT_SPLIT_FRACTIONS,
    names: Sequence[str] = DEFAULT_SPLIT_NAMES,
    method: str = "hash",
    heldout_sources: Sequence[str] = ("maze",),
) -> dict[str, str]:
    """Assign ``record.split`` in place, at the environment level. Returns the env -> split map.

    Environments whose ``source`` is in ``heldout_sources`` are pulled out entirely and labelled
    :data:`HELDOUT_SPLIT` (F4 §4: the maze set is reserved for qualitative evaluation).
    """
    heldout = {r.env_id for r in records if r.source in heldout_sources}
    splittable = [r.env_id for r in records if r.env_id not in heldout]
    assignment = assign_splits(splittable, seed, fractions, names, method=method)
    assignment.update({env: HELDOUT_SPLIT for env in heldout})
    for rec in records:
        rec.split = assignment[rec.env_id]
    return assignment


# --------------------------------------------------------------------------------------------
# Building records from disk
# --------------------------------------------------------------------------------------------


def _csv_measurements(csv_path: Path) -> dict[str, Any]:
    """Read a trajectory CSV and derive duration, sample rate and payload path length.

    Path length is computed from the saved payload positions rather than the optimizer's own
    ``path_length``, which upstream never persists. The two agree to interpolation error.
    """
    frame = pd.read_csv(csv_path, usecols=["time", *_POSITION_COLUMNS])
    n = int(len(frame))
    if n == 0:
        return {"n_timesteps": 0, "total_time_s": None, "dt_s": None,
                "sample_rate_hz": None, "path_length_m": None}

    time = frame["time"].to_numpy()
    total = float(time[-1] - time[0]) if n > 1 else 0.0
    dt = float((time[-1] - time[0]) / (n - 1)) if n > 1 else None
    positions = frame.loc[:, list(_POSITION_COLUMNS)].to_numpy(dtype=float)
    steps = positions[1:] - positions[:-1]
    path_length = float(np.linalg.norm(steps, axis=1).sum())
    return {
        "n_timesteps": n,
        "total_time_s": total,
        "dt_s": dt,
        "sample_rate_hz": (1.0 / dt) if dt else None,
        "path_length_m": path_length,
    }


def record_from_csv(
    csv_path: Path,
    params_dir: Path,
    subdir: str,
    solve_info: Mapping[str, Mapping[str, Any]] | None = None,
) -> TrajectoryRecord:
    """Build one :class:`TrajectoryRecord` from a generated CSV plus its params YAML."""
    csv_path = Path(csv_path)
    stem = csv_path.stem
    yaml_path = Path(params_dir) / subdir / f"{stem}.yaml"
    identity = parse_stem(stem)

    record = TrajectoryRecord(
        stem=stem,
        yaml_path=str(yaml_path) if yaml_path.exists() else None,
        csv_path=str(csv_path),
        subdir=subdir,
        source=identity["source"],
        env_id=identity["env_id"],
        forest_type=identity["forest_type"],
        env_seed=identity["env_seed"],
    )

    try:
        measurements = _csv_measurements(csv_path)
    except Exception as exc:  # a truncated / half-written CSV must not abort the whole scan
        record.extra["csv_read_error"] = f"{type(exc).__name__}: {exc}"
        return record

    for key, value in measurements.items():
        setattr(record, key, value)

    info = (solve_info or {}).get(stem)
    if info:
        record.solver_status = info.get("solver_status")
        record.iterations = info.get("iterations")
        record.solve_time_s = info.get("solve_time_s")
        record.status_source = info.get("status_source", "solver_log")

    return record


def build_records(
    csv_dir: Path,
    params_dir: Path,
    subdirs: Sequence[str] | None = None,
    solve_info: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[TrajectoryRecord]:
    """Scan ``csv_dir/<subdir>/*.csv`` and build one record per trajectory found on disk."""
    csv_dir = Path(csv_dir)
    params_dir = Path(params_dir)
    if subdirs is None:
        subdirs = sorted(p.name for p in csv_dir.iterdir() if p.is_dir())

    records: list[TrajectoryRecord] = []
    for subdir in subdirs:
        root = csv_dir / subdir
        if not root.is_dir():
            continue
        for csv_path in sorted(root.glob("*.csv")):
            records.append(record_from_csv(csv_path, params_dir, subdir, solve_info))
    return records


# --------------------------------------------------------------------------------------------
# JSONL I/O
# --------------------------------------------------------------------------------------------


def write_manifest(records: Iterable[TrajectoryRecord], path: Path) -> Path:
    """Write ``records`` as JSONL. Written to a sibling temp file then renamed, so a crashed
    run never leaves a half-written manifest that downstream would happily consume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def iter_manifest(path: Path) -> Iterator[TrajectoryRecord]:
    """Stream records from a manifest JSONL file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield TrajectoryRecord.from_dict(json.loads(line))


def read_manifest(path: Path) -> list[TrajectoryRecord]:
    """Read a manifest JSONL file into memory."""
    return list(iter_manifest(path))


def load_solve_info(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Merge solver-log sidecars (``solve_info.jsonl``) written by the generation driver.

    Later files win, so re-solving an environment updates its recorded status.
    """
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                stem = payload.get("stem")
                if stem:
                    merged[stem] = payload
    return merged


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


def summarize(records: Sequence[TrajectoryRecord], **filter_kwargs: Any) -> dict[str, Any]:
    """Aggregate manifest statistics: counts, solver statuses, splits, duration distribution."""
    kept, rejected = filter_records(records, **filter_kwargs)
    reason_counts: Counter[str] = Counter()
    for reasons in rejected.values():
        reason_counts.update(reasons)

    durations = sorted(r.total_time_s for r in kept if r.total_time_s is not None)
    lengths = sorted(r.path_length_m for r in kept if r.path_length_m is not None)
    steps = sorted(r.n_timesteps for r in kept if r.n_timesteps is not None)

    def quantiles(values: Sequence[float]) -> dict[str, float] | None:
        if not values:
            return None
        n = len(values)

        def q(p: float) -> float:
            return float(values[min(n - 1, max(0, int(round(p * (n - 1)))))])

        return {
            "min": float(values[0]),
            "p25": q(0.25),
            "median": q(0.50),
            "p75": q(0.75),
            "max": float(values[-1]),
            "mean": float(sum(values) / n),
        }

    return {
        "n_records": len(records),
        "n_pass": len(kept),
        "n_fail": len(rejected),
        "n_environments": len({r.env_id for r in records}),
        "n_environments_pass": len({r.env_id for r in kept}),
        "sources": dict(Counter(r.source for r in records)),
        "solver_status": dict(
            Counter(r.solver_status if r.solver_status else "<unrecorded>" for r in records)
        ),
        "status_source": dict(Counter(r.status_source for r in records)),
        "splits_trajectories": dict(Counter(r.split or "<unassigned>" for r in kept)),
        "splits_environments": dict(
            Counter(split for split in {r.env_id: (r.split or "<unassigned>") for r in kept}
                    .values())
        ),
        "rejection_reasons": dict(reason_counts),
        "duration_s": quantiles(durations),
        "path_length_m": quantiles(lengths),
        "n_timesteps": quantiles(steps),
        "sample_rate_hz": sorted({round(r.sample_rate_hz, 3) for r in records
                                  if r.sample_rate_hz is not None}),
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    """Render :func:`summarize` output as a short human-readable report."""
    lines: list[str] = []
    lines.append(f"trajectories        : {summary['n_records']}")
    lines.append(f"  pass quality      : {summary['n_pass']}")
    lines.append(f"  fail quality      : {summary['n_fail']}")
    lines.append(f"environments        : {summary['n_environments']} "
                 f"({summary['n_environments_pass']} with passing trajectories)")
    lines.append(f"sources             : {summary['sources']}")
    lines.append(f"solver status       : {summary['solver_status']}")
    lines.append(f"status provenance   : {summary['status_source']}")
    lines.append(f"sample rate (Hz)    : {summary['sample_rate_hz']}")
    lines.append(f"split (trajectories): {summary['splits_trajectories']}")
    lines.append(f"split (environments): {summary['splits_environments']}")
    if summary["rejection_reasons"]:
        lines.append(f"rejection reasons   : {summary['rejection_reasons']}")
    for key, label in (("duration_s", "duration (s)"),
                       ("path_length_m", "path length (m)"),
                       ("n_timesteps", "timesteps")):
        stats = summary.get(key)
        if stats:
            lines.append(
                f"{label:<20}: min={stats['min']:.3f} p25={stats['p25']:.3f} "
                f"med={stats['median']:.3f} p75={stats['p75']:.3f} max={stats['max']:.3f} "
                f"mean={stats['mean']:.3f}"
            )
    return "\n".join(lines)
