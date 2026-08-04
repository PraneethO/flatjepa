"""Dataset assembly for F4.

Walks the F1 manifest, cuts each trajectory into windows (F2 ground truth attached), computes
normalisation statistics **on the training split only**, and writes memory-mappable arrays plus
metadata sufficient to reproduce the build.

Two invariants this module exists to enforce, both of which silently inflate every downstream
metric when violated:

* **Splits are environment-level, never window-level.** Windows within a trajectory overlap by
  construction; trajectories within an environment share an obstacle field. A random window split
  places near-duplicates on both sides. Split assignment comes from the manifest and is never
  re-derived here.
* **Normalisation statistics come from the training split alone.** Computing them over the full
  corpus leaks the test distribution into training.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np

from flatjepa.data.flatness import SystemParams
from flatjepa.data.targets import TARGET_NAMES, TOTAL_TARGET_DIM, target_slices
from flatjepa.data.windows import WindowConfig

if TYPE_CHECKING:  # pragma: no cover
    from flatjepa.data.manifest import TrajectoryRecord

# `csv_io` pulls in pandas and `manifest` pulls in the CSV stack; both are needed only to *build* a
# dataset, never to read one. Reading is what the training container does, and it has no pandas.
# Keeping these imports inside the build path lets the training image stay minimal.

SPLITS: tuple[str, ...] = ("train", "val", "test")

# Arrays written per split. Name -> (per-window shape suffix).
_ARRAY_NAMES: tuple[str, ...] = (
    "state_hist",
    "action_hist",
    "action_future",
    "state_future",
    "targets",
)


@dataclass
class NormalizationStats:
    """Per-channel mean/std for each array, computed on the training split only."""

    mean: dict[str, list[float]]
    std: dict[str, list[float]]
    source_split: str = "train"
    n_windows: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping) -> "NormalizationStats":
        return cls(
            mean={k: list(v) for k, v in d["mean"].items()},
            std={k: list(v) for k, v in d["std"].items()},
            source_split=d.get("source_split", "train"),
            n_windows=int(d.get("n_windows", 0)),
        )

    def apply(self, name: str, arr: np.ndarray) -> np.ndarray:
        """Normalise ``arr`` (..., C) using the stats for ``name``."""
        mean = np.asarray(self.mean[name], dtype=arr.dtype)
        std = np.asarray(self.std[name], dtype=arr.dtype)
        return (arr - mean) / std


def compute_normalization(
    arrays: Mapping[str, np.ndarray], eps: float = 1e-6, source_split: str = "train"
) -> NormalizationStats:
    """Per-channel statistics over every leading axis except the last.

    A channel with (near-)zero variance gets std 1.0 rather than ``eps``: dividing by a tiny number
    would amplify float noise into a large normalised value, which is worse than leaving a constant
    channel at zero.
    """
    mean: dict[str, list[float]] = {}
    std: dict[str, list[float]] = {}
    n = 0
    for name, arr in arrays.items():
        if arr.size == 0:
            width = arr.shape[-1] if arr.ndim else 0
            mean[name] = [0.0] * width
            std[name] = [1.0] * width
            continue
        flat = arr.reshape(-1, arr.shape[-1])
        n = max(n, arr.shape[0])
        m = flat.mean(axis=0)
        s = flat.std(axis=0)
        s = np.where(s < eps, 1.0, s)
        mean[name] = [float(x) for x in m]
        std[name] = [float(x) for x in s]
    return NormalizationStats(mean=mean, std=std, source_split=source_split, n_windows=n)


@dataclass
class BuildReport:
    """What a build actually did, including everything it refused."""

    windows_per_split: dict[str, int] = field(default_factory=dict)
    trajectories_per_split: dict[str, int] = field(default_factory=dict)
    skipped_too_short: list[str] = field(default_factory=list)
    skipped_all_invalid: list[str] = field(default_factory=list)
    skipped_load_error: dict[str, str] = field(default_factory=dict)
    windows_dropped_invalid: int = 0
    sample_rates: dict[str, int] = field(default_factory=dict)

    def total_windows(self) -> int:
        return sum(self.windows_per_split.values())

    def format(self) -> str:
        lines = ["windows per split:"]
        for s in SPLITS:
            lines.append(
                f"  {s:<6} {self.windows_per_split.get(s, 0):>8} windows"
                f"  from {self.trajectories_per_split.get(s, 0):>5} trajectories"
            )
        lines.append(f"total windows        : {self.total_windows()}")
        lines.append(f"dropped (F2 invalid) : {self.windows_dropped_invalid}")
        lines.append(f"skipped too short    : {len(self.skipped_too_short)}")
        lines.append(f"skipped all-invalid  : {len(self.skipped_all_invalid)}")
        lines.append(f"skipped load errors  : {len(self.skipped_load_error)}")
        if self.sample_rates:
            rates = ", ".join(f"{hz} Hz x{n}" for hz, n in sorted(self.sample_rates.items()))
            lines.append(f"sample rates         : {rates}")
        return "\n".join(lines)


def _trajectory_windows(
    record: "TrajectoryRecord",
    config: WindowConfig,
    params: SystemParams,
    report: BuildReport,
) -> dict[str, np.ndarray] | None:
    """Cut one trajectory, returning packed arrays or ``None`` if it yielded nothing usable."""
    from flatjepa.data.csv_io import load_trajectory_csv

    try:
        traj = load_trajectory_csv(record.csv_path)
    except Exception as exc:  # noqa: BLE001 - we want the reason, not a crash mid-corpus
        report.skipped_load_error[record.stem] = f"{type(exc).__name__}: {exc}"
        return None

    reach = (config.span - 1) * config.stride
    if traj.n_steps <= reach:
        report.skipped_too_short.append(record.stem)
        return None

    hz = int(round(1.0 / traj.dt)) if traj.dt == traj.dt and traj.dt > 0 else 0
    report.sample_rates[hz] = report.sample_rates.get(hz, 0) + 1

    from flatjepa.data.flatness import flat_outputs_from_trajectory

    flat = flat_outputs_from_trajectory(traj, params=params)

    # How many windows would have existed without the validity filter, so the drop count is real.
    from flatjepa.data.windows import extract_windows, window_indices

    n_candidate = window_indices(traj.n_steps, config).size
    windows = extract_windows(traj, flat=flat, config=config)
    report.windows_dropped_invalid += max(0, n_candidate - windows.n_windows)

    if windows.n_windows == 0:
        report.skipped_all_invalid.append(record.stem)
        return None

    from flatjepa.data.targets import pack_targets

    targets = pack_targets(flat, windows.anchor_index, origin=windows.anchor_origin)

    return {
        "state_hist": windows.state_hist.astype(np.float32),
        "action_hist": windows.action_hist.astype(np.float32),
        "action_future": windows.action_future.astype(np.float32),
        "state_future": windows.state_future.astype(np.float32),
        "targets": targets.astype(np.float32),
    }


def build_dataset(
    records: Sequence["TrajectoryRecord"],
    out_dir: str | Path,
    config: WindowConfig | None = None,
    params: SystemParams | None = None,
    normalize: bool = True,
    extra_metadata: Mapping | None = None,
) -> BuildReport:
    """Build a windowed dataset from manifest ``records`` into ``out_dir``.

    ``records`` must already carry split assignments (see :mod:`flatjepa.data.manifest`); records
    with a split outside :data:`SPLITS` -- for example the held-out maze set -- are ignored.
    """
    config = config or WindowConfig()
    params = params or SystemParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = BuildReport()

    per_split: dict[str, dict[str, list[np.ndarray]]] = {
        s: {name: [] for name in _ARRAY_NAMES} for s in SPLITS
    }
    traj_counts = {s: 0 for s in SPLITS}

    for record in records:
        split = getattr(record, "split", None)
        if split not in SPLITS:
            continue
        packed = _trajectory_windows(record, config, params, report)
        if packed is None:
            continue
        for name, arr in packed.items():
            per_split[split][name].append(arr)
        traj_counts[split] += 1

    # Concatenate per split.
    concatenated: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        concatenated[split] = {
            name: (
                np.concatenate(chunks, axis=0)
                if chunks
                else _empty_for(name, config)
            )
            for name, chunks in per_split[split].items()
        }
        report.windows_per_split[split] = int(concatenated[split]["state_hist"].shape[0])
        report.trajectories_per_split[split] = traj_counts[split]

    if report.windows_per_split.get("train", 0) == 0:
        raise ValueError(
            "training split produced zero windows; refusing to write a dataset that cannot be "
            "normalised. Check the manifest split assignment and trajectory lengths."
        )

    stats = compute_normalization(concatenated["train"])

    for split in SPLITS:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in concatenated[split].items():
            if normalize and name in stats.mean:
                arr = stats.apply(name, arr).astype(np.float32)
            np.save(split_dir / f"{name}.npy", arr)

    metadata = {
        "window": {
            "history": config.history,
            "horizon": config.horizon,
            "stride": config.stride,
            "observed_fields": list(config.observed_fields),
            "obs_dim": config.obs_dim,
            "channel_names": config.channel_names(),
            "require_valid": config.require_valid,
        },
        "targets": {
            "names": list(TARGET_NAMES),
            "total_dim": TOTAL_TARGET_DIM,
            "slices": {k: [v.start, v.stop] for k, v in target_slices().items()},
        },
        "system_params": asdict(params) if hasattr(params, "__dataclass_fields__") else {},
        "normalized": normalize,
        "splits": {s: report.windows_per_split.get(s, 0) for s in SPLITS},
        "trajectories": {s: report.trajectories_per_split.get(s, 0) for s in SPLITS},
        "sample_rates_hz": report.sample_rates,
        "note": (
            "Forest data is 10 Hz native (planner knots). Do not resample; the 20 Hz target in an "
            "early revision of F4 is unreachable."
        ),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    (out_dir / "normalization.json").write_text(json.dumps(stats.to_json(), indent=2, sort_keys=True))
    (out_dir / "build_report.txt").write_text(report.format() + "\n")

    return report


def _empty_for(name: str, config: WindowConfig) -> np.ndarray:
    if name == "state_hist":
        return np.empty((0, config.history, config.obs_dim), dtype=np.float32)
    if name == "state_future":
        return np.empty((0, config.horizon, config.obs_dim), dtype=np.float32)
    if name == "action_hist":
        return np.empty((0, config.history, 3), dtype=np.float32)
    if name == "action_future":
        return np.empty((0, config.horizon, 3), dtype=np.float32)
    if name == "targets":
        return np.empty((0, TOTAL_TARGET_DIM), dtype=np.float32)
    raise KeyError(name)


class WindowedDataset:
    """Memory-mapped read access to one split of a built dataset.

    Deliberately not a ``torch.utils.data.Dataset`` subclass: this module stays torch-free so the
    data layer can be exercised without a torch install. Wrapping it for torch is a one-liner in the
    training harness.
    """

    def __init__(self, root: str | Path, split: str, mmap: bool = True):
        self.root = Path(root)
        self.split = split
        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"no such split directory: {split_dir}")

        mode = "r" if mmap else None
        self.arrays: dict[str, np.ndarray] = {
            name: np.load(split_dir / f"{name}.npy", mmap_mode=mode) for name in _ARRAY_NAMES
        }

        meta_path = self.root / "metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        norm_path = self.root / "normalization.json"
        self.normalization = (
            NormalizationStats.from_json(json.loads(norm_path.read_text()))
            if norm_path.exists()
            else None
        )
        self.target_slices = {
            k: slice(v[0], v[1])
            for k, v in self.metadata.get("targets", {}).get("slices", {}).items()
        }

    def __len__(self) -> int:
        return int(self.arrays["state_hist"].shape[0])

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        return {name: np.asarray(arr[i]) for name, arr in self.arrays.items()}

    def target(self, name: str) -> np.ndarray:
        """All windows' values for one named probe target, shape (N, width)."""
        if name not in self.target_slices:
            raise KeyError(f"unknown target {name!r}; have {sorted(self.target_slices)}")
        return np.asarray(self.arrays["targets"][:, self.target_slices[name]])

    def flat_inputs(self) -> np.ndarray:
        """History window flattened to (N, H*D). This is F7's raw-input-window control."""
        sh = self.arrays["state_hist"]
        return np.asarray(sh).reshape(sh.shape[0], -1)
