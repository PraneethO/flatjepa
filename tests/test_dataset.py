"""Tests for F4 dataset assembly.

Focused on the invariants whose violation is silent and inflates every downstream metric:
train-only normalisation, environment-level splits, and metadata sufficient to reproduce a build.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from flatjepa.data.dataset import (
    SPLITS,
    NormalizationStats,
    WindowedDataset,
    build_dataset,
    compute_normalization,
)
from flatjepa.data.manifest import TrajectoryRecord
from flatjepa.data.targets import TOTAL_TARGET_DIM
from flatjepa.data.windows import WindowConfig


def _write_csv(path: Path, n: int = 60, dt: float = 0.1, offset: float = 0.0) -> None:
    """Write a planner-format CSV with the full 35-column schema."""
    t = np.arange(n) * dt
    pos = np.stack([np.sin(t) + offset, np.cos(t), 2.0 + 0.1 * t], axis=-1)
    vel = np.stack([np.cos(t), -np.sin(t), 0.1 * np.ones_like(t)], axis=-1)
    acc = np.stack([-np.sin(t), -np.cos(t), np.zeros_like(t)], axis=-1)
    jerk = np.stack([-np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1)

    cols = {"time": t}
    for i in range(3):
        cols[f"sol_x_{i}"] = pos[:, i]
        cols[f"sol_x_{i + 3}"] = vel[:, i]
        cols[f"sol_x_{i + 6}"] = acc[:, i]
        cols[f"sol_u_{i}"] = jerk[:, i]
        cols[f"sol_quad_x_{i}"] = pos[:, i]
        cols[f"sol_quad_x_{i + 3}"] = vel[:, i]
        cols[f"sol_quad_x_{i + 6}"] = acc[:, i]
        cols[f"sol_payload_rpy_{i}"] = np.zeros(n)
    for i in range(4):
        cols[f"sol_quad_quat_{i}"] = np.full(n, 1.0 if i == 3 else 0.0)
    for i in range(6):
        cols[f"rot_mat_{i}"] = np.full(n, 1.0 if i in (0, 4) else 0.0)

    header = ",".join(cols)
    rows = np.stack(list(cols.values()), axis=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


@pytest.fixture
def corpus(tmp_path: Path):
    """Six trajectories across three environments, split env-wise."""
    records = []
    layout = [("train", 3), ("val", 1), ("test", 2)]
    k = 0
    for split, count in layout:
        for _ in range(count):
            stem = f"forest_{k:03d}_f0_s{1000 + k}"
            csv = tmp_path / "csvs" / f"{stem}.csv"
            _write_csv(csv, n=60, offset=float(k))
            records.append(
                TrajectoryRecord(
                    stem=stem,
                    yaml_path=None,
                    csv_path=str(csv),
                    subdir="forests",
                    source="forest",
                    env_id=f"env_{k}",
                    split=split,
                )
            )
            k += 1
    return records, tmp_path


# ------------------------------------------------------------------ normalization


def test_normalization_is_per_channel():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=[1.0, -5.0, 100.0], scale=[0.5, 2.0, 10.0], size=(500, 4, 3))
    stats = compute_normalization({"x": arr})
    np.testing.assert_allclose(stats.mean["x"], [1.0, -5.0, 100.0], atol=0.2)
    np.testing.assert_allclose(stats.std["x"], [0.5, 2.0, 10.0], rtol=0.15)
    out = stats.apply("x", arr)
    np.testing.assert_allclose(out.reshape(-1, 3).mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(out.reshape(-1, 3).std(axis=0), 1.0, atol=1e-6)


def test_constant_channel_gets_unit_std_not_epsilon():
    """Dividing a constant channel by eps would amplify float noise into huge values."""
    arr = np.concatenate(
        [np.random.default_rng(1).normal(size=(100, 2, 1)), np.zeros((100, 2, 1))], axis=-1
    )
    stats = compute_normalization({"x": arr})
    assert stats.std["x"][1] == 1.0
    assert np.isfinite(stats.apply("x", arr)).all()


def test_normalization_json_roundtrip():
    stats = compute_normalization({"x": np.random.default_rng(2).normal(size=(50, 3, 2))})
    restored = NormalizationStats.from_json(json.loads(json.dumps(stats.to_json())))
    assert restored.mean == stats.mean
    assert restored.std == stats.std
    assert restored.source_split == "train"


# ------------------------------------------------------------------ build


def test_build_produces_all_splits(corpus):
    records, tmp = corpus
    out = tmp / "ds"
    report = build_dataset(records, out, config=WindowConfig(history=10, horizon=20))

    assert report.total_windows() > 0
    for split in SPLITS:
        assert (out / split / "state_hist.npy").exists()
        assert report.windows_per_split[split] > 0
    assert report.trajectories_per_split["train"] == 3
    assert report.trajectories_per_split["test"] == 2


def test_normalization_uses_training_split_only(corpus):
    """The leak test: stats must be identical whether or not val/test exist at all."""
    records, tmp = corpus
    cfg = WindowConfig(history=10, horizon=20)

    build_dataset(records, tmp / "full", config=cfg)
    full = json.loads((tmp / "full" / "normalization.json").read_text())

    train_only = [r for r in records if r.split == "train"]
    # Keep one val/test record so the build does not fail on an empty split.
    keep = [r for r in records if r.split != "train"][:2]
    build_dataset(train_only + keep, tmp / "partial", config=cfg)
    partial = json.loads((tmp / "partial" / "normalization.json").read_text())

    assert full["mean"] == partial["mean"], "val/test data leaked into normalization statistics"
    assert full["std"] == partial["std"]
    assert full["source_split"] == "train"


def test_records_outside_known_splits_are_ignored(corpus):
    records, tmp = corpus
    for r in records:
        if r.split == "test":
            r.split = "heldout"
    report = build_dataset(records, tmp / "ds", config=WindowConfig(history=10, horizon=20))
    assert report.windows_per_split["test"] == 0
    assert report.trajectories_per_split["test"] == 0


def test_empty_training_split_refuses_to_build(corpus):
    records, tmp = corpus
    for r in records:
        r.split = "test"
    with pytest.raises(ValueError, match="training split"):
        build_dataset(records, tmp / "ds", config=WindowConfig(history=10, horizon=20))


def test_short_trajectories_are_skipped_not_crashed(corpus, tmp_path):
    records, tmp = corpus
    short = tmp / "csvs" / "forest_short_f0_s9999.csv"
    _write_csv(short, n=8)
    rec = TrajectoryRecord(
        stem="forest_short_f0_s9999",
        yaml_path=None,
        csv_path=str(short),
        subdir="forests",
        source="forest",
        env_id="env_short",
        split="train",
    )

    report = build_dataset(records + [rec], tmp / "ds", config=WindowConfig(history=10, horizon=20))
    assert "forest_short_f0_s9999" in report.skipped_too_short


def test_unreadable_csv_is_recorded_not_fatal(corpus):
    records, tmp = corpus
    bad = tmp / "csvs" / "broken.csv"
    bad.write_text("not,a,planner,csv\n1,2,3,4\n")
    rec = TrajectoryRecord(
        stem="broken",
        yaml_path=None,
        csv_path=str(bad),
        subdir="forests",
        source="other",
        env_id="env_bad",
        split="train",
    )

    report = build_dataset(records + [rec], tmp / "ds", config=WindowConfig(history=10, horizon=20))
    assert "broken" in report.skipped_load_error
    assert report.total_windows() > 0, "one bad file must not abort the corpus"


# ------------------------------------------------------------------ metadata and reading


def test_metadata_records_everything_needed_to_reproduce(corpus):
    records, tmp = corpus
    cfg = WindowConfig(history=8, horizon=12, observed_fields=("payload_pos", "payload_vel"))
    build_dataset(records, tmp / "ds", config=cfg, extra_metadata={"commit": "abc123"})

    meta = json.loads((tmp / "ds" / "metadata.json").read_text())
    assert meta["window"]["history"] == 8
    assert meta["window"]["horizon"] == 12
    assert meta["window"]["observed_fields"] == ["payload_pos", "payload_vel"]
    assert meta["window"]["obs_dim"] == 6
    assert meta["commit"] == "abc123"
    assert meta["targets"]["total_dim"] == TOTAL_TARGET_DIM
    assert 10 in {int(k) for k in meta["sample_rates_hz"]}


def test_windowed_dataset_reads_back(corpus):
    records, tmp = corpus
    cfg = WindowConfig(history=10, horizon=20)
    report = build_dataset(records, tmp / "ds", config=cfg)

    ds = WindowedDataset(tmp / "ds", "train")
    assert len(ds) == report.windows_per_split["train"]

    item = ds[0]
    assert item["state_hist"].shape == (10, 9)
    assert item["action_future"].shape == (20, 3)
    assert item["targets"].shape == (TOTAL_TARGET_DIM,)

    assert ds.target("cable_dir").shape == (len(ds), 3)
    assert ds.target("R_quad_cols").shape == (len(ds), 6)
    assert ds.flat_inputs().shape == (len(ds), 10 * 9)

    with pytest.raises(KeyError):
        ds.target("no_such_target")


def test_missing_split_directory_raises(corpus):
    records, tmp = corpus
    build_dataset(records, tmp / "ds", config=WindowConfig(history=10, horizon=20))
    with pytest.raises(FileNotFoundError):
        WindowedDataset(tmp / "ds", "nonexistent")


def test_unnormalized_build_preserves_raw_values(corpus):
    records, tmp = corpus
    cfg = WindowConfig(history=10, horizon=20)
    build_dataset(records, tmp / "raw", config=cfg, normalize=False)
    ds = WindowedDataset(tmp / "raw", "train")
    # Anchor position is exactly zero by construction when values are not normalised.
    np.testing.assert_allclose(np.asarray(ds.arrays["state_hist"])[:, -1, 0:3], 0.0, atol=1e-6)
