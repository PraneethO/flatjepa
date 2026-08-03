"""Tests for the F1 manifest: round-trip, quality predicates, and leak-free splits."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pytest

from flatjepa.data import manifest as m


DEFAULT_STEM = "forest_000_f0_g15_0_0_m0p5_v0_0_0_s123"


def make_record(stem: str = DEFAULT_STEM, **overrides) -> m.TrajectoryRecord:
    identity = m.parse_stem(stem)
    defaults = dict(
        stem=stem,
        yaml_path=f"/params/forests/{stem}.yaml",
        csv_path=f"/csvs/forests/{stem}.csv",
        subdir="forests",
        source=identity["source"],
        env_id=identity["env_id"],
        forest_type=identity["forest_type"],
        env_seed=identity["env_seed"],
        solver_status="Solve_Succeeded",
        status_source="solver_log",
        iterations=120,
        solve_time_s=6.4,
        path_length_m=16.0,
        total_time_s=5.6,
        n_timesteps=57,
        dt_s=0.1,
        sample_rate_hz=10.0,
        split="train",
    )
    defaults.update(overrides)
    return m.TrajectoryRecord(**defaults)


# ---------------------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------------------


def test_manifest_roundtrip_preserves_every_field(tmp_path: Path) -> None:
    records = [
        make_record("forest_000_f0_g15_0_0_m0p5_v0_0_0_s123"),
        make_record("forest_004_f2_g15_n3_0_m0p6_v1_0_0_s999", split="test", iterations=None,
                    solver_status=None, status_source="unknown", solve_time_s=None),
        make_record("maze_1", subdir="experiments", split=m.HELDOUT_SPLIT),
    ]
    path = m.write_manifest(records, tmp_path / "manifest.jsonl")
    loaded = m.read_manifest(path)

    assert len(loaded) == len(records)
    for original, restored in zip(records, loaded):
        assert restored.to_dict() == original.to_dict()


def test_manifest_has_every_field_required_by_f1_section_6(tmp_path: Path) -> None:
    path = m.write_manifest([make_record()], tmp_path / "manifest.jsonl")
    payload = m.read_manifest(path)[0].to_dict()
    required = {
        "stem", "yaml_path", "csv_path", "solver_status", "iterations", "solve_time_s",
        "path_length_m", "total_time_s", "split",
    }
    assert required <= set(payload)
    assert all(payload[key] is not None for key in required)


def test_manifest_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = m.write_manifest([make_record()], tmp_path / "manifest.jsonl")
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_from_dict_tolerates_unknown_keys(tmp_path: Path) -> None:
    payload = make_record().to_dict()
    payload["a_field_from_a_future_version"] = 7
    restored = m.TrajectoryRecord.from_dict(payload)
    assert restored.stem == payload["stem"]
    assert restored.extra["a_field_from_a_future_version"] == 7


# ---------------------------------------------------------------------------------------
# Stem parsing / environment identity
# ---------------------------------------------------------------------------------------


def test_parse_stem_groups_sibling_trajectories_into_one_environment() -> None:
    siblings = [
        "forest_000_f0_g15_0_0_m0p5_v0_0_0_s1390851128",
        "forest_001_f0_g15_0_0_m0p5_v1_0_0_s1390851128",
        "forest_003_f0_g15_n3_0_m0p5_v1_0_0_s1390851128",
        "forest_008_f0_g15_3_0_m0p6_v0_0_0_s1390851128",
    ]
    env_ids = {m.parse_stem(s)["env_id"] for s in siblings}
    assert env_ids == {"forest_f0_s1390851128"}
    assert m.parse_stem(siblings[0])["env_seed"] == 1390851128
    assert m.parse_stem(siblings[0])["forest_type"] == 0


def test_parse_stem_distinguishes_forest_types_and_seeds() -> None:
    a = m.parse_stem("forest_000_f0_g15_0_0_m0p5_v0_0_0_s5")["env_id"]
    b = m.parse_stem("forest_000_f2_g15_0_0_m0p5_v0_0_0_s5")["env_id"]
    c = m.parse_stem("forest_000_f0_g15_0_0_m0p5_v0_0_0_s6")["env_id"]
    assert len({a, b, c}) == 3


def test_parse_stem_recognises_mazes_and_unknowns() -> None:
    assert m.parse_stem("maze_7") == {
        "source": "maze", "env_id": "maze_7", "forest_type": None, "env_seed": None
    }
    assert m.parse_stem("something_else")["source"] == "other"


# ---------------------------------------------------------------------------------------
# Quality filtering (F1 §7)
# ---------------------------------------------------------------------------------------


def test_good_record_passes() -> None:
    assert m.passes_quality(make_record()) is True
    assert m.quality_issues(make_record()) == []


@pytest.mark.parametrize("status", sorted(m.GOOD_SOLVER_STATUSES))
def test_accepted_solver_statuses(status: str) -> None:
    assert m.passes_quality(make_record(solver_status=status))


@pytest.mark.parametrize(
    "status", ["Infeasible_Problem_Detected", "Maximum_Iterations_Exceeded", "Solver_Failed"]
)
def test_rejected_solver_statuses(status: str) -> None:
    assert "solver_status" in m.quality_issues(make_record(solver_status=status))


def test_unknown_status_is_configurable() -> None:
    record = make_record(solver_status=None, status_source="unknown")
    assert m.passes_quality(record, allow_unknown_status=True)
    assert "unknown_solver_status" in m.quality_issues(record, allow_unknown_status=False)


def test_failure_sentinels_are_caught_not_treated_as_measurements() -> None:
    assert "total_time_sentinel" in m.quality_issues(
        make_record(total_time_s=m.FAILURE_TIME_SENTINEL)
    )
    assert "path_length_sentinel" in m.quality_issues(
        make_record(path_length_m=m.FAILURE_PATH_LENGTH_SENTINEL)
    )
    assert "solver_iteration_sentinel" in m.quality_issues(
        make_record(iterations=m.FAILURE_ITERATION_SENTINEL)
    )


def test_degenerate_path_is_rejected() -> None:
    assert "degenerate_path" in m.quality_issues(make_record(path_length_m=1e-9))
    assert "degenerate_duration" in m.quality_issues(make_record(total_time_s=0.0))


def test_missing_csv_is_rejected() -> None:
    issues = m.quality_issues(make_record(csv_path=None, n_timesteps=None))
    assert "missing_csv" in issues


def test_trajectory_shorter_than_one_window_is_rejected() -> None:
    # H + T = 30 samples required.
    assert m.passes_quality(make_record(n_timesteps=30), h=10, t=20, stride=1)
    assert "too_short_for_window" in m.quality_issues(make_record(n_timesteps=29), h=10, t=20)


def test_window_length_check_accounts_for_resampling_stride() -> None:
    # 57 samples at stride 1 is plenty; at stride 25 (500 Hz -> 20 Hz) it is 3 samples.
    record = make_record(n_timesteps=57)
    assert m.passes_quality(record, h=10, t=20, stride=1)
    assert "too_short_for_window" in m.quality_issues(record, h=10, t=20, stride=25)
    assert m.passes_quality(make_record(n_timesteps=1962), h=10, t=20, stride=25)


def test_resampled_length_matches_numpy_slicing_semantics() -> None:
    for n in range(1, 60):
        for stride in (1, 2, 3, 7, 25):
            assert m.resampled_length(n, stride) == len(range(0, n, stride))
    with pytest.raises(ValueError):
        m.resampled_length(10, 0)


def test_filter_records_reports_reasons_per_stem() -> None:
    good = make_record("forest_000_f0_g15_0_0_m0p5_v0_0_0_s1")
    bad = make_record("forest_001_f0_g15_0_0_m0p5_v0_0_0_s1", n_timesteps=5, path_length_m=0.0)
    kept, rejected = m.filter_records([good, bad])
    assert [r.stem for r in kept] == [good.stem]
    assert set(rejected[bad.stem]) == {"degenerate_path", "too_short_for_window"}


# ---------------------------------------------------------------------------------------
# Splits (F1 §4, F4 §4)
# ---------------------------------------------------------------------------------------


ENV_IDS = [f"forest_f0_s{i}" for i in range(500)]


@pytest.mark.parametrize("method", ["hash", "shuffle"])
def test_split_assignment_is_deterministic_given_a_seed(method: str) -> None:
    first = m.assign_splits(ENV_IDS, seed=1234, method=method)
    second = m.assign_splits(list(reversed(ENV_IDS)), seed=1234, method=method)
    assert first == second


@pytest.mark.parametrize("method", ["hash", "shuffle"])
def test_split_assignment_changes_with_the_seed(method: str) -> None:
    a = m.assign_splits(ENV_IDS, seed=0, method=method)
    b = m.assign_splits(ENV_IDS, seed=1, method=method)
    assert a != b


def test_hash_split_is_stable_when_new_environments_are_added() -> None:
    """Incremental generation must not reshuffle already-assigned environments."""
    before = m.assign_splits(ENV_IDS[:100], seed=7, method="hash")
    after = m.assign_splits(ENV_IDS, seed=7, method="hash")
    assert all(after[env] == split for env, split in before.items())


def test_shuffle_split_hits_the_exact_requested_ratios() -> None:
    assignment = m.assign_splits(ENV_IDS, seed=7, fractions=(0.8, 0.1, 0.1), method="shuffle")
    counts = Counter(assignment.values())
    assert counts == {"train": 400, "val": 50, "test": 50}


def test_hash_split_approaches_the_requested_ratios() -> None:
    env_ids = [f"forest_f0_s{i}" for i in range(20000)]
    counts = Counter(m.assign_splits(env_ids, seed=3, method="hash").values())
    for name, expected in zip(m.DEFAULT_SPLIT_NAMES, m.DEFAULT_SPLIT_FRACTIONS):
        assert math.isclose(counts[name] / len(env_ids), expected, abs_tol=0.02)


@pytest.mark.parametrize("method", ["hash", "shuffle"])
def test_every_environment_lands_in_exactly_one_split(method: str) -> None:
    assignment = m.assign_splits(ENV_IDS, seed=99, method=method)
    assert set(assignment) == set(ENV_IDS)
    assert all(v in m.DEFAULT_SPLIT_NAMES for v in assignment.values())


def test_rejects_fractions_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError):
        m.assign_splits(ENV_IDS, seed=0, fractions=(0.8, 0.1, 0.2))
    with pytest.raises(ValueError):
        m.assign_splits(ENV_IDS, seed=0, fractions=(0.5, 0.5))
    with pytest.raises(ValueError):
        m.assign_splits(ENV_IDS, seed=0, method="stratified")


def _corpus() -> list[m.TrajectoryRecord]:
    records: list[m.TrajectoryRecord] = []
    for seed in range(60):
        for idx in range(9):
            records.append(
                make_record(f"forest_{idx:03d}_f0_g15_0_0_m0p5_v0_0_0_s{seed}", split=None)
            )
    for i in range(1, 10):
        records.append(make_record(f"maze_{i}", subdir="experiments", split=None))
    return records


@pytest.mark.parametrize("method", ["hash", "shuffle"])
def test_no_trajectory_appears_in_more_than_one_split(method: str) -> None:
    records = _corpus()
    m.apply_splits(records, seed=42, method=method)
    by_stem: dict[str, set[str]] = {}
    for record in records:
        assert record.split is not None
        by_stem.setdefault(record.stem, set()).add(record.split)
    assert all(len(splits) == 1 for splits in by_stem.values())


@pytest.mark.parametrize("method", ["hash", "shuffle"])
def test_no_environment_is_shared_between_splits(method: str) -> None:
    """The leak F4 §4 warns about: sibling trajectories from one obstacle field must not straddle
    the train/test boundary."""
    records = _corpus()
    m.apply_splits(records, seed=42, method=method)
    by_env: dict[str, set[str]] = {}
    for record in records:
        by_env.setdefault(record.env_id, set()).add(record.split)
    offenders = {env: splits for env, splits in by_env.items() if len(splits) > 1}
    assert offenders == {}

    # And the split partitions the environments: intersections are empty.
    members = {name: set() for name in (*m.DEFAULT_SPLIT_NAMES, m.HELDOUT_SPLIT)}
    for env, splits in by_env.items():
        members[next(iter(splits))].add(env)
    all_envs = set(by_env)
    assert set().union(*members.values()) == all_envs
    assert sum(len(v) for v in members.values()) == len(all_envs)


def test_mazes_are_held_out_entirely() -> None:
    records = _corpus()
    m.apply_splits(records, seed=42)
    maze_splits = {r.split for r in records if r.source == "maze"}
    assert maze_splits == {m.HELDOUT_SPLIT}
    assert all(r.split != m.HELDOUT_SPLIT for r in records if r.source == "forest")


def test_apply_splits_is_deterministic_across_calls() -> None:
    a, b = _corpus(), _corpus()
    m.apply_splits(a, seed=5)
    m.apply_splits(b, seed=5)
    assert [r.split for r in a] == [r.split for r in b]


# ---------------------------------------------------------------------------------------
# Building from disk
# ---------------------------------------------------------------------------------------


def _write_csv(path: Path, n: int, dt: float = 0.1, step: float = 0.3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "time,sol_x_0,sol_x_1,sol_x_2\n"
    rows = "".join(f"{i * dt},{i * step},0.0,0.0\n" for i in range(n))
    path.write_text(header + rows, encoding="utf-8")


def test_build_records_derives_measurements_from_csv(tmp_path: Path) -> None:
    stem = "forest_000_f0_g15_0_0_m0p5_v0_0_0_s42"
    _write_csv(tmp_path / "csvs" / "forests" / f"{stem}.csv", n=51)
    (tmp_path / "params" / "forests").mkdir(parents=True)
    (tmp_path / "params" / "forests" / f"{stem}.yaml").write_text("dt: 0.1\n", encoding="utf-8")

    records = m.build_records(tmp_path / "csvs", tmp_path / "params")
    assert len(records) == 1
    record = records[0]
    assert record.n_timesteps == 51
    assert record.total_time_s == pytest.approx(5.0)
    assert record.dt_s == pytest.approx(0.1)
    assert record.sample_rate_hz == pytest.approx(10.0)
    assert record.path_length_m == pytest.approx(15.0)
    assert record.yaml_path is not None
    assert record.env_id == "forest_f0_s42"


def test_build_records_merges_solver_provenance(tmp_path: Path) -> None:
    stem = "forest_000_f0_g15_0_0_m0p5_v0_0_0_s42"
    _write_csv(tmp_path / "csvs" / "forests" / f"{stem}.csv", n=51)
    solve_info = {stem: {"solver_status": "Solve_Succeeded", "iterations": 88,
                         "solve_time_s": 6.1, "status_source": "solver_log"}}
    record = m.build_records(tmp_path / "csvs", tmp_path / "params", solve_info=solve_info)[0]
    assert record.solver_status == "Solve_Succeeded"
    assert record.iterations == 88
    assert record.status_source == "solver_log"


def test_build_records_survives_an_unreadable_csv(tmp_path: Path) -> None:
    """A half-written CSV from a job still in flight must not abort the whole scan."""
    root = tmp_path / "csvs" / "forests"
    root.mkdir(parents=True)
    (root / "forest_000_f0_g15_0_0_m0p5_v0_0_0_s1.csv").write_text("not,a,valid\nheader", "utf-8")
    _write_csv(root / "forest_001_f0_g15_0_0_m0p5_v0_0_0_s1.csv", n=51)

    records = m.build_records(tmp_path / "csvs", tmp_path / "params")
    assert len(records) == 2
    broken = next(r for r in records if r.stem.startswith("forest_000"))
    assert "csv_read_error" in broken.extra
    assert "missing_csv" in m.quality_issues(broken)


def test_load_solve_info_merges_with_later_entries_winning(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text('{"stem": "x", "solver_status": "Solver_Failed"}\n', encoding="utf-8")
    second.write_text('{"stem": "x", "solver_status": "Solve_Succeeded"}\n'
                      '{"stem": "y", "solver_status": "Solve_Succeeded"}\n', encoding="utf-8")
    merged = m.load_solve_info([first, second, tmp_path / "missing.jsonl"])
    assert merged["x"]["solver_status"] == "Solve_Succeeded"
    assert set(merged) == {"x", "y"}


def test_summarize_counts_and_formats(tmp_path: Path) -> None:
    records = _corpus()
    records.append(make_record("forest_000_f0_g15_0_0_m0p5_v0_0_0_s999", n_timesteps=4))
    m.apply_splits(records, seed=0)
    summary = m.summarize(records)
    assert summary["n_records"] == len(records)
    assert summary["n_pass"] + summary["n_fail"] == summary["n_records"]
    assert summary["rejection_reasons"]["too_short_for_window"] == 1
    assert "split (environments)" in m.format_summary(summary)
