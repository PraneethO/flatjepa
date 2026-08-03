"""Tests for the F1 generation driver.

These exercise everything that does not require docker: command construction, output-name
prediction, log parsing, exit-code accounting and idempotent skipping. The container itself is
covered by ``scripts/generate_data.py smoke``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from flatjepa.data import generate as gen
from flatjepa.data import manifest as m


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "polyfly_ral"
    (repo / "src" / "poly_fly").mkdir(parents=True)
    (repo / "data" / "csvs" / "forests").mkdir(parents=True)
    (repo / "data" / "params" / "forests").mkdir(parents=True)
    return repo


@pytest.fixture()
def shim(tmp_path: Path) -> Path:
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "sitecustomize.py").write_text("import matplotlib\n", encoding="utf-8")
    return shim_dir


@pytest.fixture()
def config(fake_repo: Path, shim: Path) -> gen.DockerConfig:
    return gen.DockerConfig(polyfly_repo=fake_repo, shim_dir=shim)


# ---------------------------------------------------------------------------------------
# Docker command (the two F1 §2 blockers)
# ---------------------------------------------------------------------------------------


def test_command_carries_both_blocker_workarounds(config: gen.DockerConfig) -> None:
    cmd = config.command(gen.PLANNER_MODULE, ("--yaml", "experiments/maze_1.yaml"))
    joined = " ".join(cmd)

    # Blocker 2.2: container UID mismatch silently produces no output.
    assert "--user" in cmd
    import os

    assert f"{os.getuid()}:{os.getgid()}" in cmd
    assert "HOME=/tmp" in cmd

    # Blocker 2.1: headless matplotlib, via the shim ahead of the upstream package.
    assert "PYTHONPATH=/shim:/workspace/src" in cmd
    assert "MPLBACKEND=Agg" in cmd
    assert f"{config.shim_dir}:/shim:ro" in joined

    assert f"{config.polyfly_repo}:/workspace:rw" in joined
    assert "POLYFLY_DIR=/workspace" in cmd
    assert cmd[-4:] == ["-m", gen.PLANNER_MODULE, "--yaml", "experiments/maze_1.yaml"]


def test_libhsl_dir_is_absent_when_unset(config: gen.DockerConfig) -> None:
    assert not any("LIBHSL" in part for part in config.command(gen.PLANNER_MODULE, ()))


def test_libhsl_dir_is_mounted_and_exported_when_set(fake_repo: Path, shim: Path,
                                                     tmp_path: Path) -> None:
    hsl = tmp_path / "hsl"
    (hsl / "lib").mkdir(parents=True)
    (hsl / "lib" / "libcoinhsl.so").write_bytes(b"")
    config = gen.DockerConfig(polyfly_repo=fake_repo, shim_dir=shim, libhsl_dir=str(hsl))
    cmd = config.command(gen.PLANNER_MODULE, ())
    assert f"LIBHSL_DIR={hsl}" in cmd
    assert f"{hsl}:{hsl}:ro" in " ".join(cmd)
    assert config.validate() == []


def test_from_env_reads_libhsl_dir(fake_repo: Path, shim: Path, monkeypatch) -> None:
    monkeypatch.setenv("LIBHSL_DIR", "/opt/hsl")
    config = gen.DockerConfig.from_env(polyfly_repo=fake_repo, shim_dir=shim)
    assert config.libhsl_dir == "/opt/hsl"
    monkeypatch.delenv("LIBHSL_DIR")
    assert gen.DockerConfig.from_env(polyfly_repo=fake_repo, shim_dir=shim).libhsl_dir is None


def test_from_env_falls_back_to_polyfly_env_vars(shim: Path, fake_repo: Path, monkeypatch) -> None:
    monkeypatch.delenv("POLYFLY_REPO", raising=False)
    monkeypatch.setenv("POLYFLY_DIR", str(fake_repo))
    assert gen.DockerConfig.from_env(shim_dir=shim).polyfly_repo == fake_repo.resolve()
    monkeypatch.delenv("POLYFLY_DIR")
    with pytest.raises(ValueError):
        gen.DockerConfig.from_env(shim_dir=shim)


def test_validate_flags_a_broken_setup(tmp_path: Path, config: gen.DockerConfig) -> None:
    bad = gen.DockerConfig(polyfly_repo=tmp_path / "nope", shim_dir=tmp_path / "also_nope")
    problems = " ".join(bad.validate())
    assert "polyfly repo not a directory" in problems
    assert "sitecustomize.py" in problems
    # A LIBHSL_DIR that does not actually contain the library must be flagged, because upstream
    # only warns and silently falls back to the slower default solver.
    with_bad_hsl = gen.DockerConfig(
        polyfly_repo=config.polyfly_repo, shim_dir=config.shim_dir, libhsl_dir=str(tmp_path)
    )
    assert any("libcoinhsl.so missing" in p for p in with_bad_hsl.validate())


# ---------------------------------------------------------------------------------------
# Output-name prediction
# ---------------------------------------------------------------------------------------


def test_forest_env_seed_matches_upstream_derivation() -> None:
    # Verified against real on-disk output: `-n 1 --seed 7` produced *_s1390851128.csv files.
    assert gen.forest_env_seed(7) == 1390851128
    assert gen.forest_env_seed(7) == gen.forest_env_seed(7)
    assert gen.forest_env_seed(7) != gen.forest_env_seed(8)


def test_expected_forest_stems_matches_real_filenames() -> None:
    stems = gen.expected_forest_stems(base_seed=7, forest_type=0)
    assert len(stems) == gen.TRAJECTORIES_PER_SEED == 9
    # These three filenames exist on disk from the verified `-n 1 --seed 7 --forest-type 0` run.
    assert stems[0] == "forest_000_f0_g15_0_0_m0p5_v0_0_0_s1390851128"
    assert stems[1] == "forest_001_f0_g15_0_0_m0p5_v1_0_0_s1390851128"
    assert stems[3] == "forest_003_f0_g15_n3_0_m0p5_v1_0_0_s1390851128"
    assert stems[6] == "forest_006_f0_g15_0_0_m0p6_v0_0_0_s1390851128"


def test_expected_stems_all_map_to_one_environment() -> None:
    stems = gen.expected_forest_stems(base_seed=11, forest_type=2)
    assert len({m.parse_stem(s)["env_id"] for s in stems}) == 1


def test_forest_jobs_are_one_environment_each() -> None:
    jobs = gen.forest_jobs(base_seed=100, n_environments=4, forest_type=0)
    assert len(jobs) == 4
    assert len({j.job_id for j in jobs}) == 4
    for job in jobs:
        assert job.args[:4] == ("-n", "1", "--sequential")[:3] + ("--seed",)
        assert len(job.expected_stems) == gen.TRAJECTORIES_PER_SEED
    # No two environments collide.
    all_stems = [s for j in jobs for s in j.expected_stems]
    assert len(set(all_stems)) == len(all_stems)


def test_smoke_mode_is_exactly_two_trajectories() -> None:
    jobs = gen.smoke_jobs()
    assert len(jobs) == 2
    assert sum(len(j.expected_stems) for j in jobs) == 2
    assert all(j.module == gen.PLANNER_MODULE for j in jobs)
    assert all(j.subdir == "experiments" for j in jobs)


# ---------------------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------------------


FOREST_LOG = """
Finding y candidates
Optimization completed in 6.51 seconds
Solver Return Status: Solve_Succeeded
Number of iterations: 143
file_dir: forests/forest_000_f0_g15_0_0_m0p5_v0_0_0_s1390851128.test
Optimization completed in 5.02 seconds
Solver Return Status: Maximum_Iterations_Exceeded
Number of iterations: 3000
optimization failed for forest_002_f0_g15_n3_0_m0p5_v0_0_0_s1390851128
Solver failed: Error in Opti::solve
optimization failed for forest_004_f0_g15_3_0_m0p5_v0_0_0_s1390851128
"""


def test_parse_planner_log_attributes_solver_stats_to_stems() -> None:
    solves = {s.stem: s for s in gen.parse_planner_log(FOREST_LOG)}
    assert len(solves) == 3

    ok = solves["forest_000_f0_g15_0_0_m0p5_v0_0_0_s1390851128"]
    assert ok.solver_status == "Solve_Succeeded"
    assert ok.iterations == 143
    assert ok.solve_time_s == pytest.approx(6.51)
    assert ok.succeeded is True

    maxed = solves["forest_002_f0_g15_n3_0_m0p5_v0_0_0_s1390851128"]
    assert maxed.solver_status == "Maximum_Iterations_Exceeded"
    assert maxed.succeeded is False

    crashed = solves["forest_004_f0_g15_3_0_m0p5_v0_0_0_s1390851128"]
    assert crashed.solver_status == "Solver_Failed"
    assert crashed.iterations == m.FAILURE_ITERATION_SENTINEL


def test_parsed_statuses_drive_the_quality_filter() -> None:
    solves = {s.stem: s.to_dict() for s in gen.parse_planner_log(FOREST_LOG)}
    good = m.TrajectoryRecord(stem="a", yaml_path=None, csv_path="a.csv", subdir="forests",
                              source="forest", env_id="e", n_timesteps=60, path_length_m=15.0,
                              total_time_s=5.9)
    good.solver_status = solves["forest_000_f0_g15_0_0_m0p5_v0_0_0_s1390851128"]["solver_status"]
    assert m.passes_quality(good)
    good.solver_status = solves["forest_002_f0_g15_n3_0_m0p5_v0_0_0_s1390851128"]["solver_status"]
    assert not m.passes_quality(good)


def test_parse_planner_log_handles_the_single_yaml_entry_point() -> None:
    log = """
    Solving experiments/maze_1.yaml
    Optimization completed in 6.20 seconds
    Solver Return Status: Solve_Succeeded
    Number of iterations: 98
    Total time = 3.92
    file_dir: experiments/maze_1.yaml
    """
    solves = gen.parse_planner_log(log)
    assert len(solves) == 1
    assert solves[0].stem == "maze_1"
    assert solves[0].total_time_s == pytest.approx(3.92)
    assert solves[0].succeeded is True


def test_a_saved_csv_from_a_failed_solve_is_not_marked_successful() -> None:
    """Observed for real on experiments/maze_3: the single-YAML entry point calls save_result
    unconditionally, so a failed solve still writes a full, plausible-looking CSV. Exit code and
    file existence both say 'fine'; only the solver status does not."""
    log = (
        "Solver failed: Error in Opti::solve\n"
        "file_dir: experiments/maze_3.yaml\n"
    )
    solve = gen.parse_planner_log(log)[0]
    assert solve.stem == "maze_3"
    assert solve.succeeded is False
    assert solve.solver_status == "Solver_Failed"

    record = m.TrajectoryRecord(
        stem="maze_3", yaml_path=None, csv_path="maze_3.csv", subdir="experiments",
        source="maze", env_id="maze_3", n_timesteps=2381, path_length_m=4.5, total_time_s=4.76,
        solver_status=solve.solver_status, iterations=m.FAILURE_ITERATION_SENTINEL,
    )
    issues = m.quality_issues(record)
    assert "solver_status" in issues and "solver_iteration_sentinel" in issues


def test_parse_planner_log_on_empty_or_noisy_input() -> None:
    assert gen.parse_planner_log("") == []
    assert gen.parse_planner_log("nothing to see here\n") == []


# ---------------------------------------------------------------------------------------
# Exit codes, idempotency, failure accounting
# ---------------------------------------------------------------------------------------


def _write_outputs(config: gen.DockerConfig, job: gen.Job, stems=None) -> None:
    root = config.polyfly_repo / "data" / "csvs" / job.subdir
    root.mkdir(parents=True, exist_ok=True)
    for stem in (stems if stems is not None else job.expected_stems):
        (root / f"{stem}.csv").write_text("time,sol_x_0,sol_x_1,sol_x_2\n0,0,0,0\n", "utf-8")


def _patch_subprocess(monkeypatch, behaviour) -> list[list[str]]:
    """Replace subprocess.run inside the driver; return the list of commands it was asked to run."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return behaviour(cmd)

    monkeypatch.setattr(gen.subprocess, "run", fake_run)
    return calls


def test_nonzero_exit_is_surfaced_and_counted(config: gen.DockerConfig, monkeypatch) -> None:
    jobs = gen.forest_jobs(base_seed=1, n_environments=2)

    def behaviour(cmd):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="boom", stderr="")

    _patch_subprocess(monkeypatch, behaviour)
    report = gen.run_jobs(jobs, config, workers=2)

    assert len(report.failed_jobs) == 2
    assert report.produced_stems == []
    assert "jobs failed          : 2" in report.format()
    assert "exit=1" in report.format()


def test_exit_zero_with_no_output_is_still_a_failure(config: gen.DockerConfig,
                                                    monkeypatch) -> None:
    """The F1 §2.2 regression: the container reports success but wrote nothing."""
    jobs = gen.forest_jobs(base_seed=1, n_environments=1)
    log = "Optimization completed in 6.0 seconds\nSolver Return Status: Solve_Succeeded\n"

    def behaviour(cmd):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=log, stderr="")

    _patch_subprocess(monkeypatch, behaviour)
    report = gen.run_jobs(jobs, config, workers=1)

    assert len(report.failed_jobs) == 1
    assert "no CSV written" in report.format()


def test_partial_output_is_reported_not_silently_accepted(config: gen.DockerConfig,
                                                          monkeypatch) -> None:
    jobs = gen.forest_jobs(base_seed=1, n_environments=1)
    job = jobs[0]

    def behaviour(cmd):
        _write_outputs(config, job, stems=job.expected_stems[:5])
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    _patch_subprocess(monkeypatch, behaviour)
    report = gen.run_jobs(jobs, config, workers=1)

    assert report.failed_jobs == []
    assert len(report.produced_stems) == 5
    assert len(report.missing_stems) == 4
    assert "trajectories missing : 4" in report.format()


def test_timeout_is_recorded_as_a_failure(config: gen.DockerConfig, monkeypatch) -> None:
    jobs = gen.forest_jobs(base_seed=1, n_environments=1)

    def behaviour(cmd):
        raise subprocess.TimeoutExpired(cmd, timeout=1.0)

    _patch_subprocess(monkeypatch, behaviour)
    report = gen.run_jobs(jobs, config, workers=1, timeout_s=1.0)

    assert report.failed_jobs[0].timed_out is True
    assert "timeout" in report.format()


def test_rerun_is_idempotent_and_skips_complete_environments(config: gen.DockerConfig,
                                                             monkeypatch) -> None:
    jobs = gen.forest_jobs(base_seed=1, n_environments=3)
    for job in jobs[:2]:
        _write_outputs(config, job)

    def behaviour(cmd):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    calls = _patch_subprocess(monkeypatch, behaviour)
    report = gen.run_jobs(jobs, config, workers=2)

    assert len(report.skipped) == 2
    assert len(calls) == 1
    assert str(jobs[2].args[-3]) in " ".join(calls[0])


def test_force_re_solves_everything(config: gen.DockerConfig, monkeypatch) -> None:
    jobs = gen.forest_jobs(base_seed=1, n_environments=2)
    for job in jobs:
        _write_outputs(config, job)

    calls = _patch_subprocess(
        monkeypatch, lambda cmd: subprocess.CompletedProcess(cmd, 0, "", "")
    )
    report = gen.run_jobs(jobs, config, workers=2, force=True)

    assert report.skipped == []
    assert len(calls) == 2


def test_empty_csv_does_not_count_as_complete(config: gen.DockerConfig) -> None:
    job = gen.forest_jobs(base_seed=1, n_environments=1)[0]
    root = config.polyfly_repo / "data" / "csvs" / job.subdir
    for stem in job.expected_stems:
        (root / f"{stem}.csv").write_text("", encoding="utf-8")
    csv_dir = config.polyfly_repo / "data" / "csvs"
    assert gen._is_complete(job, csv_dir) is False


def test_solve_info_sidecar_is_written(config: gen.DockerConfig, monkeypatch,
                                       tmp_path: Path) -> None:
    jobs = gen.forest_jobs(base_seed=7, n_environments=1)
    job = jobs[0]
    log = (f"Optimization completed in 6.51 seconds\nSolver Return Status: Solve_Succeeded\n"
           f"Number of iterations: 143\nfile_dir: forests/{job.expected_stems[0]}.test\n")

    def behaviour(cmd):
        _write_outputs(config, job, stems=job.expected_stems[:1])
        return subprocess.CompletedProcess(cmd, 0, log, "")

    _patch_subprocess(monkeypatch, behaviour)
    sidecar = tmp_path / "out" / "solve_info.jsonl"
    gen.run_jobs(jobs, config, workers=1, solve_info_path=sidecar, log_dir=tmp_path / "out/logs")

    info = m.load_solve_info([sidecar])
    assert info[job.expected_stems[0]]["solver_status"] == "Solve_Succeeded"
    assert info[job.expected_stems[0]]["iterations"] == 143
    assert (tmp_path / "out" / "logs" / f"{job.job_id}.log").read_text().startswith("Optimization")


def test_rebuild_manifest_reads_disk_and_assigns_splits(config: gen.DockerConfig,
                                                        tmp_path: Path) -> None:
    for job in gen.forest_jobs(base_seed=1, n_environments=3):
        _write_outputs(config, job)
    manifest_path = tmp_path / "data" / "manifest.jsonl"

    records = gen.rebuild_manifest(config.polyfly_repo, manifest_path, split_seed=0)

    assert manifest_path.exists()
    assert len(records) == 3 * gen.TRAJECTORIES_PER_SEED
    assert all(r.split is not None for r in records)
    # Environment-level split integrity survives the round-trip through the file.
    by_env: dict[str, set[str]] = {}
    for record in m.read_manifest(manifest_path):
        by_env.setdefault(record.env_id, set()).add(record.split)
    assert all(len(v) == 1 for v in by_env.values())


def test_generate_end_to_end_with_a_stubbed_container(config: gen.DockerConfig, monkeypatch,
                                                      tmp_path: Path) -> None:
    jobs = gen.forest_jobs(base_seed=1, n_environments=2)
    stems_by_seed = {j.job_id: j.expected_stems for j in jobs}

    def behaviour(cmd):
        joined = " ".join(cmd)
        for job_id, stems in stems_by_seed.items():
            seed = job_id.rsplit("seed", 1)[1]
            if f"--seed {seed} " in joined + " ":
                root = config.polyfly_repo / "data" / "csvs" / "forests"
                root.mkdir(parents=True, exist_ok=True)
                for stem in stems:
                    (root / f"{stem}.csv").write_text(
                        "time,sol_x_0,sol_x_1,sol_x_2\n" +
                        "".join(f"{i*0.1},{i*0.3},0,0\n" for i in range(50)), "utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _patch_subprocess(monkeypatch, behaviour)
    report, records = gen.generate(jobs, config, tmp_path / "manifest.jsonl", workers=2)

    assert report.failed_jobs == []
    assert len(records) == 2 * gen.TRAJECTORIES_PER_SEED
    kept, rejected = m.filter_records(records, h=10, t=20)
    assert len(kept) == len(records), rejected
