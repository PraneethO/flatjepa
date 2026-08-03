# F1 — Data Generation

**Purpose:** produce a large corpus of optimal cable-suspended payload trajectories from the
upstream PolyFly planner, headlessly and in parallel.

**Serves:** all experiments. This is the blocking dependency for everything else.

---

## 1. Verified facts

These were confirmed empirically on the target machine before this document was written:

| Fact | Value |
|------|-------|
| Planner solve time | ~6.5 s per trajectory (single core, default IPOPT) |
| Output shape | 1962 timesteps × 35 columns for `experiments/maze_1.yaml` |
| Host cores | 16 |
| Estimated throughput | ~8k trajectories/hour at 16-way parallelism |
| Solver | Default IPOPT. MA57/HSL absent — see §5 |

## 2. Two blockers and their resolutions

Both were discovered by running the planner, and both must be handled by any batch driver.

### 2.1 Headless matplotlib

`src/poly_fly/optimal_planner/global_planner.py:7` executes `matplotlib.use('TkAgg')` at import
time. This raises `ImportError` with no display attached.

**Resolution:** a `sitecustomize.py` shim placed ahead of the upstream package on `PYTHONPATH`,
which forces the `Agg` backend and makes interactive backend selection a no-op. This is preferred
over patching upstream because it keeps the PolyFly checkout pristine and version-agnostic.

### 2.2 Container UID mismatch

The `poly-fly:latest` image ends with `USER mambauser`. Writing to a bind-mounted host directory
then fails with `PermissionError` on `data/csvs`. The failure is *silent in the summary output* —
the solve reports success and only the exit code reveals the problem.

**Resolution:** run with `--user "$(id -u):$(id -g)"` and `HOME=/tmp`. Outputs are then owned by the
host user.

> **Regression risk:** a batch job that ignores exit codes will appear to run all night and produce
> nothing. The driver must check exit status per trajectory and fail loudly.

## 3. Invocation

The verified working command, which the batch driver wraps:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$POLYFLY_REPO:/workspace:rw" \
  -v "$SHIM_DIR:/shim:ro" \
  -e POLYFLY_DIR=/workspace \
  -e PYTHONPATH=/shim:/workspace/src \
  -e MPLBACKEND=Agg \
  -e HOME=/tmp \
  --workdir /workspace \
  poly-fly:latest \
  /opt/conda/envs/poly_fly/bin/python -m poly_fly.optimal_planner.planner --yaml <relative.yaml>
```

## 3b. Authored upstream config — an assumption to revisit

The upstream repository ships `data/params/experiments/base.yaml` but **not**
`data/params/forests/base.yaml`, which `generate_forest.py` requires. Forest generation fails
without it.

That file was therefore authored for this project. Two things about it are judgment calls rather
than upstream fact, and both should be revisited if trajectory quality looks wrong:

1. **State bounds.** `forest_params.py` places obstacles over `x ∈ [0, 16]`, `y ∈ [−7, 7]` with the
   goal at `x = 15`, but the shipped experiments config bounds position to `x ≤ 5`. Bounds were
   widened to `state_max: [17, 7.5, 0.75, ...]` / `state_min: [-1, -7.5, 0, ...]` to cover the
   forest extent. Velocity, acceleration, and input bounds were left at the upstream values —
   notably acceleration remains ±10 m/s², which is what makes the slack regime reachable (see F3).
2. **`tube_distance: 10`.** Required by the `MPC` dataclass, absent from the experiments base
   config, present as `10` in the generated maze YAMLs. Copied from those.

**Risk:** if these differ from what the PolyFly authors used, generated trajectories are still
valid optimal solutions to *a* problem, but not necessarily to the same problem as the paper. This
does not affect the flatness-recovery experiments, which are self-contained — but it does affect any
comparison to published PolyFly results.

## 4. Environment sampling

Trajectory diversity comes from varying the environment YAML, not from noise. Two sources:

1. **Forest environments** — upstream `generate_forest.py` produces randomized obstacle fields with
   varying start/goal positions. This is the scalable source.
2. **Maze environments** — 9 fixed hand-designed environments. Useful as a held-out structured test
   set, too few to train on.

**Split policy:** split by *environment*, never by timestep. Windows drawn from the same trajectory
are highly correlated; a random timestep split leaks the test set into training and will inflate
every probe R² in F7. This is the single easiest way to get a meaningless result.

## 5. Solver configuration

The upstream planner warns `LIBHSL_DIR environment variable not set. Using default IPOPT solver.`
MA57 is roughly 3× faster but requires an academic HSL license.

**Requirement:** the driver reads `LIBHSL_DIR` from the environment and passes it through when set,
so acquiring a license later is a zero-code-change speedup.

## 6. Output contract

Per trajectory, upstream writes:

- `data/csvs/<subdir>/<stem>.csv` — 35 columns: `time`, `sol_x_*` (9, payload pos/vel/acc),
  `sol_u_*` (3, jerk), `sol_quad_x_*` (9), `sol_quad_quat_*` (4), `sol_payload_rpy_*` (3),
  `rot_mat_*` (6, first two rotation-matrix columns)
- `data/params/<subdir>/<stem>.yaml` — the parameters used, including obstacle geometry

F1 additionally writes a **manifest** (`manifest.jsonl`, one record per trajectory) capturing: stem,
yaml path, csv path, solver status, iterations, solve time, path length, total trajectory time, and
the environment-split assignment. Downstream features consume the manifest, not a directory listing,
so that failed or degenerate solves are excluded explicitly rather than by accident.

## 6b. A failed solve still writes a valid-looking CSV *(verified — data-integrity hazard)*

Discovered live during smoke testing: `experiments/maze_3` failed with `Solver_Failed`, and upstream
`run()` **still called `save_result`** on the debug solution values. The result:

- exit code **0**
- a CSV present on disk, 2381 rows, 4.76 s duration, 4.50 m path length
- structurally **indistinguishable** from a good trajectory — every structural predicate in §7 passes

Compounding this: **upstream never persists solver status.** Status, iteration count, and solve time
are printed to stdout and discarded (`planner.py:966-971`). A manifest rebuilt from CSVs alone
therefore cannot recover them.

**Consequences, all mandatory:**

1. The generation driver must capture planner stdout and record the IPOPT status to a sidecar
   (`solve_info.jsonl`) at generation time. It cannot be recovered later.
2. Success must be derived from the recorded solver status — **not** from exit code, not from file
   presence, and not from the `file_dir:` log line (which prints on *entry* to `save_result`, before
   any of the work, so it appears even for failed solves).
3. Any corpus generated without this capture has unknown provenance and may silently contain failed
   solves presented as optimal trajectories.

The forest generator appears to write only on success, which is why status-unknown records from that
path are tolerated — but that is a property of a different code path and should not be assumed of the
single-YAML entry point.

## 7. Quality filtering

Not every solve is usable. The manifest must record and the dataset builder must be able to exclude:

- solver status not in {`Solve_Succeeded`, `Solved_To_Acceptable_Level`}
- degenerate paths (near-zero path length, or trajectory time at the failure sentinel `1000`)
- trajectories shorter than one full training window (H + T)

## 8. Acceptance criteria

- [ ] Generates N trajectories in parallel across configurable cores, with per-trajectory exit-code
      checking
- [ ] Non-zero exit codes are surfaced, counted, and do not silently reduce the dataset
- [ ] Manifest written with all fields in §6
- [ ] Environment-level split assignment is deterministic given a seed
- [ ] Re-running is idempotent: existing valid trajectories are skipped, not recomputed
- [ ] A smoke mode generates 2 trajectories end-to-end in under 2 minutes for CI
