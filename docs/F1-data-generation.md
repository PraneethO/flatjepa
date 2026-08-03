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
