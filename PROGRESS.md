# Progress / Handoff

Last updated: 2026-08-02, late evening. Written mid-session because of a usage limit
(resets 3:20am America/Los_Angeles).

Repo: https://github.com/PraneethO/flatjepa (public, PraneethO). All work pushed through
commit `b4e5194`.

---

## Where things stand

| Feature | State |
|---------|-------|
| F1 generation driver + manifest | **Done**, 68 tests pass |
| F2 flat-output extractor | **Done**, agrees with upstream to ~1e-15 |
| F3 tension / taut-slack | **Done**, histogram produced, E4 decision surfaced |
| F4 dataset builder | **Not started** — next critical path item |
| F5 JEPA core | **Done** (encoders, predictor, SIGReg, jepa, diagnostics) |
| F6 physics prober | **Nearly done** — 1 failing test, see below |
| F7 measurement suite | **Not started** |
| F8 training harness | **Not started** |
| F9 evaluation/figures | **Not started** |
| F10 baselines | **Not started** |

Test suite: **188 passed, 1 failed**.

## The one failing test

`tests/test_prober.py::test_residual_head_can_fit_non_zero_targets`

```
assert float(loss) < 1e-4
E   assert 0.0005926934536546469 < 0.0001
```

**This is not a bug in the prober.** It is F6 §3's "dead head" control — proving the residual head
*can* fit non-zero targets, so that "residual ≈ 0" in E5-a means something. The head is demonstrably
learning; it just has not converged in the 400 Adam steps the test allows.

**Correct fix: raise the step count** (try 1500–2000), *not* loosen the threshold. Loosening the
tolerance to make it green would defeat the test's purpose. Verify the meaningful assertion on the
next line (`residual.abs().mean() > 0.1`) also passes.

The F5/F6 agent was terminated by the usage limit immediately after writing this test, so it was
never run to completion.

## Data generation

**Still running** in the background (`gen_all.sh`, nohup). 3 forest types × 4 seeds × `-n 40`,
`--mp --pin` across 16 cores. ~71+ forest CSVs at last check, target ~4300.

Check status:
```bash
pgrep -f gen_all.sh && find ~/Desktop/polyfly_ral/data/csvs/forests -name '*.csv' | wc -l
```

Logs in the session scratchpad: `gen_ft{0,1,2}_s{101,202,303,404}.log`.

**If it died**, re-run `gen_all.sh` — generation is idempotent at the environment level via F1's
driver, though `gen_all.sh` itself is a simple loop and will redo work.

## Findings that change the plan

These are all documented in `docs/` with evidence. They matter more than the code.

### 1. Forest CSVs are 10 Hz, not 500 Hz
`generate_forest.py:537` calls `save_result(..., dt=0.1)`. Only the maze corpus goes through 500 Hz
interpolation. **F4's plan to stride-25 down to 20 Hz is impossible** and has been dropped —
windowing runs at native 10 Hz. Trajectories are ~47–122 rows, giving ~20 windows each at H=10/T=20.

### 2. CSV attitude is NOT the flat-map attitude
Upstream applies `get_yaw_along_trajectory` *after* `differential_flatness` (`planner.py:1350`), so
`R_csv = R_z(ψ)·R_flat` with |ψ| up to 1.39 rad (~80°). Position/velocity/acceleration columns are
written *before* that step and do agree to ~1e-15. **Anything probing attitude must use the flat-map
attitude, not the CSV columns.**

### 3. A failed solve still writes a valid-looking CSV
`maze_3` failed with `Solver_Failed` but produced a 2381-row CSV, exit code 0, passing every
structural check. Upstream never persists solver status — it prints to stdout and discards it. **Any
corpus generated without capturing stdout has unknown provenance.**

### 4. E4 is blocked but recoverable — **needs a decision**
The forest corpus contains **zero** near-slack timesteps at any threshold (min margin 0.84; free
fall = 0.0). Cause: payload z is bounded to a 0.75 m corridor while x/y span ~18×15 m.

Two probe trajectories were generated to test whether scenario redesign helps:

| Scenario | Drop | Vel bound | margin min | % below τ=0.2 |
|---|---|---|---|---|
| Forest (baseline) | — | ±5 | 0.841 | 0% |
| `descent/probe_1` | 6 m | ±8 | 0.600 | 0% |
| `descent/probe_2` | **20 m** | **±20** | **0.152** | **8.4%** |

**E4 is recoverable with a dedicated descent corpus.** A 6 m drop is not enough. The binding
constraint is the jerk bound (±10 m/s³), not acceleration.

**Open decision for Praneeth:** generate a separate descent corpus (kept distinct from the forest
data, since mixing would confound E1–E3), or drop E4. E1/E2/E3 are unaffected either way.

### 5. E1 risks being tautological — unresolved design issue
See `docs/F4-dataset.md` §2. If the model input is `sol_x = (p,v,a)` and the probe target is
`(p,v,a)`, the network is handed its own answer. **This must be resolved before E1 means anything.**
Planned mitigation: partial observation as the headline config, attitude/cable-direction as the
physically meaningful targets, random-init encoder as the control. Not yet implemented.

## Next steps, in order

1. **Fix the prober test** (raise step count). Small.
2. **F4 dataset builder** — the critical path blocker for everything downstream. Must respect: 10 Hz
   native rate, environment-level splits (F1's manifest already provides them), train-split-only
   normalization, relative positions, and the E1 tautology mitigation above.
3. **F8 training harness** → first JEPA training run on GPU (use the `ros-jazzy:pytorch` container,
   torch 2.13+cu130, verified working on the RTX 3090 with `--gpus all`).
4. **F7 measurement suite** → E5-a zero-residual check first (it gates trust in everything), then
   E1/E2/E3.
5. Decide E4.

## Environment notes

- Host venv: `/home/praneetho/Desktop/flatjepa/.venv` (created with `virtualenv`, since `python3 -m
  venv` needs the `python3.10-venv` system package). torch 2.13 CPU, numpy, scipy, pandas, pytest.
- Tests: `./.venv/bin/python -m pytest tests/ -q`
- Data generation container: `poly-fly:latest`, python at
  `/opt/conda/envs/poly_fly/bin/python`. Requires `--user "$(id -u):$(id -g)"`, `HOME=/tmp`, the
  `scripts/shim/sitecustomize.py` first on `PYTHONPATH`, and `MPLBACKEND=Agg`.
- GPU training container: `ros-jazzy:pytorch` with `--gpus all`.
- `gh` CLI installed at `~/.local/bin/gh`.

## Files authored in the upstream polyfly_ral checkout

These are new/untracked in `~/Desktop/polyfly_ral` and are **not** part of upstream:

- `data/params/forests/base.yaml` — required by `generate_forest.py`, not shipped. State bounds
  widened to cover the forest extent; `tube_distance: 10` copied from generated mazes. Assumptions
  documented in `docs/F1-data-generation.md` §3b.
- `data/params/descent/{base,probe_1,probe_2}.yaml` — the E4 descent probes.

Note the planner **rewrites** these YAMLs in place on solve (it writes the global plan back into
`payload_pos_init`), so they will look different from as-authored.
