# Progress / Handoff

Last updated: 2026-08-03 morning.

**This document is written to be self-contained.** A fresh session with no prior context should be
able to pick up the project from here. Read this, then `docs/00-overview.md`.

Repo: https://github.com/PraneethO/flatjepa (public, owner `PraneethO`).

---

## 0. Ground rules

- **Do not add any AI/Claude/Anthropic attribution anywhere** — not in code, comments, docstrings,
  commit messages, or co-author trailers. This is the author's explicit standing instruction. All
  commits are authored as Praneeth Otthi <potthi@berkeley.edu>.
- Never modify the upstream checkout at `~/Desktop/polyfly_ral` except for the parameter files
  listed in §8, which this project authored.
- Do not loosen a test tolerance to make a test pass. If a threshold is unreachable, replace the
  criterion with a better-posed one and say so.

## 1. What this project is

**Hypothesis:** a differentially flat system provides analytic ground truth for the minimal
sufficient latent of its own dynamics, so on such a system JEPA representation quality can be
measured *directly* rather than inferred from downstream task performance.

Standard JEPA evaluation is indirect — you measure downstream reward and cannot separate "the
representation captured the state" from "the predictor learned the dynamics" from "the controller is
good." A differentially flat system has a closed-form map from a few flat outputs to the entire
state, so the correct latent is known in both dimension and content.

The system is a quadrotor with a cable-suspended payload, using trajectories from the PolyFly
optimal planner ([arXiv:2510.15226](https://arxiv.org/abs/2510.15226)). Architecture is adapted from
SkyJEPA ([arXiv:2606.23444](https://arxiv.org/abs/2606.23444)): TCN state/action encoders, GRU latent
predictor, SIGReg anti-collapse, and a physics-inspired prober using differentiable kinematic
integration.

Experiments (full detail in `docs/00-overview.md` §4):

| ID | Question |
|----|----------|
| E1 | Does the latent recover the flat outputs? (linear probe, R²) |
| E2 | Is effective dimensionality ≈ **9**, the known minimal state? |
| E3 | **Headline.** Does SIGReg help or hurt recovery? Flat coordinates are intrinsically non-isotropic, so enforcing isotropy may distort them |
| E4 | Does the latent discover the taut/slack hybrid boundary? (**blocked — see §6**) |
| E5 | Does the prober residual localize flatness-assumption violations? |

## 2. Current state

| Feature | Status | Doc |
|---------|--------|-----|
| F1 generation driver + manifest | **Done** | `docs/F1-data-generation.md` |
| F2 flat-output extractor | **Done**, agrees with upstream to ~1e-15 | `docs/F2-flat-outputs.md` |
| F3 tension / taut-slack | **Done** | `docs/F3-taut-slack.md` |
| F4 dataset builder | **NOT STARTED — critical path** | `docs/F4-dataset.md` |
| F5 JEPA core | **Done** | `docs/F5-jepa-core.md` |
| F6 physics prober | **Done** | `docs/F6-physics-prober.md` |
| F7 measurement suite | **NOT STARTED** | `docs/F7-measurement-suite.md` |
| F8 training harness | **NOT STARTED** | `docs/F8-training-harness.md` |
| F9 evaluation / figures | **NOT STARTED** | `docs/F9-evaluation.md` |
| F10 baselines | **NOT STARTED** | `docs/F10-baselines.md` |
| F11 perception | Deferred by design | `docs/F11-deferred-perception.md` |

**Test suite: 189 passed, 0 failed.**

Nothing has been trained yet. No model has seen data. Everything so far is data plumbing, model
code, and the measurement plan.

## 3. Environment

Machine: Ubuntu 22.04 host, 16 cores, 30 GB RAM, RTX 3090 (20 GB), ~500 GB free.

### Python
Host has **no** numpy/torch in system python. Use the project venv:
```bash
cd ~/Desktop/flatjepa
./.venv/bin/python -m pytest tests/ -q          # 189 should pass
```
Created with `virtualenv` (not `python3 -m venv`, which needs the `python3.10-venv` system package).
Contains torch 2.13 **CPU**, numpy, scipy, pandas, pyyaml, matplotlib, pytest.

### Containers (two, by design — do not merge)

**`poly-fly:latest`** — planning/data generation. CasADi + IPOPT, CPU only.
Python at `/opt/conda/envs/poly_fly/bin/python` (NOT on `PATH` under `bash -lc`).

Three things are mandatory or it fails, two of them silently:
```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$HOME/Desktop/polyfly_ral:/workspace:rw" \
  -v "$HOME/Desktop/flatjepa/scripts/shim:/shim:ro" \
  -e POLYFLY_DIR=/workspace -e PYTHONPATH=/shim:/workspace/src \
  -e MPLBACKEND=Agg -e HOME=/tmp \
  --workdir /workspace poly-fly:latest \
  /opt/conda/envs/poly_fly/bin/python -m poly_fly.optimal_planner.planner --yaml experiments/maze_1.yaml
```
- `--user "$(id -u):$(id -g)"` + `HOME=/tmp` — the image ends with `USER mambauser`, which cannot
  write to the bind mount. **The solve reports success and only the exit code reveals the failure.**
- `scripts/shim/sitecustomize.py` first on `PYTHONPATH` — upstream hardcodes
  `matplotlib.use('TkAgg')` at import, which raises headless. The shim neutralizes it without
  patching upstream.

**`ros-jazzy:pytorch`** — GPU training. Ubuntu 24.04, ROS 2 Jazzy, Python 3.12, torch 2.13.0+cu130.
GPU verified working:
```bash
docker run --rm --gpus all ros-jazzy:pytorch python3 -c "import torch;print(torch.cuda.is_available())"
```
Missing pandas/casadi — install into the image or the training code should avoid pandas.

### Tools
`gh` CLI at `~/.local/bin/gh`, authenticated as `PraneethO`. Git remote uses SSH.

## 4. Data generation

**Currently RUNNING** (restarted 2026-08-03 09:44).

```bash
cd ~/Desktop/flatjepa
scripts/gen_launch.sh status     # running? how many trajectories?
scripts/gen_launch.sh start      # launch detached (survives SSH disconnect)
scripts/gen_launch.sh stop       # stop driver + any planner containers
```

Driver log: `logs/gen_all.log`. Per-phase logs: `logs/gen_ft{TYPE}_s{SEED}.log`. `logs/` is
gitignored.

Tunables (env vars): `FOREST_TYPES`, `SEEDS`, `N_PER_SEED`, `TARGET_CSVS`, `LIBHSL_DIR`.

**Current corpus:** ~198+ forest trajectories, all forest-type 0, at
`~/Desktop/polyfly_ral/data/csvs/forests/`. Plus 1 maze and 2 descent probes.

**⚠️ A previous run was lost to a reboot.** The machine rebooted at 2026-08-02 23:45, killing the
job and wiping `/tmp`, where the original driver script and all logs lived. That is why the driver
now lives in `scripts/` and logs in `logs/`. `gen_launch.sh` survives SSH disconnect but **not a
reboot** — after any reboot, re-run `scripts/gen_launch.sh start`.

Throughput: ~15.6 s/trajectory sequential; with `--mp --pin` the machine saturates all 16 cores.
MA57/HSL would be ~3× faster but needs an academic license; the driver passes `LIBHSL_DIR` through
if you ever set it, no code change needed.

## 5. Findings that change the plan

**These matter more than the code.** All are verified, not assumed, and are documented in the
relevant feature doc.

### 5.1 Forest CSVs are 10 Hz, not 500 Hz
`generate_forest.py:537` calls `save_result(..., dt=0.1)`. Only the maze corpus goes through the
500 Hz interpolation. Forest trajectories are **47–122 rows**. F4's original plan to stride-25 down
to 20 Hz is **impossible** and has been dropped — window at native 10 Hz. At H=10/T=20, a ~53-row
trajectory yields only ~20 windows.

### 5.2 CSV attitude is NOT the flat-map attitude
Upstream applies `get_yaw_along_trajectory` *after* `differential_flatness` (`planner.py:1350`), so
`R_csv = R_z(ψ)·R_flat`, |ψ| up to 1.39 rad (~80°). Position/velocity/acceleration columns are
written *before* that step and do agree to ~1e-15. **Any probe targeting attitude must use the
flat-map attitude**, not the CSV columns, or it is silently wrong by up to 80° of yaw.

### 5.3 A failed solve still writes a valid-looking CSV
`experiments/maze_3` failed with `Solver_Failed`, yet `run()` still called `save_result`, producing a
2381-row CSV with exit code 0 that passes every structural check. Compounding it, **upstream never
persists solver status** — it prints to stdout and discards it (`planner.py:966-971`).

Therefore: capture planner stdout at generation time and record IPOPT status to a sidecar. It cannot
be recovered afterwards. Do not infer success from exit code, file presence, or the `file_dir:` log
line (which prints on *entry* to `save_result`, before any work).

### 5.4 Splits must key on environment, not trajectory
One forest seed yields multiple trajectories sharing one obstacle field. A per-trajectory split puts
the same layout on both sides. `src/flatjepa/data/manifest.py` already keys splits on `env_id`
parsed from the stem, and uses per-environment *hashing* rather than seeded shuffling so that
appending new environments to a growing corpus cannot silently move a previously-tested environment
into training. This matters because generation is ongoing.

### 5.5 E1 risks being tautological — mitigation redesigned, not yet implemented
See `docs/F7-measurement-suite.md` §1b and `docs/F4-dataset.md` §2.

The naive failure: if the input is `sol_x = (p,v,a)` and the probe target is `(p,v,a)`, R² ≈ 1.0
proves nothing.

**The subtler failure, which the original mitigation walked into:** with a *history window* as
input, every time-derivative is a **linear functional of that window** — velocity is a finite
difference, acceleration a second difference, jerk a third. So feeding position only (the original
proposed fix) leaves `v`, `a`, `j` linearly recoverable from the raw window. The "raw input window"
control would also score ~1.0, and the result would still be vacuous while *appearing* controlled
for.

**Current design.** Probe for the **nonlinear** consequences of the flat map, which are its actual
content:

| Target | Linear in window? | Role |
|---|---|---|
| `p`, `v`, `a`, `j` | Yes (finite differences) | Sanity check only; expect ~1.0 and label it |
| tension `‖a+g·e₃‖` | No (norm) | Eligible |
| cable direction | No (normalization) | Eligible |
| quad attitude `R_flat` | No (cross products, SO(3)) | **Preferred headline** |
| `p_quad` | No (nonlinear via `b₃`) | Eligible |

Enforced by a **linear-decodability audit** (F7 §1b): every candidate target is first probed from
the raw input window, and anything the raw window already solves (R² > 0.9) is disqualified as a
headline target automatically. Not yet implemented — build it as part of F7, before any E1 number
is reported.

Remember §5.2 when probing attitude: use the **flat-map** attitude, not the CSV columns.

## 6. Open decisions (need the author)

### 6.1 E4 — blocked, but recoverable
The forest corpus contains **zero** near-slack timesteps at every threshold (min tension margin
0.841; free fall = 0.0). Cause: payload z is bounded to a 0.75 m corridor while x/y span ~18×15 m,
so a minimum-time horizontal objective never builds vertical speed.

Two probes were generated to test whether scenario redesign helps:

| Scenario | Drop | Vel bound | `a_z` min | margin min | % below τ=0.2 |
|---|---|---|---|---|---|
| Forest (baseline) | — | ±5 | −1.57 | 0.841 | 0% |
| `descent/probe_1` | 6 m | ±8 | −3.97 | 0.600 | 0% |
| `descent/probe_2` | **20 m** | **±20** | **−8.32** | **0.152** | **8.4%** |

**E4 is recoverable with a dedicated descent corpus** built from the `descent/probe_2` config with
randomized drop heights/offsets/obstacles. A 6 m drop is not enough. The binding constraint is the
**jerk** bound (±10 m/s³), not acceleration: swinging `a_z` from 0 to −9.81 and back costs ~2 s.

Keep such a corpus **separate** from the forest data — the environment distribution differs
substantially and mixing would confound E1–E3.

**Decision needed:** build the descent corpus, or drop E4 and report the diagnosis. E1/E2/E3 are
unaffected either way.

### 6.2 `forests/base.yaml` was authored by this project
Upstream does not ship it and `generate_forest.py` cannot run without it. State bounds were widened
to cover the forest extent and `tube_distance: 10` was copied from generated mazes. Documented in
`docs/F1-data-generation.md` §3b. If these differ from what the PolyFly authors used, trajectories
are still valid optimal solutions but not to the same problem as the paper — fine for the
self-contained flatness experiments, **not** fine for any comparison to published PolyFly results.

## 7. Next steps, in order

### Step 1 — F4 dataset builder *(critical path, blocks everything)*
Spec: `docs/F4-dataset.md`. Must respect:
- **10 Hz native rate** (§5.1) — do not resample
- environment-level splits — consume `src/flatjepa/data/manifest.py`, do not re-derive
- normalization statistics from the **training split only**
- positions **relative** to the window's last history frame, never absolute world coordinates
- exclude timesteps failing F2's validity mask
- the E1 tautology mitigation (§5.5) — make the observed-state subset configurable
- no window may cross a trajectory boundary

Existing pieces to build on: `src/flatjepa/data/csv_io.py` (loader), `flatness.py` (ground truth),
`tension.py` (labels), `manifest.py` (splits + quality filters).

### Step 2 — F8 training harness
Two-stage: (1) encoders + predictor on `L_pred + λ·L_SIGReg`; (2) freeze stage 1, train the prober.
**Assert the freeze** — verify stage-1 params have zero gradient in stage 2, or every claim about
probing *frozen* representations is void. Log collapse diagnostics every epoch with an alarm
(`src/flatjepa/models/diagnostics.py` exists) so an overnight run does not spend hours training a
constant function.

### Step 3 — first training run on GPU
Use `ros-jazzy:pytorch` with `--gpus all`.

### Step 4 — F7 measurement suite
**Run E5-a first** — the zero-residual calibration check. Trained on flatness-consistent data the
residual must converge to ~0. It has an unambiguous physically-determined correct answer and it
gates trust in everything else. The companion control (residual head *can* fit non-zero targets) is
already tested in `tests/test_prober.py`.

Then E1/E2/E3. Every probe result needs all three controls from F7 §1 (random-init encoder, raw
input window, shuffled labels) reported *in the same figure*. Multi-seed (≥3) with spread by
default.

### Step 5 — F10 baselines, F9 figures, then decide E4.

## 8. Files this project authored inside the upstream checkout

New/untracked in `~/Desktop/polyfly_ral`, **not** part of upstream:

- `data/params/forests/base.yaml` — required by `generate_forest.py`, not shipped (see §6.2)
- `data/params/descent/{base,probe_1,probe_2}.yaml` — the E4 descent probes

Note the planner **rewrites these YAMLs in place** on solve, writing the global plan back into
`payload_pos_init`, so they no longer look as authored.

## 9. Quick health check

```bash
cd ~/Desktop/flatjepa
./.venv/bin/python -m pytest tests/ -q                                    # expect 189 passed
scripts/gen_launch.sh status                                              # generation state
find ~/Desktop/polyfly_ral/data/csvs/forests -name '*.csv' | wc -l        # corpus size
git log --oneline -5
```
