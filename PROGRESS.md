# Progress / Handoff

Last updated: 2026-08-03 morning.

**This document is written to be self-contained.** A fresh session with no prior context should be
able to pick up the project from here.

Reading order: **this file** (state) → **`docs/BACKGROUND.md`** (why the project is shaped this way,
including alternatives already rejected) → **`docs/00-overview.md`** (the experimental plan).
A ready-to-paste startup prompt is in **`START_PROMPT.md`**.

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
| F4 dataset builder | **Done** — 37,180 windows built | `docs/F4-dataset.md` |
| F5 JEPA core | **Done** | `docs/F5-jepa-core.md` |
| F6 physics prober | **Done** | `docs/F6-physics-prober.md` |
| F7 measurement suite | **Done** | `docs/F7-measurement-suite.md` |
| F8 training harness | **Done** | `docs/F8-training-harness.md` |
| F9 evaluation / figures | **Done** | `docs/F9-evaluation.md` |
| F10 baselines | **Code done**, not yet run | `docs/F10-baselines.md` |
| F11 perception | Deferred by design | `docs/F11-deferred-perception.md` |

**Test suite: 230 passed, 0 failed.**

Nothing has been trained yet. No model has seen data — but the dataset now exists and the E1
methodology has been validated against it (see §5.8).

Built dataset: `data/windows_v1/` — 37,180 windows from 1,187 forest trajectories
(27,796 train / 4,566 val / 4,818 test), 10 Hz, environment-level splits, train-only normalisation.
Rebuild with the snippet in §7. The F1 manifest is at `data/manifest.jsonl` (1,190 records,
238 environments).

**Training works.** ~1 s/epoch on 27,796 windows with GPU-resident data. Run the full experiment:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w \
  ros-jazzy:pytorch bash -c 'PYTHONPATH=/w/src python3 scripts/experiment.py \
  --out outputs/e1e3 --runs-root runs/e1e3 --seeds 0 1 2 \
  --lambdas 0.0 0.01 0.1 1.0 10.0 --widths 4 8 16 24 48 96 --epochs 60 --jobs 6'
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w \
  ros-jazzy:pytorch bash -c 'PYTHONPATH=/w/src python3 scripts/report.py'
```

Note `--user` and `HOME=/tmp`: without them the container writes root-owned files into the repo,
same as the planner image.

## RESULTS EXIST — see `outputs/e1e3/RESULTS.md`

30 runs, 3 seeds, λ and width sweeps. **E1 and E2 are negative results.** The trained encoder scores
*below* a random-init encoder on 6 of 7 targets, and effective dimensionality tracks allocated width
(1.3–5.1) rather than converging on the theoretical 9. F7 §1's stopping condition is met.

E3 shows the predicted dissociation but contradicts two design premises: λ=0 does **not** collapse
and gives the lowest prediction loss, while λ=1 collapses 2/3 seeds to participation ratio 1.01.
A direct probe (`scripts/sigreg_rank_probe.py`) shows the penalty is **not monotone in rank** —
rank-9 scores 0.5148 vs rank-1 at 0.1668.

**Top open item: verify the SIGReg implementation against
[rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa)**, specifically whether embeddings are
standardised before the projection test. That would remove the scale confound and is the difference
between "our implementation has a bug" and "the method has this property". Do not claim the latter
until checked.

**Then: F10 baselines (code written, not run), and the E4 decision (§6.1).**

## 3. Environment

Machine: Ubuntu 22.04 host, 16 cores, 30 GB RAM, RTX 3090 (20 GB), ~500 GB free.

### Python
Host has **no** numpy/torch in system python. Use the project venv:
```bash
cd ~/Desktop/flatjepa
./.venv/bin/python -m pytest tests/ -q                  # 230 pass, ~35 s
./.venv/bin/python -m pytest tests/ -q -m "not slow"    # 229 pass, ~5 s
```
`tests/conftest.py` pins torch to a single thread. Without it the suite takes **12+ minutes** during
a generation run (torch's per-core threads fight the 14 pinned planner workers) and looks hung. Do
not remove it.
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

**Current corpus:** ~1187 forest trajectories at `~/Desktop/polyfly_ral/data/csvs/forests/`,
spanning forest types 0 and 2 (the only two that exist — see §5.9). Plus 1 maze and 2 descent
probes, both excluded from the built dataset: the maze is 500 Hz and the probes are a different
environment distribution.

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

### 5.8 The E1 methodology is validated on real data
`scripts/linear_audit.py` fits a ridge probe from the raw input window to every candidate target.
On the real corpus:

```
v               linear_trivial    1.0000   DISQUALIFIED
a               linear_trivial    1.0000   DISQUALIFIED
j               linear_trivial    0.9919   DISQUALIFIED
cable_dir       nonlinear         0.6945   eligible
R_quad_cols     nonlinear         0.3996   eligible
p_quad          nonlinear         0.6945   eligible
tension_margin  nonlinear         0.5035   eligible
```

Probing for the state the model was fed would have scored a perfect 1.0000 and meant nothing —
confirmed empirically, not hypothetically. The nonlinear targets sit at 0.40–0.69 from the raw
window, leaving real headroom for a trained encoder to improve on, which is what makes E1 a
question rather than a formality.

The audit also caught a degenerate target of my own: payload position at the anchor is identically
zero once positions are anchor-relative. Removed, with tests preventing recurrence. Re-run the audit
whenever the window config changes.

### 5.10 The prober must be fed physical units, not normalised state
Its nominal model integrates `ṗ = v`, `v̇ = a`, `ȧ = u`, which hold only in physical units:
per-channel normalisation gives p/v/a different scales (std 1.11 / 1.53 / 2.00) and non-zero
offsets. The rollout still ran while integrating the wrong quantities, and nothing in the loss curve
showed it. Fixed via `GPUResidentSplit.denorm`; see `docs/F6-physics-prober.md` §3b.

### 5.11 E5-a cannot literally pass, for a known reason
With physical units restored and the residual disabled, the nominal integrator leaves RMSE
0.26/0.39/0.45 over a 20-step rollout (data std ≈ 1.9). The raw CSVs are only *approximately*
self-consistent — per-step Euler mismatch 0.006/0.012/0.021 — because upstream fits states and
inputs with independent cubic splines on different knots (§5.6). Since the prober integrates
exactly, that mismatch is upstream interpolation, not physics. **E5-b must control for it or it will
just rediscover the spline inconsistency.**

### 5.9 Only forest types 0 and 2 exist
`forest_params.py` defines `FOREST_SMALL_OBS id=0` and `FOREST_LARGE_OBS id=2`; `generate_forest.py`
hits a bare `raise Exception()` for anything else — **per trajectory, while still exiting 0**, so a
type-1 phase reports "0/360 succeeded" and looks like a real run. `gen_all.sh` now defaults to
`"0 2"` and warns on anything else.

### 5.6 `sol_u` and `d(acc)/dt` are not interchangeable
Upstream `interpolate()` (`planner.py:1186`) fits states and inputs with **independent** cubic
splines, and inputs use knots `time_points[:-1]` (interval starts). So on the 500 Hz maze corpus the
`sol_u` channel is not the exact derivative of the acceleration channel: RMS relative error 9–16%
(52% on one short file), correlation 0.95–0.99 per axis, best alignment at a ~10–20 ms lag.

Use `sol_u`. It is what upstream actually fed into the flat map to produce `sol_quad_x`, so it is
the self-consistent choice — but do not treat it as `d(acc)/dt`, and do not substitute a numerical
derivative for it. On the 10 Hz forest corpus this is moot: those are the optimizer's own knot
values, not a spline resampling.

### 5.7 Smaller upstream landmines
- **`experiments/maze_2` crashes upstream** with `ValueError: Initial velocity must be near zero.`
  from `get_yaw_along_trajectory`. A genuine exit-1, correctly caught and reported by F1's driver.
- **The flatness cross-check does not import PolyFly.** `tests/upstream_ref.py` parses upstream's
  source with `ast`, lifts the exact function definitions, and executes them in a NumPy/SciPy
  namespace. This is deliberate: CasADi is not installed outside the planning container, and
  hand-copying the math would make the cross-check circular. Read that file's docstring before
  touching it — it raises rather than silently diverging if upstream renames anything.
- **The corpus is currently forest-type 0 only.** The first run died before reaching types 1 and 2;
  the restarted run covers all three. Check the type distribution before treating the corpus as
  diverse:
  `ls ~/Desktop/polyfly_ral/data/csvs/forests/ | sed 's/.*_f\([0-9]\)_.*/f\1/' | sort | uniq -c`

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

### Step 0 — rebuild the dataset when the corpus grows *(done once, repeat as needed)*
```bash
cd ~/Desktop/flatjepa
./.venv/bin/python scripts/generate_data.py manifest --polyfly-repo ~/Desktop/polyfly_ral
./.venv/bin/python -c "
from pathlib import Path; import subprocess
from flatjepa.data.manifest import read_manifest, filter_records
from flatjepa.data.dataset import build_dataset
from flatjepa.data.windows import WindowConfig
kept,_ = filter_records(read_manifest(Path('data/manifest.jsonl')))
forest = [r for r in kept if r.source == 'forest']
c = subprocess.run(['git','rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip()
print(build_dataset(forest,'data/windows_v1',config=WindowConfig(10,20),
                    extra_metadata={'commit':c,'corpus':'forests'}).format())"
./.venv/bin/python scripts/linear_audit.py --dataset data/windows_v1
```

### Step 1 — F4 dataset builder *(DONE)*
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
./.venv/bin/python -m pytest tests/ -q                                    # expect 230 passed, ~35 s
scripts/gen_launch.sh status                                              # generation state
find ~/Desktop/polyfly_ral/data/csvs/forests -name '*.csv' | wc -l        # corpus size
git log --oneline -5
```

## 10. Resuming from a fresh SSH session

Host `praneetho`, LAN address `192.168.4.43`, sshd active. The `claude` CLI is at
`~/.local/bin/claude`.

### Always work inside tmux

`claude` dies with the SSH connection. A dropped WiFi link mid-task loses the session, so start or
reattach a tmux session first — it is already installed at `/usr/bin/tmux`.

```bash
ssh praneetho@192.168.4.43
tmux new -A -s flatjepa          # create, or reattach if it exists
cd ~/Desktop/flatjepa
```

Detach with `Ctrl-b d`. Reattach later with `tmux attach -t flatjepa`. Long jobs launched inside
tmux survive disconnect; **nothing survives a reboot** (see §4).

### Orient before doing anything

```bash
cd ~/Desktop/flatjepa
git log --oneline -5
scripts/gen_launch.sh status
./.venv/bin/python -m pytest tests/ -q -m "not slow"      # ~9 s
```

If generation is not running and the corpus is short of target:

```bash
cd ~/Desktop/flatjepa && scripts/gen_launch.sh start
```

### Starting the assistant

Run `claude` from `~/Desktop/flatjepa` so it picks up the repo as its working directory, then paste
the prompt from **`START_PROMPT.md`** (which also has shorter variants for checking status, resuming
mid-feature, and recovering from a reboot).

Note the prompt restates the no-attribution rule explicitly. A fresh session does not inherit it,
and violating it in a commit trailer is silent.

**Closing your laptop lid** (the machine you SSH *from*) drops the connection and SIGHUPs anything
in the SSH session's foreground, including `claude`. Work inside tmux survives, and data generation
survives regardless because `gen_launch.sh` detaches with `setsid nohup`. The desktop itself has no
lid and cannot suspend — `sleep.target` and `suspend.target` are masked. Only a reboot loses work.

### What a fresh session most needs to know

Ordered by how much damage getting it wrong causes:

1. **§0** — no AI attribution anywhere; do not loosen test tolerances.
2. **§5.5** — the E1 tautology. The linear-decodability audit is specified but **not implemented**.
   Building it is part of F7 and must precede any reported E1 number.
3. **§5.1** — forest data is 10 Hz. Do not resample. Several docs originally said 20 Hz and were
   corrected; if you find a stale reference, fix it.
4. **§5.2** — CSV attitude is not flat-map attitude. Probing the wrong one is silently wrong by up
   to 80° of yaw.
5. **§3** — the container flags that fail *silently* when omitted.
6. **§6** — two decisions that are the author's to make, not the assistant's.

### Where the work is

- Design specs: `docs/F1`–`F11`, each with acceptance criteria. The specs are the contract; if
  reality contradicts a spec, correct the spec **and** say so in the commit, as was done for the
  10 Hz and E1 findings.
- Nothing has been trained. F4 is the blocker for F7/F8/F9/F10.
