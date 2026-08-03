# F4 — Dataset Builder

**Purpose:** turn per-trajectory CSVs plus F2/F3 derived quantities into windowed training tensors
with normalization statistics and leak-free splits.

**Serves:** all experiments.

---

## 1. The sampling-rate problem

This is the most important detail in this document and it is easy to miss.

> **CORRECTION (verified empirically).** The two corpora are written at **different rates**, and the
> original version of this section was wrong about the one that matters.
>
> | Corpus | dt | Rate | Rows/traj | Source |
> |--------|-----|------|-----------|--------|
> | `experiments/*` (mazes) | 0.002 | 500 Hz | ~1962 | `save_result` cubic-spline interpolation |
> | `forests/*` (**the training corpus**) | 0.1 | **10 Hz** | **~53** | planner knots, no interpolation |
>
> The forest generation path writes at the optimizer's own knots rather than through the 500 Hz
> interpolation. **Consequences that override §1 as originally written:**
>
> 1. There is nothing to downsample. Forest data is already at 10 Hz and **cannot be resampled up
>    to the 20 Hz target** — that target is unreachable and is hereby dropped. Windowing operates at
>    the native 10 Hz.
> 2. At H=10 / T=20, a 53-row trajectory yields only **~23 windows**, not the ~48 estimated earlier.
>    Window budget must be recomputed from trajectory count accordingly.
> 3. H=10 now covers **1.0 s** of history and T=20 covers **2.0 s** of prediction — which is
>    physically reasonable, so the window sizes survive even though the reasoning changed.
> 4. Knot-level `sol_u` is the optimizer's actual decision variable rather than a spline
>    resampling, which makes it *cleaner* as an action signal than the interpolated 500 Hz version.
>
> Retained below for the maze corpus, where the 500 Hz reasoning still applies.

Upstream `save_result` interpolates with `dt=0.002` — **500 Hz**. A measured trajectory
(`experiments/maze_1`) has 1962 rows over ~3.92 s, confirming this.

SkyJEPA's window sizes (H=10 history, T=20 future) assume **20 Hz**. Applied naively at 500 Hz:

- H=10 covers **20 ms** of history — essentially a single instant, carrying no dynamic information
- T=20 predicts **40 ms** ahead — a trivial extrapolation any linear model solves

Either mistake would produce results that look fine and mean nothing.

**Requirement:** resample to a target rate (default 20 Hz, stride 25) before windowing. The target
rate is a config parameter and must be recorded in the dataset metadata.

At 20 Hz, a ~3.9 s trajectory yields ~78 timesteps → ~48 windows with H=10/T=20. At the F1
throughput estimate of ~8k trajectories, that is on the order of 400k windows, which is ample.

**Resampling method:** subsample by stride, do not average. These are smooth optimizer outputs
already interpolated from a coarse solution; averaging would low-pass filter jerk, which is the
action signal. Record the stride so it is reproducible.

## 2. Window contents

Per window, at the resampled rate:

| Tensor | Shape | Source |
|--------|-------|--------|
| `state_hist` | (H, D_s) | CSV state columns |
| `action_hist` | (H, 3) | `sol_u_*` (payload jerk) |
| `action_future` | (T, 3) | jerk over the prediction horizon |
| `state_future` | (T, D_s) | targets for latent prediction |
| `flat_future` | (T, 12) | F2 ground truth, for probes |
| `regime_future` | (T,) | F3 labels |
| `valid_mask` | (H+T,) | F2 validity mask |

**Action definition.** The action here is **payload jerk** (`sol_u`), not motor forces. This differs
from SkyJEPA deliberately: jerk is the actual control input of PolyFly's triple-integrator model.
Documented so the difference is a choice, not an oversight.

**State definition.** Default `D_s = 9`: payload position, velocity, acceleration (`sol_x`). Note
that under the flat map the quadrotor state is a deterministic function of these, so including it
adds no information — but an ablation that includes it tests whether redundant inputs change what
the latent learns.

> **Design tension worth being explicit about.** If the model's input is `sol_x = (p, v, a)` and the
> probe target is `(p, v, a)`, then E1 is close to tautological: the network is handed the answer.
> **This must be addressed or E1 is meaningless.** Options, to be decided empirically:
>
> 1. Feed only a *partial* observation (e.g. position only, or position + velocity) so recovering
>    acceleration and jerk is genuinely inferential.
> 2. Probe for quantities not directly in the input: quadrotor attitude, cable direction.
> 3. Rely on the random-init encoder baseline to quantify how much is trivially present.
>
> Recommendation: **do all three.** Option 1 as the headline configuration, option 2 as the
> physically meaningful target, option 3 as the control that makes either interpretable. A version
> of E1 that hands the model its own probe target and reports R² ≈ 1.0 is not a result.

## 3. Normalization

- Statistics computed on the **training split only**, then applied to all splits. Computing them
  over the full corpus leaks test distribution into training.
- Per-channel mean/std.
- Position is translation-relative: express positions relative to the window's final history frame,
  not in world coordinates. Absolute world position is an artifact of environment layout and would
  let the model memorize environments.
- Stats are serialized with the dataset and versioned.

## 4. Splits

**Split by environment, at the trajectory level** — never by window. Windows within a trajectory
overlap by construction and are strongly correlated; a random window split places near-duplicates on
both sides and inflates every metric in F7.

Default 80/10/10 train/val/test, deterministic given a seed, assignment recorded in the F1 manifest.

Held-out structured set: the 9 maze environments are reserved entirely for qualitative evaluation.

## 5. Storage

Precompute windows to a memory-mappable array format rather than windowing on the fly. Windows are
cheap to store and the training loop should not be doing CSV parsing.

Persist alongside the tensors: target rate, stride, H, T, normalization stats, split assignment,
F1 manifest hash, and the flatjepa commit that produced them.

## 6. Acceptance criteria

- [ ] Resampling to target rate implemented and recorded; default 20 Hz
- [ ] Splits are environment-level and deterministic given a seed
- [ ] Normalization statistics computed on training split only — verified by test
- [ ] No window crosses a trajectory boundary — verified by test
- [ ] Windows exclude timesteps failing the F2 validity mask
- [ ] Positions are relative, not absolute world coordinates
- [ ] Dataset metadata records every parameter needed to reproduce it
