# flatjepa — What Was Built, Why, and What It Found

A plain-language write-up of this project from start to finish. Written to be readable end-to-end
without needing the code open.

**One-sentence summary:** we set out to test whether a JEPA world model learns the physically
correct internal representation of a drone-with-payload system, built the measurement apparatus to
check it against exact ground truth, and found that it does not — the trained model's
representation is no better than an untrained one.

---

## 1. Where this started

The original idea was: *take PolyFly's trajectory data and fit a transformer instead of an RNN, and
see if it predicts trajectories better.*

That's a reasonable engineering question but a weak research one, for a specific reason. PolyFly's
trajectories come out of an optimizer, not a real drone. They are noise-free and governed by a
**triple integrator** — position's derivative is velocity, velocity's derivative is acceleration,
acceleration's derivative is the control input. That's exactly integrable. Almost any sequence model
fits it near-perfectly. A benchmark where every method scores ~100% measures nothing.

So the question had to move from *architecture* to *representation*.

### The JEPA angle

A **JEPA** (Joint Embedding Predictive Architecture) predicts in a compressed internal space rather
than predicting raw observations. Its advantage is that it can throw away whatever it can't predict,
instead of wasting capacity modeling noise.

Its problem is the flip side: **you can't see what it kept.** On standard benchmarks there's no
"true" answer for what the internal representation should contain, so the field measures it
indirectly — train a controller on top, see if the robot does well. That conflates three separate
things: did the representation capture the state, did the predictor learn the dynamics, and is the
controller any good? When the number is bad you don't know which failed.

### The insight that made this a project

This particular system is **differentially flat**. That means there's an exact, closed-form formula
mapping a small set of quantities (the payload's position and its derivatives) to the *entire*
system state — where the drone is, how it's tilted, which way the cable points.

If that formula exists, then **the correct internal representation is known in advance**, in both
size and content. Which turns "is this representation any good?" from a guess into a measurement.

That's the project: *use differential flatness as a measuring instrument for JEPA representations.*

### What we deliberately did not do

- **MPPI vs. a distilled policy.** Considered as the headline comparison, rejected — TD-MPC2 already
  ships both and the tradeoff is well characterized. Reproducing it isn't a contribution.
- **Isaac Gym simulation.** Would give a genuine sim-to-real-style residual, but it needs Python 3.8
  and is deprecated; this machine has 3.10 and 3.12.
- **Real flight logs.** None exist on this machine.
- **Camera/depth perception.** Deferred — it would add an entire subsystem before any result exists.

---

## 2. What existed before

The `polyfly_ral` repository: an optimal trajectory planner for a quadrotor carrying a
cable-suspended payload. It could generate trajectories. It had **no** machine learning code, **no**
generated data (`data/csvs/` was empty), and the host machine had no numpy, no PyTorch — nothing.

So essentially everything below was built from zero.

---

## 3. What was built

A new repository, [flatjepa](https://github.com/PraneethO/flatjepa), with eleven components:

| Part | What it does |
|---|---|
| **F1** Data generation | Runs the planner in parallel to produce thousands of trajectories |
| **F2** Flat-output extractor | Computes the exact ground-truth answer from physics |
| **F3** Cable tension | Detects when the cable would go slack (physics breaks down there) |
| **F4** Dataset builder | Cuts trajectories into training windows without leaking data |
| **F5** JEPA model | The encoders, predictor, and anti-collapse regularizer |
| **F6** Physics prober | Decodes the representation using known physics equations |
| **F7** Measurement suite | The actual experiment: probes plus the controls that make them mean something |
| **F8** Training harness | Two-stage training with checks that the model is really frozen |
| **F9** Reporting | Tables and figures, regenerable from raw results |
| **F10** Baselines | Simple methods to compare against |
| **F11** Perception | Deliberately deferred; design recorded |

Every part has a design document written *before* the code, with explicit pass/fail criteria.
**235 automated tests** pass.

### Three things that had to be fixed just to generate data

None of these were documented anywhere; each was found by running the thing and watching it fail.

1. **The planner crashes without a screen.** It hard-codes an interactive plotting backend. Solved
   with a small shim rather than modifying the upstream repo.
2. **The container can't write files, silently.** It runs as a different user than the host. The
   planner *reports success* and only the exit code reveals the failure — a batch job ignoring exit
   codes would run all night and produce nothing.
3. **A required config file doesn't ship.** Forest generation needs `forests/base.yaml`, which
   upstream doesn't include. We wrote one, and documented the two values we had to guess.

---

## 4. Things discovered along the way

These matter more than the code, and several changed the experiment's design.

### 4.1 The data is 10 Hz, not 500 Hz

The original plan assumed 500 Hz and called for downsampling. Measurement showed the forest
trajectories are written at 10 Hz — the plan was impossible. Windowing now runs at native rate.

### 4.2 The recorded drone orientation is not the physics orientation

Upstream applies an extra yaw rotation *after* computing the physics, and saves that. Using the
saved orientation as ground truth would be wrong by up to **80 degrees**. Position and velocity are
saved *before* that step and are correct to 15 decimal places.

### 4.3 A failed solve still writes a perfectly normal-looking file

One trajectory failed to solve, yet produced a 2381-row file with a success exit code that passes
every structural check. And the solver's status is printed to the screen and then **thrown away** —
never saved. So it must be captured at generation time or it's gone forever.

### 4.4 The physics prober was being fed the wrong units

The prober integrates "position's derivative is velocity." That's only true in real physical units.
The data was normalized, which gives position, velocity, and acceleration different scales — under
which the equation is simply false. **It ran fine and nothing in the loss curve showed it.** This is
exactly the class of silent error the calibration check exists to catch.

### 4.5 The residual has a known, non-physical source

Once units were fixed, the physics still didn't predict the data perfectly. Investigation: upstream
fits the states and the control inputs with **separate** interpolating curves, so the recorded jerk
isn't quite the jerk that produced the recorded acceleration. The leftover error is an interpolation
artifact, not physics. Worth knowing, because otherwise a future experiment would "discover" it and
mistake it for something meaningful.

### 4.6 The slack-cable experiment (E4) has no data

We wanted to test whether the model discovers the moment a cable goes slack. Measurement: **zero**
such moments exist in the corpus. Cause: the planner confines the payload to a 0.75 m vertical
corridor while flying 18 m horizontally, so it never builds enough downward speed.

Rather than guess whether that was fixable, we generated test trajectories:

| Scenario | Drop height | Closest to slack (0 = slack) |
|---|---|---|
| Normal corpus | — | 0.841 |
| 6 m descent | 6 m | 0.600 |
| **20 m descent** | 20 m | **0.152** |

**E4 is recoverable but needs its own dataset.** A modest descent isn't enough. This is an open
decision for you.

---

## 5. The central design problem, and how it was fixed

This is the most important section.

**The trap:** if you feed the model position, velocity and acceleration, and then test whether its
representation "contains" position, velocity and acceleration — of course it does. You handed it the
answer. R² ≈ 1.0, and it means nothing.

**The subtler trap** — which the original plan walked straight into: the obvious fix is to feed the
model *less* (say, position only) and test whether it recovered the rest. **This doesn't work.**
Because the model sees a *window* of past positions, velocity is just a difference between
consecutive positions, and acceleration is a difference of differences. These are *linear* functions
of the input. A trivial linear fit recovers them exactly, with no model involved.

That's worse than the original trap, because it *looks* like it was controlled for.

**The fix:** test only quantities the physics computes **non-linearly** — the drone's orientation,
the cable's direction, the cable tension. These involve normalizations and rotations that no linear
fit can produce from the raw input.

**And enforce it mechanically.** Every candidate quantity is first tested with a plain linear fit on
the raw input. Anything the raw input already solves is *automatically disqualified*. Measured on
real data:

| Quantity | Type | Linear fit on raw input | Verdict |
|---|---|---|---|
| velocity | linear | **1.0000** | disqualified |
| acceleration | linear | **1.0000** | disqualified |
| jerk | linear | 0.9919 | disqualified |
| cable direction | nonlinear | 0.6945 | usable |
| drone orientation | nonlinear | 0.3996 | usable |
| cable tension | nonlinear | 0.5035 | usable |

The first three confirm the trap was real, not hypothetical. This check also immediately caught a
mistake of our own: one intended target was mathematically always zero.

---

## 6. Results

**30 training runs**: 3 random seeds × (5 regularizer strengths + 5 model sizes). 60 epochs each.

Every measurement is reported against three controls:
- an **untrained** model of identical architecture,
- a plain **linear fit on the raw input**,
- **scrambled labels** (the chance floor).

### Result 1 — the trained model is no better than an untrained one

| Quantity | Trained | Untrained | Raw input | Trained − Untrained |
|---|---|---|---|---|
| velocity | 0.634 | 0.719 | 1.000 | **−0.085** |
| acceleration | 0.593 | 0.748 | 1.000 | **−0.155** |
| jerk | 0.485 | 0.537 | 0.994 | **−0.053** |
| cable direction | 0.548 | 0.581 | 0.695 | **−0.034** |
| drone orientation | 0.420 | 0.455 | 0.485 | **−0.035** |
| drone position | 0.548 | 0.581 | 0.695 | **−0.034** |
| cable tension | 0.407 | 0.375 | 0.505 | +0.032 |

On **six of seven** quantities the trained model scores *below* the untrained one. The raw input
beats both on all seven. The one positive (+0.032) is smaller than the run-to-run variation.

**This is a negative result, and the stopping condition we wrote in advance says to report it as
one.** Training did not produce a more useful representation than random initialization.

Note the numbers 0.41–0.63 would have looked respectable in isolation. Only the untrained-model
control reveals they're worthless.

### Result 2 — the representation is smaller than physics requires

The physics says 9 numbers are needed. Measured:

| Model size | Effective size used | Untrained model |
|---|---|---|
| 4 | 1.83 | 2.77 |
| 8 | 1.84 | 3.05 |
| 16 | 1.27 | 3.05 |
| 24 | 2.07 | 3.35 |
| 48 | 4.46 | 3.74 |
| 96 | 5.08 | 4.27 |

It never reaches 9. It tracks how much room you give it rather than what the physics needs. And
training makes it **smaller** than random initialization at most sizes.

That's consistent with Result 1: a representation that threw away information can't reveal it later.

### Result 3 — the anti-collapse regularizer causes the collapse it should prevent

"Collapse" is JEPA's known failure mode: the model outputs the same thing for every input, which
makes prediction trivially perfect and the representation useless. SIGReg exists to prevent it.

| Regularizer strength λ | Prediction loss | Effective size |
|---|---|---|
| 0.0 (off) | 0.0001 | **4.97** |
| 0.01 | 0.0001 | 4.19 |
| 0.1 | 0.0001 | 4.30 |
| 1.0 | 0.0128 | **2.07** |
| 10.0 | 0.1548 | 2.06 |

Two surprises:

1. **With the regularizer off, nothing collapses** — and prediction is *best*. Our design document
   predicted collapse here.
2. **Turning it on collapses things.** At λ=1.0, two of three runs collapse to an effective size of
   **1.01** — a single dimension.

We tested the regularizer directly with synthetic inputs of known size:

| Input | Penalty | Actual size |
|---|---|---|
| ideal (full size) | 0.0003 | 23.85 |
| constant (total collapse) | 0.4089 | 0.00 |
| **1-dimensional** | **0.1668** | 1.00 |
| **9-dimensional** | **0.5148** | 6.45 |

**The penalty gets it backwards.** A 9-dimensional representation is penalized *more* than a
1-dimensional one. The mechanism: the check tests whether random 1-D shadows of the data look like a
bell curve. A collapsed representation's shadows *do* look like bell curves. So it catches a totally
constant output but not a collapsed-to-one-line output.

**Important caveat:** this is a statement about *our implementation and settings*, not about the
published method. The top open item is checking it against the reference implementation —
specifically whether it rescales the data before testing, which would remove the confound. That is
the difference between "we have a bug" and "the method has a weakness," and we are not claiming the
second.

---

## 7. What this means

**The honest reading:** on this system, with this architecture and this much training, the JEPA did
not learn a physically meaningful representation. It learned to predict its own outputs very well
(prediction error 0.0001) while its representation stayed no more useful than random.

That gap is the point. "The training objective was optimized well" and "a useful representation was
learned" are usually assumed to travel together. Here they came apart, and the flatness ground truth
is what made that visible. That's the instrument working, even though the model didn't.

**What's solid regardless of the negative result:**
- The disqualification check works, and changed the experiment *before* it produced a misleading
  number.
- The controls work. Without the untrained-model comparison, Result 1 would have read as a success.
- The freeze check, the units check, and the zero-variance-column handling each caught a real bug.

---

## 8. Limitations

- **60 epochs, one architecture.** The encoder was inherited from SkyJEPA, where it was sized to run
  at 100 Hz on embedded hardware — a constraint that doesn't apply here. A bigger model might behave
  differently.
- **The regularizer implementation is unverified** against the reference.
- **1,187 trajectories** from one environment generator. Two environment types, not a wide variety.
- **The config file we wrote** for forest generation involved two guessed values. Fine for these
  self-contained experiments; not fine for comparing against published PolyFly numbers.
- **No baselines run yet.** The code exists; the comparison doesn't.

---

## 9. What to do next

**Decisions that are yours, not the code's:**

1. **Verify the regularizer** against [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa).
   Highest value per hour of anything on this list — it determines whether Result 3 is a finding or
   a bug.
2. **E4 (slack cable)**: build the descent dataset, or drop the experiment and report the diagnosis.
   The evidence says it's recoverable.
3. **Whether to push on the negative result.** Options: train much longer, try a larger encoder, or
   report as-is. A rigorous negative with working controls is publishable in the right venue, and is
   more useful than a fragile positive.

**Mechanical next steps:** run the baselines (code written), and generate more data — the run is
still going.

---

## 10. Where everything is

```
flatjepa/
├── WRITEUP.md              this file
├── PROGRESS.md             current state + how to resume from a fresh session
├── START_PROMPT.md         paste-ready prompt for a new session
├── docs/
│   ├── BACKGROUND.md       why the project is shaped this way
│   ├── 00-overview.md      the experimental plan
│   └── F1..F11             one design document per component
├── outputs/e1e3/
│   ├── RESULTS.md          full results with interpretation
│   ├── results.json        raw numbers from all 30 runs
│   └── fig1..fig3          figures
├── src/flatjepa/           the code
├── scripts/                entry points
└── tests/                  235 tests
```

Everything is committed and pushed. All figures regenerate from `results.json` with one command.
