# F10 — Baselines and Controls

**Purpose:** the comparison points without which no measurement in F7 is interpretable.

**Serves:** E1–E5.

---

## 1. These are not optional

The distinction between a baseline and a control matters here:

- A **control** establishes whether an effect exists at all.
- A **baseline** establishes whether the method is better than an alternative.

This project's central risk is a *spurious positive* — reporting that the latent "recovers" the flat
outputs when that information was trivially present in the input all along. Controls address that
risk and are therefore higher priority than baselines. They are specified in F7 §1 and repeated here
because they are the most skippable-looking and least skippable part of the project.

## 2. Controls (mandatory)

| Control | Implementation | Interpretation if it matches the trained model |
|---------|---------------|-----------------------------------------------|
| **Random-init encoder** | Identical architecture, no training, frozen | Training accomplished nothing; E1 is vacuous |
| **Raw input window** | Probe directly on the flattened normalized input | Information was already linearly present |
| **Shuffled labels** | Probe against permuted targets | Establishes the chance floor for this probe/dataset |

## 3. Model baselines

| Baseline | Why it is here |
|----------|----------------|
| **Direct state regression** | Same encoder, trained to predict future *states* rather than latents. Isolates what latent-space prediction contributes, since this is JEPA's core claim |
| **GRU/LSTM in observation space** | The classical recurrent alternative the project set out to compare against |
| **Linear/constant-velocity extrapolation** | The floor. On smooth optimizer trajectories at 20 Hz this may be surprisingly strong over short horizons, and reporting it prevents overclaiming |

The last one deserves emphasis: these are smooth, optimal, noise-free trajectories. A trivial
extrapolator will do well over short horizons. If it is not reported, short-horizon numbers will
look far more impressive than they are.

## 4. Explicitly deferred

**MPPI vs. distilled policy.** Considered and set aside. The planning-versus-policy tradeoff is
well covered by existing work (TD-MPC2 ships both a policy prior and sampling-based planning), so a
head-to-head here would reproduce known results rather than contribute. It may appear later as a
downstream demonstration that the learned model is useful — not as a research question.

## 5. Acceptance criteria

- [ ] All three controls implemented and run for every reported probe result
- [ ] Direct-regression and recurrent baselines share the encoder architecture, so comparisons
      isolate the objective rather than confounding it with capacity
- [ ] Trivial extrapolation baseline reported for all prediction-horizon results
- [ ] Baselines run under the same seeds, splits, and normalization as the main model
