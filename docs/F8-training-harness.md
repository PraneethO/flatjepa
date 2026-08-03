# F8 — Training Harness

**Purpose:** configuration, training loops, checkpointing, logging, reproducibility.

**Serves:** all experiments.

---

## 1. Two-stage training

Mirrors SkyJEPA's structure, and the ordering matters:

**Stage 1 — latent dynamics.** Train `Enc_θ`, `Enc_φ`, `Pred_ψ` jointly on `ℒ_pred + λ_sig·ℒ_SIGReg`.

**Stage 2 — prober.** Freeze everything from stage 1. Train only the physics prober (F6) on the
supervised metric-state rollout loss.

The freeze must be verified, not assumed — assert that stage-1 parameters have zero gradient during
stage 2. A silent unfreeze would invalidate every claim about probing *frozen* representations,
which is the basis of E1, E3, E4, and E5.

## 2. Configuration

Config-file driven (YAML), with every experiment fully specified by its config plus a seed. No
experiment-relevant constants in code.

Each run records: full resolved config, config hash, git commit, dataset metadata hash, seed,
hardware, and library versions. A result that cannot be traced to these is not reportable.

## 3. Reproducibility

- Seeds set for Python, NumPy, and Torch; seeded DataLoader workers
- Deterministic algorithms enabled where the throughput cost is acceptable
- Multi-seed is the *default execution mode*, not an afterthought — F7 §7 requires spread on every
  number, so the harness should make single-seed runs the special case

## 4. Logging

Local-first (TensorBoard or plain JSONL); no dependency on an external service for core results.

Logged every epoch, beyond the loss:

- **Collapse diagnostics** (F5 §4): latent variance, effective rank, participation ratio — with an
  automatic alarm that halts or flags a collapsed run rather than letting it burn hours
- SIGReg penalty separately from prediction loss
- Residual vs. nominal magnitude separately (stage 2)
- Per-step prediction error across the T-step rollout, so error compounding is visible

## 5. Checkpointing

Save on a fixed interval and on best validation. Checkpoints must be loadable by F7 standalone,
carrying enough metadata to reconstruct the dataset and normalization used.

Long unattended runs must be resumable from the last checkpoint.

## 6. Execution environment

Training runs in the `ros-jazzy:pytorch` container (torch 2.13+cu130, CUDA verified on the RTX
3090). Data generation runs in `poly-fly:latest` (F1). These are separate images by design and
should not be merged — the planner needs CasADi and CPU only; training needs CUDA.

## 7. Acceptance criteria

- [ ] Two-stage training with verified parameter freezing in stage 2
- [ ] Fully config-driven; runs reproducible from config + seed
- [ ] Provenance recorded per run (commit, config hash, dataset hash, seed)
- [ ] Collapse alarm halts or flags degenerate runs automatically
- [ ] Resumable from checkpoint
- [ ] Multi-seed sweep runnable as one command
