# Startup Prompt

Copy the block below into a fresh `claude` session started from `~/Desktop/flatjepa`.

Start it inside tmux first, or closing your laptop lid kills it:

```bash
ssh praneetho@192.168.4.43
tmux new -A -s flatjepa
cd ~/Desktop/flatjepa
claude
```

---

## The prompt

```
This is the flatjepa research project. Before doing anything, read these three files in
order: PROGRESS.md, docs/BACKGROUND.md, docs/00-overview.md.

Then read the design doc for whatever feature you're about to work on. The docs are the
spec; the code implements them.

Standing rules for this repo:
- Never add AI/Claude/Anthropic attribution anywhere: not in code, comments, docstrings,
  commit messages, or co-author trailers. All commits are authored as me.
- Never modify ~/Desktop/polyfly_ral except the parameter files listed in PROGRESS.md §8.
- Never loosen a test tolerance to make a test pass. If a threshold is unreachable,
  replace the criterion with a better-posed one and say so explicitly.
- If reality contradicts a design doc, fix the doc as well as the code, and say so in the
  commit message. This has already happened three times and those corrections are the most
  valuable content in the repo.
- Verify claims by running things. Several important findings in this project came from
  running code that "obviously" worked and finding it silently didn't.

Before changing anything, run the health check in PROGRESS.md §9 and tell me the current
state: test count, whether data generation is running, corpus size, and the last few
commits.

The next task is F4, the dataset builder (docs/F4-dataset.md). It blocks F7/F8/F9/F10.
Pay particular attention to PROGRESS.md §5.1 (data is 10 Hz, do not resample) and §5.5
(the E1 tautology, and the linear-decodability audit that has been specified but not yet
implemented).

Two decisions in PROGRESS.md §6 are mine to make, not yours. Raise them, don't resolve
them.
```

---

## Variants

**Continuing mid-feature** — replace the last two paragraphs with:

```
Pick up where the last session left off. Check `git log` and `git status` first, then tell
me what state the work is in before continuing.
```

**Just checking on things** — no need for the full prompt:

```
Read PROGRESS.md §9 and run the health check. Report the current state. Don't change
anything.
```

**Resuming after a reboot** — generation does not survive one:

```
The machine rebooted. Read PROGRESS.md §4, restart data generation, confirm it's running,
and tell me how many trajectories exist.
```

## Why the prompt is shaped this way

- **Reading order is specified** because PROGRESS.md is state, BACKGROUND.md is reasoning, and
  00-overview.md is the plan. A session that reads only the plan will re-derive decisions that were
  already made and rejected for stated reasons.
- **The no-attribution rule is repeated** because a fresh session does not inherit it, and it is
  easy to violate silently in a commit trailer.
- **"Verify by running things"** because the highest-value findings in this project — the silent
  permission failure, the 10 Hz sampling rate, failed solves writing valid-looking CSVs — all came
  from running code rather than reading it.
- **The decision boundary is explicit** because an assistant asked to work autonomously will
  otherwise resolve open research questions by picking one, and the E4 and `base.yaml` decisions
  have real scientific consequences.
