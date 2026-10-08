# AGENTS.md

This repo studies concept-cone steering: directions built from company-level buy/sell
preference differences (DIM, single neurons, multi-dimensional cones) are added to a decoder
LLM's residual stream to test whether its generated investment decision flips. It holds one
report per task. These rules keep `main` clean while experiments stay reproducible.

## Branches

| Branch | Purpose | Rules |
|---|---|---|
| `task/T<NN>-<name>` | one per task; experiments and trial and error | Anything goes, but never rebase, force-push or delete it |
| `clean/T<NN>-<name>` | branched from `task/T<NN>-<name>` when the task ends; holds only what passes the file rule | Opened as a PR to `task/T<NN>-<name>`. Deleted once that PR is merged |
| `chore/<name>` | changes outside any task: refactors, documentation, tooling, these rules | Short-lived. Opened as a PR to `main` and deleted once merged. A refactor must not change steering or evaluation results |
| `main` | runnable code and every finished task | Every new task branches from it. Receives a task only by merging its `task/` branch after the `clean/` PR |

Task numbers are two digits and never reused: `T01`, `T02`, ...

Tasks may run in parallel. Each task branches from the current `main` and works in its own git
worktree (`git worktree add ../llm-bias-T<NN> task/T<NN>-<name>`), so tasks never share a working
tree. Parallel tasks must not change the same code; when one needs another's change, it waits for
that task to reach `main`.

Files every task touches (the task tables in `README.md` and this file) are changed only in step 4,
after the latest `main` has been merged in.

Branches named `research/*`, `implement/*` and the remote branches without a prefix predate
these rules. They are history, not tasks.

## Workflow per task

1. Branch `task/T<NN>-<name>` from `main`. Commit the task's `PROTOCOL.md` before any GPU run
   whose results go into the report. Run GPU jobs with `lab run`, which executes the pushed
   commit.
2. For every run whose results go into the report:
   - copy its run directory from the lab host to
     `artifacts/<model-slug>/concept-cone-steering/runs/<run-id>/`, check the sha256 of every
     file against the host copy, then delete the host copy;
   - tag the commit it ran with an annotated tag `exp/T<NN>-<run-id>`. The tag message records
     what the report leaves out: model slug, job id, the repo-relative paths of the run's files
     under `artifacts/`, and their sha256. No host names: `lab show <job id>` gives the host.

   A tag never points to a different commit. Its message may be corrected by re-creating
   the tag on the same commit.
3. Branch `clean/T<NN>-<name>` from `task/T<NN>-<name>`. Move, rewrite or delete files until
   only what passes the file rule is left, and open a PR to `task/T<NN>-<name>`.
4. After that PR is merged, merge the latest `main` into `task/T<NN>-<name>` (a merge, never a
   rebase), update the shared files, and open a PR from `task/T<NN>-<name>` to `main`.
5. Keep the `task/` branch and its tags after the merge. Files deleted in step 3 stay
   reachable through them.

## File rule

A file goes to `main` only if it is needed to:

- **run** a steering experiment: model loading, hooks and interventions, operators, prompts,
  parsing, entry points, environment setup;
- **reproduce** a reported number or figure: protocols, experiment configs, eval data,
  analysis scripts;
- **understand** the repo: `README.md`, this file, task READMEs.

Everything else (debugging, abandoned attempts, superseded scripts and figures) is deleted on
the `clean/` branch. It stays in the `task/` branch's history and tags.

## Layout

```
llm_bias/core/                                  shared code: models, hooks, steering, parsing, provenance
llm_bias/entity_to_dial/ llm_bias/balanced_evidence_gap/   upstream prompt and direction code: frozen
scripts/                                        entry points (run as `uv run python scripts/<name>.py`)
analysis/                                       analysis tools used by two or more tasks
tests/                                          unit tests with fake models and temporary directories
tasks/T<NN>-<name>/
  PROTOCOL.md                                   pre-registered design, committed before the runs
  REPORT.md                                     results only
  README.md                                     how to reproduce: commands, configs, code changes
  configs/                                      experiment configs (dose grids, seeds, splits)
  data/                                         compact eval data behind every reported number
  figures/                                      figures used in REPORT.md
  *.py                                          analysis used only by this task
```

- A change that can alter steering or evaluation results goes behind a new argument or config
  key whose default keeps the original behaviour, so every earlier result still reproduces
  from `main`. Changes that only affect logging or which files are kept need no switch.
- New operators and runners go in new modules or scripts, not in edits to the ones earlier
  tasks ran.
- A task's analysis script moves to `analysis/` only when a second task needs it. The same PR
  updates the commands in earlier tasks' READMEs.

## Research rules

- A fixed-answer margin (`log p(buy) − log p(sell)`) only shows a readout change. A claim that
  steering flips a decision needs the decision-flip rate and parse rate of real greedy
  generations.
- Do not save raw activations, residuals, gradients or KV caches. Save compact statistics and
  provenance only.
- Finished runs, their numbers and committed protocols are never rewritten. A change of
  direction source, primary metric, controls or gate is a new task.
- A new run writes to a new `<run-id>` and never overwrites an earlier run.

## Reports

- `REPORT.md` presents results: setup, numbers, figures, comparisons, limitations.
- No artifact paths, hashes, host names or job logs in reports. Those go in the `exp/` tag
  messages.
- Every number and figure in a report must be regenerable from `data/` with a command in the
  task's `README.md`, and `data/` must be regenerable from the tagged runs under `artifacts/`.
  Numbers quoted from other sources (papers, other repos) name their source instead.

## Data and artifacts

Models, datasets, run outputs and job logs are never committed.

- Run outputs and job logs are kept on this machine, in `artifacts/` at the repo root
  (git-ignored), and referenced from `exp/` tag messages. `git clean -x` deletes `artifacts/`;
  do not run it.
- Models (`.cache/models/<slug>`) and datasets are inputs and can be re-downloaded. They may
  stay on the lab hosts as a cache.
- `third_party/jacobian-lens` is the editable `jlens` workspace member. Clone it as described
  in `README.md`, then `uv sync`. It is not committed.

## Working rules

- Change only what the task needs. Check `git status` before starting.
- Shared logic goes in `llm_bias/core/`. One-off experiment logic stays in its script. Do not
  abstract ahead of need.
- Commands, paths and states in documents must match the code and artifacts. Mark anything
  unverified as proposed.
- Set up the environment with `uv sync` and add packages with `uv add`.

## Verification

```bash
uv lock --check
uv run pytest -q
```

Unit tests use fake models and temporary directories and never load a real checkpoint. GPU
runs are checked separately as smoke runs.

## Tasks

| Task | Branch | Status |
|---|---|---|
| T01 layer localization | `task/T01-layer-localization` | Report in `tasks/T01-layer-localization/` |
| T02 steering operators | `task/T02-steering-operators` | Report in `tasks/T02-steering-operators/` |
| T03 control limits | `task/T03-control-limits` | Report in `tasks/T03-control-limits/` |
| T04 stance rebuild | `task/T04-stance-rebuild` | Code in `tasks/T04-stance-rebuild/`; no report |
