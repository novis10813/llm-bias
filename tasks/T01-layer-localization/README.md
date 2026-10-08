# T01 layer localization: how to reproduce

Every number in `REPORT.md` comes from `data/`, and `data/` comes from three runs under `artifacts/`. The
`exp/T01-*` tag messages list each run's files and sha256.

## Report tables from `data/`

```bash
uv run python tasks/T01-layer-localization/t01.py tables
```

## `data/` from the runs

```bash
uv run python tasks/T01-layer-localization/t01.py build
```

`build` reads the runs below for the four models and checks that each Stage 1 `analyze/summary.json` matches
the sha256 bound in `llm_bias/core/steering/protocol.py:MODEL_REGISTRY`.

| Run | Path under `artifacts/<slug>/` | Tag |
|---|---|---|
| Stage 1 | `balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/` | `exp/T01-phase2b-v2-427-01` |
| Overlap exclusion | `concept-cone-steering/runs/audit-v1-20260925/` | `exp/T01-audit-v1-20260925` |
| Steering-prompt patching | `concept-cone-steering/runs/confirmation-v1-20260925-full-01/{c2v3,c2v3_gen}/` | `exp/T01-confirmation-v1-20260925-full-01` |

## The runs

Check out the tag of the run first (`git switch --detach <tag>`), then run `uv sync`. Models are read from
`.cache/models/<slug>`.

Stage 1 (GPU, one model per call; GPT-OSS adds `--dtype native`). The runner runs Phase 2A
(`phase2a-v2-427-01`) and then the Phase 2B sweep. If the Phase 2A gate fails, it passes `--gate-override`.
Qwen ran through `scripts/downloads/run_v2_427_qwen4b.sh`, which runs the same steps with the model and GPU
fixed.

```bash
bash scripts/downloads/run_v2_427_crossmodel.sh .cache/models/<slug> <slug> <gpu> [--dtype native]
```

Overlap exclusion (CPU):

```bash
uv run python scripts/summarize_c2_overlap_exclusion.py --run-id audit-v1-20260925
```

Steering-prompt patching is part of the confirmation-v1 full run (GPU, one model per call). T01 uses only its
`c2v3` and `c2v3_gen` arms. The runner refuses to start in `full` mode with uncommitted code.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
  uv run python scripts/probe_steering_confirmation.py --model .cache/models/<slug> \
  --phase full --run-id confirmation-v1-20260925-full-01 --arms all
```

## Code changes

None. T01 adds only `t01.py`, which reads stored results and runs no model.

## Provenance gaps

- The Stage 1 runs record no git commit. They finished before `7bf2e8f`, the first commit with the v2 code and
  protocol, so the tag on `7bf2e8f` marks the inferred code version.
- The overlap-exclusion run records no git commit either. Its output files are 20 seconds newer than
  `e5b4a96`, the commit that added `summarize_c2_overlap_exclusion.py`, so the tag is on that commit.
- The confirmation run records `06efa5d` with no uncommitted code for all four models.

## Differences from the paper draft

The paper draft of 2026-10-08 describes these experiments differently in the places below. The paper text has
not been changed. The IDs are shared with the T02 and T03 READMEs.

| ID | Paper draft | Experiment |
|---|---|---|
| P01 | §4.1: Stage 1 uses "a fixed set of 402 construction companies" | 427 companies from `data/baseline/investment-dial/exploratory-v1.json`, 89 of them in the evaluation set. Without them, the peaks and T are unchanged (Table 1) |
| P02 | §3 and the Appendix A conditions table: positive, negative, mixed and zero conditions are used for layer localization | Stage 1 uses only `pos` and `neg`. `mixed2` and `zero` were run only in the confirmation `evidence` and `anon` arms, and the paper reports neither |
| P03 | Appendix A: GPT-OSS uses low reasoning effort. Appendix G mentions only "a different reasoning setting" | GPT-OSS L14 was selected under `medium` reasoning, and every steering run uses `low` |
| P04 | Appendix H: "the protocols were committed before the corresponding runs" | Stage 1's protocol was committed after its runs, marked development, with no layer-selection rule fixed in advance. The Gemma and GLM Phase 2A gates failed and were overridden |
| P05 | Appendix G: the readout is the realized-path margin read during the model's own greedy generation | The primary readout is a teacher-forced margin on the target's unsteered generation, so a patched forward is read on the target's path, not on its own |
| P20 | Appendix G table: GPT-OSS 70% band "1–8" | The band is layers 1 and 8 only. Layers 2–7 fall below 0.7 × peak |
