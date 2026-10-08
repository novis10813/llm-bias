# T02 steering operators: how to reproduce

Every number in `REPORT.md` comes from `data/operators.json`, and that file comes from three runs under
`artifacts/`. The `exp/T02-*` tag messages list each run's files and sha256.

## Report tables from `data/`

```bash
uv run python tasks/T02-steering-operators/t02.py tables
```

## `data/` from the runs

```bash
uv run python tasks/T02-steering-operators/t02.py build
```

`build` computes the dose-grid cells with `scripts/summarize_confirmation_paper_tables.py` (the summarizer
behind the paper tables) and the C8 verdict with `scripts/summarize_confirmation.py`.

| Run | Path under `artifacts/<slug>/concept-cone-steering/runs/` | Tag |
|---|---|---|
| Freeze | `confirmation-v1-freeze-20260925/` | `exp/T02-confirmation-v1-freeze-20260925` |
| Full run | `confirmation-v1-20260925-full-01/` (arms `gates`, `alpha0`, `dim`, `ops`) | `exp/T02-confirmation-v1-20260925-full-01` |
| Dose supplement | `confirmation-v1-supp-20261006-full-01/` (arms `dim`, `ops`) | `exp/T02-confirmation-v1-supp-20261006-full-01` |

## The runs

Check out the tag of the run first (`git switch --detach <tag>`), then run `uv sync`. Models are read from
`.cache/models/<slug>`. Both GPU runners refuse to start in `full` mode with uncommitted code.

Freeze (CPU):

```bash
uv run python scripts/freeze_steering_protocol.py --run-id confirmation-v1-freeze-20260925
```

Full run (GPU, one model per call). `scripts/run_confirmation_wave.sh` runs several models in sequence on one
GPU.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
  uv run python scripts/probe_steering_confirmation.py --model .cache/models/<slug> \
  --phase full --run-id confirmation-v1-20260925-full-01 --arms all
```

Dose supplement (GPU, one model per call). It reads the full run and adds only the doses in `SUPPLEMENT`.
Gemma ran only the `random` arm, with `--allow-operator-drift`, because its DIM direction did not reproduce on
the newer GPU (see T03).

```bash
uv run python scripts/probe_steering_confirmation_supplement.py --model .cache/models/<slug> \
  --phase full --run-id confirmation-v1-supp-20261006-full-01
```

## Code changes

T02 merges `research/confirmation-v1-supplement`, which adds the supplement runner, its test and the paper-table
summarizer. It does not change existing code. `t02.py` reads stored results and runs no model.

## Provenance

- The full run records `06efa5d` and the supplement records `de2aae5`, both with no uncommitted code, for all
  four models.
- The freeze records no git commit. Its outputs are 8–14 seconds newer than `e5b4a96`, the commit that added
  `scripts/freeze_steering_protocol.py`, so the tag is on that commit.

## Differences from the paper draft

The paper draft of 2026-10-08 describes these experiments differently in the places below. The paper text has
not been changed. The IDs are shared with the T01 and T03 READMEs.

| ID | Paper draft | Experiment |
|---|---|---|
| P06 | Table 3 uses one grid {±0.25, ±2, ±4, ±8, ±16} for all models. Appendix H: grids come from a pre-specified rule, and protocols were committed before the runs | The calibrated grids are ±{1, 2, 4, 8, 16, 32, 64} (Qwen, GLM), ±{0.25, 2, …, 64} (Gemma) and ±{0.125, 0.5, 2, …, 32} (GPT-OSS). The ±0.25 cells of Qwen, GLM and GPT-OSS come from the 2026-10-06 supplement, chosen after the full results were known, with no protocol document |
| P07 | §4.2: "DIM flip rates are monotone in dose, and no DIM cell falls below the 90% parse threshold"; Appendix E: "DIM stays parsable across the whole grid" | True only up to \|α\| = 16. Gemma at −64 flips 0.00 with every output parsed. Qwen at +64 and GPT-OSS at −32 parse 0.00 and 0.02 (Table 4) |
| P08 | §4: the cone axes are the leading principal directions of `R[p]`, computed at each position | The axes are shared across positions: the 100 residuals at all K tokens are unit-normalized and pooled, the leading eigenvectors of their uncentered Gram matrix are taken, and each is orthogonalized per token against `d̂[p]` (`llm_bias/core/steering/directions.py:cone_axes`). Signs come from the Pearson correlation with the margins of the 20 Top/Bottom companies |
| P09 | §4: a single MLP neuron "steers entirely along `w_{n*}`" | Not stated in the paper: the write vector is rescaled to `‖d[p]‖` at every token; Gemma and GLM include the post-MLP norm gain; GPT-OSS selects an expert neuron (expert 22, neuron 1617) from 92,160 candidates; the median per-token cosine to DIM is 0.020–0.091 |
| P10 | §4.2: "our dose grid is too coarse to compare transition widths formally" | A pre-registered smoothness rule (C8, on fixed-prefix margin steps) was run. It fails in all four models and holds only for sell→buy in Gemma and GPT-OSS (Table 5) |
| P11 | Appendix B: the steer-suffix "trailing token ids are identical across the four tokenizers" | K is 100, 98, 100 and 99, and the steer-suffix id hashes differ between models. The ids are identical across the five evidence conditions within each model |
| P12 | §5.2 compares DIM with random directions descriptively | The pre-registered C5 rule (Gain lower bound > 0 at ±α_50 and ±α_hi, DIM above jitter and its own off-target rate) fails in Gemma (lower bound −0.026 at −0.25 and 0 at −64). The jitter and shuffled-label controls were run and are not reported |
| P18 | Figure 1: Gemma-4-12B at layer 27 with one positive and one negative fact | No script or stored row produces `Example_Plot.png`. One positive and one negative fact is the `mixed2` condition, which was run only on the reduced grid |
| P19 | Appendix A: each model's pinned chat template is used | Gemma's template already adds `<bos>` and the loader adds another, so every Gemma run has a double BOS |
