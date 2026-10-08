# T03 control limits: how to reproduce

Every number in `REPORT.md` comes from `data/validation.json`, and that file comes from two runs under
`artifacts/`. The `exp/T03-*` tag messages list each run's files and sha256.

## Report tables from `data/`

```bash
uv run python tasks/T03-control-limits/t03.py tables
```

## `data/` from the runs

```bash
uv run python tasks/T03-control-limits/t03.py build
```

`build` computes the validation cells with `scripts/summarize_confirmation_paper_tables.py` (the summarizer
behind the paper tables), and the C5 rule and readout disagreement counts with `scripts/summarize_confirmation.py`.

| Run | Path under `artifacts/<slug>/concept-cone-steering/runs/` | Tag |
|---|---|---|
| Full run | `confirmation-v1-20260925-full-01/` (arms `alpha0`, `cal`, `dim`, `ops`, `random`, `jitter`, `evidence`, `anon`) | `exp/T03-confirmation-v1-20260925-full-01` |
| Dose supplement | `confirmation-v1-supp-20261006-full-01/` (arms `dim`, `random`, `evidence`, `anon`) | `exp/T03-confirmation-v1-supp-20261006-full-01` |

## The runs

Both runs are the ones T02 uses. `tasks/T02-steering-operators/README.md` gives their commands.

## Code changes

None. `t03.py` reads stored results and runs no model.

## Provenance

The full run records `06efa5d` and the supplement records `de2aae5`, both with no uncommitted code, for all
four models.

## Differences from the paper draft

The paper draft of 2026-10-08 describes these experiments differently in the places below. The paper text has
not been changed. The IDs are shared with the T01 and T02 READMEs.

| ID | Paper draft | Experiment |
|---|---|---|
| P13 | Abstract and conclusion: against opposing evidence "three of four models never flip"; §5.3: no dose up to \|α\| = 16 flips any company in GLM, Gemma or GPT-OSS | True only up to \|α\| = 16. GLM flips 0.74 at −32 and 1.00 at −64 with every output parsed. Only Gemma never flips within the parsable range (up to ±64). GPT-OSS outputs break at ±32 |
| P14 | Table 4: "Random: flip rate of matched-norm random directions (maximum over five seeds), and Gain = Steering − Random" | Rand is the largest share, over five seeds, of all companies whose decision changes in either direction. DIM's denominator is the source class only, so Gain subtracts rates with different denominators |
| P15 | Limitations: "Part of the Gemma random baseline was run on different GPU hardware (†)"; Table 4 marks † only on Gemma sell→buy +2 to +16 | The supplement random directions are scaled to a DIM re-estimated on the newer GPU, with a median norm about 14% below the original, so they are not matched-norm. Gemma buy→sell −2 to −16 come from the same supplement and carry no † |
| P16 | Limitations: "Fixed-prefix margins disagree with generated decisions for 24% of outputs overall" | 24% is GPT-OSS alone: 22.7% with the pre-registered C4 arms (alpha0, dim, ops, evidence), 24.1% over all generated rows. Over all four models the share is 7.1% and 7.5% |
| P17 | Not mentioned | GPT-OSS evidence rows are not fully reproducible: regenerating one stored row at −32 changes its decision. This affects the two supplement Opposing cells at ±0.25 (both 0.00) |
