# Entity Cell Localization and Downstream Attribution: Status

**Status:** implementation complete; no model inference, discovery, calibration, or held-out test run.

**Protocol:** frozen V1 in [the proposal](proposal.md).

**Provenance:** branch `data/entity-cell-generic-baseline`, based on main commit `7d289e1`. No model experiment, numerical result, or confirmation verdict exists.

## Implemented

- `entity-cell prepare` validates split-bound identities, renders the frozen header variants, records E2 source groups, and materializes deterministic donor contracts for `original`, `anonymous_identity`, `same_sector_swap`, and `name_form_control`.
- E1, E2 attribution/readout/patch preparation, and E3 discovery expose lifecycle-aware public APIs.
- Discovery summaries report counts and exclusions without issuing a confirmatory verdict.
- `prepare-confirmation-config` freezes E1/E2/E3 selections and doses. `analyze-confirmation` validates calibration/test records, computes ticker-level bootstrap intervals and sign-flip tests, applies the frozen Holm family, and handles authorization fail-closed.
- Confirmation output uses the compact `analyze/confirmation.json` schema when the output path is chosen under a run's `analyze/` directory. Parent hashes and configuration provenance are recorded in the run manifest.

## Evidence status

No formal model run has been performed. The adapted generic baseline is now available at `data/entity-cell/generic-baseline-qwen3.5-9b-v1.jsonl` with identity `adapted:qwen3.5-9b-generic-cloze-v1` and SHA-256 `9e38d79887ba6fc313e9df0f98c63b4816e9a5c7067ec82421b9e0165a905179`. It contains exactly 399 validated records and is not the paper's exact Appendix A list. Calibration and held-out test remain unrun; this branch does not authorize a test without evaluator output from a successful frozen calibration artifact.

Tokenizer-only preparation is now the next step. This update reports dataset availability only; it does not claim a model experiment result.

## Next executable smoke command

After restoring the ignored editable workspaces and providing the frozen input files, prepare a tokenizer-only smoke artifact:

```bash
uv run entity-cell prepare \
  --input data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv \
  --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
  --baseline data/entity-cell/generic-baseline-qwen3.5-9b-v1.jsonl \
  --baseline-identity adapted:qwen3.5-9b-generic-cloze-v1 \
  --model .cache/models/qwen3.5-4b \
  --run-id entity-cell-prepare-smoke \
  --artifact-root artifacts
```

The input CSV and split manifest remain required local inputs. This preparation step loads a tokenizer only; do not run model inference here.
