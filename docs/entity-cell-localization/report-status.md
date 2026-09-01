# Entity Cell Localization and Downstream Attribution: Status

**Status:** implementation complete; no model inference, discovery, calibration, or held-out test run.

**Protocol:** frozen V1 in [the proposal](proposal.md).

**Provenance:** branch `feat/entity-cell-confirmation`, based on T5 commit `ab20ca7`. The T6 implementation is intended for one follow-up commit on this branch. No numerical result or confirmation verdict exists.

## Implemented

- `entity-cell prepare` validates split-bound identities, renders the frozen header variants, records E2 source groups, and materializes deterministic donor contracts for `original`, `anonymous_identity`, `same_sector_swap`, and `name_form_control`.
- E1, E2 attribution/readout/patch preparation, and E3 discovery expose lifecycle-aware public APIs.
- Discovery summaries report counts and exclusions without issuing a confirmatory verdict.
- `prepare-confirmation-config` freezes E1/E2/E3 selections and doses. `analyze-confirmation` validates calibration/test records, computes ticker-level bootstrap intervals and sign-flip tests, applies the frozen Holm family, and handles authorization fail-closed.
- Confirmation output uses the compact `analyze/confirmation.json` schema when the output path is chosen under a run's `analyze/` directory. Parent hashes and configuration provenance are recorded in the run manifest.

## Evidence status

No formal run has been performed. The repository contains no new E1/E2/E3 or confirmation results from this implementation. Calibration and held-out test remain unrun; this branch does not authorize a test without evaluator output from a successful frozen calibration artifact.

## Next executable smoke command

After restoring the ignored editable workspaces and providing the frozen input files, prepare a tokenizer-only smoke artifact:

```bash
uv run entity-cell prepare \
  --input data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv \
  --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
  --baseline <adapted-baseline.jsonl> \
  --baseline-identity adapted:<version> \
  --model .cache/models/qwen3.5-4b \
  --run-id entity-cell-prepare-smoke \
  --artifact-root artifacts
```

The baseline path and adaptation identity remain placeholders because the frozen 399-record paper list is not present in this checkout. Do not run this command until an explicit adapted baseline contract exists.
