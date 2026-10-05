# CPU candidate panels are implemented, GPU execution is not

**D1:** The fixed historical-data-informed comparison panels are implemented in `llm_bias/core/stance_localization_candidate_panel.py`. No new localization effects, global peak or band are claimed. [Proposal](proposal.md) owns the protocol and interpretation limits, [development spec](../../dev/stance_localization_candidate_panel_spec.md) owns the API and acceptance contract.

| Canonical model | Zero-indexed layers | Full fit/validation primary cells |
|---|---|---:|
| qwen3.5-4b | 0, 3, 14, 15, 16, 31 | 72,384 |
| glm4-9b-0414 | 0, 4, 19, 20, 21, 39 | 72,384 |
| gemma4-12b-it | 0, 5, 23, 26, 27, 28, 47 | 84,448 |
| gpt-oss-20b | 0, 2, 7, 8, 9, 11, 14, 23 | 96,512 |

**R1:** GPT remains blocked on known full-parent replay drift. Strict continuation equality/no-op gates and frozen baselines remain unchanged. Current V1 partial effects were not read or used for panel selection. No GPU runner is implemented in this phase.

**A1:** Stop/archive authorization covers old jobs21/22/23/25/28/34/35 only. Expected handoff artifact: `artifacts/maintenance-snapshots/localization-stopped-for-candidate-panel-v2.json`. It is absent locally, so stopped states and SHA-backed partial archives are not verified here. Jobs27/41 are unaffected.

**A2:** Next phase is a separately reviewed panel-bound runner using all3,016 fit/validation pairs, all four whole spans, existing strict execution checks and fresh write-once run roots. Future top3 selection is within-panel only. CPU tests do not certify GPU research eligibility.

**Verified CPU acceptance:** Focused panel suite:26 passed. Panel plus existing alignment/pair suites:139 passed/1 skipped (optional actual-GLM parent check, whose ignored artifact is absent). `uv lock --check`, `git diff --check`, and unchanged `uv.lock` checks passed. No real checkpoint or remote scheduler was used.
