# The candidate runner supports three plain routes, with GPT explicitly unsupported

**D1:** `scripts/run_stance_localization_candidates.py` implements the separate
panel-bound production entry for GLM/Qwen/Gemma. Only layer coverage changes:
all 3,016 fit/validation pairs, all four post/full spans, original parent policies,
strict full-output gates, grouped execution and write-once records remain required.
See the [runner spec](../../dev/stance_localization_candidates_runner_spec.md).
The earlier CPU-only `status.md` and frozen proposal remain unchanged.

**R1:** GPT execution fails explicitly before loading inputs or creating artifacts.
Its fixed 96,512-cell panel is planning-only. Strict merged native Harmony parent
replay and row-specific recovery support require a separate implementation. Known
parent drift is not bypassed. No GPU completeness, global peak/band or generated
effect is claimed for any model.

**A1:** Main can review this runner for first GLM/Qwen/Gemma dispatch using fresh
model-specific run roots. Main owns the previously verified seven-job stop/archive
handoff. This implementation did not inspect partial effects, operate a scheduler,
load real checkpoints or depend on deleted remote worktrees.

CPU verification: candidate runner + panel suites **48 passed / 6 skipped**.
Accepted plain/grouped/Harmony, transient-capture and hook-cleanup suites
**352 passed / 36 skipped**, with 15 expected transient-capture error-path hook
warnings. `uv lock --check`, `git diff --check` and unchanged `uv.lock` checks pass.
Skips include incompatible fake routes and absent optional recorded parent artifacts.

The additional full `PYTHONPATH=$PWD uv run --no-sync python -m pytest -q`
attempt exceeded the 900-second command timeout at approximately 38% progress.
No failures appeared before termination. Full-suite completion is not certified.
