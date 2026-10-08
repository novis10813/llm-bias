# T04 stance rebuild: code and how to run it

T04 rebuilds the decision pipeline on schema-constrained generated decisions: 503 S&P 500 tickers (2024
list, 500 issuers), shared FactSet 2024-11-15 market-context evidence in four conditions (`++`, `+-`, `-+`,
`--`), and four models (Qwen3.5-4B, Gemma-4-12B-IT, GLM-4-9B-0414, GPT-OSS-20B). It adds the runners for
baseline, localization, DIM, neuron discovery and cone training.

T04 reports no results and has no `REPORT.md`. Its protocol was never frozen (`PROTOCOL.md`), every run
records `research_eligible=false`, and several stages did not finish. Its runs have no `exp/` tags.

## Setup

```bash
uv sync
uv run pytest -q
```

Inputs are not committed. The compiler reads the constituents CSV (`data/sp500_constituents_2020_2025.csv`,
sha256 fixed in the script) and the FactSet PDF (sha256 fixed in
`configs/concept-cone-steering/rebuild-v1/inputs.json`), and refuses any other bytes:

```bash
uv run python scripts/compile_stance_inputs.py \
  --source-pdf <factset_20241115.pdf> --out-dir data/concept-cone-steering/rebuild-v1/compiled
```

`evidence-review.md` gives the source of each evidence item and the checks made on it.

## Entry points

All GPU runners take `--model <checkpoint> --inputs <compiled> --output-dir <run-dir>`. Runs are write-once
and resumable. A run's registration binds the code, model metadata, runtime and physical GPU UUID, so a run
resumes only on the same commit and the same GPU.

| Stage | Entry point | Extra arguments | State |
|---|---|---|---|
| Checkpoint smoke | `scripts/smoke_stance_checkpoint.py` | `--output`, `--stop-token-id` | Passed for all four models on one row |
| Baseline | `scripts/run_stance_baseline.py` | `--stop-token-id`, `--max-new-tokens`, `--timeout-seconds`, `--use-cache true`; GPT-OSS adds `--channel-policy harmony_no_tools --dtype native` | Complete for all four models, 2012 rows each |
| GPT-OSS truncation retry | `scripts/recover_stance_baseline_truncations.py` | `--parent <baseline>`, `--max-new-tokens 4096 --timeout-seconds 1200` | Complete: the 31 truncated rows |
| GPT-OSS merged baseline (CPU) | `scripts/merge_stance_baseline_recovery.py` | `--original`, `--recovery` | Complete |
| DIM fit | `scripts/fit_stance_dim.py` | `--parent` | Complete for GLM only |
| Cone teachers (CPU) | `scripts/compile_stance_cone_teachers.py` | `--parent` | Complete for GLM only |
| Cone training | `scripts/train_stance_cone.py` | `--parent`, `--teachers`, `--dimension`, `--seed` | GLM, k ∈ {2, 4} × three seeds |
| Cone validation | `scripts/run_stance_cone_validation.py` | `--parent`, `--training-dir`, `--training-audit` | Not finished |
| Neuron discovery | `scripts/run_stance_neuron_discovery.py` | `--parent`, `--shard-index`, `--num-shards` | Not finished |
| Localization discovery-v3 | `scripts/run_stance_localization_discovery_v3.py` | `--model-slug`, `--parent`, `--phase discovery` | Complete for GLM, Qwen and Gemma |
| Localization validation-v3 | same | `--phase validation --discovery-run <discovery-run-dir>` | Complete for GLM only |

`run_stance_localization.py`, `run_stance_localization_grouped.py`, `run_stance_localization_plain_crossmodel.py`
and `run_stance_localization_candidates.py` are earlier localization runners. None finished a run, but
discovery-v3 imports their record, gate and store code, so they stay.

Example (discovery-v3, GLM):

```bash
uv run python scripts/run_stance_localization_discovery_v3.py --model-slug glm4-9b-0414 \
  --model <checkpoint> --inputs <compiled> --parent <glm-baseline-run> \
  --output-dir artifacts/glm4-9b-0414/concept-cone-steering/runs/<run-id> --phase discovery
```

## Code changes

- Shared code, default behaviour unchanged: `llm_bias/core/inference/interventions.py` removes hooks that were
  registered before an error; `mlp_addition` gains an optional `selector` (the default still edits every
  position) and rejects non-integer coordinates and boolean deltas.
- New shared modules: `llm_bias/core/{population,stance_*,experiment_contract}.py`,
  `llm_bias/core/prompt_input/decision_prompt.py` and `llm_bias/core/inference/{structured_output,stance_*,harmony_*}.py`.
- `xgrammar==0.2.8` is a pinned dependency; `structured_output.py` works around an escape issue of that release.
- Two files sit outside the task layout because their paths are recorded in run identities:
  `configs/concept-cone-steering/rebuild-v1/` (input policy and decision schema) and
  `docs/concept-cone-steering/c2-v3-steering-prompt/status.md` (the historical candidate source, bound by
  sha256 in `llm_bias/core/stance_localization_candidate_panel.py`).

## Removed on `clean/T04-stance-rebuild`

These stay reachable on `task/T04-stance-rebuild` (last commit before clean-up: `d89ddd0`):

- `docs/dev/*_spec.md`: implementation specs used while building the runners.
- Status and proposal documents under `docs/concept-cone-steering/` (`PROTOCOL.md` points to the proposals).
- `scripts/run_stance_localization_harmony.py` and `llm_bias/core/inference/stance_localization_harmony_grouped.py`:
  the GPT-OSS localization runner. All three of its runs failed.
- `scripts/watch_stance_baseline_job.py`: job-monitoring tool.
- `llm_bias/core/decision_metrics.py`: metrics no runner uses.
