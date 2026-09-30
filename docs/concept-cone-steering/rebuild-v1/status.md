# Rebuild-v1 has not produced a research result

**State:** implementation in progress; no new GPU experiment has been submitted. Unit tests do not establish experimental effects or checkpoint compatibility.

## P0 population code passes tests and awaits independent review

`llm_bias/core/population.py` enforces the frozen 2024 CSV digest, 503 ticker coverage, explicit issuer metadata and issuer-disjoint fit/validation/calibration/evaluation roles. Focused tests: 88 passed. Implementation full regression: 347 passed, 1 skipped, one existing tensor-to-scalar warning. Independent code/spec review is running; formal issuer metadata and role allocation are not frozen.

The editable `third_party/jacobian-lens` dependency was restored at `581d398613e5602a5af361e1c34d3a92ea82ba8e`. Before implementation, `uv lock --check` passed and the full suite returned 259 passed, 1 skipped, one existing warning.

Implementation contracts and sequential acceptance commands: [`../../dev/stance_rebuild_p0_spec.md`](../../dev/stance_rebuild_p0_spec.md).

## Shared evidence requires external-source acquisition and review

The user selected externally sourced shared two-slot evidence. The old hypothetical sentences and company-specific 427-ticker pool do not supply this input. Source research is running. No factual/evidence review, pair artifact or research prompt has been approved.

The population input was recovered from idlab; its SHA-256 matches `27c454d250ce513fda2016b43e7009ccbe0f5381e6c7648d3efe0d418c501a38`. The selected rows contain 503 tickers, including 29 Unspecified sectors. All share classes remain present; no complete sector-disjoint claim is authorized.

## Physical GPU placement still gates execution

Requested placement is idlab GPU1, idlab1 GPU1, idlab2 GPUs0/1. Existing lab cannot enforce index pinning. The user authorized an allowlist extension in `/home/novis/idlab`; implementation tests independently returned 314 passed, and code/spec review is running. The updated tool and inventory have not yet been activated.

Remote read-only preflight located checkpoint/input/dependency stores. Observed VRAM is a snapshot, not an availability guarantee; no other process was stopped and no device was claimed.

## Historical sources remain historical

Immutable compact `confirmation-v1-20260925-full-01` artifacts were copied locally for Qwen3.5-4B, Gemma4-12B-IT, GLM4-9B-0414 and GPT-OSS-20B. Source manuscript/bibliography copies were recovered into untracked `data/concept-cone-steering/rebuild-v1/source/` for audit. These are not rebuild-v1 experiments and have not been numerically re-audited.

Pending milestones: remaining P0 prompt/evidence/plan/metrics, schema/channel decoding and no-op/cache/scope checks; frozen protocol; baseline; all-layer generated localization; real neuron discovery; 1D generated-label DIM; independent sequence-trained cone; common controls/calibration; geometry/component-removal; reason/counterfactual/annotation; robustness; eligible summaries and paper updates. Missing, unsupported, untestable, unsuccessful training and negative effects must remain distinguishable.
