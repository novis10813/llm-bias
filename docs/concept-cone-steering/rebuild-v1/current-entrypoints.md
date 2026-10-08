# Only rebuild-v1 entrypoints may feed its main tables

**State:** implementation in progress. No current script has produced an eligible rebuild-v1 research result. Listing a path below does not mean it exists or passed GPU validation.

## New contracts are distinct from historical experiment orchestration

Implemented and independently accepted: `llm_bias/core/population.py`, `stance_evidence.py`, `prompt_input/decision_prompt.py`, `experiment_contract.py`, `inference/structured_output.py` (narrow plain ByteLevel BPE capability only). Core modules are not result-producing research entrypoints.

Diagnostic entrypoint: `scripts/smoke_stance_checkpoint.py` loads the approved full manifest and runs one row with a real CUDA checkpoint and four-arm no-op test; all outputs explicitly remain research_eligible=false. This is a smoke, not a complete baseline cohort or formal result.

Accepted compiler: `scripts/compile_stance_inputs.py`; input materialization only, never a generated result. It produced the complete503-ticker/2012-row pack in ignored `data/concept-cone-steering/rebuild-v1/compiled/`. Planned primary stages: `stance_baseline.py`, `stance_localize.py`, `stance_neurons.py`, `stance_dim.py`, `stance_cone.py`, `stance_compare.py`, `stance_geometry.py`, `stance_reasons.py`, `stance_summarize.py`. These become current only after their specs, full-manifest/gate tests and actual capability checks pass. Primary result eligibility binds protocol, input, schema/template, model/code/backend, selection/calibration and parent identities and exact coverage.

## Historical scripts stay available but are not rebuild producers

| Preserved family | Historical use / exclusion |
|---|---|
| `probe_dim_steering.py`, `probe_operator_comparison.py`, `probe_concept_cone.py` | Legacy/upstream ranking and adapted controls; not approved generated-label rebuild operators. |
| `probe_steering_confirmation.py`, `core/steering/{protocol,prompts,directions,evaluate,summary}.py` | Reproduce confirmation-v1, including its own renderer/parser/margin diagnostics; cannot relabel as schema-constrained rebuild. |
| `probe_rdo_cone*.py`, `probe_rdo_evidence.py` | Margin-objective historical learned directions, distinct from new sequence CE/ablation/KL cone. |
| `probe_evidence_scan.py`, `build_evidence_pool.py` | Historical 427-company/four-slot/company-specific evidence; not shared full503 input compiler. |
| `balanced_evidence_gap_phase2*.py`, `entity_to_dial_heldout_transfer.py` | Frozen upstream studies. No new method may import their constants/rankings/margins as primary rebuild selection. |
| Existing `plot_*`, `summarize_*`, `reparse_*` | Their historical schemas only; a CPU audit is separately labeled, never a fresh generation experiment. |

No bulk move to `scripts/legacy/` is needed. Preserve imports, historical commands/provenance and frozen proposals/status/artifacts. New summarizer must allow only registered rebuild schema/stages, not scan arbitrary `result.json` files. Missing/unsupported/untestable/negative/incomplete statuses stay explicit.
