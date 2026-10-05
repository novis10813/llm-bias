# Historical candidates replace the all-layer localization map

**D1:** This new version compares a fixed historical candidate panel, not a new global peak or band. The user approved replacing the old all-layer localization design with 6–8 layers per model. Old protocols, partial outputs and results remain frozen. This phase implements only the CPU panel/protocol/tests, with no GPU runner or launch.

## D2: Historical values and independent anchors fix every layer

For each canonical model, union the prior C2 v3 teacher-forced `steer_suffix` peak ±1, the single prior v2 steering injection layer, and independent anchors L0, `floor((L-1)/8)`, `floor((L-1)/2)`, L−1. Clip neighbors to `[0,L)`, deduplicate and sort zero-indexed layers. Actual loaded layer count must equal the registered expected count before execution. No unchecked CLI layer count or model alias can define a panel.

| Canonical slug | Actual/expected L | v3 peak | v2 injection | Frozen layers | Development cells |
|---|---:|---:|---:|---|---:|
| qwen3.5-4b | 32 | 15 | 16 | 0, 3, 14, 15, 16, 31 | 72,384 |
| glm4-9b-0414 | 40 | 20 | 19 | 0, 4, 19, 20, 21, 39 | 72,384 |
| gemma4-12b-it | 48 | 27 | 27 | 0, 5, 23, 26, 27, 28, 47 | 84,448 |
| gpt-oss-20b | 24 | 8 | 14 | 0, 2, 7, 8, 9, 11, 14, 23 | 96,512 |

Historical source: [C2 v3 status](../c2-v3-steering-prompt/status.md), repository-relative `docs/concept-cone-steering/c2-v3-steering-prompt/status.md`, SHA-256 `8a7762e97b934562167c6450bfdf441dca43c6815dc94bd97c20b2b22669b1b7`. The CPU constructor checks local source bytes against this pin. Hashes are provenance, not checkpoint authentication. The canonical record binds model slug, expected/actual count, historical values/source/hash, rule and interpretation limits.

**R1:** Historical margin localization is not generated-decision proof. C2 v3 used a different steering prompt and construction-only cross-company pairs, teacher-forced clean-path primary margin and potentially off-path fixed-prefix secondary margin. Qwen/GLM R7 had zero opposite-clean denominators, not 0% effects. GPT's historical v3 band excludes v2 injection L14. These historical observations motivate candidates but do not establish current-prompt sites.

**R2:** This is a historical-data-informed prospective candidate rule, not untouched preregistration. Current V1 effects were partially collected, but were not read or used to select this panel or its spans. Do not inspect partial effects to revise it. The four whole-span arms are fixed for every candidate layer.

## D3: Only layer coverage changes from the approved population and execution contract

Retain approved 503 tickers/500 issuers and all 4,024 role-preserving pairs. Development is every fit/validation pair, 2,416 + 600 = 3,016, with no ticker subset, donor resampling, or outcome-based inclusion. Calibration 208 and evaluation 800 stay in the global table but are not development sampling sources. The counts above are exactly `3016 × 4 spans × panel size`, excluding gates and clean/donor calls.

Primary arms remain entity/evidence1/evidence2/instruction, **post-block/full selector**, unchanged original prompt compilation, exact-token mapping when token tuples match and explicitly recorded relative-rank transport otherwise. Existing alignment sensitivity and position selectors remain separate phases. Retain current transient capture/cache behavior, parent-specific precision/wrapper/schema/generation policy, write-once provenance, full parent-match checks, no-op gates and ITT failure accounting. No saved raw activations, residuals, gradients or KV caches. See [original pairing/alignment/selection contract](../rebuild-v1/localization-proposal.md) and [execution contract](../../dev/stance_localization_runner_spec.md) for unchanged requirements. Those frozen documents still describe V1's all-layer coverage and are not rewritten by this proposal.

**R3:** GPT's known full-parent replay drift persists. GPT remains blocked until strict full native continuation parent replay and no-op gates pass. Do not overwrite its baseline, substitute regex/payload-only matching, remove analysis/envelope tokens, or weaken a gate to make execution proceed.

## D4: Future selection is restricted to the frozen panel

For each family/contrast/span, rank only these candidate layers by fit-role company-first toward-source ITT among opposite-clean pairs, descending, ties ascending layer. Select top3 defined scores, or none if all are NA. Failures remain no-flip outcomes in the fixed denominator, directional ITTs and source counts remain visible, same-clean retention is separate, and zero eligible denominator is NA. Validation reports these fixed candidates without reranking or replacement. Later evaluation tests only frozen selected candidates, with Holm correction over the full evaluation contrast/span/candidate family, counting directions separately if formal tests are reported. Do not infer an unmeasured global peak or publish a new global/descriptive band.

Neuron discovery, DIM fitting and cone choices are **not conditioned on this panel**. Jobs27/41 are unaffected. Candidate replacement comparison does not prove the optimal additive steering layer or require agreement between operators.

## A1: Stop/archive handoff precedes any new runner

User authorized stopping only old localization jobs21/22/23/25/28/34/35 and preserving partial outputs with SHA manifests at `artifacts/maintenance-snapshots/localization-stopped-for-candidate-panel-v2.json`. That file is absent in this local checkout at panel implementation time. This phase neither operates the scheduler nor certifies remote stops/backups. Main must verify the manifest and stopped-job states before new dispatch. No unrelated job or service may be stopped.

New execution, once separately implemented and accepted, must use fresh `artifacts/<model-slug>/concept-cone-steering/runs/<run-id>/` roots and panel-bound plans, never resume an all-layer V1 root as V2. CPU panel acceptance is not GPU completeness or research eligibility.
