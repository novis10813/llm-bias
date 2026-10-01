# Recovered historical artifacts do not answer rebuild hypotheses

**Type:** read-only historical audit findings, not a new model run. Source run is `confirmation-v1-20260925-full-01`; original outputs are unchanged. Independent machine artifact verification checked JSON keys, generated text and selected source hashes. A separate deterministic audit compiler and full per-dose table remain pending.

## Gemma preserves decisions but not all reason text in self replacement

`artifacts/gemma4-12b-it/concept-cone-steering/runs/confirmation-v1-20260925-full-01/c2v3_gen/result.json`, SHA-256 `b07281ae56389e4536838bb9719346af8913d0ecfd89e21b430270f35835f990`: `summary.self_patch_identical=false`. Ten self-patch rows give 7/10 identical generated texts and 10/10 identical complete-object decisions; ALGN/BX/CHRW reason wording differs. Strict-decision equality is not valid-output equality: all ten are unparsed under the strict parser due to fences.

This does not establish full generated token identity: compact `realized_token_ids` are the answer-token readout pair, not complete continuation sequences. Do not infer why the text changed from small numerical/margin differences. New no-op tests must compare repeated baseline, zero hook and native self replacement in the same checkpoint/cache/kernel/precision configuration and fail eligibility when the frozen identity gate fails.

## GPT-OSS's historical band is a two-element set

`artifacts/gpt-oss-20b/concept-cone-steering/runs/confirmation-v1-20260925-full-01/c2v3/result.json`, SHA-256 `c149cf86ca21c5281b018ba040b171fb886a8a47f6714ee01407690535cec295`: `summary.peak=8`, `summary.band=[1,8]`. Band rule is teacher-forced steer-suffix T≥0.7×peak, not a continuous interval. L4 is below threshold; confirmation injection L14 is outside the set. The old layer-check L4-inside-band rationale is invalid. Preserve the old run/proposal and record this erratum; new candidate selection is generated-outcome development-only.

## Historical DIM denominators belong to its own parser/protocol

All four model `dim` arms planned 101 evaluation tickers, seed20260923. Complete-object α0 classes: Qwen buy0/sell101; GLM buy101/sell0; Gemma buy58/sell43; GPT-OSS buy44/sell57. Sell→buy denominators are respectively101/0/43/57; buy→sell0/101/58/44. Zero source class is NA, not a zero-percent effect. Invalid steered rows remain nonflips in historical ITT; paired parsed denominators are separate diagnostics. These are not schema-constrained rebuild outputs.

Raw C2 overlap inputs are missing locally: for each model `balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/{sweep/records.jsonl,analyze/summary.json}`. Overlap exclusion remains input-blocked, not recomputed or disproved.

## The recovered manuscript differs from TODO's audited draft

Untracked source manuscript SHA-256 `e4d7d5b859588695fc8bf17dbefe4a19d5e17585e51da89958292fd5a997416a` is a 365-line earlier single-Qwen/200-cohort pilot draft. It has no four-model appendix lines704/716/725/672 referenced by TODO. Do not apply those line-number corrections to an unrelated version or restore a deleted manuscript from tags without authorization.

Recovered bibliography SHA-256 `e8768318493b968a80548552ca995ee0d21f83fb3abdbc1008b65f62fb1f638a` names `Paleka, Dhruv`; the Arditi source author is Daniel Paleka. A new bibliography should correct the primary-source name. Existing source copy remains unchanged.

Every future audit table will record its source file hashes, parser, denominator, dose, historical protocol and analysis timestamp. Margin C8, posthoc generated monotonicity and independently trained rebuild cone are different tests. Preserve genuine historical effects and failed training as historical evidence with their scope.
