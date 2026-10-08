# LC3 binds a full pair table before any intervention effect is observed

**Goal:** Deterministically construct the two localization pair families for all approved parent keys, preserving roles and issuer separation. Donors are selected without decision labels. Pair enumeration is one slice; it does not render prompts, choose layers, launch GPU runs or define eligibility.

## 1. Verified inputs and explicit new API

【已驗證: core/stance_baseline_parent.py:CompletedBaseline】 has plan, rows, parent_sha256 and immutable result lookup. Its constructor is trusted, not factory-attested. 【已驗證: core/stance_baseline_inputs.py:BaselineInputs】 roles['assignments'] maps tickers to issuer-disjoint roles; members supply issuer IDs. 【已驗證: core/experiment_contract.py:RowKey,ExecutionRow,GenerationOutcome】 baseline keys and primary_valid status already exist. 【新設計】 Pairing policy is in `docs/concept-cone-steering/rebuild-v1/localization-proposal.md`: same-role next-issuer ring and fixed reciprocal evidence contrasts. This design is data-informed, not untouched preregistration.

Create only `llm_bias/core/stance_localization_pairs.py`, `tests/test_stance_localization_pairs.py`. Do not modify LC0/LC1/LC2, input pins, historical code/results or dependencies.

```python
build_localization_pairs(inputs: BaselineInputs, parent: CompletedBaseline) -> LocalizationPairTable
```

## 2. All4024 pairs remain planned even without a flip denominator

**REQ-LC3-1 【新設計】** Validate the full approved parent plan once with build_baseline_plan(inputs,parent.plan.identity) canonical equality; require exact2012 parent rows/keys/issuer-bound outcomes, all roles present, parent hash64hex, completed diagnostic summary correct executed/planned/missing/complete/gates/researchfalse. Do not rebuild per row or authenticate a direct parent constructor from a hash alone. The consumer uses parent.rows primary labels, not parsing or reloading directory data; full result integrity remains LC0 loader responsibility. Public constructor misuse that changes plan, missing/duplicate rows or row issuer/key must reject, not silently shrink the table.

**REQ-LC3-2 【新設計】** For each role independently group member tickers by issuer. Sort issuer IDs by `(sha256_json({'seed':20261002,'role':role,'issuer_id':issuer}), issuer)`; require at least two issuer groups. Next issuer cyclically supplies donor; choose donor issuer's lexicographically smallest ticker. For every target baseline row, construct cross_company donor key with same condition/trial, donor ticker. Target issuer must differ, role same. No decision or sector influences donor selection. All target tickers retained, including share classes, and all four conditions retained.

**REQ-LC3-3 【新設計】** For each target baseline row construct cross_evidence donor with same ticker/trial and reciprocal condition map {'++':'--','--':'++','+-':'-+','-+':'+-'}. Contrast names: cross_company => entity_context; cross_evidence ++/-- => polarity_content; cross_evidence +-/ -+ => evidence_order. An order contrast swaps exact approved P1/N1 items; do not substitute arbitrary trials. Full fixed table4024 rows, role counts fit2416/validation600/calibration208/evaluation800. Execution phase selectors later are fixed role/arm partitions of this global table, not arbitrary research target lists.

**REQ-LC3-4 【新設計】** Frozen LocalizationPair fields: target_key: RowKey, donor_key: RowKey, role: str, family: str, contrast: str, clean_relation: str, pair_sha256: str. clean_relation exactly opposite/same/invalid_parent; computed from both valid parent outcomes without changing donor. If either parent outcome not primary_valid => invalid_parent; otherwise equal legal decisions=>same elseopposite. Never omit invalid/same rows. Constructor validates accepted enums, proper baseline keys/dose0, family condition/ticker/trial relationships and pair hash; input-specific issuer/roles validated by builder. Exact to_dict exports keys as dicts, all seven fields. Pair hash sha256_json(other six-field export). Hash includes relation as observed provenance but relation never selects donor.

**REQ-LC3-5 【新設計】** Frozen LocalizationPairTable fields parent_sha256: str, inputs_manifest_sha256: str, policy_sha256: str, pairs: tuple[LocalizationPair,...], table_sha256: str. policy hash covers literal closed record {'kind':'localization_pairs_v1','seed':20261002,'cross_company':'role_issuer_hash_ring_next_lexicographic_ticker','cross_evidence':reciprocal_map,'contrast_names':{'cross_company':'entity_context','cross_evidence':{'++':'polarity_content','--':'polarity_content','+-':'evidence_order','-+':'evidence_order'}}}. table hash sha256_json(export minus table_sha256); export includes complete pair list. Sort pairs canonically by (target_key, family, donor_key) with genuine RowKey order; pair rows unique by (target_key,family). Public constructor validates hashes and structural sorted/unique tuples, not full-population authentication. Builder validates counts/eachrole/nonmissingdonorkey. to_dict defensive. No actualprompt IDs/tensors/reasons needed. parent_sha256 and table_sha256 change if the parent changes even when donor choice remains identical; policy_sha256 is parent-independent by design.

## 3. Synthetic tests cannot become a smaller public cohort

Use pinned full inputs and2012 direct ExecutionRows with synthetic buy/sell/failure outcomes. Build a trusted synthetic CompletedBaseline using consistent full plan and immutable fields (or reuse LC0 fixture helper after reading it); avoid2012 writes/parent loads just for pairing. One optional integration test loads actualGLM via LC0 and derives4024 without using evaluation labels for site selection; skip this test only when the ignored GLM run directory is absent. Main/independent acceptance on this working tree must run that actual-parent check (it is not skipped here). Test issuer rings independent of parent row order/label flips, shareclass role isolation, evidence reciprocal involution, sorted/hash reproducibility and mutation defenses, zero-opposite still4024, primary-invalid retention, constructor malformed/duplicate/foreignissuer/missingpairplan/hash rejection. Independently calculate donor choices for a few known role issuers; do not mirror the implementation as sole oracle. Assert plan builder called once. No public reduced-cohort option.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -q tests/test_stance_localization_pairs.py tests/test_stance_baseline_parent.py tests/test_stance_baseline_plan.py
uv lock --check
git diff --check
```

Independent read-only preflight then one implementation dispatch then main/independent acceptance. Full regression runs separately. Plan/store/runner and scientific site-ranking/no-op/resource contracts remain separate unfinished deliverables.
