# Discovery-v3 selects layers on 32 fit issuers and replicates them on all validation pairs

**D1:** The user approved shrinking discovery because the v2 candidate panel needed 72k–84k development cells per model. V3 keeps the v2 candidate panel, spans, pairs, parent policies, gates and ITT accounting. It changes three things: discovery uses 32 fit issuers, validation runs only the selected layers, and interventions run as same-prompt row batches. V1/V2 protocols, partial outputs and results stay frozen. Entry: `scripts/run_stance_localization_discovery_v3.py`.

## D2: Discovery cohort is fixed by issuer hash

Sort the 300 fit issuers by `(sha256_json({'seed': 20261007, 'role': 'fit', 'issuer_id': issuer}), issuer)` and keep the first 32. Every fit ticker of a chosen issuer is a target, so share classes stay together. The rule never reads decisions, sectors, margins or effects. Donors come from the unchanged role-preserving pair table (`localization_pairs_v1`), so a donor may lie outside the 32.

On the current compiled inputs the 32 issuers map to 32 tickers: AFL, AOS, AXON, BA, BBY, BKR, BMY, CAT, CINF, COP, CRL, EIX, ETN, EW, GRMN, HOLX, HON, HSIC, HUM, HWM, KIM, LOW, MKTX, NRG, OXY, PG, PHM, PWR, RCL, SHW, TROW, YUM. That gives 32 × 4 conditions × 2 families = 256 pairs.

| Model | Panel layers | Discovery cells |
|---|---|---:|
| qwen3.5-4b | 0, 3, 14, 15, 16, 31 | 6,144 |
| glm4-9b-0414 | 0, 4, 19, 20, 21, 39 | 6,144 |
| gemma4-12b-it | 0, 5, 23, 26, 27, 28, 47 | 7,168 |

## D3: Discovery selects at most two layers per site

A site is one family/contrast/span: 3 contrasts (`entity_context`, `evidence_order`, `polarity_content`) × 4 spans = 12 sites. For each site and panel layer, the score is the company-first toward-source ITT over opposite-clean fit pairs, pooling both source directions. Failed or invalid interventions count as no flip. Rank defined scores descending, ties to the lower layer, and keep the top 2. A site with no defined score is `untestable`. It gets no validation cells and no substitute layer. A site with one defined score keeps one layer.

## D4: Validation replicates fixed layers on all 600 validation pairs

Validation binds to one complete discovery run of the same parent, pair table, panel, issuers, model, template, grammar and generation policy. It recomputes the selection from the stored discovery scores and refuses a mismatch. Each of the 600 validation pairs runs every span at its site's selected layers, at most 4,800 cells. Validation reports these layers without reranking. Calibration and evaluation are not used.

## D5: Interventions run as same-prompt row batches with a control row

All cells of one pair share one target prompt, so rows need no padding and share positions. Per pair:

1. Batch-one donor generation with transient capture, then batch-one clean target. Both must equal the parent exactly, as in v2. Otherwise the pair halts.
2. One `generate` call per chunk of at most `max_rows` rows (default 32, recorded in the descriptor). Row 0 is an unpatched control and rows 1.. are cells. Each row has its own grammar matcher, stop state, deadline and replacement layer/span. Matchers fill one shared bitmask in parallel (`xgrammar.BatchGrammarMatcher`) and one kernel applies it to the open rows. Batching requires `use_cache=True`, because without a cache a finished row would keep receiving its replacement while other rows decode.
3. The control row's full output is compared with the batch-one clean target and stored as `control_match` in `records/batch_<sha>.json`. A mismatch is recorded, never repaired or retried.

**R1:** Batching changes GEMM shapes, so BF16 rows may diverge from batch-one at near-tie tokens. Batched execution is v3's declared protocol, and primary ITT uses all cells. The summary also reports flips restricted to calls whose control matched (`eligible_control_matched`, `flip_control_matched`) and the overall control match rate. Comparing batched cells against stored batch-one v2 cells is a separate proposed check and is not implemented.

**R2:** Timeout semantics change. Each row's 180 s deadline starts with the shared call and is checked when that row closes. A row failure closes only that row. An exception outside row callbacks fails every open row.

**R3:** 32 issuers give lower power than the full fit set. Discovery picks layers and validation estimates effects, so a noisy top-2 choice shows up as weak validation effects, not as inflated ones.

**R4:** GPT remains unsupported until strict merged Harmony replay passes. Qwen3.5 batched numerics are untested on GPU, so the control match rate must be read per model.

## Execution

```bash
uv run python scripts/run_stance_localization_discovery_v3.py --model <checkpoint> \
  --inputs <compiled-inputs> --parent <baseline-run> --model-slug glm4-9b-0414 \
  --phase discovery --output-dir artifacts/glm4-9b-0414/concept-cone-steering/runs/localization-discovery-v3-<job-id>

uv run python scripts/run_stance_localization_discovery_v3.py --model <checkpoint> \
  --inputs <compiled-inputs> --parent <baseline-run> --model-slug glm4-9b-0414 \
  --phase validation --discovery-run <discovery-run-dir> \
  --output-dir artifacts/glm4-9b-0414/concept-cone-steering/runs/localization-discovery-v3-validation-<job-id>
```

Runs are write-once and resumable. Resume revalidates every gate, cell, batch and halt record. A cell must reference a batch record that lists it at its row. No shard, ticker, layer, span or token-budget option exists.
