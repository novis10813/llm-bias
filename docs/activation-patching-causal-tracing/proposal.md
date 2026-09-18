# Activation Patching Causal Tracing: Proposal

This experiment uses hierarchical residual-stream resample patching to locate
(layer, prompt-span) combinations that carry sufficient state to change the
fixed Buy/Sell continuation margin. It patches model-produced states from one
valence condition into the opposite condition; it does not fit or inject a
Buy/Sell direction.

**Protocol status:** Draft 1 frozen and implemented; results are recorded in [the experiment report](report.md)
**Model for first run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)
**CLIs:** `jspace-intervention run-activation-patching` and `jspace-intervention analyze-activation-patching-confirmation`
**Package:** `llm_bias/jspace_intervention/activation_patching.py`

## Motivation

Prior J-space intervention experiments established:

1. An outcome axis $d_l$ exists with clear Buy/Sell vocabulary semantics
   (V2 direction decode).
2. Steering along $d_l$ reliably flips decisions (V2 Buy-steering 9/9,
   p = 0.003).
3. The steering effect is **not position-specific**: injecting $d_l$ at the
   final answer position produces the same 100 % flip rate as evidence
   positions (V2 position specificity null result).
4. Sector representation and outcome axis are nearly orthogonal (82°–92°,
   outcome direction geometry).

These results characterize a pre-chosen direction. What remains unknown is
**where the model's own forward computation builds up the representation
content that distinguishes a Buy outcome from a Sell outcome** — without
imposing an externally fitted direction. Activation patching answers this
by using the model's own natural activation differences between two prompts
that produce opposite decisions.

## Method

### Input: Valence Pairs

Each pair is produced by `build_valence_pair()` from one raw trial row:

- **Source prompt (positive):** header + buy_qual + buy_quant + JSON instruction
- **Target prompt (negative):** header + sell_qual + sell_quant + JSON instruction

The header and JSON instruction text match across conditions. Only the evidence
body differs. Because decoder states depend only on prior tokens, source and
target header states should match exactly. Instruction states can differ even
when the instruction text matches because they occur after the evidence.

### Patching Protocol

For each valence pair and patching condition `(layer_set, position_set)`:

1. Run the source prompt forward → cache residual stream
   $h^\text{src}_{l,p}$ at every (layer $l$, position $p$) within the
   requested `layer_set` (in GPU memory; never persisted).
2. Run the target prompt forward with hooks: at each $(l, p)$ in the
   patching condition, replace the target residual with the cached source
   residual $h^\text{src}_{l,p}$.
3. Measure the Buy/Sell margin $M = \log P(\text{buy}) - \log P(\text{sell})$
   under the patched forward pass, using FP32 tail-logit scoring.
4. Record $\Delta M = M_\text{patched} - M_\text{target\_clean}$ and whether
   a discrete decision flip occurred.

Bidirectional: repeat with source ↔ target swapped.

### Token Alignment

The header and JSON instruction share identical text and therefore identical
token sequences. Evidence bodies differ in content and possibly token count.
Position mapping for evidence spans:

- **Header positions**: 1:1 aligned (same tokens).
- **Evidence span positions**: Target-complete nearest-normalized mapping.
  Every target evidence token receives the source token at the nearest
  fractional location. If source and target spans contain the same number of
  tokens, mapping reduces to one-to-one alignment by offset.
- **JSON instruction positions**: Aligned by offset from the instruction start.
  Their activation states need not match because the preceding evidence differs.
- **Final position**: Always position `seq_len - 1` of the target prompt.

Character-to-token mapping uses existing `token_span()` from
`llm_bias.core.prompt_input.encoding`.

## Experiment Phases

### Phase 0: Baseline Confirmation

Verify that each valence pair produces naturally opposed decisions:

- Source (positive) → $M > 0$ (Buy)
- Target (negative) → $M < 0$ (Sell)

Only pairs with **both conditions cleanly decided** (margin signs match
expected valence) are included. This is a gate, not an assumption.

**Output per pair:** `ticker`, `source_margin`, `target_margin`,
`clean_decision_match` (bool).

### Phase 1: Layer-Band × All-Position Patching

Patch **all** token positions within a layer band simultaneously. This
establishes the upper bound of patching effect at each layer granularity.

**Sweep design:**

| Condition          | Layers patched                                       |
| ------------------ | ---------------------------------------------------- |
| Single-layer sweep | Each layer individually: L0, L1, ..., L30            |
| Cumulative band    | L14–L16, L14–L18, L14–L20, L14–L22, L14–L24, L14–L26 |
| Full band          | L14–L26 (known margin-sensitive band)                |
| Late band          | L24–L30                                              |
| Early band         | L0–L13                                               |

For each condition: patch all positions in those layers → measure $\Delta M$.

**Output per (pair, condition):** `layer_condition`, `layers`,
`patching_direction` (`positive→negative` or `negative→positive`),
`delta_margin`, `flip` (bool), `patched_margin`.

### Phase 2: Span-Level Patching

Within the effective layer range(s) identified in Phase 1, patch only
specific semantic spans.

**Prompt spans** (character boundaries from `render_valence_prompt_with_spans`
mapped to token positions):

| Span ID               | Content                                                                                         | Same across conditions?                |
| --------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------- |
| `header`              | Ticker, name, header text                                                                       | Yes (identical tokens)                 |
| `evidence_qual`       | Evidence item 1 (qualitative)                                                                   | No                                     |
| `evidence_quant`      | Evidence item 2 (quantitative)                                                                  | No                                     |
| `instruction_context` | Post-evidence instruction and assistant-prefix positions, excluding the final decision position | Same text; evidence-conditioned states |
| `instruction`         | `instruction_context` plus final decision position                                              | Same text; evidence-conditioned states |
| `all_evidence`        | `evidence_qual` ∪ `evidence_quant`                                                              | No                                     |
| `final_position`      | Last token only                                                                                 | N/A (positive control)                 |

**Patching combinations** (within effective layer band):

| Condition             | Positions patched                                             | Expected role            |
| --------------------- | ------------------------------------------------------------- | ------------------------ |
| `all_evidence`        | All evidence token positions                                  | Primary test             |
| `evidence_qual`       | Qualitative evidence only                                     | Component test           |
| `evidence_quant`      | Quantitative evidence only                                    | Component test           |
| `header`              | Header positions only                                         | Negative control         |
| `instruction_context` | Post-evidence positions excluding the final decision position | Primary aggregation test |
| `instruction`         | `instruction_context` plus final position                     | Aggregation upper bound  |
| `final_position`      | Final token only                                              | V2 comparison control    |

**Key contrasts:**

- `all_evidence` vs `final_position`: Does evidence carry decision info
  beyond what is already at the answer position?
- `all_evidence` vs `header`: Is the effect evidence-specific?
- `evidence_qual` vs `evidence_quant`: Which evidence component carries more?
- `header` ≈ 0: Confirms identical-header patching is a null intervention.

### Phase 3: Layer × Span Interaction (Precision Scan)

Cross Phase 1 (effective layers) with Phase 2 (effective spans):

For each effective single layer $l$ and each effective span $s$:

- Patch only $(l, s)$ → measure $\Delta M$

Output: A (layer × span) causal matrix.

## Statistical Design

### Sample

- **Discovery:** 35 Technology discovery tickers × `trials_per_ticker`
  valence pairs (using existing `select_valence_trials` with seed=0).
- **Held-out confirmation:** If Phase 1–2 identify effective (layer, span)
  combinations on discovery tickers, confirm on calibration + test split
  tickers.

### Metrics Per Condition

| Metric                 | Definition                                                              |
| ---------------------- | ----------------------------------------------------------------------- |
| `delta_margin`         | $M_\text{patched} - M_\text{target\_clean}$                             |
| `flip`                 | $\text{sign}(M_\text{patched}) \ne \text{sign}(M_\text{target\_clean})$ |
| `flip_rate`            | Fraction of pairs where flip = true                                     |
| `mean_delta_margin`    | Mean $\Delta M$ across tickers                                          |
| `bidirectional_effect` | $\frac{1}{2}[\Delta M_{+→-} - \Delta M_{-→+}]$                          |

### Controls and Comparisons

- **Negative control:** `header`-only patching (identical tokens → expected
  $\Delta M \approx 0$).
- **Positive control:** `final_position`-only patching (known V2 comparison).
  Draft 1 implements the header and final-position controls. A random unrelated
  source prompt is reserved for the confirmatory version because it changes the
  control family and requires a frozen ticker-matching rule.

### Discovery and Confirmatory Analysis

Draft 1 writes per-record effects plus equal-ticker mean $\Delta M$ and flip
rates. Phase 1 and Phase 2 are discovery scans. Before a held-out run, freeze
one or more effective layer/span combinations and add the exact paired test,
confidence interval, and Holm family to a confirmatory protocol version.
Discovery results do not support a confirmatory significance claim.

## Artifact Schema

### Output Directory

```
artifacts/qwen3.5-4b/jspace-causal-tracing/runs/<run-id>/
├── manifest.json
├── prepare/
│   └── metadata.json
├── forward/
│   ├── phase0_baseline.jsonl
│   └── <phase>_records.jsonl
└── analyze/
    └── summary.json
```

### Record Fields (JSONL)

Each line contains:

```json
{
  "record_id": "record_<24hex>",
  "ticker": "AAPL",
  "source_trial_key": "...",
  "phase": "phase1",
  "patching_direction": "positive_to_negative",
  "layer_condition": "single_L18",
  "layers": [18],
  "span_condition": "all_positions",
  "positions_patched": [12, 13, 14, ...],
  "position_count": 45,
  "source_clean_margin": 1.234,
  "target_clean_margin": -2.345,
  "patched_margin": 0.567,
  "delta_margin": 2.912,
  "flip": true,
  "score": {"positive": {...}, "negative": {...}},
  "prompt_template": "canonical-valence-v1",
  "evidence_item_hashes": {...}
}
```

No raw activations, residuals, hidden states, gradients, or KV caches are
persisted. Only compact per-condition scalar outputs with provenance.

## Memory and Compute Budget

### Per-Pair Overhead

- **Source forward pass:** One standard forward pass. Residual cache for
  `n_layers` layers × `seq_len` positions × `d_model=2560` dimensions
  depends on the model activation dtype. For BF16, 13 layers × 200 tokens ×
  2560 dimensions use about 13 MiB; 31 layers use about 30 MiB, excluding
  framework overhead.
- **Patched forward pass:** One forward pass per patching condition (hooks
  read from cached source residuals; no additional storage).

### Phase 1 Compute

Per eligible pair, the default scan runs 31 single-layer conditions plus ten
band conditions in both directions. With 105 pairs this yields about 8,600
patched forwards, plus clean and source-cache forwards. Runtime must be measured
from the first smoke run; the protocol does not assume a fixed per-forward time.

### Phase 2 Compute

With one frozen layer band, six span conditions, 105 pairs, and two directions,
Phase 2 yields at most 1,260 patched forwards. The Phase 0 gate can reduce this
count.

### Phase 3 Compute

Phase 3 accepts explicit layer and span conditions selected from discovery.
Record the selected grid before execution and report its actual forward count.

## Implementation Notes

### Core Function

```python
def run_activation_patching_record(
    *,
    model,
    tokenizer,
    source_prompt: str,
    target_prompt: str,
    source_spans: dict[str, list[int]],  # char spans
    target_spans: dict[str, list[int]],  # char spans
    layers: Sequence[int],
    span_condition: str,  # "all_positions" | "all_evidence" | "evidence_qual" | ...
    positive_candidate: str = " Buy",
    negative_candidate: str = " Sell",
    device: Any,
) -> dict:
```

### Residual Caching

Uses `record_residuals()` from `llm_bias.core.inference.forward` to capture
full-sequence residuals `[1, seq_len, d_model]` at requested layers. These
remain in GPU memory for the patching forward pass, then are discarded.

### Patching Hooks

Uses `residual_interventions()` from `llm_bias.core.inference.interventions`.
Each hook replaces `tensor[:, target_positions, :]` with the corresponding
`source_residuals[layer][:, source_positions, :]` (after position alignment).

### Scoring

The pipeline applies the tokenizer chat template, appends the frozen decision
prefix `{"decision": "`, and scores the one-token continuations `buy` and
`sell` with `score_single_token_margin_fp32()`. This matches the fixed-choice
V1/V2 margin semantics. `instruction_context` excludes the scoring prompt's
last position, so it tests post-evidence state transfer without directly
replacing the decision-position state.

### Command Examples

Phase 0 smoke on one prepared pair:

```bash
uv run jspace-intervention run-activation-patching \
  --pairs artifacts/qwen3.5-4b/jspace-valence-readout/runs/valence-technology-evidence-balanced-20260827T043734Z/prepare/valence_pairs.jsonl \
  --model .cache/models/qwen3.5-4b \
  --run-id causal-trace-phase0-smoke \
  --phase phase0 \
  --max-records 1
```

Phase 1 discovery uses the default single-layer and band sweep:

```bash
uv run jspace-intervention run-activation-patching \
  --pairs artifacts/qwen3.5-4b/jspace-valence-readout/runs/valence-technology-evidence-balanced-20260827T043734Z/prepare/valence_pairs.jsonl \
  --model .cache/models/qwen3.5-4b \
  --run-id causal-trace-phase1-discovery \
  --phase phase1
```

Phase 2 requires the layer band selected from Phase 1. Example syntax:

```bash
uv run jspace-intervention run-activation-patching \
  --pairs artifacts/qwen3.5-4b/jspace-valence-readout/runs/valence-technology-evidence-balanced-20260827T043734Z/prepare/valence_pairs.jsonl \
  --model .cache/models/qwen3.5-4b \
  --run-id causal-trace-phase2-discovery \
  --phase phase2 \
  --layer-condition selected_L14-L26=14-26
```

## Success Criteria

This experiment does **not** have a single pass/fail gate. Instead, it
produces a descriptive causal map. The key questions and their interpretive
outcomes:

| Observation                                                                                               | Interpretation                                                                                                 |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Evidence-span patching at mid-layers flips decisions; final-position patching at the same layers does not | Evidence positions carry decision-relevant info that propagates forward — **evidence-position causal pathway** |
| Both evidence-span and final-position patching flip decisions at the same layers                          | Decision info is available at multiple positions by mid-layers — consistent with V2 position non-specificity   |
| Only late-layer (L24+) patching is effective regardless of span                                           | Decision is formed very late; evidence influence is indirect                                                   |
| No single-layer patching flips decisions but multi-layer bands do                                         | Decision info accumulates gradually across layers                                                              |
| `evidence_qual` is much stronger than `evidence_quant` (or vice versa)                                    | One evidence modality dominates the decision pathway                                                           |

## Relation to Prior Experiments

- **V2 outcome direction flip:** V2 used a fitted outcome gradient $d_l$
  and showed position non-specificity. This experiment uses the model's own
  natural activation differences (no fitted direction) and can reveal whether
  position specificity holds for natural patching even though it failed for
  direction-based steering.
- **Sector intervention:** Sector prototypes were orthogonal to the outcome
  axis. Activation patching bypasses the sector/outcome decomposition entirely
  and asks: does the full activation at evidence positions matter?
- **V1 token screen:** Vocabulary-nominated directions were too weak. Activation
  patching uses high-dimensional full-residual replacement, not rank-1
  direction steering.

## Frozen Held-Out Confirmation Design

Discovery fixes a 3 × 3 layer/span matrix:

| Layer | Discovery-selected role                       |
| ----: | --------------------------------------------- |
|    L6 | Early evidence-state site                     |
|   L16 | Middle post-evidence instruction-context site |
|   L30 | Late final-position site                      |

Each layer is crossed with `all_evidence`, `instruction_context`, and
`final_position`, in both patching directions. Calibration and test use newly
prepared Technology valence pairs from the existing split manifest; no prompt
or outcome crosses splits.

Primary per-pair metric remains normalized transfer $T$. For each ticker,
trials are averaged within direction, directions are averaged, and tickers
receive equal weight. The three primary row-dominance contrasts are:

$$
C_6 = T(\mathrm{L6, evidence}) - \frac{T(\mathrm{L6, context}) + T(\mathrm{L6, final})}{2},
$$

$$
C_{16} = T(\mathrm{L16, context}) - \frac{T(\mathrm{L16, evidence}) + T(\mathrm{L16, final})}{2},
$$

$$
C_{30} = T(\mathrm{L30, final}) - \frac{T(\mathrm{L30, evidence}) + T(\mathrm{L30, context})}{2}.
$$

Calibration proceeds only if at least nine tickers and 24 pairs pass the clean
outcome gate. Test success requires all of the following frozen conditions:

1. At least nine test tickers and 24 pairs pass the clean-outcome gate.
2. Equal-ticker mean diagonal transfer is at least `0.75` at L6 evidence,
   `0.40` at L16 instruction context, and `0.75` at L30 final position.
3. Each row-dominance contrast is greater than `0.20`.
4. Each contrast has a ticker-bootstrap 95% CI lower bound above zero.
5. One-sided exact ticker sign-flip tests for the three contrasts remain below
   `0.05` after Holm correction.
6. Both patching directions have positive mean normalized transfer for each of
   the three diagonal conditions.

Calibration checks whether the frozen matrix remains viable; it does not alter
layers, spans, thresholds, contrasts, or the test gate. If calibration fails,
the Draft 1 test is not run. The frozen machine-readable config is:

`artifacts/qwen3.5-4b/jspace-causal-tracing/configs/position-transfer-confirmation-v1.json`

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**

- [jspace-token-experiments](../jspace-token-experiments/report.md)（研究承接）：V2 未建立位置特異性，改用模型自然狀態差定位決策充分性。
- [baseline-trial](../baseline-trial/proposal.md)（資料／產物依賴）：沿用 trial rows 的正負證據配對；residual patch 本身不要求 lens。

**後續**

- [sector-context-followup](../sector-context-followup/proposal.md)（研究承接）：依 evidence→instruction context→final 的層級轉移，設計 A/B/C 延伸。
- [balanced-evidence-gap](../balanced-evidence-gap/report.md)（研究承接）：明確證據下 header 效應弱，改在多空對稱條件確認 entity 影響。
- [entity-to-dial](../entity-to-dial/report.md)（方法參考）：沿用 bidirectional residual patch、固定答案 margin 與 self-source no-op 契約。

最終／最新結果見本研究的 [report](report.md)。
