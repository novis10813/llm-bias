# Entity Cell Localization: V2 Discovery Report (E1 V2)

**Status:** V2 discovery complete — 1/35 trusted candidate entity cell (FTNT).
Calibration and held-out test not run; no confirmation freeze yet.

**Protocol:** [proposal-v2](proposal-v2.md). Version index: [README](README.md).
V1 result: [report-v1](report-v1.md).

## V2 runs on record

All under `artifacts/qwen3.5-4b/entity-cell-localization/runs/`.

| run | state | note |
|---|---|---|
| `entity-cell-prepare-discovery-v2` | complete | same inputs as V1 prepare plus 420 frame variants and the template-only control (`--localization-family v2-frames`) |
| `entity-cell-e1-smoke-v3` | failed (preserved) | CUDA OOM at startup (GPU full) |
| `entity-cell-e1-smoke-v4` | complete | one-ticker V2 smoke (CPU) |
| `entity-cell-e1-discovery-v2` | complete | V2 discovery, all four E1 stages (GPU 0, bf16, ~37 min) |

## Implementation

Branch `feat/entity-cell-e1v2` (merged as `f2f6cd3`): frozen twelve
natural-sentence frames (F0–F7 localization, H0–H3 held), frame-family
surface controls (`anonymous_name_frames`, `name_form_control_frames`)
rendered and re-tokenized at run time, the template-only control prompt with
a frozen template signature (top-5 by absolute z-score), the V1 header family
re-run for comparison only, the deterministic non-degenerate wrong-entity
rule with a `degraded_control` flag, and the four V2 gates
(held overlap, form-robust, template-robust, amnesia endpoint). V1 prepared
outputs remain byte-identical under the default `v1-header` family (verified
against `entity-cell-prepare-discovery-v1`).

## V2 discovery result (`entity-cell-e1-discovery-v2`)

- **1/35 tickers passed all four frozen gates: FTNT.** Its top-1 cell (L0,
  N104, stability score 363.4) is form-robust (absent from both frame-family
  surface-control top-5s), not in the template signature, retained in the
  held set (top-5 overlap 1, held rank 4 via its rank-2 cell L1 N575), and
  passed the amnesia endpoint gate on 2/3 eligible financial prompts at
  $\alpha=-3$ (target progress +0.067 / +0.265, each above both the
  wrong-entity and matched-random controls; the third prompt fails with
  negative target progress). No ticker had a degenerate wrong-entity control.
- **Caveat (cross-ticker selectivity):** FTNT's top-1 cell (L0, N104) is the
  shared top-1 of 17/35 tickers in the frame family; the other 16 share it
  but fail form-robust (their surface controls fire on it). The gates are
  per-ticker, so the pass is a boundary case: (L0, N104) is a candidate,
  not yet a demonstrated FTNT-specific identity cell. Selective
  intervention (E3) is the next test of entity specificity.
- **Structure:** V2 top-1 collision is (L0, N104)×17, (L0, N5101)×6,
  (L0, N4485)×4, then singletons — template neuron (L0, N4485) dropped from
  31/35 (V1) to 4/35; 25/35 tickers have zero frame∩header top-5 overlap,
  5 tickers overlap 3, 5 overlap 4. The template-only signature top-5 is
  led by (L0, N4485) (|z| 277), reproducing the V1 collision neuron as the
  template-reactive landmark; 10/35 tickers have a candidate in the
  signature. Only 1/35 tickers is form-robust; 9/35 pass the amnesia
  endpoint gate alone (AKAM, AMAT, CTSH, FFIV, FTNT, IBM, JKHY, MU, TXN),
  all but FTNT excluded by form-robust, template, or held overlap.

## Next steps and scientific closure

1. **科學定性（證偽假設 H1）**：
   大模型在 Qwen3.5-4B 規模下，並未展現出如 7B PopQA 論文所稱的普遍單神經元實體編碼特性（35 家僅 1 家通過，且該神經元為 17 家共享）。此結果**正式證偽了「可為整個產業族群建立單一 MLP 實體神經元字典」之假設**。
2. **研究轉向個案解剖**：
   FTNT (L0, N104) 不再被視為群體代表，而是轉入專屬的因果干預實驗（E3）作為單點個案研究（Case Study），直接檢驗該神經元究屬 FTNT 專屬特異性單元，亦或群體共享的通用語法/實體槽位。
3. **E1 階段結案**：
   E1 定位階段至此正式結案歸檔。未來若需推進群體實體定位，需另立 V3（引入跨 Ticker 選擇性硬門檻或多維表徵分析）。
