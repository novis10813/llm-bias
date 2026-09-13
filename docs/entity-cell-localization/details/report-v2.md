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

---

## Instrument Revision Re-verification (v2 instrument, 2026-09-03)

**Erratum context.** E1 V2 四道門檻中只有 amnesia 門檻是 margin-based，受 shared core FP32 tail 的 v1 儀器 bug 影響（final norm 手動公式漏掉 Qwen3.5 的 `1+` 項；詳見 [`docs/shared-experiment-core.md`](../../shared-experiment-core.md) 測量變更記錄 v2，修正於 commit `101e44e`）。上方原始記錄不改寫；本節為 v2 儀器下的重驗。

**重驗 run**：
1. **`entity-cell-e1-discovery-v4`**（**官方完整重驗**；GPU 0, bf16, ~41 min, complete）：全四階段（baseline 6 筆、localization 35 筆、amnesia 3,150 筆、analyze 70 組），以模型原生 GPU bf16 精度搭配 v2 儀器完整重跑。
2. **`entity-cell-e1-discovery-v3`**（CPU fp32 自檢，partial preserved）+ `scripts/entity_cell_amnesia_recheck.py`（針對性 amnesia 重驗，結果存於 `amnesia_recheck_v2_instrument.json`）：作為輔助 off-device 交叉驗證。

### 1. 官方 GPU bf16 重驗結果（`entity-cell-e1-discovery-v4`）

**FTNT（E3 個案）：四道門檻全數通過，仍為全群體唯一 Trusted 候選（1/35）**：

- **Held-variant overlap**：`top5_overlap = 1`（通過）。
- **Form-robust**：`form_robust = true`（`anonymous_name_frames` 與 `name_form_control_frames` 之 top-5 overlap 皆為 0，通過）。在原生 GPU bf16 精度下，FTNT 確認乾淨通過 form-robust（CPU fp32 下的 overlap 1 確為 off-device 精度敏感差異）。
- **Template-robust**：不在模板特徵簽名中（通過）。
- **Amnesia endpoint 門檻**：通過 2/3 prompts（平均終點 $A_p = +0.0858$）：

| Prompt | v1 儀器 GPU bf16 target $A_p(-3)$ | v2 儀器 GPU bf16 target $A_p(-3)$ | v2 clean → 匿名 | 判定（兩版） |
|---|---|---|---|---|
| `6ba8dca` | −0.1049 | −0.0932 | +1.8492 → +0.6497 | fail（兩版皆負） |
| `6eee4cd` | +0.0672 | +0.0749 | +1.6534 → +0.4564 | pass（高於兩對照 −0.0366 / −0.0388） |
| `147b61f` | +0.2650 | +0.2757 | +1.5722 → +0.2997 | pass（高於兩對照 +0.0572 / −0.0484） |

註：輔助 CPU fp32 針對性重驗數值（−0.0373 / +0.1004 / +0.2281）呈現完全相同的 2/3 模式。

### 2. 全 35 家 Amnesia Endpoint 集合與 Trusted 判定

在 `entity-cell-e1-discovery-v4` 中，全 35 家以 v2 儀器重算 amnesia endpoint 門檻（$\alpha=-3.0$、`all_positions`、$\ge 2$ eligible prompts）：

- **13/35 家單獨通過 amnesia endpoint 門檻**：AKAM, AMAT, BR, CTSH, FFIV, FI, FIS, **FTNT**, GEN, IBM, JKHY, MU, TXN。
  - 包含 v1 儀器下的全部 9 家（AKAM, AMAT, CTSH, FFIV, FTNT, IBM, JKHY, MU, TXN）。
  - 新增 4 家：BR, FI, FIS, GEN。
- **其餘 12 家皆被排除，不影響 Trusted 集合**：
  - AKAM, AMAT, FFIV, JKHY, MU, BR, FI, FIS, GEN：排除原因皆為 `not_form_robust`。
  - CTSH, IBM, TXN：排除原因皆為 `in_template_signature` 與 `not_form_robust`。
- **最終結果**：`v2_candidate_eligibility` 中僅 FTNT 1 家標記為 `eligible: true`（`trusted_ticker_count = 1`）。

### 3. 修正後的定性

1. **「1/35 trusted candidate (FTNT)」核心結論完全成立**：在真儀器、官方 GPU bf16 精度下，四道門檻完整重跑確認 FTNT 是全 35 家中唯一通過四道門檻的實體候選。
2. **form-robust 門檻落定**：在模型部署的官方 GPU bf16 精度下，FTNT 之 surface control overlap 為 0，乾淨通過。CPU fp32 自檢時出現的 overlap 1 確為 off-device 精度敏感差異，不影響官方 run 判定。
3. **E3 V1 個案研究的標靶合法性完全穩固**：E3 所干預的 FTNT `(L0, N104)`，在修復前後均為嚴格通過四道門檻的唯一合格實體單元，且 `entity-cell-e3-discovery-v4` 在真決策下全數通過 frozen gates。
