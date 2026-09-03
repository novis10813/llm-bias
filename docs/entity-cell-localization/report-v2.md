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

**Erratum context.** E1 V2 四道門檻中只有 amnesia 門檻是 margin-based，受 shared core FP32 tail 的 v1 儀器 bug 影響（final norm 手動公式漏掉 Qwen3.5 的 `1+` 項；詳見 [`docs/shared-experiment-core.md`](../shared-experiment-core.md) 測量變更記錄 v2，修正於 commit `101e44e`）。上方原始記錄不改寫；本節為 v2 儀器下的重驗。

**重驗方式**：全 35 家 amnesia 在 CPU 上不可行（當前機器負載下預估 ~8 天），因此：(a) `entity-cell-e1-discovery-v3`（CPU fp32，partial preserved，manifest status=failed 並註明中停原因）用於 activation-based 部分的自檢；(b) 針對性 amnesia 重驗（FTNT 完整 dose curve + v1 儀器下 endpoint-pass 的 8 家，frozen candidates 與 frozen gate 規則，CPU fp32）以 `scripts/entity_cell_amnesia_recheck.py` 執行，結果保存在該 run 目錄的 `amnesia_recheck_v2_instrument.json`（含 provenance）。

### 1. Amnesia 門檻重驗（v2 儀器，CPU fp32）

**FTNT（E3 個案）：門檻通過，2/3 prompts**（與 v1 儀器相同的 2/3 模式）：

| Prompt | v1 儀器 target $A_p(-3)$ | v2 儀器 target $A_p(-3)$ | v2 clean → 匿名 | 判定（兩版） |
|---|---|---|---|---|
| `6ba8dca` | −0.1049 | −0.0373 | +1.9470 → +0.6034 | fail（兩版皆負） |
| `6eee4cd` | +0.0672 | +0.1004 | +1.7098 → +0.4383 | pass（兩版皆高於兩對照） |
| `147b61f` | +0.2650 | +0.2281 | +1.5643 → +0.3928 | pass（兩版皆高於兩對照） |

**endpoint-gate 集合的變化**（v1 儀器下 9/35 單獨通過 amnesia endpoint 的 tickers，以同一 gate 規則重驗）：AKAM、AMAT、**FTNT**、MU、TXN 仍通過（5 家）；CTSH、FFIV、IBM、JKHY 不再通過（4 家）。此集合變化**不影響 trusted 集合**：這 8 家在官方 run 中本就因 form-robust / template / held-overlap 被排除，trusted 仍需四道門檻全過。

### 2. Localization 自檢與 form-robust 的精度 near-tie（重要附帶發現）

`entity-cell-e1-discovery-v3`（CPU fp32）的 localization 與 v2（GPU bf16）對比：top-1 在 32/35 不變（IBM/IT/NTAP 為 near-tie rank swap），FTNT top-1 (L0, N104) 穩定。但 **FTNT 的 form-robust 門檻是 bf16/fp32 精度 near-tie**：官方 bf16 run 中 `form_robust=true`（兩個 frame-family surface control top-5 皆不含候選），fp32 自檢 run 中 `name_form_control_frames` 出現 top-5 overlap 1（`form_robust=false`）。form-robust 是 activation-based gate，不受 norm 修復影響；此差異純粹是 bf16→fp32 的 rank 邊界敏感度。

### 3. 修正後的解讀

1. **FTNT 的 amnesia 資格在 v2 儀器下維持**：真決策下 clean/匿名皆為 Buy（gap 约 −1.17~−1.34），`(0,104)` 抑制在 2/3 prompts 上把 margin 拉向匿名基線且高於兩對照，與 v1 儀器相同的 2/3 模式。
2. **「1/35 trusted」結論在官方 bf16 run 語義下維持**，但 form-robust 成分被標記為精度 near-tie：若未來在 GPU bf16 下以 v2 儀器完整重跑四道門檻（留待 GPU 空檔），該 near-tie 會以官方精度重新落定。在此之前，FTNT 應視為「amnesia-verified、form-robust 為 boundary case」的候選——這與上方 §V2 discovery result 原有的 boundary case 註記一致。
3. **E3 V1 的個案地位不受影響**：E3 的 frozen 設計已围绕 FTNT (0,104) 執行，且 `entity-cell-e3-discovery-v4`（見 [report-e3-v1 §6](report-e3-v1.md)）在真決策下全數通過 frozen gates。
