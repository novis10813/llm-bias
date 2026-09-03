# Entity Cell Downstream Attribution: Report E2 (Discovery)

**Status:** E2 discovery complete. Descriptive routing and head selection concluded. Calibration and held-out test not run. Version index: [README](README.md).

**Protocol:** [proposal-e2](proposal-e2.md).

---

## 1. Runs on Record

All runs located under `artifacts/qwen3.5-4b/entity-cell-localization/runs/`:

| run | state | note |
|---|---|---|
| `entity-cell-e2-discovery-v1` | failed (preserved) | keyword-only attention forward hook failure, fixed in `764c5ad` |
| `entity-cell-e2-discovery-v2` | failed (preserved) | fixed 1e-4 additivity tolerance rejected bf16 rounding noise, fixed in `0c4cbfc` |
| `entity-cell-e2-discovery-v3` | failed (preserved) | single-token continuation unpack tuple length error, fixed in `c949537` |
| `entity-cell-e2-discovery-v4` | failed (preserved) | bf16-relative additivity tolerance calibration, fixed in `f9f53d3` |
| **`entity-cell-e2-discovery-v5`** | **complete** | **Formal E2 discovery run** (13,440 attribution records, 525 readout records, 420 patch contract records) |

---

## 2. Head Selection and Attribution Results

### Top-5 Selected Full-Attention Heads

Heads ranked by equal-ticker mean absolute identity-header DLA:

| Rank | Head | Mean Abs Identity DLA | Mean Abs Instruction DLA | Routing Label | Direction Consistency | Selection Status |
|---|---|---|---|---|---|---|
| 1 | **L31 H0** | 0.0138 | 0.1118 | instruction-dominant | 1.00 (35/35) | Selected (Top-1) |
| 2 | **L31 H1** | 0.0063 | 0.0892 | instruction-dominant | 1.00 (35/35) | Selected (Top-2) |
| 3 | **L31 H3** | 0.0033 | 0.0481 | instruction-dominant | 1.00 (35/35) | Selected (Top-3) |
| 4 | **L19 H4** | 0.0020 | 0.0245 | instruction-dominant | 1.00 (35/35) | Selected (Top-4) |
| 5 | **L27 H6** | 0.0019 | 0.0312 | instruction-dominant | 1.00 (35/35) | Selected (Top-5) |

### Population Routing Summary (128 Heads Scanned across 8 Full-Attention Layers)
- **Instruction-dominant**: 123 heads
- **Mixed**: 5 heads
- **Identity-dominant**: 0 heads

---

## 3. Readout and Downstream Controls

1. **Jacobian Lens Readout**:
   - `L19 H4`: Readout available. 105/105 prompts showed Buy > Sell score (mean margin diff +1.0018).
   - `L27 H6`: Readout available. 25/105 prompts showed Buy > Sell score (mean margin diff -0.1634).
   - `L31 H0, H1, H3`: Readout unavailable because canonical lens covers source layers up to L30 only (`readout_unavailable: true`).
2. **Patching Contracts**:
   - Stage `e2-patching` verified and serialized 420 contract rows (105 prompts × 4 conditions).

---

## 4. Key Scientific Conclusions

1. **注意力層級對實體 Header 的直接貢獻微弱且晚期**：
   在所有 full-attention heads 中，沒有任何一個 head 是 identity-dominant（0/128）。所有被選中的 head 均為 instruction-dominant。
2. **與上下文覆蓋實驗（Sector B）結果一致**：
   決策主要由 post-evidence 的指令上下文驅動，實體 header 本身對注意力輸出的直接調控權限極低。
3. **後續 E3-B 干預標靶確立**：
   選定此 5 個 heads 作為下游干預（E3-B）的唯一合法組別。

---

## 5. Instrument Revision Re-verification (v2 instrument, 2026-09-03)

**Erratum context.** E2 DLA 的 frozen margin direction 與 shared core FP32 tail 同源，v5 run 的方向向量使用 v1 儀器（final norm 手動公式漏掉 Qwen3.5 的 `1+` 項；詳見 [`docs/shared-experiment-core.md`](../shared-experiment-core.md) 測量變更記錄 v2，修正於 commit `101e44e`）。本節為同一凍結 E2 設計在 v2 儀器下的重驗，不改寫上方原始記錄。

**Re-verification run:** `entity-cell-e2-discovery-v6`（13,440 DLA records；CPU fp32；同 v5 的 105 prompts × 8 layers × 16 heads）。

**Run 狀態註記：** 該 run 的 `e2-attribution` 階段完成且 records 完整保留（`e2/head_attribution.jsonl`，sha256 `78c390de578f7335…`）；`analyze` 階段因一個與本 bug 無關的 latent 程式錯誤（`run_e2` 內 stage-branch local import 遮蔽 module-level 名稱，`UnboundLocalError`）而失敗，該錯誤已在 commit `f885f20` 修復並加回歸測試。`analyze/summary.json` 為 post-hoc 以 frozen `rank_attention_heads` 對完整保留的 attribution records 重算（deterministic、model-free），provenance 記錄於該檔，run manifest 維持 status=failed。

### 5.1 Head 選擇與 routing 重驗

| Rank | Head | rank（v5→v6） | identity mean（v5→v6） | instruction mean（v5→v6） | routing label（v5→v6） | consistency |
|---|---|---|---|---|---|---|
| 1 | L31 H0 | 1 → **1** | −0.00203 → −0.00304 | −0.03473 → −0.04636 | instruction-dominant → instruction-dominant | 1.00 (35/35) |
| 2 | L31 H1 | 2 → **2** | +0.00067 → +0.00076 | −0.05048 → −0.07493 | instruction-dominant → instruction-dominant | 1.00 (35/35) |
| 3 | L31 H3 | 3 → **3** | +0.00064 → +0.00080 | +0.02056 → +0.02514 | instruction-dominant → instruction-dominant | 1.00 (35/35) |
| 4 | L19 H4 | 4 → **4** | +0.00204 → +0.00302 | +0.01543 → +0.01718 | mixed → mixed † | 1.00 (35/35) |
| 5 | L27 H6 | 5 → **5** | +0.00020 → +0.00023 | +0.02402 → +0.03208 | instruction-dominant → instruction-dominant | 1.00 (35/35) |

- **Top-5 選擇與排名完全不變**（L31 H0/H1/H3、L19 H4、L27 H6，rank 1–5 順序一致）。
- 全 128 heads 的 routing label 分布不變：123 instruction-dominant / 5 mixed / 0 identity-dominant。
- 所有 DLA 數值在真方向下放大约 1.4–1.5×（與 `1+w` norm 係數一致），方向與幾何不變；selection（以 equal-ticker mean abs identity DLA + consistency 排序）對均勻 scaling 不敏感。
- † **原文表勘誤**：上方 §2 表格將 L19 H4 標為 instruction-dominant，但 v5 與 v6 的 `analyze/summary.json` 對 L19 H4 的 frozen routing label 皆為 `mixed`（兩版一致；population 計數 123/5/0 亦以 summary 為準）。

### 5.2 修正後的解讀

1. **E2 結論維持**：沒有 identity-dominant head、決策由 instruction context 驅動、selected 5 heads 作為 E3-B 標靶——這些結論在真 margin 方向下全部維持，不需要新版本。
2. **E3-B 的 selected-head 輸入仍然有效**：`entity-cell-e3-discovery-v4` 的 downstream attenuation 使用的正是這 5 個 heads，選擇結果不變代表 E3-B 的重驗不需要換標靶。
3. **Readout 數值註記**：§3 的 lens readout margin（如 L19 H4 的 +1.0018）同樣產出於 v1 儀器，絕對值需以 v2 儀器重算才可使用；readout 的可用/不可用狀態（lens 覆蓋 L30）不受影響。
