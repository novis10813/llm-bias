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
