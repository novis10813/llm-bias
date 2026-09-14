# Entity Cell Localization: Report V1 (Header Family Discovery)

**Status:** V1 discovery complete. Negative result: protocol template-dominated. 0/35 trusted candidates. Superseded by V2. Version index: [README](README.md).

**Protocol:** [proposal-v1](proposal-v1.md).

---

## 1. Runs on Record

All runs located under `artifacts/qwen3.5-4b/entity-cell-localization/runs/`:

| run | state | note |
|---|---|---|
| `entity-cell-prepare-discovery-v1` | complete | 35 tickers, 420 header variants, 399 baseline, 105 financial prompts |
| `entity-cell-e1-smoke-v1` | failed (preserved) | metadata registration bug, fixed in `80b9a0c` |
| `entity-cell-e1-smoke-v2` | complete | one-ticker V1 smoke |
| **`entity-cell-e1-discovery-v1`** | **complete** | **Formal E1 V1 discovery run** (all 4 stages: baseline, localization, amnesia, analyze) |

---

## 2. E1 V1 Discovery Results

- **神經元高度撞車（31/35 共享同一神經元）**：
  31/35 家 Technology tickers 共享完全相同的 top-1 神經元：**(L0, N4485)**；其餘 4 家共享第二顆神經元 **(L0, N5101)**。
- **門檻通過率：0/35**：
  沒有任何一家公司通過預設的失憶（amnesia）門檻。
- **機制剖析（模板主導）**：
  V1 的 12 個變體共享完全相同的三行 Header 前綴結構（`Stock Ticker: [X]` / `Stock Name: [Y]` / `--- Evidence ---`）。對 Header 模板反應強烈的神經元，在不同變體之間的激活方差趨近於 0，導致其穩定度分數 $S_{\ell j} = (\mathbb{E} z)^2 / (\operatorname{Std} z + \varepsilon)$ 在所有公司身上均名列第一。定位算法找到的是「三行 Header 模板檢測器」，而非個別公司的實體單元。
- **已知實作限制（已被 V2 修復）**：
  在 V1 程式碼中，定位階段的表面形式控制組重用了原 prompt 的 `input_ids`（未重新 tokenize），導致報告中的 5/5 overlap 屬無效對照。但「31/35 撞車」與「0/35 通過失憶門檻」的事實不受此影響，模板主導的結論確立。

---

## 3. 結案結論

V1 證明固定的三行 Header 前綴無法分離模板反應與實體表徵。本版本正式結案並歸檔，不進入 Calibration 或 Test。促使建立 [proposal-v2](proposal-v2.md)（自然句 Frames）。
