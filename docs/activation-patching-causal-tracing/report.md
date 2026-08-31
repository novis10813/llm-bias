# Activation Patching Causal Tracing: 實驗報告

**Formal Held-Out Confirmation Verdict:** `success=true`

實驗操作契約、干預定義、殘差抽樣協定與凍結之驗證門檻見 [實驗提案與操作契約](proposal.md)。

---

## 執行摘要

本實驗採用層級式殘差流抽樣補丁（Hierarchical Residual-Stream Resample Patching），在 Qwen3.5-4B 模型上系統性定位決定 Buy/Sell 投資決策的內部表徵位置（Layers × Semantic Spans）。

在未見過的 Held-Out 測試集（11 個獨立 Technology Tickers、33 對提示詞）上，實驗正式確認了內部因果充分性（Causal Sufficiency）隨模型計算深度的動態轉移：

1. **早期層（L6）**：決策充分性完全集中於財務證據區塊（`all_evidence` 正規化轉移度 **0.9756**），下游 Context 與 Final Position 均為完全的 No-op。
2. **中期層（L16）**：決策充分性轉移至證據之後的指令前綴區塊（`instruction_context` 正規化轉移度 **0.6314**），即使排除最後一個輸出 Token，仍足以顯著轉移決策 Margin。
3. **晚期層（L30）**：決策充分性最終收斂於最後決策位置（`final_position` 正規化轉移度 **0.9136**）。

在預先凍結的 3×3 對角主導檢定中，所有 11 個測試實體均展現 100% 對角優勢，精確排列檢定（Exact Permutation Test）之 Holm 調整後顯著性均達 **$p = 0.001465$**，正式宣告 Confirmation 成功。

---

## 1. 診斷性 Smoke 實驗

Smoke 實驗使用單一 Analog Devices (`ADI`) 探索提示對，驗證程式邏輯、Mapping 正確性並確立干預效果的上界，不作為母體統計推論依據。

### Phase 0: 基準決策驗證
- **Run ID**: `artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-phase0-smoke-20260828`
- **結果**: Positive Prompt Margin 為 `+5.389980`（Buy）；Negative Prompt Margin 為 `-5.937601`（Sell），順利通過 Clean Decision 門檻。

### Phase 1: 全位置補丁上界診斷
- **Run ID**: `artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-phase1-minimal-smoke-20260828`
- **干預**: 同時替換 L14–L26 區間內的所有 Token 位置殘差。

| 干預方向 | 目標 Clean Margin | 補丁後 Margin | Margin 變化量 ($\Delta M$) | 離散決策翻轉 (Flip) |
|---|---:|---:|---:|:---:|
| Positive $\to$ Negative | -5.937601 | +5.415966 | +11.353567 | 是 |
| Negative $\to$ Positive | +5.389980 | -5.960407 | -11.350387 | 是 |

補丁後的 Margin 幾乎完全重現來源條件的數值，證實 L14–L26 全位置干預能提供完整的狀態轉移上界。

### Phase 2: 語義區間（Semantic Spans）診斷
- **Run ID**: `artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-phase2-span-smoke-20260828`

| 干預方向 | 語義區間 (Span) | Margin 變化量 ($\Delta M$) | 離散翻轉 (Flip) |
|---|---|---:|:---:|
| Positive $\to$ Negative | `all_evidence` | +4.651005 | 否 |
| Positive $\to$ Negative | `evidence_qual` | +1.596579 | 否 |
| Positive $\to$ Negative | `evidence_quant` | +1.325550 | 否 |
| Positive $\to$ Negative | `header` | 0.000000 | 否 |
| Positive $\to$ Negative | `instruction` (含輸出位) | +11.185770 | 是 |
| Positive $\to$ Negative | `final_position` | +6.619417 | 是 |
| Negative $\to$ Positive | `all_evidence` | -7.118902 | 是 |
| Negative $\to$ Positive | `evidence_qual` | -2.088142 | 否 |
| Negative $\to$ Positive | `evidence_quant` | -3.890133 | 否 |
| Negative $\to$ Positive | `header` | 0.000000 | 否 |
| Negative $\to$ Positive | `instruction` (含輸出位) | -11.067641 | 是 |
| Negative $\to$ Positive | `final_position` | -7.225910 | 是 |

**診斷結論**: Header 區間補丁呈現精確的 No-op（$\Delta M = 0.000000$），符合因果前綴一致性；單獨抽換證據區塊具備顯著因果效應但未對稱翻轉；後續指令區塊與輸出位置則展現強烈的因果翻轉能力。

---

## 2. Phase 1 Discovery: 全位置跨層掃描與上界確認

- **Run ID**: `artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-phase1-discovery-20260830`
- **規模**: 35 個 Technology Tickers、105 對提示詞；共 8,320 筆補丁記錄。
- **門檻篩選**: 104 對通過 Phase 0 Clean Gate（唯一未通過者為 KEYS，其正面提示詞仍偏向 Sell，Margin 為 `-0.515882`）。

### 正規化狀態轉移度（Normalized Transfer）定義

$$T = \frac{M_{\mathrm{patched}} - M_{\mathrm{target}}}{M_{\mathrm{source}} - M_{\mathrm{target}}}$$

其中 $T = 1$ 代表補丁後完全重現來源 Clean Margin；$T = 0$ 代表無干預效果。

### 跨層掃描結果
從 L4 起，單層全位置抽換在雙方向的翻轉率均達 100%。等權雙向正規化轉移度自 L0 的 `0.9197` 迅速攀升至 L14 的 `0.9977` 與 L30 的 `0.9999`。平均絕對誤差 $|M_{\mathrm{patched}} - M_{\mathrm{source}}|$ 隨層數從 L0 的 `1.06` 降至 L14 的 `0.15`，L30 則低於 `0.01`。

**分析**: 單層替換所有 Token 位置會將來源計算完整帶入下游，因此無法單獨解碼決策的語義位置。據此，Phase 2 將語義區間探索區間凍結於 **L14–L26**，聚焦精細的語義跨度。

---

## 3. Phase 2 Discovery: 語義區間因果隔離

- **Run ID**: `artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-phase2-discovery-20260830`
- **規模**: L14–L26、104 對合格提示詞、1,248 筆記錄。

| 干預方向 | 語義區間 (Span) | 平均 $\Delta M$ | 平均正規化轉移度 ($T$) | 等權翻轉率 |
|---|---|---:|---:|---:|
| Positive $\to$ Negative | `all_evidence` | +4.401945 | 0.3708 | 0.95% |
| Negative $\to$ Positive | `all_evidence` | -7.870503 | 0.6601 | 99.05% |
| Positive $\to$ Negative | `evidence_qual` | +1.050084 | 0.0906 | 0.95% |
| Negative $\to$ Positive | `evidence_qual` | -3.834784 | 0.3241 | 27.62% |
| Positive $\to$ Negative | `evidence_quant` | +1.559078 | 0.1308 | 0.00% |
| Negative $\to$ Positive | `evidence_quant` | -5.065151 | 0.4255 | 53.33% |
| Positive $\to$ Negative | `final_position` | +7.400028 | 0.6197 | 82.86% |
| Negative $\to$ Positive | `final_position` | -7.545092 | 0.6311 | 99.05% |
| Positive $\to$ Negative | `instruction` (含輸出位) | +11.857183 | 0.9938 | 100.00% |
| Negative $\to$ Positive | `instruction` (含輸出位) | -11.856134 | 0.9938 | 100.00% |

### 提出 `instruction_context`（排除輸出位置）
為驗證效果是否僅來自最後決策 Token，進一步將指令區塊嚴格切分為「輸出前指令前綴（`instruction_context`）」與「最後輸出位置（`final_position`）」。

- **Follow-up Run**: `artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-instruction-context-discovery-20260830`

| 干預方向 | 平均 $\Delta M$ | 平均正規化轉移度 ($T$) | 等權翻轉率 |
|---|---:|---:|---:|
| Positive $\to$ Negative | +8.519363 | 0.7159 | 99.05% |
| Negative $\to$ Positive | -10.339682 | 0.8654 | 100.00% |

**關鍵發現**: 即使**完全不替換最後一個決策輸出位置**，L14–L26 的 `instruction_context` 殘差狀態已具備足夠的因果充分性，使 99% 以上的提示對產生雙向翻轉。這證明了在輸出端上游存在高度集中的決策表徵。

---

## 4. Phase 3 Discovery: Layer × Span 全矩陣定位

- **Run ID**: `artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-phase3-layer-span-discovery-20260830`
- **分析產出**: `analyze/phase3_localization.json`（L0–L30 × 3 種 Spans × 雙方向，共 19,344 筆記錄）。

### 三大語義區間的跨層動態剖面

下圖展示了模型在不同層級與語義區間下的因果充分性轉移曲線（Panel a）以及最終在 Held-Out 測試集確認的 3×3 轉移矩陣（Panel b）：

![因果轉移曲線與 3x3 驗證矩陣](../assets/jspace-causal-tracing/causal_tracing_position_transfer.png)

1. **財務證據區塊 (`all_evidence`)**:
   - L0 即達 `0.9195`（95% CI `[0.8988, 0.9389]`），於 **L7 達到峰值 0.9912**，L9 以前均維持在 0.97 以上。
   - 自 L10（`0.8992`）起快速衰退：L12（`0.7312`）$\to$ L14（`0.5156`）$\to$ L15（`0.2842`）$\to$ L20（`0.0057`）。
2. **指令前綴區塊 (`instruction_context`)**:
   - L0–L9 轉移度接近零，自 L10（`0.0734`）起顯著增長，於 **L16 達到峰值 0.6386**。
   - 自 L19（`0.5624`）起逐步交棒：L21（`0.4983`）$\to$ L22（`0.4009`）$\to$ L30（`0.0844`）。
3. **最後輸出位置 (`final_position`)**:
   - L0–L15 轉移度接近零，自 L16（`0.1340`）開始爬升。
   - 自 L21（`0.4958`）與 Context 交叉，在 **L22（0.5916）確立主導地位**，最終在 **L30 達到 0.9131**。

### 兩大因果遞交區間（Position-Transfer Intervals）
- **Evidence $\to$ Context 遞交區間**: 位於 **L14 至 L15** 之間（L14 Evidence $0.5156 > \text{Context } 0.4041$；L15 Context $0.6114 > \text{Evidence } 0.2842$）。
- **Context $\to$ Final Position 遞交區間**: 位於 **L21 至 L22** 之間（L21 兩者相當：Context $0.4983 \approx \text{Final } 0.4958$；L22 Final $0.5916 > \text{Context } 0.4009$）。

### 雙干預方向的獨立動態曲線

![雙方向轉移曲線分離圖](../assets/jspace-causal-tracing/causal_tracing_direction_split.png)

如上圖所示，將 Positive $\to$ Negative 與 Negative $\to$ Positive 兩方向分別繪製，層級遞交的先後順序（Early Evidence $\to$ Middle Context $\to$ Late Final Position）在兩個方向完全一致。在 L16 處，Negative $\to$ Positive 的 Context 轉移度（`0.7612`）高於 Positive $\to$ Negative（`0.5160`），但兩者均在 L16 形成局部峰值。

---

## 5. 預註冊 Calibration 驗證

- **Run ID**: `artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-confirmation-calibration-20260830`
- **分析產出**: `analyze/confirmation_evaluation_v1.json`
- **規模**: 12 個獨立 Calibration Tickers、36 對提示詞（全部通過 Phase 0 Clean Gate）。

### 凍結矩陣檢驗結果

| 預註冊主對角條件 | 平均正規化轉移度 | 正向 $\to$ 負向 | 負向 $\to$ 正向 |
|---|---:|---:|---:|
| **L6 `all_evidence`** | **0.9884** | 0.9699 | 1.0068 |
| **L16 `instruction_context`** | **0.6395** | 0.5117 | 0.7673 |
| **L30 `final_position`** | **0.9129** | 0.8978 | 0.9279 |

- **列主導優勢（Row Dominance）**:
  - L6 Evidence 主導差值: `0.9894`（95% CI `[0.9721, 1.0050]`）
  - L16 Context 主導差值: `0.4825`（95% CI `[0.4772, 0.4887]`）
  - L30 Final 主導差值: `0.8700`（95% CI `[0.8658, 0.8739]`）
- **統計顯著性**: 12/12 Tickers 均達成正向優勢；單尾精確符號翻轉檢定（Exact Sign-Flip Test）$p = 0.000244$，Holm 調整後顯著性均為 **$p = 0.000732$**。

Calibration 門檻全數通過，授權以完全不變的配置進入 Held-Out 測試。

---

## 6. Held-Out Formal Confirmation 驗證

- **Run ID**: `artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-confirmation-test-20260830`
- **分析產出**: `analyze/confirmation_evaluation_v1.json`
- **規模**: 11 個未見過的 Held-Out Technology Tickers、33 對提示詞（全部通過 Phase 0 Clean Gate）。

### 凍結 3×3 轉移矩陣

| 補丁層級 | 財務證據 (`all_evidence`) | 指令前綴 (`instruction_context`) | 最後輸出位 (`final_position`) |
|---:|---:|---:|---:|
| **L6** | **0.9756** | 0.0034 | -0.0007 |
| **L16** | 0.1762 | **0.6314** | 0.1346 |
| **L30** | 0.0001 | 0.0847 | **0.9136** |

### 雙方向轉移細部數據

| 預註冊主對角條件 | 正向 $\to$ 負向 ($T$) | 負向 $\to$ 正向 ($T$) |
|---|---:|---:|
| **L6 `all_evidence`** | 0.9562 | 0.9950 |
| **L16 `instruction_context`** | 0.5211 | 0.7417 |
| **L30 `final_position`** | 0.9030 | 0.9243 |

### 列主導對比與統計檢定（Row Dominance & Significance）

| 檢驗對比 | 平均優勢差值 | Ticker-Bootstrap 95% CI | 正向實體數 | 單尾 Exact $p$ | Holm 調整後 $p$ |
|---|---:|---:|---:|---:|---:|
| **L6 Evidence Dominance** | **0.9743** | [+0.9515, +0.9926] | 11/11 | 0.000488 | **0.001465** |
| **L16 Context Dominance** | **0.4759** | [+0.4606, +0.4922] | 11/11 | 0.000488 | **0.001465** |
| **L30 Final Dominance** | **0.8713** | [+0.8686, +0.8739] | 11/11 | 0.000488 | **0.001465** |

**驗證結論**: 測試完全滿足樣本數門檻、對角轉移門檻、列主導門檻、Bootstrap 下界門檻、雙向對稱門檻與 Holm 多重檢定校正。**Formal Verdict 宣告 `success=true`**。

---

## 7. 科學解讀與方法學邊界

1. **因果充分性（Sufficiency）而非必要性（Necessity）**:
   - 激活值重抽樣補丁（Resample Patching）證明了指定區域的狀態足以轉移固定輸出的決策 Margin；這不等於證明該區域是模型計算的唯一通道或不可或缺的必要成分。
2. **分散式表徵整合**:
   - 實驗否定了「決策僅在最後一個輸出 Token 臨時形成」的假設；證實模型在中期層（L16）已將證據充分編碼並寫入下游 Instruction Context 殘差流中。
3. **不可外推為離散推理鏈**:
   - 本因果路徑描述的是連續向量空間中決策充分性的層級流動，不應被解釋為離散的 Chain-of-Thought 或符號邏輯推演步。

---

## 8. 可重現性驗證指令與圖表生成

### 驗證評估指令
```bash
uv run jspace-intervention analyze-activation-patching-confirmation \
  --records artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-confirmation-test-20260830/forward/phase3_records.jsonl \
  --baseline artifacts/qwen3.5-4b/jspace-causal-tracing/runs/causal-trace-confirmation-test-20260830/forward/phase0_baseline.jsonl \
  --config artifacts/qwen3.5-4b/jspace-causal-tracing/configs/position-transfer-confirmation-v1.json \
  --output /tmp/confirmation_evaluation.json \
  --split test
```

### 圖表渲染指令
```bash
uv run python scripts/render_causal_tracing_figures.py
```

圖表產出保存於 `docs/assets/jspace-causal-tracing/`，並在 `figures_provenance.json` 中以 SHA-256 綁定輸入數據與繪圖參數。

---

## Version Record

| Version | Date | Status |
|---|---|---|
| Draft 1 | 2026-08-30 | **Completed**; Discovery 定位跨層遞交，Calibration 通過，Held-out Confirmation 獲得 `success=true` ($p = 0.001465$) |
