# Sector and Context Follow-up Experiments: Discovery Report

本報告記錄 A V1（cross-sector header-state patching）、B V1（cross-sector context overriding under negative evidence）與 C V1（L16 instruction-context Jacobian-lens readout）的 discovery runs 實驗結果、對照組分析、bootstrap 信賴區間與科學解讀。

實驗問題、干預定義、對照設計與 artifact contract 見 [實驗提案](../proposal.md)。本報告中所有 discovery 結果均為描述性定位與探索性證據，未凍結 formal confirmation gate 前不支撐驗證性統計結論。

---

## 1. 執行概覽與 Run 清單

所有 discovery 與 diagnostic runs 均在 Qwen3.5-4B（`.cache/models/qwen3.5-4b`）上執行完畢，所有 patch records 均使用 target-complete nearest-normalized position mapping，並通過 split contract、manifest SHA-256 與 self-source exact no-op 驗證。

### A V1: Cross-Sector Header Patching Runs

- **AAPL↔JPM Smoke**：`artifacts/qwen3.5-4b/cross-sector-header-patching/runs/cross-sector-header-aapl-jpm-smoke-20260831`（56 records，self-source $|\Delta M| = 0.00000000$）。
- **廢棄 Discovery Run**：`cross-sector-header-discovery-20260831`（因使用修正前非 ROT13 name-form 輸入且未受限 cache，手動停止並標記 failed，結果不予採納）。
- **正式 Corrected Population Discovery**：`artifacts/qwen3.5-4b/cross-sector-header-patching/runs/cross-sector-header-discovery-rot13-bounded-20260831`
  - Prepared Input：`prepared/discovery_pairs_v1-rot13.jsonl`（602 records、29 個跨產業 pairs、平衡 positive/negative 證據層）。
  - Patching：L0–L30 全層 sweep、雙方向。
  - 產出：30,132 compact records；Manifest 狀態 `complete`；7,192 筆 self-source records 之最大 $|\Delta M| = 0.00000000$。

### B V1: Cross-Sector Context Overriding Runs

- **AAPL↔JPM Smoke**：`artifacts/qwen3.5-4b/cross-sector-context-overriding/runs/cross-sector-context-aapl-jpm-smoke-20260831`（30 records，self-source $|\Delta M| = 0.00000000$）。
- **Primary Population Discovery**：`artifacts/qwen3.5-4b/cross-sector-context-overriding/runs/cross-sector-context-discovery-20260831`
  - Prepared Input：301 negative-evidence prepared records（29 個跨產業 pairs）。
  - Patching：L14–L21；包含 primary `instruction_context`、`header` control 與 `final_position` control 三種 spans。
  - 產出：11,664 compact records；Manifest 狀態 `complete`。
- **ROT13 Name-form Control Follow-up V2**：`artifacts/qwen3.5-4b/cross-sector-context-overriding/runs/cross-sector-context-name-form-rot13-followup-v2-20260831`
  - Prepared Input：`discovery_name_form_negative_v1-rot13.jsonl`（58 prepared records）。
  - Patching：L14–L21；三種 spans。
  - 產出：2,784 compact records；Manifest 狀態 `complete`。

### C V1: L16 Context Readout Runs

- **Technical Smoke**：`artifacts/qwen3.5-4b/l16-context-readout/runs/l16-context-readout-smoke-20260831`（8 compact records，canonical lens SHA-256 驗證通過）。
- **Technology Discovery**：`artifacts/qwen3.5-4b/l16-context-readout/runs/l16-context-readout-discovery-20260831`（105 pairs、35 Technology tickers、840 compact records）。
- **Financial Services Discovery V2**：`artifacts/qwen3.5-4b/l16-context-readout/runs/l16-context-readout-financial-services-discovery-v2-20260831`（108 pairs、36 Financial Services tickers、864 compact records）。
- **跨產業彙總分析**：`artifacts/qwen3.5-4b/l16-context-readout/analyze/cross_sector_discovery_v1.json`（10,000 次 ticker-level bootstrap）。

---

## 2. 實驗 A: Cross-Sector Header-State Patching Discovery 結果

### 評估指標

Primary estimand 為朝向來源抬頭條件 clean margin 的移動量（Toward-Source $\Delta M$）：

$$\operatorname{sign}(M_{\mathrm{source}} - M_{\mathrm{target}}) \cdot \Delta M$$

正值表示抽換 header 狀態後，target 提示詞的 Buy/Sell margin 朝 source clean margin 移動。對 29 個 cross-sector identity pairs 等權平均，並計算 95% ticker-pair bootstrap 信賴區間。

### L0 至 L30 逐層剖面

| Layer | Cross-Sector Toward-Source $\Delta M$ | 95% Bootstrap CI | Cross-Sector $|\Delta M|$ | 同產業 Peer $|\Delta M|$ | ROT13 Control $|\Delta M|$ |
|---:|---:|---:|---:|---:|---:|
| 0 | +0.62732 | [+0.54257, +0.70950] | 0.67886 | 0.69062 | 1.30573 |
| 1 | +0.65259 | [+0.58393, +0.72456] | 0.68411 | 0.66608 | 1.27028 |
| 2 | +0.66222 | [+0.58838, +0.73576] | 0.69313 | 0.66172 | 1.28556 |
| 3 | +0.64519 | [+0.57184, +0.72215] | 0.68318 | 0.64346 | 1.28444 |
| 4 | +0.63626 | [+0.55602, +0.71819] | 0.68291 | 0.61960 | 1.33254 |
| 5 | +0.62383 | [+0.53730, +0.71484] | 0.67776 | 0.62832 | 1.31702 |
| 6 | +0.61321 | [+0.52173, +0.70588] | 0.67070 | 0.59309 | 1.31547 |
| 7 | +0.47884 | [+0.38977, +0.57054] | 0.56646 | 0.52337 | 1.17674 |
| 8 | +0.46154 | [+0.37941, +0.55293] | 0.54675 | 0.50893 | 1.14845 |
| 9 | +0.42853 | [+0.34743, +0.51087] | 0.52978 | 0.49170 | 1.06378 |
| 10 | +0.33335 | [+0.25840, +0.41383] | 0.43429 | 0.36340 | 0.97527 |
| 11 | +0.09132 | [+0.05605, +0.12918] | 0.18498 | 0.14992 | 0.41701 |
| 12 | +0.06615 | [+0.03612, +0.10023] | 0.12845 | 0.11106 | 0.31963 |
| 13 | +0.05380 | [+0.02399, +0.08606] | 0.11792 | 0.09640 | 0.27612 |
| 14 | +0.04574 | [+0.01444, +0.07873] | 0.11536 | 0.09378 | 0.24890 |
| 15 | -0.00009 | [-0.01271, +0.01324] | 0.06400 | 0.05926 | 0.14382 |
| 16 | -0.00690 | [-0.01960, +0.00510] | 0.04997 | 0.04599 | 0.08318 |
| 17 | -0.00438 | [-0.01689, +0.00722] | 0.04635 | 0.03978 | 0.06209 |
| 18 | -0.00292 | [-0.01544, +0.00894] | 0.04481 | 0.03999 | 0.05903 |
| 19 | -0.00565 | [-0.01725, +0.00498] | 0.04158 | 0.03222 | 0.05308 |
| 20 | -0.00364 | [-0.01562, +0.00692] | 0.03908 | 0.03001 | 0.04854 |
| 25 | -0.00534 | [-0.01744, +0.00552] | 0.03600 | 0.02916 | 0.06399 |
| 30 | -0.00070 | [-0.00375, +0.00224] | 0.01302 | 0.01317 | 0.02165 |

### 正負證據層分層分析

- **Positive Evidence Stratum**：L0 Toward-Source $= +0.85252$（95% CI $[+0.69045, +1.02394]$，100% 正向 pairs）；L6 $= +0.84693$；L14 $= +0.06062$；L16 $= -0.01094$（CI 跨越 0）；L30 $= -0.00006$。
- **Negative Evidence Stratum**：L0 Toward-Source $= +0.40212$（95% CI $[+0.30591, +0.49935]$，96.6% 正向 pairs）；L6 $= +0.37950$；L14 $= +0.03085$；L16 $= -0.00287$（CI 跨越 0）；L30 $= -0.00134$。

### A Discovery 科學解讀與邊界

1. **中晚期 Header 因果充分性歸零**：從 L15 起至 L30，單獨抽換 Header 狀態的 Toward-Source $\Delta M$ 完全歸零（95% CI 均包含 0）。這證實 pre-evidence header 位置在中晚期不再直接向下游 decision 輸出因果信號。
2. **早期 Header 效應為泛用 Entity Replacement**：在 L0–L6，跨產業 Header 抽換產生的 $|\Delta M| \approx 0.68$ 與同產業 Peer Header 抽換產生的 $|\Delta M| \approx 0.66$ 幅度高度重疊，ROT13 亂碼控制組產生更大擾動（$|\Delta M| \approx 1.31$）。這表明早期 Header 的干預效果屬於實體 Token 替換對後續計算的通用擾動，而非特定於 Sector 的獨立偏倚機制。

---

## 3. 實驗 B: Cross-Sector Context Overriding Discovery 結果

### 實驗設計與問題

在完全相同的 negative financial evidence 下，測試 Technology 與 Financial Services 抬頭所衍生出的 post-evidence `instruction_context` 殘差狀態（排除 final decision position），是否足以改變 target 的 Buy/Sell decision margin。

### L14 至 L21 逐層剖面與對照組比較

| Layer | CS Context (Toward-Source $\Delta M$) | 95% Bootstrap CI | CS Header Control | CS Final Position Control | Peer Context $|\Delta M|$ | ROT13 Context $|\Delta M|$ |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | +0.17320 | [+0.1116, +0.2335] | +0.03085 | -0.00561 | 0.13732 | 0.26365 |
| 15 | +0.30284 | [+0.2044, +0.4045] | +0.00895 | +0.00539 | 0.20707 | 0.51330 |
| **16** | **+0.31806** | **[+0.2255, +0.4151]** | **-0.00287** | **+0.03861** | **0.21436** | **0.57141** |
| 17 | +0.30111 | [+0.2355, +0.3742] | +0.00325 | +0.13954 | 0.18056 | 0.53843 |
| 18 | +0.24144 | [+0.1869, +0.2992] | +0.00246 | +0.19470 | 0.14329 | 0.43845 |
| 19 | +0.24056 | [+0.1821, +0.3000] | -0.00003 | +0.21324 | 0.14739 | 0.44948 |
| 20 | +0.21334 | [+0.1587, +0.2685] | -0.00055 | +0.24368 | 0.13354 | 0.40372 |
| 21 | +0.20613 | [+0.1555, +0.2595] | -0.00042 | +0.25458 | 0.12526 | 0.38288 |

### L16 峰值處之關鍵對比

- **Context vs. Header Control**：差值為 **+0.32093**（95% CI $[+0.22780, +0.41608]$）。在 L16，Header 抽換為完全的 No-op（$-0.00287$），而 Context 抽換產生強烈 Toward-Source 移動（$+0.31806$）。
- **Context vs. Final Position Control**：差值為 **+0.27945**（95% CI $[+0.20166, +0.36124]$）。在 L16，Context 效應顯著大於 Final Position（$+0.03861$）。
- **動態轉移轉折點**：在 L14–L18，Context 效應顯著優於 Final Position；自 L19–L21 起，Final Position 效應逐步上升（L21 達 $+0.25458$）並超越 Context（$+0.20613$），與主幹因果追蹤的 L21–L22 Context $\to$ Final Position handoff 一致。

### B Discovery 科學解讀與邊界

在相同看跌負面財務證據下，模型在 L15–L17 的 `instruction_context` 區域形成了攜帶實體與產業先驗特徵的殘差表徵。抽換此處狀態能顯著移動決策 Margin，且顯著優於同層 Header 與 Final Position 控制組。

---

## 4. 實驗 C: L16 Context Jacobian-Lens Readout Discovery 結果

### 實驗設計與問題

使用凍結的 Jacobian Lens，解碼 L16 `instruction_context` 輸送至輸出詞表的機率質量對齊（Vocabulary Alignment），檢驗其是否與凍結的 12-Token 財務證據詞表或 4-Token 產業原型詞表對齊。

### 詞表定義與 SHA-256 凍結

- Frozen Config：`artifacts/qwen3.5-4b/l16-context-readout/configs/l16-context-readout-v1.json`（SHA-256: `ec8d4676e04b5b3977e4b9ec0da845c2d2e7ede11aca3adf7db6e93039f10e50`）。
- **12-Token Financial Evidence Family**：`potential`, `predicted`, `risks`, `downgrade`, `impacts`, `risk`, `justify`, `Industry`, `upgrade`, `increase`, `justified`, `partnership`。
- **4-Token Technology Family**：`quantum`, `spin`, `cloud`, `manufacturer`。
- **4-Token Financial Services Family**：`provision`, `border`, `deposit`, `scrutiny`。

### 跨產業詞表機率質量與差異（Technology vs. Financial Services）

彙總來源：`artifacts/qwen3.5-4b/l16-context-readout/analyze/cross_sector_discovery_v1.json`（10,000 次獨立 Ticker-Bootstrap）。

| Readout Site    | Family             |   Technology Positive |       Technology Negative |           FS Positive |               FS Negative |                                      Difference-in-Differences (95% CI) |
| --------------- | ------------------ | --------------------: | ------------------------: | --------------------: | ------------------------: | ----------------------------------------------------------------------: |
| **L16 Context** | Financial Evidence | $6.36 \times 10^{-4}$ | **$7.97 \times 10^{-4}$** | $6.45 \times 10^{-4}$ | **$8.47 \times 10^{-4}$** | $+4.02 \times 10^{-5}$ ($[-2.37 \times 10^{-5}, +1.06 \times 10^{-4}]$) |
| L16 Context     | Technology Sector  | $1.54 \times 10^{-5}$ |     $1.49 \times 10^{-5}$ | $1.58 \times 10^{-5}$ |     $1.51 \times 10^{-5}$ | $-2.45 \times 10^{-7}$ ($[-5.30 \times 10^{-7}, +3.17 \times 10^{-8}]$) |
| L16 Context     | FS Sector          | $2.85 \times 10^{-5}$ |     $3.41 \times 10^{-5}$ | $2.88 \times 10^{-5}$ |     $3.40 \times 10^{-5}$ | $-4.12 \times 10^{-7}$ ($[-1.23 \times 10^{-6}, +3.87 \times 10^{-7}]$) |
| **L16 Header**  | Financial Evidence | $6.52 \times 10^{-4}$ |     $6.51 \times 10^{-4}$ | $5.86 \times 10^{-4}$ |     $5.87 \times 10^{-4}$ | $+6.87 \times 10^{-7}$ ($[-2.31 \times 10^{-8}, +1.40 \times 10^{-6}]$) |
| **L6 Evidence** | Financial Evidence | $9.46 \times 10^{-4}$ |     $9.38 \times 10^{-4}$ | $9.42 \times 10^{-4}$ |     $9.41 \times 10^{-4}$ | $+6.43 \times 10^{-6}$ ($[-4.84 \times 10^{-6}, +1.77 \times 10^{-5}]$) |
| **L30 Final**   | Financial Evidence | $5.76 \times 10^{-7}$ |     $2.18 \times 10^{-6}$ | $6.55 \times 10^{-7}$ |     $1.97 \times 10^{-6}$ | $-2.94 \times 10^{-7}$ ($[-8.68 \times 10^{-7}, +2.79 \times 10^{-7}]$) |

### C Discovery 科學解讀與邊界

1. **價態對齊在兩產業中高度重現**：在 Technology 與 Financial Services 中，負面條件在 L16 Context 對 12-Token 財務證據詞表的輸送機率質量均顯著高於正面條件（Positive $-$ Negative 分別為 $-1.62 \times 10^{-4}$ 與 $-2.02 \times 10^{-4}$）；而在 L16 Header 控制組中，正負條件差異接近零（$\sim 10^{-7}$ 數量級）。這表明 L16 Context 確實承載了來自證據區塊的價態語義輸送。
2. **表面產業詞彙未在 L16 Context 分離**：4-Token 產業原型詞表的絕對機率質量極低（$10^{-5}$ 數量級），且跨產業 DiD 的 95% Bootstrap CI 均跨越 0。
3. **方法學定位**：Jacobian Lens 為一階線性輸送解碼，反映的是詞表對齊，不能直接當作因果證據；C 目前維持 `formal_success_gate = false`，不支撐基於產業表面詞彙分離的驗證性結論。

---

## 5. 綜合科學結論

```
[Prompt 輸入]
   │
   ├─> 早期層 (L0-L6):
   │   Header (Ticker/Name) 與 Evidence 進行底層語義編碼。
   │   此處抽換 Header 產生的是通用 Token/Entity 擾動 (Peer 與 ROT13 均產生同等或更大效應)。
   │
   ├─> 中期層 (L14-L17) ──【關鍵決策與價態整合中樞】：
   │   Evidence 自身的因果充分性開始衰退，轉移至 post-evidence Instruction Context (L16 Peak)。
   │   在此處：
   │   (1) 價態因果轉移達顯著峰值 (Transfer = 0.6314, p < 0.001)。
   │   (2) 相同負面證據下，不同實體/產業衍生出的 Context 表徵足以驅動 Margin 偏向來源 (dM ~ +0.32)。
   │   (3) Jacobian Lens 解碼證實此處 Context 具備穩定的負面價態詞表輸送對齊。
   │
   └─> 晚期層 (L20-L30) ──【輸出路由】：
       Context 的充分性在 L20-L21 逐步交棒給 Final Position，最終在 L30 投射至 Buy/Sell Margin。
```

---

## 6. Version Record

| Experiment                         | Version | Status                                                                                                                                                                |
| ---------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Activation Patching Causal Tracing | V1      | **Held-out confirmed (`success=true`)**；證明 L6 Evidence $\to$ L16 Context $\to$ L30 Final Position 因果轉移                                                         |
| Cross-sector header-state patching | A V1    | **Discovery complete**（`cross-sector-header-discovery-rot13-bounded-20260831`）；L0–L6 為泛用實體替換，L15–L30 效應歸零；calibration/test gate 未凍結                |
| Cross-sector context overriding    | B V1    | **Discovery complete；calibration `success=false`**。L16 Context effect 與 Context–Header contrast 通過 effect/statistical gates，但 same-sector peer specificity gate 未通過；`test_authorized=false`，見 [confirmation report](../report.md) |
| L16 instruction-context readout    | C V1    | **Discovery complete**（Technology 與 FS runs 完成）；證實價態對齊重現，產業表面詞彙未分離；`formal_success_gate = false`                                             |
