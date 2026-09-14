# Entity Cell Localization: Proposal E3 V1 (Upstream Suppression with Cross-Ticker Specificity and Downstream Component Attenuation)

**Document status:** frozen E3 V1 Discovery protocol; discovery run complete (see [report-e3-v1](report-e3-v1.md)). Calibration and test not run. Version index: [README](README.md).

**Model for run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)  
**Depends on:**
- Prepared inputs: `artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-prepare-discovery-v2` (105 financial prompts across 35 Technology discovery tickers, 3 prompts/ticker).
- Completed E1 V2 discovery: `artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-e1-discovery-v2` (sole trusted candidate: `FTNT`, cell `(L0, N104)`, stability score 363.4; baseline stats in `e1/baseline_stats.json`).
- Completed E2 discovery: `artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-e2-discovery-v5` (selected heads: `(31, 0)`, `(31, 1)`, `(31, 3)`, `(19, 4)`, `(27, 6)`).

---

## 1. 核心假說與文獻邊界（Scientific Question & Literature Boundary）

### 研究問題
E1 V2 中唯一通過四道門檻的實體單元 FTNT `(L0, N104)`，究竟是「真正的 FTNT 專屬實體神經元」，還是「17 家公司共享的通用實體語法/槽位神經元」？
透過因果壓制（suppression）與跨 Ticker 對照，能否確立該神經元對 FTNT 的專屬因果必要性？壓制該單元是否會特異性地調控下游 E2 full-attention heads 的實體 DLA 貢獻，同時保留證據（evidence）貢獻？

### 文獻依據與差異對照表

| 維度 | Barzilay et al. (2026) / Chughtai et al. (2024) 原始設定 | 本專案 E3 適應性修改（Adaptations） | 理論風險與邊界限制 |
|---|---|---|---|
| **干預對象** | Barzilay: 對通過 amnesia filter 的實體神經元進行單元壓制 | 包含目標公司（FTNT）、同撞車公司（ADI, MU）與異組公司（FTV） | 本實驗首創「跨 Ticker 同單元抑制對照」，以因果手段直接檢驗單義性 vs 通用語法槽位。 |
| **劑量網格** | Barzilay: 抑制倍率 $\alpha \in \{1, 0, -1, -2, -3\}$ | 完全相同：$\alpha \in \{1.0, 0.5, 0.0, -1.0, -2.0, -3.0\}$ | 負倍率表示反向激活（negative ablation）。 |
| **下游衰減** | Chughtai: 玩具任務 IOI 上的整體 head 衰減 | E3-B: 僅對 E2 選定 5 個 heads 之**重構實體 source 向量**進行衰減（$\beta \in \{1.0, 0.75, 0.5, 0.25, 0.0\}$） | 不直接抑制整顆注意力頭，而是保留證據與指令向量，僅精確衰減實體更新路徑。 |
| **控制組基準** | 常規隨機神經元抽樣 | 同層 matched-random、確定性錯實體（wrong-entity）、以及同單元跨 Ticker 對照 | 三重對照交叉鎖定特異性。 |

---

## 2. 預期 Input / Output 契約

### Input 契約
- `financial_prompts.jsonl`：包含 105 題真實財務提示詞。
- `cells.jsonl`（來自 E1 V2）：包含 35 家公司候選單元清單。
- `summary.json`（來自 E1 V2）：必須包含 `v2_candidate_eligibility` 欄位。**合格 Trusted Ticker 判定嚴格以 `v2_candidate_eligibility[ticker]["eligible"] == True` 為準，禁止僅依賴 V1 的 amnesia 局部欄位**。
- `head_attribution.jsonl`（來自 E2 v5）：提供選定 heads 名單。

### Output 契約
- **輸出路徑**：`e3/suppression.jsonl`（上游）、`e3/downstream.jsonl`（下游）、`analyze/summary.json`、`manifest.json`。
- **資料格式**：嚴格 compact JSONL。每筆記錄包含 ticker、prompt_id、phase、scope、dose、margin、clean_margin、anonymous_margin、anonymous_progress ($A_p$)、flip、mediation_deltas、controls 與 provenance。
- **禁令**：嚴禁儲存任何未聚合之 raw activations、hidden states、residuals、KV caches 或注意力權重。
- **數值約束**：所有純量必須為 finite float。

### CLI 契約 1:1 綁定
本協議綁定專屬子命令：
```bash
entity-cell run-intervention \
  --prepared-dir <prepared_dir> \
  --model .cache/models/qwen3.5-4b \
  --run-id <run_id> \
  --artifact-root artifacts \
  --stages e3-upstream e3-downstream analyze \
  --e1-run-root <e1_run_root> \
  --e2-run-root <e2_run_root> \
  --peer-tickers ADI MU FTV
```

---

## 3. 邊界情況與防禦性行為（Edge Cases & Fail-Safe Policies）

1. **下游 Random Subset 分組無空洞保證（Partition Integrity）**：
   在 E3-B 的 `random_subset` 控制組中，從非實體 token 隨機抽樣 $K$ 個 token 作為假實體時，**原始的 `identity_header` token 位置必須強制併入 `other_prefix`**。系統在 forward hook 執行前必須強制斷言：
   $$\bigcup_{g \in \text{SOURCE\_GROUPS}} \text{positions}(g) == \text{range}(\text{query}), \quad \text{且組間交集為空}$$
   若有任何位置遺失或重疊，立即阻斷執行。
2. **錯實體單元不等性防禦（Non-Degenerate Wrong-Entity Rule）**：
   在為 Trusted Ticker 選取 `wrong_entity` 時，必須同時滿足兩項條件：
   - Ticker 來源不同（依 discovery 字母序回繞）；
   - **神經元絕對不同**：`c["layer"] == target["layer"]` 且 `c["neuron"] != target["neuron"]`。
   若同 split 內的所有備選均等於目標神經元，標記為 `degraded_control: True`，失憶門檻自動改由 matched-random 單獨判定。嚴禁出現「目標為 (0, 104) 且錯實體也是 (0, 104)」之退化對照。
3. **合格資格識別邊界（V2 Eligibility Binding）**：
   解析 E1 run 時，若發現 `v2_candidate_eligibility` 欄位存在，則必須採用該字典判定 `trusted` 資格（本 run 中僅 FTNT 1 家合格）；若無該欄位則向後相容退回 V1 判定。嚴禁將未通過 form-robust 的同撞車公司混入 trusted 集合。

---

## 4. 評估指標與預註冊解讀標準

1. **Anonymous Progress**：
   $$A_p(\alpha) = \frac{(m(\alpha) - m_{\text{clean}}) \cdot g}{g^2 + \varepsilon}, \quad g = m_{\text{anon}} - m_{\text{clean}}$$
2. **跨 Ticker 特異性對照（Specificity Contrast）**：
   $$\Delta A_p^{\text{specificity}}(\alpha) = A_p^{\text{FTNT}}(\alpha) - \frac{1}{|\mathcal{P}|} \sum_{P \in \mathcal{P}} A_p^P(\alpha), \quad \mathcal{P} = \{\text{ADI}, \text{MU}\}$$
3. **預註冊解讀標準**：
   - **實體專屬性成立**：$A_p^{\text{FTNT}}(-3.0) > 0$ 且高於 wrong-entity / matched-random，同時 $A_p^{\text{FTNT}}(-3.0) - A_p^{\text{ADI}}(-3.0) > 0.10$ 與 $A_p^{\text{FTNT}}(-3.0) - A_p^{\text{MU}}(-3.0) > 0.10$。
   - **通用語法/槽位神經元成立**：$A_p^{\text{FTNT}}(-3.0) \approx A_p^{\text{ADI}}(-3.0) \approx A_p^{\text{MU}}(-3.0) > 0$。

---

## 5. 版本分立觸發條件（Version Break Triggers）

以下任一變更必須另立新版（如 `proposal-e3-v2.md`），禁止原地修改本文件：
1. 變更 $\alpha$ 抑制劑量網格或 $\beta$ 衰減網格；
2. 增減跨 Ticker 對照組名單；
3. 改變下游衰減的四種 mode（identity, evidence, random_subset, whole_head）；
4. 更改主要評估指標。
