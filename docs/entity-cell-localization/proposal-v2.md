# Entity Cell Localization: Proposal V2 (Natural-Sentence Frame Family)

**Document status:** frozen V2 protocol (frozen 2026-09-02, commit `91bd657`); discovery run complete (see [report-v2](report-v2.md)). 1/35 trusted candidate (FTNT); shared-top-1 caveat. Calibration and test not run. Version index: [README](README.md).

**Model for run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)  
**Depends on:**
- Prepared inputs: `artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-prepare-discovery-v2` (contains 420 natural-sentence frame variants across 35 Technology discovery tickers, plus template control).
- Generic baseline: `data/entity-cell/generic-baseline-qwen3.5-9b-v1.jsonl` (399 prompts).
- Financial prompts: 105 prompts across 35 Technology discovery tickers.

---

## 1. 核心假說與文獻邊界（Scientific Question & Literature Boundary）

### 研究問題
在自然語言句型（消除固定前綴結構）中，能否定位到不依賴特定表面句式的公司實體神經元？實體候選單元能否通過表面形式控制組（ROT13 與 Anonymous）與模板特徵檢驗？

### 文獻依據與差異對照表
本方法改進自 **Barzilay et al. (2026)** 的表面可變化實體提示詞定位。

| 維度 | Barzilay et al. (2026) 原始設定 | 本專案 V2 適應性修改（Adaptations） | 理論風險與邊界限制 |
|---|---|---|---|
| **句式設計** | 100 個屬性模板（`The <attribute> of <entity>`） | 12 個固定自然敘述句框（F0–F7 localization, H0–H3 held），公司名為純文本嵌入 | 句框數量由 100 縮減為 12，多樣性不及論文，但足以打破固定三行 Header。 |
| **表面控制方向** | 別名/拼寫變體「通用化測試」（要求不同別名激活同一神經元） | 匿名與 ROT13「特異性測試」（要求名稱被抹除或擾動時神經元**不得**激活） | **重大方向差異**：本專案測試的是實體特異性（specificity），非別名泛化性（generalization）。 |
| **模板基線控制** | 無顯式模板控制 prompt | 新增中立實體 Header 控制組（`[NEUT]`）建立 Template Signature | 顯式篩除任何對 Header 殘餘格式起反應的神經元。 |
| **實體 token 位置** | 屬性句末尾或實體末尾 token | 公司名稱 span 的最後一個 content token | 由字元偏移與 tokenizer offset mapping 精確解析。 |

---

## 2. 預期 Input / Output 契約

### Input 契約
- `frame_variants.jsonl`：包含 35 家公司各 12 個自然句框（F0–F7 localization，H0–H3 held-out）。
- `template_control.json`：包含 1 題中立 Header 模板控制題。
- `baseline_prompts.jsonl`：399 題 generic cloze 提示詞。
- `financial_prompts.jsonl`：105 題真實財務提示詞。

### Output 契約
- 產出路徑：`e1/baseline_stats.json`、`e1/cells.jsonl`、`e1/template_signature.json`、`e1/amnesia.jsonl`、`analyze/summary.json`。
- 資料格式：compact JSONL。每筆記錄包含 `v2_candidate_eligibility`、`collision_selectivity`、四道 gate 逐項檢驗結果。
- 禁令：嚴禁儲存未聚合之原始激活向量。
- 數值約束：所有數值必須為 finite float。

### CLI 契約 1:1 綁定
本協議綁定專屬子命令：
```bash
entity-cell run-localization \
  --prepared-dir <prepared_dir> \
  --model .cache/models/qwen3.5-4b \
  --run-id <run_id> \
  --artifact-root artifacts \
  --localization-family v2-frames \
  --stages e1-baseline e1-localization e1-amnesia analyze
```

---

## 3. 邊界情況與防禦性行為（Edge Cases & Fail-Safe Policies）

1. **四道合格門檻（Fail-Closed Gates）**：
   - Gate 1 (`held_variant_overlap >= 1`): Top-5 候選在 4 個未見句框中至少重疊 1 顆。
   - Gate 2 (`form_robust == True`): Top-5 候選單元**均不得**出現在 `anonymous_name_frames` 或 `name_form_control_frames` 的 top-5 排名中（重新 tokenize）。
   - Gate 3 (`template_robust == True`): Top-5 候選單元**均不得**出現在中立模板簽名 top-5 中。
   - Gate 4 (`amnesia_endpoint_pass >= 2`): 在 $\alpha = -3.0$ 下，至少 2 題財務題目之 $A_p(-3) > 0$ 且高於 wrong-entity 與 matched-random。
2. **錯實體退化防禦（Deterministic Non-Degenerate Fallback）**：
   依字母順序選取下一個 ticker，循序檢查其候選單元，直至找到與目標單元不同之神經元。若全部相同，則標記為 `degraded_control: True`，且僅比對 matched-random。
3. **跨 Ticker 撞車警示（Cross-Ticker Collision Caveat）**：
   若某神經元通過門檻，但同時為多個其他 ticker 的 top-1 單元（如 FTNT 的 (0, 104) 亦為 17 家公司的 top-1），此單元僅視為**候選單元（Candidate）**，不得直接宣稱具備單一公司專屬實體性，必須移交 E3 進行跨 ticker 抑制驗證。

---

## 4. 版本分立觸發條件（Version Break Triggers）

以下任一變更必須另立新版（如 `proposal-v3.md`），禁止原地修改本文件：
1. 新增跨 ticker 選擇性硬門檻（如 collision count limit）；
2. 引入論文式的別名泛化測試（Alias Generalization）；
3. 修改 12 個自然句框的文本模板；
4. 變更四道門檻的及格邏輯。
