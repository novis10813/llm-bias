# Entity Cell Localization: Proposal V1 (Header Family)

**Document status:** frozen V1 protocol; implemented; discovery run complete (see [report-v1](report-v1.md)). Negative result (template-dominated). Superseded by [proposal-v2](proposal-v2.md). Version index: [README](README.md).

**Model for run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)  
**Depends on:**
- Split manifest: `artifacts/qwen3.5-4b/jspace-intervention/splits.json`
- Generic baseline: `data/entity-cell/generic-baseline-qwen3.5-9b-v1.jsonl` (399 generic cloze prompts, identity `adapted:qwen3.5-9b-generic-cloze-v1`)
- Financial prompts: 105 prompts across 35 Technology discovery tickers (3 prompts/ticker).

---

## 1. 核心假說與文獻邊界（Scientific Question & Literature Boundary）

### 研究問題
大模型是否在早期 MLP 層（L0–L5）存在單一神經元級別的公司實體表徵（entity cells）？在固定的財務 Header 結構下，能否藉由跨變體標準化穩定度指標定位這些神經元？

### 文獻依據與差異對照表
本方法基於 **Barzilay et al. (arXiv 2604.01404, 2026)** 的 Entity Cell 定位與失憶（Amnesia）過濾架構。

| 維度 | Barzilay et al. (2026) 原始設定 | 本專案 V1 適應性修改（Adaptations） | 理論風險與邊界限制 |
|---|---|---|---|
| **模型架構** | Qwen2.5-7B (純 Full-Attention Transformer) | Qwen3.5-4B (Hybrid: Linear + Full Attention) | 4B 參數量較小，可能更偏向分散式多義表徵（superposition）。 |
| **任務與語料** | PopQA 事實檢索（例如「法國的首都在哪」） | 財務決策情境（35 家科技公司財務分析提示詞） | 財務文本含多行結構化數據，上下文遠比短事實句複雜。 |
| **定位 Prompt 構造** | 100 個多樣化屬性模板（`The <attribute> of <entity>`），$K=2$ 題/實體 | 12 個結構固定的三行 Header 前綴變體 | **重大理論風險**：所有變體共享相同的三行 Header 模板，可能導致模板反應神經元方差極小而主導穩定度排名。 |
| **候選層範圍** | 早期 MLP 層（通常為前 20% 層數） | 凍結在 L0–L5（32 層中的前 6 層） | 符合早期 MLP 實體編碼假設。 |
| **穩定度指標** | $S_{\ell j} = (\mathbb{E}_i z)^2 / (\operatorname{Std}_i z + \varepsilon)$ | 完全相同，$\varepsilon = 10^{-6}$，基於 399 題 generic baseline 做 z-score 標準化 | 若模板神經元在各變體激活恆高且標準差極低，分母趨近於 0 會造成分數暴增。 |

---

## 2. 預期 Input / Output 契約

### Input 契約
- `header_variants.jsonl`：包含 35 家公司各 12 個三行 Header 提示詞變體（8 個 localization，4 個 held-out）。
- `baseline_prompts.jsonl`：399 題 generic cloze 提示詞。
- `financial_prompts.jsonl`：105 題真實財務提示詞。

### Output 契約
- 產出路徑：`e1/baseline_stats.json`、`e1/cells.jsonl`、`e1/amnesia.jsonl`、`analyze/summary.json`。
- 資料格式：compact JSONL。每筆記錄僅保存 per-ticker top-5 候選元、穩定度分數、held-variant overlap、控制組 overlap、失憶劑量曲線指標。
- **禁令**：嚴禁儲存未聚合之原始激活向量、神經元激活矩陣或全詞表 logits。
- 數值約束：所有數值必須為 finite float。

### CLI 契約 1:1 綁定
本協議綁定專屬子命令：
```bash
entity-cell run-localization \
  --prepared-dir <prepared_dir> \
  --model .cache/models/qwen3.5-4b \
  --run-id <run_id> \
  --artifact-root artifacts \
  --localization-family v1-header \
  --stages e1-baseline e1-localization e1-amnesia analyze
```

---

## 3. 邊界情況與防禦性行為（Edge Cases & Fail-Safe Policies）

1. **模板撞車退化（Template Dominance Degeneracy）**：
   若 $\ge 80\%$ 的公司共享完全相同的 top-1 神經元，表示定位算法被固定模板主導，標記為退化並直接觸發版本升級。
2. **對照組退化（Degenerate Wrong-Entity Control）**：
   在失憶測試中，若錯實體對照組抽出的神經元與目標神經元完全相同，該門檻退化為無效，必須依賴 matched-random 對照判定。
3. **失憶門檻分母保護**：
   當 $|M_{\text{anon}} - M_{\text{clean}}| < 0.1$ 時，分母過小導致 progress 不穩定，將該 prompt 排除於 gate 計算之外。

---

## 4. 版本分立觸發條件（Version Break Triggers）

一旦發生以下任一變更，**必須另立新版（即 V2），嚴禁原地修改本文件**：
1. 修改定位提示詞構造方式（如從三行 Header 改為自然句 Frames）；
2. 調整表面形式控制組的實作邏輯或重新分詞機制；
3. 修改四道合格門檻（Gates）；
4. 變更 generic baseline 數據集或候選層範圍。
