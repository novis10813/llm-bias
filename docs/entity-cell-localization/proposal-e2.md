# Entity Cell Downstream Attribution: Proposal E2 (Full-Attention DLA)

**Document status:** frozen E2 protocol; implemented; discovery run complete (see [report-e2](report-e2.md)). Calibration and test not run. Version index: [README](README.md).

**Model for run:** Qwen3.5-4B (`.cache/models/qwen3.5-4b`)  
**Depends on:**
- Prepared inputs: `artifacts/qwen3.5-4b/entity-cell-localization/runs/entity-cell-prepare-discovery-v2` (105 financial prompts with disjoint source spans).
- Canonical lens: `artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt` (covers source layers up to L30).

---

## 1. 核心假說與文獻邊界（Scientific Question & Literature Boundary）

### 研究問題
在殘差流中，公司實體資訊如何透過注意力頭傳遞至最終決策位置？注意力層級的 identity contribution 是否足以改變 Buy/Sell margin，還是被證據與指令上下文壓制？

### 文獻依據與差異對照表
本方法基於 **Chughtai et al. (2024)** 的 source-token Direct Logit Attribution (DLA)。

| 維度 | Chughtai et al. (2024) 原始設定 | 本專案適應性修改（Adaptations） | 理論風險與邊界限制 |
|---|---|---|---|
| **模型架構** | 標準 Full Transformer (如 GPT-2, LLaMA) | Qwen3.5-4B (Hybrid: 8 Full Attention + 24 Recurrent Linear Attention) | **線性注意力層無法直接應用 source DLA**，本協議僅掃描 8 個 full-attention 層（L3, L7, L11, L15, L19, L23, L27, L31）。 |
| **任務類型** | 玩具任務 IOI (Indirect Object Identification, "John gave Mary...") | 財務分析決策（真實長文本多行 Prompt，約 400~500 tokens） | 文本長度增加導致 attention softmax 稀釋，且 query 位置包含 decision prefix。 |
| **Attention 機制** | 標準 MHA / GQA | Qwen3.5 含 `query_dependent output gate` (`attn_output_gate`) 與 RoPE/GQA | 必須在 hook 重建中精確重現 output gate 與 repeat_interleave，否則數值無法閉合。 |
| **數值精度** | FP32 | 模型為 BF16，DLA 重建為 FP32 | 存在約 0.3%~0.6% 的 BF16 浮點捨入噪聲，需採用尺度相對容差。 |

---

## 2. 預期 Input / Output 契約

### Input 契約
- `financial_prompts.jsonl`：包含 105 題 financial prompts。每筆記錄必須包含：
  - `prompt_id` (str)
  - `ticker` (str)
  - `final_query_position` (int, 嚴格大於所有 source tokens)
  - `source_groups` (Mapping): 必須包含 `identity_header`, `evidence`, `instruction_context`, `other_prefix` 四組，且**必須互斥且 100% 覆蓋 `range(final_query_position)`**。

### Output 契約
- **輸出路徑**：`e2/head_attribution.jsonl`、`e2/component_readout.jsonl`、`e2/patch_contracts.jsonl`、`analyze/summary.json`。
- **資料格式**：嚴格 compact JSONL。每筆記錄包含 `layer`, `head`, `group`, `frozen_scale_margin`, `additive_error` 等純量。
- **禁令**：**嚴禁保存完整注意力矩陣、全層 KV cache、未聚合之 head output vectors 或 full-vocabulary logits**。
- **數值約束**：所有數值必須為 finite float。

### CLI 契約 1:1 綁定
本協議綁定專屬子命令：
```bash
entity-cell run-attribution \
  --prepared-dir <prepared_dir> \
  --model .cache/models/qwen3.5-4b \
  --run-id <run_id> \
  --artifact-root artifacts \
  --e2-layers 3 7 11 15 19 23 27 31
```

---

## 3. 邊界情況與防禦性行為（Edge Cases & Fail-Safe Policies）

1. **加法重建容差（Additivity Relative Tolerance）**：
   重構 head outputs 之和與原始 attention block 輸出在 final query position 的殘差，必須滿足相對尺度容差公式：
   $$\max_{i} |r_i - c_i| \le \max\left(10^{-3},\; 2\times 10^{-2} \cdot \max_j |c_j|\right)$$
   若超過此門檻，視為注意力重構結構性錯誤（如 RoPE 偏差或 gate 遺失），直接拋出例外終止。
2. **Lens 覆蓋邊界**：
   Canonical lens 僅涵蓋源層至 L30。若選出的 head 位於 L31，`component_readout` 必須明確標註為 `readout_unavailable: true`，不得靜默替換層數。
3. **分組覆蓋斷言**：
   在執行任何投影前，強制檢查 $\bigcup_{g} \text{positions}(g) == \text{range}(\text{query})$。任何漏掉的 token 都會直接拋出 `ValueError`。

---

## 4. 版本分立觸發條件（Version Break Triggers）

以下任一變更必須另立新版（如 `proposal-e2-v2.md`），禁止原地修改本文件：
1. 納入 Recurrent Linear Attention 狀態歸因；
2. 修改 10:1 的 routing ratio 或篩選門檻；
3. 修改四個 source groups 的切分定義；
4. 改變 primary outcome（從 frozen-scale DLA 改為其他 estimand）。
