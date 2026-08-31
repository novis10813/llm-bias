# Sector and Context Follow-up Experiments: Proposal

本文件定義 activation-patching causal tracing 之後的三個獨立實驗。三者重用 active
baseline inputs、Qwen3.5-4B 與既有 split manifest，但各自維護 estimand、artifact 與
verdict。開始 model run 前先凍結本文件所列的 input、controls 與 analysis contract。

- **A V1: cross-sector header-state patching**：固定 evidence，抽換 identity-header
  residual states。
- **B V1: cross-sector context overriding under negative evidence**：固定 negative
  evidence，抽換 post-evidence `instruction_context` states。
- **C V1: L16 instruction-context Jacobian-lens readout**：解碼 L16 context 的 transported
  vocabulary alignment。

A 與 B 是 residual resample-patching sufficiency 實驗。C 是 descriptive transported
representation readout。三者都不保存 raw residuals、activations、hidden states、Jacobians、
gradients 或 full-vocabulary arrays。

## Shared Inputs and Split Contract

Population inference 使用
`artifacts/qwen3.5-4b/jspace-intervention/splits.json` 的 Technology 與 Financial Services
assignments。Discovery、calibration、test 不共用 ticker。Raw structured attribute trials
來自 active baseline trial plan；prompt rendering 只讀 structured evidence fields，不解析
原 prompt body。

每個 matched record 包含兩個 identities 與一組完全相同的 qualitative/quantitative
evidence。Preparation 必須：

1. 依固定 seed 在同一 split 內 deterministic pairing Technology 與 Financial Services
   tickers；population analysis 以 identity pair 為聚合單位。
2. 平衡 evidence origin sector，避免所有 evidence 都來自同一 sector。
3. 排除 evidence body 含任一 paired ticker 或完整 company name 的 records。
4. 只允許 bracketed ticker/name header 在 matched prompts 間不同；evidence 與 instruction
   必須 byte-identical。
5. 保存 source trial identity、evidence hashes、identity pair、split manifest hash 與 input
   hash。

Apple (`AAPL`) ↔ JPMorgan (`JPM`) 只作 smoke pair。Formal population claim 不以單一
entity pair 為推論單位。

## A V1: Cross-Sector Header-State Patching

### Question

在 evidence 固定時，哪個 layer 的 identity-header residual states 足以改變固定 Buy/Sell
continuation margin？Early-layer effect 與 middle-layer effect 分開報告，不把 layer profile
稱為 attention 或唯一 information route。

### Intervention

對每個 matched pair 執行 Technology→Financial Services 與 Financial
Services→Technology 兩方向 patching。每次只替換 `header` span，在 L0–L30 逐一 patch
single layer。Company names token length 不同，因此 mapping 使用 target-complete
nearest-normalized position mapping。

### Discovery Outcomes

每個 layer/direction 報告：

- clean source/target margin；
- patched target margin 與 margin change；
- discrete flip；
- clean source-target margin gap 非零時的 normalized source-state transfer；
- equal-pair mean、direction-specific mean 與 pair-level consistency。

Discovery 只定位 effect onset 與候選 layers。看完 discovery 前不設定 formal effect
threshold。Calibration/test gate 必須在 discovery 後另行凍結。

### Controls

- **Self-source patch**：同 prompt states patch 回自身，patched margin 必須在數值容差內等於
  clean margin。
- **Same-sector peer header patch**：控制 company-specific identity replacement。
- **Name-form control**：保留字元形式但不代表真實 entity。
- **Evidence-origin strata**：Technology-origin 與 Financial-Services-origin evidence 分開
  報告。

跨 sector effect 只有在超過 same-sector peer control 且 evidence-origin strata 方向一致時，
才可描述為 sector-conditioned header-state effect。

### Interpretation Limits

Effect 表示 source header state 在該 layer 對固定 margin 具有 resample-patching
sufficiency。它不證明該 state 必要、不等同 attention effect，也不單靠一個
Apple↔JPMorgan pair 建立 sector-level claim。

## B V1: Cross-Sector Context Overriding under Negative Evidence

### Question

在完全相同的 negative evidence 下，sector identity 是否改變 L14–L21 的 post-evidence
context state，使抽換該 state 足以移動或翻轉 target decision？

### Intervention

只使用 negative qualitative 與 quantitative evidence。Matched prompts 的 evidence 與
instruction 完全相同，header 分別使用 Technology 與 Financial Services identity。
Primary intervention 在 L14–L21 逐層抽換 `instruction_context`，排除 final decision
position，並執行兩個 sector directions。

Existing same-ticker positive→negative context patching 只作 outcome-state-transfer positive
control。它直接搬運 valence-conditioned state，不能當作 sector-bias estimand。

### Outcomes

Primary outcome 是 equal-pair mean target-margin change。另報：

- 相對 clean source-target gap 的 normalized transfer（denominator 非零時）；
- discrete flip；
- pair-level directional consistency；
- layer profile；
- evidence-origin strata。

### Controls

- 同 layer 的 `header`-only patch；
- 同 layer 的 `final_position`-only patch；
- same-sector peer `instruction_context` patch；
- self-source `instruction_context` patch；
- existing positive→negative same-ticker context patch 作 positive control。

若 cross-sector context effect 沒有超過 same-sector peer control，不得稱為 sector bias。
若 effect 只在某個 evidence-origin stratum 出現，結果按 stratum 報告，不合併成 general
sector claim。

### Interpretation Limits

B 測量 fixed negative evidence 下 sector-conditioned context state 的 sufficiency。它不證明
模型忽略 evidence，也不把 L14–L21 稱為唯一 decision computation。`flip` 與 normalized
transfer 分開解讀，因為 clean margins 可能不對稱。

## C V1: L16 Instruction-Context Jacobian-Lens Readout

### Question

L16 `instruction_context` 的 transported vocabulary alignment 較接近 frozen
financial-evidence token family，或 frozen sector token families？

### Discovery-Only Vocabulary Nomination

在任何 calibration/test L16 readout 前，從既有 discovery artifacts 凍結 token IDs：

1. **Financial-evidence family**：
   `artifacts/qwen3.5-4b/jspace-valence-readout/runs/valence-technology-evidence-balanced-20260827T043734Z/analyze/frozen_candidate_suggestions.json`
   的 12 個 positive/negative valence candidates。
2. **Technology sector family** 與 **Financial Services sector family**：
   `artifacts/qwen3.5-4b/jspace-intervention/config-tfidf-tech-to-financial.json` 的 frozen
   contrastive-TF-IDF prototype token IDs。
3. 移除跨 family 重複 token IDs；不在查看 L16 outputs 後手動加詞。

Frozen config 保存每個 source artifact path、SHA-256、token ID、decoded token、family 與
nomination rule。V1 不新增 frequency-matched control family；若加入，建立新版本。

### Readout Contract

對每個 prepared prompt：

1. 在 memory 中 capture L16 residual states。
2. 解析 `instruction_context`，排除 final decision position。
3. 對 span 每個 position 使用 validated canonical Jacobian lens transport。
4. 套用 FP32 final norm 與 LM head，再計算完整 vocabulary softmax。
5. 先平均 prompt 內所有 span positions 的完整 softmax，再於 condition/ticker 層級平均。
6. 完成 aggregate 後才選 top-k。Disk 只保存 compact top-k、family mass/rank、entropy 與
   provenance。

Primary site 固定為 L16 `instruction_context`。L6 `all_evidence`、L30 `final_position` 是
known-site comparisons；L16 `header` 是 span control。

### Outcomes

每個 split、condition、layer/span 報告：

- 每個 frozen family 的 total probability mass；
- family token mean/median rank；
- positive-minus-negative family-mass difference；
- matched Technology-minus-Financial-Services family-mass difference；
- aggregate top-k 與 entropy diagnostics。

Discovery 只能檢查 readout viability。Calibration/test 的 primary contrasts、bootstrap CI
與 gate 必須在讀取 test outputs 前寫入 frozen config。Vocabulary 不得由 calibration/test
重新提名。

### Interpretation Limits

Jacobian lens 提供 local first-order transported representation readout。Token mass、rank 與
top-k 描述 vocabulary alignment，不是 causal effect、attention map、chain-of-thought、離散
reasoning step 或完整 mediation path。C 的結果不能取代 A/B 的 patching evidence。

## Artifact Layout

三個 experiment 使用獨立 dataset slugs：

```text
artifacts/<model-slug>/cross-sector-header-patching/runs/<run-id>/
artifacts/<model-slug>/cross-sector-context-overriding/runs/<run-id>/
artifacts/<model-slug>/l16-context-readout/runs/<run-id>/
```

Run 遵循 `prepare → forward → analyze` stages 與 manifest finalization。每個 formal output
綁定 input、split manifest、frozen config，C 另綁定 canonical lens SHA-256。Slug root 另
保存 frozen configs 與 prepared pair inputs（例如
`artifacts/<model-slug>/l16-context-readout/configs/` 與
`artifacts/<model-slug>/cross-sector-header-patching/prepared/`）；B 重用 A 的 prepared
records，因此 `cross-sector-context-overriding/` 沒有 slug-level prepared 目錄。

Discovery run records 見 [實驗報告](report-discovery.md)。
