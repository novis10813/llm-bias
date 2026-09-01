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

## B V1: Frozen Confirmation Design

本節定義 B V1 cross-sector context overriding 的 calibration 與 held-out test gate。
Discovery 結果見 [report-discovery.md](report-discovery.md) §3。

本節在讀取 calibration 或 test outputs 前凍結，凍結後不修改 primary contrast、
thresholds、layers、spans 或統計方法。若需修改，建立 B V2。

### Discovery Summary（gate 設計依據）

Discovery（29 個 identity pairs、301 條 negative-evidence prepared records）在 L14–L21
找到以下 cross-sector `instruction_context` 效應：

| Layer | Toward-Source ΔM | 95% CI | Positive-fraction |
|---:|---:|---:|---:|
| 14 | +0.1732 | [+0.1152, +0.2336] | 86.2% |
| 15 | +0.3028 | [+0.2049, +0.4015] | 89.7% |
| **16** | **+0.3181** | **[+0.2256, +0.4106]** | **89.7%** |
| 17 | +0.3011 | [+0.2331, +0.3713] | 100% |
| 18 | +0.2414 | [+0.1845, +0.3000] | 100% |
| 19 | +0.2406 | [+0.1820, +0.3007] | 96.6% |
| 20 | +0.2133 | [+0.1617, +0.2700] | 96.6% |
| 21 | +0.2061 | [+0.1555, +0.2600] | 96.6% |

L16 Context 顯著超過同層 Header control（差 +0.3209）與 Final Position control
（差 +0.2795）。Same-sector peer context |ΔM| 為 0.2144，ROT13 name-form context
|ΔM| 為 0.5714。

### Primary Estimand

Context-Header Contrast at L16：

$$
C_{\mathrm{B}} = \overline{\Delta M}_{\mathrm{L16,\,context}}^{\mathrm{toward}} - \overline{\Delta M}_{\mathrm{L16,\,header}}^{\mathrm{toward}}
$$

其中 $\overline{\Delta M}^{\mathrm{toward}}$ 是 equal-pair mean Toward-Source margin
change，計算方式為：先在 identity pair 內平均兩個方向與兩個 evidence-origin strata，
再於 pairs 間等權平均。

### Confirmation Gates

Calibration 與 test 使用相同 gate。Test 只在 calibration 通過後執行。

#### 資料品質 gate

1. 至少 8 個 identity pairs 通過 clean-outcome gate（兩個 sector 的 clean margin
   符號均與 negative evidence 一致）。
2. Self-source patch 的 |ΔM| 對所有 records 均等於 0（FP64 exact no-op）。

#### Effect gate

3. L16 cross-sector context Toward-Source ΔM 的 equal-pair mean 大於 `0.10`。
4. L16 Context-Header Contrast $C_{\mathrm{B}} > 0.10$。
5. L16 cross-sector context Toward-Source ΔM 的 pair-bootstrap 95% CI 下界大於 0。
6. L16 Context-Header Contrast $C_{\mathrm{B}}$ 的 pair-bootstrap 95% CI 下界大於 0。

#### Specificity gate

7. L16 cross-sector context |ΔM| 大於 same-sector peer context |ΔM|。
8. Evidence-origin strata 一致性：Technology-origin 與 Financial-Services-origin
   evidence 的 L16 Toward-Source ΔM 均為正。

#### Statistical gate

9. L16 Toward-Source ΔM 的 one-sided exact pair sign-flip test $p < 0.05$。
10. L16 $C_{\mathrm{B}}$ 的 one-sided exact pair sign-flip test $p < 0.05$。
11. Gates 9–10 的兩個 p-values 經 Holm correction（$\alpha = 0.05$，$k = 2$）後
    仍通過。

Bootstrap 使用 10,000 次 pair-level resampling，seed `20260901`。

### Threshold 選擇依據

- `0.10` 的 effect threshold 約為 discovery peak（0.318）的 31%。Discovery 中
  最弱的層（L14）Toward-Source ΔM 為 0.173，仍超過此 threshold。設為 0.10 允許
  calibration/test 因 pair 組成不同而自然衰減，但仍排除 noise-level 效應。
- 不使用 discovery 的 threshold 值（0.318）作為 gate，因為 calibration/test 的
  pair 數少於 discovery（calibration 預期 ~10 pairs vs. discovery 29 pairs），
  point estimate 會有更大的抽樣變異。

### Aggregation

Pairs 內平均兩個方向（Tech→FS 與 FS→Tech）與兩個 evidence-origin strata；pairs
間等權平均。不對 pairs 按 clean margin gap 加權。

### Layers 與 Spans

Primary layer 固定為 L16。Calibration/test 同時執行 L14–L21 作為 descriptive
layer profile，但只有 L16 進入 formal gate。

每層執行三個 span conditions：`instruction_context`（primary）、`header`（control）、
`final_position`（control）。

### Input Preparation

Calibration 使用 split manifest 的 calibration tickers（Technology 12 + Financial
Services 12），test 使用 test tickers（Technology 11 + Financial Services 12）。
Prepare 命令重用既有 `prepare-cross-sector-patching` CLI，只改 `--split` 參數。
B 重用 A 的 prepared records，只在 forward 階段篩選 negative evidence。

### Artifact Layout

```text
artifacts/qwen3.5-4b/cross-sector-context-overriding/
├── configs/
│   └── b-v1-confirmation-v1.json          ← frozen gate config
├── runs/
│   ├── cross-sector-context-calibration-<date>/
│   │   ├── prepare/
│   │   ├── forward/
│   │   ├── analyze/
│   │   └── manifest.json
│   └── cross-sector-context-test-<date>/
│       ├── prepare/
│       ├── forward/
│       ├── analyze/
│       └── manifest.json
```

### Calibration Rule

Calibration 只檢查 frozen gates 是否通過。不修改 thresholds、layers、spans 或
contrasts。若 calibration 未通過，不執行 test，B V1 的 confirmation verdict 為
`success=false`。

### Test Rule

所有 gates（3–11）通過時 verdict 為 `success=true`。任何 gate 未通過時 verdict
為 `success=false`。Verdict 與逐條 gate 結果記錄於 confirmation report。

### Interpretation Limits

B V1 confirmation 測量 fixed negative evidence 下 sector-conditioned context state
對 fixed margin 的 resample-patching sufficiency。Confirmation 不證明模型忽略
evidence、不把 L16 稱為唯一 decision computation、也不建立 attention-based 因果
機制。Effect 與 specificity 結合只支持「sector identity 透過 L16 context 表徵調節
fixed-evidence 下的 decision margin」這一因果充分性聲明。

### Execution Commands

```bash
# 1. Prepare calibration pairs
uv run jspace-intervention prepare-cross-sector-patching \
  --input ../baseline/runs/paper-local-qwen36-27b/trial_plan.jsonl \
  --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
  --split calibration \
  --output artifacts/qwen3.5-4b/cross-sector-header-patching/prepared/calibration_pairs_v1.jsonl

# 2. Run B V1 context overriding on calibration
uv run jspace-intervention run-cross-sector-context-overriding \
  --prepared-pairs artifacts/qwen3.5-4b/cross-sector-header-patching/prepared/calibration_pairs_v1.jsonl \
  --model .cache/models/qwen3.5-4b \
  --run-id cross-sector-context-calibration-$(date -u +%Y%m%d) \
  --layers 14-21

# 3. If calibration passes, prepare test pairs
uv run jspace-intervention prepare-cross-sector-patching \
  --input ../baseline/runs/paper-local-qwen36-27b/trial_plan.jsonl \
  --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
  --split test \
  --output artifacts/qwen3.5-4b/cross-sector-header-patching/prepared/test_pairs_v1.jsonl

# 4. Run B V1 context overriding on test
uv run jspace-intervention run-cross-sector-context-overriding \
  --prepared-pairs artifacts/qwen3.5-4b/cross-sector-header-patching/prepared/test_pairs_v1.jsonl \
  --model .cache/models/qwen3.5-4b \
  --run-id cross-sector-context-test-$(date -u +%Y%m%d) \
  --layers 14-21
```
