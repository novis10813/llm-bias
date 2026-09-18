# Shared experiment core

`llm_bias/core/` 保存 active experiments 共用、且不帶單一研究線語意的 workflow mechanics。
Experiment package 仍擁有自己的研究定義、config schema、artifact schema 與呈現方式。
本文件說明 core 的 ownership、相容入口與驗證範圍；artifact identity 的完整契約見
[Artifact identity and run manifest contract](artifact-contract.md)。

## Workflow 對應

Shared experiment workflow 是 `prepare → forward → analyze → finalize`：

| Stage | Core package | 責任 |
|---|---|---|
| `prepare` | `llm_bias.core.prompt_input` | chat template、tokenization、character/token span 對齊與 continuation token 檢查 |
| `forward` | `llm_bias.core.inference` | batch encoding、model forward、generation、logit extraction 與暫態 residual intervention hooks |
| `analyze` | `llm_bias.core.analysis` | transported readout、distribution statistics、top-k records、direction 與 paired statistics |
| `finalize` | `llm_bias.core.artifacts` | atomic serialization、parent hash 驗證、stage lifecycle、manifest 與 finalization postcheck |

`finalize` 不是 artifact tree 中的獨立 stage。`ArtifactRun.finalize()` 只在 required stages
完成後，把 manifest 標為 `complete`。各 experiment 可以使用自己的 stage names；例如
prompt-analysis 使用 `readout`、`forward`、`backward`，J-space intervention 與 span
sensitivity 使用 `prepare`、`forward`、`analyze`。

## Package ownership

### `prompt_input/`

`contracts.py` 定義 `TokenSpan`。`encoding.py` 擁有 `format_messages`、`format_prompt`、
`input_ids`、`token_span`、`span_record`、`all_occurrences`、`continuation_token_ids` 與
`find_token_subsequence`。新的 prompt formatting 或 token alignment 共用能力放在此處，
不要在 experiment package 複製。

### `inference/`

- `adapter.py` 定義 model adapter protocol 與 injected adapter。
- `forward.py` 提供 `encode_batch`、`forward_batch`、`capture_final_residuals` 與
  intervention workflow 使用的 `record_residuals`。
- `logits.py` 統一處理 Hugging Face output 形狀與 final-position logits。
- `mlp.py` 提供 dense MLP down projection 前的單位置記錄、縮放與替換；只接受
  batch-one 與絕對 token 位置，離開 context 後移除 hooks，不保存 raw values。
- `continuations.py` 對已驗證的 prompt/suffix IDs 作完整續接 FP32 計分；單 token
  候選共用一次 forward，多 token 使用 teacher forcing，不改動 prompt 干預位置。
- `coordinate_screen.py` 的 `coordinate_derivatives(..., save_on_cpu=False)` 可選擇
  以 `torch.autograd.graph.save_on_cpu(pin_memory=False)` 將反向傳播所需張量暫存在
  CPU RAM，使用時回傳原裝置；不寫磁碟、不改權重精度、目標量或篩選層。
  `investment-dial run-check` 與 `run-screen` 以 `--save-on-cpu` 啟用，並在
  `prepare/protocol.json` 記錄同名 `save_on_cpu` 布林欄位；預設關閉，舊產物缺少
  此欄位表示未啟用。此選項不改完整 logits 計算或 JSON decision/reason 生成，
  也不對 calibration/evaluation 增加參數。CPU/GPU 傳輸可能增加耗時；GPU 峰值
  仍需以真實模型工程檢查測量，不能保證固定顯存上限。
- `generation.py` 定義 `GenerationConfig`、`generate_tokens` 與 `finish_reason`。
- `interventions.py` 提供 exception-safe forward-hook context manager。Hooks 只在 forward
  期間修改 residual；離開 context 後必須移除。

部分 API 只從 submodule 匯入，例如 `record_residuals`、`next_logits` 與
`residual_interventions`。不要假設 `llm_bias.core.inference` re-export 所有 symbol。

### `analysis/`

- `transport.py` 擁有 Jacobian transport 與 method metadata。
- `distributions.py` 擁有 restricted/full-vocabulary statistics、effective temperature，
  以及先平均完整 vocabulary softmax 再選 top-k 的 `mean_full_vocabulary`。
- `records.py` 產生 compact token 與 attribution records。
- `statistics.py` 擁有 bootstrap、paired test、Holm correction、cosine statistics 與
  direction hash。
- `directions.py` 擁有 online direction accumulation 與 quantile helpers。
- `decision_flips.py` 解析真實生成的 JSON Buy/Sell decision，並計算 paired flip
  count、方向與 valid-pair rate；不接受 fixed-choice margin sign 作為替代。

Research-specific estimand、success gate 或 vocabulary 定義留在 owning experiment。

### `artifacts/`

`ArtifactRun`、`StageContext` 與 `run_context` 管理 stage 狀態。`io.py` 的 atomic writers
預設拒絕 overwrite，並拒絕 tensor、array、non-finite value 與 raw activation/gradient
payload。`load_parent_jsonl` 驗證 supplied hash、previous-stage hash 與 sidecar hash。

`artifacts/provenance.py` 提供 local checkpoint 檔案、tokenizer 與 Python source
指紋；實驗 package 擁有需要核對哪些來源的研究契約。

`artifacts/manifest.py` 與 `artifacts/paths.py` 是 package namespace facades；canonical
實作位於 `core/artifact_manifest.py` 與 `core/artifact_paths.py`。Run root 固定為：

```text
artifacts/<model-slug>/<dataset-slug>/runs/<run-id>/
```

同一路徑的 artifact 重新註冊時，`RunManifest` 會取代舊 reference 並重算 record counts。
一般 workflow 不應靠此行為覆寫已完成 run；建立新 run ID 保留 provenance。

## Top-level core modules

- `model.py`：tokenizer/model loading、diagnostics 與 Qwen 27B multi-GPU device maps。
- `continuation_scoring.py`：teacher-forced continuation likelihood、fixed-choice margin、
  categorical KL 與 FP32 single-token margin。`fp32_next_token_logits` 提供 final norm 加
  unembedding 的 FP32 logits tail；`fp32_next_token_log_probs` 在同一 tail 後套用
  log-softmax。V2 direction decode 重用前者，確保 fitting target 與 vocabulary
  decode 使用相同 unembedding convention。Final norm 直接呼叫模型的 `_final_norm`
  module（FP32 輸入），不對 norm 公式做手動再實作（見下方測量變更記錄 v2）。
- `artifact_paths.py` / `artifact_manifest.py`：run identity、hash、atomic path helpers 與
  schema-version 1 manifest。
- `lens_artifacts.py`：canonical/candidate/archive path、schema-version 2 metadata 與 model
  shape/layer coverage validation。
- `lens_registry.py`：解析 `config/pretrained_lenses.json`，只接受 exact model identity
  與 pinned revision。
- `lens_loader.py`：載入 lens 後執行 model identity、shape、coverage 與 metadata 驗證。

`prompt_input.continuation_token_ids` 只回傳 continuation suffix；
`continuation_scoring.continuation_token_ids` 回傳 prompt IDs 與 suffix IDs。兩者同名但契約
不同，匯入時使用完整 module path。

## 測量變更記錄（Instrument Change Log）

shared core 的計分 tail 一旦改變，所有下游 margin/方向數值的絕對值都會變，因此在此
明確定版。此記錄只描述 shared core 的測量定義；各 experiment 的 run 狀態與重驗結果
寫在對應 experiment 的 README/report。

### v2（2026-09-03，branch `fix/fp32-tail-norm`）：final norm 改由模型 module 計算

- **變更**：`fp32_next_token_logits` 不再手動重實作 final norm（舊式
  `x·rsqrt(mean(x²)+ε)·w`），改為直接呼叫模型的 `_final_norm` module（FP32
  輸入、FP32 輸出）。`entity_cell.attention_attribution.frozen_margin_direction` 同步修正：
  per-dimension multiplier `A` 改由 `norm(1)/rsqrt(1+ε)` 從實際 module 提取，使 frozen
  direction 與 v2 tail 精確一致。
- **原因**：舊手動公式對 standard RMSNorm（`norm·w`，例如 Llama-3.2-1B）與 LayerNorm
  精確，但 Qwen3.5 的 `Qwen3_5RMSNorm` 是 Llama-3 風格（weight 初始為 0，forward 為
  `norm·(1+w)`），舊式漏掉 `1+`，導致所有 Qwen3.5 的 margin 都經過一個與模型真實
  unembedding 不同的固定線性變換（等價於把 normalized residual 先乘一個固定的正對角
  矩陣再投影）。
- **影響範圍**：所有在 Qwen3.5-4B / Qwen3.5-9B 上經 `score_single_token_margin_fp32`、
  `fp32_next_token_log_probs`、`fp32_next_token_logits` 或 `frozen_margin_direction` 產出的
  margin、DLA 方向與 readout logits——包含 `entity_cell`（E1 amnesia、E2 DLA、E3、
  lens readout）與 `jspace_intervention`（outcome_flip、activation_patching、
  cross_sector_patching、context_readout、context_overriding、prior_probe、runner、
  outcome_decode）的全部 Qwen3.5 runs。Llama-family 的 runs 數值不變（舊公式對其精確）。
- **結構性影響說明**：同一 probe 內部的相對比較（劑量方向、pair 對照、匿名推進度、
  DLA 分組佔比）在舊 run 內部仍然自洽，但「margin 符號 = 模型決策」與任何以 0 為界的
  判定（decision flip、邊界題的 sign）必須以 v2 儀器重驗；絕對 margin 值一律以 v2
  重跑或重算為準。
- **驗證**：`tests/test_continuation_scoring.py`（standard 與 Llama-3 兩種 norm 風格的
  module 一致性與 true logprobs 回歸測試）、`tests/test_entity_cell_e2.py`（frozen
  direction 與 true margin / core tail 的一致性）；並對 Qwen3.5-4B 真實模型驗證
  v2 tail 與模型真實 logits maxdiff = 0。
- **重驗狀態（2026-09-03，entity_cell 線，Qwen3.5-4B）**：
  - E3 V1：`entity-cell-e3-discovery-v4`（CPU fp32）完成，全部 frozen gates 通過；
    「匿名 = Sell」的 decision-conflict 前提被推翻（真決策下匿名基線為 Buy），324 records 0 flips。
    見 [`entity-cell-localization/report-e3-v1.md` §6](entity-cell-localization/details/report-e3-v1.md)。
  - E2：`entity-cell-e2-discovery-v6`（CPU fp32，13,440 DLA records）以真方向重算，
    top-5 head 選擇、排名與 routing labels 全部不變（DLA 放大约 1.4–1.5×）。
    見 [`entity-cell-localization/report-e2.md` §5](entity-cell-localization/details/report-e2.md)。
  - E1 V2：`entity-cell-e1-discovery-v4`（GPU bf16，完整四階段全重驗）完成。
    官方原生 GPU bf16 精度下 FTNT 通過全部四道門檻（`form_robust` 在 bf16 官方精度下
    確認為 pass，overlap=0），維持 1/35 trusted 唯一候選；全 35 家 amnesia endpoint gate
    共有 13 家通過（含 v1 的 9 家與 4 家新通過），其餘 12 家皆被 form-robust 或
    template 排除。輔助 CPU fp32 針對性重驗（`entity_cell_amnesia_recheck.py`）作為
    off-device 交叉驗證對照。
    見 [`entity-cell-localization/report-v2.md` 附錄](entity-cell-localization/details/report-v2.md)。
  - `jspace_intervention` 線的 Qwen3.5 runs 同受影響（絕對 margin/readout 值），為 frozen 線，
    不回填；未來若啟用該線需先以 v2 儀器重驗。

## Lens ownership boundary

Core 只負責 lens identity、path、metadata、registry lookup 與 validated loading：

- fitting 與 candidate evaluation 位於 `llm_bias/lens_fitting/`；
- pinned artifact download/install 位於 `llm_bias/lens_install/`；
- candidate promotion 由 `scripts/promote_qwen_lens_candidate.py` 執行；
- experiment workflow 只能消費已驗證的 canonical lens，不得 fitting、安裝、promotion
  或覆寫 lens。

Canonical lens、candidate-selection 例外與 promotion 條件見
[Qwen Jacobian-lens selection](jacobian-lens-selection/proposal.md)。

## Compatibility facades

以下入口支援既有 callers，修改或移除前先搜尋 active 與 frozen code：

- `core/prompting.py` re-export `prompt_input`；`lens_fitting` 仍使用此入口。
- `core/readout.py` re-export `core.analysis`，並保留 `last_unmasked_positions`；
  prompt-analysis 仍使用此入口。
- `core/directions.py` 為 frozen archive 保留舊 direction imports。
- `core/artifacts/manifest.py` 與 `core/artifacts/paths.py` re-export top-level canonical
  implementations。

`baseline_trial` 直接使用部分 `prompt_analysis` modules 是既有 compatibility exception。
不要新增同類依賴；把新的共用 mechanics 放入 core。

## 保存與 import 邊界

Core 不得包含：

- Buy/Sell、sector、entity pair 等 experiment-specific research semantics；
- 對任何 experiment package 的 import；
- raw activations、residuals、hidden states、gradients、Jacobians 或 KV caches 的持久化；
- lens fitting、下載、安裝或 promotion；
- experiment-specific artifact schema、dashboard 或 figure rendering。

Forward 執行期間可以在記憶體中使用 residual 或 gradient。Serializer 與 manifest
只接受 compact derived outputs，例如 top-k、rank、scalar statistics、token IDs/text、
probabilities 與 provenance。

## Verification

修改 core 時先執行受影響的 targeted tests：

```bash
uv run pytest -q \
  tests/test_prompt_input.py \
  tests/test_core_inference.py \
  tests/test_core_analysis.py \
  tests/test_core_artifacts.py \
  tests/test_artifact_manifest.py \
  tests/test_artifact_paths.py \
  tests/test_continuation_scoring.py \
  tests/test_workflow_boundaries.py
```

Lens identity、path 或 loading 變更另執行：

```bash
uv run pytest -q \
  tests/test_lens_artifacts.py \
  tests/test_lens_loader.py \
  tests/test_lens_registry.py \
  tests/test_lens_install.py \
  tests/test_lens_promotion.py
```

`test_lens_install.py` 與 `test_lens_promotion.py` 驗證 core 邊界外的 installer/promotion，
但它們會鎖定 core path 與 metadata contract。完整 repository gate 仍以 root
`AGENTS.md` 的 Verification 為準。