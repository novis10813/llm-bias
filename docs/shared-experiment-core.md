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

Research-specific estimand、success gate 或 vocabulary 定義留在 owning experiment。

### `artifacts/`

`ArtifactRun`、`StageContext` 與 `run_context` 管理 stage 狀態。`io.py` 的 atomic writers
預設拒絕 overwrite，並拒絕 tensor、array、non-finite value 與 raw activation/gradient
payload。`load_parent_jsonl` 驗證 supplied hash、previous-stage hash 與 sidecar hash。

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
  categorical KL 與 FP32 single-token margin。
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

## Lens ownership boundary

Core 只負責 lens identity、path、metadata、registry lookup 與 validated loading：

- fitting 與 candidate evaluation 位於 `llm_bias/lens_fitting/`；
- pinned artifact download/install 位於 `llm_bias/lens_install/`；
- candidate promotion 由 `scripts/promote_qwen_lens_candidate.py` 執行；
- experiment workflow 只能消費已驗證的 canonical lens，不得 fitting、安裝、promotion
  或覆寫 lens。

Canonical lens、candidate-selection 例外與 promotion 條件見
[Qwen Jacobian-lens selection](qwen-jacobian-lens-selection.md)。

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