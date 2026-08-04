# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案定位

本 repository 研究 entity-sensitive representation 是否會在 decoder language model 的 residual stream 中形成，以及將 source entity activation 替換為 target entity activation 是否會改變答案分布。主要 workflow 是：

- `counterfactual-patching`：建立 entity-span pairs，執行 residual activation patching，分析 answer-margin transfer，並提供互動式 dashboard。
- `prompt-analysis`：對 CSV prompts 執行逐層 Jacobian-lens readout、uncertainty、generated-token attribution、ablation validation 與結果視覺化。
- `prepare-edgar-8k`：將外部 extracted EDGAR 8-K JSON 清理成可驗證的 JSONL staging artifacts。
- `prepare-counterfactual-data`：從 staged 8-K earnings events 建立 point-in-time entity histories、人工審核資料與 entity-only counterfactual pairs。
- `fit-jacobian-lens`：獨立的 lens fitting 工具；兩個 experiment workflow 只消費已存在的 lens，不可自行 fitting。

`jlens` 的 Jacobian readout 是 transported representation，不是模型的 chain-of-thought 或離散 reasoning path。

## 開發環境與依賴

- Python 版本為 3.13（見 `.python-version`），Python 依賴與 lockfile 一律由 `uv` 管理；新增套件使用 `uv add`。
- root project 以 editable workspace member 使用 `third_party/jacobian-lens/` 與 `third_party/jspace-viz/`。這些 checkout 被 `.gitignore` 忽略；新 checkout 必須先恢復它們再執行 `uv sync`：

  ```bash
  mkdir -p third_party
  git clone https://github.com/anthropics/jacobian-lens.git third_party/jacobian-lens
  git clone https://github.com/Festyve/jspace-viz.git third_party/jspace-viz
  uv sync
  ```

- 模型、lens binary、實驗輸出與第三方 checkout 不要加入 root Git；分別使用 `.cache/`、`artifacts/` 與 `third_party/`。
- 每個模型只能有一個 active canonical lens：`artifacts/lenses/<model>/jacobian_lens.pt`。partial/stride fitting 與 checkpoint 放在 `artifacts/archive/`，不要混入 active model directory。
- 預設模型是 `.cache/models/llama-3.2-1b-instruct`；Qwen3.5-4B 使用不同 residual width/layer count 的 model-specific lens，應使用 `scripts/run_qwen_lens_candidates.sh` 的候選選擇流程，而不是用小型 smoke corpus 覆寫 canonical lens。
- `prepare-counterfactual-data annotate` 的 optional dependency 是 `langextract[openai]`，可用 `uv run --extra extraction ...` 啟用；它預期本地 OpenAI-compatible llama.cpp endpoint。

## 常用驗證指令

```bash
uv sync
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
node --check llm_bias/static/counterfactual.js
```

目前 root package 沒有已配置的 Python lint command；不要把 third-party checkout 自己的 Ruff 設定當成 root lint 設定。若修改 static asset，除了 JavaScript syntax check 與 pytest，也要同步提高對應 HTML 的 query-string 版本，避免瀏覽器使用舊 asset。

單一 deterministic test 可用：

```bash
uv run pytest -q tests/test_interventions.py::test_normalized_span_mapping_uses_nearest_target_centers
```

測試應優先使用 deterministic unit tests、fake model、monkeypatch 與 temporary directories；不要為一般 unit test 載入大型 checkpoint。模型/GPU inference 應明確視為 smoke 或 integration test。

## 主要 CLI 與執行方式

所有 entry points 都定義在 `pyproject.toml`。`uv run python -m llm_bias` 只列出 workflow，不會執行實驗。

### Jacobian lens

```bash
uv run fit-jacobian-lens \
  --model .cache/models/llama-3.2-1b-instruct \
  --calibration-prompts 16
```

fitting 產生 model-specific canonical lens 與 metadata；checkpoint/resume artifacts 使用 `artifacts/archive/lens_checkpoints/`。Lens fitting 與 experiment package 保持分離。

### Counterfactual patching

```bash
uv run counterfactual-patching prepare-data
uv run counterfactual-patching run \
  --lens artifacts/lenses/llama-3.2-1b-instruct/jacobian_lens.pt
uv run counterfactual-patching summarize
uv run counterfactual-patching visualize
uv run counterfactual-patching serve --host 0.0.0.0 --port 8321
```

常用 smoke 選項是 `prepare-data`/`run` 的 `--max-pairs 4`。主要輸出位於 `artifacts/counterfactual_patching/`，包含 `pairs.jsonl`、`patch_results.jsonl`、layer summary、transfer/heatmap 圖與 dashboard 所需資料。

### Prompt analysis

```bash
uv run prompt-analysis readout \
  --input sp500_r1k_r2k_entityBiasPrompt.csv \
  --model .cache/models/qwen3.5-4b \
  --lens artifacts/lenses/qwen3.5-4b/jacobian_lens.pt \
  --max-rows 2

uv run prompt-analysis attribute \
  --model .cache/models/qwen3.5-4b \
  --max-new-tokens 64

uv run prompt-analysis validate-attribution \
  --attribution artifacts/prompt_analysis/qwen3.5-4b/generated_attribution/generated_token_attribution.jsonl \
  --model .cache/models/qwen3.5-4b

uv run prompt-analysis visualize \
  --attribution artifacts/prompt_analysis/qwen3.5-4b/generated_attribution/generated_token_attribution.jsonl \
  --tokenizer .cache/models/qwen3.5-4b

uv run prompt-analysis serve \
  --model .cache/models/qwen3.5-4b --host 0.0.0.0 --port 8322
```

也可使用 `scripts/run_prompt_analysis.sh` 與 `scripts/visualize_prompt_analysis.sh`；前者預設使用 tmux session `prompt_analysis`，需要前景執行時設定 `RUN_IN_TMUX=0`。runner 可透過 `MODEL`、`LENS`、`RUN_ROOT`、`INPUT_CSV` 等環境變數重用於其他 ticker-style prompt table。`--max-rows 2` 與 `--no-input-attribution` 適合快速 integration check。

### EDGAR staging

```bash
uv run prepare-edgar-8k clean \
  --input ../10-k/edgar-crawler/datasets/EXTRACTED_FILINGS/8-K \
  --output artifacts/edgar_8k/cleaned \
  --max-files 100
uv run prepare-edgar-8k validate --input artifacts/edgar_8k/cleaned
```

cleaner 不修改外部 crawler dataset，而是輸出 `filings.jsonl`、`sections.jsonl`、`quality_report.json` 與 `manifest.json`；分析文字與移除的 boilerplate spans 都需保留可追溯資訊。

### Entity-only counterfactual data

```bash
uv run prepare-counterfactual-data entities
uv run prepare-counterfactual-data sample --count 500 --seed 20260730
uv run --extra extraction prepare-counterfactual-data annotate
uv run prepare-counterfactual-data review-bundle --count 200
uv run prepare-counterfactual-data promote --review <review.jsonl>
uv run prepare-counterfactual-data build-pairs
uv run prepare-counterfactual-data render --model .cache/models/llama-3.2-1b-instruct
uv run prepare-counterfactual-data validate
```

流程順序為 `entities → sample → annotate → review-bundle → promote → build-pairs → render → validate`。資料在通過至少 200 筆人工 review 及 precision/recall、grounding、semantic outcome、identity leakage gates 前，不得標記為 validated。

## 架構與責任邊界

- `llm_bias/core/`：共用 model loading、chat/raw prompt formatting、token alignment 與 lens artifact metadata/validation；只放與研究語意無關的共用基礎設施。
- `llm_bias/lens_fitting/`：calibration corpus、fit/evaluation/promotion 與 checkpoint recovery；不 import 任一 experiment package。
- `llm_bias/counterfactual_patching/`：`Pair` serialization、source/target residual recording、normalized span mapping、patched logits、transfer metrics、summary/figures 與 FastAPI dashboard。
- `llm_bias/prompt_analysis/`：CSV prompt readout、完整 vocabulary aggregation、generated-token gradient attribution、zero-vector ablation validation、plots 與 standalone/interactive dashboard。
- `llm_bias/edgar_preparation/`：extracted 8-K 的 schema/quality cleaning、SEC item taxonomy 與 manifest validation。
- `llm_bias/counterfactual_data/`：point-in-time CIK/name/SIC history、deterministic event sampling、local annotation、review promotion、entity contrasts、model-specific token rendering 與 validation。
- `llm_bias/static/`：counterfactual dashboard 的無 build-step frontend，呼叫 `/api/info`、`/api/pairs`、`/api/counterfactual`。HTML grid backend 必須保留完整 top-k，前端只負責顯示 top-1 與 hover 詳細資料。

兩個 experiment package `counterfactual_patching` 與 `prompt_analysis` 不可互相 import；共同功能應先確認不含研究語意後才放入 `core/`。`lens_fitting` 也不得由 experiment CLI 隱式啟動。

## 研究語意與資料保存約束

- `Pair` 必須保留 entity token start/end 與完整 token-id span，並支援舊 single-token pair。
- 不同長度 span 使用 normalized span-internal token centers 的 nearest mapping；不得插入/刪除 sequence token，也不得合成 activation。batch 結果需保留 source span、target span、position mapping 與 mapping strategy。
- source/target prompt 可有不同 token 長度；answer logits 讀各自 final position，control patch 使用各自最後一個非-entity position。interactive dashboard 目前對不同長度 pair 應明確回傳 validation error。
- 不保存完整 raw activations；只輸出 compact top-k、rank、統計量、token IDs/text、probabilities 與 provenance。
- bias-specific pairs 必須共用相同 headline/context 與 expected outcome；分開報告 `real_vs_real`、`real_vs_anonymous`、`real_vs_synthetic`、`synthetic_vs_synthetic`，使用固定 outcome options 的 logit margin，而不是 factual answer-transfer 公式。
- 不要把不同 token 的 top-1 probability 差直接當成 causal effect；使用固定答案 token probability、logit margin 或 normalized transfer。
- prompt readout 的 aggregate 必須先平均每個 condition 的完整 vocabulary softmax，再選 top-k，不可只平均每個 prompt 的 top-k entries。Attribution 是 local first-order sensitivity，不是 attention map 或 standalone causal claim。

更具體的目錄規則仍由各層 `AGENTS.md` 補充，尤其是 root、`llm_bias/`、`tests/` 與 `llm_bias/static/` 中的檔案。
