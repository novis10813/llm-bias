# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## 專案定位

本 repository 研究 entity-sensitive representation 是否會在 decoder language
model 的 residual stream 中形成，以及將 source entity activation 替換為
target entity activation 是否會改變答案分布。主要 workflow 是：

- `baseline-trial`：把 `data/baseline/` 的 trial-plan CSV 跑過 prompt-analysis 的
  `prepare → forward → analyze → finalize` stages，並加上 lens-forward 逐層
  readout 階段。
- `prompt-analysis`：對 CSV prompts 執行逐層 Jacobian-lens readout、uncertainty、
  generated-token attribution、ablation validation 與結果視覺化；也是
  `baseline-trial` 的執行基礎。
- `jacobian-lens fit`：獨立的 lens fitting 工具；experiment workflow 只消費
  已存在的 lens，不可自行 fitting。

Counterfactual、synthetic 與 10-K 線的實驗程式（含各自的 CLI）已移至
`archive/`（frozen、可還原），見 `archive/README.md`。

`jlens` 的 Jacobian readout 是 transported representation，不是模型的
chain-of-thought、離散 reasoning path、attention map 或 standalone causal
proof。

研究設計與 execution roadmap 位於：

- [`docs/proposal/entity-bias-research-proposal.md`](docs/proposal/entity-bias-research-proposal.md)
- [`docs/proposal/entity-bias-roadmap.md`](docs/proposal/entity-bias-roadmap.md)

J-space evaluation 是從 Jacobian-lens working-space literature 延伸出的
**optional、proposed、non-runnable auxiliary preflight**，只評估 synthetic
task-local J-space-candidate evidence；它不建立 global workspace 結論、不 gate
entity-bias milestones，也不取代每個模型自己的 entity-only causal protocol。
文件位於 [`docs/j-space-evaluation.md`](docs/j-space-evaluation.md)。

## Operational documentation

完整 CLI、參數、artifact、schema、runner 與 dashboard 操作只維護在各自的
canonical workflow 文件；不要在本文件複製完整 command blocks。

- [Baseline trial plan prompts](docs/baseline-trial-plan-prompts.md)
- [Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md)
- [Prompt-analysis reproducibility](docs/prompt-analysis-reproducibility.md)
- [Interactive prompt-lens dashboard](docs/interactive-prompt-lens-dashboard.md)
- [Research proposals and roadmap](docs/proposal/README.md)

已 archive 實驗的操作文件在 `docs/archive/`。

README 保留 fresh-checkout setup 與 public quickstart；不要把 future
`jspace-eval` commands 當成目前可執行的 entry point。所有正式 entry points
都定義在 `pyproject.toml`。

## 開發環境與依賴

- Python 版本為 3.13（見 `.python-version`），Python 依賴與 lockfile 一律由
  `uv` 管理；新增套件使用 `uv add`。
- root project 以 editable workspace member 使用
  `third_party/jacobian-lens/` 與 `third_party/jspace-viz/`。這些 checkout 被
  `.gitignore` 忽略；新 checkout 請依照 `README.md` 恢復後再執行 `uv sync`。
- 模型、lens binary、實驗輸出與第三方 checkout 不要加入 root Git；分別使用
  `.cache/`、`artifacts/` 與 `third_party/`。
- 每個模型只能有一個 active、完整逐層的 canonical lens：
  `artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt`。partial/stride fitting 與
  checkpoint 放在 `artifacts/archive/<model-slug>/jacobian-lens/checkpoints/`，不要混入
  active model directory。
- Qwen3.5-4B 使用不同 residual width/layer count 的 model-specific lens。優先
  使用 pinned pretrained registry 中 exact identity 相符且通過完整驗證的 artifact；
  `docs/qwen-jacobian-lens-selection.md` 的本地 bilingual 候選選擇是研究替代流程。
  不要用小型 smoke corpus 覆寫 canonical lens。
- `langextract[openai]` 這個 optional extra 是給已 archive 的
  `prepare-counterfactual-data annotate` 用的（預期本地 OpenAI-compatible
  llama.cpp endpoint）；為了還原時免重新安裝，extra 仍保留在 `pyproject.toml`。

## 架構與責任邊界

- `llm_bias/core/`：共用 model loading、prompt formatting、token alignment 與
  lens artifact metadata/validation；只放與研究語意無關的共用基礎設施。
- `llm_bias/lens_fitting/`：calibration corpus、fit/evaluation/promotion 與
  checkpoint recovery；不 import 任一 experiment package。
- `llm_bias/prompt_analysis/`：CSV prompt readout、完整 vocabulary aggregation、
  generated-token gradient attribution、zero-vector ablation validation、plots
  與 standalone/interactive dashboard。
- `llm_bias/baseline_trial/`：baseline trial-plan CSV 的 stage runner
  （含 lens-forward 逐層 readout 擴展）；建立在 `prompt_analysis` 之上。
- `llm_bias/static/`：prompt-analysis dashboard 的無 build-step frontend。
- `archive/`：frozen 的 counterfactual/synthetic/10-K 實驗程式與 tests；
  還原步驟見 `archive/README.md`。

`baseline_trial` 可以 import `prompt_analysis` 作為執行基礎；其他 experiment
package 不可互相 import。共同功能應先確認不含研究語意後才放入 `core/`。
`lens_fitting` 也不得由 experiment CLI 隱式啟動。

## Shared experiment workflow contract

The shared experiment workflow is `prepare → forward → analyze → finalize`. Reuse the four core subpackages—`llm_bias/core/prompt_input`, `llm_bias/core/inference`, `llm_bias/core/analysis`, and `llm_bias/core/artifacts`—for cross-experiment workflow mechanics whenever those packages exist. Experiment packages must not sink shared prompt preparation, model forward execution, common analysis, artifact serialization, manifest/provenance, or lifecycle finalization into local copies; keep research-specific semantics and presentation in the owning experiment package. Compatibility rules are mandatory: preserve existing public CLI/API behavior and artifact schemas unless a canonical workflow document explicitly versions a change; `counterfactual_patching` and `prompt_analysis` must not import each other, and shared infrastructure must not import either experiment. Experiment workflows consume an existing validated canonical lens and must not fit, mutate, or replace one implicitly. Never persist raw activations, residuals, hidden states, or gradients; emit only compact derived outputs with provenance. These rules apply even before one or more of the four core subpackages has been created.

## 研究語意與資料保存約束

- 不保存完整 raw activations；只輸出 compact top-k、rank、統計量、token IDs/text、
  probabilities 與 provenance。
- 不要把不同 token 的 top-1 probability 差直接當成 causal effect。
- prompt readout 的 aggregate 必須先平均每個 condition 的完整 vocabulary softmax，
  再選 top-k。Attribution 是 local first-order sensitivity，不是 attention map
  或 standalone causal claim。

Counterfactual 線的 Pair span、normalized span mapping、control patch 與
bias-specific pair 協議等研究語義隨程式一併移至 `archive/README.md`。

## 常用驗證

```bash
uv sync
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
node --check llm_bias/static/prompt_readout.js
```

單一 deterministic unit test 範例：

```bash
uv run pytest -q \
  tests/test_prompt_input.py::test_token_span_and_record_cover_overlapping_tokens
```

測試應優先使用 deterministic unit tests、fake model、monkeypatch 與 temporary
directories；不要為一般 unit test 載入大型 checkpoint。模型/GPU inference 應明確
視為 smoke 或 integration test。

若修改 static asset，除了 JavaScript syntax check 與 pytest，也要同步提高對應
HTML 的 query-string 版本，避免瀏覽器使用舊 asset。
