# Repository guidance

## 環境需求

- 使用 `uv` 管理 Python 虛擬環境。
- 新增 Python 套件使用 `uv add`。
- 這份文件適用於整個 repository；子目錄中的 `AGENTS.md` 會補充更具體的模組規則。

## Repository scope

本 repo 實驗 entity-level bias 如何在 decoder LLM 的 residual stream 中形成，以及把
source entity 的 activation 替換成 target entity 後，是否會改變最終答案分布。

目前的實驗圍繞 `data/baseline/` baseline 資料集展開。主要 workflow 的完整操作
文件如下：

- [Baseline trial plan prompts](docs/baseline-trial-plan-prompts.md)
- [Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md)
- [Prompt-analysis reproducibility](docs/prompt-analysis-reproducibility.md)
- [Interactive prompt-lens dashboard](docs/interactive-prompt-lens-dashboard.md)
- [J-space sector intervention: methods, calibration, results](docs/jspace-sector-intervention-interim.md)
- [Entity-bias proposal and roadmap](docs/proposal/README.md)

Counterfactual、synthetic 與 10-K 線的實驗程式已移至 [`archive/`](archive/README.md)
（frozen、可還原）；其操作文件在 `docs/archive/`。

J-space evaluation 位於 [`docs/j-space-evaluation.md`](docs/j-space-evaluation.md)。它是
從 Jacobian-lens working-space literature 延伸出的 optional、proposed、non-runnable
auxiliary preflight，只評估 synthetic task-local J-space-candidate evidence；它不建立
global workspace 結論、不 gate entity-bias milestones，也不取代每個模型自己的
entity-only causal protocol。可執行的 J-space sector intervention 實驗線是
`llm_bias/jspace_intervention/`（CLI `jspace-intervention`），方法、calibration 與
interim 結果見 [J-space sector intervention](docs/jspace-sector-intervention-interim.md)。
它屬於每個模型自己的 entity-only causal protocol，與上述 synthetic preflight 分開。

## Shared experiment workflow contract

The shared experiment workflow is `prepare → forward → analyze → finalize`. Reuse the four core subpackages—`llm_bias/core/prompt_input`, `llm_bias/core/inference`, `llm_bias/core/analysis`, and `llm_bias/core/artifacts`—for cross-experiment workflow mechanics. Experiment packages must not sink shared prompt preparation, model forward execution, common analysis, artifact serialization, manifest/provenance, or lifecycle finalization into local copies; keep research-specific semantics and presentation in the owning experiment package. Compatibility rules are mandatory: preserve existing public CLI/API behavior and artifact schemas unless a canonical workflow document explicitly versions a change; experiment packages (`baseline_trial`, `jspace_intervention`, `prompt_analysis`) must not import each other, and shared infrastructure must not import any experiment package. Experiment workflows consume an existing validated canonical lens and must not fit, mutate, or replace one implicitly. Never persist raw activations, residuals, hidden states, gradients, Jacobians, or KV caches; emit only compact derived outputs with provenance.

## Research semantic boundaries

- 不保存完整 raw activations；只輸出 compact top-k、rank、統計量、token IDs/text、probabilities 與 provenance。
- 不要把不同 token 的 top-1 probability 差直接當成 causal effect；使用固定答案 token probability、logit margin 或明確定義的 normalized transfer。
- prompt readout 的 aggregate 必須先平均每個 condition 的完整 vocabulary softmax，再選 top-k。Attribution 是 local first-order sensitivity，不是 attention map 或 standalone causal claim。
- Jacobian lens 是 transported representation readout，不是 chain-of-thought、離散 reasoning path 或 standalone causal evidence。
- Counterfactual 線的 Pair/span-mapping/control-patch/bias-specific pair 研究語義隨程式一併移至 [`archive/README.md`](archive/README.md)。

## 依賴與外部 checkout

- Python 依賴與 lockfile 一律由 `uv` 管理；新增套件使用 `uv add`。
- `third_party/jacobian-lens` 與 `third_party/jspace-viz` 是 editable workspace members，但整個 `third_party/` 被 `.gitignore` 忽略。
- 新環境請依照 `README.md` clone 兩個外部 repo 後再執行 `uv sync`。
- 不要把模型權重、patch 結果、lens binary 或第三方 checkout 加入 root Git repository。這些內容位於 `.cache/`、`artifacts/`、`third_party/`。
- 每個 model 只有一個 active、完整逐層的 canonical lens：`artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt`。Partial/stride 實驗與 fitting checkpoint 必須放在 `artifacts/archive/<model-slug>/jacobian-lens/checkpoints/`，不可混入 active model folder。

## 常用驗證

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
node --check llm_bias/static/prompt_readout.js
node --check llm_bias/static/attribution_dashboard.js
```

測試應優先使用 deterministic unit tests、fake model、monkeypatch 與 temporary directories；不要為一般 unit test 載入大型 checkpoint。模型/GPU inference 應明確視為 smoke 或 integration test。
