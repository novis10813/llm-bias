# Repository guidance

## 環境需求

- 使用 `uv` 管理 Python 虛擬環境。
- 新增 Python 套件使用 `uv add`。
- 這份文件適用於整個 repository；子目錄中的 `AGENTS.md` 會補充更具體的模組規則。

## Repository scope

本 repo 實驗 entity-level bias 如何在 decoder LLM 的 residual stream 中形成，以及把
source entity 的 activation 替換成 target entity 後，是否會改變最終答案分布。

主要 workflow 的完整操作文件如下：

- [Counterfactual patching](docs/counterfactual-patching.md)
- [8-K counterfactual entity dataset](docs/counterfactual-dataset-generation.md)
- [EDGAR 8-K preparation](docs/edgar-8k-preparation.md)
- [Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md)
- [Prompt-analysis reproducibility](docs/prompt-analysis-reproducibility.md)
- [Interactive prompt-lens dashboard](docs/interactive-prompt-lens-dashboard.md)
- [Entity-bias proposal and roadmap](docs/proposal/README.md)

J-space evaluation 位於 [`docs/j-space-evaluation.md`](docs/j-space-evaluation.md)。它是
從 Jacobian-lens working-space literature 延伸出的 optional、proposed、non-runnable
auxiliary preflight，只評估 synthetic task-local J-space-candidate evidence；它不建立
global workspace 結論、不 gate entity-bias milestones，也不取代每個模型自己的
entity-only causal protocol。

## Research semantic boundaries

- `Pair` 必須保留 entity token start/end 與完整 token-id span，並支援舊 single-token pair。
- 不同長度 span 使用 normalized span-internal token centers 的 nearest mapping；不得插入/刪除 sequence token，也不得合成 activation。
- source/target prompt 可有不同 token 長度；batch answer logits 讀各自 final position，control patch 使用各自最後一個非-entity position。
- batch 結果必須保留 source span、target span、position mapping 與 mapping strategy。
- 不保存完整 raw activations；只輸出 compact top-k、rank、統計量、token IDs/text、probabilities 與 provenance。
- bias-specific pairs 必須共用相同 headline/context 與 expected outcome，分開報告 `real_vs_real`、`real_vs_anonymous`、`real_vs_synthetic`、`synthetic_vs_synthetic`，使用固定 outcome options 的 logit margin，而不是 factual answer-transfer 公式。
- 不要把不同 token 的 top-1 probability 差直接當成 causal effect；使用固定答案 token probability、logit margin 或明確定義的 normalized transfer。
- prompt readout 的 aggregate 必須先平均每個 condition 的完整 vocabulary softmax，再選 top-k。Attribution 是 local first-order sensitivity，不是 attention map 或 standalone causal claim。
- Jacobian lens 是 transported representation readout，不是 chain-of-thought、離散 reasoning path 或 standalone causal evidence。

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
node --check llm_bias/static/counterfactual.js
```

測試應優先使用 deterministic unit tests、fake model、monkeypatch 與 temporary directories；不要為一般 unit test 載入大型 checkpoint。模型/GPU inference 應明確視為 smoke 或 integration test。
