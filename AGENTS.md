# Repository guidance

這份文件是 repository 的 AI 協作入口。先讀本檔；工作落在子目錄時，再讀該目錄的
`AGENTS.md`。若祖先目錄有多份 instruction files，越靠近目標檔案者優先；下層只
補充局部規則，未覆蓋的規則繼續沿用上層。詳細分工與維護方式見
[Documentation and instruction system](docs/documentation-system.md)。

## Repository Scope

本 repo 研究 decoder LLM 的 entity-sensitive 與 sector-sensitive representation、
Jacobian-lens transported readout，以及 residual/J-space intervention 對固定答案分布
的影響。Active experiments 使用 `data/baseline/`，目前包含 baseline trial、prompt
analysis、span sensitivity 與 J-space intervention。

本 repo 不是 production trading system，也不把 lens readout 當成 chain-of-thought、
離散 reasoning path 或單獨的 causal proof。Counterfactual、synthetic 與 10-K 線已
frozen；還原方式見 [`archive/README.md`](archive/README.md)。

主要 workflow 的完整操作文件如下：

- [Baseline trial plan prompts](docs/baseline-trial-plan-prompts.md)
- [Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md)
- [Prompt-analysis reproducibility](docs/prompt-analysis-reproducibility.md)
- [Interactive prompt-lens dashboard](docs/interactive-prompt-lens-dashboard.md)
- [J-space sector intervention: methods, calibration, results](docs/jspace-sector-intervention-interim.md)
- [J-space valence vocabulary readout](docs/jspace-valence-vocabulary-readout.md)
- [J-space token experiment versions](docs/jspace-token-causal-screen.md)
  - [V1: representation-nominated token directions](docs/jspace-token-causal-screen-v1.md)（completed；shortlist empty）
  - [V2: outcome-conditioned Buy/Sell decision flips](docs/jspace-outcome-direction-flip-v2.md)（Draft 1 已實作；第一次正式 pipeline 的 verdict 為 `success=false`）
  - [V2 outcome direction 幾何投影分解](docs/jspace-outcome-direction-geometry.md)（輔助診斷；描述性幾何，非 causal）
- [Shared experiment core](docs/shared-experiment-core.md)
- [Technology header-span sensitivity](docs/technology-header-span-sensitivity.md)
- [Entity-bias proposal and roadmap](docs/proposal/README.md)
- [Artifact identity and run manifest contract](docs/artifact-contract.md)

J-space evaluation 位於 [`docs/j-space-evaluation.md`](docs/j-space-evaluation.md)。它是
optional、proposed、non-runnable auxiliary preflight，只評估 synthetic task-local
J-space-candidate evidence；它不建立 global workspace 結論，也不 gate active
experiment milestones。可執行的 sector intervention 位於
`llm_bias/jspace_intervention/`（CLI `jspace-intervention`），與該 synthetic preflight
及 archived entity-only patching protocol 分開。

## Shared experiment workflow contract

The shared experiment workflow is `prepare → forward → analyze → finalize`. Reuse the four core subpackages—`llm_bias/core/prompt_input`, `llm_bias/core/inference`, `llm_bias/core/analysis`, and `llm_bias/core/artifacts`—for cross-experiment workflow mechanics. Experiment packages must not sink shared prompt preparation, model forward execution, common analysis, artifact serialization, manifest/provenance, or lifecycle finalization into local copies; keep research-specific semantics and presentation in the owning experiment package. Compatibility rules are mandatory: preserve existing public CLI/API behavior and artifact schemas unless a canonical workflow document explicitly versions a change; experiment packages (`baseline_trial`, `jspace_intervention`, `prompt_analysis`, `span_sensitivity`) must not import each other, and shared infrastructure must not import any experiment package. Experiment workflows consume an existing validated canonical lens and must not fit, mutate, or replace one implicitly. Never persist raw activations, residuals, hidden states, gradients, Jacobians, or KV caches; emit only compact derived outputs with provenance.

目前 `baseline_trial` 仍直接重用部分 `prompt_analysis` modules，屬於待收斂的 legacy
compatibility exception；不要新增同類依賴。新 shared mechanics 必須放進 `core/`。

## Research semantic boundaries

- 不保存完整 raw activations；只輸出 compact top-k、rank、統計量、token IDs/text、probabilities 與 provenance。
- 不要把不同 token 的 top-1 probability 差直接當成 causal effect；使用固定答案 token probability、logit margin 或明確定義的 normalized transfer。
- prompt readout 的 aggregate 必須先平均每個 condition 的完整 vocabulary softmax，再選 top-k。Attribution 是 local first-order sensitivity，不是 attention map 或 standalone causal claim。
- Jacobian lens 是 transported representation readout，不是 chain-of-thought、離散 reasoning path 或 standalone causal evidence。
- Counterfactual 線的 Pair/span-mapping/control-patch/bias-specific pair 研究語義隨程式一併移至 [`archive/README.md`](archive/README.md)。

## 設定與檔案放置

- Python 3.13 與 workspace 依賴由 `.python-version`、`pyproject.toml`、`uv.lock`
  定義；使用 `uv sync` 建環境，新增套件使用 `uv add`。
- Pinned lens registry 放在 `config/pretrained_lenses.json`；修改 model identity、revision
  或 SHA-256 時，依 [Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md)
  重新驗證。
- 可追蹤的詳細政策與 workflow 放 `docs/`；script ownership map 見
  [Research scripts reference](docs/research-scripts.md)。
- Input/provenance 放 `data/`；模型與 Hugging Face cache 放 `.cache/`；run outputs 與
  lens artifacts 放 `artifacts/`。這些大型或 generated 內容遵守 `.gitignore`，不要
  加入 root Git。
- `.pi/` 保存本地 agent runtime，`graphify-out/` 保存 generated repository diagrams，
  `.worktrees/` 保存本地 Git worktrees；它們不是 main source tree，也不要加入 root Git。
- `third_party/jacobian-lens` 與 `third_party/jspace-viz` 是 editable workspace members，但整個 `third_party/` 被 `.gitignore` 忽略。
- 新環境請依照 `README.md` clone 兩個外部 repo 後再執行 `uv sync`。
- 每個 model 只有一個 active、完整逐層的 canonical lens：
  `artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt`。一般 partial/stride fitting
  checkpoint 放 `artifacts/archive/<model-slug>/jacobian-lens/checkpoints/`；受控
  candidate-selection workflow 可依
  [Qwen Jacobian-lens selection](docs/qwen-jacobian-lens-selection.md) 使用
  `artifacts/<model-slug>/jacobian-lens/candidates/` 的 candidate-adjacent digest
  checkpoints，但不得把 candidate 當 active lens。

## Instruction Index

以下只列 root 直接子目錄中的 instruction files：

- [`archive/AGENTS.md`](archive/AGENTS.md)：frozen code、還原邊界與 archive 內入口。
- [`docs/AGENTS.md`](docs/AGENTS.md)：canonical 文件分類、引用與狀態維護。
- [`llm_bias/AGENTS.md`](llm_bias/AGENTS.md)：active Python packages 的 ownership 與局部驗證。
- [`scripts/AGENTS.md`](scripts/AGENTS.md)：research operators、diagnostics 與 renderers 的慣例。
- [`tests/AGENTS.md`](tests/AGENTS.md)：regression test 地圖與 fake-model 測試規則。

目前 `config/` 與 `data/` 是未來候選：registry schema 或 dataset-specific provenance
規則變得無法用一兩句覆蓋時，再在該目錄新增 `AGENTS.md`。其他目錄也採同一門檻。
新增後，只更新最近一層祖先 `AGENTS.md` 的 Instruction Index，不在 root 枚舉更深層
檔案。

## Working Rules

- 維持既有 package ownership、public CLI/API 與 artifact schema；需要版本變更時，
  先更新對應 canonical workflow 文件。
- 只改任務要求的範圍，不做順手重構，也不覆蓋不相干的 dirty changes。開始前先看
  `git status` 與相關 diff。
- 文件中的命令、path、run 狀態與架構描述要能對上 code、config、tests 或 artifact
  provenance；不確定的內容標成 proposed 或 note。
- 實驗術語的定義必須在文件中：提及概念時只使用 repo 文件已定義的英文術語（原文
  照用）；沒有術語的概念用完整中文描述句。需要為新概念命名時，先將定義寫入 owning
  experiment 的 canonical workflow 文件再使用；不得引入任何文件中查無出處的英文複合詞，
  也不得在回覆中直接使用未定義的新詞。
- 新增詳細規則時先更新 `docs/`，再讓 `AGENTS.md` 連結該文件，避免兩處維護完整副本。
- 討論 J-space token 實驗時必須標明 V1 或 V2。若 direction source、primary outcome、
  controls 或 gate 改變，依
  [experiment versioning](docs/documentation-system.md#experiment-versioning) 建新版本，
  不把新設計回填成舊版本結果。

## Verification

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
node --check llm_bias/static/prompt_readout.js
node --check llm_bias/static/attribution_dashboard.js
```

測試應優先使用 deterministic unit tests、fake model、monkeypatch 與 temporary directories；不要為一般 unit test 載入大型 checkpoint。模型/GPU inference 應明確視為 smoke 或 integration test。
