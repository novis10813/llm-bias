# `llm_bias/` scope

這份文件說明主實驗 Python package 的責任邊界；上層規則見 repository root
的 `AGENTS.md` 與 `CLAUDE.md`，其中完全相同的 **Shared experiment workflow contract**
定義 canonical `prepare → forward → analyze → finalize` 流程與跨 package 規則。
本 scope 文件只補充 `llm_bias/` 內的 package ownership，不重複 canonical workflow
commands 或取代各 workflow 文件。

## Package 邊界

- `core/`：承接 shared experiment workflow mechanics，以及 model loading、
  continuation scoring、artifact identity 與 validated lens loading。修改時先讀
  [`core/AGENTS.md`](core/AGENTS.md)；完整 ownership 與 compatibility map 見
  [`../docs/shared-experiment-core.md`](../docs/shared-experiment-core.md)。
- `lens_fitting/` / `lens_install/`：lens fitting 與安裝；不可 import 任一 experiment。
- `baseline_trial/`：擁有 baseline trial prompt 準備、forward 與 artifact pipeline。
- `jspace_intervention/`：擁有 sector coordinate intervention、valence readout、J-space
  token V1/V2、activation patching，以及 sector/context follow-up A/B/C。修改時先讀
  [`jspace_intervention/AGENTS.md`](jspace_intervention/AGENTS.md)；各 protocol、artifact
  identity 與 evidence status 由該檔連結的 experiment proposal/report 定義。
- `prompt_analysis/`：擁有 CSV prompt readout、generated-token attribution、
  attribution validation 與結果視覺化。
- `span_sensitivity/`：擁有單一產業 identity-header conditions、固定 Buy/Sell
  continuation margin 與 ticker-clustered paired analysis；workflow 見
  [`../docs/span-sensitivity/proposal.md`](../docs/span-sensitivity/proposal.md)。
- `financial_soundness/`：擁有財務短句探索定位與獨立因果驗證；CLI 為
  `financial-soundness`，協議見 [定位提案](../docs/financial-soundness-localization/proposal.md)
  與 [因果驗證提案](../docs/financial-soundness-causal-validation/proposal.md)。
- `investment_dial/`：擁有 arXiv:2608.22852 的 Qwen3.5-4B 本地方法復現；
  `investment-dial` CLI 分開提供工程檢查、梯度篩選、A/B 校準與獨立公司評估。
  使用 JSON decision/reason、全 token 加法；不得把合成工程輸入當作 baseline 結果。
- `selective_intervention/`：擁有 V1 推論期 entity-difference 子空間移除、對照與
  gate 評估；operator 為 `scripts/selective_intervention_v1.py`，協議見
  [V1 proposal](../docs/selective-intervention/details/proposal-v1.md)。
- `entity_cell/`：擁有 entity-cell localization、attention attribution 與
  suppression intervention 工作流；CLI 入口為 `entity-cell`。
- `entity_concept_decision/`：Phase 1 基礎工具與 development runner；無 CLI；材料驗證、概念方向 fitting 與 caller-owned `prepare → forward → analyze` development workflow 由 [proposal](../docs/entity-concept-decision/proposal.md) 與 [implementation](../docs/entity-concept-decision/details/implementation-phase1-development.md) 定義。runner 接受任意概念集合（`concept_ids`）並做通用 stance 正交化；`development_materials` 含 round-1（C+S）與 round-2（表格格式，任意概念）兩種 builder；`layer_scan` 為多層完整空間概念-vs-stance 分離性掃描（development only）。runner 只接受已載入模型、tokenizer、CPU basis 與明確 development provenance，不是 formal audit pipeline。
- `counterfactual_patching/`、`synthetic_entity_bias/`、`ten_k_change_data/`、
  `edgar_preparation/`：已隨程式移至 `archive/llm_bias/`（frozen），操作文件在
  `docs/archive/`。

Shared infrastructure 不可 import 任一 experiment package。`baseline_trial` 目前為
legacy compatibility 直接重用部分 `prompt_analysis` modules；不要擴大這個例外，新增
跨實驗共用能力應移入 `core/`。其餘 experiment packages 不可互相 import。
CLI 入口是 `jacobian-lens`、`prompt-analysis`、`baseline-trial`、
`jspace-intervention`、`span-sensitivity`、`entity-cell`、`financial-soundness` 與
`investment-dial`；experiment CLI
不可自行 fitting lens。

## CLI 設計原則

- **頂層命令維持一個 Package 一個**：保持 `pyproject.toml` 中的 `[project.scripts]`
  精簡（如 `entity-cell`、`jspace-intervention`），不為單一子實驗註冊全局命令，避免環境污染。
- **子命令依研究階段（Phase / Milestone）解耦分立**：每個獨立實驗階段必須有專屬的
  Subcommand（如 `run-localization`、`run-attribution`、`run-intervention`，或如
  `jspace-intervention` 的 `run-token-screen`、`run-outcome-flip`）。
- **封閉獨立的參數空間**：各子命令只宣告自己需要的參數，必要參數在 `argparse` 層級設為
  `required=True`，不可在單一命令中混裝跨階段參數。
- **嚴禁巨石 Dispatcher 反模式**：嚴禁把定位、歸因、干預等多個不同階段硬塞在同一個通用
  `run` 裡，嚴禁在程式碼內部使用 `if any(stage.startswith(...)):` 等猜測使用者意圖的脆弱分發邏輯。

## Instruction Index

以下只列 `llm_bias/` 直接子目錄中的 instruction files：

- [`core/AGENTS.md`](core/AGENTS.md)：shared workflow mechanics、compatibility facades、
  artifact 與 lens boundaries。
- [`jspace_intervention/AGENTS.md`](jspace_intervention/AGENTS.md)：J-space V1/V2、activation
  patching、sector/context A/B/C 的 package 分工與局部驗證。

Experiment package 只有在出現無法由 canonical workflow 文件覆蓋的局部架構時才新增
`AGENTS.md`。新增後，只更新本節，不在上層枚舉更深入口。

## 修改與驗證原則

- 維持 package ownership；不要為單一 experiment 把 research semantics 下沉到 `core/`。
- 修改 intervention/hook 語意或 artifact schema 時，至少執行 `uv run pytest -q`
  並補 regression test。
- 不要儲存完整 raw activations；只輸出 compact top-k/rank、scalar dose
  diagnostics 與 provenance。
- 模型載入與 GPU inference 屬於 smoke/integration，不得讓 `uv run pytest -q`
  依賴特定 GPU。
