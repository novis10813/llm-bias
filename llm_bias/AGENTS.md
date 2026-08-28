# `llm_bias/` scope

這份文件說明主實驗 Python package 的責任邊界；上層規則見 repository root
的 `AGENTS.md` 與 `CLAUDE.md`，其中完全相同的 **Shared experiment workflow contract**
定義 canonical `prepare → forward → analyze → finalize` 流程與跨 package 規則。
本 scope 文件只補充 `llm_bias/` 內的 package ownership，不重複 canonical workflow
commands 或取代各 workflow 文件。

## Package 邊界

- `core/prompt_input/`、`core/inference/`、`core/analysis/`、`core/artifacts/`：承接
  shared experiment workflow 的 prepare、forward、analyze、finalize mechanics。
- `core/` 其他模組：只放 model loading、prompt formatting、token alignment、
  continuation scoring、artifact paths 與 lens artifact metadata/validation
  等模型無關的共用基礎設施。
- `lens_fitting/` / `lens_install/`：lens fitting 與安裝；不可 import 任一 experiment。
- `baseline_trial/`：擁有 baseline trial prompt 準備、forward 與 artifact pipeline。
- `jspace_intervention/`：擁有 J-space sector 座標 swap/gain intervention、
  dose-matched controls、valence vocabulary readout、V1 token causal screen 與 V2
  outcome-conditioned decision flip（`prepare-outcome-flip-config`、`run-outcome-flip`）
  及 analysis。
  詳細語意見 [`../docs/jspace-sector-intervention-interim.md`](../docs/jspace-sector-intervention-interim.md)、
  [`../docs/jspace-valence-vocabulary-readout.md`](../docs/jspace-valence-vocabulary-readout.md)
  與 [`../docs/jspace-token-causal-screen.md`](../docs/jspace-token-causal-screen.md) 的版本入口。
  V2 使用獨立的 `jspace-outcome-direction-flip` slug 與
  `outcome_flip_*` artifacts（見
  [`../docs/jspace-outcome-direction-flip-v2.md`](../docs/jspace-outcome-direction-flip-v2.md)），
  不可把 V1 CLI/schema 當成 V2 implementation。
- `prompt_analysis/`：擁有 CSV prompt readout、generated-token attribution、
  attribution validation 與結果視覺化。
- `span_sensitivity/`：擁有單一產業 identity-header conditions、固定 Buy/Sell
  continuation margin 與 ticker-clustered paired analysis；workflow 見
  [`../docs/technology-header-span-sensitivity.md`](../docs/technology-header-span-sensitivity.md)。
- `counterfactual_patching/`、`synthetic_entity_bias/`、`ten_k_change_data/`、
  `edgar_preparation/`：已隨程式移至 `archive/llm_bias/`（frozen），操作文件在
  `docs/archive/`。

Shared infrastructure 不可 import 任一 experiment package。`baseline_trial` 目前為
legacy compatibility 直接重用部分 `prompt_analysis` modules；不要擴大這個例外，新增
跨實驗共用能力應移入 `core/`。其餘 experiment packages 不可互相 import。
CLI 入口是 `jacobian-lens`、`prompt-analysis`、`baseline-trial`、
`jspace-intervention` 與 `span-sensitivity`；experiment CLI 不可自行 fitting lens。

## Instruction Index

目前 `llm_bias/` 的直接子目錄沒有 `AGENTS.md`。`core/` 是未來優先候選：shared
infrastructure 若再增加獨立 lifecycle 或 compatibility 規則，於 `core/` 新增
`AGENTS.md` 並只更新本節。Experiment package 只有在出現無法由 canonical workflow
文件覆蓋的局部架構時才新增。

## 修改與驗證原則

- 維持 package ownership；不要為單一 experiment 把 research semantics 下沉到 `core/`。
- 修改 intervention/hook 語意或 artifact schema 時，至少執行 `uv run pytest -q`
  並補 regression test。
- 不要儲存完整 raw activations；只輸出 compact top-k/rank、scalar dose
  diagnostics 與 provenance。
- 模型載入與 GPU inference 屬於 smoke/integration，不得讓 `uv run pytest -q`
  依賴特定 GPU。
