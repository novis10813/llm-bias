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
  dose-matched controls、position/direction controls 與 analysis。
- `prompt_analysis/`：擁有 CSV prompt readout、generated-token attribution、
  attribution validation 與結果視覺化。
- `counterfactual_patching/`、`synthetic_entity_bias/`、`ten_k_change_data/`、
  `edgar_preparation/`：已隨程式移至 `archive/llm_bias/`（frozen），操作文件在
  `docs/archive/`。

三個 experiment package 不可互相 import；shared infrastructure 不可 import 任一
experiment package。共同能力必須先確認確實與研究語意無關，才可放進 `core/`。
四個 CLI 入口分別是 `jacobian-lens`、`prompt-analysis`、`baseline-trial` 與
`jspace-intervention`；experiment CLI 不可自行 fitting lens。

## 修改與驗證原則

- 小改用精確文字替換（edit tool），避免整檔重寫。
- 修改 intervention/hook 語意或 artifact schema 時，至少執行 `uv run pytest -q`
  並補 regression test。
- 不要儲存完整 raw activations；只輸出 compact top-k/rank、scalar dose
  diagnostics 與 provenance。
- 模型載入與 GPU inference 屬於 smoke/integration，不得讓 `uv run pytest -q`
  依賴特定 GPU。
