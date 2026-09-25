# Repository guidance

本 repo 只做一條研究線：**concept-cone steering**。它在 decoder LLM 的 residual stream
注入由公司好惡差值建構的方向（DIM 或多維 cone），觀察投資 buy/sell 判定是否翻轉。
研究內容、結論與狀態都在 [`docs/concept-cone-steering/`](docs/concept-cone-steering/)；
先讀 `claim-to-evidence.md`，再依需要讀各版本資料夾的 `status.md`／`proposal.md`。

2026-09-25 以前的其他研究線（Jacobian-lens readout、J-space intervention、entity cell、
investment dial 等）已從工作樹刪除，完整狀態保存在 git tag `pre-cleanup`。除非使用者
要求，不要從該 tag 取回舊 code 或舊文件。

## Layout

- `scripts/`：實驗入口，每支獨立用 `uv run python scripts/<name>.py` 執行。
  - `probe_concept_cone.py`、`probe_dim_steering.py`、`probe_operator_comparison.py`：steering 主實驗。
  - `reparse_concept_cone_decisions.py`、`summarize_concept_cone_decisions.py`：決策解析與彙整。
  - `plot_*.py`：`docs/concept-cone-steering/*/figures/` 的圖。
  - `balanced_evidence_gap*.py`、`downloads/run_*.sh`：產生 steering 用的
    `balanced-evidence-gap-phase2/runs/phase2b-*/pairs/directions.json`（上游）。
  - `entity_to_dial_heldout_transfer.py`：產生 200 家 construction cohort（上游）。
- `llm_bias/core/`：模型載入（經 `jlens.from_hf` 包裝）、residual hook／intervention、
  generation、continuation scoring、run manifest 與 artifact path。
- `llm_bias/entity_to_dial/`、`llm_bias/balanced_evidence_gap/`：上游 prompt template、
  span 與 direction 產生邏輯；steering scripts 直接 import 其中的 template 與 margin 函式。
- `docs/balanced-evidence-gap/details/`、`docs/entity-to-dial/details/`：上游協議，已凍結。
  其中指向已刪除文件的連結是歷史引用，不要修。
- `data/`（輸入）、`artifacts/<model-slug>/...`（run 輸出）、`.cache/`（模型）都不進 git。
- `third_party/jacobian-lens` 是 `jlens` 的 editable workspace member（不進 git），
  依 `README.md` clone 後再 `uv sync`。

## 研究規則

- 固定答案 token 的 margin（`log p(buy) − log p(sell)`）只代表讀出改變；宣稱「翻轉決策」
  必須附真實 greedy generation 的 decision-flip 率與 parse rate。
- 不保存 raw activations、residuals、gradients 或 KV cache；只輸出 compact 統計與 provenance。
- 已完成的 run、數值與凍結協議不回頭改寫。設計（direction 來源、主要指標、controls、gate）
  改變時，在 `docs/concept-cone-steering/` 開新的版本資料夾（`proposal.md` + `status.md`），
  不把新結果回填到舊版本。
- 新 run 寫到 `artifacts/<model-slug>/concept-cone-steering/runs/<run-id>/`，不覆蓋舊 run。

## Working rules

- 只改任務範圍內的東西；開始前看 `git status`。
- 新共用邏輯放 `llm_bias/core/`；一次性實驗邏輯留在 script 裡即可，不要預先抽象化。
- 文件裡的命令、path 與狀態必須對得上 code 與 artifact；不確定就標 proposed。
- 用 `uv sync` 建環境、`uv add` 加套件。

## Verification

```bash
uv lock --check
uv run pytest -q
```

Unit test 用 fake model 與 temporary directory，不載入真 checkpoint；GPU run 另外當 smoke 執行。
