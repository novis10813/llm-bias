# Archived experiment code

這個目錄保存已退出 active execution path 的 frozen 實驗程式與其可還原的資料、測試及
canonical 文件；多數是舊 counterfactual/synthetic/8-K/10-K workflow，另包含
financial-soundness 探索線。Active tree（`llm_bias/`）保留 shared core、baseline
workflow、現行實驗與 lens infra；完整 active research navigation 見
[`../docs/README.md`](../docs/README.md)。

**Frozen：不要在此目錄開發新功能。** 若新的 baseline 實驗需要這裡的實作，
先評估能否重寫成更小的版本放入 active tree；確需沿用時按下方步驟還原。

## 內容

| 路徑 | 原位置 | 說明 |
|---|---|---|
| `llm_bias/counterfactual_patching/` | `llm_bias/counterfactual_patching/` | 8-K counterfactual activation patching（`counterfactual-patching` CLI） |
| `llm_bias/counterfactual_data/` | `llm_bias/counterfactual_data/` | 8-K counterfactual entity pair 生成（`prepare-counterfactual-data` CLI） |
| `llm_bias/edgar_preparation/` | `llm_bias/edgar_preparation/` | EDGAR 8-K 清洗（`prepare-edgar-8k` CLI） |
| `llm_bias/synthetic_entity_bias/` | `llm_bias/synthetic_entity_bias/` | synthetic entity bias 實驗（`synthetic-entity-bias` CLI） |
| `llm_bias/ten_k_change_data/` | `llm_bias/ten_k_change_data/` | 10-K metadata-change 實驗（`prepare-10k-change-data` CLI） |
| `archive/llm_bias/financial_soundness/`; `archive/tests/test_financial_soundness.py`; `archive/data/baseline/financial-soundness-localization/`; `docs/archive/financial-soundness-localization/`; `docs/archive/financial-soundness-causal-validation/` | `llm_bias/financial_soundness/`; `tests/test_financial_soundness.py`; `data/baseline/financial-soundness-localization/`; `docs/financial-soundness-localization/`; `docs/financial-soundness-causal-validation/` | Frozen exploratory financial-judgment localization and causal-validation workflow; no certification claim. The package depends on active `llm_bias/core`; its archived test is not in the root pytest suite. |
| `llm_bias/static/counterfactual.*` | `llm_bias/static/` | counterfactual dashboard frontend |
| `tests/` | `tests/` | 上述 package 的 20 個 test 檔 |
| `scripts/` | `scripts/` | `run_synthetic_*.sh`、`build_index_constituents.py` |
| `../docs/archive/` | `docs/` | 對應的 6 個 canonical workflow 文件 |

相依關係（還原時若只還原部分 package，注意這些邊界）：

- `counterfactual_data` → `counterfactual_patching.data`
- `ten_k_change_data` → `prompt_analysis`（active）
- 其餘 package 只依賴 `llm_bias/core`（active）

## 還原步驟

以 `synthetic_entity_bias` 為例：

```bash
git mv archive/llm_bias/synthetic_entity_bias llm_bias/
git mv archive/tests/test_synthetic_entity_*.py tests/
git mv archive/scripts/run_synthetic_*.sh scripts/
git mv docs/archive/synthetic-entity-bias.md docs/
```

接著在 `pyproject.toml` 加回 entry point：

```toml
synthetic-entity-bias = "llm_bias.synthetic_entity_bias.cli:main"
```

還原 financial-soundness workflow 時，必須把完整 execution unit 搬回原位置，再在
`pyproject.toml` 加回相同的 project script；不要只還原 package：

```bash
git mv archive/llm_bias/financial_soundness llm_bias/
git mv archive/tests/test_financial_soundness.py tests/
git mv archive/data/baseline/financial-soundness-localization data/baseline/
git mv docs/archive/financial-soundness-localization docs/
git mv docs/archive/financial-soundness-causal-validation docs/
```

```toml
financial-soundness = "llm_bias.financial_soundness.cli:main"
```

還原後才可執行該 archived test；未還原時它不屬於 root pytest collection。

視情況同步更新 `llm_bias/__main__.py` 的 workflow 清單、
`tests/test_workflow_boundaries.py` 與 `AGENTS.md`/`README.md`
中的文件清單，然後跑完整驗證：

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
```

## Archived research semantics（counterfactual 線）

以下研究語義隨程式一併移入；若還原這些 package，這些規則重新生效：

- `Pair` 必須保留 entity token start/end 與完整 token-id span，並支援舊
  single-token pair。
- 不同長度 span 使用 normalized span-internal token centers 的 nearest
  mapping；不得插入/刪除 sequence token，也不得合成 activation。interactive
  dashboard 對不同長度 pair 應明確回傳 validation error。
- source/target prompt 可有不同 token 長度；batch answer logits 讀各自 final
  position，control patch 使用各自最後一個非-entity position。
- batch 結果必須保留 source span、target span、position mapping 與 mapping
  strategy。
- bias-specific pairs 必須共用相同 headline/context 與 expected outcome，分開
  報告 `real_vs_real`、`real_vs_anonymous`、`real_vs_synthetic`、
  `synthetic_vs_synthetic`，使用固定 outcome options 的 logit margin，而不是
  factual answer-transfer 公式。

## 備註

- `pyproject.toml` 的 `[project.optional-dependencies] extraction`
  （`langextract`）是 archived 的 `prepare-counterfactual-data annotate` 用的；
  為了還原時免重新裝，這個 extra 仍保留在 active `pyproject.toml`。
- `data/*.csv` 與 `artifacts/` 是 gitignored 的本地資料/輸出，不在 archive 範圍。
- 搬移日期：2026-08（baseline-focused archive plan，見 git log）。
