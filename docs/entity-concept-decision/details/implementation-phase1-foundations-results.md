# Phase 1 基礎工具三片已驗收，完整 regression 仍有四項缺檔失敗

**範圍：工程交付，不是研究結果。**依[已批准 spec](implementation-phase1-foundations.md) 依序完成 Slices 1–3，沒有 CLI／runner、真實概念材料、checkpoint 載入或 GPU 實驗。完整 Phase 1 pipeline 仍未實作。

## 1. 交付檔案與能力

| Slice | 實作 | 新增測試 |
|---|---|---|
| 1 | `llm_bias/core/artifacts/verification.py`：完成 manifest 的身份／登記／digest／counts 驗證、required inputs 限定、路徑與非有限值拒絕 | `tests/test_core_input_verification.py` |
| 2 | `llm_bias/core/analysis/subspaces.py`：FP64 Gram 檢查、CPU FP32 投影、shape／dtype／overflow 防禦 | `tests/test_core_subspaces.py` |
| 3 | `llm_bias/entity_concept_decision/{__init__,materials,concepts}.py`：材料審核標記／split 檢查、家族等權 fitting、近零退化、unit direction scoring | `tests/test_entity_concept_materials.py`、`tests/test_entity_concept_directions.py` |

更新 `llm_bias/AGENTS.md` 與 `tests/AGENTS.md` 入口，研究文件標明只完成無模型基礎工具。沒有修改舊實驗程式、既有 API、依賴或 artifacts。各片經 general 與 reviewers 審查；Slice 2 自動 spec-reviewer 無回傳文字，另補一次明確 spec-reviewer，結果 PASS。

## 2. 主流程親自執行的驗收

| 檢查 | 結果 |
|---|---|
| Slice 1 的 spec gate（含既有 regression） | 30 passed |
| Slice 2 的 spec gate（含既有 regression） | 58 passed |
| Slice 3 的 spec gate（含既有 regression） | 39 passed |
| 四份新測試合併執行 | **50 passed** |
| `uv run pytest -q` | **758 passed、4 failed、1 skipped、11 warnings**；151.80 秒 |
| `uv lock --check` | pass |
| `uv run python -m compileall -q llm_bias` | pass |
| `uv build` | pass；wheel 包含新 package 與 core helpers |
| 兩份 static JS 的 `node --check` | pass |
| `git diff --check` | pass |

逐片 counts 含重複 regression，不可相加當作獨立測試總數。測試僅使用 CPU、合成資料／既有 fake models，未執行研究模型 smoke 或 formal run。

## 3. 四項失敗皆落在本次未修改的缺檔依賴

1. `tests/test_workflow_boundaries.py::test_root_guidance_shares_exact_workflow_contract`：root `CLAUDE.md` 不存在；先前已核 HEAD 亦无該檔。
2. `tests/test_context_overriding.py::test_confirmation_uses_toward_source_pair_aggregation_and_all_gates`。
3. `tests/test_context_overriding.py::test_confirmation_fails_closed_on_config_and_required_matrix_mismatch`。
4. `tests/test_context_overriding.py::test_confirmation_artifact_writer_and_cli_dispatch`。

後三项單獨重跑，均在 `_confirmation_config()` 讀取 `artifacts/qwen3.5-4b/cross-sector-context-overriding/configs/b-v1-confirmation-v1.json` 時發生 `FileNotFoundError`，尚未進入被測分析。這份測試檔與相應 production code 本次未改動。未建立假資料、skip／xfail 或變更 assertions 繞過。

因此只能宣稱三片已驗收，不能宣稱全 repository verification pass。後續可獨立修正既有測試對本機 artifacts 的依賴，以及 root 相容文件契約；不納入本次批准範圍。

## 4. 下一步需要另行設計與批准

準備真實候選材料、token/span 對齊、canonical lens readout 與完整 softmax 聚合、upstream 整合、校準／audit gates、runner／CLI 及真實 smoke。當前純函式的 `ok` 只代表數學方向可計算，`approved` 只是 reader 要求的人工審核標記；兩者均不是概念或因果假說已通過的證據。

變更保留在 working tree，未 commit。
