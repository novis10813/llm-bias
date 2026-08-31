# `scripts/` guidance

本目錄保存 active research operators、data preparers、diagnostics 與 report renderers。
每支 script 的用途、input/output、owner workflow 與 test coupling 見
[`../docs/research-scripts.md`](../docs/research-scripts.md)。

## Local conventions

- 從 repository root 執行 `uv run python scripts/<name>.py` 或
  `bash scripts/<name>.sh`。`scripts/` 不是 Python package，不在 scripts 之間建立隱式
  import dependency。
- Python operator 使用 argparse 並寫 compact derived outputs 與 provenance。禁止保存
  raw activations、residuals、hidden states、gradients 或 Jacobians。
- Shell runner 使用 `set -euo pipefail`、解析 `SCRIPT_DIR`/`REPO_ROOT` 後回到 root；長任務
  的 tmux/env contract 要與 canonical workflow 文件同步。
- 不 hardcode model slug。使用 `llm_bias.core.lens_artifacts.model_slug`，canonical lens
  path 維持 `${ARTIFACT_ROOT}/${MODEL_SLUG}/jacobian-lens/jacobian_lens.pt`。
- 只有 `promote_qwen_lens_candidate.py` 可在完成 hash、shape、metadata 驗證並 archive
  舊 artifact 後寫入 active canonical lens。
- `render_jspace_intervention_report_figures.py` 綁定既有 report runs。加入新 run 前先在
  `docs/jspace-sector-intervention/report.md` 記錄 provenance。

`promote_qwen_lens_candidate.py`、`jspace_tfidf_analysis.py` 與 shell runners 有直接
regression-test coupling；重命名 script、public symbol、env variable 或 path contract
前先查 [`../docs/research-scripts.md`](../docs/research-scripts.md) 的 Test-coupled
interfaces。

## Instruction Index

目前 `scripts/` 沒有含 `AGENTS.md` 的直接子目錄。新增 script family 子目錄後，只有在
它形成獨立 workflow 或慣例時才加局部 `AGENTS.md`，並更新本節。

## Verification

```bash
uv run pytest -q tests/test_lens_artifacts.py tests/test_lens_promotion.py \
  tests/test_jspace_token_normalization.py tests/test_workflow_boundaries.py
uv run python -m compileall -q scripts
```

修改 shell file 後另執行 `bash -n scripts/<name>.sh`；修改 Python script 時執行該 script
對應的 targeted tests，不以 GPU/model run 取代 deterministic tests。
