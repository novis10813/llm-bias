# `docs/` guidance

本目錄保存政策、canonical workflow、artifact contract、研究方法與結果。上層解析規則
見 root `AGENTS.md`；文件分類、source of truth 與維護順序見
[`documentation-system.md`](documentation-system.md)。

## 文件慣例

- Active workflow 的命令、參數、artifact layout 與 interpretation limits 只在對應
  canonical 文件維護；README 與 AGENTS 只提供摘要和連結。
- 實驗結果要附 run ID 或 artifact path，並標明 discovery、calibration、held-out、
  diagnostic 或 formal status。不要用新結果覆寫舊 run 的 historical record。
- Direction source、primary outcome、controls 或 gate 改變時，依
  [`documentation-system.md#experiment-versioning`](documentation-system.md#experiment-versioning)
  建立 version index 與版本文件；每版明寫 implemented/proposed/evidence status。
- `proposal/` 區分 research design 與完成狀態；code 移入 archive 後，要同步更新 roadmap
  的 code/protocol status。
- `archive/` 文件描述 frozen workflows。Archived CLI 不可寫成 active entry point；
  restore steps 統一連到 [`../archive/README.md`](../archive/README.md)。
- 圖表放 `assets/` 時，在來源 workflow 文件記錄 renderer、input run 與可重建方式。
- Markdown 連結使用相對路徑；修改或搬移文件後，檢查所有 inbound links。

Research operators 與 renderers 的詳細對照見
[`research-scripts.md`](research-scripts.md)；artifact lifecycle 見
[`artifact-contract.md`](artifact-contract.md)。

## Instruction Index

目前 `docs/` 的直接子目錄沒有 `AGENTS.md`。若 `archive/` 或 `proposal/` 出現超過本檔
一兩句能覆蓋的獨立維護規則，再於該目錄新增 `AGENTS.md`，並只回來更新本節。

## Verification

修改 root guidance 或 workflow boundary 時執行：

```bash
uv run pytest -q tests/test_workflow_boundaries.py
```

修改 command block 時，對照 `pyproject.toml` entry points 與對應 CLI parser；修改相對
連結後，從該 Markdown 檔案所在目錄解析目標，不能只從 repository root 判斷。
