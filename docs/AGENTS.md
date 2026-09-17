# `docs/` guidance

本目錄保存研究規範、原始紀錄與結論。先遵守 root `AGENTS.md`；完整分工見
[文件系統](documentation-system.md)。一般文件同步（含 `/hey-doc`）必須遵守下列鐵則，
不能自行改寫這些規則以擴大當次修改範圍。

## 文件更新鐵則

- **讀取範圍不等於修改範圍**：先列本次可寫檔案與理由；只更新直接受影響的紀錄、
  必要的 report 段落及總覽那一列。祖先目錄、被引用文件與其他研究不自動納入修改。
- **研究結果不改 AGENTS**：AGENTS 不保存 run ID、分數或逐實驗進度；只有規則本身或
  入口契約需要調整時才修改。一般同步不得放寬本節；規則變更須由使用者明確要求。
- **原始協議與歷史結果預設不改**：不改 frozen 判準、controls、split、數字、run ID、
  模型／lens identity 或 provenance；新版本另立，勘誤保留原文與來源。只有經授權的
  搬移／刪除造成連結失效，才可僅修連結目標，不回寫 artifacts。
- **不自動增加文件或重整目錄**：不強制成對 proposal/report、不補空報告、不建替代
  index、不重建已刪頂層 proposal；新 AGENTS、搬移、刪除與改名需明確範圍授權。
- **先查依據，不升格結論**：探索不是正式確認、沒有翻轉不等於零效應；來源有衝突須
  標示待查，不能改寫來源讓結論一致。文件同步不授權 run，也不宣告收線。

適用範圍、例外與 `/hey-doc` 同步點限制見
[文件同步的最小修改範圍](documentation-system.md#文件同步的最小修改範圍)。

## 寫哪裡、怎麼寫

- `report.md`：讀者入口，按研究問題萃取結論，分清觀察、解讀與限制；只保留關鍵證據，
  連到原始紀錄。遵守 [報告寫作原則](documentation-system.md#報告寫作原則)。
- `details/`：原始版本協議、執行／階段結果與診斷依據；不是可隨手改寫的草稿。
  既有頂層 proposal 若本身是原始協議仍保留，不強制搬移；純導覽不必另立 proposal。
- 小型探索優先補既有紀錄，不套完整正式協議模板；適用範圍、升級條件與最小 provenance
  見 [小型驗證](documentation-system.md#小型想法驗證不預設建立完整研究線)。
- 新正式協議須遵守 [版本規則](documentation-system.md#experiment-versioning) 與
  [最低要求](documentation-system.md#正式協議的最低要求)；不得因精簡文件省略 controls、
  數值容差、fail-closed 或真實模型端到端 smoke，也不得事後改 frozen 設計。
- `docs/README.md` 結果表遵守 [表格規則](documentation-system.md#總覽表格填寫規則)：
  四欄、單一固定狀態、先限定研究範圍；不把 gate 結果塞入進度狀態。
- `proposal/` 保留跨實驗計畫；一次實驗更新不自動重寫 roadmap。`archive/` 描述 frozen
  workflows，還原方式連到 [archive 入口](../archive/README.md)，不把 archived CLI 寫成 active。
- Markdown 使用相對連結；有授權的刪除或搬移須修 inbound links。圖表的 renderer 與
  input run 記在對應報告，不在索引複製。

完整分類見 [文件分類](documentation-system.md#詳細文件分類)；共用機制見
[shared core](shared-experiment-core.md)、[research scripts](research-scripts.md) 與
[artifact contract](artifact-contract.md)。

## Instruction Index

目前 `docs/` 直接子目錄沒有 `AGENTS.md`。不要為一次實驗更新新增 instruction file；
確有無法由本層涵蓋的局部規則時，先提出需求，再更新最近祖先的 index。

## Verification

改 root guidance 或 workflow boundary 時執行：

```bash
uv run pytest -q tests/test_workflow_boundaries.py
```

檢查相對連結時，從來源 Markdown 目錄解析。檢查 diff 只含可寫清單；原始協議、歷史
數字與 artifacts 未變。若有既有測試失敗，分開報告，不以新增副本或放寬測試掩蓋。
