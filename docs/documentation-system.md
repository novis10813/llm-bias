# Documentation and instruction system

本 repository 使用兩層文件系統：`docs/` 保存詳細政策與 workflow，分層
`AGENTS.md` 提供 AI 協作入口。README 面向首次進入 repository 的讀者，提供專案摘要、
setup、workflow 與文件地圖。

## 解析順序

1. 先讀 root `AGENTS.md`。
2. 修改某個目錄時，再讀該目錄的 `AGENTS.md`。
3. 若目標路徑的祖先有多份 `AGENTS.md`，離目標檔案最近者優先。
4. `AGENTS.md` 只補充所在目錄的規則；未覆蓋的規則繼續沿用上層文件。

每層 `AGENTS.md` 的 Instruction Index 只列出直接子目錄中的 `AGENTS.md`。完整
repository map 留在 README 或詳細文件，避免 root instruction file 隨目錄數量成長。

## 文件責任

| 文件 | 讀者與用途 | 應包含 | 不應包含 |
|---|---|---|---|
| root `AGENTS.md` | 所有 coding agents | 專案 scope、全域規則、檔案放置、直接子層入口、常用驗證 | 完整 CLI 操作手冊、整棵目錄樹 |
| 子目錄 `AGENTS.md` | 修改該目錄的 agents | 該目錄特有的 ownership、慣例、局部驗證、直接子層入口 | root 已定義的通用規則 |
| `docs/<experiment-name>/` 與 `details/` | 維護者與實驗執行者 | 頂層研究入口與最終／最新報告；details 保存原始協議、分階段結果、artifact schema 與操作限制 | 無 code 或 artifact 依據的推測 |
| `README.md` | 新使用者 | setup、active workflows、quickstart、文件地圖 | 重複 canonical workflow 的完整參數表 |
| `CLAUDE.md` | Claude Code 相容入口 | root 規則的工具特定摘要與必要相容內容 | 與 root `AGENTS.md` 衝突的第二套政策 |

`tests/test_workflow_boundaries.py` 要求 root `AGENTS.md` 與 `CLAUDE.md` 各包含一次
完全相同的 `Shared experiment workflow contract`。修改該段時必須同步兩份文件。

## 詳細文件分類

- `docs/<experiment-name>/`：研究線的 canonical 目錄，頂層只保留 `proposal.md` 與
  `report.md`。已收線者使用收線報告；未收線者保留最新完整報告，明寫 discovery、
  exploratory 或 proposed 等實際狀態，不因檔名統一而視為完成。
- 單一協議的 `proposal.md` 保存問題、protocol、controls、estimands、gates、artifact
  contract 與 CLI。多版本或分階段研究的 `proposal.md` 是全線入口，說明研究問題、
  階段順序、狀態與原始協議連結；它不合併各階段 CLI，不取代 frozen protocol，亦不
  構成新實驗版本或 formal run 授權。
- `docs/<experiment-name>/details/`：保存版本化／分階段原始 proposal、階段 report、
  diagnostic、smoke 紀錄與舊版 README。原始協議仍是對應版本的 source of truth；
  移入此處不代表 obsolete 或 frozen。尚在 proposed 的延伸必須在頂層入口明列。
  沒有中間文件的研究不建立空目錄。Run outputs 留在 `artifacts/`，圖表留在既有
  `assets/`，本次整理不搬移或改寫歷史 artifacts。
- `docs/README.md`：15 條研究線的閱讀入口、狀態與前後關係。關係分為「資料／產物
  依賴」、「研究承接」與「方法參考」；每條關係要附來源文件，不以日期先後推定依賴。
  各研究的 `proposal.md` 列上游與後續研究，連回總覽。
- `docs/proposal/`：跨實驗研究計畫與 roadmap，不保存單一 active experiment 的完整
  protocol 或 dated results。
- `docs/archive/`：已移至 `archive/` 的 frozen workflow 文件。文件保留還原與
  historical artifact 語意，但不可把 archived CLI 寫成 active entry point。
- `docs/assets/`：文件引用的可重建圖表。產生圖表的 script 與來源 run 應在對應
  report 中記錄。
- `docs/*.md`：只保留 shared policy、operations、contracts、dashboard 或跨實驗索引；
  不再新增單一實驗的 proposal/report 到 docs root。

點時間的實驗數字應附 run ID、artifact path 或日期。後續實驗不覆寫舊結果；新增或
延伸 report，並說明哪個 run 是正式結果、哪個只作 diagnostic。

## Experiment versioning

同一研究線若改變 direction source、primary outcome、split/freeze sequence、control
family 或 success gate，建立新的 version 文件，不在舊文件中把新 protocol 寫成
「下一步」後混用結果。版本化研究線使用：

- 頂層 `proposal.md` 作為不帶版本號的 index/router，列出各版本 status、primary
  outcome、implementation 與 evidence；
- 每個版本在 `details/` 保存詳細文件，檔名帶 `-v1`、`-v2`；分階段協議保留原有
  phase identity，不將最後一階段改名冒充全線協議；
- root `AGENTS.md` 與 README 連到頂層入口，特定版本的協議引用直接指向 `details/`；
- run/config/report 明確寫 version。第一次 formal run 後若改 estimand 或 gate，建立新
  version，不回填舊版本。

## 文件搬移與歷史引用

搬移文件時修正 Markdown 相對連結、圖片路徑、內文文件路徑，以及程式／script 中的
協議引用。原始 protocol 的 gates、數字、版本、run ID 與命令保持不變；頂層入口新增
的說明與原始協議分開維護。舊路徑與新路徑可由 Git rename history 追查。

已產出的 artifacts 與其 provenance 不回寫。程式中供未來輸出使用的 protocol 路徑
改指目前文件位置，不改 schema 或實驗 identity。此次頂層入口統一只改文件編排，
不觸發研究版本變更。

## Source of truth

- Active CLI：`pyproject.toml` 的 `[project.scripts]` 與各 package 的 CLI parser。
- Python 與 workspace 依賴：`pyproject.toml`、`uv.lock`、`.python-version`。
- Runtime/data 忽略規則：`.gitignore`。
- Shared workflow core ownership：[`shared-experiment-core.md`](shared-experiment-core.md)。
- Artifact lifecycle：[`artifact-contract.md`](artifact-contract.md) 與各 workflow
  文件。
- Canonical lens 選擇與 candidate checkpoint 例外：
  [`jacobian-lens-selection/proposal.md`](jacobian-lens-selection/proposal.md)。
- Active 與 frozen code 邊界：root `README.md`、`archive/README.md`。
- Research scripts map：[`research-scripts.md`](research-scripts.md)。

文件中的命令或 path 若與上述來源衝突，先依 code/config 修正詳細文件，再更新引用它
的 `AGENTS.md`。

## 維護流程

1. 先用 `git status` 與 `git diff` 確認既有工作，不覆蓋不相干的 dirty changes。
2. 從 code、config、tests 與 artifact schema 查核事實。
3. 先更新 `docs/`、README 或 archive 說明。
4. 再縮短或更新各層 `AGENTS.md`，讓它們引用已存在的詳細文件。
5. 檢查 Markdown 相對連結，並執行受文件中命令或 contract 影響的驗證。

目錄出現獨立架構、非自明內部結構或專屬 workflow 後，才新增該目錄的
`AGENTS.md`。新增後，只更新最近一層祖先 `AGENTS.md` 的 Instruction Index；不要在
root 列出更深層入口。
