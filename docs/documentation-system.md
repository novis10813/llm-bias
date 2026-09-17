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
| root `AGENTS.md` | 所有 coding agents | 專案 scope、全域規則、檔案放置、直接子層入口、常用驗證 | 逐實驗結果與進度、完整 CLI 操作手冊、整棵目錄樹 |
| 子目錄 `AGENTS.md` | 修改該目錄的 agents | 該目錄特有的 ownership、慣例、局部驗證、直接子層入口 | root 已定義的通用規則 |
| `docs/<experiment-name>/` 與 `details/` | 維護者與實驗執行者 | 頂層研究入口與最終／最新報告；details 保存原始協議、分階段結果、artifact schema 與操作限制 | 無 code 或 artifact 依據的推測 |
| `README.md` | 新使用者 | setup、active workflows、quickstart、文件地圖 | 重複 canonical workflow 的完整參數表 |
| `CLAUDE.md`（目前缺檔） | 歷史 Claude Code 相容入口 | 若恢復，沿用 root 規則 | 與 root `AGENTS.md` 衝突的第二套政策 |

`tests/test_workflow_boundaries.py` 仍要求 root `AGENTS.md` 與 `CLAUDE.md` 各包含一次
完全相同的 `Shared experiment workflow contract`，但目前 Git 未追蹤 `CLAUDE.md`，
該測試因缺檔失敗。這是既有相容性問題，本輪不新增副本或放寬測試；shared contract 保持不變。

## 詳細文件分類

- `docs/<experiment-name>/report.md`：面向讀者的研究結論入口，從原始協議與執行紀錄
  萃取發現；不是逐次 run 日誌。只有階段結果時，總覽可直接連到階段報告，不建空報告。
- 頂層 `proposal.md` **不是必備檔案**。既有檔案若本身就是原始協議，保留其效力；
  若只是重複 report 的導覽，經使用者批准可移除並修復引用，不另建 index 或轉址檔。
  未指定的研究不隨本次整理一起遷移，也不得自動重建已移除的頂層 proposal。
- `docs/<experiment-name>/details/`：保存版本化／分階段原始 proposal、階段 report、
  diagnostic、smoke 紀錄與舊版 README。原始協議仍是對應版本的 source of truth；
  移入此處不代表 obsolete 或 frozen。尚在 proposed 的延伸必須在頂層入口明列。
  沒有中間文件的研究不建立空目錄。Run outputs 留在 `artifacts/`，圖表留在既有
  `assets/`，本次整理不搬移或改寫歷史 artifacts。
- `docs/README.md`：跨實驗的一句話結論與狀態只在此彙整，各列連到來源報告及協議；
  root README 與 AGENTS 只連到本頁，不另維護逐實驗結果。詳細前後關係放在可展開區，
  區分「資料／產物依賴」、「研究承接」與「方法參考」，每條關係附來源。
  原始 proposal 保留執行條件與階段授權；上下游總覽不另在每份結論入口複製一遍。
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

## 先讀結論，需要重跑時才讀協議

- 快速了解結果：讀 `docs/README.md`；查證據與限制：讀該列連結的報告；執行或重跑：
  讀對應版本的 proposal。`details/` 供查核，不是逐篇必讀清單。
- 總覽依下節的固定欄位與狀態填寫；run ID、gate 編號、命令與歷次修訂留在來源文件。
- 報告先說「問什麼、觀察到什麼、在哪些條件下成立」，再列證據；可用白話解釋，
  但保留精確指標與來源。未通過檢驗寫成「本次未支持」，不改寫為「證明不存在」。
- 收線報告提供目前結論，原始協議仍規範各歷史版本；收線本身不使協議失效，
  不因縮短入口而刪除歷史紀錄、改寫 gate 或新增 superseded 判定。

## 報告寫作原則

`report.md` 是從 `details/` 萃取出的結論；`details/` 是可查核的研究依據，不是可隨手
改寫或丟棄的草稿。未產生結論的嘗試仍可有價值，但不必全搬進主文。

1. 開頭寫研究範圍、模型、狀態與一句話發現；主文按研究問題組織，不按 run 時序重播。
2. 每個問題先給回答，再列決定結論的數據與對照，分清「觀察」與「解讀／限制」。
   白話解釋指標，說清楚干預做了什麼；零翻轉不等於零效應，null 不等於不存在。
3. 對不同模型、版本、公司、精度及探索／正式確認分開陳述；不把某一組的結果擴成全線結論。
   來源衝突就標出差異與待查項，不挑有利數字，也不為了文句順暢自行裁決。
4. 主文以一分鐘能讀出問題、發現與邊界為目標；保留必要表格。命令、完整判準、歷次
   嘗試與工程排錯留在原始文件，用連結查證，不再複製完整協議／run 清單。
5. 結尾提供精簡查證入口，連到支持主張的紀錄、run ID／artifact path 與原始協議。
   只有影響解讀的版本轉折才進主文；研究已收線不必再附一份推測性的「下一步」。
6. 新結果先記入 owning experiment 的本次紀錄；只有核心結論、重要限制或狀態變動才更新
   report 與總覽那一列。純重跑、smoke 或工程修復不觸發全線報告重寫。
   重要解讀更正附簡短編輯說明，保留原始數值與來源，不改歷史判定。

## 文件同步的最小修改範圍

本節也適用於 `/hey-doc`。一般「更新文件／同步結果」不授權重整目錄或改寫研究歷史。

- **先列可寫清單與理由**：每個擬修改文件必須對應本次已確認的結果、code 差異或使用者
  明確要求。讀過、被引用、位於祖先目錄都不構成修改理由。沒有需要同步的事實就零修改。
- **區分查閱與寫入**：可讀 root／局部 AGENTS 理解規則，但一次 run 不改 AGENTS、全域政策、
  root README、其他研究報告或 roadmap。只有它們自身的契約／事實確實受影響時才做最小修改。
- **禁止自動補齊結構**：不因模板要求而建立 proposal/report 成對文件、版本索引、空目錄、
  新 AGENTS 或額外摘要；不搬移、刪除、改名既有文件，不順手重寫未涉及的段落。
  這類整理需要使用者明確指定範圍，單一研究的刪除授權不擴及其他研究。
- **歷史依據預設不改**：frozen proposal、舊 run 結果、判準、split、controls、數值、
  run ID、模型／lens identity、artifact provenance 不因同步而改寫。新版本另立協議；
  發現錯誤以有來源的勘誤／新增紀錄說明，不把修正悄悄填回舊結果。
- **限制例外**：已授權搬移／刪除造成的失效連結，可只修連結目標；歷史 artifacts 不回寫。
  原始協議本身不因「收線」「格式統一」而失效，未執行也不能寫成已完成或已獲授權。
- **同步點只是線索**：`git log` 最近一次 docs commit 不保證已同步所有研究；先核對本次
  指定的實驗／run 與工作區差異。不明確時詢問範圍，禁止退回全 repo 重寫。
- **鐵則不能自行放寬**：一般同步不得修改本節或 AGENTS 中的保護條款以配合當次輸出；
  修改規則本身需要使用者明確要求。收尾列出實際改檔、理由及未解差異，檢查 diff 未越界。

## 總覽表格填寫規則

`docs/README.md` 的結果表固定四欄；模型不得自行增加狀態或把多種狀態串在同一格。

| 欄位 | 填寫要求 |
|---|---|
| 研究 | 使用既有研究名稱；若只總結一個版本、階段或流程建置，名稱必須標明範圍，探索也須標明。 |
| 狀態 | 只能選下表的一個固定值，不加括號、分號或 pass/fail；它表示工作進度，不表示結果好壞或證據強度。 |
| 一句話發現 | 用一句白話說明受測問題的結果及必要限制，最多兩組數字；探索不能寫成正式確認，未執行寫「尚無結果」，工具建置不能寫成研究成功。 |
| 查證／執行 | 直接連到支持本列結論與狀態的報告／探索紀錄，以及對應協議；只有提案時只連提案，不建立空報告。 |

### 狀態只使用六個固定值

| 狀態 | 使用條件 |
|---|---|
| 規劃中 | 本列範圍尚無研究執行，仍在提案、實作或等待授權／啟動；程式已寫好或協議已凍結不算已完成。 |
| 進行中 | 已開始本列範圍的研究執行或分析，仍有工作尚未結束；多階段研究僅部分完成且主線仍在推進，也用此值。 |
| 已完成 | 本列所指工作已有結果與來源紀錄，但沒有明確的全線收線決定；只完成探索或校準，須在研究名稱限定該範圍。 |
| 已收線 | 來源文件明確決定結束本列研究線或版本；可為陽性、負結果或前提不成立，不能只因 gate fail 或沒有下一次 run 就推定收線。 |
| 暫停 | 來源文件明確記載暫停且未收線；等待一般後續工作或單純久未更新不算暫停。 |
| 待確認 | 缺乏判定進度的依據，或來源文件互相衝突；不得猜測，先指出缺少的依據或衝突並查核。 |

填寫時先固定本列範圍，再查來源；來源明確收線／暫停時採用該決定，否則按
是否開始、是否完成選值。未執行的後續提案不使已收線主線自動變成進行中。
總覽只是摘要，不能透過改狀態授權 run、宣告收線或改變 frozen protocol。

「探索」「formal」「held-out」是證據階段，「通過／未通過」是檢驗結果，
「協議凍結」是協議狀態，都不是上述進度值。必要時寫入研究範圍或一句話發現，
不要把全部細節塞回狀態欄。例：`J-space token V1（探索）｜已收線｜候選名單為空…`；
不要寫 `探索 completed／gate fail／未授權下一階段`。

更新順序：先核對 owning experiment 的結果／階段文件，再改總覽那一列；不因改摘要
而回寫歷史結果。新增狀態前必須先修改本節定義，不能在 README 臨時造詞。

## 小型想法驗證不預設建立完整研究線

本節適用於尚未授權 formal run、只想判斷一個想法是否值得繼續的探索；不能以規模小、
只有一個 gate 或結果不理想為由，降級既有正式實驗。

- 優先在 owning experiment 的既有探索紀錄加一節；沒有合適文件時，才建立一份
  `details/note-<topic>.md`。尚無 owning experiment 時，可先用
  `docs/<topic>/note.md`，不要求配套 proposal、report 與空的 details 目錄。
- 每個問題以一頁內為目標，分開記錄執行前的問題／最小方法，以及執行後的結果／限制／
  是否繼續。方法至少連到模型、輸入、對照（若無則註明）、指標、config 或命令；
  結果附日期與 run ID 或 artifact path，未執行就明寫未執行。
- 探索中的方法改動按日期保留差異，不覆寫既有結果，也不為每次修改建立 proposal/report
  成對文件。需要正式確認、凍結判準或產出供下游依賴的研究結論時，先建立正式協議並取得
  run 授權；舊探索不可事後改稱 formal confirmation。
- 新的 smoke、review、materials 與 implementation 過程不各自產生研究報告。
  純工程紀錄依需要留在 `docs/dev/` 的既有文件或本地工作紀錄；影響研究解讀的材料、
  偏差與驗證證據，保留在 owning experiment 對應文件中並附來源。既有文件本輪不搬移。
- 輕量紀錄仍遵守 shared workflow、provenance 與禁止保存 raw activations 等規則；
  它只減少文件，不豁免執行安全、資料契約或 formal run 前的真實模型 smoke。

## Experiment versioning

正式協議若改變 direction source、primary outcome、split/freeze sequence、control
family 或 success gate，建立新的 version 文件，不在舊文件中把新 protocol 寫成
「下一步」後混用結果。版本化研究線使用：

- 由 report 的查證入口直接連到版本協議，不要求另建頂層 proposal 或版本索引；
- 每個版本在 `details/` 保存詳細文件，檔名帶 `-v1`、`-v2`；分階段協議保留原有
  phase identity，不將最後一階段改名冒充全線協議；
- root `AGENTS.md` 與 README 連到研究總覽；總覽連到報告與適用的原始協議，版本協議直接指向 `details/`；
- run/config/report 明確寫 version。第一次 formal run 後若改 estimand 或 gate，建立新
  version，不回填舊版本。

### 正式協議的最低要求

以下只規範新立或經授權修訂的正式協議，不要求對舊 frozen 文件補模板，也不套用於小型
探索 note。已執行結果另記，尚無結果時不必預建 report。

- 問題與文獻邊界：列方法出處、與原始設定的差異及風險；無文獻復現目標則明說。
- Input／Output：記錄輸入路徑、schema、tokenizer 條件、依賴 hashes，以及 compact 輸出
  與 finite 數值限制；不保存 raw activations／tensors／KV caches。
- 執行契約：命令對應該階段 CLI subcommand，不能讓讀者自行猜測互斥參數或跨階段分支。
- 假說、指標、controls、判準與停止條件：執行前確定；定義退化分佈、缺少對照或缺少
  合格樣本時的 fail-closed／fallback。若使用位置分割，須明確覆蓋且互斥；數值容差須
  依模型精度給公式或有依據的底限。
- 版本分立：prompt family、estimand、direction source、controls、split/freeze sequence
  或 gate 改變時另立版本，不原地改已凍結規範；有結果才寫該版報告，不配套建立空檔。
- Formal discovery/calibration/test 前，須以真實模型與 tokenizer 完成至少一筆 prompt
  的端到端 smoke，涵蓋 hooks、controls、下游模組與 analyze，確認無例外且符合 schema；
  通過 smoke 不等於獲得 formal run 授權。

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

現行操作文件的命令或 path 若與上述來源衝突，先查證再作最小修正；frozen 協議若
與現行 code 不符，保留歷史契約並記錄差異，不將今日 code 自動回填成舊規範。
只有 AGENTS 本身的規則或引用失效才更新它，不因下游報告更新而跟著改。

## 維護流程

1. 先用 `git status` 與 `git diff` 確認既有工作，不覆蓋不相干的 dirty changes。
2. 從 code、config、tests 與 artifact schema 查核事實。
3. 依本次可寫清單更新直接受影響的文件，遵守最小修改範圍。
4. 只有規則或入口本身需要變更，才更新 AGENTS；日常實驗結果不更新 AGENTS。
5. 檢查 Markdown 相對連結，並執行受文件中命令或 contract 影響的驗證。

目錄出現獨立架構、非自明內部結構或專屬 workflow 後，才新增該目錄的
`AGENTS.md`。新增後，只更新最近一層祖先 `AGENTS.md` 的 Instruction Index；不要在
root 列出更深層入口。
