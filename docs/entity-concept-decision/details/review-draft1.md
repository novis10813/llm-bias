# Draft 1 已完成文件審查，但既有 regression 有一項環境缺口

**範圍**：僅研究規劃與文件；沒有新增實驗程式、執行模型或產生新研究結果。入口見 [proposal](../proposal.md)。

## 文件與原始資料核對

- 三個 explorer 分別核對程式重用、實際 upstream artifacts、測試與 lifecycle；主流程另直接讀 code、JSON keys、16 公司名冊、basis shape 與 SHA-256。
- 另一個 explorer 對五份新文件作 preflight。依回饋釐清 recording helper 自行 forward、live restore 的狀態來源、named→anonymous instruction 對齊、fake model 的證據限制。
- 主流程另補 rank-1 norm-matched shuffled control 可能退化的檢查；不把相同 patch 當獨立 specificity 證據。
- 五份設計文件中的相對檔案連結全部存在；`git diff --check` 通過。這不是對所有歷史文件或所有 heading anchors 的全面查核。

## 執行過的測試與未處理缺口

執行 [共通契約 §6.1](design-and-validation.md#61-現有-regression命令存在) 列出的全部十個 test files，結果 **108 passed，1 failed**。

失敗：`tests/test_workflow_boundaries.py::test_root_guidance_shares_exact_workflow_contract` 讀取 repository root 的 `CLAUDE.md` 時出現 `FileNotFoundError`。`git ls-tree HEAD CLAUDE.md` 無項目，working tree 亦無該檔；本次沒有刪除它。屬既有 repository 相容入口與測試不一致，未在本研究規劃任務中新增檔案或改測試繞過。

因此不能宣稱 regression 全綠；此問題需獨立修復後重跑。沒有執行完整 pytest、build、GPU smoke、calibration 或 formal。現有單元測試通過不代表新概念假說成立，也不代表尚未實作的新增 tests 已存在。

## 交付後仍須決定

先審閱 Phase 1 的單方向概念定義、材料預算與停止政策，再撰寫 bounded implementation spec。Phase 2/3 仍為條件式計畫；不因本文件通過審閱取得模型執行授權。
