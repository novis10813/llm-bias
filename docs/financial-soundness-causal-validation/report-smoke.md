# 因果驗證流程通過真實模型 smoke，尚未認證財務神經元

**日期**：2026-09-07。**性質**：工程 smoke，非正式因果確認。協議見 [因果驗證提案](proposal.md)。

## 全部干預條件已完成，恢復檢查與輸出 hashes 通過

run ID 為 financial-smoke-causal-v1，輸出位於 artifacts/qwen3.5-4b/financial-soundness-causal-validation/runs/financial-smoke-causal-v1/。模型為 Qwen3.5-4B，GPU0，bf16 forward 與 FP32 完整續接計分。

上游為 [定位 smoke](../financial-soundness-localization/report-smoke.md) 的 financial-smoke-localization-v1。程式核對來源 manifest、compact artifacts、模型及 tokenizer identity，使用其 24 個 L0 候選，在未參與定位的 12 對 evidence/comparison 題上完成驗證。

總共輸出 4,032 筆記錄：clean 576、scale 1,152、random 1,152、restore 576、donor 576。scale/random 各包含 0.5 與 0.0 兩個劑量；donor 包含每對的雙向替換。

576 筆 restore 的 margin 與 clean 最大絕對差為 0。四個輸出 artifact 的 SHA-256 全部驗證通過，manifest 狀態為 complete。定位與因果流程合計執行 8 分 34 秒，包含模型載入及來源 hashes 計算。

## 工程通過不等同財務概念的因果認證

restore 只檢查同一座標撤銷干預後能否回到原結果，不能作獨立的概念儲存證據。本次只查 L0、採縮減樣本，且定位 smoke 已顯示答案映射敏感與部分 clean 題目答錯。

因此，本次只確認定位到干預的端到端流程可執行。不能由 complete 狀態推論候選具有財務特異性，也不能宣稱模型財務判斷能力遭破壞。正式研究仍需人工檢查提示詞、增加句式與數值覆蓋，並先凍結行為及因果 gate。

## 重跑必須保留來源並使用新 run ID

使用 financial-soundness run-causal-validation，將上述定位 run 目錄指定為 --source-run，模型路徑指定為 --model，並提供未使用過的 --run-id。以 CUDA_VISIBLE_DEVICES=0 選擇 GPU。下游會驗證來源，不接受在驗證過程中更換候選或重新排名。

## 本次程式驗證仍有四項不相關的既有測試失敗

46 項相關測試通過，lock、compile、build、JavaScript 語法及新文件相對連結檢查通過。全套測試檢查時為 464 通過、4 失敗；後續新增兩項測試亦已通過。既有失敗來自三項 context-overriding 測試缺少設定 artifact，以及一項 workflow-boundary 測試缺少 CLAUDE.md。本次未修改這些不相關缺失，不宣稱全套測試通過。
