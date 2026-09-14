# 多數候選的反應方向能跨句式重現，但回答仍受答案形式影響

**狀態**：exploratory V1 completed，非 formal confirmation。日期：2026-09-07。協議見 [proposal](proposal.md)。

## 全層定位產生 72 個候選，尚未認證其財務功能

run ID：financial-localization-exploratory-v1。輸出：artifacts/qwen3.5-4b/financial-soundness-localization/runs/financial-localization-exploratory-v1/。

Qwen3.5-4B 在 GPU0 執行 3 分 9 秒；使用全部 234 對提示詞、JNJ 與 BAC，掃描 L0–L31，各 family/concept/answer_mode 保留 discovery top-3，共 72 個不同座標。資料為 data/baseline/financial-soundness-localization/exploratory-v1.json。模型、資料與程式 identity 以及輸出 hashes 均保留於 run 內。

72 個候選中，70 個在 held-out 的平均配對反應差與 discovery 同號。這僅是方向重現，不代表效應幅度穩定、財務特異性或因果重要性。held-out 保留未見句式及數值；公司名稱仍是 JNJ/BAC，不是未見公司驗證。

## 同一財務問題更換答案形式後，正確率仍會改變

下表各格為 8 道題的正確率。direct 使用完整正負詞彙續接；labels-forward 使用 A 對應正向答案；labels-reversed 交換兩個標籤的語義。topic/company 沒有客觀正確答案，不計正確率。

| 任務與答案形式 | discovery | calibration | held-out |
| --- | --- | --- | --- |
| 財務 evidence / direct | 50% | 87.5% | 87.5% |
| 財務 evidence / labels-forward | 87.5% | 62.5% | 87.5% |
| 財務 evidence / labels-reversed | 50% | 50% | 62.5% |
| 非財務 comparison / direct | 50% | 50% | 50% |
| 非財務 comparison / labels-forward | 50% | 50% | 75% |
| 非財務 comparison / labels-reversed | 75% | 87.5% | 100% |

財務與非財務對照的難度並不一致，答案標籤也影響判斷。因此，候選排名不能單獨用來命名財務概念。全部答錯題皆保留，沒有以刪題提高通過率。

## 因果驗證使用全部候選，不回頭挑選 held-out 表現較好的單元

後續 [因果探索報告](../financial-soundness-causal-validation/report.md) 使用同一來源的全部 72 個候選。正式認證尚需改善行為測量、擴充獨立樣本並凍結門檻；本次沒有 success gate。
