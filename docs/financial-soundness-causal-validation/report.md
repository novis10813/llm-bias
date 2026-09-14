# 單座標抑制只造成小幅判斷變化，本輪尚未建立財務特異性

**狀態**：exploratory V1 completed，非正式因果認證。日期：2026-09-07。協議見 [proposal](proposal.md)。

## 全部 72 個候選已完成因果驗證

run ID：financial-causal-exploratory-v1。輸出：artifacts/qwen3.5-4b/financial-soundness-causal-validation/runs/financial-causal-exploratory-v1/。

使用 [全層定位](../financial-soundness-localization/report.md) 的 72 個候選，在 calibration/held-out 共 48 對 evidence/comparison 題上執行 clean、0.5/0.0 縮放、同層隨機對照、同題恢復及雙向 donor 替換，共 48,384 筆記錄。GPU0 上耗時 54 分 43 秒。

來源及所有輸出 hashes 通過核對，manifest 為 complete。所有 restore 的 margin 與 clean 最大絕對差為 0；這只驗證同位置撤銷干預的工程一致性。

## 抑制效應較小，最負的組別也有相近隨機效應

以下是 held-out、縮放係數 0.0 時，各候選與答案形式組別的平均正確答案優勢變化範圍。單位為 nats；負值表示正確答案相對錯誤答案的優勢下降。每組先平均同數值組的改寫，再平均數值組。

| 題型 | 候選抑制 | 同層隨機抑制 |
| --- | --- | --- |
| 財務 evidence | −0.015174 至 +0.006781 | −0.015085 至 +0.021271 |
| 非財務 comparison | −0.008878 至 +0.016802 | −0.007905 至 +0.023106 |

範圍重疊本身不是統計等效證明，不能據此排除每個候選。具體例子是 L10 N585：財務 labels-reversed 的平均變化為 −0.015174，同層隨機對照為 −0.015085；同一候選在 direct 為 −0.001996，在 labels-forward 則為 +0.003021。這個事後挑出的最負組別沒有呈現一致的跨答案形式損害，不能當作確認發現。

## 翻轉紀錄不能替代能力損害與特異性檢驗

held-out 財務題在 0.0 候選抑制下，共 1,728 筆候選×題目記錄，其中 37 筆的 margin 正負號與 clean 不同。這不是 37 道獨立題，也不是 Buy/Sell 翻轉；同一道題會被不同候選重複測量，且翻轉可能改善或惡化答案。

完整的不同劑量、donor 與條件區分變化皆保留在 forward/effects.jsonl 及 analyze/summary.json。本摘要沒有從其中另挑新的認證 gate，也沒有把跨候選比較當作已校正的顯著性檢驗。

## 下一輪應先改善行為測量，而不是宣稱財務神經元不存在

定位報告顯示答案映射敏感，以及財務與非財務題正確率不匹配。每個 split 只有兩個獨立數值組，bootstrap 區間亦不足以支撐穩健的人口推論。

本次最多支持：依這套排名選出的單座標，在 prompt 句末、指定劑量與受測短句下，沒有提供足以認證財務特異性的證據。這不排除其他 token 位置、候選排名或多座標共同作用，也不證明公司偏好必然分散。下一步先審查提示詞與答案形式，再決定是否另立版本；不事後修改本輪數據或宣稱神經元認證成功。
