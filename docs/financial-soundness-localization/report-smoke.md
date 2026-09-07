# 定位流程完成真實模型 smoke，但提示詞尚不適合正式認證

**日期**：2026-09-07。**性質**：工程 smoke，非 formal discovery 或 held-out confirmation。協議見 [定位提案](proposal.md)。

## 真實 Qwen3.5-4B 已完成定位與 compact artifact 輸出

run ID：financial-smoke-localization-v1。輸出位於 artifacts/qwen3.5-4b/financial-soundness-localization/runs/financial-smoke-localization-v1/。

GPU0、bf16 forward、FP32 完整續接計分；只定位 L0，各 family/concept/answer_mode 保留 top-1。內建完整資料含 234 對，本次每個 family/concept/mode/split 保留第一對，company 額外保留兩家名稱，合計 99 對、198 筆行為結果。包含 topic 27 對、company 54 對、evidence 9 對及 comparison 9 對。公司為 Johnson & Johnson 與 Bank of America。

流程產生 24 筆候選，座標去重後也是 24 個。它們是探索排名，不是已認證的財務神經元。prepare、forward、analyze 與 manifest finalization 均完成，forward 約 38 秒。完整模型載入與 hashes 計算時間不含在此數字。

編碼輸入 SHA-256：86b86b53beb458eca0f1a71af9769c9cf792ea226394c269465f46bda2bed9df。模型檔案與 tokenizer identity、程式指紋記於 prepare/protocol.json，各輸出 hashes 記於 manifest.json。

## 答案映射仍會影響行為，不能把 smoke 當成能力驗證

本次有正確答案的每個 family/mode/split 只有一對，即兩道題。evidence 的直接詞彙題在 calibration 與 held-out 均為 2/2 正確，但 discovery 為 1/2；交換選項標籤後，三個 split 都只有 1/2。comparison 的直接詞彙題在三個 split 也都只有 1/2。

這些數字顯示我們需要繼續檢查答案形式與題目難度。模型的正負條件分數差即使大於零，也不代表它把兩個條件都答對。不得刪除答錯資料後宣稱定位成功，也不能將後续干預變化直接稱作財務能力崩塌。

## 重跑使用既有 smoke 輸入與新的 run ID

資料位於 data/baseline/financial-soundness-localization/smoke-v1.json；它是上述確定性縮減的資料，不是另一套正式 prompt family。使用 financial-soundness run-localization，指定此檔為 --prompts、.cache/models/qwen3.5-4b 為 --model、--layers 0、--top-k 1，以及未使用過的 --run-id。CUDA_VISIBLE_DEVICES=0 控制 GPU，不能覆寫既有 run。

因果驗證使用獨立命令與 run，依 [因果驗證提案](../financial-soundness-causal-validation/proposal.md) 執行。正式研究仍需完整句式/數值檢查、人工審查與 gate 凍結。
