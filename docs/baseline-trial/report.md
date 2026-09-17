# Baseline Trial：外部金融提示詞流程接通與批量透視鏡解碼（流程建置）

**狀態：已完成（流程建置）。** 完成狀態依各資料集與執行階段獨立判定，非單一全線 verdict。完整重現指南與操作契約見 [實驗計畫](proposal.md)。

**一句話發現：** 將外部金融偏誤試驗提示詞成功轉換並接通 prompt-analysis 分析管線（50-stock 轉出 800 cells，paper 規模轉出 30,744 cells），批量 lens-forward 實測將 4B 逐層解碼耗時自 ~24 分鐘降至 ~4 分鐘，產物合約通過驗證。

## 1. 外部金融試驗提示詞如何接入本專案管線？格式轉換與驗證完成

我們透過轉換腳本將外部 trial-plan 提示詞轉換為本專案的 legacy-wide CSV 格式：

**觀察：**
- **50-stock 規模**：50 檔股票、16 個 prompt columns（涵蓋 attribute, volume, intensity, strategy 等條件），共轉出 **800 個** 非空 cells。
- **Paper 規模**：427 檔股票、72 個 prompt columns，共轉出 **30,744 個** 非空 cells。
- **序列長度**：Token 長度中位數約 245–333 tokens，最大長度達 660–733 tokens，統一使用 `--max-seq-len 1024` 避免截斷財務證據。
- 輸入格式與欄位對齊通過 `prompt-analysis inspect-input` 驗證。

**解讀：** 外部試驗提示詞已無損接入本 repo 的分析環境，每筆資料旁均附有 `.provenance.json` 記錄原始 SHA-256 與轉換矩陣。

## 2. 批量化透視鏡解碼效能與數值一致性為何？耗時顯著降低，數值達 bf16/ulp 容差一致

在執行透視鏡解碼（lens-forward / readout）時，我們針對逐層 Jacobian-lens transported readout 進行批量化（batched lens-forward）優化：

**觀察：**
- **解碼效能**：在 Qwen3.5-4B 上執行 5 層 readout 時，批量化推論將整體耗時由約 **24 分鐘降低至約 4 分鐘**。
- **數值精度**：批量化推論結果在 bf16 浮點數及 ulp 容差下，與逐筆循序推論之 log-probability 完全一致。

**解讀：** 批量優化大幅提升了大規模資料集的逐層表徵讀出效率，且未引入任何數值漂移或精度損失。

## 3. 各階段產物合約與下游解讀邊界為何？

**契約與邊界：**
1. **三階段管線契約**：管線嚴格解耦為 `readout`（逐層詞彙讀出與不確定度）、`generate`（唯一執行自回歸生成的階段）與 `attribute-generated`（僅讀取既有生成記錄計算梯度歸因，不重複生成），遵循 `RunManifest` 規範。
2. **不保存原始張量**：產物僅輸出 compact top-k、rank、統計量與數值摘要，嚴禁持久化未聚合的 raw activations。
3. **非單一偏誤結論**：本工作為基礎設施與管線重現建置，各階段通過產物合約驗證，不代表已在特定股票上確立單一因果或偏誤結論。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| 原始執行計畫與轉換契約 | [實驗計畫](proposal.md)；轉換腳本 `scripts/convert_baseline_trial_plan.py`。 |
| 資料集路徑 | `data/baseline/qwen36-27b-50stocks/` 與 `data/baseline/paper-local-qwen36-27b/`。 |
| 執行管線腳本 | `scripts/run_prompt_analysis.sh`；專用 CLI `baseline-trial`。 |

**本次編輯說明：** 本報告按三項核心問題改寫，聚焦於資料轉換規模、批量解碼效能與產物合約邊界；原始計畫 `proposal.md` 完整保留。
