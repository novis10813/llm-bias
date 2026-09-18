# J-space Outcome 決策翻轉 V2：Buy 轉移通過但 Sell 檢定力不足且位置不特異（success=false）

**狀態：已完成（Formal pipeline completed，`success=false`；V1 探索收線 shortlist 為空）。** 模型為 Qwen3.5-4B（bf16）；2026-08-28 完成。V1/V2 原始協議在 `details/`，本報告總結 V2 formal 結果與輔助診斷。

**一句話發現：** Outcome 梯度方向在 Buy 方向達成 9/9 翻轉（Holm $p = 0.003$ 通過），但 Sell 方向因測試集僅 2 檔股票導致統計檢定力結構性不足而失敗，且最後位置對照組同樣達成 100% 翻轉，位置特異性不成立（整體 verdict `success=false`）。

## 1. Outcome 梯度方向能否特異性翻轉 Buy/Sell 決策？Buy 方向成立，但 Sell 方向統計檢定力不足

在 11 檔未見過的 Held-out 科技股上，我們使用在 discovery 階段擬合的 outcome 梯度方向 $d_l$（L10–L30，強度 $r = 0.4$）進行介入：

**觀察：**
- **Buy 方向轉移顯著（9 個 clean-Sell 股票）**：主臂產生 **9/9 次** 決策翻轉（sell $\to$ buy），而 Matched-random 對照組為 **0/9**、Label-permutation 對照組為 **0/9**；單尾精確檢定 $p = 0.0015$，Holm 校正後 **$p = 0.003$**，通過 gate。
- **Sell 方向檢定力結構性不足（僅 2 個 clean-Buy 股票）**：主臂產生 **2/2 次** 翻轉，對照組為 **0/2**；但當 $n = 2$ 時，exact paired test 的理論最小單尾 $p$ 值為 $0.25$，數學上不可能達到 $\le 0.05$ 門檻。

**解讀：** 依預先註冊門檻，因 Sell 方向檢定力不足，整體 Formal Pipeline 宣告未通過（`success=false`）。失敗原因為 test split 中 clean-Buy 樣本不足（受限於模型強大的 Sell Prior），而非效應缺席。

## 2. 該效果是否具備證據位置特異性（Position Specificity）？否，最後位置注入同樣 100% 翻轉

**觀察：** 在輸出端最後位置（final position）注入相同的方向向量時，在買賣雙方向同樣產生了 **100% 的決策翻轉**（Buy 方向 9/9 翻轉、Sell 方向 2/2 翻轉）。

**解讀：** 效果在輸出位置可被完全重現，位置特異性不成立。V2 方向干預本質上是對決策輸出軸的通用調控（outcome-axis steering），不能解讀為作用於證據位置的因果推論。

## 3. 輔助幾何與讀出診斷帶來何種理解？

**觀察與診斷：**
1. **幾何正交性**：[幾何診斷](details/report-v2-geometry.md) 顯示，科技與金融業的狀態差異在各層皆幾乎正交於 outcome 梯度方向，兩者非同一語義軸。
2. **零證據 Prior Probe**：在無證據模板下，12 個條件全數為 Sell（margin −4.93 至 −3.39 nats），ticker 與 sector 差異可忽略，證實 Sell 先驗為模板共有結構。
3. **逐層透視鏡解碼**：透過 Jacobian lens 解碼，+d 方向的 Buy-minus-Sell logit margin 隨深度自 L10（−0.16）逐步擴大至 L24（+21.84），展現層級累積特徵。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| V1 詞彙篩選協議與報告 | [V1 協議](details/proposal-v1.md)；[V1 報告](details/report-v1.md)（840 次干預 1 次翻轉，shortlist 空）。 |
| V2 Formal 協議與執行記錄 | [V2 協議](details/proposal-v2.md)（Draft 1 frozen）；test run `outcome-flip-tech-test-20260828T051624Z`，位於 `artifacts/qwen3.5-4b/jspace-outcome-direction-flip/runs/`。 |
| 幾何投影診斷 | [幾何報告](details/report-v2-geometry.md)。 |
| 版本演進與歷史索引 | [版本索引](details/README.md)（記錄 CUDA 決定論修復等工程歷史）。 |

**本次編輯說明：** 本報告按三項核心問題改寫，明確載明 V2 Formal 驗證各方向數值與位置特異性否定結論；移除純導覽頂層 proposal，直接以本報告為閱讀入口。
