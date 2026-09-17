# J-space 產業介入實驗：殘差小幅敏感，但未通過方向與位置特異性（Held-out 未支持）

**狀態：已完成（Held-out 測試完成，特異性未支持）。** 模型為 Qwen3.5-4B（bf16）；2026-08-26 完成。原始操作契約與 dose 設計見 [實驗提案](proposal.md)。

**一句話發現：** 在 L14–L26 候選層置換產業座標能引起小幅 Buy/Sell 機率差移動，但安全劑量下 0/23 離散決策翻轉，且方向特異性（$C_{\text{direction}} = -0.0114$）與位置特異性（$C_{\text{position}} = -0.0587$）雙雙未勝過對照組，未支持產業特異因果。

## 1. 置換產業 J-space 座標能否特異性改變買賣決策方向？未通過方向特異性檢驗

我們在獨立定位確定的 L14–L26 候選 workspace 層，將 Technology 與 Financial Services 的 J-space 產業座標進行雙向置換（以安全劑量 `swap_fraction = 0.5`）：

**觀察（Held-out 測試集，11 個科技股與 12 個金融股）：**
- **方向特異性 Estimand（$C_{\text{direction}}$）**：
  $$C_{\text{direction}} = E(\text{sector, evidence}) - E(\text{random, evidence}) = \mathbf{-0.0114}$$
  Held-out 估計值為負，未勝過同範數的隨機對照組（Matched-random control）。
- **無離散決策翻轉**：在主要證據置換條件下，23 個 held-out 股票中無任何 prompt 發生 Buy $\leftrightarrow$ Sell 偏好翻轉（0/23 flips）。

**解讀：** 雖然介入微幅改變了 continuation margin，但該移動並未優於隨機方向擾動，未能支持特定產業方向的引導效應。

## 2. 介入效果是否具備證據位置特異性（Position Specificity）？未通過，輸出位置效應更大

**觀察：**
- **位置特異性 Estimand（$C_{\text{position}}$）**：
  $$C_{\text{position}} = E(\text{sector, evidence}) - E(\text{sector, final}) = \mathbf{-0.0587}$$
  估計值顯著為負。
- 在最後輸出位置（final position）注入產業向量時，在 Financial Services $\to$ Technology 方向引發的 margin 偏移（+0.0938 nats）甚至遠大於證據位置（+0.0104 nats），且在最終位置產生了 1 個翻轉。

**解讀：** 效果在輸出位置比在證據位置更強烈，無法證明介入特異地作用於證據語義整合區間。

## 3. 劑量校準確立了何種安全邊界與機制限制？

**觀察與限制：**
1. **安全擾動上限**：校準確立了 `swap_fraction = 0.5`（相對擾動 ~1.5%）與 `gain = 1.10`（~2.5%）為安全上限，更高劑量（如 gain $\ge 1.2$ 或 full swap）累積擾動達 8%–19%，會破壞殘差流穩定性。
2. **保守科學結論**：模型對 L14–L26 的小幅殘差微調存在數值敏感度，但本實驗未能在未見股票上建立產業方向特異性或證據位置特異性。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| 原始提案與 Estimands 定義 | [實驗提案](proposal.md)（含校準與 held-out 判準）。 |
| Held-out 執行記錄 | run `artifacts/qwen3.5-4b/jspace-intervention-calibration/runs/` 下的 test runs（Technology 11、Financial 12）。 |
| 校準劑量與對照組記錄 | runs `jspace-v2-swap-*`、`jspace-v3-control-*`。 |
| 圖表生成 | 腳本 `scripts/render_jspace_intervention_report_figures.py`，圖表於 `docs/assets/jspace-sector-intervention/`。 |

**本次編輯說明：** 本報告按三項核心問題改寫，明確載明方向與位置特異性未通過之數值；原始協議 `proposal.md` 完整保留。
