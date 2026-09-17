# Evidence-insensitivity：S&P 500 全人口證據不敏感性與 Prior（Phase 1 報告）

**狀態：進行中。** 模型為 Qwen3.5-4B（bf16；Gemma-4-E2B 同步執行中）；2026-09-16 Phase 1 formal run 完成、四項 gate 全過（G-P1 parse 0.9998、G-P2 符號一致 0.9738、G-P3 決定論 0 mismatch / 0.0 ΔM、G-P4 分組 50/453）、分組表凍結；Phase 2 capture-layer 對比協議已凍結（待啟動）；Phase 3 上游因果定位規劃中。

**一句話發現：** 在 S&P 500 全人口（503 家）共用中立證據下，零證據 Sell 為普遍預設（503/503，平均 margin −5.40 nats）；僅 50 家（9.9%）決策跟隨證據極性，453 家（90.1%）固定判 Sell；項目順序展現強烈非對稱 recency 效應（正項在尾 69–90% 翻轉為 Buy）。

## 1. 在無證據與多空證據下，模型在全人口的基準決策行為為何？強烈 Sell Prior 壓制決策

我們在 503 家標普 500 成分股上，使用完全相同的中立多空證據梯級（主項 6/8/10/15%、次項 5%）測試模型的 Buy/Sell 生成決策與 logit margin：

**觀察：**
- **503/503 家零證據全數判 Sell**：在無任何財務資訊的純身分提示下，全人口無一例外輸出 Sell，平均 margin $M(\text{zero}) = -5.40$ nats（IQR 0.91）。
- **身分剝除（Anonymous）亦為強 Sell**：匿名提示詞之零證據 margin 為 −6.75 nats，顯示保守 Sell 為評估框架的通用底層預設。
- **全體 503 家在 logit 層均跟隨證據極性**：全人口對比 $C_c = M(\text{P15}) - M(\text{N15}) > 0$（範圍 0.66 至 2.04 nats）；自零證據至正向證據梯級（zero $\to$ P）之跨條件移動約 4.9 nats。

**解讀：** 證據極性在連續 logit 空間對每家公司都產生顯著推力，但離散決策閾值被強大的 Sell 先驗大幅抬高，導致 90.1% 的公司在正項在前時無法推過多空門檻。

## 2. 全人口的分組結構與產業分布為何？50 家跟隨極性，IT 顯著過代表

依據預先註冊之分組判準（$D(\text{N15}) = \text{sell}$ 且 $D(\text{P15}) = \text{buy}$）：

**觀察：**
- **凍結分組結構**：
  - **Evidence-responsive（響應組）**：**50 家（9.9%）**，包含 AAPL、AMZN、GOOG、MSFT、NVDA、JPM 等大型龍頭企業；
  - **Fixed-sell（固定賣出組）**：**453 家（90.1%）**；
  - **Fixed-buy / Mixed**：0 家（0%）。
- **產業顯著偏向**：Information Technology 佔 16/65（**24.6%**），遠高於人口平均基率 9.9%；Consumer Staples（0/33）與 Utilities（1/31）則極少或無跟隨。

**解讀：** 模型對公司身分的敏感度高度分化，只有少數高知名度或科技類企業能克服先驗阻力跟隨正面證據轉向 Buy。

## 3. 證據呈現順序如何影響決策？強烈、非對稱的近因效應（Recency）

在 100 家公司的 order-swap 臂中，我們將正項與負項的呈現順序對調（次項先、主項後）：

**觀察：**
- **366/800 對決策翻轉，全數為 sell $\to$ buy**：正項移到末尾時，引發大規模向 Buy 翻轉。
- **極不對稱的條件響應**：正面主導的 P 條件翻轉率極高（P6 69 家、P8 90 家、P10 88 家、P15 89 家）；而負面主導的 N 條件翻轉極少（N15 僅 2 家）。
- **位移幅度相當於極性效應**：P 條件（P8–P15）下順序對調引發之 $|\Delta M| \approx 1.30 \sim 1.35$ nats（P6 為 0.71 nats），與淨極性本身的效應量級相當。

**解讀：** 模型決策嚴重受最後出現的資訊主導（近因效應），且 Sell 是強吸引子（Attractor）：正項在尾強烈促成 Buy，但負項在尾無法翻轉 Sell。

## 4. 跨模型診斷與研究邊界

- **模型本體特性確認**：跨模型診斷（Llama-3.2-1B/3B、Gemma-4-E2B）證實：零證據保守 Sell 是評估模板的共有底層，但「對證據極性的跟隨結構與分化」是 Qwen 本體模型特質（Llama 全面固定保守，Gemma 則呈現反向的「有證據即 Buy」模式），排除了模板假象。
- **無因果宣稱**：Phase 1 僅為行為層面的全人口篩選與分組，不包含任何內部因果介入主張。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| Phase 1 協議與判準 | [Phase 1 協議](details/proposal-phase1.md)（Rev 2 frozen）。 |
| Phase 1 執行結果與分組名冊 | run `phase1-gpu-bf16-01`，位於 `artifacts/qwen3.5-4b/evidence-insensitivity/runs/`；完整 503 家分組表收錄於 `summary.json`；詳細報告見 [Phase 1 詳細報告](details/report-phase1.md)。 |
| 跨模型對照診斷 | [Cross-model 診斷報告](details/diagnostic-cross-model-probe.md)。 |
| 後續階段規劃 | [全線進度協議](proposal.md)；[Phase 2 協議](details/proposal-phase2.md)（Rev 1.1 frozen）。 |

**本次編輯說明：** 本報告按三項核心問題總結 Phase 1 正式結果與行為分組；全線進度與多階段協議在 `proposal.md` 完整保留。
