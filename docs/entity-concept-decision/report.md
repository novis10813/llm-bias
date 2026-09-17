# 公司身分的中間概念：概念前提否決與 Stance 通道收斂描述

**狀態：已收線（探索收線）。** 模型為 Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構，32 層）；2026-09-15 收線，原三階段 pipeline 前提被否決、未實作，formal 從未授權。

**一句話發現：** 跨層與跨公司檢驗一致否決「存在可與立場分離之公司身分中間概念」之假設；1D stance 軸在 L15 解釋 60.5% margin 變異但因果極弱（屬描述讀數而非因果把手）；軸分解證實 16 家 L15 狀態 99.5% 共享，決策實為 0.5% 微小公司差的高增益非線性讀出。

## 1. 模型內部是否存在可與評價立場分離的公司身分中間概念？多輪一致否決

我們檢驗「在相同財務證據下，公司身分是否透過可獨立辨識的中間概念（如集中度 C、地理多元度 G）影響決策」：

**觀察：**
- **Round 1/2 概念正交化檢驗**：候選 C 方向 88% 是通用 stance 軸（$\cos = -0.88$）；刻意挑選之「stance 中性」地理多元度 G 與 stance 相關達 0.62，正交化後四家公司全部坍縮至同一點。
- **Layer-scan（L8–L26 完整空間 7 層）與 16 家擴展**：跨層掃描顯示國內 vs 全球公司在各層的幾何差異均趨近於 0，移除 stance 後 16 家公司完全無法分開。

**解讀：** L15 的 $k=8$ 子空間及各層殘差空間均不承載可從通用評價立場獨立分離的公司身分概念，原假設被否決，研究問題轉向 stance 通道本身。

## 2. 1D Stance 通道是什麼，它與實體訊號的關係為何？L15-localized 決策讀數，非實體訊號

**觀察：**
- **單一 Stance 軸在 L15 解釋 60.5% 變異**：由中立評價對獨立 fit 的 eval-stance 方向，在 L15 解釋 16 家公司 buy/sell margin 變異的 60.5%（$R^2 = 0.605$，$r = 0.778$，95% CI [0.21, 0.84]），層輪廓在 L15 達到全局峰值（L8 0.005 $\to$ L15 0.605 $\to$ L19–26 0.30–0.44）。
- **撤回 8D 假象**：初報 $k=8$ 迴歸 $R^2 = 0.953$ 經 5-fold CV 證實為 overfit 假象（CV $R^2$ 為負），可靠核心為 1D stance 方向。
- **Stance 軸垂直於身分主軸**：stance 軸與公司間主差軸 PC1 幾乎正交（$\cos = 0.003$），僅占公司間狀態變異 **1.14%**。

**解讀：** Stance 通道是（近）1D、在 L15 高度局域化的決策相關變數，但它不是 entity signal 的重新參數化，而是另一個「預測決策」的薄軸。

## 3. Stance 軸與公司間微小差異如何驅動決策？因果極弱，決策為高增益非線性讀出

**觀察：**
- **因果探查：弱因果槓桿**：直接對 stance 軸施加加性 push（$\pm 0.02 \sim 0.10$），6 個劑量格符號全部符合，但最大劑量（約 5.5 倍公司間差）僅移動 margin **+0.010 nats**（零翻轉）；實測因果斜率約 0.10 nats/單位，僅為表觀相關斜率（≈62）的 **1/600**。
- **軸分解：高增益非線性讀出**：
  1. 16 家的 L15 instruction 狀態 **99.5% 為共享內容**（公司間差僅占狀態范數的 0.52%）；
  2. 決策局部敏感度 $\|w\| \approx 12.9$ nats/單位且高度分散，內容方向僅佔敏感度范數平方（$\|w\|^2$）的 0.18%（公司間差 0.15%、stance 0.01%、均值 0.03%），99.8% 分散於非內容方向；
  3. 局部線性預測僅能解釋約 0.03 nats 的 margin 展開（實際為 1.66 nats，即 **~2%**）；公司間決策差異主要來自下游對 0.5% 微小差異的**高增益（50–100×）非線性讀出**。

**解讀：** 模型內部不存在低維局部線性因果軸介導「公司身分 $\to$ 決策」；stance 是決策走向的描述性讀數而非強槓桿，全人口 sell 先驗由 99.5% 的共享內容設定。

## 4. 研究宣稱之邊界與未涵蓋事項

1. **樣本規模局限於 16 家企業**：結論基於 2A test split 的 16 家公司，未外推全域企業庫。
2. **單模型與英文模板**：基於 Qwen3.5-4B 與英文財務模板，未外推跨模型。
3. **局部線性化屬量級估計**：$\|w\|$ 為局部權重估計，非全局精確線性模型。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| 原三階段協議與材料設計 | [Phase 1 協議](details/proposal-phase1.md)；[Phase 1 V2](details/proposal-phase1-v2.md)；[Phase 2](details/proposal-phase2.md)；[Phase 3](details/proposal-phase3.md)；[材料設計與驗證](details/design-and-validation.md)。 |
| 公司概念否決報告 | [Round 1 smoke-03](details/report-phase1-v2-development-smoke-03.md)；[Round 2 CG 報告](details/report-phase1-v2-round2-cg.md)；[Layer scan 報告](details/report-phase1-v2-layerscan.md)；[Broadened 16 家報告](details/report-phase1-v2-layerscan-broad16.md)。 |
| Stance 通道與因果探查 | [Stance 特徵描述報告](details/report-stance-characterization.md)；[Stance 因果探查報告](details/report-stance-causal.md)。 |
| 軸分解分析 | [軸分解報告](details/report-axis-decomposition.md)；run `phase1-v2-axis-decomp16-01`。 |

**本次編輯說明：** 本報告依既有紀錄按三項核心問題改寫，移除純導覽的頂層 proposal，直接以本報告為閱讀入口；不變更任何原始數據、門檻或執行歷史。
