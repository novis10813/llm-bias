# Stance 軸因果探查:推天平會移動決策,方向對、有劑量依賴,但是弱槓桿

**run**:`phase1-v2-stance-causal16-01`(`artifacts/qwen3.5-4b/entity-concept-stance-causal/runs/phase1-v2-stance-causal16-01`,RTX 3060 12GB,bf16,32 層,seed 1729,528 forwards,16 家公司,8m43s)。
**scientific_status**:`not_evaluated`。**purpose**:`development`。無正式 gate,描述性探查。
**上游**:`phase1-v2-stance-char16-01`(prompts + materials 凍存,stance 方向同公式重 fit)、2A `phase2a-gpu-bf16-01` prompts。

## 問題

[stance 通道描述](report-stance-characterization.md)顯示 L15 的 1D stance 方向解釋 16 家公司 buy/sell margin 變異的 60%(R²=0.605),但那是**相關**。**entity 比較**(見該報告補充節)又顯示這根軸很薄:只占公司間狀態變異的 1.14%,與主要身分軸幾乎垂直。問題:這根軸是**控制**決策的因果成分,還是只是跟著變的讀數?若是因果,效力多大?

此前沒有任何一條線測過這根軸本身:2A Phase 3 測的是 entity coordinate(additive intervention,null);investment-dial 測的是單一 neuron;selective-intervention 測的是 entity-difference 子空間。

## 方法

- **介入**:在第 15 層 block 輸出後,把 `劑量 × 方向` 加到 instruction span 位置(與 stance 讀取位置相同;2B 的因果峰值也在 L15 instruction span)。
- **方向**:stance 軸(同 formula 重 fit:6 個 evaluation 對的均值差,unit 化;sha256 記錄於 forward metadata)＋ 4 個 seed 化的隨機 unit 方向(對照,同劑量)。
- **劑量**:±0.02 / ±0.05 / ±0.10(stance 分數單位;公司間 stance 全距只有 0.018,所以 0.10 ≈ 5.5 倍公司間差異)。
- **測量**:答案位置 buy−sell margin 的變化(ΔM),以及 p(buy) 二分類概率(看是否翻轉)。
- **精度**:state norm ≈ 10.0(instruction span 平均),推入分量遠大於 bf16 量化步長,精度已排除。

## 結果

### 1. 符號一致、劑量依賴、顯著於隨機對照

| 劑量 | stance ΔM(均值±sd) | 隨機 ΔM(均值±sd) | 配對 gap(t) |
|---|---|---|---|
| +0.10 | **+0.0103**(0.030) | −0.0023(0.031) | **+0.0126(t=+3.18)** |
| +0.05 | +0.0049(0.020) | +0.0025(0.027) | +0.0024(t=+0.64) |
| +0.02 | −0.0002(0.035) | −0.0013(0.026) | +0.0011(t=+0.20) |
| −0.02 | −0.0040(0.029) | −0.0007(0.030) | −0.0033(t=−0.66) |
| −0.05 | −0.0056(0.021) | +0.0034(0.028) | −0.0090(t=−1.87) |
| −0.10 | −0.0070(0.026) | +0.0072(0.029) | **−0.0141(t=−2.37)** |

- **6 個劑量格的符號全部符合預測**(往 buy 推 → margin 上升;往 sell 推 → 下降)。
- 最大劑量下,stance 軸相對於同劑量隨機方向的配對差**統計顯著**(t=±2.4~3.2);隨機方向在負劑量下符號混亂(無系統結構)。
- → stance 軸**不是旁觀讀數,它有因果作用**。這是本線第一個對這根軸的因果證據。

### 2. 但它是弱槓桿

- 最大劑量(+0.10 ≈ 5.5 倍公司間 stance 差異)只把 margin 平均移動 **+0.010 nats**;**零家**公司翻轉(16 家全部維持 sell)。
- 直接尺度對照:2A Phase 2B 把 L15 instruction span 狀態**整段換成另一家公司**(完整狀態交換)的 toward-source ΔM = **0.604 nats**。本 run 的 1D stance 推(5.5 倍幅度)只移動 0.010——**差約 60 倍**。
- 顯隱對照:公司間回歸的表觀斜率 ≈ 62 nats/單位 stance 分數;實測因果斜率 ≈ 0.10 nats/單位——**約 1/600**。強相關不等於強因果槓桿。

### 3. 公司間異質性大

逐公司看,推 stance 軸的反應不齊:
- HON、NSC:所有劑量(含負劑量)都往 buy 移(像一般性擾動效應,非劑量依賴)。
- AMA、BDX、DE、GLW:多數劑量往 sell 移。
- ABT、DHR、GS、IT、SYK:混合。
配對統計(每家公司自己的 stance−隨機差)已把這種公司特異的底層偏移扣除;上表是扣除後的軸特異效應。

## 詮釋

1. **stance 軸是因果成分,但贡献很小**:推它會同向移動決策(符號一致、劑量依賴、顯著於隨機),所以它不是純粹的表觀讀數。
2. **強的公司間相關(R²=0.605)主要反映共變,不是槓桿**:stance 分數隨公司狀態共變、同時 margin 也隨公司狀態變化,兩者共享上游,但直接推這根軸只能移動決策的一小塊。
3. **決策的因果主力在更寬的 L15 instruction 狀態**:完整狀態交換移動 0.60 nats,stance 軸(占公司間變異 1.14%)只移動 0.01 nats——量級與它的「薄」成比例。決策相關的狀態內容在 stance 軸之外的成分上。
4. 與既有結論一致:2A 與 investment-dial 都指出 L15 的 stance 是**通用**(非 entity-specific)的;本 run 量化了「通用」的因果含量——薄。

## 限制

- n=16;單層(L15)、單 span(instruction span)。位置選擇有依據(2B 因果峰值),但未测其他位置/層。
- 劑量最大 5.5 倍公司間差異,零翻轉;更大劑量未測(非線性/飽和未排除,劑量反應在 0.02→0.10 呈超線性增長)。
- 公司間異質性大,均值是混合;逐公司的軸效應未單獨建模。
- margin 是單一答案位置,非多題平均。

## 下一步（已執行）

本問題由軸分解回答：決策的局部因果權重**不在**身分差軸（0.15%）、stance 軸（0.01%）或共享均值方向（0.03%）上，而是**分散**的；公司間 1.66 nats 的決策差異是微小（0.5%）公司間差的**高增益非線性讀出**，base-rate sell 由共享內容設定。見 [report-axis-decomposition.md](report-axis-decomposition.md) 與[收線報告](../report.md)。
