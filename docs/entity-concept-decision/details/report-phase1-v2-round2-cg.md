# Round 2 報告：stance 中性概念（G）與 stance 平衡概念（C）

**run**：`phase1-v2-round2-cg-smoke-01`（`artifacts/qwen3.5-4b/entity-concept-decision-development/runs/phase1-v2-round2-cg-smoke-01`，RTX 3060 12GB，bf16，32 層，layer 15，k=8 basis，seed 1729，37 forwards，bit-exact repeat 通過）。
**scientific_status**：`not_evaluated`。**purpose**：`development`。本報告不含正式 gate。

## 目的

Round 1 的 smoke-03 校正指出：固定 L15 k=8 子空間主要是通用 stance／評價通道，「客戶集中度」的讀出是 stance 產物。Round 2 用兩個對照概念做決定性測試，直接問：**k=8 是否承載「可從 stance 分離的公司相關概念」？**

- **G（地理多元度，設為 stance 中性）**：多國營運 vs 單一市場。假設一個模型不太會有好壞評價的規模事實，若能乾淨讀出（與 stance 正交、勝 random、把公司分開），才支持 k=8 承載公司概念。
- **C（客戶集中度，stance 平衡版）**：把正負兩極寫成同等正面（皆「financially healthy」），只改集中度事實。若控住明示 stance 後集中度仍能解耦（低 cos、正交化後不塌、公司可分），才支持集中度是獨立於 stance 的訊號。

材料見 [materials-round2-g.md](materials-round2-g.md) 與 [materials-round2-c.md](materials-round2-c.md)。runner 已泛化以接受任意概念集合（`concept_ids`），並在分析階段做通用 stance residualization。

## 結果（皆為描述統計，非正式判定）

### 1. G（stance 中性地理多元度）

- **cos_with_stance = +0.618**：即便是設為中性的地理多元度，仍與通用 stance 方向有 0.62 的正相關（模型把「全球佈局」讀成較正面）。G 的來源差分只有 14.5% 投影進 k=8（`retained_fraction` 0.145）。
- **leave-one-group-out primary 未勝 random**：G1 0.0180、G2 0.0158、G3 0.0190，全部**低於** random primary q95（0.0228，96 deltas）。即地理多元度在 held-out 上沒有可勝過隨機方向的讀出。
- **公司不分開**：raw 分數 IT −1.453、BLK −1.413、BDX −1.450、NSC −1.433（spread 0.040），順序**不反映真實地理多元**（NSC 實質單一國家，卻排在中間）；stance 正交化後四家全部塌到 ≈ −2.25（無可分差異）。

### 2. C（stance 平衡集中度）

- **cos_with_stance = −0.860**：把兩極都寫成「financially healthy」之後，集中度方向仍與通用 stance 方向有 0.86 的反相關。明示 stance 平衡**沒有**把集中度與 stance 解耦——模型仍把「客戶集中」讀成風險。
- **leave-one-group-out primary 勝 random**：C1 0.0453、C2 0.0446、C3 0.0358，全部**高於** random primary q95（0.0228）。集中度（較顯眼的財務概念）在 held-out 上確有勝過隨機的讀出。
- **公司不分開**：raw 分數 IT 0.399、BDX 0.394、NSC 0.373、BLK 0.345，rank **精確等於 sell 立場強度**（IT > BDX > NSC > BLK，與 margins −3.18/−2.98/−1.90/−1.52 同序）；stance 正交化後四家全部塌到 ≈ +1.65（無可分差異）。

### 對照表

| 概念 | cos_with_stance | LO-group 勝 random？ | 正交化後公司可分？ | raw 公司差異由何驅動 |
|---|---|---|---|---|
| G（中性） | +0.618 | 否（3/3 < q95） | 否（皆 ≈ −2.25） | 無真實地理訊號 |
| C（平衡） | −0.860 | 是（3/3 > q95） | 否（皆 ≈ +1.65） | sell 立場強度 |

## 結論

1. **L15 k=8 是通用 stance／評價通道，不是公司概念通道。** 無論設為中性（G）還是刻意平衡（C），概念方向都大幅綁在通用 stance 上（+0.62／−0.86），且**stance 正交化後四家公司在兩個軸上都塌到同一點**——没有任何可從 stance 分離的公司相關概念訊號殘留。
2. **round-1 的校正被獨立證實並擴充**：round-1 以為「客戶集中度」是 readable 公司概念，實為 stance 產物；round-2 進一步顯示，連一個刻意選的中性屬性（地理多元度）也會被 stance 通道吸收。
3. **方法學收益**：「勝 random 的可讀性」（C 是、G 否）與「stance 移除後的公司可分性」（兩者皆否）被清楚拆開。一個概念可以在 k=8 內被讀出（C 勝 random），但那個可讀成分仍主要是 stance，而非公司特定的概念內容。

## 材料設計限制（誠實記錄）

- **G 並非真正 stance 中性**：我假設「地理多元度」是模型沒有好壞觀點的規模事實，但模型其實把「全球佈局」讀成正面（cos +0.62）。在財務決策語境下，**大多數公司屬性都帶有模型隱含的 stance**，真正 stance 中性的概念極難找到。這使「找一個中性概念證明 k=8 承載公司概念」的路径本身受限。
- LO-group 只有 3 個群（每概念），比 round-1 的 8 家族弱；但結論由「正交化後公司全部塌到同點」主導，不依賴 LO-group 的顯著性。
- 四家公司各屬一業，跨產業驗證仍弱（且已被 stance 主導的發現取代）。

## 對研究計畫的意義（真正的方向分岔）

本線最初的方向源假設——「因果有效的 L15 k=8 公司狀態差子空間承載可獨立識別的公司概念」——**在 L15 上不被支持**。k=8 主要是 stance 通道。這是一個需要決定下一步的研究分岔：

- **選項 1（接受 L15-k=8 null，改方向源／層級）**：承認 k=8 是 stance 通道，改從其他位置找公司特定概念——例如不同層、或由 entity-span 差分（而非整句公司狀態差）導出的方向、或更高維的子空間。
- **選項 2（改研究問題本身）**：把「stance 通道」當作真正有因果作用的對象（它確實驅動決策），研究「模型如何用單一 stance 軸把各種財務屬性匯成交付決策」，而不是假設存在独立於 stance 的公司概念。
- **選項 3（擴大材料／概念池）**：再試更多概念與層級，確認「stance 通道主導」是 k=8／L15 的局部現象，還是模型-wide 的結構。

我暫未執行任何一個選項，等待你決定方向。
