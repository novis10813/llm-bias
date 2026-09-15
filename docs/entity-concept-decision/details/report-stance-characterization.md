# Stance 通道描述（第一輪）：它是接近 1D、在 L15 被讀出的決策變數

**run**：`phase1-v2-stance-char16-01`（`artifacts/qwen3.5-4b/entity-concept-layer-scan/runs/phase1-v2-stance-char16-01`，RTX 3060 12GB，bf16，32 層，seed 1729，49 forwards，16 家公司，bit-exact repeat 通過）。
**scientific_status**：`not_evaluated`。**purpose**：`development`。無正式 gate。

## 背景與問題

前三輪（round-1/round-2/layer-scan/broad16）一致顯示：固定 L15 k=8 與 L8–L26 完整空間都不承載可從 stance 分離的公司概念；模型以一個通用 stance／評價通道驅動 buy/sell。研究問題因此轉向 **stance 通道本身**。第一件要判定的事：**那個 stance 通道有沒有結構**——是不是接近 1D 的買/賣軸、在哪一層被讀出？這決定這條線值不值得深入研究。

## 方法

- **stance 方向**：用 evaluation 對（公司事實相同、只改 favorable/unfavorable）fit 出的「評價好壞」方向，每個候選層在完整空間獨立 fit。
- **決策變數**：16 家公司（4/業）的最終 buy/sell margin（答案位置 buy−sell logit 差）。
- **指標**：每層，stance 方向對 16 家公司的分數，與最終 margin 的 Pearson R²（n=16）。

## 結果

### 單一 stance 方向就能解釋決策的大部分變異，且在 L15 最強

| 層 | R²（stance 分數 vs margin） |
|---|---|
| 8 | 0.005 |
| 12 | 0.16 |
| **15** | **0.605** |
| 19 | 0.441 |
| 20 | 0.374 |
| 23 | 0.301 |
| 26 | 0.403 |

- **L15 的單一 eval-stance 方向解釋 16 家公司 buy/sell margin 變異的 60%**（r=0.78）。n=16 下 r 的 95% CI 約 [0.56, 0.99]，下界遠大於 0——是真實訊號，非小樣本假象（但估計不精確）。
- **層輪廓**：R² 在 L8 幾乎為零、L12 上升、**L15 達峰**、之後回落（L19–L26 維持 0.30–0.44）。即 stance 通道在 **L12–L15（instruction／entity handoff 區）被寫入，L15 最強**。
- **與因果證據收斂**：L15 正是 selective-intervention 已證實「因果有效」的決策子空間所在層。表徵層（stance 方向最強）與因果層（介入最有效）落在同一層。

### 直覺：微小的表徵差異被放大成決策差異

L15 上 16 家的 stance 分數 spread 只有 **0.018**（+0.2735~+0.2919），但 margin spread 是 **1.66**（−3.18~−1.52）。順序乾淨一致：**IT**（stance 最低 +0.2735）→ 最賣（−3.18）；**BLK**（stance 最高 +0.2899）→ 最不急著賣（−1.52）。即模型對每家公司的「好壞」有一個微妙的相對排序，那個排序被放大成買/賣決策。

### 兩個要分開的東西

1. **base-rate sell bias**：16 家 margin 全部為負（−1.52~−3.18），即模型對這批公司整體偏賣。這是一個**全域偏移**。
2. **per-company 排序**：公司在「多正面」上有序，那個順序（stance 方向）預測 margin 順序（L15 R²=0.60）。

stance 通道研究要同時處理這兩者：全域偏賣從哪來（instruction 的 prior？還是逐公司算出？），以及 per-company 排序如何形成、哪些事實寫入它。

## 補充：stance 方向 vs 因果有效的 k=8 子空間（`phase1-v2-stance-k8-16-01`、`phase1-v2-stance-decomp16-01`）——**已修正**

最初（`stance-k8-16-01`）曾報告「margin ~ 8 個 k8 座標的 OLS R²=0.953」，並據此推論「決策是 ≤8D 的 k=8 現象」。**這個推論已撤回**：

- 用同一份存檔的 16×8 k8 座標重跑 `_ols_r2`（SVD 穩定解）得到的是 **0.515**，不是 0.953；in-sample R² 在不同精確度/資料下於 0.51~0.95 之間不穩定。
- **5-fold 交叉驗證 R² 為負**（多個 seed 平均 −5~−107）：8 個高度共線的 k8 座標 fit 16 點，in-sample 高 R² 是 overfit，out-of-sample 完全不泛化。
- 逐軸看，**沒有任何單一 k8 軸與 margin 強相關**（各軸 R² 0.003~0.142）。

所以「k=8 捕捉 95% 決策變異」是 overfit 假象，**k=8 子空間並不可靠地預測 margin**。「stance vs 非 stance 分解 k=8 決策內容」因此失去基礎（它建立在已撤回的 k=8 決策含量上），只保留描述性意義。

## 補充:stance 方向 vs 2A 的 L15 entity signal——**不同的 object**（`analyze/stance_vs_entity_compare.json`，同 run 重算）

問題：stance 方向是不是 2A 線 L15 entity signal（公司身分差）的重新參數化？重算 48 支 prompt 的 L15 instruction-span 均值狀態（sanity:stance vs margin R²=0.6051 重現），比較：

| 面向 | 結果 |
|---|---|
| 公司間狀態變異的分佈 | 分散：PC1 只占 29%、PC2 16%、前 8 個 PC 才 79%，無單一主導軸 |
| cos(stance, PC1) | **0.003**（幾乎垂直）；cos(PC2)=0.228；其餘 0.03–0.13 |
| stance 軸占公司間變異 | **1.14%** |
| vs 2B 的 8 個 entity-pair contrast | cosine 全低（0.12–0.19） |

結論：**stance 方向不是 2A 的 entity signal**，是另一個「決策預測」的薄軸。它預測 margin（R²=0.605）但只占公司間狀態變異的 1.14%——公司身分的主體分散在其他「較不決策相關」的軸上。這與 2A H4／2C／investment-dial「L15 stance 非 entity-specific」一致，但量化了結構（薄、垂直於身分主軸）。這是新的：2A 做的是分散的 entity 定位，沒有找出這個單一的決策預測方向，也沒有特徵化它與 entity signal 的關係。

後續因果探查（[report-stance-causal.md](report-stance-causal.md)）進一步顯示：這根薄軸**有**因果作用但**很弱**（5.5 倍公司間幅度只移 0.01 nats；完整狀態交換移 0.60 nats）——強相關主要是共變/讀數，不是強槓桿。

## 可靠的核心發現（重新定位）

1. **stance 方向（1D、完整空間）可靠地預測 buy/sell margin**：L15 R²=0.605（r=0.778），95% CI R²=[0.21, 0.84]；**leave-one-out 穩定**（任剔一家，R² 落在 0.54~0.67，非單一公司驅動）。stance 方向由 evaluation 材料獨立定義（非 fit 到公司 margin），所以這是真實相關、非 overfit。
2. **stance 通道最好描述為一個 1D 方向（評價軸）**，不是 8D 子空間。層 profile 峰在 L15（0.605），與因果有效層收斂。
3. **k=8 子空間不是可靠的決策表示**：它是 entity-to-dial 的 Δs 子空間（因果有效，對「抹掉它會改變決策」意義上有效），但用 OLS 去解釋 16 家公司的 margin 變異時 overfit，不能當作「決策是 8D」的證據。

## 結論

1. **stance 方向是真實、可靠、（近）1D、層 localize（L15 峰）的決策相關方向**——解釋 ~60% 的公司間 buy/sell 變異（CI [0.21, 0.84]，leave-one-out 穩定）。這是本線目前最穩的正面發現。
2. **撤回「決策是 8D k=8 現象」**：那是 overfit 的 OLS 假象（CV R² 負、逐軸 max 0.14、in-sample R² 不穩定）。
3. **後續**（已執行，研究線已收線，見[收線報告](../report.md)）：(a) entity 比較 → stance 軸 ≠ entity signal（薄、垂直）；(b) 因果探查 → stance 軸有因果但弱（見 [report-stance-causal.md](report-stance-causal.md)）；(c)＋(d) base-rate sell 與決策因果主力 → 軸分解：base-rate 由共享內容設定、決策是微小公司間差的高增益非線性讀出（見 [report-axis-decomposition.md](report-axis-decomposition.md)）。

## 限制

- n=16：stance 的 1D CI 寬（R² [0.21, 0.84]），但 leave-one-out 穩定、方向獨立於 margin，所以是真實但精確度有限。
- k=8 基來自 entity-to-dial e-01 的 L15 Δs；它「因果有效」（selective-intervention 證實抹掉它改變決策），但「解釋 16 家 margin 變異 95%」是 overfit，不作為決策維度證據。
- stance 方向由 G/C 的 evaluation 對定義（「被告知好壞時的評價反應」），作為通用評價方向的 proxy。
- margin 是單一 buy/sell 答案位置，非多題平均。
