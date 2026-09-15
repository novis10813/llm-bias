# Broadened 16-company layer-scan 報告：null 是否只是小池子偶然

**run**：`phase1-v2-layerscan-broad16-01`（`artifacts/qwen3.5-4b/entity-concept-layer-scan/runs/phase1-v2-layerscan-broad16-01`，RTX 3060 12GB，bf16，32 層，seed 1729，49 forwards 各捕捉 8 層，bit-exact repeat 通過）。
**scientific_status**：`not_evaluated`。**purpose**：`development`。無正式 gate。

## 目的

[Layer-scan](report-phase1-v2-layerscan.md) 用 4 家公司（IT/BDX/NSC/BLK，各一業、G 只有 1 家國內 vs 3 家全球）已顯示 null。但 4 家、各一業、1-vs-3 的對照很弱——null 可能是小池子偶然。本 run 把公司池擴到 **16 家**（balanced-evidence-gap-2A 全池，4 家×4 業），且 G 變成更乾淨的 **2 家國內（NSC、CSX，皆美國鐵路）vs 14 家全球** 對照，以確認 null 是否穩健。

公司池（4/業）：Health Care（ABT/BDX/DHR/SYK）、Information Technology（AMAT/GLW/HPE/IT）、Financials（AXP/BLK/C/GS）、Industrials（CSX/DE/HON/NSC）。材料（G 地理多元、C 集中度）與候選層（L8/12/15/19/20/23/26）皆同 [layer-scan](report-phase1-v2-layerscan.md)。

## 結果

### G：公司間無地理多元差異（任何層、raw 與正交化皆然）

國內（NSC、CSX）vs 14 家全球的平均分數差，在**每一層**都近乎為零：

| 層 | cos_with_stance | raw 國內−全球 | ortho 國內−全球 |
|---|---|---|---|
| 8 | +0.231 | −0.001 | −0.001 |
| 12 | +0.142 | +0.001 | +0.001 |
| 15 | +0.143 | +0.001 | +0.002 |
| 19 | +0.133 | −0.002 | −0.003 |
| 20 | +0.137 | −0.003 | −0.003 |
| 23 | +0.093 | −0.006 | −0.006 |
| 26 | +0.091 | −0.006 | −0.006 |

全 16 家的 ortho spread 也只有 0.009–0.126（極小）。即：**即便有乾淨的 2-vs-14 對照與 4 倍大的公司池，模型在材料定義的 G 方向上仍幾乎不編碼公司間的地理多元差異**——NSC 與 CSX 沒有沿 G 軸脫離其餘 14 家（raw 或移除 stance 後皆然）。

### C：raw 公司差異追蹤 sell 立場，正交化後 16 家全塌

raw C 公司分數與 sell 立場（margin）的 Pearson 相關：L15 **−0.71**、L19 −0.64、L20 −0.56、L23 −0.45、L26 −0.35（中後層皆為中等至強負相關）；L8/L12 弱。移除 stance 後，16 家的 ortho spread 只有 0.008–0.033，**全部塌到同一點**（無任何公司可分）。

## 結論

1. **null 穩健，不是小池子偶然**：把公司池從 4 家（各一業、G 1-vs-3）擴到 16 家（4/業、G 2-vs-14），G 的公司分離仍是零、C 移除 stance 後仍不分開。
2. **合併證據（4 家 + 16 家、L8–L26、完整空間、G + C）**：Qwen3.5-4B 在這些財務概念上，用通用 stance／評價通道驅動 buy/sell，而不把公司特定財務屬性編碼成可從 stance 分離、可由材料辨識的獨立概念方向。G（最有利於公司分離、且有乾淨真實世界順序的概念）是最強的反證——它仍零分離。
3. **仍非字面「模型-wide 定理」**：這是「這些材料、這些 16 家、L8–L26」上的強 null，不是對每個概念/每家公司的數學證明。要繼續擴大（更多概念、更多公司、其他模型）可以，但上限受限：財務屬性大多帶隱含 stance，且真正 stance 中性的概念難找。

## 限制

- 公司池仍全部來自 balanced-evidence-gap-2A（同一起源、同一 instruction 格式）。
- G 的「2 國內 vs 14 全球」中，14 家全球彼此跨不同產業/規模；但正因如此，若模型真有公司特定 G 訊號，NSC/CSX 脫離的訊號反而更該被看見——它沒有。
- 材料定義的方向（非直接幾何測試）；見 [layer-scan 報告的限制](report-phase1-v2-layerscan.md#限制誠實記錄)。
