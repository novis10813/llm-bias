# generalization-v1：company-disjoint 與複製已完成，sector-disjoint 不可概括到四模型

**狀態：**依 [事前協議](proposal.md)，四模型的 R9 `loso`、R10 `loso_construction` 和 R11 `split_seed` 在 `confirmation-v1-20260925-full-01` 全量完成；執行與同步來源見 [confirmation-v1/status.md](../confirmation-v1/status.md)。結果針對各模型**自建方向的 DIM**，不得外推到 cone。

各模型的 `artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-20260925-full-01/{loso,loso_construction,split_seed}/result.json` 均為 `complete=true`。固定 seed `20260923` 的 503 家母體分成 construction 402／evaluation 101，受測公司不參與 ranking、方向擬合或 CAL；R9 的 eval 內 94 家有明確 sector、7 家 Unspecified 不當 LOSO target。R11 另外用 `20260924`／`20260925` 兩個 seed 重新切分與擬合，逐模型保存 101 家受測列及 α0／生成結果。這是**切分複製**，不是 random-direction seed 對照。

## 預先登記的 C10 門檻

「sector-disjoint」需要同時符合：R9 在 `±α_50` 有分母的符號下，comparator−LOSO 的 company-bootstrap 95% CI **上界 <0.10**；R10 每個規定 sector（construction n≥20，或在**任一模型的** Top／Bottom10 中占≥3 家）在 LOSO fold 至少一個 on-target flip。此處按四模型的 sector 聯集核對，不只依單模型名單。以下 R9 數字取各模型 `loso/result.json → summary.contrasts`；無起始類別分母記「—」，不是通過該方向。

| 模型 | R9 −α₅₀／+α₅₀ 的 CI 上界 | R10 規定 sector 至少一 flip | C10 sector-disjoint 的可寫範圍 |
|---|---|---|---|
| Qwen3.5-4B | —／0.0000 | 是 | 僅有 sell→buy 分母的方向符合；buy→sell 未檢驗 |
| Gemma-4-12B | 0.0000／0.0909 | 是 | 兩方向在本 split／網格符合（0.0909 **低於** 0.10） |
| GLM-4-9B | 0.0000／— | 是 | 僅有 buy→sell 分母的方向符合；sell→buy 未檢驗 |
| GPT-OSS-20B | **0.1875**／0.0769 | 是 | 負向 R9 未過門檻；**不能**概括為此模型 sector-disjoint 達標 |

R10 的 required-sector 判定取四模型的規定 sector 聯集（本次為 Consumer Discretionary、Consumer Staples、Energy、Financials、Health Care、Industrials、Information Technology、Materials、Real Estate、Utilities），四模型這十個 sector 在 LOSO fold 各有至少一個 on-target flip；逐 sector 的 flips 保存在 `loso_construction/result.json → summary.per_sector`，不能把整體效果率取代「每個 sector 至少一 flip」。Qwen、GLM 的另一符號為零分母；在論文只寫已測方向，而不寫「兩向均達標」。所有四模型的 **company-disjoint held-out** 可依實際切分稱呼；LOSO 只支持上述條件式的 DIM sector 表述，不把舊 200-company 5/5 construction-overlap 筆記升格為 held-out cone 結果。
