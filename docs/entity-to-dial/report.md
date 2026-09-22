# Entity-to-Dial：L15 狀態差子空間與雙通道匯流飽和

**狀態：Phase A–F 已收線；held-out transfer V1 已完成。** 模型為 Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構，32 層）；2026-09-13 收線，Phase E1 通過、Phase E2b/F1 未通過（雙通道表示被否決），最終採納 L15 instruction span 的 $k=8$ 殘差子空間描述收線；2026-09-22 完成 frozen $V_8$ 的 200-company held-out fixed-margin transfer evaluation（未達 0.8 high-recovery target）。

**一句話發現：** 在 Phase E 的 16 家 construction set 上，L15 指令區間的 $k=8$ 殘差子空間可恢復 98.3% 的 fixed-margin 置換效果；但 frozen $V_8$ 在全 sector 200-company held-out evaluation 的 median recovery 為 sector-within 0.6759、cross-sector 0.6596，均低於 0.8 target，故只支持 construction-set state-transfer sufficiency，不支持該 high-recovery 程度的 entity-held-out generalisation。單一 Dial 瓶頸與「$v_1$ + Dial」可加表示仍未通過既有檢驗。

## 1. 實體訊號是否經由實體 token 或單一 Dial 瓶頸傳遞？均被排除

在多空平衡財務證據下，我們追查實體訊號自底層（L0–11）進入決策層的因果路徑：

**觀察：**
- **實體 Token 區間在高層失去充分性（Phase A，Gate A1 fail）**：在 L12 至 L18 跨實體置換 ticker 或 name token，normalized transfer 均值全數落於 0 附近（95% 信賴區間均涵蓋 0），訊號在 L12 前已移出實體字元位置。
- **單一 Block 增量未主導轉移（Phase B，Gate B fail）**：在 L12–15 交接窗口，單獨置換 Attention 或 MLP block 的增量差值，transfer 均 < 0.20；但在 L15 指令區間置換 full residual 時，決策轉移達到全局峰值（+0.604 nats）。
- **Dial 單坐標無法承載主體（Phase C，Gate C fail）**：移植 source 實體的 L15/N8490 dial 激活值至 target 實體，未解釋落差高達 90% 以上；dial 屬通用立場調節，非實體偏誤專屬瓶頸。
- **Raw Channel 介入重現抹除簽章（Phase D，Gate D1/D2 fail）**：L12–18 指令區間 MLP down-projection 的 block patch 效應一致偏向 sell（8 方向均值 ≈ 0），且 19 個 top channel 的產業符號一致性多數僅 0/4 或 1/4，缺乏跨產業一致因果機制。

**解讀：** 實體訊號不走實體 token，亦不走單一 dial 瓶頸，而轉由指令上下文區間（instruction span）的殘差狀態差值承載。

## 2. L15 指令區間狀態差的幾何結構為何？8 維殘差子空間承載 98.3% 充分性

在轉移峰值層 L15 解剖指令區間狀態差值（Phase E）：

**觀察：**
- **Block Delta 組合具自洽因果充分性（Gate E1 pass）**：L15 Attention 與 MLP 增量差值相加的 joint patch，effect ratio 中位數達 **0.575**（$\ge 0.5$），勝出 pre-L 殘留假說（$H_{\text{carryover}} \approx 0.187$）。
- **狀態差值低維集中（E2a 降維曲線）**：對 8 個方向狀態差值實施 PCA，$k=1$ 恢復 73.7%，$k=3$ 達 88.5%，**$k=8$ 達 98.3%**（已飽和 full swap），$k=16$ 達 100.1%。2560 維殘差流中的實體資訊實質約束於 8 維線性子空間內。
- **Dial 構成正交平行通道（Gate E2b fail 但具實質效應）**：位置受限 dial 通道移植達成 43.4% 效應比值（中位數 0.434 < 0.5 未達 gate）；事後幾何確認 $\cos(v_1, \text{dial\_footprint}) = \mathbf{-0.020}$，兩通道幾何嚴格正交。
- **立場轉移不對稱性**：4 個 bottom→top 方向（sell 為 source）轉移極強（+0.95 至 +1.44 nats），而 4 個 top→bottom 方向轉移微弱或反向（+0.27 至 −0.26 nats），係由 pre-L 狀態差扮演抑制煞車。

**解讀：** L15 狀態差值高度低維集中於 8 維子空間，Dial 是與其正交但平行的同效應通道。

## 3. 「$v_1$ + Dial」是否構成線性可加的雙通道表示？否決，下游讀出飽和匯流

為了驗證「$v_1$（殘差方向）+ dial（MLP channel）」的極致雙通道表示，實施單一 forward 聯合介入與中性 push 檢驗（Phase F）：

**觀察：**
- **加法性檢驗（Gate F1 fail）**：在 6 個有效方向上，dual-hook 聯合介入的 additivity ratio 中位數為 **0.7329**（$< 0.85$ 門檻，Gate F1 fail）。4 個 bottom→top 方向的 additive residual 全數為負（**−0.491 至 −0.550 nats**），combined 效果量甚至小於 $v_1$ 單臂。
- **匿名推注無本征立場 Loading（H_F2 描述性判定）**：在中性匿名文本施加 $\pm\alpha \cdot \text{push\_base}$ 介入，$v_1$ 與 dial footprint 兩臂的全部 8 個判定點位移幅度均落於 $|\Delta M| \le 0.05$ nats 擾動帶內，判定為 **`context_dependent_or_null`**。

**解讀：** 兩通道在幾何上雖正交，但在下游決策讀出時競爭同一非線性飽和帶，無法維持線性加性；$v_1$ 脫離上下文後無獨立立場 loading。依協議 fallback，否決雙通道緊湊表示，**採納 $k=8$ 殘差子空間（恢復 98.3% 效應量）作為 L15 段完整因果描述**。

## 4. Frozen $V_8$ 在 200 家 held-out entities 上只保留部分 transfer

Held-out Transfer V1 從 2024 S&P 500 的 503 家中，排除 Phase E construction 的 16 家與 M6 external-removal 的 12 家，對餘下 476 家按 GICS sector 比例抽取 200 家。selection variant 的 clean margin 僅用於建立 pair graph；另三個 frozen variants 評估 full L15 state swap、frozen $V_8$ projected transplant、四個 random 8D projector 與 self-source no-op。formal run `entity-to-dial-heldout-transfer-v1-01` 的 1,200 no-op records 全為 0.0，394 directed transfers 均有完整 seven-arm × three-variant matrix。

| Stratum | Eligible directed transfers | Excluded（full reference < 0.2 nats） | $V_8$ median recovery（95% pair-bundle bootstrap CI） | $V_8$ minus median random-8D ratio（95% CI） | Predeclared status |
|---|---:|---:|---:|---:|---|
| Sector-within | 124 | 70 | 0.6759 [0.6420, 0.7033] | 0.6656 [0.6450, 0.6929] | not supported |
| Cross-sector | 140 | 60 | 0.6596 [0.6343, 0.7089] | 0.6629 [0.6321, 0.6982] | not supported |

兩個 stratum 的 random-control advantage 均為正，但 median recovery 與 CI lower bound 都沒有達到 frozen 0.8 high-recovery target。因此，frozen $V_8$ 比任意同維 random projector 更保留 source→target fixed-margin transfer，卻沒有把 Phase E construction-set 的近飽和 recovery 外推到這個 held-out population。high→low 與 low→high 的 ratio 都低於 0.8：sector-within 為 0.6754／0.6764，cross-sector 為 0.6764／0.6569。

此結果只量 fixed-answer Buy/Sell continuation margin；沒有生成 decision-flip outcome，也不證明 generated decision 改變。它不否定 L15 state transfer 對 construction directions 的充分性，亦不支持 universal $V_8$ coordinate system、完整 entity circuit 或跨模型結論。

## 5. 研究宣稱之邊界與未涵蓋事項

1. **樣本與方向母體限制**：Phase E construction-set 結論基於 16 家標竿企業與 8 個極端對比方向（TOP = {NSC, BLK}，BOTTOM = {IT, BDX}）。Held-out V1 擴展至按 2024 S&P 500 sector composition 抽取的 200 家，但只測 Qwen3.5-4B、frozen shared-evidence prompts、L15 $V_8$ 與 pair-specific transplant。
2. **非電路級完全拆解**：dual-hook 與 residual transplant 屬狀態介入，不能還原內部計算圖突觸權重。
3. **模型架構專一性**：結論針對 Qwen3.5-4B（Gated DeltaNet 混合架構），不外推其他純 Transformer 模型。
4. **行為層限制**：Held-out V1 的 margin transfer 不能當作 generated decision effect；source→target transplant 若要提出行為層主張，仍需另立生成協議。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| Phase A–C 三段式 discovery | [Phase A–C 協議](details/proposal-phase-abc.md)；[報告](details/report-phase-abc.md)；run a-01 / b-01 / c-01。 |
| Phase D 指令區間 block 掃描 | [Phase D 協議](details/proposal-phase-d.md)；run `entity-to-dial-d-02`（D1/D2 結果收錄於該協議）。 |
| Phase E 雙 block 與 SVD 降維 | [Phase E 協議](details/proposal-phase-e.md)；run `entity-to-dial-e-01`（E1/E2 結果收錄於該協議，含 $k=8$ 基礎矩陣）。 |
| Phase F 雙通道加法性與匿名推注 | [Phase F 協議](details/proposal-phase-f.md)；run `entity-to-dial-f-02`（F1/F2 結果收錄於該協議）。 |
| Frozen $V_8$ entity-held-out transfer | [Held-out Transfer V1 協議](details/proposal-heldout-transfer-v1.md)；run `entity-to-dial-heldout-transfer-v1-01`（200-company sector-within / cross-sector fixed-margin evaluation）。 |

**本次編輯說明：** 本報告依既有紀錄按三項核心問題改寫，移除純導覽的頂層 proposal，直接以本報告為閱讀入口；不變更任何原始數據、門檻或執行歷史。
