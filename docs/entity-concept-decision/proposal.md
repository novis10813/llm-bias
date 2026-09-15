# 公司身分的中間概念：從 L15 有效狀態差驗證決策路徑

**狀態：收線（2026-09-15）**，見[收線報告](report.md)。原假設（可分離公司概念）被多輪真實模型開發否決；研究問題曾轉向 stance 通道本身並取得收斂描述（1D stance 方向 L15 解釋 margin 變異 60%、≠ entity signal、因果探查為弱槓桿、軸分解顯示決策是微小公司間差的高增益非線性讀出）。本文件保留為分階段導覽與開發記錄索引；原三階段 pipeline 未執行，formal 從未授權。

## 1. 要驗證的是兩段連結，不是替八個方向命名

核心問題：在同一份財務證據下，公司身分是否透過可獨立辨識的中間概念影響 buy/sell margin？本線的「候選概念」指有人工定義、成對語句與獨立改寫驗證的內容差異；不是單一讀出 token，也不預設每個 SVD 基底對應一個概念。

- **公司身分 → 概念**：早層公司狀態置換，是否改變 L15 的候選概念分量？
- **概念 → 決策**：只置換該分量，是否改變公司相關決策差異？恢復原公司的該分量，是否削弱上游置換的作用？

成立時僅支持所測概念分量參與所測路徑。不能據此宣稱唯一中介、完整推理鏈、Bias 的全部來源，或把公司名稱效應直接等同有害偏見。未知概念不強行命名。

## 2. 三階段必須依序決定是否繼續

| 階段 | 原始協議 | 主要交付 | 停止條件 |
|---|---|---|---|
| 1：辨識概念 | [原始 Draft 1](details/proposal-phase1.md)／[V2 材料協議](details/proposal-phase1-v2.md) | 最多 3 個有獨立語義驗證的 L15 候選；固定材料、方向與 provenance | 無可辨識候選、只分辨詞面或一般正負立場，停止此輪 |
| 2：驗證決策作用 | [Phase 2](details/proposal-phase2.md) | 候選／剩餘分量／完整 k=8 與對照介入，建立公司相關作用 | 只見全局推注、效應不勝 controls，或證據不足，不進 Phase 3 |
| 3：驗證上游連結 | [Phase 3](details/proposal-phase3.md) | 早層公司置換與 L15 分量恢復的組合檢驗 | 上游效應不合格，或恢復不具特異性，不宣稱路徑成立 |

Draft 1 的文件核對與測試結果見[審查紀錄](details/review-draft1.md)（108 passed、1 個既有缺檔失敗；非實驗結果）。

每階段分開 `prepare → forward → analyze → finalize`，有自己的 config、run ID、gate 與批准。Phase 2/3 現為条件式設計，不能從 Phase 1 自動啟動。[共通契約與實作驗證](details/design-and-validation.md) 是各階段的規範性附件；數值門檻、資料名冊等凍結項未填齊時，formal 必須拒絕執行。

## 3. 現有證據支持起點，不保證語義與泛化

- Entity-to-Dial `entity-to-dial-e-01` 的 L15 k=8 effect ratio 中位數為 0.983，分母合格方向為 6/8；基底與評估來自相同研究配對，並非 98.3% 全域偏見解釋率。[原始 E 協議](../entity-to-dial/details/proposal-phase-e.md)
- `entity-to-dial-f-02` 的匿名推注為 `context_dependent_or_null`，只限所測劑量與情境，不證明概念一定不存在。[原始 F 協議](../entity-to-dial/details/proposal-phase-f.md)
- Selective-intervention V1 的 full-strength gate fail：縮小差異但產生全局與匿名位移。本線研究機制，不沿用其「控制成功」門檻。[V1 報告](../selective-intervention/report.md)
- J-space token **V1** 的 12 候選皆未通過篩選；本線不用單詞 transported 方向充當概念介入方向。[V1 報告](../jspace-token-experiments/details/report-v1.md)
- Entity Cell 的事實回想作用不等於決策作用；不強制將其作上游端點。[收線報告](../entity-cell-localization/report.md)

## 4. 先批准方法，再批准有限預算

三階段研究規劃與[Phase 1 首批實作 spec](details/implementation-phase1-foundations.md) 已分別獲使用者批准。無模型的上游驗證、投影與材料／方向純函式已實作並完成逐片驗收，見[交付紀錄](details/implementation-phase1-foundations-results.md)。開發級 model-injected workflow（材料轉換＋L15 k=8 方向 fitting＋LO-group／random 對照＋通用 stance 正交化）已實作並跑過真實 smoke：[smoke-01](details/report-phase1-v2-development-smoke.md) 初判 C 可讀出，但 [smoke-03](details/report-phase1-v2-development-smoke-03.md) 以 stance 正交化修正——C 方向 88% 是通用 stance 軸、公司間差異消失。[Round 2（G 中性 + C stance 平衡）](details/report-phase1-v2-round2-cg.md) 決定性地確認並擴充該結論：連刻意選的「stance 中性」地理多元度（G）都與 stance 方向 0.62 相關、LO-group 未勝 random、正交化後四家全部塌到同點；stance 平衡的集中度（C）仍 0.86 綁 stance、raw 公司 rank 精確等於 sell 立場強度、正交化後同樣不分開。**即固定 L15 k=8 子空間是通用 stance／評價通道，不承載可從 stance 分離的公司相關概念。** [Layer-scan（L8–L26 完整空間，7 層）](details/report-phase1-v2-layerscan.md) 將該 null 擴充到所有掃描層；[broadened 16-company layer-scan](details/report-phase1-v2-layerscan-broad16.md) 再將公司池從 4 家（各一業、G 1-vs-3）擴到 16 家（4/業、G 2-vs-14），G 的國內（NSC/CSX）vs 全球公司在任何層的差仍近零、C 移除 stance 後 16 家全塌——**即 null 不是小池子偶然**。原前提在 L15 k=8 與 L8–L26 完整空間（4 家與 16 家）皆被否決，模型以單一 stance 通道驅動決策。**使用者已決定把研究問題轉向 stance 通道本身**（不再找可分離公司概念）。[Stance 通道第一輪描述](details/report-stance-characterization.md)：單一 eval-stance 方向在 **L15 解釋 16 家 buy/sell margin 變異的 60%**（層輪廓 L8≈0→L15 峰 0.60→L19–26 0.30–0.44），與 selective-intervention 的因果有效層收斂於 L15；stance 通道是（近）1D、層 localize 的決策變數。k=8 對 margin 的 OLS（`phase1-v2-stance-k8-16-01`）初報 R²=0.953，經以存檔 16×8 座標重算＋ 5-fold CV 確認是 **overfit 假象**（in-sample R² 0.51~0.95 不穩、CV R² 負、逐軸 max 0.14），**撤回「決策是 8D k=8 現象」**；可靠的核心是 **1D stance 方向**（R²≈0.60，CI [0.21,0.84]，leave-one-out 穩定，L15 峰）。[Stance 方向 vs 2A L15 entity signal 比較]：stance 軸**不是** entity signal 的重新參數化——它與 PC1（公司間主差軸）幾乎垂直（cos 0.003）、只占公司間狀態變異 1.14%，是另一個「決策預測」的薄軸（與 2A H4／investment-dial「L15 stance 非 entity-specific」一致但量化了結構）。[Stance 軸因果探查](details/report-stance-causal.md)（`phase1-v2-stance-causal16-01`，528 forwards）：直接推這根軸**有**因果作用（6 個劑量格符號全對、配對 t=+3.18/−2.37 顯著於同劑量隨機方向），但**很弱**（5.5 倍公司間幅度只移 0.01 nats、零翻轉；2B 完整狀態交換移 0.60 nats，差約 60 倍；表觀相關斜率 ≈62 vs 實測因果斜率 ≈0.10 nats/單位，約 1/600）——**強公司間相關主要是共變/讀數，決策的因果主力在 L15 instruction 狀態的更寬成分上**。均為 development、非正式 audit。正式 pipeline 與各 gate 另行定版。每步都要獨立批准真實模型執行；本提案不包含算力授權。先量測每筆 forward 時間與峰值記憶體，再用 config 展開後的唯一 forward 數估算預算，不沿用舊 run 的分鐘數。

**軸分解（已執行，`phase1-v2-axis-decomp16-01`，1456 forwards）**：把 L15 instruction 狀態對 margin 的局部因果權重分解到成分軸（[report](details/report-axis-decomposition.md)）——16 家的 L15 狀態 **99.5% 是共享內容**（公司間差只有狀態范數的 0.52%）；決策局部權重 ‖w‖≈12.9 nats/單位且**分散**（公司間差子空間 0.15%、stance 軸 0.01%、共享均值 0.03%）；讀出**非線性**（隨機方向劑量飽和、內容軸大致線性）；公司間 1.66 nats 的決策差異只有 ~2% 能被局部線性讀出——**決策是微小公司間差異的高增益非線性讀出，不存在乾淨的低維局部線性因果軸**。stance 是有用的描述讀數（R²=0.605）但不是槓桿（0.01%）。base-rate sell 由共享內容設定。

Phase 1 V2 的兩候選與 64 句製作預算已批准；[待審材料](details/materials-phase1-v2-draft.md)不是 approved inputs，不含 audit，無模型結果。原始 Draft 1 保留歷史預算；V2 分開記錄控制與分割調整。

下列項目仍為 **freeze blockers**：候選概念材料與人工標記、全新公司名冊、模板家族分割、概念可辨識與最小決策效應門檻、數值誤差上限、樣本量／精確度目標及各階段專屬 CLI。它們需要 calibration 或人工審查，不能假裝已由 codebase 決定。

## 5. 上游關係與文件責任

| 上游 | 關係 | 使用範圍 |
|---|---|---|
| Balanced Evidence Gap Phase 2A | 資料／產物依賴 | 舊 64 prompts 僅作 discovery 與重現參照 |
| Entity-to-Dial Phase E | 資料／產物依賴 | 固定前 8 個右奇異向量；不重新 fitting 舊基底 |
| Jacobian-lens selection | 資料／產物依賴 | 唯一 validated canonical lens；僅語義提名需要 |
| Entity Cell、Selective-intervention V1、J-space token V1 | 研究承接／方法參考 | null、對照與解讀限制，不新增跨實驗程式 import |

回到[研究總覽](../README.md)。本線不改上游 frozen 文件、不新增虛構 report；有實際 run 才寫結果。
