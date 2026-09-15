# 公司身分的中間概念研究線收線報告

**狀態**：收線（2026-09-15）
**對象模型**：Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構，32 層）
**核心結論**：原假設「公司身分透過可從通用 stance 分離的中間概念影響 buy/sell margin」在 L15 固定 k=8 子空間與 L8–L26 完整空間、4 家與 16 家公司下**一致被否決**——沒有可獨立辨識的公司概念。研究問題隨之轉向 **stance 通道本身**：它是一個 1D、在 L15 被讀出的決策相關方向（解釋 16 家公司 margin 變異 60%），但**不是** entity signal（與公司間差主軸幾乎垂直）、也**不是**強因果槓桿（5.5 倍公司間幅度只移 0.01 nats，完整狀態交換移 0.60 nats）。最終軸分解顯示：16 家的 L15 instruction 狀態 **99.5% 是共享內容**（公司間差只有狀態范數的 0.52%），決策是這 0.5% 微差異的**高增益、非線性讀出**，局部因果權重 ‖w‖≈12.9 且**分散**（公司間差子空間 0.15%、stance 0.01%、共享均值 0.03%）。**不存在乾淨的低維局部線性因果軸**介導「公司身分 → 決策」；stance 是決策走向的**描述讀數**，不是**因果把手**。

**名詞定義**：

- **stance 方向**：由 round-2 evaluation 對（公司事實相同、只改 favorable/unfavorable）fit 出的「評價好壞」1D 方向；在本線各 run 中於完整層狀態空間獨立 fit。
- **margin（M）**：答案位置 buy 與 sell token 的 log-prob 差（$\log p(\text{buy})-\log p(\text{sell})$），nats。
- **company-contrast 子空間**：16 家公司 instruction-span 狀態差分的主成分張開的子空間（公司間差／entity-contrast）。
- **‖w‖（局部因果權重）**：L15 instruction-span 狀態對 margin 的局部（一階）敏感度的范數，以隨機 unit 方向 r 的 $E[(w\cdot r)^2]=\|w\|^2/d_{\text{model}}$ 估計。
- **additive push**：於指定層 block 輸出後、對 instruction-span 位置施加 $劑量\times unit方向$ 的加性干預。

## 1. 原假設被否決：L15 不承載可從 stance 分離的公司概念

在固定財務證據、只換公司身分的受控條件下，L15 的固定 k=8 有效狀態差與候選概念方向被逐輪检验：

- **Round 1（smoke-03）**：候選 C 方向 88% 是通用 stance 軸（cos_with_stance=−0.88），公司間差異消失。
- **Round 2（G 中性 + C stance 平衡）**：連刻意選的「stance 中性」地理多元度（G）都與 stance 方向 0.62 相關、LO-group 未勝 random、正交化後四家全部塌到同點；stance 平衡的集中度（C）仍 0.86 綁 stance。**決定性 null**。
- **Layer-scan（L8–L26 完整空間，7 層）**：把 null 擴充到所有掃描層。
- **Broadened 16-company（4/業）**：G 的國內（NSC/CSX）vs 全球公司在任何層的差仍近零、C 移除 stance 後 16 家全塌。**null 不是小池子偶然**。

使用者據此決定把研究問題轉向 stance 通道本身（不再找可分離公司概念）。原三階段 pipeline（概念→決策→上游）的 Phase 1 停止條件「無可辨識候選、只分辨一般正負立場」已滿足。

## 2. Stance 通道是一個 1D、L15-localize 的決策相關讀數方向

- **單一 eval-stance 方向在 L15 解釋 16 家 buy/sell margin 變異的 60%**（R²=0.605，r=0.778，95% CI [0.21, 0.84]，leave-one-out 穩定 0.54–0.67）。stance 方向由 evaluation 材料獨立定義（非 fit 到公司 margin），所以是真實相關、非 overfit。
- **層輪廓**：R² 在 L8≈0.005、L12 0.16、**L15 峰 0.605**、L19–L26 維持 0.30–0.44——stance 通道在 L12–L15（instruction／entity handoff 區）被寫入、L15 最強，與 selective-intervention 的因果有效層收斂。
- **撤回「決策是 8D k=8 現象」**：k=8 對 margin 的 OLS 初報 R²=0.953，經以存檔 16×8 座標重算＋5-fold CV 確認是 overfit 假象（in-sample 0.51–0.95 不穩、CV R² 負、逐軸 max 0.14）。可靠核心是 1D stance 方向。

## 3. Stance 方向 ≠ entity signal：一個薄的「決策預測」軸

- 16 家的 L15 instruction-span 狀態**99.5% 是共享的**：狀態范數 ≈10.0，公司間差 ‖state_c − 均值‖ ≈0.052（**0.52%** of 范數）。
- stance 軸與公司間差主成分**幾乎垂直**：cos(stance, PC1)=**0.003**、cos(PC2)=0.228、其餘 0.03–0.13；stance 只占公司間變異 **1.14%**；vs 2B 的 8 個 entity-pair contrast 全低（0.12–0.19）。
- → stance 不是 2A entity signal 的重新參數化，是另一個「決策預測」的薄軸。與 2A H4／investment-dial「L15 stance 非 entity-specific」一致，但**量化了結構**（薄、垂直於身分主軸）。

## 4. 因果探查：stance 軸是因果成分，但是弱槓桿

- 直接推 stance 軸（±0.02/0.05/0.10 × 16 家 × 4 隨機對照）：**6 個劑量格符號全部符合**（往 buy 推→margin 升、往 sell 推→降）；最大劑量下相對於同劑量隨機的配對差**顯著**（t=+3.18/−2.37）；隨機方向無此符號結構。**stance 軸不是旁觀讀數，它有因果作用**。
- **但很弱**：最大劑量（≈5.5 倍公司間 stance 差異）只把 margin 平均移動 **+0.010 nats**、**零家**翻轉；2A 2B 的完整狀態交換（同層同 span）移 **0.604 nats**——差約 60 倍。表觀相關斜率 ≈62 nats/單位 vs 實測因果斜率 ≈0.10——**約 1/600**。**強公司間相關主要是共變/讀數，不是強因果槓桿**。

## 5. 軸分解：決策是微小公司間差的高增益非線性讀出

- **決策局部權重大而分散**：‖w‖≈12.9 nats/單位（768 隨機方向測量，最大單筆占 2.3%，穩）；內容軸佔 ‖w‖²——公司間差子空間 **0.15%**、stance **0.01%**、共享均值 **0.03%**、結構化總和 **0.18%**。**99.8% 的局部決策敏感度不在這些內容方向上**。
- **個別內容軸局部斜率**（nats/單位，劑量上大致線性）：pc7 −0.30（t−9.8）、pc1 −0.25（t−8.3）、mean_state +0.21（t+6.3）、pc5 −0.19、pc6 −0.17、stance +0.13（t+3.6）、pc4 −0.11、pc8 +0.08、pc3 −0.07、pc2 +0.05；隨機 12 方向均值 −0.006、sd 0.089（中心在 0，無結構）。
- **讀出非線性**：隨機（非內容）方向每單位回應 @0.08=0.274、@0.40=0.086（**比值 0.31，飽和**）；內容軸大致線性（0.74–1.1）。
- **高增益**：局部線性讀出公司間差只預測 ≈0.03 nats margin spread，實際 1.66 nats——**~2%**。公司間決策差異 ~98% 是高增益（≈50–100×）、非線性/情境依賴地讀出那 0.5% 微差異。2B 狀態交換不對稱（朝好公司 IT→BLK +1.44、朝壞公司 BLK→IT +0.27）與此一致。
- **base-rate sell**：16 家均值 margin −2.30、全 sell，由**共享內容**（相同證據被讀成偏賣）設定，對所有公司相同。

## 6. 研究宣稱之邊界與未涵蓋事項

- **樣本規模局限於 16 家**（4/業，來自 2A test split），未向外擴展到全域企業庫；「公司間差主成分」是**這 16 家**的子空間，不是一般性 entity-identity 子空間。
- **單模型、單語言、單答案位置**：Qwen3.5-4B、英文財務模板、單一 buy/sell 答案位置；不外推至不同規模模型、多語言或多題平均。
- **單層（L15）、單 span（instruction span）**：高增益非線性讀出**發生在哪一層**未定位（2B 已知 instruction span 轉移峰在 L15，但未掃「公司間差敏感度」的層輪廓）。
- **局部線性化**：‖w‖ 是量級性局部權重（劑量混合、會飽和），不是精確線性模型；全部 development、非正式 audit、無 gate。
- **bf16**：所有觀測於 bf16；介入幅度遠大於量化步長（已排除精度偽象）。

## 7. 收線判定：原假設被否決，stance 通道取得收斂的描述

本研究線之核心任務「公司身分是否透過可獨立辨識的中間概念影響 buy/sell margin」已取得解答：
1. **概念層**：L15 k=8 與 L8–L26 完整空間**不承載**可從 stance 分離的公司概念（4 家與 16 家、多輪一致 null）——原假設被否決；
2. **stance 層**：stance 是 1D、L15-localize 的決策相關**讀數**方向（R²≈0.60），但**不是** entity signal、**不是**強因果槓桿；
3. **機制層**：決策是微小（0.5%）公司間差在 99.5% 共享背景上的**高增益非線性讀出**，局部權重分散——**不存在乾淨的低維局部線性因果軸**。

本線既定之研究問題已全數取得自洽解答，正式收線。後續若有新探索方向，應開立獨立研究線：
- **高增益非線性讀出的層/位置定位**：掃「公司間差敏感度」的層輪廓，找到把 0.5% 差放大成 1.66 nats 的層；
- **非線性劑量反應的直接量測**：內容軸上更密劑量網（0.01–0.2），畫放大/飽和曲線；
- **跨模型／跨語言**對公司身分決策偏誤之橫向比對。

## 產物與數據索引

- **原三階段協議（未執行，前提被否決）**：
  - 導覽：`docs/entity-concept-decision/proposal.md`
  - Phase 1：`details/proposal-phase1.md`／`details/proposal-phase1-v2.md`
  - Phase 2：`details/proposal-phase2.md`；Phase 3：`details/proposal-phase3.md`
  - 材料：`details/materials-round2-c.md`／`details/materials-round2-g.md`／`details/materials-phase1-v2-draft.md`
- **公司概念 null（Round 1/2 + layer-scan）**：
  - 報告：`details/report-phase1-v2-development-smoke-03.md`、`details/report-phase1-v2-round2-cg.md`、`details/report-phase1-v2-layerscan.md`、`details/report-phase1-v2-layerscan-broad16.md`
  - run：`artifacts/qwen3.5-4b/entity-concept-decision-development/runs/phase1-v2-round2-cg-smoke-01`、`entity-concept-layer-scan/runs/{phase1-v2-layerscan-01, phase1-v2-layerscan-broad16-01}`
- **Stance 通道（第一輪描述 + k=8 撤回）**：
  - 報告：`details/report-stance-characterization.md`
  - run：`entity-concept-layer-scan/runs/{phase1-v2-stance-char16-01, phase1-v2-stance-k8-16-01, phase1-v2-stance-decomp16-01}`
- **Stance vs entity 比較 + 因果探查 + 軸分解**：
  - 報告：`details/report-stance-causal.md`、`details/report-axis-decomposition.md`
  - run：`entity-concept-stance-causal/runs/phase1-v2-stance-causal16-01`（528 forwards）、`entity-concept-axis-decomposition/runs/phase1-v2-axis-decomp16-01`（1456 forwards）
- **程式與測試模組**：
  - 核心套件：`llm_bias/entity_concept_decision/`（`development.py`、`development_materials.py`、`layer_scan.py`、`attribution.py`、`stance_causal.py`、`axis_decomposition.py`）
  - 執行腳本：`scripts/entity_concept_development.py`、`scripts/entity_concept_layer_scan.py`、`scripts/entity_concept_attribution.py`、`scripts/entity_concept_stance_causal.py`、`scripts/entity_concept_axis_decomposition.py`、`scripts/entity_concept_stance_vs_entity.py`
  - 回歸測試：`tests/test_entity_concept_*.py`（新增 15 項含於全套 795 passed、4 項既有失敗與本線無關）
