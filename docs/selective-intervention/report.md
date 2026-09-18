# Selective-intervention：推論期移除 L15 k=8 子空間（全強度負結果）

**狀態：已完成（V1 formal；M6-V2 compatibility formal）。** V1 使用 Qwen3.5-4B（bf16）於 2026-09-14 完成，formal run `selective-intervention-v1-gpu-bf16-01`，門檻判定為 `fail`（G1a/G1b/G2 pass、G3/G4 fail；full-strength 負結果）。M6-V2 使用目前 Hugging Face default runtime 於 GPU1 完成，run `m6-v2-gpu1-01`；它不改寫 V1，也不宣稱 exact V1 center compatibility。

**一句話發現：** V1 在原始 population 內成功使公司間決策落差減半但因全局副作用收口為 full-strength 負結果；M6-V2 在外部 12 家 current-runtime compatibility population 的 spread ratio 為 **0.6401**（95% CI **[0.2837, 2.1144]**），未達預註冊的 ≤0.5 primary gate，且 G3′/G4′ 均 fail，因此沒有確認外部機制泛化。

## M6-V2 外部 population compatibility validation

- **Population**：12 家、4 sectors、48 prompts；selection manifest 固定 seed 42。
- **Primary**：clean IQR = 0.3364，full-strength main IQR = 0.2153，spread ratio = **0.6401**，reduction = 35.99%。paired company bootstrap（10,000、seed 42）95% CI = **[0.2837, 2.1144]**。
- **Interpretation**：`fail`（point estimate > 0.5）；不是 confirmed generalization。
- **Specificity**：random control pass；random reduction = −2.00%，相對主臂縮減未達 25% 門檻。
- **Safety diagnostics**：G3′ fail（mean shift +0.4546 nats）、G4′ fail（anonymous shift −0.2557 nats）。兩者為 secondary diagnostics，不改變 primary 判定。
- **Greedy generation**：48/48 comparable prompts，main sell→buy flip rate = 0。
- **Compatibility boundary**：V2 current center digest `f1fcf6b4...` 不等於 V1 artifact digest `99e5dcfb...`；因此本結果不得被描述為 exact frozen-V1-center replication。

Formal artifact：`artifacts/qwen3.5-4b/selective-intervention-m6-v2/runs/m6-v2-gpu1-01/`。
Protocol：[M6-V2 protocol](details/proposal-m6-v2.md)。

## 1. 移除 L15 $k=8$ 實體差異子空間能否縮減決策落差，且具備特異性？有效性與特異性雙陽性

我们在推論期對 L15 指令區間施加投影移除操作（$P_{\perp} = I - U_k U_k^T$，採用 cloud centering）：

**觀察：**
- **主臂縮減效果（G1a/G1b pass）**：在全強度 $\alpha = 1.0$ 下，frozen group gap 由 1.2697 縮減至 **0.5863**（$\le 0.6348$，縮減 **53.8%**，G1a pass）；entity-contrast IQR 由 0.5640 縮減至 **0.2800**（$\le 0.2820$，縮減 **50.2%**，G1b pass）。
- **隨機子空間對照（G2 pass）**：隨機 8 維正交子空間對照僅縮減 group gap 0.49%（1.2697 $\to$ 1.2635，相對縮減比例 0.0049，mean shift +0.0016 nats），遠低於主臂縮減量的 25% 門檻（相對縮減 $\le 0.1346$，即主臂 53.82% 的四分之一，G2 pass），證實縮減效果高度特異於該實體差異子空間。

**解讀：** L15 的 8 維實體差異子空間確實承載了公司間決策落差的主要推力，移除該子空間能特異性地消減實體落差。

## 2. 全強度移除是否具備無副作用的選擇性？未通過，引發全局推注與匿名偏移

**觀察：**
- **全局推注（G3 fail）**：全體 64 題平均 margin 由 −1.7500 推至 −1.4201（正向偏移 **+0.3299 nats**，超過 $\le 0.15$ 門檻）。
- **匿名提示詞偏移（G4 fail）**：匿名無實體提示詞（identity-stripped prompt）的 margin 由 −3.2280 移至 −3.4968（偏移 **−0.2689 nats**，超過 $|\cdot| \le 0.10$ 門檻）。

**解讀：** 子空間移除在消除實體落差的同時，帶走了一個全局立場分量；依預先註冊決策表（「G3 或 G4 fail $\to$ full-strength 有全局副作用；V1 於 full strength 為負結果」），本線於全強度收口為負結果，不事後預註冊劑量救援。

## 3. 干預效果是否具備層級與上下文特異性？L15 最有效，cloud centering 最佳

**觀察：**
- **層級特異性**：在 canonical 16-prompt 測試下，L15 最有效（gap 0.7661），顯著優於周邊層（L13 1.0960、L14 0.8945、L16 1.0100、L17 1.1622），與 entity-to-dial 收線定位一致。
- **Centering 比較**：cloud centering（shift +0.3299）相比 zero centering (+1.0948) 與 anon centering (+1.0611) 減少了近 70% 的全局漂移。
- **Position Scope**：instruction span（gap 0.5863）足以承載大部分效果，full sequence (0.1236) 雖縮減更深但全局偏移略大。
- **劑量單調性**：$\alpha \in \{0.25, 0.50, 0.75, 1.00\}$ 呈現單調近線性（gap 依序為 1.0784、0.9120、0.7394、0.5863），全 population 64 題全程維持 sell 判定（零翻轉）。

## 4. 研究宣稱之邊界與未涵蓋事項

1. **樣本母體限制**：基底取自 16 家企業，group gap 定義於 4 家極端企業（TOP = {BLK, NSC}、BOTTOM = {BDX, IT}），未外推全域企業庫。
2. **全強度負結果**：本研究提供機制探索證據，不構成可直接部署之推論期去偏演算法。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| V1 協議與凍結決策表 | [V1 協議](details/proposal-v1.md)（Rev 1.5）。 |
| M6-V2 協議與結果 | [M6-V2 protocol](details/proposal-m6-v2.md)；run `m6-v2-gpu1-01`，位於 `artifacts/qwen3.5-4b/selective-intervention-m6-v2/runs/`。 |
| Formal run 執行記錄與產物 | V1 run `selective-intervention-v1-gpu-bf16-01`，位於 `artifacts/qwen3.5-4b/selective-intervention/runs/`；646 arm forwards + 17 calibration forwards。 |

**本次編輯說明：** 本報告依既有紀錄按三項核心問題改寫，移除純導覽的頂層 proposal，直接以本報告為閱讀入口；不變更任何原始數據、門檻或執行歷史。
