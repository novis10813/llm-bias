# Qwen3.5-4B Jacobian-lens 校準與候選選擇：簡體中文獲選為 Operational Winner

**狀態：已完成（儀器選擇與部署完成）。** 模型為 Qwen3.5-4B；2026-07-30 完成。設計、fitting 與 selection rule 見 [提案與操作契約](proposal.md)。

**一句話發現：** 評估 English-only、Simplified-Chinese-only 與 bilingual mixed 三組校準條件，簡體中文在預設規則下得分最高獲選為 operational winner（mean log10 rank 3.826620），但 32 組語意對 bootstrap 95% CI 均跨過 0（vs English 為 [-0.1779, 0.0478]、$p=0.148$；vs mixed 為 [-0.1209, 0.0110]、$p=0.057$），統計上尚未證實其顯著優於另外兩組。

## 1. 三組校準條件中哪一組在預設評選規則下勝出？簡體中文獲選為 Winner

我們在 Qwen3.5-4B 的 32 個雙語語意對（bilingual semantic pairs）上評估三組校準條件的 balanced native mean log10 rank（越低越好）：

**觀察：**
- **`chinese_simplified`**：balanced native mean log10 rank 為 **3.826620**（選定分數 −3.826620）；
- **`mixed`**：mean log10 rank 為 **3.881337**；
- **`english`**：mean log10 rank 為 **3.889848**。

**解讀：** 依據預先註冊的評選規則，`chinese_simplified` 取得最佳數值，被選為 canonical promotion winner。

## 2. 該勝出者在統計上是否顯著優於其他語言校準組？CI 跨 0，統計優勢未確立

我們以 32 個語意對為配對單位，執行 10,000 次確定性 bootstrap 與單尾 sign-flip 置換檢定：

**觀察：**
- **vs English-only**：平均差值 −0.06323，95% CI **[−0.17790, +0.04780]**，單尾 $p = 0.1483$；
- **vs Mixed**：平均差值 −0.05472，95% CI **[−0.12093, +0.01096]**，單尾 $p = 0.0573$。

**解讀：** 兩組比較的 95% 信賴區間均跨過 0。因此，簡體中文僅為預定規則下的**操作性獲勝者（operational winner）**，現有 holdout 尚未提供其在統計上顯著優於另外兩組的證據。

## 3. 透視鏡產物在後續研究中的角色與邊界為何？

**部署與限制：**
1. **Canonical Promotion**：`chinese_simplified` 候選透視鏡已透過原子化腳本晉升為 active canonical lens，存放於 `artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt`，完整覆蓋所有中介層。
2. **描述性投影非思維鏈**：透視鏡讀出僅代表殘差向量經線性映射回詞表的投影分數，不是模型的離散推理鏈或 Chain-of-Thought，亦不能單獨作為因果證據。
3. **未證明語言一般性**：本評估旨在選擇工具，未證明簡體中文校準在一般多語言任務上普遍具備優越性。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| 校準設計與評選契約 | [提案與操作契約](proposal.md)。 |
| 評選結果產物 | `artifacts/qwen3.5-4b/jacobian-lens/candidates/evaluation.json`。 |
| Active Canonical Lens | `artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt`（SHA-256：`3691d7b2...`）。 |
| 晉升執行腳本 | `scripts/promote_qwen_lens_candidate.py`。 |

**本次編輯說明：** 本報告按三項核心問題改寫，明確區分操作性獲勝與統計顯著性邊界；原始協議 `proposal.md` 完整保留。
