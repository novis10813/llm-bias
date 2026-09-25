# Evidence sensitivity v1：逆向證據、零證據與匿名身分下的 DIM steering

**狀態：**事前協議，2026-09-25，smoke 前凍結。**上層協議：**[confirmation-v1](../confirmation-v1/proposal.md)。**Claims：**C7（steering 仍對金融證據敏感）、C6（operator 是 global stance control）；另補 C1 的另一方向分母。

## Conditions

| condition | evidence body | 網格 | 用途 |
|---|---|---|---|
| balanced | P1、P2、N1、N2 | full（取自 `dim` arm） | 參照 |
| pos | P1、P2 | full | buy→sell 的逆向證據 |
| neg | N1、N2 | full | sell→buy 的逆向證據 |
| zero | 空（保留兩個標記，使用者決定） | full | 無證據 |
| mixed2 | P1、N1 | `G_red` | 控制證據量（兩句）的混淆 |

- 所有 condition 共用同一 skeleton；五個 family 的 K 與 steer-suffix token ids 完全相同（freeze 已驗證），所以同一個 `d[p]` 可以不經重對齊直接注入。方向一律來自 balanced construction（不為各 condition 重擬合）。
- 每個 condition 的 α0 各自生成一次；flip 以同一公司、同一 condition 的 α0 決策為條件。

## 估計量（`summary.py:flip_dose`、`scripts/probe_steering_confirmation.py:c7_summary`）

- **Flip dose：**該公司在該 condition 下，某一符號網格上第一個 on-target flip 的 |α|。
- **Censor：**`blocked` = 該符號所有 steered 列都可解析但沒有翻；`collapsed` = 在任何翻轉之前先出現 unparsed。只有 blocked 能支持 C7；collapsed 表示輸出崩壞，不算「證據阻擋」。
- **Contrast 規則（不是清單）：**只在同一公司、兩個 condition 的 α0 類別都屬於該方向起始類別時比較。預先登記的對照為 `neg` vs `balanced`／`mixed2`（sell→buy，α>0）與 `pos` vs `balanced`／`mixed2`（buy→sell，α<0）。報告可比公司數、逆向條件下較高／相同／較低的 flip dose 數，以及兩邊的 blocked／collapsed 數；可比公司 n < 10 時只作描述。
- C7 的可寫範圍：「逆向證據提高翻轉所需劑量，或在可解析範圍內阻擋翻轉（blocked，不含 collapsed）」。`reason` 文字只作質性材料，不作 faithfulness 量測。

## Anonymous（R5，C6）

- 10 個預先登記的匿名身分 × {balanced, pos, neg, zero} × full 網格；方向與 K 與 named arm 相同。runner 在生成前檢查格式化文字中沒有任何真實 ticker（`[TICKER]` 形式）或公司全名。
- 估計量在 identity 層級（n = 10）：各 α 的 ITT on-target flip 率，對照 101 家 named 公司在同一 α、同一 condition 的分布，用 identity bootstrap，明寫 n = 10。α0 全屬同一類時，只報有分母的方向。
- 可寫範圍：「在 identity 層級（n = 10）與 global stance control 相容」；不宣稱方向不含 entity 資訊。

## 成本（full）

每模型約 101 × (3 × full + 1 × `G_red`) ＋ 10 × 4 × full 列，另加 α0（101 × 4 ＋ 40）。
