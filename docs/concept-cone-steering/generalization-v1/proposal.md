# Generalization v1：leave-one-sector-out（R9／R10）與 split-seed 複製（R11）

**狀態：**事前協議，2026-09-25，smoke 前凍結。**上層協議：**[confirmation-v1](../confirmation-v1/proposal.md)。**Claims：**C10（跨公司與跨產業泛化）、C12（跨模型）。

## 已成立的部分

402/101 split 已是 company-disjoint：受測 101 家從不參與排序、方向擬合或 CAL。論文可寫「company-disjoint held-out」。「sector-disjoint」必須滿足下方規則才能寫。

## LOSO（`arm_loso`）

- 對受測公司所在的每個非 Unspecified sector S：
  - **loso fold：**construction 排序中排除 S 與所有 Unspecified 公司，重選 Top/Bottom10 並擬合 `d_fold`。
  - **comparator：**只排除 Unspecified（所有 sector 共用一個）。把 comparator 與 loso 分開，避免「排除 Unspecified」與「排除 S」混在一起。
  - **placebo fold（R9）：**排除 Unspecified，以及與 S 同數量、隨機抽出的非 S 非 Unspecified 公司（seed `20260927 +` sector 依字母序的索引）。
- Unspecified 公司（construction 22、eval 7）不當 target。
- R9：101 家中的 94 家非 Unspecified 受測公司，各自在自己 sector 的 loso 與 placebo fold 下、以及 comparator 下，跑 `G_red` 非零點；α0 與方向無關，沿用共用 α0。
- R10（`loso_construction`）：380 家非 Unspecified construction 公司，各自在自己 sector 的 loso fold 與 comparator 下跑 `G_red`；落在 comparator Top/Bottom10 內的公司，不列入對比（標記在 `targets_in_comparator_construction`）。
- 記錄每個 fold 的 Top/Bottom10、`cos(d_fold, d_main)` 的 median（特別注意 Gemma 的 Energy fold）。

## C10 判準（預先登記）

- 在 `±α_50` 各符號：comparator 減 loso 的 ITT on-target flip 率差，company bootstrap 95% CI；另報 discordance（兩個方向下生成決策不同的公司數）。
- 「sector-disjoint」措辭必須同時滿足：
  1. R9 在 `±α_50`（有分母的符號）上，comparator − loso 的 CI 上界 < 10 個百分點；
  2. R10 中每個「construction n ≥ 20，或在任一模型的 Top/Bottom10 中佔 ≥ 3 家」的 sector，loso fold 下至少有一個 on-target flip。
- 未滿足時只寫 company-disjoint。每個 sector 的 eval n 很小（約 5–20），per-sector 結果只作描述。另報 89／12 分層（是否在 C2-427 公司集合中）。

## Split-seed 複製（R11，`arm_split_seed`）

- seed `20260924`、`20260925` 各產生新的 402/101 split；新 construction 的排序直接取自 503 家乾淨 margin（ranking arm），重選 Top/Bottom10、擬合方向，在新的 101 家上跑 `G_red`，新 eval 公司缺 α0 者由 job 補生成。
- CAL 公司取自三個 seed 的 construction 交集，所以不會出現在任何新 eval 中（runner 會檢查）。報告新舊 eval 的重疊數。

## 成本（full）

R9 約 94 × 3 × 4 列；R10 約 380 × (1 + 2 × 4) 列；R11 約 2 × 101 × 4 列，加新 eval 的 α0。
