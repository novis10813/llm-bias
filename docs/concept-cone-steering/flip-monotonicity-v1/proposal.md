# flip-monotonicity-v1：以生成決策的 flip-rate 曲線重算 cone 的平滑度

**狀態：**事前協議，2026-09-30 於計算前寫定。**性質：**探索性、事後新增（margin 版 C8 的結果已知），純 CPU 重新分析 confirmation-v1 已有的逐劑量結果；不跑模型。**上層：**[operator-comparison-v2](../operator-comparison-v2/status.md)（C8 原判準）與 [claim-to-evidence.md](../claim-to-evidence.md)。

## 為什麼做

C8 的平滑度指標算的是「固定前綴 margin 的平均變化」在相鄰劑量間是否上升。這個 margin 在 Gemma、GPT-OSS 是 off-path，其符號與生成決策也會不一致（見 C4），所以以它判斷 cone 是否比 DIM 平滑並不可靠。本版改用**生成決策的 ITT on-target flip rate**當曲線，其餘結構沿用 C8。

本版**不取代、不改寫** C8 及其結果；C8 仍是 margin 版的「測試後不成立」。本版只新增一個以生成決策為準的對照。

## 資料

- `artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-20260925-full-01/dim/result.json` 的 `summary.dim.per_alpha`，以及 `ops/result.json` 的 `summary.<op>.per_alpha`（`cone2`、`cone4`、`cone4_projection`、`dim_orth_rand4`、`neuron`）。四個模型：Qwen3.5-4B、Gemma-4-12B、GLM-4-9B、GPT-OSS-20B。
- 每個點的 `on_flip_itt`：α 的目標方向上，α=0 決策屬於來源類別的公司中，生成決策翻成目標類別者的比例；未能解析的輸出計為未翻（含 collapse）。`on_class_n = 0` 的符號沒有分母，不評。

## 曲線與指標

對每個模型、operator、符號，取該符號上 `on_class_n > 0` 的網格點，依 |α| 由小到大排序，得到 flip-rate 序列 `y_1, …, y_m`。

- **monotone-step fraction：**相鄰步 `y_{k+1} ≥ y_k` 的比例。平台（相等）算單調；這與 C8 用的「嚴格上升」不同，因為 flip rate 會飽和在 1.0，嚴格上升會把飽和誤判為不平滑。
- **drops：**`y_{k+1} < y_k` 的步數。高劑量端的下降通常來自輸出崩壞（unparsed 計為未翻），這是刻意保留的，因為它反映抗飽和性。
- **max drop：**最大單步下降幅度（描述用）。

## 判準（沿用 C8 的結構）

比較 `cone4`、`dim_orth_rand4`、`dim`，只用三者共同擁有的網格點。某符號的判準成立，當且僅當：

1. `cone4` 的 monotone-step fraction **嚴格高於** `dim_orth_rand4`，且**不低於** `dim`；
2. `cone4` 的 drops **不多於** `dim_orth_rand4` 與 `dim`。

模型層級：兩個符號都有分母時，兩個符號都成立才算成立。只有一個符號有分母的模型（Qwen 只有 sell→buy，GLM 只有 buy→sell），只評該符號，並明確標為單符號；兩符號都沒有分母則不評。三者相等時「嚴格高於」不成立，並如實列出該平手。

`cone2`、`cone4_projection`、`neuron` 只報同樣的指標，不進判準。

## 解讀界線

- 探索性：判準與指標在計算前寫定，但 margin 版結果已知，不當成新的預先登記檢定。
- 只描述 flip-rate 曲線的形狀；不宣稱 cone 的軸有語意，也不延伸到未測的劑量。
- flip rate 的分母是各方向的來源類別公司數（Qwen 101、GLM 101、Gemma 43／58、GPT-OSS 57／44）；不用 bootstrap。

## 輸出

`artifacts/<slug>/concept-cone-steering/runs/flip-monotonicity-v1-20260930-01/result.json`（各輸入檔的 SHA-256、每個 operator 與符號的曲線與指標、判準結果）；腳本 `scripts/summarize_flip_monotonicity.py`。
