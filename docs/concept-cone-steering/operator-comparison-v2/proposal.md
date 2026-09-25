# Operator comparison v2：single neuron → DIM → cone，與 random／jitter／shuffle controls

**狀態：**事前協議，2026-09-25，smoke 前凍結。**上層協議：**[confirmation-v1](../confirmation-v1/proposal.md)（劑量、CAL、readout、flip 定義皆沿用）。**Claims：**C3（operator 可比）、C5（learned direction 勝過 matched random）、C8（cone 較平滑）。原 v1 協議檔 `proposal-p2-operator-comparison-v1.md` 在 repo 與 git 歷史中都不存在，不重建；本檔是第一份可執行的 operator 比較協議。

## 共同設定

- 同一模型、同一主要層、同一 101 家受測公司、同一 balanced prompt、同一 steer suffix（prefill only）。
- 所有 operator 以 DIM raw 差 `d[p]` 為錨；dose 表記錄每個 operator 在 α=1 的注入 norm、`cos(B, d̂)`、投影與 `‖B‖/median‖h‖`。
- 投影劑量 `α_eff = α × median(proj_d̂ B) / median‖d‖`；DIM、cone4 等投影為 1，cone4 等範數為 0.5，neuron 為其 `cos(w, d̂)`。

## Operators（`scripts/probe_steering_confirmation.py:arm_ops/arm_random/arm_jitter/arm_shuffle`）

| 名稱 | 構造 | 劑量 | 網格 |
|---|---|---|---|
| `dim` | `d[p]`（`dim` arm） | raw | full |
| `neuron` | 主要層 MLP 的單一 neuron 寫入向量（見下），所有 token 共用 | 等範數 | full |
| `cone2`、`cone4` | `(d̂[p] + Σ_{j=2..k} v_j[p]) / √k` | 等範數 | full |
| `cone4_projection` | 同 cone4 | 等投影（總 norm 為 2‖d‖） | full |
| `dim_orth_rand4` | 同 cone4，但 v2–v4 換成共用 random 向量（seed 20–22），逐 token 對 `d̂` 與彼此正交化 | 等範數 | full |
| `random_s0..4` | CPU fp32 generator 以 seed 0–4 抽一個單位向量，所有 token 共用 | 等範數 | random |
| `jitter_s10..12` | 同 random（seed 10–12） | 等範數 | `±0.05·α_50` |
| `shuffle_s100..102` | construction 扣除真 Top/Bottom10 後，以 seed 抽兩組各 10 家算差向量，取單位方向 | 逐 token 等範數於 `d` | `G_red` |

### Cone 軸（不依賴配對）

對每個 token，取 10×10 共 100 個 Top−Bottom 差向量，扣掉在 `d̂[p]` 上的分量並逐列單位化；把所有 token 的列合併，取 Gram 矩陣前 3 個特徵向量為共用軸 v2–v4，再逐 token 對 `d̂[p]` 與前面的軸做 Gram–Schmidt。軸 1 固定為 `d̂[p]`，所以 cone1 ≡ DIM，不另跑。

v2–v4 與 `d` 正交，Top 與 Bottom 的均值差在其上的投影為 0，因此正負號由預先登記的規則決定：20 家 construction 公司的 token 平均狀態在該軸上的投影，與其乾淨固定前綴 margin 的 Pearson 相關取非負（為 0 時取絕對值最大分量為正）。另記解釋變異比例、|相關|，以及逐 token 正交化前後的平均 |cos|（跨 token 一致性）。

### Single neuron（使用者決定：四個模型都做）

- 規則：主要層上 `argmax_n cos(write_n, mean_p d̂[p])`（逐 token 單位化後的平均，等於最大化共用寫入向量與各 token `d̂[p]` 的平均 cos），只用 construction 資料。候選集合與寫入向量：
  - Qwen3.5-4B：`mlp.down_proj.weight` 的第 n 欄。
  - Gemma-4-12B：第 n 欄乘上 `post_feedforward_layernorm.weight`（逐元素 gain）。
  - GLM-4-9B：第 n 欄乘上 `post_mlp_layernorm.weight`。
  - GPT-OSS-20B：32 個 expert × 2880 個 neuron，從 checkpoint 的 MXFP4 `down_proj_blocks/_scales` 反量化為 `[expert, intermediate, hidden]` 後取 `[e, n, :]`；標記為 `expert:neuron`。
- 權重直接從 safetensors 在 CPU 讀取，不經過載入後的模組（避免 MXFP4 kernel 格式差異）。
- 寫入向量以等範數注入於 steer suffix（與其他 operator 同一劑量慣例），不是改寫 neuron 活化值。Sandwich norm 對 MLP 輸出整體做 RMS 正規化，gain 加權欄向量是該 neuron 貢獻在 norm 後的方向（差一個逐 token 純量），本協議把它當作「該 neuron 的寫入方向」。
- 記錄：選中的 neuron、其 cos、第二名 cos、候選數。Qwen 另報同規則在 L15 會選哪個 neuron，以及舊 dial N8490 在 L15 的 cos 與名次（只作歷史對照，不生成）。

## C5 判準（預先登記）

- 在 `±α_50` 與 `±α_hi` 四個點（各符號分開）：DIM 的 ITT on-target flip 率，減去 5 個 random seed 中「任一方向 flip 率」（分母為 α0 可解析公司、steered unparsed 記為未翻）的最大值；company bootstrap（B=2000）95% CI 下界 > 0 記為該點支持。
- C5 成立：四個點中每個「DIM on-target 分母非 0」的點都支持，且 DIM on-target flip 率同時高於三個 jitter seed 的任一方向 flip 率與 DIM 自身的 off-target flip 率。
- 其餘 random 網格點（`±α_90`）與 margin 差距只作描述；GPT-OSS 與 Gemma 的 margin 標 off-path。
- shuffle（R3d，tier2）為補充 null：同樣報 ITT on-target flip 率，不改變上述判定。

## C8 判準（預先登記）

- 對每個 operator、每個符號，以 `α_eff` 為橫軸、平均 ΔM（固定前綴）與 ITT on-target flip 率為縱軸，計算：相鄰點 ΔM 朝目標方向上升的比例（primary）、Spearman、反轉次數、高劑量段與低劑量段的斜率比、collapse dose 與 flip dose。
- C8 成立（「cone 較平滑」）：兩個符號上，`cone4` 的 primary 指標都**嚴格高於** `dim_orth_rand4` 且**不低於** `dim`，並且反轉次數不多於兩者。否則 C8 不成立；因為 V1 已顯示 DIM 的 flip 遠多於 cone，本判準預期可能否證 C8，結果照實報告。
- Gemma 有 `final_logit_softcapping=30`，跨模型比較 margin 時只用 |ΔM| < 20 的點。

## C3 報告

同層、同一批公司、同一劑量慣例下的描述性比較表：neuron／DIM／cone2／cone4（等範數與等投影）的 ITT flip 率、parse 率、collapse dose、flip dose、ΔM 與 realized-path margin，並附 dose 表。不做 operator 間的顯著性檢定。

## Tier

`ops`：Qwen、Gemma 在 tier1；GLM、GPT-OSS 在 tier2（原 R3c）。`random`、`jitter`：四模型 tier1。`shuffle`：Qwen、Gemma tier2。

## 修訂紀錄

- 2026-09-25（Qwen smoke 後、任何 full 結果之前）：single-neuron 目標由 `mean_p d[p]` 改為 `mean_p d̂[p]`。Colab L4 smoke 顯示 `‖d[p]‖` 在 steer suffix 內相差約 20 倍（0.09–2.04），原目標被少數高範數 token 主導，選中的 neuron 與各 token `d̂[p]` 的 cos 中位數約 −0.001（等同 random）。新目標與逐 token 等範數劑量慣例一致。

