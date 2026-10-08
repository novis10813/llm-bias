# 論文與實驗的差異紀錄

**用途：**逐項記錄論文稿（2026-10-08 版，"The Invisible Hand"）與實際協議、程式、artifact 不一致之處。論文內容不在此修改；改稿時以本檔為清單。

**核對基礎：**`confirmation-v1-20260925-full-01`、`confirmation-v1-supp-20261006-full-01`、`phase2b-v2-427-01`、`audit-v1-20260925`、`confirmation-v1-freeze-20260925`（後三者 2026-10-08 自 idlab2 取回，sha256 與 idlab、idlab1 的副本一致）；summarizer 為 branch `research/confirmation-v1-supplement` 的 `scripts/summarize_confirmation_paper_tables.py`。路徑皆相對於 `artifacts/<slug>/`。

**分組：**P01–P05、P20 屬 T01（layer localization），P06–P12 屬 T02（steering operators），P13–P17 屬 T03（control limits），P18–P19 跨 task。

## T01 layer localization

### P01 Stage 1 的公司集合不是 402 家 construction

- **論文：**§4.1「We construct paired prompts using a fixed set of 402 construction companies.」
- **實驗：**使用 `data/baseline/investment-dial/exploratory-v1.json` 的 427 家，其中 89 家在 101 家 evaluation 集內。
- **影響：**A2 稽核（`concept-cone-steering/runs/audit-v1-20260925/c2_overlap_exclusion.json`）排除這 89 家後剩 338 家（676 directions），四模型 instruction peak 與 T 不變：Qwen L16 0.407、GLM L19 0.460、Gemma L27 0.190、GPT-OSS L14 0.532。Table 2 數字本身可重現（`balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json`）。

### P02 Stage 1 只用了 pos 與 neg 兩個條件

- **論文：**§3「For layer localization, we also vary the prompt using positive-only, negative-only, mixed, and zero-evidence combinations.」Appendix A 的 conditions 表把 positive、negative、mixed、zero 都標為「Layer localization」。
- **實驗：**Stage 1（[phase2-v2 協議](../balanced-evidence-gap/details/proposal-phase2-v2.md)）只用同公司的 `pos`（P1,P2）↔ `neg`（N1,N2）。`mixed2` 與 `zero` 只在 confirmation 的 `evidence`、`anon` arm 跑過，論文沒有報告這兩個條件的結果。
- **影響：**conditions 表的「Used for」欄應為：positive／negative 用於 localization 與 opposing evidence，mixed／zero 未報告。

### P03 GPT-OSS 的注入層在 medium reasoning 下選出

- **論文：**Appendix A「GPT-OSS uses low reasoning effort」，正文未區分 Stage 1 與 Stage 2。Appendix G 只寫「used a different reasoning setting and readout」。
- **實驗：**[confirmation-v1 協議](confirmation-v1/proposal.md)的模型登錄表註明 L14 是在 `medium` reasoning、固定前綴 readout 下選出；所有 steering 用 `low`。

### P04 Stage 1 不是預先登記的協議

- **論文：**Appendix H「Pre-commitment: the protocols were committed before the corresponding runs.」
- **實驗：**phase2-v2 協議標為 development（非 protocol-final），選層規則沒有事前固定；Gemma 與 GLM 的 Phase 2A gate 未通過，以 override 續跑（[C2 v2-427 狀態](c2-v2-427/status.md)）。

### P05 Appendix G 的 readout 不是 realized-path margin

- **論文：**Appendix G「The readout is the realized-path margin (Appendix C)」，Appendix C 定義為「read at the step of the model's own greedy generation」。
- **實驗：**[C2 v3 協議](c2-v3-steering-prompt/proposal.md)的 primary readout 是 teacher-forced clean-path margin：把 target 公司自己的 α0 生成 token 接在 prompt 後，在決策值 token 前讀 margin。patched forward 讀的是 target 的 α0 路徑，不是 patched 模型自己生成的路徑。

### P20 GPT-OSS 的 70% band 不是連續的 1–8 層

- **論文：**Appendix G 表格「70% band (layers)」GPT-OSS 為「1–8」。
- **實驗：**`confirmation-v1-20260925-full-01/c2v3/result.json` 的 `summary.band` 為 `[1, 8]`：teacher-forced steer-suffix 曲線上只有 L1 與 L8 達到 0.7 × peak（peak L8，T=0.103），L2–L7 都低於門檻。

## T02 steering operators

### P06 Dose grid 不是單一預先登記網格，±0.25 是事後補點

- **論文：**Table 3 對四模型用同一組 {±0.25, ±2, ±4, ±8, ±16}。Appendix H「each model's grid comes from a pre-specified rule applied to construction-set companies only」及「the protocols were committed before the corresponding runs」。
- **實驗：**CAL 產生的 full-01 網格：Qwen、GLM 為 ±{1,2,4,8,16,32,64}，Gemma 為 ±{0.25,2,4,8,16,32,64}，GPT-OSS 為 ±{0.125,0.5,2,4,8,16,32}。Qwen、GLM、GPT-OSS 的 ±0.25（`dim`、`ops`、`evidence`、`anon`）與四模型 Random 欄的多數 dose 來自 2026-10-06 的 supplement run，在 full-01 結果揭露後才加，沒有協議文件。
- **影響：**Table 3、Table 4 中 Qwen／GLM／GPT-OSS 的 ±0.25 欄，以及 Random 欄（Qwen +0.25/+2/+8/+16、GLM −0.25/−2/−4/−16、Gemma ±2～±16、GPT-OSS ±0.25/±2/±8）不在預先登記範圍。

### P07 ±16 以外 DIM 不單調、也不一定可解析

- **論文：**§4.2「DIM flip rates are monotone in dose, and no DIM cell falls below the 90% parse threshold」；Appendix E 段落「DIM stays parsable across the whole grid」。
- **實驗：**`dim/result.json` 的 full-01 網格超出表格範圍時：Gemma α=−64 flip 0.00、parse 1.00（steering 失效，不是格式崩壞）；Qwen α=+32 parse 0.98、+64 parse 0.00；GPT-OSS α=−32 flip 0.02、parse 0.02，α=+32 flip 0.60、parse 0.66。
- **影響：**這兩句只在 |α|≤16 成立。

### P08 Cone 軸跨 token 共用，不是逐位置 PCA

- **論文：**§4 Concept cone「We stack the 100 residuals into R[p], take its leading k−1 principal directions」，即每個位置 p 各自做 PCA。
- **實驗：**`llm_bias/core/steering/directions.py:cone_axes`：100 個殘差在所有 K 個位置逐列單位化後合併（100K 列），取未中心化 Gram 矩陣的前 k−1 個特徵向量作為跨 token 共用軸，再逐 token 對 d̂[p] 做 Gram–Schmidt。正負號由 Top/Bottom 20 家的 token 平均狀態投影與固定前綴 margin 的 Pearson 相關決定。

### P09 Single neuron 的實作細節未寫出

- **論文：**「a single MLP neuron whose write vector w_n best aligns with the DIM direction … steers entirely along w_{n*}」。
- **實驗：**選擇規則與論文等價（`argmax cos(w_n, mean_p d̂[p])`）。未寫出的是：注入時逐 token 縮放為 ‖d[p]‖（等範數）；Gemma、GLM 的 w_n 乘上 post-FFN／post-MLP norm 的 gain；GPT-OSS 是 MoE，候選為 32 experts × 2880 共 92160 個，選中 expert 22 的 neuron 1617（cos 0.144）。這是在 residual 加上 write 方向，不是改寫 neuron activation。

### P10 C8（cone 較平滑）有預先登記的正式檢定，結果不支持

- **論文：**§4.2「our dose grid is too coarse to compare transition widths formally」。
- **實驗：**[operator-comparison-v2](operator-comparison-v2/proposal.md) 預先登記了 C8 判準（monotone-step fraction、反轉次數等），已在四模型執行，四模型都未通過（[claim-to-evidence](claim-to-evidence.md) C8）。論文的結論方向一致，但寫成「無法正式比較」與事實不符。

### P11 Steer suffix 的 token ids 跨 tokenizer 不相同

- **論文：**Appendix B「These trailing token ids are identical across the four tokenizers.」
- **實驗：**`concept-cone-steering/runs/confirmation-v1-freeze-20260925/manifest.json`：K 為 Qwen 100、GLM 98、Gemma 100、GPT-OSS 99，`steer_suffix_ids_sha256` 四模型各不相同。相同的是同一模型內五個 evidence condition 的尾段 ids。

### P12 預先登記的 C5 判定未報告

- **論文：**§5.2 以描述方式比較 DIM 與 random。
- **實驗：**C5 判準要求 ±α_50 與 ±α_hi 各點 Gain 的 bootstrap 下界 > 0，且 DIM 高於 jitter 與自身 off-target。Gemma −0.25 下界 −0.0258、−64 下界 0，未通過（[claim-to-evidence](claim-to-evidence.md) C5）。jitter、shuffle 兩個對照有跑但論文未提。

## T03 control limits

### P13 Opposing evidence 只報到 |α|≤16，GLM 在更高 dose 會翻

- **論文：**摘要「against opposing evidence, three of four models never flip」；§5.3「In GLM, Gemma, and GPT-OSS, no dose up to |α|=16 flips any company」；結論「three of four models never flipped」。
- **實驗：**`evidence/result.json`（full-01）：
  - GLM（pos，buy→sell）在 −32 flip 0.74、−64 flip 1.00，parse 皆 1.00。所需 dose 是 balanced（−4）的 8 到 16 倍。
  - Gemma 到 ±64 都不翻，parse 1.00（blocked）。
  - GPT-OSS 到 ±16 不翻。±32 時輸出崩壞（neg +32 parse 0.72，pos −32 parse 0.02），無法判斷是否翻轉。
  - Qwen 和論文一致：neg +8 為 0.88、+16 為 1.00。
- **影響：**「三個模型從不翻」只在 |α|≤16 成立。在完整網格內，只有 Gemma 確定在可解析範圍內不翻。

### P14 Random 欄的定義和 DIM 欄的分母不同

- **論文：**Table 4 caption「Random: flip rate of matched-norm random directions (maximum over five seeds), and Gain = Steering − Random」。
- **實驗：**summarizer 的 `rand_stat`：每個 seed 計算「全部 source 或 target 類公司中，決策往任一方向改變的比例」，取五個 seed 的最大值。DIM 欄的分母則只有 source-class 公司。Gain 是兩個不同分母的比例相減（paper-tables README 已註明）。

### P15 Gemma 的 Random 不是 matched-norm，† 標記不完整

- **論文：**Limitations「Part of the Gemma random baseline was run on different GPU hardware (†)」；Table 4 只在 Gemma sell→buy 的 +2～+16 標 †。
- **實驗：**Gemma supplement 的 random 在不同 GPU 上無法重現原 DIM，random 方向對齊的是重新估出的 DIM，其 norm 中位數比原 DIM 低約 14%（`artifacts/paper-tables/confirmation-v1-supp-20261006/README.md`）。`cells.json` 中 buy→sell 的 −2、−4、−8、−16 也帶 operator drift，論文沒有標 †。
- **影響：**這 8 格的 random 比 DIM 小約 14%，Gain 可能偏向 DIM。

### P16 「24% 的輸出 margin 與決策不一致」只是 GPT-OSS 的數字

- **論文：**Limitations「Fixed-prefix margins disagree with generated decisions for 24% of outputs overall」。
- **實驗：**以 full-01 的 `alpha0`、`dim`、`ops`、`random`、`evidence`、`anon` 中所有可解析列重算（margin 正負與決策不一致）：GPT-OSS 24.1%（3578/14845）、Qwen 1.2%、GLM 2.0%、Gemma 2.2%，四模型合計 7.5%（4359/57959）。

### P17 GPT-OSS evidence arm 重跑結果不固定

- **論文：**未提。
- **實驗：**重跑原 run 中 α=−32 的一列時決策會改變（paper-tables README 第 4 點）。受影響的是 supplement 補的 GPT-OSS ±0.25 Opposing 兩格（皆為 0.00）。

## 跨 task

### P18 Figure 1 找不到來源

- **論文：**Figure 1 為 Gemma-4-12B、layer 27、一正一負兩個 fact 的範例。
- **實驗：**repo 與 artifacts 中沒有產生 `Example_Plot.png` 的 script 或對應的列。一正一負是 `mixed2` 條件，它只跑了縮減網格 `G_red`。範例的公司與 dose 無法核對。

### P19 Gemma 的 double BOS 未說明

- **論文：**Appendix A「We render the prompt with each model's pinned chat template」。
- **實驗：**Gemma 的 chat template 自帶 `<bos>`，loader 又加一次，所有 Gemma run 都是 double BOS（confirmation-v1 協議已記錄，未修改）。
