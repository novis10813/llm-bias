# operator-comparison-v2：四模型比較已執行，cone 平滑優勢未獲支持

**狀態：**依 [事前協議](proposal.md)於 `confirmation-v1-20260925-full-01` 完成；整體執行／同步紀錄見 [confirmation-v1/status.md](../confirmation-v1/status.md)。本頁記錄已同步 artifact 的核對結果，不改寫凍結協議或舊 pilot。Gemma／GLM／GPT-OSS 的 `job_summary.json` 只含最後一次 invocation 的 arm；每個 arm 應以自己的 `result.json` 為準。

## 執行入口與範圍

四模型各自的 `artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-20260925-full-01/` 下：`dim/result.json`、`ops/result.json`、`random/result.json`、`jitter/result.json` 均為 `complete=true`；Qwen、Gemma 的 `shuffle/result.json` 亦完成。GLM／GPT-OSS 沒有 shuffle 是[上層協議](../confirmation-v1/proposal.md)的 tier 定義，不是漏跑。各 run 均有 `cal/calibration.json`、`invocations.jsonl` 及具 checkpoint／template／split／C2 來源 SHA 的 arm metadata。實際執行 host、commit 與續跑紀錄以 invocations 和 [上層狀態](../confirmation-v1/status.md) 為準，不從本頁推測命令。

`ops` 包含 construction-only 選出的 single-neuron 寫入方向、cone2、cone4、等投影 cone4、DIM⊥random4；各 operator 保存每-token 劑量統計、逐公司 margin／生成／parse、ITT flip 與 collapse。此 cone **不等於**舊 `smoke-cone4d` 的 sector-demeaned SVD basis。比較以同一模型內的 matched-norm／有效投影為準，不以不同模型的 raw α 比較。

## 預定判準的實際結果

| 項目 | 核對結果 | 可寫界線 |
|---|---|---|
| C3 operator comparison | neuron、DIM、cone2／cone4、等投影 cone 和 controls 的劑量與逐公司行為皆已產出 | 可以描述相同目標／prompt／層下的 dose-response；不能稱各 operator 的同一 raw α 等效，也不能把「比較完成」寫成 cone 勝出 |
| C5 DIM vs random | `random/result.json → summary.c5` 已做五個 matched-norm random seed 與 company bootstrap。Qwen 有分母的 +4／+32 下界 1.0／0.9505、GLM 負向點與 GPT-OSS 四點下界均 >0；Gemma −0.25 下界 −0.0258、−64 下界 0，未達 `>0` | 此 summary 只判 random 這一關；C5 **完整**判準還要求 DIM on-target 同時勝三個 jitter seed 的任一方向 flip 率及自身 off-target flip 率，報告時須逐點列出。零分母不是陰性結果；不可宣稱四模型全面支持 C5 |
| C8 cone smoother | `ops/result.json → summary.c8` 有兩符號的 monotone-step fraction、Spearman、reversals 與斜率比；依 proposal 的「兩符號均嚴格勝 DIM⊥random4、不低於 DIM，且反轉不較多」，**四模型均未通過**。Qwen cone4 正向 0.667 vs DIM⊥random4 0.833、負向 0.333 vs DIM 0.667；Gemma、GLM、GPT-OSS 亦至少一符號不符合 | 已測而不支持，不是尚未執行；不得引用舊筆記宣稱 4D cone 優於 1D |

翻轉率使用上層協議的 **complete-object** primary parse 與 α0 條件 ITT 分母（steered unparsed 記未翻）；strict JSON 是另一個指標。C5 的嚴格跨模型判定、C3 的模型別比較表尚須整理成論文表格，但**不是未跑的 GPU arm**。
