# evidence-sensitivity-v1：配對證據與匿名實驗已完成，blocked 不等於 collapsed

**狀態：**[事前協議](proposal.md)所列 `evidence`、`anon` 已於四模型的 `confirmation-v1-20260925-full-01` 執行完成；host、commit、同步等整體 provenance 見 [confirmation-v1/status.md](../confirmation-v1/status.md)。本頁只審查已同步的 artifact，不修改舊 exploratory renderer 的結果。

各模型在 `artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-20260925-full-01/` 的 `dim/result.json`、`evidence/result.json`、`anon/result.json` 均為 `complete=true`。五個 condition（balanced 取自 `dim`，另外 pos／neg／zero／mixed2 取自 `evidence`）共用凍結 skeleton、各 condition 有自己的 α0；`anon` 則含 10 個預先登記身分 × {balanced, pos, neg, zero}。各 arm metadata 保留 prompt family／模型／split SHA；runner 結果保留逐列生成、parse、固定前綴與 realized-path margin。

## C7：已計算 flip dose，須把輸出崩壞分開

`evidence/result.json → summary.c7.contrasts` 已有同公司、同起始決策類別的可比數、higher／equal／lower flip dose 與 blocked／collapsed：

| 模型與預登記對比 | 可比公司與觀察 | 判讀 |
|---|---|---|
| Qwen3.5-4B：neg vs balanced，sell→buy | n=101，101 higher dose；neg vs mixed2，n=98，98 higher | 支持這兩個可比分母下「負向證據延後翻轉」；pos vs balanced 無 buy→sell 分母 |
| GLM-4-9B：pos vs balanced，buy→sell | n=101，101 higher；pos vs mixed2，n=101，33 higher／20 equal／48 lower | 正向證據相對 balanced 延後翻轉，但相對 mixed2 不能一概如此；neg vs balanced 無 sell→buy 分母 |
| Gemma-4-12B：neg vs balanced、pos vs balanced | 可比 n=43／58，分別 43／58 **blocked**，未記為 collapsed | 在掃描可解析範圍內未翻；不等於證明無限劑量都不會翻 |
| GPT-OSS-20B：neg vs balanced | 可比 n=57，45 blocked／12 collapsed；pos vs balanced n=44，1 blocked／43 collapsed | collapsed 是無法解析，**不能**算「證據抵抗 steering」；可寫範圍比其他模型窄 |

`blocked` 指各相關 steered 列仍可解析而未翻；`collapsed` 指翻轉前已出現 unparsed。報告同時列出 parse rate 與起始類別分母，不把固定前綴 margin 或生成 `reason` 當作決策或 rationale faithfulness。

## C6：匿名身分的效果已觀測，不是 entity-free 證明

`anon/result.json` 含 10 個匿名身分 × 四條件的生成與 ITT 摘要；例如 Qwen balanced α0 為 10/10 sell，α=+4 翻轉 10/10。這與超出單一實體身分的 stance control **相容**，但不證明 direction 完全不包含公司資訊。若比較匿名與 named 的誤差範圍，應依[協議](proposal.md)以 identity 為單位呈現 bootstrap（n=10），不可當成 101 家公司數。
