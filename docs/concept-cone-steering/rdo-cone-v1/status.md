# rdo-cone-v1：狀態

**狀態：**已完成（2026-09-29，run `rdo-cone-v1-20260929-full-01`，Qwen3.5-4B，`idlab2` 的 lab job `rdo-6`，約 1 小時 24 分）。協議見 [proposal.md](proposal.md)。**性質：**探索性，單一訓練 seed，只有 sell→buy 一個方向；不改寫 operator-comparison-v2 的 C8，也不併入 confirmation-v1 的數字。

產物：`artifacts/qwen3.5-4b/concept-cone-steering/runs/rdo-cone-v1-20260929-full-01/{training,directions,result}.json`（`result.json` 的 `complete=true`）。smoke 為 `rdo-cone-v1-20260929-smoke-01`（在 `idlab2` 的 job 工作目錄，沒有拉回本機）。程式 commit `703b172`（分支 `rdo-cone-v1`）；超參數為協議預設值：150 步、batch 4、lr 1e-3、`λ_ret`=1、訓練劑量 1（DIM 範數的倍數）、seed 20260930、cone 維度 4、8 個抽樣方向。

## 結果（101 家評估公司，balanced，α0 為 101 sell／0 buy）

sell→buy 的 ITT flip 率／parse 率（單位為 DIM 範數的倍數）：

| 方向 | α=1 | α=2 | α=4 | α=8 | 翻一半的劑量 |
|---|---|---|---|---|---|
| `dim` | 0.00 | 0.10 | 1.00 | 1.00 | 4 |
| `rand1` | 0.00 | 0.00 | 0.00 | 0.00 | 無 |
| `rdo1` | 0.84 | 1.00 | 1.00 | 1.00 | 1 |
| `rco_b1` | 0.91 | 1.00 | 1.00 | — | 1 |
| `rco_b2` | 0.10 | 1.00 | 1.00 | — | 2 |
| `rco_b3` | 0.00 | 0.65 | 1.00 | — | 2 |
| `rco_b4` | 0.00 | 0.29 | 1.00 | — | 4 |
| `rco_centroid` | 0.91 | 1.00 | 1.00 | — | 1 |

所有格子的 parse 率都是 1.00，沒有出現崩壞（測到的最高劑量：`rdo1`／`dim`／`rand1` 為 8，cone 各方向為 4）。

## 預先登記問題的判讀

- **Q1（梯度方向比 DIM 有效率）：是。**`rdo1` 翻一半的劑量為 1，門檻為 ≤2。
- **Q2（基底各自有效）：是。**四條基底在 α≤4 都達到 100%；效率隨基底順序遞減（1、2、2、4）。
- **Q3（cone 有沒有用）：**8 個抽樣方向在 α=1 的 flip 率介於 0.09 與 1.00（平均 0.60、中位數 0.72），α≥2 全部 1.00。best-of-8 在 α=1 為 101/101，單一 `rdo1` 為 0.84。沒有比較 best-of-N 與「同一方向多次抽樣」的對照，所以不能把差距解讀為方向互補。崩壞劑量：測到的最高劑量內都沒有崩壞，因此不能比較抗崩壞。
- **Q4（reasoning 是否不同）：**只讀了 ABNB、AEP、AES 三家。DIM、`rdo1`、`rco_b2`、`rco_b4`、`rco_s4` 翻成 buy 之後的理由是同一個論述（強勁營收與自由現金流蓋過毛利壓縮與下修指引），只有措辭差異。沒有看到不同方向產生不同理由的證據。這不是系統性檢查。

## 訓練與方向的診斷

- `rdo1` 的訓練收斂：訓練劑量下的 margin 由 −2.75 升到約 +0.4，KL（答案前綴上、buy／sell 以外 token 的分布變化）約 0.009。
- `rdo1` 與 DIM 幾乎正交：逐 token cosine 的中位數為 −0.003；`rand1` 為 0.001。
- RCO 第一條基底 `rco_b1` 與 `rdo1` 的 cosine 為 0.92（初始化不同），`rco_b2`–`rco_b4` 與 `rdo1` 的 cosine 為 0.08–0.13。四條基底彼此正交（Gram–Schmidt）。
- cone 的訓練沒有完全收斂：結束時基底在訓練劑量（α=1）上的平均 margin 為 −0.74，損失曲線起伏大。`rco_b2`–`rco_b4` 要到 α=2–4 才有效。

## 尚未回答／不能下的結論

- 尚未檢驗 `rdo1` 是否對證據有反應（可能只是「說 buy」的開關）。後續見 [rdo-cone-evidence-v1](../rdo-cone-evidence-v1/proposal.md)。
- 單一訓練 seed，沒有變異估計；只有 Qwen；沒有 buy→sell 分母。
- 方向是直接對「翻成 buy」訓練的，翻得多是預期中的結果；與 DIM 的比較不能解讀為「DIM 較差」。
- 未修改 claim ledger；若要將任何結果升級為論文主張，須另開協議（多 seed、多模型、雙向分母）。
