# evidence-scan-v1：多樣證據下，哪些公司與證據組合讓 Qwen3.5-4B 回答 buy？

**狀態：**事前協議，2026-09-29 於 screen 開始前凍結。**性質：**探索性、只讀 baseline（沒有任何 steering 或訓練）。**上層：**[rdo-cone-v2](../rdo-cone-v2/status.md) 與其檢討（見下）。

## 為什麼做

檢討 rdo-cone-v1／v2 時發現兩個和論文（Wollschläger et al., [arXiv 2502.17420](https://arxiv.org/abs/2502.17420)）差很大的地方，而這個實驗只處理第二點：

1. loss 與目標不同（論文對整段目標回應算 cross-entropy，並同時有 addition、ablation、retain 三項）。**本版不處理。**
2. 訓練資料沒有變化：v1／v2 的 322 個訓練 prompt 只有 ticker 與公司名不同，四句證據與整段指令完全相同；Qwen 在這份 balanced prompt 上對全部 503 家公司都回答 sell，所以也沒有「原本是 buy」的 prompt。論文用約 1,184 個內容各異的 prompt，且有「原本會拒絕」與「原本不拒絕」兩類。

本版先只做一件事：換用每家公司各自的、多樣的證據，量測 Qwen3.5-4B 在哪些公司與證據組合上回答 buy、哪些回答 sell。結果決定之後能不能做有兩個方向的訓練，不在本版判斷。

## 只換證據，其他不動

- 模型 Qwen3.5-4B、pinned chat template、prompt 骨架（指示、Stock Ticker／Name、`— Evidence —`、四個 bullet、JSON 輸出說明）、固定前綴 `{"decision": "`、complete-object 解析，全都與 [confirmation-v1](../confirmation-v1/proposal.md) 相同。
- 唯一不同的是四個 bullet 的文字。

## 證據池

- **來源：**外部 baseline 專案的 trial plan（其產生器不在本 repo），已由 `scripts/convert_baseline_trial_plan.py` 轉成 `data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv`（SHA-256 `6aad67c9…2570`），427 檔股票、每檔 72 個 prompt。這 427 檔全部在我們的 503 家名單內，其中 338 家在建構組、89 家在評估組（切分 seed `20260923`，沿用）。
- **池的結構：**每家公司有 8 則證據（各約 60 字，含具體數字，並各自宣告約 ±5% 的股價影響），4 則指向上漲、4 則指向下跌。`attribute` 條件的 30 個 prompt 是從這 8 則中隨機抽 2 則正向 + 2 則負向、隨機排列的組合。這與「從金融偏誤資料抽出子集、隨機組合」的描述一致，也對應我們要求的「證據正負各二」。
- **正負方向的還原：**CSV 不含方向標記。`scripts/build_evidence_pool.py` 用結構還原：唯一使 30 個 attribute prompt 都是 2 + 2 的 8 選 4 分割（除了左右對調外不可有第二種），再以「price increase／decrease」字面計數決定哪一邊是正向。427 家全部有唯一分割；3 家（CAG、CF、HCA）字面計數差距小於 3，已人工核對三家的證據句，符號正確。
- **輸入檔：**`evidence_pool.json`（SHA-256 `76bb2fc95c3def0ab492ab9d39d4995c87f748cdf1e61efcdabd89975367bab2`）不進 git（2.3 MB，可由上述腳本從 CSV 重建），放在 lab 主機的 `/mnt/train-data-1-hdd/sam/lab-assets/llm-bias-rdo/evidence_pool.json`；每個 run 的 metadata 記錄它的 SHA。

## 設計

每家公司掃 31 個單位：

- `a00`–`a29`：該公司的 30 個 attribute 組合（證據與排序照 CSV，不重抽）。
- `ref`：凍結的 balanced 四句（P1、P2、N1、N2；與先前所有實驗相同），作為與舊結果的銜接。

每個單位做兩種選項順序：`"buy" or "sell"`（**canonical**，與先前實驗相同）與 `"sell" or "buy"`（reversed）。

| phase | 公司 | 內容 | 列數 |
|---|---|---|---|
| `screen` | 建構組 338 家 | 每個單位 × 2 順序的固定前綴 margin（FP32 讀出，`M = log p(buy) − log p(sell)`） | 338 × 31 × 2 = 20,956 |
| `confirm` | 評估組 89 家 | 同上，加 canonical 的 greedy 生成（最多 192 個新 token）與 complete-object 解析 | 89 × 31 × 2 = 5,518，其中 2,759 個有生成 |
| `analyze` | — | CPU 彙整 | — |

- **buy 的定義：**screen 中 `M > 0`（僅是讀出，不宣稱決策）；confirm 中生成解析出的 `decision == "buy"`。
- **不用評估組選擇任何東西。**screen 與 confirm 各自獨立彙整，confirm 用來檢查 screen 的規律在未參與的公司上是否成立。
- **切 shard：**依公司順序 `i mod n`，各 shard 用不同 run id（`evidence-scan-v1-20260929-<phase>-NN`）。

## 預先登記的問題（僅描述，不做顯著性檢定）

- **Q1（有沒有 buy）：**建構組 attribute 的 canonical buy 比例；每家公司 30 個 prompt 的 buy 比例分成全 sell／混合／全 buy 的公司數與分布；各 sector 的平均。
- **Q2（誰決定答案）：**canonical margin 對「公司」、對「四個位置的正負排列（6 種）」、兩者相加的 R²；各位置為正時的邊際 margin 變化（公司固定效應）；第一項與最後一項的正負對 buy 比例的影響。
- **Q3（選項順序）：**reversed 相對 canonical 的平均 margin 差與 buy 比例差。
- **Q4（與舊結果銜接）：**`ref` 的 canonical／reversed buy 比例。先前 balanced 對全部公司都是 sell，這裡應複現（是檢查，不是假設檢定）。
- **Q5（複現與讀出一致性，confirm）：**評估組的 attribute canonical buy 比例是否與建構組同量級；生成 parse 率；margin 符號與生成決策的一致率。

## 預先登記的判讀規則（描述性門檻，不執行後續）

以建構組 attribute 的 canonical buy 比例 `r` 與 confirm 的 parse 率為準：

| 結果 | 判讀 |
|---|---|
| `0.1 ≤ r ≤ 0.9`，且混合公司佔一成以上，且 confirm parse 率 ≥ 0.95 | 這個證據族在 Qwen3.5-4B 上同時有 buy 與 sell 的 baseline，可作為兩個方向的訓練與評估材料；下一步另開新版本協議。 |
| `r < 0.1` | Qwen 對這個證據族仍幾乎都回 sell；需要換模型或換操作，才可能有 buy baseline。 |
| `r > 0.9` | 對稱：幾乎都回 buy。 |
| confirm parse 率 < 0.95 | 生成格式在這個證據族上不穩，fixed-prefix margin 不能當決策讀出，須先處理格式。 |

## 解讀界線

- 「balanced」在這裡只是結構上正負各二，不代表語意上平衡；證據的強度與措辭由外部產生器決定，且證據句含公司名，證據與公司身分不可分開解讀。
- 只有 Qwen3.5-4B、只有 427 家公司子集、只有這一個外部證據族；不推論到其他證據型式。
- fixed-prefix margin 只是讀出；「回答 buy」的宣稱以 confirm 的真實生成為準。
- 本版沒有 steering 或訓練，結果不改 claim ledger，也不改寫先前任何版本。

## 實作與執行

- 腳本：`scripts/build_evidence_pool.py`（建池）、`scripts/probe_evidence_scan.py`（screen／confirm／analyze）；測試：`tests/test_probe_evidence_scan.py`。
- 輸出：`artifacts/qwen3.5-4b/concept-cone-steering/runs/evidence-scan-v1-<date>-<phase>-NN/`（`result.json`、`scan.log`；analyze 另有 `summary.json`）。lab job 結束後把 job log 拉回同一目錄存成 `job_output.log`。
- 成本：待 smoke 實測後補在 status（不在此估算）。
