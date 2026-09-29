# rdo-cone-v2：修正訓練流程後重做梯度訓練的方向與 cone（Qwen3.5-4B）

**狀態：**事前協議，2026-09-29 於 tune run 開始前凍結（網格與 seed 數經使用者確認）。**上層：**[rdo-cone-v1](../rdo-cone-v1/status.md)（流程有缺陷，見下）。**性質：**探索性，單一模型、只有 sell→buy；結果不改寫 v1，也不改 claim ledger。

## 為什麼做

rdo-cone-v1 的結論中，「存在比 DIM 更有效率的訓練方向」與「它與 DIM 幾乎正交」站得住；但與 cone 有關的比較站不住，因為訓練流程有這些問題：

| v1 的問題 | 影響 | v2 的修正 |
|---|---|---|
| cone 沒有收斂（最後一步基底在 α=1 的平均 margin −0.74） | `rco_b2`–`b4` 要 α=2–4 才有效，cone 與單向的比較不公平 | 預先登記收斂判準；沒通過就不進 full |
| 沒有驗證集，只看訓練 batch 的 margin（batch 4，起伏大） | 無法判斷收斂或過擬合 | 402 家建構公司切成 322 訓練／80 驗證；定期在驗證集上評估 |
| 超參數只在 5 步 smoke 後就固定 | 沒有依據 | 在訓練／驗證上跑小網格，依預先登記的規則選 |
| 固定學習率、150 步、batch 4 | 收斂不足、雜訊大 | 300 步、batch 8、warmup + cosine 衰減 |
| cone 損失中基底合計權重只有 1（每條 1/n） | 基底被訓練得少 | 錐內樣本與基底每條權重相同 |
| 單一訓練 seed | 沒有變異估計 | full 用 3 個訓練 seed |
| 訓練 log 只在 lab 的 job 目錄 | 需要手動拉回 | 腳本自己寫 `train.log` 到 run 目錄 |

## 設定（與 v1 相同的部分）

模型 Qwen3.5-4B，L16，balanced prompt，steer suffix（K=100）prefill 注入，complete-object 解析；切分 seed `20260923`（402 建構／101 評估）；DIM 的 Top/Bottom10 取自 confirmation-v1 的 Qwen ranking（只用建構公司）；等範數劑量 `α · ‖d[p]‖ · u`，`u` 為所有 suffix token 共用的單位向量；訓練劑量 `α_train = 1`；retain KL（答案前綴上 buy／sell 以外 token 的分布）權重 `λ_ret = 1`；隨機初始化；cone 維度 n=4；沒有 ablation 損失。

## 訓練流程

- **訓練／驗證切分：**402 家建構公司依 seed `20260929` 抽 80 家為驗證、322 家為訓練。評估組 101 家完全不參與訓練、調參與選擇。
- **損失：**`softplus(τ − M) + λ_ret · KL`，`M = log p(buy) − log p(sell)`。`τ = 0` 即 v1 的損失。
- **最佳化：**Adam，300 步，batch 8（每步不放回抽 8 家訓練公司），前 30 步線性 warmup，之後 cosine 衰減到 0.1 × lr；每步後 Gram–Schmidt。
- **cone 每步：**每家公司抽 2 個錐內方向，加上 4 條基底，共 6 個單位向量，損失取平均（每個向量權重相同）。
- **驗證：**第 0、50、…、300 步，在 80 家驗證公司上以 α=1 算固定答案 margin 與 KL（只做 forward，不生成）；單向為 `rdo1`，cone 為 4 條基底、centroid 與 2 個固定的錐內樣本。
- **最終方向：**取第 300 步（不按驗證挑 checkpoint）。
- **紀錄：**每步的訓練指標與每次驗證寫進 `training.json`；stdout 同時寫到 run 目錄的 `train.log`。

## 預先登記的收斂判準

在最後一次驗證（第 300 步，α=1）時，`rdo1` 與 4 條 RCO 基底**每一條**在驗證集上 `M > 0` 的比例都 ≥ 0.8，記為「收斂」。

## 階段一：調參（只用訓練／驗證，不碰評估組）

- 網格：`lr ∈ {1e-3, 3e-3}` × `τ ∈ {0, 2}`，4 組；訓練 seed `20260930`。run id 依網格順序為 `rdo-cone-v2-20260929-tune-01`（1e-3, 0）、`-02`（1e-3, 2）、`-03`（3e-3, 0）、`-04`（3e-3, 2）。每組訓練 `rdo1` 與 RCO，只做驗證，不生成。
- 選擇規則：在通過收斂判準的組中，選驗證集 KL（RCO 基底與 `rdo1` 的平均）最小者；相同時取網格順序（先 lr 小、再 τ 小）在前者。
- **四組都沒通過：**不進入階段二。在 status 記錄各組最後的驗證數字，並把「n=4 的 cone 在 α_train=1 下訓練不起來」當作結果回報；是否改 n 或 α_train 另行討論，不在本協議內追加。

## 階段二：full（3 個訓練 seed + 評估）

- 用選出的超參數，以訓練 seed `20261001`、`20261002`、`20261003` 各訓練一次（與調參的 seed 不同），同樣只用 322 家訓練公司。
- 每個 seed 各自在 101 家評估公司上評估（greedy，complete-object 解析，ITT 分母為 α0 為 sell 的公司）：

| 名稱 | 劑量（DIM 範數的倍數） |
|---|---|
| `rdo1` | 1, 2, 4, 8, 16, 32 |
| `rco_b1`…`rco_b4` | 1, 2, 4 |
| `rco_centroid` | 1, 2, 4, 8, 16, 32 |
| `rco_s1`…`rco_s8`（固定 seed 從錐裡抽） | 1, 2 |

- 對照只在第一個 seed 的 job 跑一次（與訓練 seed 無關）：`dim` 1, 2, 4, 8, 16, 32；`rand1` 1, 2, 4, 8。

## 預先登記的問題（僅描述，不做顯著性檢定）

- **V1（收斂）：**各 seed 是否通過收斂判準；未通過的 seed 仍評估，但在結果中標明。
- **V2（seed 穩定性）：**3 個 seed 的 `rdo1` 翻一半劑量；3 個 `rdo1` 兩兩之間與它們與 DIM（逐 token cosine 中位數）的 cosine。
- **V3（基底各自有效）：**每個 seed 的 4 條基底在 α=1 的 ITT flip 率；「全部 ≥ 0.5」記為在訓練劑量上有效。
- **V4（cone 相對單向）：**（a）8 個錐內樣本在 α=1、2 的 flip 率分布（最小／中位數／最大）與同 seed 的 `rdo1` 比較；（b）best-of-N 的對照：以 seed 1 的 `rco_s1`–`rco_s3` 取 best-of-3，對照 3 個 seed 的 `rdo1` 取 best-of-3（3 條獨立訓練的單向），都在 α=1。
- **V5（崩壞）：**`rdo1`、`rco_centroid`、`dim` 的崩壞劑量（parse 率 < 0.9 的最小劑量），測到 32。

## 解讀界線

- 方向是直接對「翻成 buy」訓練的，翻得多是預期；與 DIM 的比較不能解讀為「DIM 較差」。
- 仍然只有 Qwen、只有 sell→buy、只有 balanced prompt；證據敏感度沿用 [rdo-cone-evidence-v1](../rdo-cone-evidence-v1/status.md) 的結論，本版不重測。
- 理由文字（v1 的 Q4）不再檢查：v1 沒看到差異，本版沒有新的判準。

## 實作與執行

- 腳本：`scripts/probe_rdo_cone_v2.py`（重用 `scripts/probe_rdo_cone.py` 的純函式，不修改該檔）；測試：`tests/test_probe_rdo_cone_v2.py`。
- run id：`rdo-cone-v2-<date>-<phase>-NN`，phase 為 `smoke`、`tune`、`full`；`tune` 只訓練與驗證，`full` 訓練後評估。
- 輸出：`artifacts/qwen3.5-4b/concept-cone-steering/runs/<run-id>/`（`training.json`、`directions.json`、`train.log`，`full` 另有 `result.json`）。lab job 結束後也把 `~/.lab/jobs/<job>/output.log` 拉回同一目錄存成 `job_output.log`。
- 成本（依 v1 的實測速度估算，未實測）：調參每組約 30 分鐘，4 組分兩張 GPU 約 1 小時；full 每個 seed 訓練約 30 分鐘加評估約 1 小時（seed 1 多約 15 分鐘的對照），3 個 seed 分兩張 GPU 約 3 小時。
