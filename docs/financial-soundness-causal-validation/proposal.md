# 財務判斷相關候選的因果驗證

**狀態：proposed，探索版 V1，程式已實作，不授權 formal run。** 本提案獨立於 [候選定位](../financial-soundness-localization/proposal.md)。目標模型 Qwen3.5-4B；本輪不包含 Buy/Sell 或真實公司財務真值驗證。真實模型端到端 smoke 已完成，見 [工程記錄](details/report-smoke.md)；工程通過不構成財務神經元認證。全部 72 個候選的後續結果見 [完整探索報告](report.md)。

## 1. 需要干預改變判斷，才能支持候選參與概念運用

本輪檢驗 discovery 選出的候選在未參與定位的 calibration / held-out 財務題中，是否影響短期償債判斷。損害有正確答案的財務題，且不以同程度損害非財務比較或改變所有答案傾向，才可能支持財務功能；單憑高激活、詞彙 readout 或一次決策翻轉均不足。

| 方法來源 | 原始設定 | 本輪適應與限制 |
| --- | --- | --- |
| [Meng et al., Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) | 事實關聯的定位、恢復與編輯 | 僅干預已選 MLP 座標，不做權重編輯；財務任務不是單一事實回想 |
| [Entity-cell V3](../entity-cell-localization/details/proposal-v3.md) | 以事實回想干預驗證候選 | 正確財務答案取代總部等事實，舊 gate 不移植 |

「恢復」在本探索版中只指將同題同座標的 clean 值寫回，驗證 hook 可逆及計分一致。這不是獨立的機制證據：在同一位置撤銷抑制本來就應恢復。跨配對替換則提供另一項方向性診斷，但仍不證明概念只存於此座標。

## 2. 候選與模型身份必須在驗證前固定

唯一上游為完成的定位 run。run-causal-validation 接受 --source-run、--model、--run-id、--artifact-root；不接受任意指定候選座標、重新排名或更換 split。固定使用上游選中的所有候選，在 calibration 與 held-out 的 evidence/comparison 題上測試。相同座標來自不同 family 時只干預一次，保留全部來源標記。

驗證 source manifest 的 complete 狀態與所有輸出 hashes，並核對資料、protocol、候選、模型 checkpoint hashes 及 tokenizer 指紋。資料篡改、身份不同、空候選或缺少對照就中止；不得以其他 run 候選補位。

## 3. 干預範圍只限 prompt 句末的單一通道

hook 位於 dense MLP down_proj 的輸入。觀測與干預位置固定為 prompt 最後一個非 padding token；teacher forcing 期間仍用相同絕對位置，不干預答案 token，不把 −1 誤當延長後的句末。

每個候選在每個 a/b 題目中都跑以下條件：

- clean：不改變輸入。
- scale：該座標乘 0.5 及 0.0；本輪不使用負倍數，避免把符號反轉等同關閉。
- random：同層、非任何已選候選的神經元，乘同樣兩個劑量。以 seed=0 固定抽樣；不是激活幅度匹配對照。
- restore：先乘 0，再寫回同題 clean 值；檢查 margin 是否在工程容差內恢復。
- donor：將同一配對另一成員的該座標值寫入當前題的句末，a→b 與 b→a 都執行。來源與目標可以不同 token 長度，但各只映射自身句末。

donor 值只在記憶體存在，不持久化。hook 發生例外也必須移除。若同層沒有非候選控制神經元，拒絕執行，不拿候選當隨機對照。

## 4. 同時看正確答案損害、條件區分與整體答案偏移

使用定位提案同一定義 M，不更換答案或 scorer。令 y=1 或 −1 為題目的正確方向：

- 正確答案優勢變化：y×(M_intervention−M_clean)，負值表示損害。
- 條件區分變化：[(M_a−M_b)_intervention−(M_a−M_b)_clean]。
- 整體答案偏移：[(M_a+M_b)_intervention−(M_a+M_b)_clean]/2。

若只看到所有題目同向偏移，不認定概念運用受損。按 family、answer_mode、split、候選、條件、劑量分組，先在 group_id 內平均改寫，再以 group_id 作 bootstrap 單位；並列 evidence 與 comparison 的變化及 random 對照。樣本很少時只報描述性數值，不作人口推論。

對 donor 另報 (M_donor−M_clean)/(M_source_clean−M_clean)。若分母絕對值不超過 max(1e−5,1e−4×abs(M_clean))，比率設 null 並記原因，不除近零量。比率可超出 [0,1]，不裁切；也不能把比率大直接當財務特異性證明。

restore 超過相同容差時視工程 postcheck 失敗，run 不得完成。探索版不設 causal certification 或 success=true：需要後續凍結效應門檻、功效、對照等效界與多重比較，才可作正式確認。

## 5. 使用獨立 CLI 與 compact artifact，保留完整失敗狀態

專屬子命令：financial-soundness run-causal-validation --source-run PATH --model PATH --run-id ID --artifact-root PATH。只執行本因果階段，不在同一命令猜測定位或干預模式。

Input 為來源 run 的 manifest、prepare/prompts.json、prepare/protocol.json、analyze/candidates.json。來源欄位與 token 契約見定位提案。新的 run 保存 source run 絕對位置與 manifest SHA-256，並複製驗證後的 compact 輸入供審計。

Outputs 位於 artifacts/<model-slug>/financial-soundness-causal-validation/runs/<run-id>/：prepare/protocol.json、prepare/prompts.json、forward/effects.jsonl、analyze/summary.json、manifest。

effects 每筆含 schema_version、pair_id、group_id、split、family、answer_mode、member、layer、neuron、control_neuron、condition、scale、clean_margin、margin、correct_margin_delta、transfer_fraction、transfer_reason。所有分數須 finite；不適用欄位為 null。禁止 raw activations、residuals、hidden states、gradients、Jacobian 或 KV cache。manifest 對各 compact 輸出計算 hashes。

shared workflow 為 prepare → forward → analyze → finalize。共用 mechanics 使用 [core](../shared-experiment-core.md)，來源研究語義由 owning package 處理，不 import entity-cell。既有 V1/V2/V3、E3/E4 結果及 schema 不變。

## 6. 工程通過與科學結論分開

位置區間、空序列、有限數值與 tokenization fail-closed 沿用定位契約。formal run 前必須真實模型完成至少一組財務/非財務配對、兩種答案模式、全部干預及 restore、donor、analyze/finalize 的 smoke。若 smoke clean 本來答錯，仍輸出結果，但不能將錯誤答案的變化稱作能力損害。

CLI 目前只授權探索，不以 unit tests 代替模型 preflight。研究結論最多是受測座標、提示詞、劑量與位置下的干預效應；沒有作用不證明神經元與財務無關，更不證明偏好分散。公司名稱效應及 Buy/Sell 留待獨立提案。

首次 formal run 後凍結；若更改候選來源、prompt family、primary outcome、位置/劑量、controls 或 gate，依 [versioning](../documentation-system.md#experiment-versioning) 另立版本，不回填。

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**

- [financial-soundness-localization](../financial-soundness-localization/proposal.md)（資料／產物依賴）：以定位產出的候選座標做獨立干預驗證；不從驗證句式回頭挑候選。

**後續**：尚無已立案的後續研究；不把報告中的建議視為已授權實驗。

最終／最新結果見本研究的 [report](report.md)。
