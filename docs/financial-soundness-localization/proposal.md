# 財務穩健判斷相關神經元定位

**狀態：proposed，探索版 V1；不授權 formal run。** 程式已實作，僅授權 smoke / exploratory；真實模型定位 smoke 已完成，見 [工程記錄](report-smoke.md)，不構成財務神經元認證。完整探索結果見 [探索報告](report-exploratory-v1.md)。目標為 Qwen3.5-4B（.cache/models/qwen3.5-4b，GPU forward 使用 bf16）。

## 1. 三類短句分別探索話題、公司評價與財務判斷

本輪問哪些 MLP 通道的反應會隨財務提示詞改變，且能跨句式重現。財務穩健指履行債務、維持資金運作的能力；有正確答案的題目先限定現金對一年內到期債務的覆蓋，不等同完整財務評估。

| family | 第一成員與第二成員 | 可解讀的差異 |
| --- | --- | --- |
| topic | The financial stability of the company is / The operational stability of the company is | 財務與非財務話題差異；沒有客觀正確答案 |
| company | The financial stability of Johnson & Johnson is / The financial stability of the company is | 公司名稱相對匿名的既有評價；不是現實公司的財務真值 |
| evidence | 現金多於到期債務 / 到期債務多於現金 | 給定條件下的短期償債判斷 |
| comparison | 庫存多於訂單 / 訂單多於庫存 | 非財務數量比較對照 |

英文配對示例：The company has USD 20 million in cash and USD 5 million in debt due within one year. Its cash coverage of that debt is: 。第二成員只交換兩個金額。不要在證據中寫入 strong / weak 等答案暗示。

topic 與 company 同時探索 stability、debt burden、liquidity 三種短句；各自使用 stable/unstable、low/high、adequate/inadequate 的固定完整續接。三種概念分開排名。不得把不同答案的 top-1 機率差當作效應。

不使用 Item 標籤、10-K 或固定 header。V1 entity-cell 曾受模板影響，但短句也可能只觸發詞彙或句法單元。V3 沿用 V2 定位句式；F0/F2/F3 是事實驗證題，不是全部定位輸入。

## 2. 文獻僅支持方法動機，不保證財務概念位於單顆神經元

| 來源 | 原始設定 | 本輪修改與風險 |
| --- | --- | --- |
| [Geva et al., Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) | 前饋層輸入模式及輸出詞彙 | 比較財務短句反應；不能由高激活推斷知識儲存 |
| [Meng et al., Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) | 事實定位與編輯 | 不執行 ROME，不把事實機制等同評價機制 |
| [Entity-cell V2](../entity-cell-localization/proposal-v2.md) / [V3](../entity-cell-localization/proposal-v3.md) | 自然句定位、事實干預 | 重用方法原則，不沿用實體專屬性及事實崩塌 gate |

## 3. 定位只使用 discovery，驗證句式不能回頭挑候選

內建資料的 discovery、calibration、held-out 各使用兩種不同句式。有數值的 family 各 split 使用兩組不同數值；相同數值組的改寫屬相關觀測。company 使用預先宣告的公司清單，跨 split 保留公司而隔離句式，因此只能稱未見句式驗證，不能稱未見公司驗證。

每對資料有 a、b 兩成員；evidence/comparison 的 a 為較充足、b 為較不足。兩種答案模式分開分析：直接詞彙續接，以及交換 A/B 對應的選項題。選項語義由答案映射指定，不以字母大小推論正負。

主要觀測為最後一個 prompt token、down_proj 前的 MLP 通道值。所有選定層皆可定位，不沿用早期 L0–L5 限制。teacher-forced 答案不能參與定位。

對每個 family、concept、answer_mode 及座標，令每對反應為 a_i、b_i，d_i=a_i−b_i。候選分數定義為 mean(d_i) / max(sqrt(mean((a_i²+b_i²)/2)), 1e−6)。尺度不超過 1e−6 的候選排除。以絕對分數取各組 top-k，並輸出符號一致率；正負方向皆保留。此值是標準化反應差，不是 causal effect。

calibration / held-out 僅回報 discovery 候選的分數，不再選擇。探索版不設認證 gate、不輸出成功認證；候選數 top-k 預設 3、seed 預設 0，均寫入 protocol。

## 4. 固定答案分數與對照一起決定可解讀範圍

M(x)=log P(正向完整續接|x)−log P(負向完整續接|x)。多 token 用逐 token 條件 log probability 總和，無長度正規化；完整字串含前導空白。core 的 prefix 驗證若失敗就拒絕，不退回獨立答案編碼。forward 為 bf16，final norm、unembedding 與 log-softmax 使用 FP32。

有正確答案的 family 回報兩題方向及 D=M(a)−M(b)，不能只看 D>0 就說兩題皆答對。topic/company 不計正確率。各組結果分開，pair_id 的改寫先聚合到數值組再計描述性 bootstrap，不能把 token 或改寫視為獨立樣本。

comparison 用來辨別一般數量比較；topic 的 operational 詞句是非財務話題對照；答案映射交換用來辨別輸出標籤傾向。它們不能充分排除所有語言能力損害。干預檢驗另見 [財務判斷因果驗證](../financial-soundness-causal-validation/proposal.md)。

## 5. Input / Output 保存 compact 結果與可驗證來源

CLI 所屬 package 為 llm_bias.financial_soundness，唯一頂層命令 financial-soundness。各階段使用獨立子命令；不修改既有 entity-cell API。

- prepare-prompts --output PATH --companies NAME [NAME ...]：產生探索版資料 JSON；不載入模型。
- run-localization --prompts PATH --model PATH --run-id ID --layers INT [INT ...] --top-k INT --artifact-root PATH：完成 prepare → forward → analyze → finalize。層預設為全部 decoder 層，top-k 預設 3；只允許 exploratory 狀態。

Input schema_version=1，protocol_version=exploratory-v1，pairs 為物件陣列。每筆有 pair_id、family、concept、split、group_id、answer_mode、a/b。a/b 各含 text、positive、negative、expected（1、−1 或 null）。文字必須非空；pair ID 唯一；split 與 family 使用封閉值。資料 hash、分組、完整 token IDs、答案 IDs 及最後位置在 prepare 輸出保存。

Outputs 位於 artifacts/<model-slug>/financial-soundness-localization/runs/<run-id>/：prepare/prompts.json、prepare/protocol.json、forward/behavior.jsonl、analyze/candidates.json、analyze/summary.json 及 run manifest。候選保存 family/concept/mode、layer/neuron、score、rank、跨 split 統計，不保存完整通道值。protocol 記錄資料 hash、模型檔案 hashes、tokenizer 指紋、程式 identity、layers、top-k、seed。來源 identity 必須供下游驗證。

共用 encoding、forward、scoring、statistics、serialization 與 lifecycle 使用 [core](../shared-experiment-core.md)。禁止保存 raw activations、hidden states、residuals、gradients、Jacobian、KV caches。所有數值必須 finite；缺失統計用 null 與原因。此流程不需要 lens。

## 6. 退化條件不產生認證結果

空輸入、未知 family、錯誤答案方向、缺少 pair 成員、重複 ID、必要對照或 split 缺失、token 超長、非法座標、非有限數值直接拒絕。若全部尺度退化，回報空候選，下游中止。模型答錯不靜默刪題；回報完整行為並標明探索限制。

prompt 非 padding 區間 [0,T) 分成 [0,T−1) 與觀測點 [T−1,T)，互斥且完整覆蓋；答案位置不在觀測區。採單樣本推論，不允許靜默截斷。

同輸入重跑與 self-restore 的 FP32 margin 容差為 max(1e−5, 1e−4×abs(clean margin))。這是工程一致性容差，不是財務效應 gate。若 bf16 backend 超標須記錄並停止驗證，不事後放寬正式判定。

formal discovery/calibration/test 仍被禁止：先以真實 tokenizer/model 完成所有 family、hook、controls、analyze/finalize 的端到端 smoke，人工審核行為，再另行凍結最小效應、樣本量與多重比較政策。fake-model 測試不能替代此 preflight。

## 7. 新設計不得回填舊版結果

目前是未執行的探索草稿；公司評價干預、Buy/Sell 與真實財報皆不是本輪認證目標。未找到候選也不證明偏好分散。首次 formal run 後凍結協議；prompt family、排名、gate、controls 或 split 改變時，依 [versioning](../documentation-system.md#experiment-versioning) 建立版本索引及 proposal-v1/v2，不覆寫歷史。
