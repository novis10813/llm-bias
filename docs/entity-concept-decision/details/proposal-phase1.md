# Phase 1：先驗證 L15 有效狀態差能否辨識概念

**狀態：proposed，Draft 1；無模型基礎工具已實作，讀出／驗證 pipeline 未實作、模型執行未授權。**本階段只提名並驗證語義，不作決策因果宣稱。原始 Input/Output、文獻適應、數值與版本政策依[共通契約](design-and-validation.md) §1–7，均為本協議的一部分。

## 1. 假說與最小交付

H1：固定 e-01 的 k=8 子空間中，至少有一個方向可在未見改寫上分辨事先定義的概念差異，且不是只分辨公司名、語法或一般正負立場。

交付最多 3 個候選，每個有人工定義、正負範例、混淆概念、方向、符號、獨立量測門檻及完整 split provenance。若只能讀出好解釋的詞，輸出 descriptive 提名並停止，不能稱「找到中間概念」。

## 2. 先檢查現有範圍，再建立獨立概念材料

### 2.1 P1a：離線核對與有限讀出

先核對 [共通契約 §2](design-and-validation.md#21-固定輸入與-shape) 的基底、上游 records、hash、公司名冊與 pair 方向。不重新 SVD、不按新 outcome 選 basis。

獲得模型執行批准後，先在舊四家配對公司 BDX/IT/BLK/NSC 的 canonical variant 上取得 L15 instruction 狀態。依每個完整 state condition 的 vocabulary softmax 先跨位置／公司平均，再取 top-k；記錄 clean company 差及 k=8 置換後的描述性讀出。不能用「先挑 top-k 再平均」，不能把單獨 delta 經 softmax 當下一詞機率。

閱讀 top-k 只提名，固定最多 3 個內容假說；保留競爭解釋，例如產業類別、公司大小、一般正負評價。這些是候選類型，不是已知結果。若詞彙主要為標點或無法命名，可只做材料先行的有限驗證，不自動擴展全層／全詞表搜尋。

### 2.2 P1b：概念對照材料與層位置一致

每個概念建立「相同語句骨架、只改目標屬性」的成對材料；不用真公司名或 buy/sell 答案。概念若天然帶好壞判斷，需額外對照一般正负評價，否則只能標「一般立場」候選，不能當公司相關內容。

建議初始預算每概念 12 個語句家族：6 fitting、3 validation、3 audit；每家族一對正負句，合計 24 prompts，三概念最多 72。此為小規模可行性預算，不是足夠統計功效的宣稱。人工審查後如材料不足以排除詞面線索，擴充需在 audit 前另批准。

材料使用與決策任務相同的 instruction suffix 與答案 prefix，前文改為不具名的概念描述；以 L15 instruction span 作測量。這是跨語境適用性的假設，必須在 Phase 2 自然公司 prompt 再驗證。跨句型家族分割，不把同義改寫當新獨立樣本。

## 3. 用獨立語義差建立方向，不按決策調整符號

以下向量為 column convention；程式 tensor 實作可 transpose，但 metadata 必須明列。

Q 為固定 2560×8 正交基底。對 fitting 家族 i，先在 instruction positions 平均各句 L15 狀態得到 h_i⁺、h_i⁻。只在 RAM 計算：

- d_c = mean_i(h_i⁺−h_i⁻)。i 指語句家族；同一家族若有多個改寫，先平均該家族的 pair 差，再跨家族等權平均，避免改寫數量改變權重。
- u_c = QQᵀd_c；若其 norm 小於 calibration 的方向下限，判退化，不正規化放大。
- v_c = u_c/||u_c||；方向正負只由材料語義決定。
- a_c(x) = mean_p(v_cᵀh_x[p])，另報每句相對 fitting 中心的 score；中心不持久化，閾值以單一 scalar 保存。

保存 retained norm fraction ||u_c||/||d_c|| 作限制說明。高 overlap 不是語義證明；低 overlap 也不否定概念在 k=8 外存在。

本版本逐概念測單一方向，最多 3 個，不進行概念聯合最佳化。候選若高度共線，標成無法區分並在 audit 前合併或只留一個；不用 Gram–Schmidt 把語義改掉再保留原名。需多維概念時另修訂 direction source。

## 4. 必要驗證與進下一階段條件

| 檢驗 | 量測 | 失敗解讀 |
|---|---|---|
| 語義辨識 | 在 validation 決定閾值後，audit 的成對 score 差與分類表現；以語句家族為單位 | 無法辨識，或只在 fitting 成立 |
| 詞面／語法對照 | 概念相同但措辭改變；詞面相近但概念不同；長度／格式匹配 | 不能排除模板或關鍵詞偵測 |
| 一般立場對照 | 獨立正負評價材料、相同概念不同評價 | 只剩一般好壞讀出，不能命名為更特定內容 |
| 隨機及標籤對照 | k=8 內 frozen seeds 隨機 unit vectors；fitting 家族內交換 labels 的 null | 任意方向同樣可分，沒有候選特異性 |
| 重現 | 改寫家族及 sign convention 不改候選定義 | 語義標記不穩定 |

**候選通過需同时滿足**：audit 的概念差超過 δ_a（>2τ_a）；對 family-level label permutation 的檢驗經候選 Holm 校正 p<0.05；勝過 frozen random controls 的預定效應界；混淆對照通過人工盲審與預先固定的統計要求。audit 若太小無法支持判準，標 `inconclusive`，不能拿 validation 充當確認。

具體 δ_a、最小 balanced accuracy、混淆對照界及 random seeds 在 calibration 後、audit 前凍結，現為 freeze blockers。同時保存所有失敗候選，不只寫勝出者。人工名稱與決策無關，Phase 1 不以 margin 篩候選。

## 5. Input/Output、CLI 與停止政策

Input：共通 refs；材料採單一 `concept_materials.json`（**proposed 路徑**），含 `schema_version:1, concepts:[...], pairs:[...]`。pairs 每列 `id,concept_id,family_id,split,text_positive,text_negative,review_status,confound_tags`；concepts 每列 `concept_id,definition,positive_pole,negative_pole,excluded_interpretations`。材料尚未建立，未審不能執行。首批 reader 與純函式契約見[基礎實作 spec](implementation-phase1-foundations.md)，不含 runner、正式材料或 semantic gate。

Output：共通 stage records；另 `analyze/candidates.json` 含定義、v_c [2560]、Q 座標 [8]、語義 test 結果與 frozen hashes，`forward/readout.jsonl` 僅 top-k/rank/probability 與 prompt/position provenance。不持久化每句高維 h、d 或完整詞表矩陣。

每一 condition 的語義判定數字須 finite；缺控制、span 不對齊、canonical lens 驗證失敗、零 norm 或 hash 改變時中止。candidate 為空則 finalize 為負結果，不開 Phase 2。CLI 未實作，不提供虛構執行命令；實作後專屬 Phase 1 subcommand 與此協議逐項綁定。

## 6. 最小 preflight 與批准節點

先跑共通 §6 unit tests，再獨立批准真實 smoke：至少一組概念 pair、一組公司 pair、所有 controls、候選空集合與 finalize 路徑。隨後才批准 P1a/P1b 小規模 calibration；凍結材料、方向規則與數值門檻後才能開 audit。任何核心材料／direction source／gate 修改觸發共通 §7 version policy。
