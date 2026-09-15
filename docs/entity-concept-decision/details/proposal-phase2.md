# Phase 2：候選概念分量必須改變公司相關決策差異

**狀態：proposed，Draft 1，依賴 Phase 1，未實作／未授權。**文獻適應、精確上游 schema、模型／tokenizer／lens、finite、position、容差、lifecycle 與版本規則依[共通契約](design-and-validation.md)，為本協議的一部分。決策介入本身不需 lens，僅附加語義讀出才需。

## 1. 假說與結果邊界

H2：在同證據公司 prompt 配對中，僅置換 Phase 1 候選概念的自然狀態差分量，會產生可辨識、勝過 matched controls 的公司相關決策轉移；且剩餘分量的表現與概念分量不同。

這不要求把 V1 的 group gap 減半，也不把 mean shift 為零當機制成立的必要條件。需要排除的是「只要加任意方向就偏 buy/sell」與破壞造成的同向 sell；副作用照實報告，不宣稱選擇性控制成功。

## 2. 公司配對與獨立確認

development 使用舊 16 家與既有模板。Phase 1 concepts 已凍結，不以 Phase 2 margin 換名稱、調符號或重新 fitting。

confirmation 建議先規劃至少 8 個不重疊的新公司 pair（16 家）、每 pair 雙向、2 個新模板家族。這是設計起點，樣本量由 development 的 pair-level 變異與預設 CI 寬度／δ_M 做功效或精確度規劃後固定。新公司須檢查歷史使用與別名；不能看確認效果後再選取。

為區分兩種泛化，另外安排新公司×舊模板、舊公司×新模板；不能讓所有格都借用同一筆校準。primary confirmation 為新公司×新模板，另兩格為診斷。同 pair 不跨 split、配對公司不能共享於不同 confirmation pairs，避免假獨立；若做不到，必須預先採公司相依性處理，不能沿用獨立 pair test。

財務證據、instruction、答案 prefix 在每公司 pair 中完全相同。不同模板只在各自模板內映射位置。固定答案 M=log p(buy)−log p(sell)；decoder bf16，FP32 tail，雙向記 raw 與 toward-source。

## 3. 投影定義與完整對照

对每個候選 v（unit column）定義 P_C=vvᵀ、P_K=QQᵀ；P_C⊂P_K。source 與 target 自然狀態差 Δ[p]=h_s[p]−h_t[p]，按 instruction 相對位置對齊。

| arm | 在 target L15 的操作 | 用途 |
|---|---|---|
| clean／self-source | 不改變 h_t | structural no-op、baseline |
| full | h_t+Δ | 整體狀態置換參照 |
| k8 | h_t+P_KΔ | 本次 in-run 有效子空間參照 |
| concept | h_t+αP_CΔ | α=1 為 primary；0.5 為 dose 描述 |
| remainder | h_t+(P_K−P_C)Δ | 去掉候選後仍可傳遞多少 |
| reconstruction | h_t+P_CΔ+(P_K−P_C)Δ | 數學／數值對照，不能當非線性效應可加的證據 |
| random | 在 K 內預先固定的隨機 rank-1 方向投影 Δ，縮放至 concept patch 的逐位置 norm | 排除維度與介入大小；零 control norm 不重抽，標無效 |
| shuffled source | 沿同一 concept 方向，使用預先凍結的錯配 source-target 係數，匹配 norm | 排除只沿方向推動；derangement 固定，不依結果選 |
| generic valence | 用獨立正負評價材料建立的方向，匹配 norm | 檢查是否只是一般立場方向 |

random seeds 建議預先固定至少 10 個，成本不足需在 confirmation 前改設計，不得只挑一個最弱控制。rank-1 逐位置 norm matching 會消除係數大小差，因此 shuffled source 若與 concept patch 僅差原本大小、匹配後逐位置相同，就不能作獨立有效對照；preflight 須計算 patch 差異與非退化覆蓋率，不足時停止此 specificity 判定，不以重抽 donor 製造差異。候選共線不作獨立概念計數。norm matching 是研究控制，不是自然狀態 transfer；記錄每個 scale，超過預先界限則 fail-closed。

匿名對照使用 source 匿名化後与 target 匿名化後的狀態差會自動為零，**不能拿此 tautological no-op 證明特異性**。應將 named pair 的同一 concept delta 按位置施加至匹配的 anonymous prompt，與同 norm controls 比較，獨立報告 raw shift。先驗證 named／anonymous 的 instruction span token 數及 token IDs 完全相同，再以相對 offset 注入；不符即中止。另以概念無關公司的 matched prompts 檢查，不把匿名一筆當完整能力評估。

## 4. 科學判準先檢查自然差，再看介入效果

1. **自然內容**：在公司 prompt 上，候選 loading 差必須超過 τ_a；Phase 1 能分語義但此處無公司差，不能稱公司中介。不能只因 ΔM 大才納入方向。
2. **主要 transfer**：T_C=sign(M_s−M_t)·(M_concept−M_t)。每個不重疊 company pair 先分模板、雙向報告，再取 pair summary；以 pair 為重抽／sign-flip 單位，不把位置／反向／seed 當獨立樣本。
3. **controls**：primary contrast 是 T_C−mean_seed(T_random)，其次為錯配與一般立場对照。候選經 Holm 校正 p<0.05，且 pair-level CI 下界超過事先固定的 δ_M（>2τ_M），才稱 effect supported。其他兩個 specificity contrasts 的最小界亦在 confirmation 前凍結。
4. **双向與抹除**：分 source margin 較高／較低兩組的 raw shifts；需兩組均有事先規定的最小合格 pair 數及正向 transfer 證據。若一組不合格，僅作情境限定結論，不以 pooled positive 宣稱一般路徑。
5. **非目標作用**：預先凍結 named 與 anonymous／無關公司 matched patch 的交互 contrast、CI 與可接受界；若無法區別，只稱一般決策方向。保留 mean shift、remainder 效果與所有 controls，不因其不符合故事而省略。

full／k8 的 raw effect 與 CI 都報告；effect ratio 只作 secondary，|denominator|≥max(0.2,4τ_M) 才計算。0.2 為承接舊保護底限，不是概念 gate。source-target clean 差低於 2τ_M 的方向無法決定符號，標 `inconclusive`。不能依某臂效果篩掉難看的方向。

確認公司數、δ_M、random/scaling seeds、雙向 eligible 下限、交互界與 power 目標仍未凍結；未填齊不可 formal。若結果不支持任一概念，Phase 2 收為 null／inconclusive，不啟動 Phase 3。

## 5. Input/Output、preflight 與停止

Input：Phase 1 completed candidate file 與 hash、獨立 split/pair manifest、相同 evidence/prefix、model/tokenizer identity。新增公司及模板材料為 proposed，尚未存在；不能把上游 2A 當確認輸入。

Output：共通 `forward/records.jsonl` 每 pair×方向×模板×arm×seed 一列，含 source/target clean margins、patched margin、raw/toward-source shift、概念 pair score 差、patch norm、scale、eligibility/reason；所有 finite 數值。analysis 同時列 pair-level 與分方向結果，ratio 未定義用 null。原始狀態僅 transient。

preflight：先 unit tests，再獲准用一組真實 pair 兩方向跑所有 arm、一筆 anonymous、全部分析與 finalize。驗證 reconstruction 數值、self-source bit-exact、hook 位置与 counts、live clean 對歷史參照；新模板無歷史數值不偽造重現要求。

任何 candidate hash 改變、漏 control、token span 不一致、稀少有效 pair、τ 超上限均中止或 inconclusive。CLI 未實作，未來專屬 Phase 2 subcommand 必須與此 schema 一致。prompt/direction/estimand/control/gate 改變依共通 version policy，不能事後調整確認結果。
