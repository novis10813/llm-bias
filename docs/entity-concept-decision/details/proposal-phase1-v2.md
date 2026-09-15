# Phase 1 V2：以具體公司屬性材料檢驗候選方向

**狀態：材料設計與 64 句初稿製作已批准；材料待獨立人工審查，non-runnable，模型執行未授權。**本版獨立記錄材料、control family 與 split/freeze sequence 的調整；不回寫 [Phase 1 原始 Draft 1](proposal-phase1.md) 的預算或 gate。本版已實作並跑過一次 development model_smoke（非正式 audit）：[開發 smoke 紀錄](report-phase1-v2-development-smoke.md)。正式協議與 gate 另定。

## 1. 假說與方法邊界

沿用原始協議 H1：固定 L15 k=8 中，至少一個方向可辨識事先定義的概念差異，且非只辨識詞面、公司名或一般評價。方法來源、TCAV 與 activation patching 的適應性差異沿用[共通契約 §1](design-and-validation.md#1-方法來源與適應性修改)。成對描述的平均狀態差並非 TCAV 演算法復現；自然公司狀態是否喚起同一概念仍需獨立量測。

主要候選：收入依賴少數客戶的程度（正極較集中），以及收入隨景氣變動的程度（正極較敏感）。符號不表示買入偏好。暫留第三名額；不按模型結果補選。

選擇理由：兩屬性可具體定義、可建立競爭解釋對照；比「好公司」或未拆解的「財務穩健」更容易查核概念特異性。沒有證據保證它們位於既定 k=8。

備選包括短期支付能力、持續合約收入、產品／市場收入分布；公司規模與業務類別先作競爭解釋。audit 後換候選須另立確認材料，不能反覆使用原 audit。

## 2. 輸入、輸出與批准範圍

固定 upstream hashes、Q shape、L15 定義、數值與 compact outputs 沿用[共通契約 §2、4、5](design-and-validation.md)。不修改 upstream、basis、模型或 lens。

本次唯一新增材料來源是[64 句待審稿](materials-phase1-v2-draft.md)，Markdown 為人審來源，不是可載入的 `concept_materials.json`。句子均由本次 AI 協作起草，屬合成描述，不宣稱真公司事實。未經獨立人工審核，不能轉為 `review_status="approved"`。

每候選 6 fitting 家族×2句、4 validation 家族×2句、2 一般評價交叉家族×4句、2 其他對照家族×2句，合計32句；兩候選64句。全數為開發材料，無 audit。家族是待人工確認的分組，數量不是統計獨立性證明。

既有 reader 要求每概念含 fit/validation/audit，且無完整對照比較結構；因此本稿刻意不轉成該 schema、不造 audit、不修改 reader。後續 spec 必須明定對照角色、比較 ID、coverage、人工審查與凍結格式。

英文描述不含真公司名或投資建議；未來完整 prompt 保留既有 instruction suffix 與答案 prefix `{"decision": "`。只有片段已撰寫，完整 prompt、token partition 與 span 尚未驗證。CLI／runner 尚未實作，不提供執行命令。

## 3. 對照與方向規則

只以 fit 家族目標屬性差進行既有等家族權重 fitting；validation、audit、所有對照不得流入方向計算。不得用 margin 翻號或選材料。

每候選兩個四句家族，交叉屬性高／低與評價好／壞：相同評價比較屬性、相同屬性比較評價。另測詞面／數值與競爭內容。這些開發對照只用來識別問題，不能冒充獨立確認。

模型可能從評價句推測未明說的事實；交叉對照不是完美隔離。人審需記錄其合理性與歧義，不把「句子沒改」當作內部概念不變的證據。

檢查家族差異一致性及投影前後保留程度；若平均抵消、低保留或對照失敗，分開報告。單方向失敗不等同概念不存在；禁止自動改成多維或重新 SVD。

## 4. Audit 預算與尚未凍結的 gate

原始6/3/3家族預算僅能作可行性探索。本版不執行三家族 audit。若以 n 個獨立家族作精確成對標籤交換，排列數為2^n；n=3時單尾最小p=0.125。三候選 Holm 第一比較要求p<0.05/3，單尾至少n=6才有數學可能，常見對稱雙尾至少n=7。解析度不是功效保證。

暫預留每候選12個獨立主要 audit 家族，另需獨立對照；最終數目待統計量、單／雙尾、候選數、對照比較與家族相依性固定後，以合成效果情境評估。不得以增加 permutation 抽樣次數或近義句數代替增加独立家族。

原始協議的語義差、Holm、random 與混淆對照要求仍是待具體化目標，非已凍結可執行 gates；δ_a、balanced accuracy、τ_a,max、random seeds、對照界及 audit 數目均為 blockers。不從舊結果杜撰門檻。

## 5. 邊界、停止與 preflight

沿用原始協議與共通契約的 zero norm／finite／hash／位置拒絕、候選空集合 finalize、compact artifact、no raw states 政策。缺對照、人工未審或 span 不對齊不得執行；僅能辨識一般評價者不能命名為特定公司內容。

後續 runner 先完成 deterministic tests，再另批准真實 tokenizer/model 的完整公司 pair 雙向與概念 pair smoke，涵蓋 controls、analyze、finalize；不是本次授權。Phase 2/3 的決策因果與上游恢復亦未授權。

## 6. 版本與狀態紀錄

- 原始 Draft 1：最多三概念，建議6/3/3家族；未有正式材料或 run。
- V2：依已批准材料決策，先選兩概念，6 fit／4 validation，另設交叉及其他對照；audit 另定。新增版本避免混用舊控制與分割。
- [V2 製作紀錄](report-phase1-v2.md)只記文件與驗證狀態，不含研究效果。
- 後續 prompt family、direction source、estimand、controls 或 gate 實質變更依[版本政策](../../documentation-system.md#experiment-versioning)另立版本；禁止以審閱本稿冒充模型執行批准。
