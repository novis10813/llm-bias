# Phase 1 V2：64 句材料待人工審查

**review_status：pending；來源：AI 合成初稿；無模型讀出、無 audit、非可載入 JSON。**

依[V2 材料協議](proposal-phase1-v2.md)。這是實驗材料的 repository 來源，不是任何真公司的財務資料。正負為概念方向，不是好壞或 buy/sell。

## 1. ID、分割與比較規則

- `C`：收入依賴少數客戶的程度，正極較集中。`S`：收入隨景氣變動的程度，正極較敏感。
- `F01–F06`：擬 fitting；`V01–V04`：擬 validation；`E01–E02`：開發用一般評價交叉；`L01`：詞面／數值對照；`X01`：競爭內容對照。每個完整前綴（如 `C-F01`）是一個暫定家族。
- 主要句子的 `P/N` 是預期概念標籤。交叉句子 `PH/PL/NH/NL` 的首字母是概念正負，後字母是正面／負面評價。對照 `A/B` 不具有目標概念正負標籤。
- 每個 F/V 家族比較 P−N。每個 E 家族比较 PH−NH 與 PL−NL（固定評價的概念差），以及 PH−PL 與 NH−NL（固定概念的評價差）。每個 L/X 家族比较 A−B，只衡量非目標變化的反應，不歸入概念分類 accuracy。
- F/V 家族均由同一作者起草，分割是待審分配，不是已證明的句型獨立性。人工須合併近義改寫群；合併後不足預算要重寫，不能保留 ID 假裝獨立。
- 每句是完整描述欄位；部分欄位含兩個文法句子。「64句」在本文件指64筆描述，不是句號數。尚未加入固定 instruction 或 tokenizer chat formatting。

## 2. C：只辨識客戶收入分布，不辨識公司大小

定義：收入是否由少數客戶占去大部分。排除總收入、員工數、客戶付款可靠性、合約期限、增長及一般評價。

### 擬 fitting：六個家族，12筆

| ID | 英文描述 |
|---|---|
| C-F01-P | The company's three largest customers account for 75% of annual revenue. |
| C-F01-N | The company's three largest customers account for 15% of annual revenue. |
| C-F02-P | Most of the company's revenue comes from a small group of customers. |
| C-F02-N | Most of the company's revenue is spread across a broad group of customers. |
| C-F03-P | Among the company's 100 customers, payments are distributed very unevenly, with a few customers providing nearly all revenue. |
| C-F03-N | Among the company's 100 customers, payments are distributed very evenly, with each customer providing roughly the same revenue. |
| C-F04-P | The company's annual revenue is $100 million; $80 million comes from one customer and the remainder comes from 99 others. |
| C-F04-N | The company's annual revenue is $100 million; $1 million comes from each of its 100 customers. |
| C-F05-P | Replacing the revenue from the company's two largest customers would require replacing most of its sales. |
| C-F05-N | Replacing the revenue from the company's two largest customers would require replacing only a small part of its sales. |
| C-F06-P | The company's customer revenue chart has a few dominant bars and many much smaller bars. |
| C-F06-N | The company's customer revenue chart has many bars of approximately equal height. |

家族審查重點：F01 百分比大小；F02 small/broad；F03 evenly 與否定方向；F04 數字推理；F05 替換情境可能喚起損失而非分布；F06 圖形描述可能只是大小辨識。F03/F04/F06 表達同一分布，需特別審查是否只是同義改寫群。

### 擬 validation：四個家族，8筆

| ID | 英文描述 |
|---|---|
| C-V01-P | On a customer-by-customer revenue ledger, the first two accounts reach half of the total when accounts are accumulated from largest to smallest. |
| C-V01-N | On a customer-by-customer revenue ledger, the first fifty accounts reach half of the total when accounts are accumulated from largest to smallest. |
| C-V02-P | Two dollars selected independently from the company's revenue are likely to have been paid by the same customer. |
| C-V02-N | Two dollars selected independently from the company's revenue are unlikely to have been paid by the same customer. |
| C-V03-P | A sales manager responsible for just one of the company's customers oversees most of the company's revenue. |
| C-V03-N | A sales manager responsible for just one of the company's customers oversees only a small fraction of the company's revenue, regardless of which customer is assigned. |
| C-V04-P | The effective number of equally contributing customers that would reproduce this company's revenue distribution is low. |
| C-V04-N | The effective number of equally contributing customers that would reproduce this company's revenue distribution is high. |

家族審查重點：V01 累積分布與 F01 的相關性；V02 用抽樣表達分布，可能不自然或難懂；V03 與 F04/F05 可能需合併；V04 使用抽象定義，需確認人審不靠專業術語猜測。V02 的「likely」無數值界，不能充當已校準分類門檻。V04 只作文字概念測試，不宣稱已計算任何真實統計量。

### 一般評價交叉：兩家族，8筆

| ID | 英文描述 |
|---|---|
| C-E01-PH | A handful of customers provides almost all of the company's revenue. An observer describes the overall business outlook as favorable. |
| C-E01-PL | A handful of customers provides almost all of the company's revenue. An observer describes the overall business outlook as unfavorable. |
| C-E01-NH | A wide range of customers provides similar portions of the company's revenue. An observer describes the overall business outlook as favorable. |
| C-E01-NL | A wide range of customers provides similar portions of the company's revenue. An observer describes the overall business outlook as unfavorable. |
| C-E02-PH | Nearly all customer receipts are tied to a small number of accounts. A separate commentary praises the company's overall situation. |
| C-E02-PL | Nearly all customer receipts are tied to a small number of accounts. A separate commentary criticizes the company's overall situation. |
| C-E02-NH | Customer receipts are divided into similar amounts across many accounts. A separate commentary praises the company's overall situation. |
| C-E02-NL | Customer receipts are divided into similar amounts across many accounts. A separate commentary criticizes the company's overall situation. |

E01/E02 不聲稱觀察者正確，不增加違約、資產品質等事實。模型仍可能從評價推測潛在公司狀況。兩家族共同測評價干擾，但獨立性待審；不要將八筆當八個獨立樣本。

### 詞面／數值與競爭內容：兩家族，4筆

| ID | 英文描述 |
|---|---|
| C-L01-A | The company's customer contract renewal rate is 75%; the distribution of revenue across customers is not specified. |
| C-L01-B | The company's customer contract renewal rate is 15%; the distribution of revenue across customers is not specified. |
| C-X01-A | The company employs 20,000 people; the distribution of revenue across customers is not specified. |
| C-X01-B | The company employs 200 people; the distribution of revenue across customers is not specified. |

L01 重用 F01 的百分比但改成續約率，亦可能引入好壞評價，故反應不能單獨歸因於數字。X01 檢查公司大小，但員工數不充分定義所有規模含義。不指定集中度不是集中度相等的自然世界證明；僅測無明示目標資訊時是否仍有反應。

## 3. S：辨識跨景氣變動，不辨識單季好壞

定義：在比較景氣擴張與收縮時，公司收入是否有顯著同向變動。正極較敏感，負極較穩定；本批不處理反景氣公司。排除當期增長、長期平均獲利、單次事故及一般評價。

### 擬 fitting：六個家族，12筆

| ID | 英文描述 |
|---|---|
| S-F01-P | Across past economic expansions and contractions, the company's revenue rose and fell substantially. |
| S-F01-N | Across past economic expansions and contractions, the company's revenue remained broadly stable. |
| S-F02-P | The difference between the company's revenue in economic upturns and downturns has been large. |
| S-F02-N | The difference between the company's revenue in economic upturns and downturns has been small. |
| S-F03-P | In earlier recessions, company revenue fell sharply and subsequently rose sharply during economic recoveries. |
| S-F03-N | In earlier recessions, company revenue changed little and subsequently changed little during economic recoveries. |
| S-F04-P | Holding company-specific conditions constant, a change in the overall economy would substantially change the company's revenue in the same direction. |
| S-F04-N | Holding company-specific conditions constant, a change in the overall economy would barely change the company's revenue. |
| S-F05-P | Over several business cycles, the company's indexed revenue ranged from 70 in contractions to 130 in expansions. |
| S-F05-N | Over several business cycles, the company's indexed revenue ranged from 98 in contractions to 102 in expansions. |
| S-F06-P | On a chart of several economic cycles, the company's revenue curve has pronounced peaks in expansions and troughs in contractions. |
| S-F06-N | On a chart of several economic cycles, the company's revenue curve stays nearly level through expansions and contractions. |

家族審查重點：F01/F02/F03/F06 高度相關，不能憑時序／圖形換詞就宣稱獨立；F04 是明示假設而非歷史觀察；F05 可能只測範圍大小或算術。每對都提及擴張與收縮，避免正極只含衰退。

### 擬 validation：四個家族，8筆

| ID | 英文描述 |
|---|---|
| S-V01-P | Knowing whether the economy was expanding or contracting would substantially improve a forecast of this company's revenue, with higher revenue expected during expansion. |
| S-V01-N | Knowing whether the economy was expanding or contracting would barely improve a forecast of this company's revenue, with similar revenue expected in either phase. |
| S-V02-P | In a comparison of otherwise similar years, years with stronger aggregate economic activity consistently had much higher company revenue. |
| S-V02-N | In a comparison of otherwise similar years, years with stronger aggregate economic activity had approximately the same company revenue. |
| S-V03-P | When estimating the company's revenue, the analyst uses markedly higher estimates for economy-wide expansion scenarios than for contraction scenarios. |
| S-V03-N | When estimating the company's revenue, the analyst uses approximately equal estimates for economy-wide expansion and contraction scenarios. |
| S-V04-P | After removing the company's long-term revenue trend, the remaining revenue rises substantially in economic expansions and falls substantially in contractions. |
| S-V04-N | After removing the company's long-term revenue trend, the remaining revenue movements are small through expansions and contractions in the economy. |

家族審查重點：V01 預測資訊與 V03 情境估計可能同群；V02/F01 的回顧性比較可能同群；V04 需確認「去除趨勢」可讀性；AI 審查後正極已補明擴張時大幅上升、收縮時大幅下降，避免只以相關性代表變動幅度。負極定義為穩定，不代表所有低敏感公司都不受其他因素影響。

### 一般評價交叉：兩家族，8筆

| ID | 英文描述 |
|---|---|
| S-E01-PH | Company revenue changes greatly as the economy moves between expansion and contraction. An observer describes the overall business outlook as favorable. |
| S-E01-PL | Company revenue changes greatly as the economy moves between expansion and contraction. An observer describes the overall business outlook as unfavorable. |
| S-E01-NH | Company revenue changes little as the economy moves between expansion and contraction. An observer describes the overall business outlook as favorable. |
| S-E01-NL | Company revenue changes little as the economy moves between expansion and contraction. An observer describes the overall business outlook as unfavorable. |
| S-E02-PH | The firm's sales rise markedly in broad economic recoveries and fall markedly in broad economic slowdowns. A separate commentary praises the company's overall situation. |
| S-E02-PL | The firm's sales rise markedly in broad economic recoveries and fall markedly in broad economic slowdowns. A separate commentary criticizes the company's overall situation. |
| S-E02-NH | The firm's sales remain steady in both broad economic recoveries and broad economic slowdowns. A separate commentary praises the company's overall situation. |
| S-E02-NL | The firm's sales remain steady in both broad economic recoveries and broad economic slowdowns. A separate commentary criticizes the company's overall situation. |

沿用相同評價句是為了跨候選比較同一干擾，不代表獨立評價證據。兩候選 E01/E02 都是開發對照，不跨入 audit。

### 詞面／數值與競爭內容：兩家族，4筆

| ID | 英文描述 |
|---|---|
| S-L01-A | The difference between the company's two office floor areas is large; no history of revenue across economic cycles is provided. |
| S-L01-B | The difference between the company's two office floor areas is small; no history of revenue across economic cycles is provided. |
| S-X01-A | Company revenue increased by 10% in the latest quarter; no history of revenue across economic cycles is provided. |
| S-X01-B | Company revenue decreased by 10% in the latest quarter; no history of revenue across economic cycles is provided. |

L01 固定公司背景，檢查 large/small 捷徑；無法單憑它排除所有數字線索。X01 檢查當期好壞與景氣敏感性的混淆；单季增減不足以推定敏感性。兩者未指定目標屬性，不可標成低敏感性負例。

## 4. 比較清單與人工審核欄位

共有28個暫定家族：每候選10主要＋4對照。主要20對；交叉家族每個4個固定比較，共16個；其他對照4對，合計40個預先列定的開發比較。這不是40個獨立檢定，統計分析須另定。

審查者逐家族記錄：

1. `family_id` 與所有句子ID，擬用途是否正確。
2. 打散、不看配對標記時，目標屬性高／低／未指定、一般評價正／負／未指定。
3. 是否改變非目標事實；數字／否定／長度／語法是否足以猜標籤。
4. 應合併的近義家族，包括跨 F/V 的家族；不因既有預算拒絕合併。
5. `retain / revise / reject` 意見、理由、審查者及修訂版本；不預填人工通過。

**AI 預審後的分組限制**：下列關係必須先裁定，裁定前 F/V 標籤不生效、不可作凍結 split：C-F01↔C-V01（累積收入占比）、C-F04/F05↔C-V03（單一客戶收入依賴）、S-F04↔S-V01/V03（情境估計）、S-F01/F02/F03/F06↔S-V02（景氣與收入對照）。同一構念必然有語義重疊，但重疊本身不自動證明洩漏；人工需按共用骨架與衍生改寫來源分群。S-V01/V03 也可能是同一家族，不能把兩個 ID 當兩個獨立樣本。

若合併後家族不足，將本批降為未分割開發材料，保留句子與原擬用途但不 fitting／validation；另提改寫清單再審。不能為湊足數量改成另一構念，例如用訂單能見度冒充景氣敏感性。

**目前已知必須處理的風險**：C 的百分比／分布家族互相關聯；S 的時間序列／情境句互相關聯；C-V02/V04 與 S-V04 較抽象。預算湊足不表示可凍結。若剩餘独立家族不足，先報不足，再提出重寫／縮減或額外預算，不將 validation 改名 audit。

本文件不設定模型 gate，不存 vectors 或 states。後續若改寫句子，需同步比較關係、家族分組與製作紀錄；正式轉檔另定 SHA-256 與 schema，不將本稿當正式 inputs。
