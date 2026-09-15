# Round 2 材料：客戶集中度（C，stance 平衡版）

**review_status：pending，AI 合成初稿，無模型讀出，非可載入 JSON。**

用途：測試「客戶集中度」能否在**控住明示 stance** 後與通用 stance 解耦。與 round-1 的 C 不同，這裡的 primary 成對句把正負兩極寫成**同等正面**（同樣的 favorable 結尾），只改集中度事實；另保留一個無明示 stance 的家族作對照。

正極 = 集中（少数客戶占大部分收入），負極 = 分散。

## 格式

每列：`ID | concept | group | role | pole | 英文描述`。
- concept = `C`。
- group：`C1`／`C2`（明示 stance 平衡，兩極皆 financially healthy）、`C3`（無明示 stance，對照）。
- role：`primary`（C1/C2/C3）、`evaluation`（C-E01，2×2）、`lexical`（C-L01）、`competitor`（C-X01）。
- pole：primary 用 P/N；evaluation 用 PH/PL/NH/NL；lexical/competitor 用 A/B。

## 行

| ID | concept | group | role | pole | 英文描述 |
|---|---|---|---|---|---|
| C-F01-P | C | C1 | primary | P | The company's three largest customers account for 75% of annual revenue. Analysts describe the company as financially healthy. |
| C-F01-N | C | C1 | primary | N | The company's three largest customers account for 15% of annual revenue. Analysts describe the company as financially healthy. |
| C-F02-P | C | C2 | primary | P | A small group of customers provides most of the company's revenue. Analysts describe the company as financially healthy. |
| C-F02-N | C | C2 | primary | N | A broad group of customers provides similar portions of the company's revenue. Analysts describe the company as financially healthy. |
| C-F03-P | C | C3 | primary | P | The company's three largest customers account for 75% of annual revenue. |
| C-F03-N | C | C3 | primary | N | The company's three largest customers account for 15% of annual revenue. |
| C-E01-PH | C | C-E01 | evaluation | PH | The company's three largest customers account for 75% of annual revenue. An observer describes the overall business outlook as favorable. |
| C-E01-PL | C | C-E01 | evaluation | PL | The company's three largest customers account for 75% of annual revenue. An observer describes the overall business outlook as unfavorable. |
| C-E01-NH | C | C-E01 | evaluation | NH | The company's three largest customers account for 15% of annual revenue. An observer describes the overall business outlook as favorable. |
| C-E01-NL | C | C-E01 | evaluation | NL | The company's three largest customers account for 15% of annual revenue. An observer describes the overall business outlook as unfavorable. |
| C-L01-A | C | C-L01 | lexical | A | The company's customer contract renewal rate is 75%; the distribution of revenue across customers is not specified. |
| C-L01-B | C | C-L01 | lexical | B | The company's customer contract renewal rate is 15%; the distribution of revenue across customers is not specified. |
| C-X01-A | C | C-X01 | competitor | A | The company employs 20,000 people; the distribution of revenue across customers is not specified. |
| C-X01-B | C | C-X01 | competitor | B | The company employs 200 people; the distribution of revenue across customers is not specified. |

## 比較

- 每個 primary 家族（C1/C2/C3）：`<family>-primary`，P−N。
- evaluation 家族（C-E01）：
  - `C-E01-concept_at_positive_evaluation`：PH−NH
  - `C-E01-concept_at_negative_evaluation`：PL−NL
  - `C-E01-evaluation_at_positive_concept`：PH−PL
  - `C-E01-evaluation_at_negative_concept`：NH−NL
- lexical（C-L01）：`C-L01-lexical`，A−B。
- competitor（C-X01）：`C-X01-competitor`，A−B。

## 判讀要點

- 若 stance 平衡有效：C 方向（fit 自 C1/C2 的 P−N）應與 stance 方向的餘弦明顯小於 round-1 的 −0.88，且正交化後概念反應不應塌掉。
- C3（無明示 stance）的 P−N 預期仍帶隱含 stance，用作對照。
- 四家公司在 C 方向上的分數若仍有可分差異（且與 sell 立場不完全同序），才支持集中度是獨立於 stance 的公司相關訊號。
