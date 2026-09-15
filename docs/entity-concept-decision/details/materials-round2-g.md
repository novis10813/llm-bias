# Round 2 材料：地理多元度（G，stance 中性公司屬性）

**review_status：pending，AI 合成初稿，無模型讀出，非可載入 JSON。**

用於決定性測試：一個 stance 中性的公司屬性是否能在固定 L15 k=8 內被乾淨讀出（與通用 stance 正交、勝 random）、並把公司分開。正極 = 高地理多元（多國營運），負極 = 低地理多元（單一市場）。刻意選一個模型不太會有風險／好壞評價的規模事實，以對照天生帶風險評價的「客戶集中度」。

## 格式

每列：`ID | concept | group | role | pole | 英文描述`。
- concept = `G`。
- group：`G1`（國數表述）、`G2`（市場廣度）、`G3`（收入地域）。G1/G2/G3 互為改寫群，供 leave-one-group-out。
- role：`primary`（G1/G2/G3）、`evaluation`（G-E01/G-E02，2×2）、`lexical`（G-L01，數值捷徑）、`competitor`（G-X01，員工數）。
- pole：primary 用 P/N；evaluation 用 PH/PL/NH/NL（首字=概念正負，次字=評價好壞）；lexical/competitor 用 A/B（無概念正負標籤）。

## 行

| ID | concept | group | role | pole | 英文描述 |
|---|---|---|---|---|---|
| G-F01-P | G | G1 | primary | P | The company operates in 40 different countries around the world. |
| G-F01-N | G | G1 | primary | N | The company operates in a single country. |
| G-F02-P | G | G2 | primary | P | The company's customers are spread across many national markets on several continents. |
| G-F02-N | G | G2 | primary | N | The company's customers are located in one national market. |
| G-F03-P | G | G3 | primary | P | The company generates revenue in dozens of countries across five continents. |
| G-F03-N | G | G3 | primary | N | The company generates all of its revenue within one region. |
| G-E01-PH | G | G-E01 | evaluation | PH | The company operates in 40 different countries around the world. An observer describes the overall business outlook as favorable. |
| G-E01-PL | G | G-E01 | evaluation | PL | The company operates in 40 different countries around the world. An observer describes the overall business outlook as unfavorable. |
| G-E01-NH | G | G-E01 | evaluation | NH | The company operates in a single country. An observer describes the overall business outlook as favorable. |
| G-E01-NL | G | G-E01 | evaluation | NL | The company operates in a single country. An observer describes the overall business outlook as unfavorable. |
| G-E02-PH | G | G-E02 | evaluation | PH | The company's customers are spread across many national markets on several continents. A separate commentary praises the company's overall situation. |
| G-E02-PL | G | G-E02 | evaluation | PL | The company's customers are spread across many national markets on several continents. A separate commentary criticizes the company's overall situation. |
| G-E02-NH | G | G-E02 | evaluation | NH | The company's customers are located in one national market. A separate commentary praises the company's overall situation. |
| G-E02-NL | G | G-E02 | evaluation | NL | The company's customers are located in one national market. A separate commentary criticizes the company's overall situation. |
| G-L01-A | G | G-L01 | lexical | A | The company has offices in 40 cities; the number of countries in which it operates is not specified. |
| G-L01-B | G | G-L01 | lexical | B | The company has offices in 1 city; the number of countries in which it operates is not specified. |
| G-X01-A | G | G-X01 | competitor | A | The company employs 50,000 people; the number of countries in which it operates is not specified. |
| G-X01-B | G | G-X01 | competitor | B | The company employs 500 people; the number of countries in which it operates is not specified. |

## 比較

- 每個 primary 家族（G1/G2/G3）：`<family>-primary`，比較 P−N。
- 每個 evaluation 家族（G-E01/G-E02）：
  - `<family>-concept_at_positive_evaluation`：PH−NH
  - `<family>-concept_at_negative_evaluation`：PL−NL
  - `<family>-evaluation_at_positive_concept`：PH−PL
  - `<family>-evaluation_at_negative_concept`：NH−NL
- lexical（G-L01）：`G-L01-lexical`，A−B。
- competitor（G-X01）：`G-X01-competitor`，A−B。

## 四家公司的真實地理多元（用於解讀，不進入材料）

IT（IBM）、BDX（Becton Dickinson）、BLK（BlackRock）皆為全球多國營運（高）；NSC（Norfolk Southern）為美國鐵路，實質單一國家（低）。若 G 方向把 NSC 與其餘三家分開，才是公司相關訊號。
