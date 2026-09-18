# J-space 正負價態詞彙 Readout：提名 12 個候選方向（Technology Discovery）

**狀態：已完成（Technology discovery 完成，表徵候選非因果證明）。** 模型為 Qwen3.5-4B（bf16）；2026-08-27 完成。操作契約與 aggregation 定義見 [實驗提案](proposal.md)。

**一句話發現：** 在 35 檔科技股的證據結尾位置，透過 Jacobian lens 讀出並提名了 12 個跨層與跨公司方向穩定的正負價態候選詞（負向如 `potential` $\Delta p = -0.0228$、`risk`；正向如 `justify` $\Delta p = +0.0059$、`upgrade`）；但後續下游因果篩選 shortlist 為空。

## 1. 能否在殘差流中讀出能區分正負證據的穩定詞彙表徵？提名 12 個候選詞彙

我們使用對齊好的 canonical Jacobian lens，在 35 檔科技股的 105 對平衡證據的結尾位置，逐層讀取完整的詞彙表 softmax 分佈：

**觀察：** 在 L14–L26 候選層中，提名出 12 個通過嚴格門檻（平均機率 $\ge 1\times 10^{-5}$、ticker 符號一致性 $\ge 70\%$、層一致性 $\ge 75\%$）的候選詞：

| 價態方向 | 候選 Token | Token ID | 層帶機率差值 ($\Delta p$) | 公司符號一致率 | 層符號一致率 |
|---|---|---:|---:|---:|---:|
| 負向 (negative) | ` potential` | 4499 | −0.022846 | 28/35 | 13/13 |
| 負向 (negative) | ` predicted` | 18569 | −0.014658 | 28/35 | 13/13 |
| 負向 (negative) | ` risks` | 14832 | −0.007027 | 34/35 | 13/13 |
| 負向 (negative) | ` downgrade` | 87250 | −0.006880 | 35/35 | 13/13 |
| 負向 (negative) | ` impacts` | 24115 | −0.006676 | 33/35 | 13/13 |
| 負向 (negative) | ` risk` | 5048 | −0.005969 | 34/35 | 13/13 |
| 正向 (positive) | ` justify` | 9079 | +0.005930 | 25/35 | 13/13 |
| 正向 (positive) | ` Industry` | 23094 | +0.002811 | 26/35 | 13/13 |
| 正向 (positive) | ` upgrade` | 13511 | +0.001839 | 32/35 | 13/13 |
| 正向 (positive) | ` increase` | 5096 | +0.001660 | 28/35 | 13/13 |
| 正向 (positive) | ` justified` | 33273 | +0.001410 | 26/35 | 11/13 |
| 正向 (positive) | ` partnership` | 14859 | +0.001279 | 33/35 | 13/13 |

**解讀：** 這些詞彙反映了模型在處理正負財務論據時在特定位置激活的表徵偏向，具備統計上的方向一致性。

## 2. 詞彙讀出是否直接構成因果決策的證明？否，僅屬表徵提名

**觀察與方法限制：**
1. **透視讀出非真實生成**：Jacobian lens 是將殘差流向量經線性映射回詞表的投影分數，它反映的是表徵向量的幾何走向，不等於模型在真實自回歸輸出端的下一個詞機率。
2. **位置敏感性**：早期在 final position 讀取的診斷 run 幾乎全被 JSON 結構標記主導；改在 evidence item end position 讀取才分離出語義詞彙。

## 3. 提名的 12 個候選在下游因果測試中表現為何？Shortlist 為空收線

**後續驗證：** 這 12 個候選詞彙方向直接傳遞至 [Token Causal Screen V1](../jspace-token-experiments/report.md) 進行小劑量 steering 介入測試。實測結果顯示：840 次干預僅產生 1 次固定選擇符號翻轉，無任何詞彙通過預先凍結的一致性門檻，Shortlist 為空收線。詞彙讀出方向無法因果性操縱決策。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| 原始提案與 aggregation 契約 | [實驗提案](proposal.md)（V1 discovery）。 |
| Discovery 執行記錄與產物 | run `valence-technology-evidence-balanced-20260827T043734Z`，位於 `artifacts/qwen3.5-4b/jspace-valence-readout/runs/`；35 科技股，105 對提示詞。 |
| 下游因果驗證報告 | [J-space token 實驗報告](../jspace-token-experiments/report.md)；[V1 詳細報告](../jspace-token-experiments/details/report-v1.md)。 |

**本次編輯說明：** 本報告按三項核心問題改寫，明確區分表徵讀出提名與下游因果驗證結果；原始協議 `proposal.md` 完整保留。
