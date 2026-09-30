# layer-check-v1：GPT-OSS-20B 換注入層，DIM 翻轉決策的能力是否改變

**狀態：**事前協議，2026-09-30，任何 run 之前寫定。**性質：**探索性，只有 GPT-OSS-20B，不改 confirmation-v1 的注入層與任何結果。**上層：**[confirmation-v1](../confirmation-v1/proposal.md)、[c2-v3-steering-prompt](../c2-v3-steering-prompt/status.md)。

## 為什麼做

GPT-OSS 的注入層 L14 是 427 家 v2 研究在 `medium` reasoning、固定前綴 readout 下選出的；confirmation-v1 起固定不重選。之後在 steering prompt、生成路徑 readout 下做的 v3 patching，peak 在 L8、band 為 L1–8，不含 L14。DIM 跨層掃描 V2 對 GPT-OSS 只跑過 L0、L12–L16、L23，從未在 L8 或 L1–8 注入過。所以目前無法回答「patching 最有效的層」與「加方向最有效的層」在 GPT-OSS 上是否一致。

## 設計

其他一律沿用 confirmation-v1：同 split（seed `20260923`，402 建方向／101 受測）、同 balanced prompt、同 K（99）、`low` reasoning、complete-object 解析、ITT 分母、由 L14 校準出的同一劑量網格。唯一改變是注入層。

- 層：**L4、L8、L14**。L8 是 v3 peak；L4 是 band 內部；L14 是同 run 內的重現，其生成須與 confirmation-v1 `dim` arm 逐字比對。
- 方向：每層以該層乾淨殘差重新計算原始 Top10−Bottom10 差（與 confirmation-v1 相同的構造與排序）。
- 劑量：α 以各層自己的 DIM norm 為單位，網格用 L14 校準的 `full_grid`，不重新校準；若某層在較小劑量就崩壞，如實報告 parse rate。
- 執行：`scripts/probe_steering_confirmation.py --arms gates ranking alpha0 cal dim_layers --layers 4 8 14`，run id `confirmation-v1-<date>-full-layer-01`；程式改動只有 `dim_layers` arm 接受 `--layers`。

## 指標與事前判讀

每層每方向報：ITT on-target flip rate 曲線、達 50% 與 100% 的最小網格劑量、各劑量的 parse rate（bootstrap 沿用 confirmation-v1）。以 confirmation-v1 的 L14（sell→buy：50% 於 α=2、100% 於 α=8；buy→sell：50% 於 α=4、100% 於 α=8）為對照。

| 結果 | 判讀 |
|---|---|
| L8 兩個方向都在不大於 L14 的劑量達 50%，且能達 100% | L14 不是唯一有效層；GPT-OSS 的 steering 對這兩層不敏感。 |
| L8 的 50% 劑量大於 L14 的兩倍以上，或達不到 100% | 早期層的 patching 最有效，但加方向最有效的層在較深處；兩者不是同一件事。 |
| 其他 | 描述，不下結論。 |

L4 只作 band 內部的補充描述，不進判讀。

## 解讀界線

- 探索性、事後；不改 confirmation-v1 的注入層、結果與論文的主要數字。
- 只有 GPT-OSS、只有這個 prompt、只有 DIM；不延伸到 cone 或其他模型。
- 劑量網格由 L14 校準，其他層的曲線可能因此在高劑量端被截斷或過早崩壞。

## 輸出

`artifacts/gpt-oss-20b/concept-cone-steering/runs/confirmation-v1-<date>-full-layer-01/`（`dim_layers/result.json` 及共用的 ranking、alpha0、cal）；lab job log 拉回同一目錄。
