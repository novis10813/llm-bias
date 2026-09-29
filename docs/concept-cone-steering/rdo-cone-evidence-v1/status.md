# rdo-cone-evidence-v1：狀態

**狀態：**已完成（2026-09-29）。協議見 [proposal.md](proposal.md)。**性質：**探索性，Qwen3.5-4B，單一訓練 seed，只有 sell→buy；使用 [rdo-cone-v1](../rdo-cone-v1/status.md) 已存的方向（`directions.json` SHA-256 `1379e332…7f70e`）。

執行方式：為了平行，每個證據條件各一個 lab job（`idlab2`，commit `5620769`），run id 分別為 `rdo-cone-v1-evidence-20260929-full-01`（neg，`rdo-9`）、`-02`（mixed2，`rdo-10`）、`-03`（zero，`rdo-11`）。三份 `result.json` 都是 `complete=true`，在本機 `artifacts/qwen3.5-4b/concept-cone-steering/runs/` 下。這等於協議中的單一 run 拆成三份，內容與協議相同。

## 結果（101 家評估公司，sell→buy 的 ITT flip 率／parse 率，單位為 DIM 範數的倍數）

| 條件（α0） | 方向 | α=1 | α=2 | α=4 | α=8 | α=16 | α=32 | 翻一半 | 崩壞 |
|---|---|---|---|---|---|---|---|---|---|
| neg（101 sell） | `dim` | 0.00 | 0.00 | 0.00 | 0.87 | 1.00 | 0.00／0.00 | 8 | 32 |
| | `rdo1` | 0.00 | 0.06 | 1.00 | 1.00 | 1.00 | 1.00 | 4 | 無 |
| | `rco_b4` | 0.00 | 0.00 | 0.07 | 1.00 | 1.00 | 1.00 | 8 | 無 |
| mixed2（98 sell／3 buy） | `dim` | 0.20 | 0.88 | 1.00 | 1.00 | 1.00 | 0.94／0.94 | 2 | 無 |
| | `rdo1` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1 | 無 |
| | `rco_b4` | 0.23 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 2 | 無 |
| zero（96 sell／5 buy） | `dim` | 0.09 | 0.29 | 0.80 | 1.00 | 0.90／0.90 | 0.00／0.00 | 4 | 32 |
| | `rdo1` | 0.47 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 2 | 無 |
| | `rco_b4` | 0.16 | 0.51 | 1.00 | 1.00 | 0.98／0.98 | 0.01／0.01 | 2 | 32 |

未標 parse 率的格子 parse 率都是 1.00。`dim` 在 neg 的翻一半劑量 8，與 confirmation-v1 相同（該 run 中 neg 為 0.88 @ α=8）。balanced（[rdo-cone-v1](../rdo-cone-v1/status.md)）的翻一半劑量：`dim` 4、`rdo1` 1、`rco_b4` 4。

## 預先登記問題的判讀

- **E1（負面證據下的反應）：**`rdo1` 在 neg 的翻一半劑量 4，除以 balanced 的 1，比值為 4（≥2，記為「證據會延後翻轉」）；且 balanced 的 1 是劑量網格的最低點（α=1 已 0.84），真實比值 ≥ 4。對照：`dim` 為 8/4 = 2，`rco_b4` 為 8/4 = 2。`rdo1` 不是不看證據的開關：α=2 時 neg 只翻 6%，balanced 是 100%。
- **E2（沒有證據）：**`rdo1` 在 zero 的翻一半劑量 2（balanced 為 1）；`dim` 4（與 balanced 相同）；`rco_b4` 2（balanced 為 4）。mixed2 下 `rdo1` 在 α=1 已全翻。僅描述。
- **E3（格式與崩壞）：**`dim` 在 neg 與 zero 的 α=32 崩壞（parse 率 0.00），`rco_b4` 在 zero 的 α=32 崩壞（0.01）；`rdo1` 在三個條件、α 測到 32，parse 率一直是 1.00。沒有測 α>32，所以不能說 `rdo1` 不會崩壞。
- **E4（理由文字）：**讀了 ABNB、AEP 在 neg 下的列。翻成 buy 的理由都承認負面證據並仍主張 buy（例如「Despite the negative margin compression and revenue guidance cut, AEP is a defensive utility…」），`dim@8`、`rdo1@4`、`rco_b4@8` 的論述與措辭幾乎一致；`rdo1@2` 未翻的列（AEP）維持 sell 並引用負面證據。沒有量化，也沒有系統性抽樣。

## 解讀界線

- 「延後翻轉」只表示需要較高劑量，不等於對證據做了推理；`reason` 不是 evidence faithfulness 的證據（C11 仍未做）。
- `rdo1` 在 α≤32 沒有崩壞，是單一方向在單一模型上的觀察，不能說成 cone 較不易飽和。
- 單一訓練 seed，只有 Qwen 與一個符號；沒有 buy→sell 分母。未改動 claim ledger。
