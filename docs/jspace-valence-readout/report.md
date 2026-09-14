# J-space 正負價態詞彙 readout：Technology discovery 報告

本報告記錄 proposal 所定義 workflow 的 discovery 結果。方法、aggregation contract、
artifacts 與 CLI 見 [實驗提案](proposal.md)。

## Technology discovery 結果

正式 discovery run：

`artifacts/qwen3.5-4b/jspace-valence-readout/runs/valence-technology-evidence-balanced-20260827T043734Z`

規模為 35 個 Technology discovery tickers、每 ticker 3 個 matched source trials、
105 pairs／210 prompts。所有 stages 完成，readout 使用 L14–L26 與 final layer。

Band L14–L26 的 frozen representation candidates：

| Readout side | Token | Token ID | Band probability diff | Ticker sign consistency | Layer sign consistency |
|---|---|---:|---:|---:|---:|
| negative | ` potential` | 4499 | −0.022846 | 28/35 | 13/13 |
| negative | ` predicted` | 18569 | −0.014658 | 28/35 | 13/13 |
| negative | ` risks` | 14832 | −0.007027 | 34/35 | 13/13 |
| negative | ` downgrade` | 87250 | −0.006880 | 35/35 | 13/13 |
| negative | ` impacts` | 24115 | −0.006676 | 33/35 | 13/13 |
| negative | ` risk` | 5048 | −0.005969 | 34/35 | 13/13 |
| positive | ` justify` | 9079 | +0.005930 | 25/35 | 13/13 |
| positive | ` Industry` | 23094 | +0.002811 | 26/35 | 13/13 |
| positive | ` upgrade` | 13511 | +0.001839 | 32/35 | 13/13 |
| positive | ` increase` | 5096 | +0.001660 | 28/35 | 13/13 |
| positive | ` justified` | 33273 | +0.001410 | 26/35 | 11/13 |
| positive | ` partnership` | 14859 | +0.001279 | 33/35 | 13/13 |

`potential`、`predicted`、`justify`、`Industry` 等詞可能反映語句模板或一般預測語彙；
它們和 `risk`、`downgrade`、`upgrade`、`increase`、`partnership` 一樣，都只保留為
後續 causal screen 的候選，不依文字語義先行刪除。

這些 candidates 已由
[token causal screen V1](../jspace-token-experiments/details/proposal-v1.md) 的 `run-token-screen`
以 SHA-bound config 直接消費，並在 discovery split 完成對稱劑量 screen；V1 shortlist
為空。後續 V2 不再由本 readout 提名詞彙方向，而改 fitting outcome-gradient
axis。兩版差異見 [J-space token experiment versions](../jspace-token-experiments/details/README.md)。

一個較早的 diagnostic run 在 final prompt position 讀取，結果幾乎全由 JSON opening
與格式 tokens 主導，沒有產生 eligible candidates：

`valence-technology-discovery-20260827T043215Z`

因此 schema v2 將 primary readout 改為兩個 evidence-item end positions。這個變更在
看到 semantic candidates 前完成，後續 intervention 只使用 schema v2 candidate artifact。

## 限制

- Readout 只取兩個 evidence item 的 end positions；不做完整 token-position scan
  （那是 baseline lens-forward 的工作）。
- Sign consistency 是必要非充分條件：70% ticker、75% layer 與
  leave-one-ticker-out 門檻只代表 readout 方向有基本穩定性，不保證介入後行為一致；
  候選 token 仍需在 calibration/test split
  上做 signed steering/gain/swap 的 causal 驗證。
- Discovery split 只用於候選提名；calibration/test tickers 不進入本 run
  （split manifest 決定）。
- `frozen_candidate_suggestions` 上限 12、contrast top 50、min mean
  probability 1e-5 都是文件化門檻，改動需要更新本文。
