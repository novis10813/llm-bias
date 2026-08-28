# J-space 正負價態詞彙 readout（Technology positive vs negative）

## 文件狀態與科學邊界

本文件描述 `jspace-intervention run-valence-readout` workflow：對同一公司的
positive（buy qual + buy quant）與 negative（sell qual + sell quant）證據對，
在 qualitative 與 quantitative evidence item 的結尾 token 用 Jacobian lens
讀取 L14–L26 與 final layer 的 transported representation，比較完整詞彙 softmax，找出在正負證據條件下
機率差異一致、且跨 ticker／跨 layer 方向穩定的 token。

**這是 transported-representation evidence，不是 causal evidence。** 輸出只
用於 *提名* 後續 signed steering / gain / swap 實驗要作用的 token（representation
candidates）；它不證明任何 token 因果上驅動 buy/sell 偏好。Jacobian lens 是
transported representation readout，不是 chain-of-thought、discrete reasoning
path 或 standalone causal claim。

本 workflow 與先前的 Technology identity-header 文字實驗
（[Technology header-span sensitivity](technology-header-span-sensitivity.md)）
是**分開的**實驗線：header 實驗比較固定 Buy/Sell continuation margin 的
prompt 表面效果；本實驗比較 lens readout 的完整詞彙分布。兩者不互相替代。

## 輸入

- **Raw trial JSONL**：baseline trial plan（例：
  `../baseline/runs/paper-local-qwen36-27b/trial_plan.jsonl`），每行含
  `condition`、`evidence`（buy/sell × qual/quant 各一）、`ticker`、`name`、
  `sector`、`prompt`、`trial_key`、`trial_index`、`set_index` 等欄位。
- **Split manifest**：既有 `jspace_intervention_splits` JSON（含
  `assignments` 與 `input_sha256`）。
- **Model** 與 **validated canonical lens**（`artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt`；
  可傳 `--lens` 指定，否則用 canonical 路徑）。workflow 只消費既有 lens，
  不 fitting、不修改 canonical lens。

## Prepare：選擇與配對

1. 只接受 `condition == "attribute"` 且 evidence 恰好是 buy/sell × qual/quant
   各一項的 raw row；每個被選中的 row 都會驗證 ticker/name/sector/prompt/
   evidence/trial identity（`trial_key` 非空字串、`trial_index`/`set_index`
   整數）。其他 condition/sector 的 row 直接略過。
2. 依 requested sector（預設 `Technology`）與 requested split（預設
   `discovery`）過濾 ticker；對每個 ticker 以 stable hash
   `sha256("{seed}:{ticker}:{trial_key}")` 排序，取前
   `--trials-per-ticker`（預設 3）個 source trials。
3. 從每個 source row 的**結構化 evidence** 渲染 canonical 兩項 prompt
   （`canonical-valence-v1`）：相同的 header/instruction，positive 用
   buy qual + buy quant，negative 用 sell qual + sell quant。
   **不解析、不重用原四項 prompt 本體。**
4. 保留 source trial identity（`source_trial_key`、`source_trial_index`、
   `set_index`）與 evidence item hash（`sha256_json({side, kind, text})`）；
   marketcap、原始 prompt、row seed 等多餘 metadata 不保存。
5. **Preflight**：在建立 run 之前，對每對 prompt 套用 chat template 並檢查
   token 長度 ≤ `--max-seq-len`（預設 1024）；超限直接失敗、不建立 run。
6. Metadata 綁定三個 hash：raw input SHA-256、split manifest SHA-256、
   split manifest 內的原始輸入 hash（`input_sha256`）。

Canonical prompt 範式（positive 示例）：

```
Refer to the evidence below to make a final investment decision.
Stock Ticker: [AES]
Stock Name: [The AES Corporation]
--- Evidence ---
1. <buy qual text>
2. <buy quant text>
---
Respond with one valid JSON object containing only the keys "decision" (buy | sell) and "reason" (brief justification). Do not choose hold.
```

## Forward：evidence-item end positions 完整 softmax readout

- Prepare 會把每個 condition 的 qualitative 與 quantitative evidence char spans
  映射到 formatted prompt token spans。Forward 在兩個 evidence item 的結尾 token
  讀取 residual，再先於 prompt 內平均兩個完整 softmax。這避開 answer instruction
  與 JSON opening token 對 final prompt position 的 motor/format domination。
- 每個 pair 的兩個 condition prompt 組成一個 right-padded batch（batch size 2）。
- 讀取層為 `--layers`（預設 14..26）加上 **final model layer**（自動附加，
  `is_output: true`）。
- Non-final layer 以 lens Jacobian transport 到 final-layer basis；所有層
  一律以 model final norm + LM head unembed（與 baseline `lens-forward` 的
  final-normalize/unembed 一致），計算**完整詞彙 softmax**。
- 為數值穩定，完整 probability sums 在記憶體中以 float64 累積：
  - 每 condition/layer 總和（condition mean 的分子）
  - 每 ticker/condition/layer 總和（sign consistency 用）
- **只 persist 每 prompt 的 compact 記錄**：每層 top-k（預設 30）token
  id/text/probability、entropy（nats 與 normalized）、effective
  temperature（final-norm residual norm 的倒數）、identifiers、condition、
  sequence length。**永不 persist 完整分布、residual、activation 或其他
  tensor。**
- Lens Jacobian 放在 lm_head device（與 baseline 實作一致），避免每次
  readout 搬移 d×d 矩陣。

## Analyze：先平均完整 softmax，再選 top-k

**Full-softmax aggregation contract（強制）：** 先對每個 condition（以及每
ticker）的完整詞彙 softmax 向量取平均，再選 top-k。絕不用 per-prompt top-k
的 union 或平均 top-k 列表。`tests/test_jspace_valence_readout.py` 中有
regression test 證明 ranking 不是 average-top-k（per-prompt top-1 union 的
token 無法被排到平均後的 rank 1）。

- Scope：每個 readout layer（`layer_<L>`，含 final layer）與 band
  aggregate（`band_<first>-<last>`，跨非 final readout layers 的 condition
  mean 等權平均；預設 L14–L26）。
- 對每個 scope、每個 side（positive / negative），在 condition mean 上計算：
  - `probability_diff = mean_positive[t] − mean_negative[t]`
  - `smoothed_log_ratio = log((mean_positive[t] + ε) / (mean_negative[t] + ε))`，ε = 1e-12
  - `js_contribution[t] = 0.5·(p log(p/m) + q log(q/m))`，`m = (p+q)/2`（nats，per-token Jensen–Shannon contribution）
- 排序以 `probability_diff` 為主（positive 側 descending、negative 側
  ascending），每個 side 每個 scope persist **top 50**（`frozen_candidate_suggestions`
  與 contrast 表的依據）。
- `frozen_candidate_suggestions`（上限 12，先為 positive/negative 各保留最多 6 個，
  再以 band |probability_diff| 填補未用名額並排序）只從 band scope 的 top-50 中提名，須同時滿足：
  1. 經既有 `single_leading_space_token` 解析為**恰好一個完整 leading-space
     token**；
  2. 不是 `buy`/`sell`、special token、punctuation-only、或 canonical prompt
     的 format boilerplate 詞（`respond`、`json`、`decision`、`reason`、
     `hold` 等，見 `FORMAT_BOILERPLATE`）；
  3. 自己 condition 的 mean probability ≥ **1e-5**（文件化的低門檻，約為
     150k 詞彙 uniform baseline 的 1.5 倍）；
  4. **sign consistency**：先在 ticker 內平均 band layers，再要求至少 70%
     tickers 與 side 同號；另在 layer 內平均 tickers，要求至少 75% band layers
     同號；leave-one-ticker-out aggregate sign 也必須保持不變。
- 每個 candidate 標註
  `label: "transported-representation candidate for later signed steering/gain/swap; not causal evidence"`。

## Artifacts

Run root：`artifacts/<model-slug>/jspace-valence-readout/runs/<run-id>/`

```
manifest.json
prepare/
  valence_pairs.jsonl            # 每 source trial 一筆：雙 condition prompt、trial identity、evidence hashes
  metadata.json                  # raw/split/split-input SHA、sector、split、seed、trials_per_ticker、params
forward/
  valence_readout.jsonl          # 每 prompt 一筆：每層 top-k + entropy + effective temperature
  metadata.json                  # layers、band、final layer、aggregation contract
analyze/
  valence_token_contrast.jsonl   # 每 (scope, side, rank) 一筆：三種分數與 condition means
  frozen_candidate_suggestions.json
  metadata.json                  # scopes、condition counts、eps、門檻、candidate label
```

Manifest 分 `prepare`/`forward`/`analyze` 三個 stage 註冊，finalize 要求三者
全部 complete。

## CLI

```bash
uv run jspace-intervention run-valence-readout \
  --input ../baseline/runs/paper-local-qwen36-27b/trial_plan.jsonl \
  --split-manifest artifacts/<model-slug>/jspace-intervention/splits.json \
  --model .cache/models/qwen3.5-4b \
  --run-id valence-technology-discovery-1 \
  [--lens artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt] \
  [--dataset jspace-valence-readout] [--artifact-root artifacts] \
  [--sector Technology] [--split discovery] \
  [--trials-per-ticker 3] \
  [--layers 14,15,16,17,18,19,20,21,22,23,24,25,26] \
  [--top-k 30] [--max-seq-len 1024] [--seed 0]
```

單一指令完成 prepare → forward → analyze → finalize。

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
[token causal screen V1](jspace-token-causal-screen-v1.md) 的 `run-token-screen`
以 SHA-bound config 直接消費，並在 discovery split 完成對稱劑量 screen；V1 shortlist
為空。後續 V2 不再由本 readout 提名詞彙方向，而改 fitting outcome-gradient
axis。兩版差異見 [J-space token experiment versions](jspace-token-causal-screen.md)。

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
