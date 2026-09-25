# 三模型 DIM 跨層掃描 V2：協議與實作規格

**狀態：**事前協議，2026-09-25；尚未執行 101 家正式評估。**取代：**[V1](proposal-crossmodel-dim-layer-sweep-v1.md)（只跑過 ABNB smoke，無正式結果）。**範圍：**Gemma-4-12B、GLM-4-9B、GPT-OSS-20B；2024 S&P 500 母體 503 家，seed `20260923` 固定 402 家建方向、101 家受測。不改寫 [Qwen DIM V1](proposal-dim-layer-sweep-v1.md)、[四模型 paper cone](../crossmodel-cone-paper/proposal.md) 或任何既有產物。以下規則在 101 家結果揭露前固定；方向、選層、劑量、主要指標或 controls 改變時另立版本。

## 為何需要 V2：V1 smoke 暴露的四個設計問題

V1 的 ABNB smoke（`artifacts/<slug>/concept-cone-steering/runs/dim-crossmodel-layer-sweep-v1-20260925-smoke-01/`，GPT-OSS 另有 `-smoke-02`）顯示下列問題會讓 V1 的正式結果無法解讀：

1. **劑量單位跨模型不可比。**V1 加的是「單位向量 × α」，α 網格沿用 Qwen。各模型 Top−Bottom 差向量的 norm 相差一到兩個數量級（tokenwise 峰值層中位數：Gemma L27 2.8、GLM L19 1.7、GPT-OSS L14 49.5），同一個 α=6 對 GPT-OSS 只推動差距的約 0.12 倍，對 GLM 卻是約 3.6 倍。GPT-OSS L14 的 margin 在 α 0→6 只變動約 0.1 nats。
2. **只有正 α，GLM 的翻轉在結構上沒有分母。**方向是 Top−Bottom（往 buy 推），但 GLM 在既有 cone 的 α0 為 101/101 buy，`sell→buy` 分母預期為 0。
3. **嚴格解析對兩個模型在結構上是 0%。**Gemma 的輸出以解碼殘留的 `thought\n` 開頭（其 thinking 已由模板關閉，後面直接是 JSON）；GPT-OSS 輸出 Harmony 的 `analysis…assistantfinal{…}`。整段 `json.loads` 對兩者都必然失敗。
4. **GPT-OSS 的 thinking 沒有被關掉。**`enable_thinking=False` 只被 Qwen／Gemma 模板讀取；GPT-OSS 的 Harmony 模板改讀 `reasoning_effort`（預設 `medium`），因此 192 token 全停在 `analysis` channel。改成 `low` 後，ABNB 在預算內進入 `final` channel 並輸出完整物件。

## 設計摘要

| 項目 | V2 規則 |
|---|---|
| 方向 | 各模型自行排序 402 家建方向公司，取 Top10／Bottom10；每層用該層乾淨殘差計 **fp32 原始**差 `d_l = mean(Top_l) − mean(Bottom_l)`，**不做單位化** |
| 劑量 | 注入 `α · d_l`；α=1 即沿該方向推滿一個 Top−Bottom 平均差距 |
| α 網格 | `[−2, −1, −0.5, 0, 0.5, 1, 2]`，對稱；α0 每家公司只生成一次、各層共用 |
| 主要解析 | 完整物件解析（6 種事前定義外殼），執行當下計算 |
| 次要解析 | 整段 `json.loads` 的嚴格解析，並列報告 |
| Chat template | Qwen／Gemma `enable_thinking=False`；模板含 `reasoning_effort` 者（GPT-OSS）另傳 `reasoning_effort="low"`；模板呼叫 `strftime_now` 者（GPT-OSS）日期固定為 `2026-09-25`；GLM 無思考開關 |
| 選層 | 沿用 V1 由 C2 427 家曲線預選的層集合，含結構上必為零的比較層（當診斷） |

## 選層與來源（沿用 V1，不因 101 家結果增刪）

先取所有 `T ≥ 0.70 × peak T` 的層；若少於 5 層才加 peak±2；最後加 L0 與最後一層。由 `scripts/probe_concept_cone.py:validate_dim_crossmodel_curve` 在每次執行時重新核對。

| 模型 slug／層數 | C2 peak `T` | 高 `T` 層 | 全部候選層 | dtype | 共同尾段 K | C2 summary SHA-256 |
|---|---:|---|---|---|---:|---|
| `gemma4-12b-it`／48 | L27，0.1903 | L26–L32 | **0,26,27,28,29,30,31,32,47** | bf16 | 100 | `295170b9c09ca453ce7b285c37a0f128d99cf68fc1a20970f390f084fdd54635` |
| `glm4-9b-0414`／40 | L19，0.4592 | L17–L21 | **0,17,18,19,20,21,39** | bf16 | 98 | `d0b7e41ac21adc1cb858fbdc6a75e34338109cab84f3630bcd6f839119b480a6` |
| `gpt-oss-20b`／24 | L14，0.5323 | L12–L16 | **0,12,13,14,15,16,23** | native（MXFP4） | 99 | `f8ccb7c24d25048d358a32562ba8fffd799384a482b1e17152d7f0bef543a506` |

C2 summary 位於 `artifacts/<slug>/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json`，每層 `n_directions=854`。選層用的 427 家 C2 公司與 101 家受測公司**重疊 89 家**：受測組只對本次排序與方向擬合留出，不是獨立選層驗證。[Gemma 與 GLM 的 Phase 2A gate 曾 override](../c2-v2-427/status.md)。

## REQ-1：族群、排序與模型身份

【已實作：`split_population`、`prepare_instruction_suffix`、`rank_construction`、`run_dim_crossmodel_v2`】

- Prompt：`_render_frozen_prompt(ticker, name, order=0, reverse=False)`，經 `format_prompt(..., use_chat_template=True, enable_thinking=False, chat_template_kwargs=template_render_kwargs(tokenizer))` 渲染。GPT-OSS 的 Harmony 模板以 `strftime_now`（即 `datetime.now()`）在 system message 寫入 `Current date`，跨午夜的 run 會混用兩種渲染；`template_render_kwargs` 在 `low_reasoning_kwargs` 之外以同名渲染變數遮蔽該 Jinja global，把日期固定為 smoke 當日 `2026-09-25`，metadata 另記 `chat_template_date`（僅限模板有此呼叫的模型；Gemma／GLM 的渲染與 metadata 不變）。此項於 2026-09-25 16:55 補入，當時 GPT-OSS 正式 run 尚未開始。
- 排序：每次執行都以 V2 的渲染，對 402 家建方向公司計算乾淨固定後綴 margin（`{"decision": "` 後的 `log p(buy) − log p(sell)`，fp32），依 `(margin, ticker)` 升序；Top10＝最末 10 家，Bottom10＝最前 10 家。完整 402 列存於 `result.json` 的 `ranking`；`top_10`／`bottom_10` 寫入 metadata，續跑時必須一致。
- **不再**讀取 V1 所用的舊 cone ranking source。Gemma 與 GLM 的渲染與舊 source 相同，status 將以描述方式比對兩者的 Top／Bottom；GPT-OSS 的舊 source 是在 `medium` 下算的，不適用。
- 共同尾段 K 由模型當下 tokenizer 對全部 503 家重新計算，必須等於上表（GPT-OSS 在 `low` 渲染下已於 smoke 確認仍為 99）。
- metadata 記錄：model 絕對路徑、slug、dtype、config／tokenizer config SHA、checkpoint 檔名與大小、tokenizer identity、`chat_template_sha256`、`chat_template_kwargs`、population／split／prompt family SHA、C2 source 與各層 `T`、固定前綴、`max_new_tokens=192`、K、`primary_parse` 與 `allowed_formats`。C2 SHA、層數、dtype、K 任一不符即停止。

## REQ-2：兩臂擬合與注入

【已實作：`extract_dim_layer_directions(..., normalize=False)`、`fit_dim_difference`、`run_cone_evaluation`】

- **主要 `tokenwise` 臂：**取 20 家乾淨 prompt 的共同 instruction 尾段 `[K, d_model]` 各層 post-block states，逐 token 計原始差 `d_l[p]`；評估時在同層該公司尾段 post-block 加 `α · d_l[p]`，只改 prefill，不改 decode token。
- **補充 `single_all` 臂：**取各層 **block 輸入**在最後一個 instruction token 的狀態，計單一原始差 `w_l ∈ R^{d_model}`；評估時在同層 block 輸入對所有 prefill 與 cached decode token 加 `α · w_l`。兩臂的同一 α **不是**等量的整段序列劑量。
- 所有 state、差與 norm 必須 finite，否則該臂停止。差向量 norm 為 0 **不再** fail closed：原始差為零向量時注入即為 no-op。只保存每層 min／median／max norm 與差向量 SHA（`difference_sha256`），不保存 raw hidden states。
- **結構上必為零的比較層照跑，作為診斷：**
  - `single_all` 的 L0：block 輸入是最後一個 instruction token 的 embedding，各公司完全相同，差為零向量（V1 smoke 三模型實測 norm=0）。預期各 α 的 margin 與生成都與 α0 完全相同，可用來檢查 hook 與生成的決定性。
  - `tokenwise` 的最後一層：注入點在最後一個 block 之後，僅改 prefill 的 instruction 位置，影響不到讀答案的位置（V1 smoke 的 margin 每一位小數都與 α0 相同）。預期同上。
  - 若這些層出現與 α0 不同的結果，status 必須記錄並停止解釋其他層的翻轉，直到找出原因。

## REQ-3：解析與指標

【已實作：`llm_bias/core/decision_parsing.py`、`parse_generation`、`dim_v2_layer_summary`、`dim_v2_alpha_table`】

每家公司在 α0 做一次、每層在 6 個非零 α 各做一次：以固定後綴 `{"decision": "` 取 margin（nats），並從無後綴的原 prompt greedy 生成最多 192 token（`skip_special_tokens=True` 解碼，保留全文）。

**主要解析（complete object）：**去掉至多一層事前定義的外殼後，剩下的整段必須恰為只含 `decision`、`reason` 兩鍵的 JSON 物件，`decision ∈ {buy, sell}`、`reason` 為非空字串。外殼只接受以下 6 種：

| 格式 | 規則 |
|---|---|
| `bare_json` | 整段即物件 |
| `fenced_json` | 整段為 `` ```json\n…\n``` `` |
| `thought_bare_json`／`thought_fenced_json` | 以 `thought\n` 開頭，其後為上兩者之一（Gemma 的 thought channel 標記解碼殘留） |
| `harmony_final_bare_json`／`harmony_final_fenced_json` | 在最後一次出現的 `assistantfinal` 處切開，其後整段為前兩者之一（GPT-OSS 的 Harmony `final` channel 標記解碼殘留；不接受 `thought` 殼或尾隨文字） |

任何其他情況（包括仍截斷在 `analysis` channel）記為 `unparsed`，**絕不**以 regex 或 margin 猜測決策。

**次要解析（strict）：**整段 `json.loads` 為物件且 `decision ∈ {buy, sell}`。

**每層、每 α 報告：**

- 主要與嚴格的已解析數 `n/101`；
- median margin；
- `mean(ΔM_α) = (1/101) Σ_i [M_i(α) − M_i(0)]`；
- 以同一公司 α0 的主要解析決策為條件，且該 α 也可解析時，報 `buy→sell` 與 `sell→buy` 的分子／有效分母，以及全部有效配對中的翻轉數。
- 基準類別或有效分母為 0 時記 `null`，表格記 `—`，**不是 0%**。
- 正 α 預期推向 buy、負 α 預期推向 sell；兩個方向在每個 α 都要報。
- 固定答案 margin 的正負與 C2 `T` 都不能充作生成翻轉。
- 另報全 run 的格式計數 `format_counts`（α0 每家計一次）。

## REQ-4：CLI、產物與續跑

【已實作：`--cohort-mode sp500_dim_crossmodel_v2`、`validate_dim_output`、`validate_dim_v2_resume`】

- 只接受上表三模型與其 dtype、`--dim-arm tokenwise|single_all`、`--dim-layers` 為候選層的升序子集、`--alphas -2 -1 -0.5 0 0.5 1 2`、`--split-seed 20260923`、balanced evidence；拒絕舊 cone 的控制參數。
- `--smoke-tickers` 只能選 101 家受測公司，metadata `mode=smoke`，不得當成正式評估。
- 輸出限定為 `artifacts/<slug>/concept-cone-steering/runs/dim-crossmodel-layer-sweep-v2-<id>/<arm>/result.json`；schema 為 `dim-tokenwise-crossmodel-v2`／`dim-single-all-crossmodel-v2`；smoke 與正式 run 使用不同 `<id>`。
- 結構：`targets[ticker]["baseline"]` 為 α0 一列，`targets[ticker]["L{layer}"]["rows"]` 為 6 個非零 α。每列只有 `alpha`、`margin`、`generated_text`、`decision`、`format`、`strict_decision`。
- 續跑時 metadata（含重新排序得到的 Top／Bottom）與方向 SHA 必須完全相同，且每列的決策欄位必須能從全文重新推導；每家公司完成 α0 或一層後原子寫入。`complete=true` 只在所選層 × 受測公司全部完成後設定；已完成的 run 唯讀。
- Qwen V1 的 `sp500_dim_paper`、舊 cone CLI、schema 與舊 artifacts 不變。

## 執行

每個模型先做 smoke（ABNB，峰值層與最後一層，`single_all` 另加 L0），檢查 margin 有限、α0 決策可解析、兩個結構零層與 α0 相同，並估計單次生成耗時。smoke 通過後，每張 GPU 一次跑一個模型：先跑主要臂的全部候選層，再跑補充臂的全部候選層。

```bash
# 以 GLM 為例；Gemma／GPT-OSS 換 --model、--dim-layers，GPT-OSS 加 --model-dtype native
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run python scripts/probe_concept_cone.py \
  --cohort-mode sp500_dim_crossmodel_v2 --model .cache/models/glm4-9b-0414 --model-dtype bf16 \
  --population-csv data/sp500_constituents_2020_2025.csv --split-seed 20260923 \
  --dim-arm tokenwise --dim-layers 0 17 18 19 20 21 39 --alphas -2 -1 -0.5 0 0.5 1 2 \
  --output-json artifacts/glm4-9b-0414/concept-cone-steering/runs/dim-crossmodel-layer-sweep-v2-<id>/tokenwise/result.json
```

GPT-OSS 的 MXFP4 需要 `kernels-community/gpt-oss-triton-kernels`；本機快取不存在時，第一次執行不能設 `HF_HUB_OFFLINE=1`。

完成後才新增 `status-crossmodel-dim-layer-sweep-v2.md`，記錄實際命令、run 路徑、SHA、各模型 × 臂的完整矩陣、結構零層診斷、Gemma／GLM 排序與舊 source 的比對，以及上述指標表（α 為欄）。未完成不可寫成「已完成」。

## 限制

- 描述性研究：沒有 random-direction control，也沒有獨立公司的確認。
- 89 家受測公司同時在 C2 選層集合內；Gemma 與 GLM 的 Phase 2A gate 曾 override。
- α 以各層原始差為單位，與 Qwen V1 的「單位向量 × α」不可直接比較。
- GPT-OSS 在 `low` 下仍可能有公司在 192 token 內未進入 `final` channel，這些會記為 `unparsed`。
- `template_render_kwargs`（`reasoning_effort="low"` 與固定日期）套用在 `probe_concept_cone.py` 所有 chat template 渲染上。因此若以現行程式重跑 GPT-OSS 的舊 cone 命令，渲染會和既有 artifact 不同；既有 artifact 本身不變。

## 驗證

```bash
uv run pytest -q tests/test_probe_concept_cone.py tests/test_decision_parsing.py tests/test_reparse_concept_cone_decisions.py tests/test_core_inference.py
uv run pytest -q && uv lock --check && git diff --check
```
