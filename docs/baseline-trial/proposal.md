# Baseline trial-plan prompts → prompt-analysis 執行計畫

把 `../baseline`（FELAB-UNIST LLM bias in finance 重建版）的 trial-plan prompts
搬進本 repo，跑 `prompt-analysis` 的 forward（readout + generate）與
backward（attribute-generated）stage 及後續驗證／視覺化。

## 1. 資料轉換（已完成，無模型）

轉換腳本：`scripts/convert_baseline_trial_plan.py`（stdlib only，不 import
baseline 程式碼，也不 import 本 repo 套件）。

```bash
uv run python scripts/convert_baseline_trial_plan.py \
  --source ../baseline/runs/qwen36-27b-50stocks/trial_plan.jsonl \
  --output data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv

uv run python scripts/convert_baseline_trial_plan.py \
  --source ../baseline/runs/paper-local-qwen36-27b/trial_plan.jsonl \
  --output data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv
```

輸出為 legacy-wide CSV（`prompt-analysis inspect-input` 已驗證通過）：

- 一行一支 ticker；靜態欄位 `Date`（固定 cohort 標籤）、`ticker`、`name`、
  `sector`、`marketcap`。
- 每個 (condition, occurrence) 一欄：`prompt_with_context_<condition>_<occurrence>`。
  baseline 的 trial 身份是 `trial_key`；同一 (ticker, condition) 的
  `(set_index, trial_index)` 可能重複出現（不同證據抽樣），所以欄位用
  per-ticker 出現順序編號，不用 trial_index。
- `ticker` 欄位會被 generate stage 自動帶進輸出記錄（allowlist 欄位）；
  condition／occurrence 可從各 stage 輸出的 `prompt_column`／`index` 還原。
- 每個 CSV 旁有 `.provenance.json`：source 路徑與 SHA-256、記錄數、
  condition×trials 矩陣、輸出 SHA-256。

已轉換規模：

| dataset | rows (tickers) | prompt columns | 非空 cells |
|---|---|---|---|
| `data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv` | 50 | 16（attribute 2, volume 2, intensity 8, strategy 4） | 800 |
| `data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv` | 427 | 72（attribute 30, volume 6, intensity 24, strategy 12） | 30,744 |

Token 長度（qwen3.6-27b tokenizer，chat template 單 user turn、
`enable_thinking=False`，與本 repo `_prepare_prompt` 一致）：

- 50-stock：min 161 / p50 245 / p99 604 / max 660
- paper：min 87 / p50 333 / p99 619 / max 733

**兩個 dataset 都用 `--max-seq-len 1024`**（預設 256 會截斷 evidence）。

注意：CSV 裡**不要**加 `condition` 或 `system_prompt` 欄位——legacy-wide 的
row filter 會拿 `row["condition"]` 對欄位 context 做匹配，baseline 資料是
單 user turn、沒有 system prompt。

## 2. 模型選擇

- **主選：qwen3.6-27b**。權重在 `.cache/models/qwen3.6-27b`，canonical lens
  在 `artifacts/qwen3.6-27b/jacobian-lens/jacobian_lens.pt`；baseline 的
  provider 模型同名（llama.cpp 量化版），本 repo 用全精度 HF 權重是正確
  白箱 protocol。
- 備選：qwen3.5-4b（lens 與權重都有，成本低，可做跨模型 sanity check；
  但與 baseline 黑箱 runs 不是同一模型）。
- 不可行：baseline 的 qwen3.5-9b runs——本 repo 沒有 qwen3.5-9b 的 canonical
  lens 與權重。

## 3. 執行 stages（GPU 空出時）

baseline dataset 有專用 CLI `baseline-trial`（`llm_bias/baseline_trial/`），它照
repo 的 `prepare → forward → analyze → finalize` 合約把 CSV 跑進既有
prompt-analysis pipeline，並多一個 baseline 特有的 `lens-forward` stage：對
forward 的每個 (prompt + generated) 序列做 per-position、per-layer 的
Jacobian-lens readout，存到 `<run-root>/lens-forward/lens_readout.jsonl`。

stages：`readout`（prompt J-lens readout）、`forward`（greedy generate，存
`<run-root>/forward/`）、`lens-forward`（生成序列的 J-lens readout，需要完整
forward artifact）、`backward`（attribute-generated）、`validate`（semantic
scope validation）。`backward`/`validate` 需要單 GPU 載入（不支援 sharded
loading），GPU 不足時先跑前三個 stage，之後用 `--forward-artifact` 補跑。

50-stock pilot（qwen3.6-27b）：

```bash
uv run baseline-trial run \
  --input data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv \
  --model .cache/models/qwen3.6-27b \
  --run-id baseline-50stocks-$(date -u +%Y%m%d) \
  --stage readout --stage forward --stage lens-forward
```

- 預設 `--generate-full`：所有 800 個 cell 都 generate（baseline trial plans
  沒有 per-date sampling 的意義）。
- GPU 佔用高時加 `--device-map qwen27b_two_gpu` 把 27B 分到兩張卡
  （`forward`、`readout` 與 `lens-forward` 都支援；`backward`/`validate` 仍
  需要單 GPU）。Sharding 時 Jacobian 會放在 lm_head 所在的 GPU（27B 為
  GPU 1，+6.6 GB fp32），兩卡各需 ~30 GB 以上空餘 VRAM。
- 只跑 prompt readout 時可加 `--max-rows` 與 `--lens-forward-layers`
  （逗號分隔的 layer 子集，預設全部 source layers + final layer）控制成本。
- lens-forward 的 readout 已批量化（每 layer 一次 batched
  transport/unembed/topk，Jacobian 先搬到 model device 常駐；4B +0.8 GB、
  27B +6.6 GB fp32 VRAM）：4B 50-stock 5 層 readout 由 ~24 min 降到 ~4 min，
  讀取值與舊版一致到 bf16/ulp 容差（top-15 尾端 rank 偶爾換位）。剩餘成本
  在 per-record decoder forward 與 JSON 組裝；layer 數多時可視分析需求調低
  `--top-k`。
- 生成 token 數預設 256（baseline 回應是 JSON `{"decision","reason"}`，
  預設 64 會截斷 reason）；readout batch 預設 8，96 GB 卡上 seq 1024 的 27B
  可往 16 調；attribution 是 backward，若 OOM 先降 batch。
- forward-only 後補跑 lens-forward / backward / validate：

  ```bash
  uv run baseline-trial run \
    --input data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv \
    --model .cache/models/qwen3.6-27b \
    --run-id baseline-50stocks-YYYYMMDD \
    --stage lens-forward --stage backward --stage validate \
    --forward-artifact artifacts/qwen3.6-27b/trial_plan_prompts/runs/baseline-50stocks-YYYYMMDD/forward/generated_outputs.jsonl
  ```
RUN_ROOT 會落在 `artifacts/<model-slug>/trial_plan_prompts/runs/<RUN_ID>/`。
兩個 baseline dataset 同名 basename，靠 `--dataset` 或 RUN_ID 區分（例：
`--dataset qwen36-27b-50stocks`、`--dataset paper-local-qwen36-27b`）。

paper-scale 同參數、換 `--input` 與 `--run-id baseline-paper-...`；30,744
prompts 的 readout + attribution 時間長很多，排程上放 pilot 之後。

## 4. 後續評估與視覺化

```bash
# attribution 驗證（需要 backward artifact）
uv run prompt-analysis validate-attribution \
  --model .cache/models/qwen3.6-27b \
  --attribution artifacts/qwen3.6-27b/trial_plan_prompts/runs/<RUN_ID>/backward/generated_token_attribution.jsonl \
  --output-dir artifacts/qwen3.6-27b/trial_plan_prompts/runs/<RUN_ID>/attribution_validation

# 圖表 + 互動 dashboard
RUN_ROOT=artifacts/qwen3.6-27b/trial_plan_prompts/runs/<RUN_ID> \
TOKENIZER=.cache/models/qwen3.6-27b \
INPUT_CSV=data/baseline/qwen36-27b-50stocks/trial_plan_prompts.csv \
MAX_SEQ_LEN=1024 \
bash scripts/visualize_prompt_analysis.sh
```

可做的分析（皆從 stage 產出即可，不需再跑模型）：

- 逐層 readout：4 個 condition 的 per-layer top-k / uncertainty 對照
  （`index` 欄位含 condition 名稱）。
- generate 輸出：解析 `generated_text` 的 JSON decision，重算 baseline 式
  的 buy/sell flip rate（sector、market-cap、momentum/contrarian）。
- 交叉驗證：與 `../baseline/runs/<run>/responses.jsonl` 比 decision 一致率。
  注意 protocol 差異——baseline 走 llama.cpp API、temperature 0.6；本 repo
  generate 是 greedy（temperature 0），一致率是 sanity check 不是等價性
  證明。

## 5. 限制

- 本 repo 的 readout/attribution 是 representation-level 證據；baseline 的
  黑箱 bias 指標（flip rate 等）要從 generate 輸出另行解析計算。
- baseline 的 `sp500_final.csv` 沒有搬進來：本 repo 已有 2020–2025 成分股
  資料（更完整），且 `synthetic-entity-bias` workflow 驗證嚴格 schema／檔名。
- `data/baseline/` 的 CSV 由 root `.gitignore` 的 `*.csv` 規則排除；provenance
  JSON 記錄 source SHA-256，讓本地資料可查核與重建。

## 6. 未納入（後續研究選項）

從 baseline evidence corpus 建 counterfactual entity-bias pairs（同 evidence
context、換 entity identity、固定 buy/sell outcome options 的 logit margin），
接 `counterfactual_patching`。需要新的 pair 建構＋review gate（參照
8-K workflow），且要先決定證據文字中重複出現的 ticker 名稱換哪些位置。
