# Qwen3.5-4B Jacobian-lens calibration 與候選選擇：提案與操作契約

這份文件記錄 Qwen3.5-4B 的兩種 model-specific Jacobian-lens 來源：預設的
pinned pretrained artifact，以及 English-only、Simplified-Chinese-only 與
bilingual mixed 三組 calibration 的本地研究替代流程。

## 預設：安裝 pinned pretrained lens

`config/pretrained_lenses.json` pin 住 `neuronpedia/jacobian-lens` 的完整 commit、
Qwen3.5-4B lens/config 路徑與 SHA-256。`jacobian-lens install` 只在 local
checkpoint 的 config 能證明 exact base identity `Qwen/Qwen3.5-4B`、architecture、
residual width、layer count 與完整 L0–L30 coverage 都相符時，才把 artifact 安裝成
canonical lens。Experiment runtime 只讀取及驗證本地 canonical artifact，不會連網。

這個公開 lens 使用 `Salesforce/wikitext:wikitext-103-raw-v1` calibration；它不是
下述本地 bilingual candidate selection 的 winner，也沒有 `Qwen3.5-4B-Instruct`
alias。兩種 provenance 不可在結果中無標示混用；若要比較，lens source 必須成為
明確的實驗條件。

## 本地研究替代：bilingual candidate selection

以下 pipeline 解決的是「哪個 calibration condition 產生較適合目前雙語
readout 任務的 lens」。它可在需要 project-specific calibration 時取代 pretrained
artifact，但不直接證明 entity bias、因果作用、chain-of-thought 或 global
workspace。

## Pipeline 概覽

```text
parallel 16-domain × 8-style calibration
        │
        ├── English-only:            128 prompts
        ├── Simplified-Chinese-only: 128 prompts
        └── Mixed:                    64 EN + 64 zh-CN
        │
        ▼
three resumable, complete L0–L30 lens fits
        │
        ▼
32 semantic pairs × 2 languages bilingual holdout
        │
        ▼
fixed L11–L23 native-token rank selection
        │
        ▼
archive old canonical → atomically promote winner
```

所有三個 candidate 都使用相同模型、chat formatting、sequence limit、layer
coverage 與 fitting 參數。Candidate lens、checkpoint、evaluation 與 logs 位於
ignored model-scoped `artifacts/<model-slug>/jacobian-lens/`，只有資料生成器、tracked calibration/holdout 和程式進入
root Git repository。

## 與 prompt-analysis stage artifacts 的邊界

Qwen lens selection 產生的是 model-scoped lens artifact，不是 prompt-analysis
run 的 forward/readout/backward 結果。Lens 必須先在
`artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt` promotion 完成，prompt-analysis
才可使用它；不得把 candidate、partial 或另一個 model 的 lens 放入 run tree。

Prompt-analysis 的 approved batch layout 是：

```text
artifacts/<model-slug>/<dataset-slug>/runs/<run-id>/
├── manifest.json
├── forward/   # generated token records only
├── readout/   # Jacobian readout/uncertainty
└── backward/  # generated-token attribution, only when enabled
```

Root manifest 保存 model/dataset/run identity、input 與 artifact SHA-256、record counts
與 stage statuses。Backward metadata 保存實際執行的 model identity、parent forward path/hash
與 per-record generated-token coverage；目前 producer 不宣稱 same-dataset 或 same-run binding。
Stage 只保存 compact readout/provenance，不得保存 residual、embedding 或 gradient activation；
這是 prompt-analysis 的 artifact contract，不是 counterfactual-patching 全面 layout migration
的宣稱。

## Calibration 設計

資料位於 `data/calibration/qwen3.5-4b/`：

| Condition | Prompt 數 | 語言組成 |
|---|---:|---|
| `english` | 128 | 128 English |
| `chinese_simplified` | 128 | 128 Simplified Chinese |
| `mixed` | 128 | 64 English + 64 Simplified Chinese |

三組共用 16 個 domain 與 8 個 discourse style。English 與 Chinese rows 依
`pair_id` 平行；mixed 使用 domain 與 style 的交錯 parity 選語言，因此每個
domain 都是 4 EN + 4 zh-CN，每個 style 都是 8 EN + 8 zh-CN。這避免 mixed
condition 的語言與 topic/style 混淆。

以 Qwen chat template、thinking disabled 格式化後，token 長度分布為：

| Condition | Mean | Standard deviation | Min–max |
|---|---:|---:|---:|
| English | 76.17 | 3.41 | 67–85 |
| Simplified Chinese | 74.09 | 2.87 | 66–82 |
| Mixed | 74.89 | 3.18 | 66–83 |

重新產生 tracked data：

```bash
uv run python scripts/prepare_qwen_calibration.py
```

Generator 會寫入 `manifest.json`，其中記錄 condition counts、domain/style
inventory 與各 JSONL 的 SHA-256。Regression tests 會檢查數量、唯一性與
mixed condition 的 domain/style 語言平衡。

## Fitting 設定與恢復

三個 candidate 的固定設定：

| 參數 | 值 |
|---|---|
| Model | `.cache/models/qwen3.5-4b` |
| Source layers | L0–L30，完整 31 層 |
| Target layer | L31 |
| Calibration prompts | 128 |
| Chat template | enabled |
| Thinking | disabled |
| `max_seq_len` | 128 |
| `skip_first` | 16 |
| `dim_batch` | 8 |
| Checkpoint interval | 每 4 prompts |

啟動完整 workflow：

```bash
bash scripts/run_qwen_lens_candidates.sh
tmux attach -t qwen_lens_candidates
```

Runner 依序 fitting `english`、`chinese_simplified`、`mixed`。已完成的
`jacobian_lens.pt` 會被跳過；未完成的 candidate 會從以 formatted calibration
digest 命名的 checkpoint 繼續。每個 candidate 完成後會產生 metadata，記錄
model shape、完整 source layers、calibration input/formatted digests、chat
settings、jlens version、fitting 參數、artifact type/schema version、binary/metadata
SHA-256 與 provenance。直接 fitting canonical output 時，resumable checkpoint 位於
`artifacts/archive/<model-slug>/jacobian-lens/checkpoints/`；本候選 runner 使用每個
candidate 目錄旁的 digest checkpoint，不會寫入該 canonical checkpoint root。

```text
artifacts/qwen3.5-4b/jacobian-lens/
├── archive/                  # replaced active lenses
├── selection.json
└── candidates/                # candidate-adjacent fit checkpoints
    ├── english/
    │   ├── fit.log
    │   ├── jacobian_lens.pt
    │   ├── jacobian_lens.pt.metadata.json
    │   └── jacobian_lens.pt.<digest>.checkpoint.pt
    ├── chinese_simplified/
    ├── mixed/
    ├── evaluation.json
    ├── evaluation.log
    └── promotion.log
```

Evaluation 會 fail closed：每個 candidate 必須實際包含 128 個成功 prompts、
完整 L0–L30、正確 calibration filename、chat template enabled、thinking
disabled，否則不得參與 selection 或 promotion。

## Bilingual holdout 與 selection rule

Holdout 位於
`data/evaluations/qwen3.5-4b/bilingual_intermediate_holdout.jsonl`，由 32 個
語意 pair 組成，每個 pair 各有一個 English 與 Simplified-Chinese prompt，
共 64 rows。它與 calibration prompts 分離。

```bash
uv run python scripts/prepare_qwen_lens_eval.py
```

每個 row 定義 native-language intermediate concept、對應 cross-lingual concept
與 target。為避免 English token 有兩次命中機會，評估只使用一個固定 canonical
token：

- English：優先使用 leading-space single token。
- Simplified Chinese：優先使用 raw single token。

在 Qwen 的 32-layer model 上，primary layer band 事前固定為 L11–L23。每個
prompt 的 native concept rank 是該 band 中的最佳 rank。Primary score 為
English 與 Chinese 各自 mean log10 rank 的平均值，越低越好；bilingual
canonical-token rank 只作同分時的第一個 tie-break。

Evaluation 在記憶體中暫存 final-position residual vectors，輸出只保存 compact
token ranks、summary 與 provenance，不保存 raw activations。

手動重跑 evaluation：

```bash
uv run python scripts/evaluate_qwen_lens_candidates.py \
  --model .cache/models/qwen3.5-4b \
  --candidate-root artifacts/qwen3.5-4b/jacobian-lens/candidates \
  --holdout data/evaluations/qwen3.5-4b/bilingual_intermediate_holdout.jsonl \
  --max-seq-len 128 \
  --expected-calibration-prompts 128 \
  --output artifacts/qwen3.5-4b/jacobian-lens/candidates/evaluation.json
```

結果、promotion 與後續狀態見 [實驗報告](report.md)。
