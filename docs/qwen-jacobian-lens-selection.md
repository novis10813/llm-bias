# Qwen3.5-4B Jacobian-lens calibration 與候選選擇

這份文件記錄 Qwen3.5-4B 的 model-specific Jacobian-lens pipeline：如何建立
English-only、Simplified-Chinese-only 與 bilingual mixed 三組 calibration，
如何完成可恢復的逐層 fitting，以及如何使用獨立 bilingual holdout 選擇唯一
active canonical lens。

這個 pipeline 解決的是「哪個 calibration condition 產生較適合目前雙語
readout 任務的 lens」。它不直接證明 entity bias、因果作用、chain-of-thought
或 global workspace。

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
SHA-256 與 provenance。

```text
artifacts/qwen3.5-4b/jacobian-lens/
├── checkpoints/              # canonical fit checkpoints
├── archive/                  # replaced active lenses
├── selection.json
└── candidates/
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

## 2026-07-30 實驗結果

| Candidate | Balanced native mean log10 rank ↓ | Selection score ↑ |
|---|---:|---:|
| `chinese_simplified` | **3.826620** | **-3.826620** |
| `mixed` | 3.881337 | -3.881337 |
| `english` | 3.889848 | -3.889848 |

依固定 selection rule，winner 是 `chinese_simplified`。Candidate lens SHA-256：

| Candidate | SHA-256 |
|---|---|
| `english` | `847a272025bcb22ceccc0e89d8b534695b30819b83caf643191929b96ae753e8` |
| `chinese_simplified` | `3691d7b2314ed654264def681a285de9e70421ae0a6591a65468da2b2416349c` |
| `mixed` | `43624e4aa8d9916ccce9bf26723af5167bd62c87d106e96df29ffe134845fb13` |

Selection uncertainty 以 32 個 semantic pairs 為 paired unit，對每個 pair 先平均
EN/zh-CN log10 rank，再執行 deterministic 10,000-resample bootstrap 與
one-sided sign-flip permutation：

| Comparison | Selected − competitor | Paired bootstrap 95% CI | One-sided p |
|---|---:|---:|---:|
| Chinese-only vs English-only | -0.06323 | [-0.17790, 0.04780] | 0.1483 |
| Chinese-only vs Mixed | -0.05472 | [-0.12093, 0.01096] | 0.0573 |

負差值代表 Chinese-only 較好，但兩個 confidence intervals 都跨 0。因此正確
結論是：

> Chinese-only 是目前 preregistered rule 下的 operational winner，但現有
> holdout 尚未提供它顯著優於另外兩組的證據。

同一份 holdout 同時負責選 winner 與描述 winner uncertainty，所以這些 p-values
是 descriptive，不是 confirmatory。後續若要提出語言 calibration 的一般性
主張，必須使用新的 untouched bilingual holdout。

## Canonical promotion

Promotion 會先把現有 canonical lens 與 metadata 複製到 timestamped archive，
再以 atomic copy 更新 active lens：

```bash
uv run python scripts/promote_qwen_lens_candidate.py \
  --model .cache/models/qwen3.5-4b \
  --evaluation artifacts/qwen3.5-4b/jacobian-lens/candidates/evaluation.json
```

Active artifact：

```text
artifacts/qwen3.5-4b/jacobian-lens/
├── jacobian_lens.pt
├── jacobian_lens.pt.metadata.json
└── selection.json
```

`selection.json` 記錄 canonical SHA-256、winner、evaluation 路徑、不確定性與舊
lens archive。任何 interactive dashboard 都要求 canonical lens 完整覆蓋所有
source layers；partial/stride experiments 不得放入 active model folder。

## Prompt-analysis dataset gates

Qwen lens promotion 與資料集 sampling 是兩個獨立 gate。Legacy-wide `generate`
stage 預設每個 condition deterministic sampling 32 個共同日期；這個數字不可套用
成 return-pairs 的完整 generation 上限。MAG7 8-K return-pairs runner 明確使用
`return-pairs` schema、`RUN_GENERATION=1`、`RUN_ATTRIBUTION=0` 與
`GEN_SAMPLE_PER_CONDITION=0`，由 runner 傳入 `--full-generation`，對 710 個 unique
pairs 保存 1,420 筆 `original`/`counterfactual` condition records。

在宣告 run 完成前，tiny-fixture contract test 應檢查 manifest identity、enabled stage
status、manifest/file hashes、backward parent SHA-256 與 coverage，以及 raw activation
artifact 不存在。這些 checks 不需要 Qwen checkpoint；Qwen full calibration/evaluation/
promotion 才需要 model inference。

## 如何解讀與下一步

這次結果證明 calibration → fitting → holdout selection → archive/promotion 的
工程 pipeline 可重現地運作。它尚未證明：

- Chinese-only calibration 普遍優於 mixed 或 English-only。
- Jacobian readout 是模型的離散 reasoning path 或 chain-of-thought。
- Readout signal 對答案有因果作用。
- 模型存在 entity-level bias。

下一步應凍結這份 holdout，先用新的 confirmatory bilingual set 檢查 winner
是否重現，再把 canonical lens 用於 span activation patch、bias-specific
counterfactual pairs、雙向 causal controls 與 paired statistics。
