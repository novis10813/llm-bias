# Qwen3.5-4B Jacobian-lens calibration 與候選選擇：實驗報告

Calibration design、fitting、holdout 與 selection rule 見 [提案與操作契約](proposal.md)。

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

## Qwen3.6-27B GPU-only single-condition workflow

27B checkpoint 的 `config.json` 可能實際宣告 `Qwen3_5ForConditionalGeneration`、
64 layers、`d_model=5120`；artifact metadata 必須保存這個實際 identity，而不能把
目錄名稱當成 architecture。使用 `llm_bias.core.model.load_model` 時，只有明確傳入
`device_map` 才啟用 Accelerate sharding；default 仍是單 GPU。27B workflow 使用兩張
GPU 的 balanced map 與顯式 `max_memory`，禁止 CPU、disk、meta offload、quantization、
DDP、DataParallel 與 compile，並保存 `model_diagnostics` 的 resolved map、layer
placement、norm/head device 與 parameter bytes。

先執行 load/forward/autograd probe，再執行 1-prompt 與 8-prompt full L0--L62
benchmarks。benchmark 與 progress/log 僅能寫入
`artifacts/qwen3.6-27b/jacobian-lens/candidates/chinese_simplified/benchmarks/`
下的 `one_prompt/`、`eight_prompts/` noncanonical scope；舊 canonical 或真正
partial/stride checkpoint 才能使用 archive。48 小時估算、finite Jacobian、資源
穩定及 checkpoint resume 任一失敗時必須 fail closed，不能建立 canonical lens。
只有 128 Chinese prompts 的完整 L0--L62 lens
通過固定 L22--L47 holdout evaluation 後，才可使用 preselected single-candidate
promotion。這不是在 27B 上重新選擇 Chinese/English/mixed condition；metadata 的
`selection_basis` 應為
`inherited_qwen3.5_4b_operational_winner_then_qwen27b_holdout_validation`。
