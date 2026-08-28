# J-space outcome-conditioned decision-flip V2

## Status and scope

**Status：Draft 1 已凍結、已實作；第一次正式 pipeline（discovery → calibration → test）已完成，test verdict 為 `success=false`**（唯一失敗 gate：sell 方向的 Holm-adjusted specificity，原因為 test split 的 sell-eligible tickers 只有 n=2，統計上不可能顯著，見下方 Revision record 的 Test run 1）。已完成的
vocabulary-direction screen 是 [V1](jspace-token-causal-screen-v1.md)，不能把 V1
results 當成 V2 evidence。版本入口見 [J-space token experiment versions](jspace-token-causal-screen.md)。

V2 問的問題是：從 discovery tickers 直接 fitting 的 outcome-conditioned residual
axis，能否在 held-out tickers 上，比 matched controls 更常造成指定方向的 Buy/Sell
decision flip？Margin magnitude 只作 fitting signal、dose calibration 與診斷，不作
成功結論。

## Outcome hierarchy

### Primary outcome：fixed-choice decision flip

對同一個 formatted prompt，以單一 token continuation 定義

\[
M = \log P(\mathrm{buy})-\log P(\mathrm{sell}),
\qquad
D = \begin{cases}
\mathrm{Buy}, & M>0,\\
\mathrm{Sell}, & M<0.
\end{cases}
\]

V2 primary outcomes：

- Buy steering：clean Sell 中的 Sell→Buy flip rate；
- Sell steering：clean Buy 中的 Buy→Sell flip rate；
- outcome direction 與 matched control 的 paired target-flip-rate difference。

`M=0` 的 tie 規則必須在 config schema 完成前固定；不得在分析時依結果決定。
Draft 1 已凍結：clean `M=0` 恰好成立的 records 排除於兩個 eligible set，計數報告（見下方 Frozen parameters）。

### Required behavioral validation：full generation

Fixed-choice 發生 flip 的 records 必須以相同 intervention 做 deterministic full JSON
generation，解析實際 `decision`。成功 claim 要同時報告：

- fixed-choice target flip；
- generated `decision` 是否同方向翻轉；
- JSON format/parse failure；
- fixed-choice 與 generated decision agreement。

只有 margin sign 跨界、但生成 decision 未改變，不算 behavioral steering success。

### Secondary diagnostics

`ΔM`、dose-response、clean margin bins、relative perturbation norm 與 format stability
只用來解釋 flip 或選 calibration dose。它們不能取代 primary decision outcome。

## Direction fitting

對 discovery prompt \(i\)、layer \(l\)、evidence position \(p\)，計算

\[
g_{i,l,p}=\nabla_{h_{i,l,p}}
[\log P(\mathrm{buy})-\log P(\mathrm{sell})].
\]

Gradient 是 direction-fitting signal，不是 V2 outcome。每個 layer 的 candidate axis：

1. 在 prompt 內平均 frozen evidence positions；
2. 對每個 prompt gradient 做 unit normalization；
3. 先在 ticker 內平均 prompts；
4. 對 discovery tickers 等權平均後 normalize，得到 \(d_l\)。

V2 只 fitting 一條 antisymmetric axis：`+d` 是 Buy steering，`−d` 是 Sell steering。
不得分別 fitting 兩條可獨立調參的 Buy/Sell directions。

### No-persistence and reproducibility

不得持久化 raw gradients、residuals、activations 或 direction vectors。每個 calibration
或 test run 以 frozen discovery inputs、model、lens、position rule 與 aggregation config
deterministic recompute direction；artifact 只保存：

- discovery input/config/model/lens hashes；
- layer-wise direction hashes與 norms；
- ticker/prompt counts、normalization rule 與 random seeds；
- compact dose、flip、generation 與 control diagnostics。

Calibration/test run 必須驗證 recomputed direction hashes 與 frozen direction identity
一致，否則 fail closed。Direction hash 是 raw float32 bytes 的 SHA-256，要求
bit-exact；而 bf16 GEMM backward 預設非 deterministic（同一 process 重複 fitting
同一 prompt 的 raw gradient 即有約 5e-3 max abs 差異），calibration run
`outcome-flip-tech-calibration-20260828T011744Z` 因此 fail closed。因此
`run_outcome_flip_pipeline` 在任何 CUDA 使用之前啟用
`torch.use_deterministic_algorithms(True)` 與 `CUBLAS_WORKSPACE_CONFIG=:4096:8`
（涵蓋 direction fitting、scoring 與 generation）；啟用後經實測 gradient hash 同
process 與跨 process 均 bit-identical。在 non-deterministic mode 下算出的
direction identity 不可被 verify，必須以 deterministic mode 重跑 discovery
重新 freeze。

## Intervention

對 frozen layer band \(L\) 與 positions \(P_i\)：

\[
h'_{i,l,p}=h_{i,l,p}+\alpha\frac{s_{i,l}}{\sqrt{|L|}}d_l,
\]

其中 \(s_{i,l}\) 是 clean prompt 的 local scale。`alpha > 0` 對應 Buy steering，
`alpha < 0` 對應 Sell steering。Local scale definition、floor、maximum relative
perturbation norm 與 allowed doses 已於 Draft 1 凍結（見下方 Frozen parameters），
在 calibration 前寫入 config；test 不調整。

## Split and freeze sequence

Ticker 是統計與 split 單位，同一 ticker 的 prompts 不跨 split。

```text
Discovery
  fit one outcome-gradient axis from discovery prompts only
        ↓
Calibration
  choose layer band, evidence-position rule, local scale, and doses
  inspect decision flips, controls, safety, and generation agreement
        ↓
Freeze V2 protocol and config hashes
        ↓
Test
  recompute the frozen direction identity and run once
```

Calibration 可報告 near-boundary prompts 以理解 dose，但不能只用 near-boundary subset
建立正式結論。Test 必須同時報告：

- 全部 held-out prompts；
- clean Sell eligible set for Buy steering；
- clean Buy eligible set for Sell steering；
- 預先定義的 clean-margin strata。

## Controls

每個 primary record 使用同一 prompt、layers、positions、local scale 與 nominal dose，
配對執行：

1. **Outcome direction**：`+d` 或 `−d`。
2. **Matched random direction**：每層 same-norm seeded random direction。
3. **Label-permutation direction**：在 discovery fitting 前以 ticker 為單位 permutation
   gradient signs，重跑相同 fitting pipeline。
4. **Shuffled-evidence positions**：相同 direction/dose，位置換到 evidence span 內預先
   凍結的 shuffled control。
5. **Final-position control**：判斷效果是否只是直接干預 answer-position motor state。
6. **No-op**：`alpha = 0`，驗證 scorer、hook 與 generation determinism。

Control intervention 的 delivered perturbation norm 必須與 paired primary 記錄 compact
matching diagnostics；不能只比較 nominal alpha。

## Estimands

對 Buy steering 的 eligible clean-Sell records：

\[
R_{S\rightarrow B}^{arm}=P(D_{intervened}=Buy\mid D_{clean}=Sell, arm).
\]

對 Sell steering的 eligible clean-Buy records：

\[
R_{B\rightarrow S}^{arm}=P(D_{intervened}=Sell\mid D_{clean}=Buy, arm).
\]

Primary specificity：

\[
C_{flip}^{Buy}=R_{S\rightarrow B}^{outcome}-R_{S\rightarrow B}^{random},
\]

\[
C_{flip}^{Sell}=R_{B\rightarrow S}^{outcome}-R_{B\rightarrow S}^{random}.
\]

同時報告 reverse/error flips。先在 ticker 內聚合，再對 tickers 等權；CI 使用 paired
ticker bootstrap。若每 ticker 只有一個 eligible binary record，另報 exact paired test。

## Frozen calibration/test parameters (Draft 1)

以下參數已於 Draft 1 凍結，必須原樣寫入 V2 config schema：

- **Tie rule**：clean `M=0` 恰好成立的 records 排除於 clean-Sell 與 clean-Buy eligible
  sets，計數報告於 test summary。Near-boundary records（`0 < |M|`）全部保留在
  eligible set，另依 strata 報告；不得因 boundary proximity 剔除。
- **Local scale**：`s_{i,l} = max(‖h_{i,l,p}‖₂, 1.0)`。因 `d_l` 單位化，每個 position
  的 delivered relative perturbation 恰為 `r = |alpha| / sqrt(|L|)`。
- **Dose grid**：`r ∈ {0.05, 0.10, 0.20, 0.40}`；`alpha = r·sqrt(|L|)` 依選定 band
  計算。參考：V1 token screen 最大 delivered relative perturbation 約 0.024，840 個
  interventions 只有 1 個 sign flip，故 V2 grid 需明顯更大。
- **Safety bound**：test records 所有 positions 的 delivered relative perturbation norm
  必須 `≤ 0.50`；超出則該 record 無效並計數報告。
- **Candidate layer bands**（Qwen3.5-4B，32 layers，0-indexed）：
  `{L14–L26, L12–L28, L10–L30}`；calibration 選定其中一個。L14–L26 是 sector
  intervention 的 layer localization band，作為候選之一而非預設偏好。
- **Candidate position rules**：`{evidence item 結尾 tokens（valence-readout 規則）,
  evidence span 內全部 tokens}`；calibration 選定其中一個，gradient fitting 與
  intervention 必須使用同一規則。
- **Clean margin strata**（test 報告用）：`|M| ≤ 0.5`、`0.5 < |M| ≤ 1.5`、`|M| > 1.5`。

### Calibration selection rule

Calibration 從 candidate set 選定 `(band, position rule, r)`，並必須在 calibration
summary 記錄以下預先指定排序：

1. `r` 取滿足 safety bound 且 calibration prompts 的 full-generation parse success
   `≥ 90%` 的最大 grid 值。
2. Band 與 position rule 取 calibration paired（outcome − matched-random）target-flip
   rate 最大的組合；全部候選組合的數值都必須報告。
3. 若沒有任何組合在 calibration tickers 上使 Buy 與 Sell 兩方向各至少出現 1 個
   target flip，V2 fail closed，不進入 test。

## Success gates (frozen in Draft 1)

- Buy 與 Sell 兩個方向都要有非零 target flips，不能只挑成功的一邊。
- Outcome direction 對 matched-random 的 paired target-flip specificity CI 下界
  必須大於 0。
- 最低實用 target flip rate 為 **10%**（test primary eligible set）。
- Outcome arm 的 reverse/error flip rate 不得高於 matched-random arm；另報告
  net specificity（target − reverse）並必須為正。
- Matched-random 與 label-permutation arms 的 target-flip rate 不得達到 10%；
  shuffled-evidence 與 final-position controls 僅報告，不設獨立 gate。
- Full generation：target-flipped records 的 JSON parse success rate `≥ 90%`，且
  generated decision 與 target direction agreement `≥ 70%`。
- Multiplicity family：`{C_flip^Buy, C_flip^Sell}` 兩個 primary tests 做 Holm
  correction；secondary diagnostics 不在 family 內。因 calibration 凍結單一 band
  與單一 dose magnitude，test 不引入其他 layer/dose variants。

## Artifact and implementation boundary

V2 應使用新的 CLI/config/artifact identity，例如獨立的
`jspace-outcome-direction-flip` dataset slug；不要擴充或覆寫 V1
`jspace-token-screen` artifacts 後仍沿用相同 schema version。

實作仍遵守 shared `prepare → forward → analyze → finalize` lifecycle：

- prepare：驗證 splits、inputs、model/lens、direction-fit config 與 sequence lengths；
- forward：在記憶體 recompute direction，執行 paired interventions 與 generation；
- analyze：ticker-level flip estimands、controls、CI 與 gates；
- finalize：只註冊 compact outputs 與 provenance。

### Implementation (Draft 1, frozen values unchanged)

V2 implementation 使用獨立 identity 並遵守 shared lifecycle：

- **Dataset slug**：`jspace-outcome-direction-flip`；runs 位於
  `artifacts/<model-slug>/jspace-outcome-direction-flip/runs/`。
- **CLI**（`jspace-intervention`）：`prepare-outcome-flip-config`（產生
  `outcome_flip_config`，fitted layers 預設為 bands 的 union）、`validate-config`
  （依 `artifact_type` 派發到 `OutcomeFlipConfig`）、`run-outcome-flip`
  （`--split {discovery,calibration,test}`；calibration/test 必須提供
  `--direction-identity`，test 另需 `--calibration-selection`，否則 fail closed）。
- **Config schema**：`outcome_flip_config`（version 1）原樣存放 Draft 1 凍結值（tie
  rule、scale floor 1.0、dose grid、safety bound 0.50、candidate bands、position
  rules、margin strata、最低 flip rate 10%、generation gates 90%/70%、fitting 與
  bootstrap seeds、split manifest hash）。
- **Artifacts**：`outcome_flip_direction_identity`（discovery 輸出 layer-wise direction
  hashes 與 norms，不存向量）、`outcome_flip_result`（JSONL 逐行 compact 結果）、
  `outcome_flip_calibration_summary` 與 `outcome_flip_selection`（calibration；selection
  綁定 config 與 identity hashes）、`outcome_flip_analysis`（test；含 frozen gates）、
  `outcome_flip_metadata`。Prepare 階段另記錄 `outcome_flip_prompt_record` 與
  `outcome_flip_prepare_metadata`；discovery 的 analyze 另記錄
  `outcome_flip_fit_summary` 與 `outcome_flip_analysis_metadata`。
- **Scorer 與 fitting target**：decision-position margin 重用既有
  `score_single_token_margin_fp32`（FP32 final norm + unembedding，於 prompt 最後
  position 計分）；gradient fitting 使用同一 tail 的可微分 primitive
  `fp32_next_token_log_probs`，因此 fitting target 與 outcome scorer 是同一函數。
  `HFLensModel` wrapper 會 freeze 所有 params，frozen leaves 的 forward 本身不建
  autograd graph；fitting 因此在第一個 fitted layer 的輸出 re-root（同 jlens
  `ActivationRecorder.start_graph_at` 慣例），retained graph 恰為 fitted layers。
  Fake-model 測試的 params 亦 freeze 以覆蓋此路徑。
- **Delivered dose**：relative perturbation 報告每 position 實際 applied `‖Δh‖` 除以
  local scale；transform 按預期應用時等於 `r`，intervention 未應用時 run fail
  closed。
- **Reverse gate**：某方向的 reverse-eligible pool 為空（沒有該 clean decision 的
  records）時，reverse flip rate 記 0，「不得高於 matched-random」gate 視為 vacuously
  satisfied；net specificity gate 仍要求 target rate 為正。

Regression tests：`tests/test_jspace_outcome_flip.py`（fake model、無 GPU）。在第一次
正式 run 前，本文件對 evidence 的立場不變：不宣稱任何 run 結果。

## Direction decode（Jacobian lens readout of d_l）

**Status：已實作；第一次正式 run 已完成**（run 與結果見下方 Direction decode 結果）。Direction decode 是 V2 的輔助診斷（與 per-(layer, position) attribution screen 同類），不是 V2 protocol 的一部分：它不引入新的 estimand、control 或 gate，也不消耗 V2 的 run-once 性質。

### 問題與定義

Direction decode 問：calibration 凍結的 outcome direction $d_l$ 經 canonical Jacobian
lens 運送到 final-layer basis 後，指向哪些詞彙 token？它定義三個量：

- **transported direction logit**：對 layer $l$，
  \[
  z_l = W_U \, N(J_l d_l),
  \]
  其中 $J_l$ 是 canonical lens 的 layer-$l$ Jacobian（把 layer-$l$ residual 映射到
  final-layer basis；transport operator 與 `JacobianLens.transport` 相同，即 row-vector
  寫法 $d_l J_l^{\top}$，column 寫法 $J_l d_l$），$N$ 是 model final norm，$W_U$ 是
  LM-head weight。$z_l$ 與 $d_l$ 的 transport、final norm 與 unembedding 一律以
  float32 計算，與 `fp32_next_token_log_probs` 同一 tail（去掉 log-softmax）。$d_l$
  是 unit-norm，但 $J_l d_l$ 一般不再 unit-norm；$N$ 的 scale invariance 只保留
  方向、不保留模長，所以 $z_l$ 的模長由 $\lVert J_l d_l \rVert$ 決定，解碼結果不
  套用任何 dose 或 $\alpha$。
- **direction probability**：transported direction logit 的完整詞彙 softmax
  $p_l = \mathrm{softmax}(z_l)$。
- **direction decode**：workflow 名稱；CLI `decode-outcome-direction`，dataset slug
  `jspace-outcome-direction-decode`。Sign convention 沿用 V2：`+d` 是 Buy steering，
  `−d` 是 Sell steering；因 final norm 為奇函數、unembedding 為線性，
  $z_l(-d) = -z_l(+d)$ 精確成立，run 內以 self-check 驗證。

### Direction source 與 binding

Direction decode 不持久化、也不接受任何 direction vector：它依 V2 no-persistence
protocol 在記憶體中 deterministic recompute $d_l$（frozen discovery inputs、model、
position rule 與 aggregation config，`torch.use_deterministic_algorithms` 與
`CUBLAS_WORKSPACE_CONFIG=:4096:8` 在任何 CUDA 使用之前啟用），layer-wise direction
hash 必須與 `outcome_flip_direction_identity` 完全一致，否則 fail closed。輸入
binding 與 V2 test run 相同：`outcome_flip_selection` 綁定 identity SHA，identity
綁定 discovery input / split manifest / config 的 SHA-256 與 record set；lens 只讀取
既有 validated canonical lens，不 fitting、不修改。Decode 的 layers 與 position rule
取自 calibration selection 凍結的 band；selection 的 `relative_dose` 只作為
provenance 記錄，不參與解碼計算。

### Answer tokens 與輸出契約

Buy/Sell answer token 是 V2 的 `positive_candidate`/`negative_candidate` 在 discovery
scoring prompt 上的 single-token continuation ID（`continuation_token_ids`，suffix
必須恰為 1 個 token）；ID 必須在全部 discovery records 上唯一，否則 fail closed。

每層、每 sign 輸出 compact top-k（預設 30）token id/text/transported direction
logit/direction probability（依 logit 降序），以及 buy/sell token 的 rank、logit、
probability、logit margin $z(\mathrm{buy}) - z(\mathrm{sell})$ 與 probability
margin。Band aggregate 遵守 valence readout 的 full-softmax aggregation contract：
先對 band 內各層的完整 direction probability 等權平均（float64），再選 top-k；band
scope 的 buy/sell margin 以平均 probability 計算。

永不持久化 direction vector、raw gradients、Jacobians、activations 或 residuals；
artifacts 只含 compact top-k、rank、scalar scores、token id/text 與 provenance
hashes。

### Interpretation limits

Direction decode 是 transported representation readout of a fitted aggregate
axis，不是 chain-of-thought、discrete reasoning path 或 standalone causal claim。
Top tokens 只描述「該層 axis 指向哪些詞彙」；它不證明任何 token 因果驅動 buy/sell
偏好，也不能把 V2 的 steering 效果歸因到解碼出的詞彙。V2 Test run 1 已顯示
position specificity 不成立；decode 結果不得用來補強或推翻任何 causal 宣稱。

### Artifacts

Run root：`artifacts/<model-slug>/jspace-outcome-direction-decode/runs/<run-id>/`

```
manifest.json
prepare/
  direction_source.json        # 輸入 binding：config/identity/selection/lens SHA、選定 band/position rule/dose、answer token ids、解碼契約
  metadata.json                # outcome_direction_decode_prepare_metadata
forward/
  direction_decode.jsonl       # 每 (layer, sign) 一筆：top-k + answer token stats + direction hash
  metadata.json                # outcome_direction_decode_metadata（deterministic mode、recompute 計數、transport/unembed 契約）
analyze/
  direction_decode_summary.json  # band scope（average-before-top-k）+ per-layer answer margin 表 + interpretation label
  metadata.json                # outcome_direction_decode_analysis_metadata
```

Artifact types：`outcome_direction_decode_source`、`outcome_direction_decode_prepare_metadata`、`outcome_direction_decode`、`outcome_direction_decode_metadata`、`outcome_direction_decode_summary`、`outcome_direction_decode_analysis_metadata`。Manifest 分 `prepare`/`forward`/`analyze` 三個 stage 註冊，finalize 要求三者全部 complete。

### CLI

```bash
uv run jspace-intervention decode-outcome-direction \
  --input data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv \
  --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
  --config artifacts/qwen3.5-4b/jspace-outcome-direction-flip/config-technology-draft1.json \
  --direction-identity artifacts/qwen3.5-4b/jspace-outcome-direction-flip/runs/outcome-flip-tech-discovery-det-20260828T015605Z/forward/direction_identity.json \
  --calibration-selection artifacts/qwen3.5-4b/jspace-outcome-direction-flip/runs/outcome-flip-tech-calibration-det-20260828T015659Z/analyze/outcome_flip_selection.json \
  --model .cache/models/qwen3.5-4b \
  --run-id outcome-decode-tech-<timestamp>Z \
  [--lens artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt] \
  [--dataset jspace-outcome-direction-decode] [--artifact-root artifacts] \
  [--top-k 30] [--max-seq-len 1024]
```

單一指令完成 prepare → forward → analyze → finalize。

### Direction decode 結果

**Run**：`outcome-decode-tech-20260828T113010Z`（diagnostic status；run root：
`artifacts/qwen3.5-4b/jspace-outcome-direction-decode/runs/outcome-decode-tech-20260828T113010Z/`，
本 worktree 的 ignored `artifacts/`）。模型 `.cache/models/qwen3.5-4b`
（`CUDA_VISIBLE_DEVICES=0`），canonical lens 以絕對路徑唯讀載入（主 repo
`artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt`）。

**Binding 與 recompute**：direction 依 V2 no-persistence protocol 在記憶體
deterministic recompute（35 個 discovery tickers、21 個 fitted layers、雙
position rule），layer-wise direction hash 與
`outcome-flip-tech-discovery-det-20260828T015605Z` 的 identity 全部 21 層 × 2
rules 一致（fail-closed 未觸發）；`z(-d) = -z(d)` self-check max abs deviation
= 0.0（bit-exact）。Selection：L10–30、`evidence_item_end`、r=0.4（dose 只
記錄於 provenance，不參與解碼）。Buy/Sell answer token：`buy`=19180、
`sell`=33680。

**逐層 buy/sell 軌跡**（`+d` = Buy steering；logit margin = z(buy)−z(sell)）：

| layer | +d logit margin | +d buy rank | −d sell rank | −d sell probability |
|---|---|---|---|---|
| 10 | −0.16 | 67660 | 190822 | 8.8e-07 |
| 15 | 2.89 | 65513 | 13162 | 2.5e-06 |
| 18 | 4.66 | 47679 | 3411 | 2.4e-05 |
| 21 | 11.84 | 118 | 13 | 1.6e-03 |
| 24 | 21.84 | 1 | 1 | 0.658 |
| 26 | 23.37 | 5 | 1 | 0.822 |
| 28 | 19.48 | 36 | 1 | 0.265 |
| 30 | 7.94 | 3108 | 17 | 3.1e-04 |

（完整 21 層 × 雙 sign 的 top-k、rank、logit、probability 在
`forward/direction_decode.jsonl`；band scope 在
`analyze/direction_decode_summary.json`。）

**代表性 top tokens**：

- `+d`：L15「驚喜 / inspiring / seamlessly」、L18「uplifting / positive /
  boost」、L24「buy / Life」、L26「 life / Life / Life」；band（L10–30 等權
  full-softmax 平均）top-6 為「 life / 「 / positive / Life / _card / buy」，
  buy 平均 probability 1.05e-3（rank 6）、sell 2.8e-7。
- `−d`：L15「failed / fails / worse」、L18「枯萎 / 凋零 / 负面」、L24
  「sell / Sell / sell」（sell probability 0.66）、L26「sell / Sell / Sell」
  （0.82）；band top-1 即「sell」（平均 probability 0.202，rank 1），其次
  failed / doomed / fails / worse / futile / failure / fatal。

**觀察（非結論）**：answer token 在 L21–L29 中層帶進入 top-30（buy 於 `+d`
L23–27、L29；sell 於 `−d` L21–30），L24–26 最尖銳；L10–12 與 L30 以
format/低資訊 token 為主，margin 明顯較小。`−d` 的 sell 解碼強度（峰
probability ≈0.82）遠高於 `+d` 的 buy（峰 ≈0.005），與 V2 calibration 中 sell
方向 flip 數（8）高於 buy（4）的觀察同向。全 21 層中 sell 從未進入任何 `+d`
top-30、buy 從未進入任何 `−d` top-30（sign separation）。

**解讀限制**：Direction decode 是 transported representation readout of a
fitted aggregate axis，不是 chain-of-thought、discrete reasoning path 或
standalone causal claim；top tokens 只描述該層 axis 指向哪些詞彙，不證明任何
token 因果驅動 buy/sell 偏好，也不能把 V2 的 steering 效果歸因到解碼出的詞彙。
V2 Test run 1 已顯示 position specificity 不成立；本 decode 不補強、不推翻
任何 causal 宣稱，也不消耗 V2 的 run-once 性質。

Regression tests：`tests/test_jspace_outcome_decode.py`（fake model/lens、無 GPU）。

## Revision record

- **Draft 0**：將 V1 vocabulary-direction screen 與 V2 outcome-gradient decision-flip
  protocol 分開；把 Buy/Sell decision flip 設為 primary outcome，full generation 設為
  必要 behavioral validation。
- **Draft 1**：凍結 tie rule（`M=0` 排除出 eligible set）、local scale
  （residual norm，floor 1.0）、dose grid `r ∈ {0.05, 0.10, 0.20, 0.40}` 與 safety
  bound `r ≤ 0.50`、candidate layer bands `{L14–L26, L12–L28, L10–L30}`、candidate
  position rules（evidence item 結尾 / evidence span 全 tokens）、clean margin strata、
  calibration selection rule 與 fail-closed criterion、最低實用 flip rate 10%、
  reverse-flip gate、full-generation gates（parse success ≥ 90%、decision agreement
  ≥ 70%）與 multiplicity family（2 primary estimands，Holm）。
- **Implementation 1**：依 Draft 1 實作 V2 CLI、config schema、artifact
  schema 與 regression tests；未修改任何 Draft 1 凍結值。Implementation clarifications：
  decision margin 重用既有 single-token FP32 scorer（於 prompt 最後 position 計分），
  fitting target 用同一 tail 的可微分版本；delivered dose 改以實際 applied
  perturbation 量測；reverse gate 對空 reverse-eligible pool 視為 rate 0。第一次正式
  discovery run 尚未執行。
- **Implementation 2**：第一次 discovery smoke run（real
  `HFLensModel`）暴露 correctness bug：wrapper 在建構時 freeze 所有 params，frozen
  leaves 的 forward 不建 autograd graph，`fit_prompt_layer_gradients` 因此拿到沒有
  `grad_fn` 的 layer output 而 fail closed（`element 0 of tensors does not require
  grad`）。修復方式是在第一個 fitted layer 的輸出 `requires_grad_(True)` re-root
  （同 jlens `ActivationRecorder.start_graph_at` 慣例），retained graph 恰為 fitted
  layers；fake-model 測試的 params 同步 freeze 以把此路徑鎖進 regression tests（移除
  修復後 5 個 fitting/pipeline 測試 fail）。不修改任何 Draft 1 凍結值、estimator 或
  artifact schema。
- **Implementation 3（current）**：第一次 discovery 正式 run
  （`outcome-flip-tech-discovery-20260828T011535Z`）完成後，第一次 calibration
  嘗試（`outcome-flip-tech-calibration-20260828T011744Z`）fail closed：recomputed
  direction hash 與 discovery identity 不一致（`evidence_item_end` layer 10）。
  Diagnosis：default bf16 GEMM backward 非 deterministic，discovery identity
  本身不可 bit-exact 重現（實測同一 process 重複 fitting 的 raw gradient max abs
  差異約 5e-3，且集中在 L10–L26）。修復：`run_outcome_flip_pipeline` 在首次 CUDA
  使用前啟用 `torch.use_deterministic_algorithms(True)` 與
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`（涵蓋 fitting、scoring、generation）；啟用後
  gradient hash 實測同 process 與跨 process bit-identical，新增 regression test
  `test_enable_deterministic_gpu_sets_cublas_workspace_and_flag`。舊 discovery
  identity 因以 non-deterministic mode 計算而不可 verify，必須以 deterministic
  mode 重跑 discovery 重新 freeze。不修改任何 Draft 1 凍結值、estimator 或
  artifact schema。
- **Test run 1（第一次正式 full pipeline，2026-08-28）**：discovery
  （`outcome-flip-tech-discovery-det-20260828T015605Z`）→ calibration
  （`outcome-flip-tech-calibration-det-20260828T015659Z`，selection：L10–30、
  `evidence_item_end`、r=0.4，24 個候選組合全部 parse 100% 且雙方向有 flip）→ test
  （`outcome-flip-tech-test-20260828T051624Z`，11 個 held-out Technology tickers）。
  Verdict：**`success=false`**。Buy 方向（9 個 clean-Sell tickers）outcome 9/9、
  matched-random 0/9、label-permutation 0/9（reverse 9/9），p_one_sided=0.0015、
  Holm-adjusted 0.003，通過；sell 方向（僅 2 個 clean-Buy tickers）outcome 2/2、
  controls 0/2，但 n=2 時 exact paired test 的最小 p_one_sided 為 0.25，結構上
  無法通過 0.05 gate——失敗原因為 test split 的 sell 方向 power 不足，非 effect
  缺席。其餘 gates 全過：specificity CI [1,1]（雙方向）、min flip rate 100% ≥ 10%、
  net specificity 正、reverse not excess、controls < 10%、generation parse 100%、
  agreement 100%、safety max 0.402 ≤ 0.50。重要診斷發現：final-position control
  在 calibration 與 test 皆雙方向 100% flip（買 9/9、sell 2/2），效果可被
  answer-position injection 完全重現，position specificity 不成立；V2 的解釋上限為
  outcome-axis steering of the decision，非 evidence-position 因果。輔助診斷：
  per-(layer, position) attribution screen（
  `scripts/jspace_outcome_token_attribution.py`，
  `artifacts/qwen3.5-4b/jspace-outcome-token-attribution/`）顯示 evidence item-end
  的 per-token 一階敏感度集中在 L0–L14、L15–L18 斷崖，與 V2 band 大部分在斷崖下方
  一致；但 aggregate direction 的跨 ticker 對齊（pre_unit_norm）在 L15–L26 最高，
  故 band 選擇與 attribution 圖不矛盾。run-once 性質已消耗；若要以更大或更平衡的
  test split 重驗 sell 方向，依 experiment versioning 開新版本。
- **Direction decode 1（2026-08-28）**：新增輔助診斷 workflow
  `decode-outcome-direction`（dataset slug `jspace-outcome-direction-decode`）：以
  canonical Jacobian lens 把 frozen $d_l$ 運送到 final-layer basis，FP32 final norm +
  unembedding 解出 transported direction logit 與完整詞彙 softmax，逐層輸出 ±d 的
  compact top-k 與 buy/sell rank/margin；direction 依 V2 no-persistence protocol
  deterministic recompute 並驗證 direction identity，lens 只讀。不修改任何 Draft 1
  凍結值、V2 estimator、gate 或 artifact schema；第一次正式 run 見
  「Direction decode 結果」。
