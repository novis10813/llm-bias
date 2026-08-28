# J-space outcome-conditioned decision-flip V2

## Status and scope

**Status：Draft 1 已凍結、已實作；第一次正式 pipeline（discovery → calibration → test）已完成，test verdict 為 `success=false`**（唯一失敗 gate：sell 方向的 Holm-adjusted specificity，原因為 test split 的 sell-eligible tickers 只有 n=2，統計上不可能顯著，見下方 Revision record 的 Test run 1）。已完成的
vocabulary-direction screen 是 [V1](jspace-token-causal-screen-v1.md)，不能把 V1
results 當成 V2 evidence。版本入口見 [J-space token experiment versions](jspace-token-causal-screen.md)。
本文件另定義 readout-only 的 zero-evidence header-only prior probe；其第一次
正式 run 已完成，結果記錄在該節的 First probe run 段落。

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

Regression tests：`tests/test_jspace_outcome_flip.py`（fake model、無 GPU）。
Run 結果只記錄在 Revision record 與下方 prior probe 段落，不在其他位置覆寫。

## Zero-Evidence Header-Only Prior Probe（V2 先驗探針）

**定義（本文檔新術語，以下各處沿用）**：「zero-evidence header-only prior
probe」（零證據標頭先驗探針）是 V2 的 readout-only 診斷：evidence body 固定為
單一中性/空白 item，只改 header 的 Ticker 與 Sector；於 scoring prompt 最後
position 逐層量測 residual 在 frozen V2 outcome direction $d_l$ 上的投影，
以及同 position 的 Buy-vs-Sell logit margin。探針不做任何 intervention、沒有
success gate；結果只支持受測 prompt/template/model 與 frozen direction 下的
association/prior evidence，不能直接宣稱 entity 本身的 causal proof。

### 目的

V2 intervention run 證明 outcome axis 可以在 evidence prompts 上 steering
decision；prior probe 問的是反向問題：在完全沒有 evidence 時，header 的
ticker 與 sector 本身會在 frozen outcome axis 上留下多大的 prior。這是
association 層觀察，用於判斷 header identity 是否與 outcome axis 有可量測的
關聯，以及該關聯集中在哪些 layers。

### Frozen template（version 1）

```text
Refer to the evidence below to make a final investment decision.
Stock Ticker: [{ticker}]
Sector: [{sector}]
--- Evidence ---
1. No evidence provided.
---
Respond with one valid JSON object containing only the keys "decision" (buy | sell) and "reason" (brief justification). Do not choose hold.
```

與 canonical data template 相比只有兩處差異：`Stock Name:` 行換成
`Sector:` 行（本探針的變因是 ticker 與 sector，公司名不作為獨立變因）；
evidence block 換成單一 frozen 中性 item `No evidence provided.`。中性 item
在所有 conditions 完全相同，不含任何方向性數字、valence 詞或 identity 資訊。

### Direction source 與 verification

Probe 重用 V2 frozen discovery direction identity
（`outcome-flip-tech-discovery-det-20260828T015605Z` 的
`forward/direction_identity.json`）與對應的 frozen `outcome_flip_config`；probe
config 以 path + SHA-256 同時綁定兩份檔案，run 前驗證：identity 的 model、
input SHA、split manifest SHA 與 config SHA 都與 probe inputs 一致，否則 fail
closed。每次 probe run 都依 V2 的 no-persistence 規則在記憶體 deterministic
recompute direction 並要求 layer-wise hash 與 frozen identity 完全一致（同
`verify_direction_identity` fail-closed 語義）。Projection 使用 position rule
`evidence_item_end`（calibration frozen 選擇的 rule）；$d_l$ 已 unit
normalize，故 projection 即內積。

### Measurement（全部於 scoring prompt 最後 token position）

Measurement position 定義為 formatted probe prompt 加 decision prefix 之後的
最後 token position，與 V2 scorer 讀 next-token 分佈的位置相同：

- **Projection**：$\text{projection}_l = \langle h_{l,p^*}, d_l \rangle$
  （FP32 內積；$\|d_l\|_2 = 1$，所以內積即投影長）。
- **Relative projection**：$\text{projection}_l / \max(\|h_{l,p^*}\|_2, 1.0)$，
  分母是 V2 frozen local scale（floor 1.0），無量綱；residual 與 $d_l$ 完全
  同向時為 $+1$，可直接對照 V2 delivered relative perturbation $r$ 的單位。
- **Margin**：$M = \log P(\mathrm{buy}) - \log P(\mathrm{sell})$，重用
  `score_single_token_margin_fp32`（FP32 final norm + unembedding），與 V2
  primary scorer 同一函數。decision 依 V2 tie rule 記 buy / sell / tie
  （$M=0$ 記 tie）。

### Conditions（第一次 run 的 frozen matched design）

6 個 tickers × 2 個 sector labels 的完整 cross，共 12 個 conditions；每個
ticker 同時出現於兩個 sector label，每個 sector label 同時配對 6 個 tickers。
6 個 tickers 都不在 direction fit 的 35 個 Technology discovery tickers 內。

| Ticker | Canonical sector | Split | Header sector labels |
|---|---|---|---|
| NVDA | Technology | calibration | Technology、Financial Services |
| AAPL | Technology | calibration | Technology、Financial Services |
| INTC | Technology | test | Technology、Financial Services |
| JPM | Financial Services | discovery | Technology、Financial Services |
| WFC | Financial Services | test | Technology、Financial Services |
| MS | Financial Services | test | Technology、Financial Services |

Sector label 使用 canonical CSV 的原始標籤（`Technology`、`Financial
Services`）。own-sector cell（ticker 的 canonical sector）與 cross-sector
condition 都在 run 中標記。

### Analysis（descriptive，無 gate）

- 每 condition 的 raw margin、per-layer projection 與 relative projection
  存於 forward artifact。
- Group means：sector label、ticker、ticker group（Technology vs Financial
  Services tickers）的 mean margin 與 per-layer mean projection。
- Contrasts：sector-label difference（每 ticker 先做 Technology − Financial
  Services 差值再對 6 個 tickers 平均，附 per-ticker paired bootstrap 95% CI，
  seed/samples 凍結於 probe config）；NVDA − JPM（對 2 個 sector labels 平均，
  只報點估計）；ticker-group difference。

### Interpretation limits

- 結果是受測 template、model 與 frozen direction 下的 association/prior
  evidence；沒有 intervention 或 control arm，不能宣稱 entity、ticker 或
  sector 對 decision 的 causal effect。
- Probe header 用 `Sector:` 取代 `Stock Name:`，與 discovery prompts 的
  template 不完全相同；header 的 token 序列因此不保證在 discovery corpus 中
  出現過。
- Direction $d_l$ 由 Technology discovery prompts fitting：sector label
  `Technology` 對 direction source 屬部分 in-sample，`Financial Services` 與
  全部 6 個 tickers 屬 out-of-sample；兩種 status 必須在解讀時分開陳述。
- 12 個 conditions 是 descriptive sample，不是 powered hypothesis test；
  contrast 的 CI 只描述 sampling variability，不構成 significance gate。

### Artifacts 與 CLI

- **Config**：`outcome_prior_probe_config`（version 1），綁定 model、input
  CSV、split manifest、`outcome_flip_config`、direction identity 五份 frozen
  inputs 的 path + SHA-256，以及 frozen conditions、pair contrast tickers
  （預設 `NVDA:JPM`）、sector-label contrast 方向（預設
  `Technology:Financial Services`，即 Technology − Financial Services）、
  position rule、中性 item、scale floor 與 bootstrap seeds。
- **Run**：`artifacts/<model-slug>/jspace-outcome-direction-flip/runs/
  outcome-flip-prior-probe-*/`（V2 dataset identity 下的 readout-only run，
  與 V2 intervention runs 以 artifact type 與 run-id prefix 區分）。
- **Artifacts**：`outcome_prior_probe_record`（prepare；每 condition 的 raw
  probe prompt 與 identity 欄位）、`outcome_prior_probe_result`（forward；每
  condition 一行 compact projection/margin，無 raw states）、
  `outcome_prior_probe_analysis`（analyze；group means 與 contrasts）、
  `outcome_prior_probe_metadata`。不保存 raw activations、residuals、
  gradients、Jacobians 或 KV caches。
- **CLI**（`jspace-intervention`）：`prepare-prior-probe-config`（驗證
  conditions 存在於 input CSV 與 sector 為 canonical label，產生 frozen
  config）、`run-prior-probe`（prepare → forward → analyze → finalize；
  direction recompute + hash verification 後逐 condition 量測）。
- **Regression tests**：`tests/test_jspace_prior_probe.py`（fake model、無
  GPU；覆蓋 template freezing、config schema、projection normalization、
  fail-closed verification 與完整 lifecycle）。

### First probe run（`outcome-flip-prior-probe-20260828T110726Z`，2026-08-28）

第一次正式 probe run，12 個 frozen conditions（6 tickers × 2 sector labels），
model `.cache/models/qwen3.5-4b`，direction 用 frozen discovery identity
（`evidence_item_end`，L10–L30 全部 21 層 hash 驗證通過，
`direction_verified=true`）。scoring prompt 固定 91 tokens；residual norm 範圍
6.6–56.7。Run 位置：`artifacts/qwen3.5-4b/jspace-outcome-direction-flip/runs/
outcome-flip-prior-probe-20260828T110726Z/`（status complete）。

主要數值結果（association-only）：

- **Margin**：12 個 conditions 全部 clean-sell，M ∈ [−4.93, −3.39]。Sector
  label means：Technology −4.021、Financial Services −4.372；ticker means：
  JPM −3.854、NVDA −3.895、AAPL −3.878、WFC −4.228、INTC −4.494、MS −4.828；
  ticker-group means：Technology tickers −4.089、Financial Services tickers
  −4.303。Zero-evidence template 在該 model 下有強 sell prior。
- **Sector-label contrast（Technology − Financial Services，per-ticker paired
  bootstrap，n=6）**：M point +0.351，95% CI [−0.019, +0.736]（含 0，弱且
  未決）；per-layer projection_relative 全部 |point| ≤ 0.0035，只有 L18 的 CI
  不含 0（+0.0029 至 +0.0042），magnitude 可忽略。
- **NVDA − JPM contrast**：M point −0.042（≈0）；per-layer
  projection_relative |point| ≤ 0.001。
- **Projection 的 template-shared 結構**（12 個 conditions 幾乎一致，非
  identity effect）：L10 ≈ +0.046、L15–L18 負（最深 ≈ −0.049）、L23–L25 正
  （最高 ≈ +0.043）、L30 ≈ −0.133（相對值）。L30（最後 decoder layer，緊接
  final norm + unembedding）的強負 projection 與 sell-leaning margin 一致。

解讀：在受測 template/model/frozen direction 下，header 的 ticker 與 sector
identity 在 decision position 的 frozen V2 outcome axis 上沒有可量測的
association（所有 per-ticker contrast 的 relative projection 皆 ≤ 0.4% 的
residual norm）；每個 condition 的大讀數（sell-leaning margin 與逐層
projection pattern）是 12 個 conditions 共有的 template-level 結構，不是
entity-specific effect。本 run 不構成任何 entity 或 sector 的 causal claim；
若未來要检验 header identity 的 causal role，需要另建 intervention 設計
（新 experiment version）。此 run 為 first formal probe run；重跑或改設計前
必須先記錄於 Revision record。

## Revision record

- **Probe protocol（Zero-Evidence Header-Only Prior Probe）**：在 V2 文件新增
  readout-only prior probe 協議：frozen template version 1（header 的
  `Stock Ticker:`/`Sector:` + 單一中性 evidence item `No evidence provided.`）、
  frozen discovery direction identity 的 in-memory recompute + hash
  verification（fail closed）、final-position projection（unit direction 的
  FP32 內積）與 V2 local-scale relative projection、V2 同函數 margin scorer、
  12-condition matched design（6 tickers × 2 sector labels）、descriptive
  analysis（group means、sector-label / pair / ticker-group contrasts、paired
  bootstrap CI）與 interpretation limits（association-only）。新增
  `prepare-prior-probe-config` / `run-prior-probe` CLI、`outcome_prior_probe_*`
  artifacts 與 `tests/test_jspace_prior_probe.py`。不修改任何 Draft 1 凍結值、
  V2 estimator 或既有 artifact schema。
- **Probe run 1（第一次正式 probe run，2026-08-28）**：
  `outcome-flip-prior-probe-20260828T110726Z`（12 conditions，direction
  verification 通過）。結果：全部 12 個 conditions clean-sell（M ∈ [−4.93,
  −3.39]）；sector-label contrast M +0.351（95% CI 含 0）、NVDA − JPM
  M −0.042；header identity 對 frozen outcome axis 的 per-ticker projection
  contrast 全部 |relative| ≤ 0.0035，無可量測的 identity association；大的
  逐層 readout（含 L30 ≈ −0.133 relative）為 12 個 conditions 共有的
  template-level 結構。解讀上限為受測 template/model/direction 下的
  association/prior evidence，非 entity causal proof。
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
