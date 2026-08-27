# J-space outcome-conditioned decision-flip V2

## Status and scope

**Status：protocol draft、not implemented、no evidence。** 本文件先凍結研究問題與
版本邊界；目前沒有 V2 CLI、config schema、artifact schema 或正式 run。已完成的
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
一致，否則 fail closed。

## Intervention

對 frozen layer band \(L\) 與 positions \(P_i\)：

\[
h'_{i,l,p}=h_{i,l,p}+\alpha\frac{s_{i,l}}{\sqrt{|L|}}d_l,
\]

其中 \(s_{i,l}\) 是 clean prompt 的 local scale。`alpha > 0` 對應 Buy steering，
`alpha < 0` 對應 Sell steering。Local scale definition、floor、maximum relative
perturbation norm 與 allowed doses 在 calibration 前寫入 config；test 不調整。

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

## Proposed success gates to freeze before implementation

以下是 draft gates，不是已凍結 threshold。實作與 calibration 前必須在 revision record
中確定數值：

- Buy 與 Sell 兩個方向都要有非零 target flips，不能只挑成功的一邊。
- Outcome direction 的 paired target-flip specificity CI 下界必須大於 0。
- Target flip 必須高於預先指定的最低實用率；目前候選值為 10%。
- Reverse/error flip 不得與 target flip 同步增加到抵銷 net benefit。
- Matched-random、label-permutation 與 position controls 不得重現 primary effect。
- Fixed-choice flips 必須達到預先指定的 full-generation agreement 與 format-success gate。
- Dose 必須通過 frozen relative-perturbation safety bound。

Holm 或其他 multiplicity correction 的 family 取決於最終 layer/dose variants；必須在
第一次 calibration run 前定義。

## Artifact and implementation boundary

V2 應使用新的 CLI/config/artifact identity，例如獨立的
`jspace-outcome-direction-flip` dataset slug；不要擴充或覆寫 V1
`jspace-token-screen` artifacts 後仍沿用相同 schema version。

實作仍遵守 shared `prepare → forward → analyze → finalize` lifecycle：

- prepare：驗證 splits、inputs、model/lens、direction-fit config 與 sequence lengths；
- forward：在記憶體 recompute direction，執行 paired interventions 與 generation；
- analyze：ticker-level flip estimands、controls、CI 與 gates；
- finalize：只註冊 compact outputs 與 provenance。

在 CLI、schema、tests 與 calibration freeze 完成前，本文件只是一份 V2 protocol draft，
不能宣稱已執行或已驗證。

## Revision record

- **Draft 0（current）**：將 V1 vocabulary-direction screen 與 V2 outcome-gradient
  decision-flip protocol 分開；把 Buy/Sell decision flip 設為 primary outcome，full
  generation 設為必要 behavioral validation。尚未凍結 tie rule、dose、minimum useful
  flip rate、generation agreement threshold 或 multiplicity family。
