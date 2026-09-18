# J-space 產業介入實驗：提案與凍結方法

## 文件狀態

本文件保存介入算子、資料切分、controls 與 held-out estimands。校準與 test 數值另見
[實驗報告](report.md)。

## 研究問題與主要 outcome

對產業 A 的公司 prompt，我們將產業 A 的 J-space 座標替換為產業 B 的座標，觀察模型的 Buy/Sell 分布是否朝產業 B 的方向移動。第二個實驗則放大 prompt 中已存在的產業 prototype coordinate。

主要分數定義為

\[
M = \log P(\text{buy}) - \log P(\text{sell}),
\]

介入效果為

\[
\Delta M = M_{\text{intervened}} - M_{\text{clean}}.
\]

正值代表機率朝 `buy` 移動，負值代表朝 `sell` 移動。目前 workflow 評估固定 Buy/Sell continuations，尚未在介入狀態下重新生成完整文字答案。

## 資料切分與 leakage control

資料以 ticker 為單位切分，避免同一家公司同時出現在 discovery、calibration 與 test。

| 產業 | Discovery | Calibration | Test |
|---|---:|---:|---:|
| Technology | 35 | 12 | 11 |
| Financial Services | 36 | 12 | 12 |
| Healthcare | 31 | 11 | 10 |

Split manifest：

`artifacts/qwen3.5-4b/jspace-intervention/splits.json`

Split SHA-256：

`5bf184df7963810cd3e729a75a3de40edc3a81f78b0481efd99f9ef52a06d91d`

初始 dose sweep 使用 12 個 Technology prompts 與 8 個 Financial Services prompts。完整 paired-control swap calibration 使用兩個產業各 12 個 calibration tickers。

## Workspace 與 prototype 建構

獨立 layer statistics 將 L14–L26 選為候選 workspace band。L31 只用作 motor/readout contrast，不作為介入層。

Discovery-only normalized informative-prior log-odds 選出的 primary sector concepts 如下：

| 產業 | Frozen concepts |
|---|---|
| Technology | lucrative、competitor、partnership、competition |
| Financial Services | risk、portfolio、platform、regulatory |

每一層的 sector prototype 以 frozen discovery scores 加權 canonical Jacobian-lens token directions。Workflow 只讀取 canonical lens，不重新 fitting，也不修改 lens：

`artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt`

Split 與 prototype configs 均在 calibration inference 前凍結。

## 介入算子

### Coordinate swap

令 source 與 target directions 組成 \(V=[d_s,d_t]\)，先計算

\[
c=V^\dagger h,
\]

再套用

\[
h' = h + \beta V(\sigma(c)-c),
\]

其中 \(\beta\) 對應 `swap_fraction`。`0` 是 no-op；`1` 代表在每個介入位置進行完整 coordinate swap。

Coordinate-swap operator 來自 Gurnee et al.；weighted sector prototypes、loaded-position selection，以及跨 L14–L26 的介入方式是本專案的調整。

### Coordinate gain

對方向 \(v\)，定義

\[
c = \frac{\langle v,h\rangle}{\|v\|^2}, \qquad c'=gc,
\]

並套用

\[
h'=h+v(c'-c).
\]

`gain = 1` 是 no-op。此算子保留 orthogonal complement，且不重新正規化整個 residual state。

Gain 連續作用於 13 層，前層介入會改變後層收到的 state。因此 band-wide `gain = 2` 的 delivered perturbation 遠大於單一 layer 的一次座標加倍。

## Position selection 與 loading gate

每個 prompt 先執行 clean forward，再從 evidence span 選出 source prototype loading 最高的位置。介入會在這些位置跨 L14–L26 執行。本文使用的 calibration prompts 全部通過 loading gate。

Controls 包括：

- evidence span 內的 shuffled positions；
- prompt final position；
- 保留原 Gram geometry 的 matched-random directions。

## Diagnostics 與 artifact policy

每一列結果只保存 compact scalar diagnostics：

- 介入前後的 coordinates；
- absolute 與 relative perturbation norm；
- next-token KL divergence；
- direction condition number；
- loading-gate status；
- paired-primary target norm 與 dose error。

Workflow 不保存 raw activations、residuals、hidden states、gradients、Jacobians 或 KV caches。

所有 no-op rows 的 margin change、KL 與 delivered perturbation 均為 0。

## Held-out estimands（test inference 前凍結）

Held-out test 將 direction specificity 與 position specificity 視為兩個獨立 estimands，不以其中一項替代另一項。

先對任一 condition \(c\) 定義雙向效果

\[
E(c)=\tfrac12[\Delta M_{F\rightarrow T}(c)-\Delta M_{T\rightarrow F}(c)].
\]

### Estimand 1：direction specificity

固定 intervention positions 為 loaded evidence，對比 sector prototype 與 matched-random direction：

\[
C_{direction}=E(\text{sector, evidence})-E(\text{random, evidence}).
\]

\(C_{direction}>0\) 才支持 sector-direction specificity。Calibration descriptive estimate 為 \(0.0521-0.0208=0.0313\)。

### Estimand 2：position specificity

固定 direction 為 sector prototype，對比 loaded evidence 與 final position：

\[
C_{position}=E(\text{sector, evidence})-E(\text{sector, final}).
\]

\(C_{position}>0\) 才支持 evidence-position specificity。Calibration descriptive estimate 僅為 \(0.0521-0.0469=0.0052\)，因此 held-out 很可能不支持此 claim；仍需按預先定義完整報告。Sector-prototype shuffled-evidence contrast 作為 secondary position control，不改變 primary position estimand。

兩個 estimands 將分別報告 ticker-clustered paired bootstrap interval。Direction specificity 成立時，不自動宣稱 position specificity；position specificity 未成立時，也不抹除 direction contrast 的結果。

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**

- [baseline-trial](../baseline-trial/proposal.md)（資料／產物依賴）：使用 trial-plan CSV 的公司與結構化證據。
- [jacobian-lens-selection](../jacobian-lens-selection/proposal.md)（資料／產物依賴）：產業座標介入使用 canonical lens 的投影。

**後續**

- [jspace-valence-readout](../jspace-valence-readout/proposal.md)（資料／產物依賴）：共用已固定的 ticker split manifest，不重分 discovery/calibration/test。
- [span-sensitivity](../span-sensitivity/proposal.md)（資料／產物依賴）：共用 split manifest；以 header 表面替換建立行為基準，不依賴介入成功。
- [jspace-token-experiments](../jspace-token-experiments/report.md)（資料／產物依賴）：V1/V2 沿用 split manifest；各版本保留獨立 config 與 gate。
- [sector-context-followup](../sector-context-followup/proposal.md)（資料／產物依賴）：A/B/C 沿用 Technology 與 Financial Services split manifest。
- [entity-cell-localization](../entity-cell-localization/report.md)（資料／產物依賴）：E1 V1 沿用 split manifest；不是以 sector intervention verdict 為 gate。

最終／最新結果見本研究的 [report](report.md)。
