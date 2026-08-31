# J-space 產業介入實驗：校準與 held-out 報告

## 文件狀態

本文件整理 Qwen3.5-4B 在 Technology（科技）與 Financial Services（金融服務）之間的 J-space 介入校準與 held-out test。Test tickers 已依預先凍結的 estimands 執行；結果未支持 sector-direction specificity 或 evidence-position specificity。完整文字生成檢查仍未完成。

實驗顯示：在候選 workspace layer band 施加小幅介入會改變固定 continuation `buy` 與 `sell` 的機率差，但 controls 與 held-out contrasts 未證明效果只來自指定產業座標與 evidence positions。

## 摘要

- 獨立 layer localization 將 **L14–L26** 選為候選 workspace band。
- 初始 dose sweep 中，`swap_fraction = 0.5` 的雙向 coordinate swap 效果最大。
- Sector-prototype gain 的安全候選上限為 `gain = 1.10`。
- `gain ≥ 1.2` 與完整 swap 的多層累積擾動過大，因此不納入主要 held-out test。
- 安全劑量會改變 buy/sell margin，但目前沒有任何 prompt 發生 Buy ↔ Sell preference sign flip。
- v3 paired-primary controls 已把 absolute perturbation norm 的平均誤差控制在 0.6% 以下。
- Final-position 與其他 controls 也會改變 margin，因此目前尚未隔離出 evidence-position-specific sector effect。

## Dose calibration

![Swap 與 gain 的 dose calibration](../assets/jspace-sector-intervention/dose_calibration.png)

**圖 1。** 左圖顯示 coordinate swap 的雙向 margin response。`swap_fraction = 0.5` 時，兩個方向的效果符號相反，雙向估計量 \(E\) 達到 dose sweep 中的最大值。中圖顯示 gain 對 margin 的影響；兩個產業 prototype 的方向並不符合單純由 baseline buy rate 推導的假說。右圖顯示 gain 在 13 層累積後的 relative perturbation；`gain ≥ 1.2` 已超過圖中的 5% 參考線。

### Gain safety calibration

| Gain | Mean relative perturbation | Approx. maximum | 決定 |
|---:|---:|---:|---|
| 1.00 | 0% | 0% | no-op |
| 1.05 | 1.0% | 1.2% | low dose |
| 1.10 | 2.5% | 2.9% | selected upper dose |
| 1.20 | 7.8–8.0% | 8.8% | 排除 |
| 1.25 | 12.1–12.5% | 13.5% | 排除 |
| 1.50 | 約 44% | 約 45% | 排除 |
| 2.00 | 約 97% | 約 97% | 排除 |

在 `gain = 1.10` 時：

| Prototype | Tickers | Mean ΔM | Ticker-bootstrap 95% CI | Nonzero ΔM | Sign flips |
|---|---:|---:|---:|---:|---:|
| Technology | 12 | −0.0208 | [−0.0521, 0.0000] | 3/12 | 0 |
| Financial Services | 8 | +0.0156 | [0.0000, 0.0469] | 2/8 | 0 |

效果方向不支持簡單的 sector baseline-rate hypothesis：放大 Technology prototype 讓 margin 下降，放大 Financial Services prototype 則讓 margin 上升。這組 calibration 結果不足以支持 confirmatory gain claim。

### Primary swap dose calibration

雙向估計量定義為

\[
E=\tfrac12[\Delta M(\text{Financial}\rightarrow\text{Technology})-\Delta M(\text{Technology}\rightarrow\text{Financial})].
\]

| Swap fraction | Technology → Financial ΔM | Financial → Technology ΔM | Bidirectional E |
|---:|---:|---:|---:|
| 0.25 | −0.0104 | +0.0156 | +0.0130 |
| 0.50 | −0.0417 | +0.0469 | +0.0443 |
| 0.75 | 約 0 | +0.0313 | +0.0156 |
| 1.00 | −0.0104 | +0.0313 | +0.0208 |

`swap_fraction = 0.5` 的 mean relative perturbation 約為 1.4–1.5%，prompt-level maximum 為 3.4%。完整 swap 在兩個方向分別達到 10.3% 與 19.3%。因此 calibration 選擇 `0.5` 作為 primary candidate，並保留 `0.25` 作為 low-dose arm。

安全劑量下沒有 prompt 跨越 zero-margin boundary。

## Paired-primary control calibration

早期 local dose matching 會因 control path 改變後續 layers 的輸入而失準。v3 workflow 針對每個 prompt 與 dose 先執行 paired primary forward，記錄每一層的 live primary delta norm，再將 controls 重新縮放至這組固定 norms。

完整 control calibration 每個方向產生 144 rows：

\[
12\ \text{tickers}\times2\ \text{doses}\times2\ \text{direction conditions}\times3\ \text{position conditions}.
\]

在 `swap_fraction = 0.5` 時：

| 方向 | Mean dose error | Maximum dose error |
|---|---:|---:|
| Technology → Financial | 0.53% | 3.38% |
| Financial Services → Technology | 0.36% | 1.39% |

剩餘誤差主要來自 BF16 rounding。所有 control arms 的 absolute delivered perturbation norms，在兩個方向分別約為 2.01 與 1.97。

![Paired-primary dose matching](../assets/jspace-sector-intervention/dose_matching.png)

**圖 2。** 左圖比較每列 control 的 paired-primary target norm 與實際 delivered norm；點大致落在虛線 \(y=x\) 上。右圖顯示 BF16 後的相對 dose error，多數 rows 低於 1%。這項檢查支持後續 control effect comparison 使用近似相同的 absolute perturbation dose。

### Technology → Financial

| Direction condition | Position condition | Mean ΔM | Nonzero ΔM | Sign flips |
|---|---|---:|---:|---:|
| sector prototype | loaded evidence | −0.0417 | 5/12 | 0 |
| sector prototype | shuffled evidence | −0.0208 | 2/12 | 0 |
| sector prototype | final position | −0.0521 | 7/12 | 0 |
| matched random | loaded evidence | −0.0208 | 2/12 | 0 |
| matched random | shuffled evidence | −0.0313 | 7/12 | 0 |
| matched random | final position | −0.0313 | 5/12 | 0 |

### Financial Services → Technology

| Direction condition | Position condition | Mean ΔM | Nonzero ΔM | Sign flips |
|---|---|---:|---:|---:|
| sector prototype | loaded evidence | +0.0625 | 9/12 | 0 |
| sector prototype | shuffled evidence | +0.0208 | 9/12 | 0 |
| sector prototype | final position | +0.0417 | 7/12 | 0 |
| matched random | loaded evidence | +0.0208 | 4/12 | 0 |
| matched random | shuffled evidence | +0.0313 | 6/12 | 0 |
| matched random | final position | +0.0104 | 8/12 | 0 |

![Primary 與 dose-matched controls](../assets/jspace-sector-intervention/control_comparison.png)

**圖 3。** 藍色與橘色 bars 分別是兩個介入方向的 mean \(\Delta M\)，綠色為雙向估計量 \(E\)。Sector prototype 加在 loaded evidence positions 時得到最大的雙向估計量，但 sector-final control 的 \(E\) 也相當接近。Matched-random controls 較小，提供部分 direction specificity evidence；final-position control 則削弱 evidence-position specificity claim。

各 condition 的 descriptive bidirectional effects 如下：

| Condition | Bidirectional E |
|---|---:|
| sector prototype, loaded evidence | +0.0521 |
| sector prototype, shuffled evidence | +0.0208 |
| sector prototype, final position | +0.0469 |
| matched random, loaded evidence | +0.0208 |
| matched random, shuffled evidence | +0.0313 |
| matched random, final position | +0.0208 |

Primary loaded-evidence effect 大於多數 controls，尤其在 Financial Services → Technology 方向。然而它沒有大於所有 controls：Technology → Financial 的 sector-final effect 比 loaded-evidence effect 更大。Calibration sample size 不大，而且各 arms 共用 prompts，因此這些 contrasts 目前只作 descriptive comparison。

## Source removal 與 target installation decomposition

在 calibration tickers 上，固定 `swap_fraction = 0.5`，將 full swap 拆成 source-removal component 與 target-installation component。兩個 component 不是可加總的 causal effects；後續 layers 會收到不同 state，因此 decomposition 只作 pathway diagnostic。

| 方向 | Source removal ΔM | Target installation ΔM | Full swap ΔM |
|---|---:|---:|---:|
| Technology → Financial Services | −0.0208 | −0.0313 | −0.0417 |
| Financial Services → Technology | +0.0208 | +0.0521 | +0.0625 |

兩個方向中，target installation 的絕對效果都大於 source removal。Financial Services → Technology 的 full-swap effect 主要來自 target installation；Technology → Financial Services 也呈現相同方向，但兩個 component 的差距較小。所有 decomposition rows 都沒有 Buy/Sell sign flip。

這項結果支持將後續 held-out 的主要 interpretation 聚焦在 target installation contribution，但不允許把 target installation component 視為與 full swap 相同的 estimand。

## 結果解讀

目前結果支持三項較窄的敘述：

1. L14–L26 的小幅 sector-prototype manipulation 會改變 Buy/Sell probability margin。
2. Half swap 在 calibration subset 呈現預期符號的 bidirectional effect。
3. Sector directions 相較 matched-random directions 顯示部分 specificity。

目前不能主張 effect 只存在於 loaded evidence positions。Sector prototype 加在 final position 時，也會產生接近 primary intervention 的雙向 effect。因此現有證據較符合「L14–L26 對 sector-prototype manipulation 敏感」，尚不足以證明「evidence positions 中的 sector workspace coordinate 專門造成決策偏差」。

安全介入沒有改變任何 prompt 的離散 Buy/Sell preference；目前觀察到的是 probability margin shift。

## Held-out 執行狀態

1. Calibration decomposition 已完成。
2. Frozen v3 held-out ticker test 已完成，結果見下節。
3. 介入狀態下的完整生成格式與實際 Buy/Sell decision 檢查仍待執行。

Individual-token gain 維持 exploratory status，不取代 sector-prototype primary estimand。

## Held-out test 結果

Held-out test 使用 test ticker：Technology 11 個、Financial Services 12 個；固定 `swap_fraction = 0.5`，並在 inference 前凍結本文件定義的兩個 estimands。

### 各 condition 的 held-out mean ΔM

| Direction condition | Position condition | Technology → Financial | Financial Services → Technology |
|---|---|---:|---:|
| sector prototype | loaded evidence | −0.0114 | +0.0104 |
| matched random | loaded evidence | −0.0341 | +0.0104 |
| sector prototype | shuffled evidence | −0.0227 | 約 0 |
| sector prototype | final position | −0.0455 | +0.0938 |
| matched random | final position | +0.0341 | −0.0208 |

各 primary evidence mean 的 ticker-bootstrap 95% CI：

- Technology → Financial：`[−0.0568, 0.0341]`
- Financial Services → Technology：`[−0.0208, 0.0417]`

兩個 frozen estimands 的 descriptive estimates：

| Estimand | Held-out estimate | Calibration estimate | 結論 |
|---|---:|---:|---|
| Direction specificity \(C_{direction}\) | −0.0114 | +0.0313 | 不支持 |
| Position specificity \(C_{position}\) | −0.0587 | +0.0052 | 不支持 |

Held-out 的 primary sector-evidence effect 沒有優於 matched-random evidence control，因此 direction-specificity claim 沒有通過。Sector prototype 的 final-position effect 在 Financial Services → Technology 方向更大，position-specificity contrast 也呈現反方向。這表示 calibration 中看到的 sector/evidence pattern 沒有泛化到 test tickers。

Held-out margin sign flip 只有 1 個，出現在 Technology → Financial 的 sector final-position control；兩個 primary evidence conditions 都沒有 sign flip。這仍是固定 continuation margin 結果，不等同於完整文字生成決策翻轉。

因此目前最保守的整體結論是：Qwen3.5-4B 在 L14–L26 對小幅 residual/J-space manipulation 有 margin sensitivity，但本 pilot 沒有在 held-out ticker 上證明 sector-direction specificity，也沒有證明 evidence-position specificity。

## Reproducibility

Primary dose runs：

- `jspace-v2-swap-logodds-tech-fin-20260826`
- `jspace-v2-swap-logodds-fin-tech-20260826`
- `jspace-v2-gain-fine-logodds-tech-20260826`
- `jspace-v2-gain-fine-logodds-financial-20260826`

Complete paired-control runs：

- `jspace-v3-control-full-swap-tech-fin-20260826`
- `jspace-v3-control-full-swap-fin-tech-20260826`

Run root：

`artifacts/qwen3.5-4b/jspace-intervention-calibration/runs/`

圖表重建：

```bash
uv run python scripts/render_jspace_intervention_report_figures.py
```

相關 commits：

- `5143c2a`：causal intervention workflow
- `e7c1bb5`：coordinate gain 與 live dose diagnostics
- `4309f00`：bounded deterministic control seeds
- `32ca399`：local control dose matching
- `74453c0`：paired-primary control doses 與 result schema v3
- `75f2d80`：paired forward 的 model adapter compatibility

v3 implementation 驗證結果：

- 205 tests passed；
- Python compilation passed；
- package build passed；
- lockfile 與 JavaScript syntax checks passed。
