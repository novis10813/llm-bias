# Sector and Context Follow-up B V1: Confirmation Report

## Status

B V1 calibration 已完成，verdict 為 `success=false`，`test_authorized=false`。依
[proposal](proposal.md#b-v1-frozen-confirmation-design) 的 frozen rule，held-out test 未執行。

- Calibration run：
  `artifacts/qwen3.5-4b/cross-sector-context-overriding/runs/cross-sector-context-calibration-20260831T092659Z`
- Prepared input：
  `artifacts/qwen3.5-4b/cross-sector-header-patching/prepared/calibration_pairs_v1.jsonl`
- Frozen config：
  `artifacts/qwen3.5-4b/cross-sector-context-overriding/configs/b-v1-confirmation-v1.json`
- Confirmation artifact：
  `artifacts/qwen3.5-4b/cross-sector-context-overriding/runs/cross-sector-context-calibration-20260831T092659Z/analyze/confirmation.json`
- Split：`calibration`
- Eligible identity pairs：12
- Model：Qwen3.5-4B（`.cache/models/qwen3.5-4b`）
- Layers：L14–L21
- Spans：`instruction_context`、`header`、`final_position`
- Manifest：`complete`

## Frozen Gate Results

| Gate | Observed | Pass |
|---|---:|:---:|
| Minimum eligible pairs | 12 ≥ 8 | Yes |
| Self-source exact no-op | all ΔM = 0 | Yes |
| L16 context Toward-Source mean | +0.17805 > 0.10 | Yes |
| L16 Context–Header contrast | +0.17690 > 0.10 | Yes |
| Context bootstrap 95% CI lower bound | +0.08417 | Yes |
| Context–Header bootstrap 95% CI lower bound | +0.06918 | Yes |
| Cross-sector context \|ΔM\| > same-sector peer context \|ΔM\| | 0.22870 < 0.35662 | **No** |
| Both evidence-origin strata positive | Technology +0.14159; Financial Services +0.21451 | Yes |
| Context exact sign-flip test | raw p = 0.00220 | Yes |
| Context–Header exact sign-flip test | raw p = 0.00488 | Yes |
| Holm correction across two tests | adjusted p = 0.00439 / 0.00488 | Yes |

Calibration 通過 10/11 gates。唯一失敗項目是 same-sector peer specificity。

## Primary Estimates

### L16 Cross-Sector Context Toward-Source Effect

- Equal-pair mean：`+0.17805`
- Pair-bootstrap 95% CI：`[+0.08417, +0.27129]`
- One-sided exact pair sign-flip p：`0.002197`
- Holm-adjusted p：`0.004395`

在 fixed negative evidence 下，跨產業抽換 L16 `instruction_context` state 仍穩定地把
target margin 往 source clean margin 移動。效應較 discovery 的 `+0.31806` 小，但通過
預先凍結的 effect、CI 與統計 gates。

### L16 Context–Header Contrast

- Equal-pair mean：`+0.17690`
- Pair-bootstrap 95% CI：`[+0.06918, +0.27631]`
- One-sided exact pair sign-flip p：`0.004883`
- Holm-adjusted p：`0.004883`

L16 context patching 仍顯著超過同層 header patching。這與 discovery 中「中期 header
state 已接近 no-op，而 post-evidence context state 保留 decision-relevant sufficiency」
的 layer/span localization 一致。

## Specificity Failure

Calibration 的 equal-pair absolute effects：

- Cross-sector L16 context：`|ΔM| = 0.22870`
- Same-sector peer L16 context：`|ΔM| = 0.35662`

Same-sector peer context replacement 的 absolute effect 比 cross-sector replacement 大
`0.12792`。因此 B V1 calibration 不能把 L16 context effect 解讀為 sector-specific
sufficiency。結果支持 broader identity-conditioned 或 company-conditioned context-state
sensitivity；現有 controls 無法把它收窄成 sector-conditioned mechanism。

Pair-level 結果也顯示 heterogeneity。一個 pair 的 Toward-Source effect 為負，另有多個
pairs 的 cross-sector effect 很小，而 same-sector peer replacement 保持較大的 absolute
perturbation。這不是單一 outlier 造成的 threshold 邊界失敗：aggregate peer control
明顯高於 cross-sector primary arm。

## Verdict

```text
calibration_success = false
test_authorized = false
formal_success = not_evaluated
```

依 frozen rule，B V1 held-out test 不執行。B V1 保留以下結論：

1. Discovery 與 calibration 都重現 L16 `instruction_context` 相對 header control 的
   resample-patching sufficiency。
2. Calibration 未重現 sector specificity，因 same-sector peer context control 的
   absolute effect 更大。
3. 目前證據支持 identity-conditioned context-state sensitivity，不支持 formal
   sector-conditioned claim。

若後續改變 peer matching、primary specificity contrast、aggregation 或 gate，必須建立
B V2；不得用 B V1 test split 調整 protocol 後回填本版本。

## Interpretation Limits

本實驗測量 fixed negative evidence 下 residual resample-patching sufficiency。它不證明
L16 state 必要、不把 L16 稱為唯一 decision computation，也不建立 attention mechanism。
Calibration failure 不能解讀成 L16 context 沒有效應；它只否定 B V1 預先定義的
sector-specific confirmation gate。
