# Technology identity-header span sensitivity: discovery report

## Status

V1 discovery 已完成。正式 discovery run：

`artifacts/qwen3.5-4b/technology-header-span-sensitivity/runs/tech-header-discovery-gpu1-20260831T090100Z`

- Model：Qwen3.5-4B（`.cache/models/qwen3.5-4b`）
- Split：`discovery`
- Sector：Technology
- Prompt column：`prompt_with_context_attribute_0`
- Tickers：35
- Prepared/forward records：245（35 tickers × 7 conditions）
- Bootstrap：2,000 次 ticker-level resampling，seed `20260827`
- Manifest：`complete`

第一次同名 run `tech-header-discovery-20260831` 在 forward 開始時因 GPU 0 記憶體不足
失敗。Artifact manifest 保留 `failed` 狀態；結果不納入分析。正式 run 固定到 GPU 1，
使用新 run ID，未覆寫失敗 artifact。

## Discovery Results

每個 condition 先在同 ticker 內減去 `original` margin。負值表示 header mutation 讓
Buy-minus-Sell margin 往 Sell 方向移動。

| Condition              | Mean ΔM | Median ticker ΔM | 95% ticker-bootstrap CI |   Raw p |  Holm p | Margin sign flips |
| ---------------------- | ------: | ---------------: | ----------------------: | ------: | ------: | ----------------: |
| `anonymous_ticker`     | -0.0107 |          -0.0000 |      [-0.0929, +0.0786] |  0.8406 |  0.8406 |                 0 |
| `anonymous_name`       | -0.1893 |          -0.2500 |      [-0.2964, -0.0857] | 0.00250 | 0.00500 |                 1 |
| `anonymous_identity`   | -0.5929 |          -0.5000 |      [-0.8179, -0.3786] | 0.00050 | 0.00300 |                 2 |
| `same_sector_swap`     | -1.3857 |          -1.2500 |      [-1.7465, -1.0500] | 0.00050 | 0.00300 |                12 |
| `constructed_identity` | -1.2107 |          -1.2500 |      [-1.5143, -0.9179] | 0.00050 | 0.00300 |                 9 |
| `name_form_control`    | -0.8286 |          -0.7500 |      [-1.0607, -0.5892] | 0.00050 | 0.00300 |                 5 |

## Go/No-Go Assessment

Discovery 不產生 confirmatory verdict。依 [proposal](proposal.md) 的 go/no-go contract：

1. `anonymous_ticker` 沒有可定位的 effect；CI 跨越 0。
2. `anonymous_name` 與 `anonymous_identity` 的 Sell-direction effect 通過 discovery
   統計檢查，但兩者的絕對 mean effect 小於 `name_form_control`。V1 discovery 因此不能
   把這兩個結果解讀為超過 surface-form variation 的 identity-header effect。
3. `same_sector_swap` 的絕對 mean effect（1.3857）大於 `name_form_control`（0.8286），
   且產生 12/35 margin sign flips。它是進入 calibration 的 primary condition。
4. `constructed_identity` 的絕對 mean effect（1.2107）也大於 `name_form_control`，但
   constructed strings 不保證模型未見過，僅作 secondary condition。
5. `name_form_control` 本身效應大，顯示 header tokenization／surface-form mutation
   能顯著移動 margin。後續不得把 V1 header-only 結果改稱 entity-only causal effect。

**Discovery decision：** 進入 calibration，但只把 `same_sector_swap` 凍結為 primary
condition；`constructed_identity` 作 secondary diagnostic，`name_form_control` 保留為
必要 surface-form control。Calibration/test protocol 必須在 inference 前另行凍結。

## Interpretation Limits

V1 只修改 bracketed ticker/name header，evidence body 未改動。結果測量既有 evidence
之上顯示 identity header 的邊際行為敏感度。它不證明完整 entity replacement 的效果，
也不是 residual mechanism、attention effect 或 standalone causal evidence。
