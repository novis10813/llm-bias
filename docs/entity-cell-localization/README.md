# Entity cell localization and downstream attribution: versions

本頁是 entity-cell 研究線（`llm_bias/entity_cell/`，CLI `entity-cell`）的版本
入口。討論、config、run ID 與結果報告必須標明版本（V1 / V2），避免把
frozen header family 與 surface-varying frame family 的結果混用。E2（downstream
component attribution）與 E3（suppression）設計跨版本共用，定義在
[proposal-v1](proposal-v1.md)。

## Version matrix

| Version | E1 direction source | Primary outcome | Implementation | Evidence status |
|---|---|---|---|---|
| [V1](proposal-v1.md) | Frozen 三行 financial header（12 個 header-prefix variants，表面固定） | Trusted candidate entity cell（held overlap + amnesia endpoint；V1 雙關） | `entity-cell prepare/run --localization-family v1-header`（預設） | Discovery 完成：0/35 通過（template-dominated，機制已識別）；E2 discovery 完成（descriptive） |
| [V2](proposal-v2.md) | 12 個 frozen 自然句 frames（F0–F7 localization / H0–H3 held，公司名 plain prose） | Trusted candidate entity cell（V2 四關：held overlap、form-robust、template-robust、amnesia endpoint） | `--localization-family v2-frames`（frame variants + template-only control + re-tokenized surface controls + 非退化 wrong-entity 規則） | Discovery 完成：1/35 通過（FTNT, L0 N104；shared-top-1 caveat） |

Calibration / held-out test 皆未凍結、未執行；confirmation freeze 只能從
V2 discovery selections 建立（V2 versioning boundary）。

## V1 結論

V1 的 frozen header family 是 template-dominated：31/35 tickers 共享同一 top-1
neuron（L0, N4485），0/35 通過 amnesia gate。已知限制：V1 localization-stage
surface controls 重用了原 prompt 的 `input_ids`（control 為 no-op），其
top-5 overlap 報告非獨立證據；template-dominated 結論依靠 cross-ticker
top-1 collision 與 amnesia gate。E2（V1 protocol 下執行，E2 設計跨版本共用）
選出 late full-attention heads（L31 H0/H1/H3、L19 H4、L27 H6），identity
attention 貢獻小且晚、instruction-dominant。細節見 [report-v1](report-v1.md)。

## V2 結果摘要

Frame family 瓦解了 template 主導（N4485 從 31/35 降到 4/35 top-1），amnesia
gate 通過 9/35，form-robust 1/35；唯一通過全部四道 frozen gates 的是
**FTNT（L0, N104）**，但其 cell 同時是 17/35 tickers 的 shared top-1——跨
ticker selectivity 尚未證實，需 E3 selective intervention 或新的 frozen
版本（例如 selectivity gate）處理。細節見 [report-v2](report-v2.md)。

## Runs

全部位於 `artifacts/qwen3.5-4b/entity-cell-localization/runs/`：
V1 prepare/discovery、E2 v1–v5（v1–v4 為保留的失敗 run）、V2
prepare/smoke-v4/discovery-v2。
