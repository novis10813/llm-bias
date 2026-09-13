# J-space token experiments 研究線入口

**文件定位：全線入口，不是新增協議。** 本頁只區分 V1 與 V2 的 direction source、primary outcome、controls、gate 與證據；各版 frozen protocol 和歷史診斷仍以 `details/` 為準。本頁不授予新的 run authorization，也不把 V1 結果回填為 V2 結果。

## 全線問題

由 J-space transported representation readout 提名的 token direction，或由 outcome gradient fitting 得到的方向，能否在 held-out prompts 上特異性改變固定 Buy/Sell decision？V1 測試 vocabulary-nominated directions；V2 改以 outcome-gradient direction 並以 decision flips 為 primary outcome，兩者不可混用。

## 版本矩陣

[歷史版本索引與舊 README](details/README.md) 保留完整 versioning rules、run IDs 與輔助診斷索引；下表列出本入口的主要路由。

| 版本 | 原始協議與報告 | 主要結果與證據狀態 |
|---|---|---|
| **V1：vocabulary token screen** | [V1 協議](details/proposal-v1.md) · [V1 報告](details/report-v1.md) | **Discovery completed、shortlist empty**：specificity 的部分描述性訊號未通過 frozen ticker-consistency 與 Holm gate；840 個 interventions 僅 1 次 fixed-choice sign flip，未進入 calibration。 |
| **V2：outcome-conditioned decision flip** | [V2 協議](details/proposal-v2.md) · [V2 root 報告](report.md) | **Formal pipeline completed，`success=false`**：buy 方向 9/9 outcome vs 0/9 controls，Holm p=0.003；sell 方向僅 2 個 eligible tickers，Holm gate 因 n=2 結構性無法通過。final-position control 雙方向皆 100% flip，position specificity 不成立。 |
| **V2 geometry diagnostic** | [幾何診斷](details/report-v2-geometry.md) · [版本索引](details/README.md) | **Diagnostic、descriptive、non-causal**：sector state difference 幾乎正交於 frozen outcome directions；不改變 V2 verdict、estimand 或 gate。 |
| **V2 direction decode diagnostic** | [版本索引中的 decode 紀錄](details/README.md) · [V2 formal report](report.md) | **Diagnostic completed**：canonical Jacobian lens 的 transported readout 顯示 ±d 的 buy/sell token separation；這不是 chain-of-thought、discrete reasoning path 或 standalone causal claim。 |

## 研究線狀態

V1 不支持 vocabulary direction 的穩健 decision control。V2 在 buy direction 顯示 outcome-axis steering 可行，但 sell direction 的正式 gate 因 test split power 不足而失敗；position control 也限制了 evidence-position 因果解讀。若要重驗 sell direction 或改變 direction source、primary outcome、controls、gate，必須另立新版本。

## 閱讀邊界

V1 的 shortlist empty 是 vocabulary screen 的結果，不是 V2 的 direction failure。V2 的 `success=false` 保留完整 formal verdict，不能以 buy arm pass 覆蓋整體結果。geometry、prior probe 與 direction decode 都是輔助診斷；transported readout 不等於 chain-of-thought、discrete reasoning path 或 standalone causal claim。

## 最終入口

[V2 研究報告](report.md) 是目前 root report，包含 formal pipeline verdict、prior probe、direction decode、run provenance 與 interpretation limits。

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**

- [jspace-valence-readout](../jspace-valence-readout/proposal.md)（資料／產物依賴）：V1 直接消費提名的 vocabulary candidates；V2 改用 outcome-gradient，不沿用 V1 direction source。
- [jspace-sector-intervention](../jspace-sector-intervention/proposal.md)（資料／產物依賴）：V1/V2 沿用 split manifest；各版本保留獨立 config 與 gate。
- [jacobian-lens-selection](../jacobian-lens-selection/proposal.md)（資料／產物依賴）：V1 詞彙方向與 V2 direction decode 等 readout 使用 canonical lens。

**後續**

- [activation-patching-causal-tracing](../activation-patching-causal-tracing/proposal.md)（研究承接）：V2 未建立位置特異性，改用模型自然狀態差定位決策充分性。
- [balanced-evidence-gap](../balanced-evidence-gap/proposal.md)（研究承接）：V2 零證據 prior probe 未確認 entity 分歧，改測平衡證據；不沿用 V2 direction。

最終／最新結果見本研究的 [report](report.md)。
