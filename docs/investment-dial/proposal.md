# Investment-dial 研究線入口

**文件定位：全線入口，不是新增協議。** 本頁整理歷史 V1、fine-a diagnostic 與 V2 formal calibration；詳細契約仍以 `details/` 文件與既有 run protocol artifact 為準。本頁不授予新的 run authorization。收線內容見 [root report](report.md)。

## 全線問題

Park et al. (2026) 提出的單一 MLP coordinate additive intervention，能否在 Qwen3.5-4B 上校準模型整體 buy/sell stance，並在不同 target 上穩定控制決策？本線是方法復現與機制驗證，不是對原論文數值的 numeric replication。

研究線已收線；V1 的粗網格失敗由 fine-a diagnostic 與 V2 formal calibration 釐清，V2 gate 通過。

## 版本與診斷矩陣

| 版本／紀錄 | 原始文件 | 主要結果與證據狀態 |
|---|---|---|
| **V1 calibration** | [歷史文件索引](details/README.md)（V1 無獨立 proposal；契約保存在 run protocol artifact） | **Implemented、exploratory completed**：沿用 L15/n8490；B 組 RMSE 0.5757，顯示粗網格校準不足。 |
| **fine-a diagnostic** | [診斷協議](details/diagnostic-fine-a.md) · [診斷報告](details/report-fine-a.md) | **Diagnostic completed**：A-only 密集觀測顯示 δ 接近 0 有 steep transition，支持取樣解析度不足的解釋，但診斷不取代 formal gate。 |
| **V2 calibration** | [V2 協議](details/proposal-v2.md) · [V2 報告](details/report-v2.md) | **Formal completed、gate pass**：15 點 A curve 與 fine-grid inversion；B 組 RMSE 0.0579、最大誤差 0.0824，均低於 frozen 門檻，`certified=true`。 |

## 研究線狀態

V2 證實同一坐標可在本模型與既有樣本設計下精確校準三個 target（−0.3、0、+0.3），並呈現單向、不可逆的 decision flips。結論仍受 B 組曾參與 V1 候選排序、未配置 control-neuron arm、未驗證完全未見 test split 等限制；這些邊界與 run provenance 不在本入口重寫。

## 閱讀邊界

V1 是 exploratory calibration；fine-a 是 diagnostic；V2 才是 formal gate。這條線證明 model-level stance control 的校準，不證明 entity-specific 或 sector-specific 控制，也不宣稱重現 Park et al. 的原始數值。V2 的 input、curve fitting、target grid 與 interpretation limits 仍以 frozen proposal 為準。

## 最終入口

[研究線收線報告](report.md) 是全線最終摘要，包含 V1、fine-a、V2 的數值、限制、artifact path 與後續另立研究線的條件。

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**：本總覽未列其他實驗為必備上游；外部資料／文獻與儀器來源依原協議。

**後續**

- [balanced-evidence-gap](../balanced-evidence-gap/proposal.md)（資料／產物依賴）：沿用 test split、prompt 格式與 L15/N8490 dial 對照；檢查全域立場調控之外的 entity 差異。
- [entity-to-dial](../entity-to-dial/proposal.md)（資料／產物依賴）：沿用 L15/N8490 與 additive intervention 語義，檢查 entity 訊號是否經由 dial。

- [selective-intervention V1](../selective-intervention/proposal.md)（方法參考）：沿用 L15/N8490 與 ±4 native-unit push 作 dial probe。

最終／最新結果見本研究的 [report](report.md)。
