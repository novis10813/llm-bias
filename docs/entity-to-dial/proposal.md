# Entity-to-Dial 研究線入口

**文件定位：全線入口，不是新增協議。** 本頁只導覽 Phase A–F 的問題、狀態與主要證據；各 phase 原始協議仍在 `details/`，其 frozen、completed 與 gate 結果維持不變。本頁不授予新的 run authorization。收線內容見 [root report](report.md)。

## 全線問題

實體訊號從 L0–11 的 entity span 承載帶進入決策形成區間後，是否經由 L15/N8490 investment dial，或改由 instruction span 的其他殘差狀態傳遞？本線分別檢驗 token 充分性、block 貢獻、dial 路徑、低維 state direction 與雙路徑加法性；後期 phase 不取代前期 phase 的問題或結果。

研究線已收線。最終描述採納 L15 instruction span 的 **k=8 殘差子空間**，不是「v1 + dial」雙通道模型。

## 階段矩陣

| 階段 | 原始協議與報告 | 主要結果與證據狀態 |
|---|---|---|
| **Phase A–C：三段式 discovery** | [Phase A–C 協議](details/proposal-phase-abc.md) · [報告](details/report-phase-abc.md) | **Formal completed，Gate A1/B/C 全 fail**：L12 後 entity token 區間不再具充分性；單一 block patch 未形成主路徑；L15/N8490 dial transplant 僅解釋約 5%，主體路徑繞過 dial。 |
| **Phase D：instruction-span block sweep** | [協議](details/proposal-phase-d.md) | **Formal completed、Gate D fail**：無合格 block 層；19/19 非 final layers 的 top-channel 關聯勝過 controls 只屬 descriptive，sector agreement 多數 0/4。D 結果保留於本 phase proposal，沒有獨立 report。 |
| **Phase E：dual-block 與 state-direction** | [協議](details/proposal-phase-e.md) | **Formal completed、Gate E1 pass、Gate E2b fail**：L15 joint ratio median 0.575；dial ratio 0.434；k=8 恢復 98.3% full-swap 效應；v1 與 dial footprint cosine −0.020。E 結果保留於本 phase proposal，沒有獨立 report。 |
| **Phase F：dual-path additivity** | [協議](details/proposal-phase-f.md) | **Formal completed、Gate F1 fail；F2 context-dependent/null**：additivity ratio median 0.7329，雙通道下游匯流飽和；依協議 fallback 採 k=8 子空間。F 結果保留於本 phase proposal，沒有獨立 report。 |

## 研究線狀態

三段式 discovery 排除 entity token、單一 block 與單一 dial 作為完整主路徑；Phase E/F 進一步顯示 L15 的 state difference 低維集中，但 v1 與 dial 的聯合介入不具加法性。結論限於既有 Qwen3.5-4B、16 家公司與 frozen directions；新方向須另立版本。

## 閱讀邊界

Phase A–C 的 null 是對各自路徑假說的排除，不是對所有可能路徑的全域否證。Phase D 的 channel ranking 屬 descriptive；Phase E 的 k=8 比例與 Phase F 的 additivity gate 各回答不同問題。所有 phase 的 block hook、方向來源、控制組與限制仍以 `details/` 原始文件為準。

## 最終入口

[研究線收線報告](report.md) 是全線最新且最終摘要，包含 Phase A–F 的完整因果鏈、限制、run IDs 與最終收線判定。

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**

- [balanced-evidence-gap](../balanced-evidence-gap/proposal.md)（資料／產物依賴）：承接 Phase 2B 的承載帶、L15 峰值與存檔對照，Phase 3 null 限定單神經元假說。
- [investment-dial](../investment-dial/proposal.md)（資料／產物依賴）：沿用 L15/N8490 與 additive intervention 語義，檢查 entity 訊號是否經由 dial。
- [activation-patching-causal-tracing](../activation-patching-causal-tracing/proposal.md)（方法參考）：沿用 bidirectional residual patch、固定答案 margin 與 self-source no-op 契約。

**後續**：尚無已立案的後續研究；不把報告中的建議視為已授權實驗。

最終／最新結果見本研究的 [report](report.md)。
