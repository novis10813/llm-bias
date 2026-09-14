# Balanced Evidence Gap 研究線入口

**文件定位：全線入口，不是新增協議。** 本頁只概覽研究問題、各階段狀態與證據；`details/` 內的原始協議與報告仍是 canonical source of truth，搬移不改變其 frozen、completed 或 proposed 狀態。本頁不授予新的 run authorization。歷史收線內容已整合至 [root report](report.md)。

## 全線問題

在多空財務證據完全對稱時，單純更換公司實體是否仍會造成可測量的決策落差？若會，實體訊號如何從早期承載帶移交至決策形成區間，且一階歸因選出的通道是否具有因果作用？

研究線已收線：Phase 1–2 確認行為落差與表徵傳遞帶，Phase 3 否決三個一階歸因 MLP 座標的單神經元因果假說。各階段結論不能由 Phase 3 單獨代表。

## 階段路由與主要證據

| 階段 | 原始協議與報告 | 主要結果與證據狀態 |
|---|---|---|
| **Phase 1：行為確認** | [協議](details/proposal-phase1.md) · [報告](details/report-phase1.md) | **Completed、三判準全過**：16 家 test-split 公司；named-vs-anonymous gap +0.432 nats，95% CI [+0.350, +0.518]；16/128 配對翻轉。 |
| **Phase 2A：cross-entity probe，Rev 1** | [Rev 1 協議](details/proposal-phase2.md) · [2A 報告](details/report-phase2a.md) | **Completed、Rev 1 gate 未過**：pure entity margin IQR 0.570 nats，但對 Phase 1 named margin 的 Spearman ρ = −0.411。這是歷史判定，不回填改寫。 |
| **Phase 2A：Rev 2 重評** | [Rev 2 協議](details/proposal-phase2-rev2.md) · [Phase 2 總報告](details/report-phase2.md) | **Rev 2 gate 通過**：改以 Phase 1 gap 參照與 group construct check；既有 records 的 CPU-only re-analysis 授權後續 2B/2C。 |
| **Phase 2B–2C：路徑與組件定位** | [Phase 2 Rev 1 協議](details/proposal-phase2.md) · [Rev 2 協議](details/proposal-phase2-rev2.md) · [報告](details/report-phase2.md) | **Completed、2A Rev 2 與 2C 通過**：entity span L0–11 承載，L12–15 handoff，instruction span L15 transfer peak +0.464；MLP attribution 找到 L19/n6334、L20/n6520、L26/n2394，attention arm null。 |
| **Phase 3：causal validation** | [協議](details/proposal-phase3.md) · [報告](details/report-phase3.md) | **Frozen、Completed，Gate 3A fail（0/3）**：三個候選的 additive `mlp_addition` 效應 ≤ matched controls，|mean ΔM| ≤ 0.012 nats，零 decision flip；investment dial 對照仍可移動約 ±1.0–1.1 nats。 |
| **收線後診斷** | [J-lens 診斷](details/diagnostic-jlens-neurons.md) | **Completed、descriptive、non-causal**：三個 entity candidate 的 transported direction readout 沒有改變 Phase 3 null，也不重開 gate。 |

## 研究線狀態

核心行為現象成立；表徵路徑定位至 L0–11 的 entity span、L12–15 的 instruction-context handoff；晚期單一 MLP 座標未通過因果驗證。後續若改變干預位置、estimand、controls 或 gate，須另立版本，不得在本入口或既有 frozen 文件中追加新協議。

## 閱讀邊界

Phase 1 是行為基準；Phase 2 是中間路徑與組件定位；Phase 3 是對候選座標的因果收口。收線後的 J-lens 診斷只提供 transported representation readout，不能把 descriptive 結果改寫成 causal claim。各階段的 input、output、control 與容差仍只由 `details/` 原文維護。

## 最終入口

[研究線收線報告](report.md) 是全線最終摘要、限制、run provenance 與產物索引。

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**

- [entity-cell-localization](../entity-cell-localization/proposal.md)（研究承接）：事實失憶但決策不翻轉，促使研究轉向平衡證據下的 entity decision gap。
- [investment-dial](../investment-dial/proposal.md)（資料／產物依賴）：沿用 test split、prompt 格式與 L15/N8490 dial 對照；檢查全域立場調控之外的 entity 差異。
- [activation-patching-causal-tracing](../activation-patching-causal-tracing/proposal.md)（研究承接）：明確證據下 header 效應弱，改在多空對稱條件確認 entity 影響。
- [jspace-token-experiments](../jspace-token-experiments/proposal.md)（研究承接）：V2 零證據 prior probe 未確認 entity 分歧，改測平衡證據；不沿用 V2 direction。

**後續**

- [entity-to-dial](../entity-to-dial/proposal.md)（資料／產物依賴）：承接 Phase 2B 的承載帶、L15 峰值與存檔對照，Phase 3 null 限定單神經元假說。

- [selective-intervention V1](../selective-intervention/proposal.md)（資料／產物依賴）：使用 Phase 2A 的 16 公司、64 prompts 與 archived margins，評估 L15 子空間移除。

最終／最新結果見本研究的 [report](report.md)。
