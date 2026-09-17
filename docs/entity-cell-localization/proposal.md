# Entity Cell Localization 研究線入口

**文件定位：全線入口，不是新增協議。** 本頁只整理 E1、E2、E3、E4 的問題、版本與證據；`details/` 內各版協議和報告仍是 canonical source of truth，原始 frozen、completed、proposed 狀態不因整理而改變。本頁不授予新的 run authorization。

## 全線問題

Qwen3.5-4B 是否存在承載特定公司客觀事實的 MLP 實體神經元？這些事實記憶單元是否等同於投資決策單元？定位、下游歸因、因果干預與 residual-stream readout 各回答不同層次，不能合併成單一協議或由 E4 取代 E1–E3。

研究線主體已收線；E4 是已完成但仍標示 **proposed probe、非 formal promotion** 的表示層診斷。

## 階段／版本矩陣

[歷史版本索引與舊 README](details/README.md) 保留完整 phase matrix、run 清單與儀器註記；收線內容見 [root report](report.md)；下表只列全線入口所需的協議與證據。

| 階段／版本 | 原始協議與報告 | 主要結果與證據狀態 |
|---|---|---|
| **E1 V1：Header family** | [協議](details/proposal-v1.md) · [報告](details/report-v1.md) | **Discovery completed、negative**：0/35 trusted candidate；31/35 撞在模板神經元 L0/N4485，否定固定 Header 定位的有效性。 |
| **E1 V2：Natural-sentence frames** | [協議](details/proposal-v2.md) · [報告](details/report-v2.md) | **Discovery completed**：1/35 trusted candidate（FTNT，L0/N104）；未形成 calibration/held-out confirmation。 |
| **E1 V2 補充批次** | [HFM 報告](details/report-hfm-discovery.md) · [HFM-2 報告](details/report-hfm2-discovery.md) · [decision probe 診斷](details/report-decision-probe.md) · [JNJ/JPM battery 診斷](details/report-jnj-jpm-battery.md) | 高事實記憶批次促成 fact-level gate 轉向；JNJ/JPM battery 顯示事實失憶與決策翻轉解離，0 次決策翻轉。 |
| **E1 V3：Fact-level amnesia gate** | [協議](details/proposal-v3.md) · 結果收錄於[收線報告](report.md) | **Frozen、calibration + hold-out 通過**：確認 4 個 entity cells：JNJ L4/N7676、PLTR L2/N5003、BAC L0/N7801、CAT L2/N7997；共享事實通道另行分類。V3 沒有獨立 report，結果以 V3 協議與收線報告為準。 |
| **E2：Full-attention DLA attribution** | [協議](details/proposal-e2.md) · [報告](details/report-e2.md) | **Discovery completed、descriptive**：128 heads 全為 instruction-dominant，選出 5 個 heads 供 E3；未執行 calibration/test。 |
| **E3 V1：Suppression 與 downstream attenuation** | [協議](details/proposal-e3-v1.md) · [報告](details/report-e3-v1.md) | **Discovery completed、causal specificity confirmed**：FTNT 候選的跨 ticker 特異性與 evidence preservation 通過；未執行 calibration/held-out test。 |
| **E4：Residual stream readout probe** | [協議](details/proposal-e4.md) · [診斷報告](details/report-e4-readout-delta.md) | **Probe completed、proposed、非 formal promotion**：4 個 frozen V3 cells 的資訊在 cell 後 1–8 層開始顯現，L16–L29 最強；readout 是 descriptive，不能升格為新因果 gate。 |
| **E5：Fact coverage profile probe** | [協議](details/proposal-e5.md) | **Proposed、未執行**：單顆 cell 壓抑下跨事實類別（HQ 雙句框／代號／年份／交易所／競爭對手，18 pairs × 4 條件）的輸出層覆蓋輪廓量測；兩輪 clean preflight 已完成，協議已擬定（含 gold 綁定與剔除記錄），正式 run 未授權。 |

## 研究線狀態

目前最穩固的結論是：4 個 entity cells 能特異性破壞客觀事實回想，但不驅動本線測試的 buy/sell 決策；E2/E3 描述其下游路徑邊界，E4 補充 residual-stream representation readout。E5 為新立 proposed 診斷探針（單顆 cell 的事實覆蓋邊界：它承載的是地理事實子集還是更廣的公司身分槽位），兩輪 clean preflight 已完成、協議已擬定，正式 run 未授權。所有版本協議、控制組、門檻與限制仍留在 `details/` 原文。

## 閱讀邊界

E1 的 fact-level amnesia gate 定義確認名冊；E2 的 DLA 是 discovery-level routing 描述；E3 的 suppression 是有限樣本的 causal specificity discovery；E4 只讀 residual stream 的表示差異；E5 的事實覆蓋輪廓是輸出層描述（null 僅結論「輸出層未受影響」，不是「未承載」，見其 §5 限制）。不能用 E4 的詞彙讀出替代 E1 的 fact gate，也不能把 E1 結果外推為通用決策機制。

## 最終入口

[研究線收線報告](report.md) 提供全線結論、E1–E4 限制、run provenance 與產物索引。

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**

- [jspace-sector-intervention](../jspace-sector-intervention/proposal.md)（資料／產物依賴）：E1 V1 沿用 split manifest；不是以 sector intervention verdict 為 gate。
- [jacobian-lens-selection](../jacobian-lens-selection/proposal.md)（資料／產物依賴）：E4 使用 pinned canonical lens；不把這項要求擴張為 E1 定位的先決條件。

**後續**

- [financial-soundness-localization](../financial-soundness-localization/proposal.md)（方法參考）：借用 V2/V3 自然句定位原則，不沿用實體專屬性與事實崩塌 gate。
- [balanced-evidence-gap](../balanced-evidence-gap/proposal.md)（研究承接）：事實失憶但決策不翻轉，促使研究轉向平衡證據下的 entity decision gap。

最終／最新結果見本研究的 [report](report.md)。
