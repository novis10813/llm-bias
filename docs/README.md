# 研究目錄與前後關係

先從各研究的 `proposal.md` 了解全線，再讀 `report.md` 的最終／最新結果。
各版本、階段、diagnostic 與 smoke 紀錄保留在 `details/`，原始協議仍有效；
本頁只整理閱讀順序與既有關係，不新增實驗、改動 gate 或授權 run。

## 關係標籤說明依賴，不代表因果證明

- **資料／產物依賴**：使用上游的資料、split、canonical lens、候選座標或凍結結果；依賴範圍以說明為準，不表示必須重跑上游。
- **研究承接**：上游發現構成下一研究的問題來源，不等於程式輸入依賴。
- **方法參考**：沿用測量或控制原則，研究仍獨立立案。

同一研究可有多個上游。下表按研究問題分組，組內順序供閱讀，並非嚴格執行順序；
實際前後關係以後面的來源對照為準。跨實驗計畫見 [research program](proposal/README.md)：
本輪以行為確認與機制解剖為主成果，跨模型與跨任務驗證仍為必要且未完成的交付；
selective intervention 列為獨立後續研究，不作本輪完成條件；V1 formal 已完成，
full-strength gate fail，見 [V1 報告](selective-intervention/report.md)。

執行與檔案責任見 [shared core](shared-experiment-core.md)、[scripts reference](research-scripts.md)
與 [文件編排規則](documentation-system.md)。Frozen code 另見 [archive](archive/README.md)。

## 共用資料與儀器先看這三份

| 研究與狀態 | 入口 |
|---|---|
| Baseline trial：共用輸入與 prompt analysis 重現；完成狀態依 dataset/run，非單一全線 verdict。 | [proposal](baseline-trial/proposal.md) · [report](baseline-trial/report.md) |
| Qwen Jacobian-lens selection：候選選擇與 canonical lens 驗證／promotion；屬儀器 workflow。 | [proposal](jacobian-lens-selection/proposal.md) · [report](jacobian-lens-selection/report.md) |
| J-space evaluation：optional、proposed、non-runnable；不 gate 任何 active milestone。 | [proposal](j-space-evaluation/proposal.md) · [report](j-space-evaluation/report.md) |

## 從表徵提名到介入與位置定位

| 研究與狀態 | 入口 |
|---|---|
| J-space sector intervention：已完成 held-out 評估，結果與限制見 report。 | [proposal](jspace-sector-intervention/proposal.md) · [report](jspace-sector-intervention/report.md) |
| J-space valence readout：Technology discovery completed；表徵候選，非 causal proof。 | [proposal](jspace-valence-readout/proposal.md) · [report](jspace-valence-readout/report.md) |
| Header-span sensitivity：V1 discovery completed；calibration/test protocol 未凍結。 | [proposal](span-sensitivity/proposal.md) · [report](span-sensitivity/report.md) |
| J-space token V1/V2：V1 shortlist 空；V2 first formal pipeline success=false，diagnostics 分開解讀。 | [proposal](jspace-token-experiments/proposal.md) · [report](jspace-token-experiments/report.md) |
| Activation patching：Draft 1 completed；unchanged held-out confirmation success=true。 | [proposal](activation-patching-causal-tracing/proposal.md) · [report](activation-patching-causal-tracing/report.md) |
| Sector/context A/B/C：discovery completed；B V1 calibration success=false，held-out test 未執行。 | [proposal](sector-context-followup/proposal.md) · [report](sector-context-followup/report.md) |

## 從事實記憶、立場調控追查實體決策差異

| 研究與狀態 | 入口 |
|---|---|
| Entity Cell：主線收線，V3 確認 4 cells；E4 probe completed、仍為 proposed。 | [proposal](entity-cell-localization/proposal.md) · [report](entity-cell-localization/report.md) |
| 財務穩健定位：探索版 V1，smoke/exploratory completed；formal run 未授權。 | [proposal](financial-soundness-localization/proposal.md) · [report](financial-soundness-localization/report.md) |
| 財務穩健因果驗證：探索版 V1；formal run 未授權，不能稱神經元認證。 | [proposal](financial-soundness-causal-validation/proposal.md) · [report](financial-soundness-causal-validation/report.md) |
| Investment-dial：方法復現線收線；V2 formal gate pass，非 numeric replication。 | [proposal](investment-dial/proposal.md) · [report](investment-dial/report.md) |
| Balanced Evidence Gap：Phase 1–3 收線；行為／層帶定位通過，Phase 3 0/3 confirmed。 | [proposal](balanced-evidence-gap/proposal.md) · [report](balanced-evidence-gap/report.md) |
| Entity-to-Dial：Phase A–F 收線；F1 fail，L15 採 k=8 子空間描述。 | [proposal](entity-to-dial/proposal.md) · [report](entity-to-dial/report.md) |
| Selective-intervention：V1 formal `selective-intervention-v1-gpu-bf16-01` completed；G1a/G1b/G2 pass、G3/G4 fail，full-strength 負結果。 | [proposal](selective-intervention/proposal.md) · [report](selective-intervention/report.md) |

## 前後關係各有來源，不串成單一成功路線

V1→V2、Phase 1→2→3、Phase A→F 與 E1→E4 的線內順序，見各頂層 proposal。
`investment-dial` 獨立復現 Park et al. (2026)，不預設依賴 Entity Cell 的成功；
`j-space-evaluation` 來自跨實驗 research program，沒有 active experiment 的必跑上下游。
表中未列的連線不據日期或相似主題自行推定。

| 上游 → 後續研究 | 關係 | 承接內容與來源 |
|---|---|---|
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [baseline-trial](baseline-trial/proposal.md) | 資料／產物依賴 | 逐層 readout 使用 validated canonical lens；不是由 baseline workflow 隱式 fitting。 [依據](baseline-trial/proposal.md) |
| [baseline-trial](baseline-trial/proposal.md) → [jspace-sector-intervention](jspace-sector-intervention/proposal.md) | 資料／產物依賴 | 使用 trial-plan CSV 的公司與結構化證據。 [依據](jspace-sector-intervention/proposal.md) |
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [jspace-sector-intervention](jspace-sector-intervention/proposal.md) | 資料／產物依賴 | 產業座標介入使用 canonical lens 的投影。 [依據](jspace-sector-intervention/proposal.md) |
| [baseline-trial](baseline-trial/proposal.md) → [jspace-valence-readout](jspace-valence-readout/proposal.md) | 資料／產物依賴 | 從 trial-plan 結構化證據建立正負 valence pairs。 [依據](jspace-valence-readout/proposal.md) |
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [jspace-valence-readout](jspace-valence-readout/proposal.md) | 資料／產物依賴 | 詞彙 transported readout 使用 canonical lens。 [依據](jspace-valence-readout/proposal.md) |
| [jspace-sector-intervention](jspace-sector-intervention/proposal.md) → [jspace-valence-readout](jspace-valence-readout/proposal.md) | 資料／產物依賴 | 共用已固定的 ticker split manifest，不重分 discovery/calibration/test。 [依據](jspace-valence-readout/proposal.md) |
| [jspace-sector-intervention](jspace-sector-intervention/proposal.md) → [span-sensitivity](span-sensitivity/proposal.md) | 資料／產物依賴 | 共用 split manifest；以 header 表面替換建立行為基準，不依賴介入成功。 [依據](span-sensitivity/proposal.md) |
| [jspace-valence-readout](jspace-valence-readout/proposal.md) → [jspace-token-experiments](jspace-token-experiments/proposal.md) | 資料／產物依賴 | V1 直接消費提名的 vocabulary candidates；V2 改用 outcome-gradient，不沿用 V1 direction source。 [依據](jspace-token-experiments/details/proposal-v1.md) |
| [jspace-sector-intervention](jspace-sector-intervention/proposal.md) → [jspace-token-experiments](jspace-token-experiments/proposal.md) | 資料／產物依賴 | V1/V2 沿用 split manifest；各版本保留獨立 config 與 gate。 [依據](jspace-token-experiments/details/proposal-v2.md) |
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [jspace-token-experiments](jspace-token-experiments/proposal.md) | 資料／產物依賴 | V1 詞彙方向與 V2 direction decode 等 readout 使用 canonical lens。 [依據](jspace-token-experiments/details/proposal-v2.md) |
| [jspace-token-experiments](jspace-token-experiments/proposal.md) → [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) | 研究承接 | V2 未建立位置特異性，改用模型自然狀態差定位決策充分性。 [依據](activation-patching-causal-tracing/proposal.md) |
| [baseline-trial](baseline-trial/proposal.md) → [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) | 資料／產物依賴 | 沿用 trial rows 的正負證據配對；residual patch 本身不要求 lens。 [依據](activation-patching-causal-tracing/proposal.md) |
| [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) → [sector-context-followup](sector-context-followup/proposal.md) | 研究承接 | 依 evidence→instruction context→final 的層級轉移，設計 A/B/C 延伸。 [依據](sector-context-followup/proposal.md) |
| [jspace-sector-intervention](jspace-sector-intervention/proposal.md) → [sector-context-followup](sector-context-followup/proposal.md) | 資料／產物依賴 | A/B/C 沿用 Technology 與 Financial Services split manifest。 [依據](sector-context-followup/proposal.md) |
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [sector-context-followup](sector-context-followup/proposal.md) | 資料／產物依賴 | 僅 C 的 L16 transported readout 需要 canonical lens；A/B 是 residual patch。 [依據](sector-context-followup/proposal.md) |
| [jspace-sector-intervention](jspace-sector-intervention/proposal.md) → [entity-cell-localization](entity-cell-localization/proposal.md) | 資料／產物依賴 | E1 V1 沿用 split manifest；不是以 sector intervention verdict 為 gate。 [依據](entity-cell-localization/details/proposal-v1.md) |
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [entity-cell-localization](entity-cell-localization/proposal.md) | 資料／產物依賴 | E4 使用 pinned canonical lens；不把這項要求擴張為 E1 定位的先決條件。 [依據](entity-cell-localization/details/proposal-e4.md) |
| [entity-cell-localization](entity-cell-localization/proposal.md) → [financial-soundness-localization](financial-soundness-localization/proposal.md) | 方法參考 | 借用 V2/V3 自然句定位原則，不沿用實體專屬性與事實崩塌 gate。 [依據](financial-soundness-localization/proposal.md) |
| [financial-soundness-localization](financial-soundness-localization/proposal.md) → [financial-soundness-causal-validation](financial-soundness-causal-validation/proposal.md) | 資料／產物依賴 | 以定位產出的候選座標做獨立干預驗證；不從驗證句式回頭挑候選。 [依據](financial-soundness-causal-validation/proposal.md) |
| [entity-cell-localization](entity-cell-localization/proposal.md) → [balanced-evidence-gap](balanced-evidence-gap/proposal.md) | 研究承接 | 事實失憶但決策不翻轉，促使研究轉向平衡證據下的 entity decision gap。 [依據](balanced-evidence-gap/details/proposal-phase1.md) |
| [investment-dial](investment-dial/proposal.md) → [balanced-evidence-gap](balanced-evidence-gap/proposal.md) | 資料／產物依賴 | 沿用 test split、prompt 格式與 L15/N8490 dial 對照；檢查全域立場調控之外的 entity 差異。 [依據](balanced-evidence-gap/details/proposal-phase1.md) |
| [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) → [balanced-evidence-gap](balanced-evidence-gap/proposal.md) | 研究承接 | 明確證據下 header 效應弱，改在多空對稱條件確認 entity 影響。 [依據](balanced-evidence-gap/details/proposal-phase1.md) |
| [jspace-token-experiments](jspace-token-experiments/proposal.md) → [balanced-evidence-gap](balanced-evidence-gap/proposal.md) | 研究承接 | V2 零證據 prior probe 未確認 entity 分歧，改測平衡證據；不沿用 V2 direction。 [依據](balanced-evidence-gap/details/proposal-phase1.md) |
| [balanced-evidence-gap](balanced-evidence-gap/proposal.md) → [entity-to-dial](entity-to-dial/proposal.md) | 資料／產物依賴 | 承接 Phase 2B 的承載帶、L15 峰值與存檔對照，Phase 3 null 限定單神經元假說。 [依據](entity-to-dial/details/proposal-phase-abc.md) |
| [investment-dial](investment-dial/proposal.md) → [entity-to-dial](entity-to-dial/proposal.md) | 資料／產物依賴 | 沿用 L15/N8490 與 additive intervention 語義，檢查 entity 訊號是否經由 dial。 [依據](entity-to-dial/details/proposal-phase-abc.md) |
| [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) → [entity-to-dial](entity-to-dial/proposal.md) | 方法參考 | 沿用 bidirectional residual patch、固定答案 margin 與 self-source no-op 契約。 [依據](entity-to-dial/details/proposal-phase-abc.md) |
| [balanced-evidence-gap](balanced-evidence-gap/proposal.md) → [selective-intervention](selective-intervention/proposal.md) | 資料／產物依賴 | V1 使用 Phase 2A 的 16 公司、64 prompts 與 archived margins。 [依據](selective-intervention/details/proposal-v1.md) |
| [entity-to-dial](entity-to-dial/proposal.md) → [selective-intervention](selective-intervention/proposal.md) | 資料／產物依賴 | V1 使用 e-01 的 L15 k=8 basis，測試移除子空間分量；transfer 效果不預設 removal 成功。 [依據](selective-intervention/details/proposal-v1.md) |
| [investment-dial](investment-dial/proposal.md) → [selective-intervention](selective-intervention/proposal.md) | 方法參考 | V1 沿用 L15/N8490 與 ±4 native-unit push 作 dial probe。 [依據](selective-intervention/proposal.md) |
