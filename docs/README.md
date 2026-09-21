# 各實驗發現與研究入口

快速看結果只需讀下表；查證據點「報告」，要執行或重跑才讀「協議」。
結論限於來源報告的模型、資料與干預條件；「未支持」不等於證明機制不存在，
「收線」也不等於假說成立。歷史版本留在 `details/`，不必按時間逐篇讀完。

下表按證據角色整理目前 active research；J-space token V1、V2 分列，資料流程與尚未執行的提案另組，不把工具完成當研究發現。本頁不改動判準或授權 run。

**狀態只表示本列範圍的進度**：規劃中、進行中、已完成、已收線、暫停、待確認。
「已完成」不代表檢驗通過；更新表格前須遵守 [欄位與狀態定義](documentation-system.md#總覽表格填寫規則)。

## Core evidence for the current paper narrative

這四條線保留正面與 null 結果，構成目前 paper narrative 的核心證據。

| 研究 | 狀態 | 一句話發現 | 查證／執行 |
|---|---|---|---|
| Entity Cell（主線） | 已收線 | 壓制單一神經元會損害公司的部分事實、也發現多家公司共用的事實通道，但受測買賣決策未翻轉，不能稱為完全忘記公司。 | [報告](entity-cell-localization/report.md) · [V3 協議](entity-cell-localization/details/proposal-v3.md) |
| Balanced Evidence Gap（Phase 1–3） | 已收線 | 相同多空證據下，換公司名稱會穩定改變買賣傾向，但歸因挑出的三個神經元在干預中都未勝過對照。 | [報告](balanced-evidence-gap/report.md) · [Phase 3 協議](balanced-evidence-gap/details/proposal-phase3.md) |
| Entity-to-Dial（Phase A–F） | 已收線 | 第 15 層的 8 維狀態差子空間可恢復近乎全部置換效果，但「一個殘差方向加一個 dial 神經元」的可加表示未通過檢驗。 | [報告](entity-to-dial/report.md) · [Phase F 協議](entity-to-dial/details/proposal-phase-f.md) |
| Evidence-insensitivity（Phase 1–3） | 已完成 | 雙模型的受測一維狀態未解釋行為分組，最後位置的狀態置換改變買賣分數卻未翻轉生成決策；Gemma Phase 2 僅作帶保留的描述。 | [報告](evidence-insensitivity/report.md) · [Phase 3 協議](evidence-insensitivity/details/proposal-phase3.md) |

## Boundary/control evidence

這些保留線限制 single-neuron、entity-concept 與 intervention claim 的解讀，不是核心 paper narrative 的替代證據。

| 研究 | 狀態 | 一句話發現 | 查證／執行 |
|---|---|---|---|
| Investment-dial（方法復現） | 已收線 | 在本模型與任務上，調整單一神經元能連續調節整體買賣立場並達到預設校準目標，完成方法復現，未以複製原論文數值為目標。 | [報告](investment-dial/report.md) · [V2 協議](investment-dial/details/proposal-v2.md) |
| 公司身分的中間概念（探索） | 已收線 | 在受測公司與層中，未找到能與買賣立場分離的公司概念；能預測立場的方向也未展現相應的強干預效果。 | [報告](entity-concept-decision/report.md) · [Phase 1 協議](entity-concept-decision/details/proposal-phase1.md) |
| Selective-intervention V1 / M6-V2 | 已完成 | V1 的 full-strength intervention 因全局與匿名副作用收口為負結果；M6-V2 外部 12 家的 current-runtime spread ratio = 0.6401（95% CI [0.2837, 2.1144]），未確認機制泛化。 | [報告](selective-intervention/report.md) · [V1 協議](selective-intervention/details/proposal-v1.md) · [M6-V2 協議](selective-intervention/details/proposal-m6-v2.md) |

## Active discovery and localisation controls

這些是 span sensitivity、J-space sector/valence/token、activation-patching、context 線與仍在規劃的 active evaluation；探索或失敗狀態維持原樣。

| 研究 | 狀態 | 一句話發現 | 查證／執行 |
|---|---|---|---|
| Header-span sensitivity（探索） | 已完成 | 同產業公司名稱置換比表面亂碼控制更能改變決策，其他匿名化操作則尚不能排除表面文字擾動。 | [報告](span-sensitivity/report.md) · [協議](span-sensitivity/proposal.md) |
| J-space sector intervention（獨立測試） | 已完成 | 置換產業的 J-space 座標未翻轉受測買賣決策，也未顯示效果專屬於產業方向或證據位置。 | [報告](jspace-sector-intervention/report.md) · [協議](jspace-sector-intervention/proposal.md) |
| J-space valence readout（科技業探索） | 已完成 | 讀出能區分正負證據的候選詞彙，但這只提供後續干預的候選，尚不能說這些詞彙驅動決策。 | [報告](jspace-valence-readout/report.md) · [協議](jspace-valence-readout/proposal.md) |
| J-space token **V1**（探索） | 已收線 | 沿候選詞彙方向做小幅干預，沒有任何候選通過預定篩選，未進入後續確認。 | [報告](jspace-token-experiments/details/report-v1.md) · [協議](jspace-token-experiments/details/proposal-v1.md) |
| J-space token **V2** | 已完成 | 依答案梯度選的方向能把受測 sell 推向 buy，但 sell 方向樣本不足，且最後位置的對照也能翻轉，未建立位置特異性。 | [報告](jspace-token-experiments/report.md) · [協議](jspace-token-experiments/details/proposal-v2.md) |
| Activation patching（獨立測試） | 已完成 | 能轉移買賣傾向的狀態，隨層數由證據區移到指令區、再到最後位置，且這個位置變化在獨立測試中重現。 | [報告](activation-patching-causal-tracing/report.md) · [協議](activation-patching-causal-tracing/proposal.md) |
| Sector/context follow-up（B V1 校準） | 已完成 | 固定負面證據下，跨產業換入指令區狀態的效果不比同產業換公司更強，未支持產業專屬的解釋。 | [報告](sector-context-followup/report.md) · [協議](sector-context-followup/proposal.md) |
| J-space evaluation | 規劃中 | 只有合成任務的 J-space 輔助評估設計，尚無研究結果，也不作為其他實驗的執行門檻。 | [狀態](j-space-evaluation/report.md) · [提案](j-space-evaluation/proposal.md) |

## Shared inputs and instruments

這些是共用輸入與儀器，不呈現為獨立的 entity-bias evidence。

| 研究 | 狀態 | 一句話發現 | 查證／執行 |
|---|---|---|---|
| Baseline trial（流程建置） | 已完成 | 已接通共用提示詞分析流程；各階段產物驗證不代表已得到單一偏誤結論。 | [報告](baseline-trial/report.md) · [協議](baseline-trial/proposal.md) |
| Qwen Jacobian-lens selection（選擇與部署） | 已完成 | 簡體中文校準的 lens 依預設規則獲選，但驗證集尚未顯示它顯著優於英文或混合語言版本。 | [報告](jacobian-lens-selection/report.md) · [協議](jacobian-lens-selection/proposal.md) |

跨實驗計畫見 [research program](proposal/README.md)；已凍結的舊工作見 [archive](archive/README.md)。
[財務判斷 frozen exploratory workflow](archive/financial-soundness-localization/proposal.md) 已移出 active execution tree，包含 [causal validation](archive/financial-soundness-causal-validation/proposal.md)。
工具與維護入口：[dashboard](interactive-prompt-lens-dashboard.md)、[shared core](shared-experiment-core.md)、
[scripts reference](research-scripts.md)、[artifact contract](artifact-contract.md)、[文件編排規則](documentation-system.md)。

## 前後關係各有來源，不串成單一成功路線

<details>
<summary>需要追查資料來源或研究承接時，展開完整關係表</summary>

### 關係標籤說明依賴，不代表因果證明

- **資料／產物依賴**：使用上游的資料、split、canonical lens、候選座標或凍結結果；不表示必須重跑上游。
- **研究承接**：上游發現構成下一研究的問題來源，不等於程式輸入依賴。
- **方法參考**：沿用測量或控制原則，研究仍獨立立案。

各研究的版本與階段順序，從報告的查證入口連到原始協議；尚存的舊版頂層 proposal 亦保留原有導覽。
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
| [jspace-valence-readout](jspace-valence-readout/proposal.md) → [jspace-token-experiments](jspace-token-experiments/report.md) | 資料／產物依賴 | V1 直接消費提名的 vocabulary candidates；V2 改用 outcome-gradient，不沿用 V1 direction source。 [依據](jspace-token-experiments/details/proposal-v1.md) |
| [jspace-sector-intervention](jspace-sector-intervention/proposal.md) → [jspace-token-experiments](jspace-token-experiments/report.md) | 資料／產物依賴 | V1/V2 沿用 split manifest；各版本保留獨立 config 與 gate。 [依據](jspace-token-experiments/details/proposal-v2.md) |
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [jspace-token-experiments](jspace-token-experiments/report.md) | 資料／產物依賴 | V1 詞彙方向與 V2 direction decode 等 readout 使用 canonical lens。 [依據](jspace-token-experiments/details/proposal-v2.md) |
| [jspace-token-experiments](jspace-token-experiments/report.md) → [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) | 研究承接 | V2 未建立位置特異性，改用模型自然狀態差定位決策充分性。 [依據](activation-patching-causal-tracing/proposal.md) |
| [baseline-trial](baseline-trial/proposal.md) → [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) | 資料／產物依賴 | 沿用 trial rows 的正負證據配對；residual patch 本身不要求 lens。 [依據](activation-patching-causal-tracing/proposal.md) |
| [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) → [sector-context-followup](sector-context-followup/proposal.md) | 研究承接 | 依 evidence→instruction context→final 的層級轉移，設計 A/B/C 延伸。 [依據](sector-context-followup/proposal.md) |
| [jspace-sector-intervention](jspace-sector-intervention/proposal.md) → [sector-context-followup](sector-context-followup/proposal.md) | 資料／產物依賴 | A/B/C 沿用 Technology 與 Financial Services split manifest。 [依據](sector-context-followup/proposal.md) |
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [sector-context-followup](sector-context-followup/proposal.md) | 資料／產物依賴 | 僅 C 的 L16 transported readout 需要 canonical lens；A/B 是 residual patch。 [依據](sector-context-followup/proposal.md) |
| [jspace-sector-intervention](jspace-sector-intervention/proposal.md) → [entity-cell-localization](entity-cell-localization/report.md) | 資料／產物依賴 | E1 V1 沿用 split manifest；不是以 sector intervention verdict 為 gate。 [依據](entity-cell-localization/details/proposal-v1.md) |
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [entity-cell-localization](entity-cell-localization/report.md) | 資料／產物依賴 | E4 使用 pinned canonical lens；不把這項要求擴張為 E1 定位的先決條件。 [依據](entity-cell-localization/details/proposal-e4.md) |
| [entity-cell-localization](entity-cell-localization/report.md) → [balanced-evidence-gap](balanced-evidence-gap/report.md) | 研究承接 | 事實失憶但決策不翻轉，促使研究轉向平衡證據下的 entity decision gap。 [依據](balanced-evidence-gap/details/proposal-phase1.md) |
| [investment-dial](investment-dial/report.md) → [balanced-evidence-gap](balanced-evidence-gap/report.md) | 資料／產物依賴 | 沿用 test split、prompt 格式與 L15/N8490 dial 對照；檢查全域立場調控之外的 entity 差異。 [依據](balanced-evidence-gap/details/proposal-phase1.md) |
| [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) → [balanced-evidence-gap](balanced-evidence-gap/report.md) | 研究承接 | 明確證據下 header 效應弱，改在多空對稱條件確認 entity 影響。 [依據](balanced-evidence-gap/details/proposal-phase1.md) |
| [jspace-token-experiments](jspace-token-experiments/report.md) → [balanced-evidence-gap](balanced-evidence-gap/report.md) | 研究承接 | V2 零證據 prior probe 未確認 entity 分歧，改測平衡證據；不沿用 V2 direction。 [依據](balanced-evidence-gap/details/proposal-phase1.md) |
| [balanced-evidence-gap](balanced-evidence-gap/report.md) → [entity-to-dial](entity-to-dial/report.md) | 資料／產物依賴 | 承接 Phase 2B 的承載帶、L15 峰值與存檔對照，Phase 3 null 限定單神經元假說。 [依據](entity-to-dial/details/proposal-phase-abc.md) |
| [investment-dial](investment-dial/report.md) → [entity-to-dial](entity-to-dial/report.md) | 資料／產物依賴 | 沿用 L15/N8490 與 additive intervention 語義，檢查 entity 訊號是否經由 dial。 [依據](entity-to-dial/details/proposal-phase-abc.md) |
| [activation-patching-causal-tracing](activation-patching-causal-tracing/proposal.md) → [entity-to-dial](entity-to-dial/report.md) | 方法參考 | 沿用 bidirectional residual patch、固定答案 margin 與 self-source no-op 契約。 [依據](entity-to-dial/details/proposal-phase-abc.md) |
| [balanced-evidence-gap](balanced-evidence-gap/report.md) → [selective-intervention](selective-intervention/report.md) | 資料／產物依賴 | V1 使用 Phase 2A 的 16 公司、64 prompts 與 archived margins。 [依據](selective-intervention/details/proposal-v1.md) |
| [entity-to-dial](entity-to-dial/report.md) → [selective-intervention](selective-intervention/report.md) | 資料／產物依賴 | V1 使用 e-01 的 L15 k=8 basis，測試移除子空間分量；transfer 效果不預設 removal 成功。 [依據](selective-intervention/details/proposal-v1.md) |
| [investment-dial](investment-dial/report.md) → [selective-intervention](selective-intervention/report.md) | 方法參考 | V1 沿用 L15/N8490 與 ±4 native-unit push 作 dial probe。 [依據](selective-intervention/details/proposal-v1.md) |
| [entity-to-dial](entity-to-dial/report.md) → [公司身分的中間概念](entity-concept-decision/report.md) | 資料／產物依賴（proposed） | 沿用 e-01 固定 k=8 基底，新增獨立概念驗證，不把基底直接命名成語義。 [依據](entity-concept-decision/details/design-and-validation.md) |
| [balanced-evidence-gap](balanced-evidence-gap/report.md) → [Evidence-insensitivity](evidence-insensitivity/report.md) | 研究承接 | 共享平衡證據下的 entity gap 現象動機化「entity × evidence」交互量測；共享證據設計與 identity-stripped 慣例沿用。 [依據](evidence-insensitivity/details/proposal-phase1.md) |
| [jspace-token-experiments](jspace-token-experiments/report.md) → [Evidence-insensitivity](evidence-insensitivity/report.md) | 資料／產物依賴 | 零證據 header-only 模板與強 sell prior 觀察（6 tickers）；零證據量測擴到 503 家。 [依據](evidence-insensitivity/details/proposal-phase1.md) |
| [entity-concept-decision](entity-concept-decision/report.md) → [Evidence-insensitivity](evidence-insensitivity/report.md) | 方法參考 | stance 軸 derive 與正交化方法於 Phase 2 沿用；不重測已收線結論。 [依據](evidence-insensitivity/details/proposal-phase2.md) |
| [entity-to-dial](entity-to-dial/report.md) → [Evidence-insensitivity](evidence-insensitivity/report.md) | 方法參考 | state difference 與 span × 層 patching 方法於 Phase 3 沿用。 [依據](evidence-insensitivity/details/proposal-phase3.md) |
| [jacobian-lens-selection](jacobian-lens-selection/proposal.md) → [公司身分的中間概念](entity-concept-decision/report.md) | 資料／產物依賴（proposed） | Phase 1 以既有 canonical lens 提名；介入不以讀出代替因果驗證。 [依據](entity-concept-decision/details/proposal-phase1.md) |

</details>
