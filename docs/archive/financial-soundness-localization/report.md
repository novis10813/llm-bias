# 財務穩健定位：跨句式候選方向重現，但受答案形式干擾（探索性結果）

**狀態：已完成（探索 V1）。** 模型為 Qwen3.5-4B（bf16）；2026-09-07 完成，探索性定位，未設 formal 認證 gate（`certified=false`）。

**一句話發現：** 全層掃描篩選出 72 個財務穩健候選座標，其中 70 個在 held-out 句式下反應方向一致；但模型正確率高度受答案形式與標籤順序干擾（Direct 87.5% vs Labels-reversed 62.5%），未獲財務功能認證。

## 1. 能否在全層 MLP 通道定位出反應方向穩定的財務穩健候選？方向重現，但公司受限

我們使用話題（topic）、公司評價（company）、財務證據（evidence）與非財務比較（comparison）四類短句，在 Qwen3.5-4B 上掃描 L0–L31 全層：

**觀察：**
- **全層掃描出 72 個候選座標**：在全部 234 對提示詞中，保留各 family/concept/answer_mode 的 top-3，共取得 72 個候選。
- **70/72 個候選在 held-out 符號同號**：在 held-out 測試集中，70 個候選的平均配對反應差與 discovery 階段同號。

**解讀：** 候選通道在未見句式與未見數值下展現了方向上的重現性，但受測公司名稱仍限於 JNJ 與 BAC，本結果僅屬「未見句式驗證」，不能外推為「未見公司驗證」，亦不代表效應幅度穩定或具備因果重要性。

## 2. 財務判斷任務是否受到答案標籤與題型難度干擾？干擾顯著

我們在 direct（直接詞彙續接）、labels-forward（A 對應正面）與 labels-reversed（交換標籤語義）三種答案形式下測試各 8 道題目（財務 evidence 與非財務 comparison）的正確率：

**觀察：**
- **財務證據題正確率**：Direct 模式在 calibration 與 held-out 均為 87.5%，但在 Labels-reversed 模式下 held-out 僅 62.5%（discovery 更低至 50%）。
- **非財務對照題正確率**：Comparison 題在 Direct 下各 split 均為 50%，而在 Labels-reversed 下 held-out 達 100%。

| 任務與答案形式 | discovery | calibration | held-out |
|---|---|---|---|
| 財務 evidence / direct | 50% | 87.5% | 87.5% |
| 財務 evidence / labels-forward | 87.5% | 62.5% | 87.5% |
| 財務 evidence / labels-reversed | 50% | 50% | 62.5% |
| 非財務 comparison / direct | 50% | 50% | 50% |
| 非財務 comparison / labels-forward | 50% | 50% | 75% |
| 非財務 comparison / labels-reversed | 75% | 87.5% | 100% |

**解讀：** 模型對答案標籤的順序映射敏感，且財務題與非財務題基準難度不一致；候選激活排名無法排除詞彙或標籤映射偏好，不可單憑排名命名財務概念。

## 3. 這些候選是否具備因果特異性？尚未認證

**觀察與後續：** 本輪為探索性篩選，未設認證 gate。全部 72 個候選無條件直接傳遞至後續因果驗證流程，未在 held-out 階段提前人為挑選表現較佳的單元。

## 4. 研究宣稱之邊界與未涵蓋事項

1. **探索性定位**：未配置正式合格門檻，非 formal confirmation。
2. **樣本規模極小**：公司僅覆蓋 JNJ 與 BAC，每個 split 僅兩組獨立數值對。
3. **無因果宣稱**：僅量測 MLP 通道激活反應差，非因果介入效果。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| 探索協議與題型定義 | [定位協議](proposal.md)（V1 proposed）。 |
| 執行記錄與產物 | run `financial-localization-exploratory-v1`，位於 `artifacts/qwen3.5-4b/financial-soundness-localization/runs/`；資料集 `data/baseline/financial-soundness-localization/exploratory-v1.json`。 |
| 工程前檢記錄 | [smoke 報告](details/report-smoke.md)。 |
| 後續因果驗證報告 | [因果驗證報告](../financial-soundness-causal-validation/report.md)。 |

**本次編輯說明：** 本報告按三項核心問題改寫，明確標記探索性定位與未獲認證狀態；原始協議 `proposal.md` 完整保留。
