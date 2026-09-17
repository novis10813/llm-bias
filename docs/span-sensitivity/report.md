# Header-span Sensitivity：同產業置換具顯著邊際敏感度（V1 Discovery）

**狀態：已完成（V1 discovery 完成，calibration/test 協議未凍結）。** 模型為 Qwen3.5-4B（bf16）；2026-08-31 完成。原始操作契約與家族設計見 [實驗提案](proposal.md)。

**一句話發現：** 在財務證據固定下，header 中同產業公司置換產生顯著決策偏移（平均 ΔM = −1.39 nats，12/35 翻轉）並超越表面亂碼控制，獲選為 primary condition；但純匿名化操作無法排除表面文字擾動。

## 1. 在證據固定下，Header 中的代號與公司名稱置換是否引起決策邊際偏移？是

我們在財務證據內文完全固定的前提下，系統性修改提示詞開頭 Header 中的股票代號與公司名稱，觀察 Buy-minus-Sell continuation margin 的變化：

**觀察（35 個 Technology 股票）：**
- **同產業置換效應顯著**：`same_sector_swap` 產生平均 $\Delta M = \mathbf{-1.3857}$ nats（中位數 −1.2500，95% CI [−1.7465, −1.0500]，Holm $p = 0.00300$），誘發 **12/35 次** margin 符號翻轉（sell $\to$ buy 或 buy $\to$ sell）。
- **人工構建身分次之**：`constructed_identity` 產生平均 $\Delta M = -1.2107$ nats（95% CI [−1.5143, −0.9179]，9/35 翻轉）。

**解讀：** 在證據不變時，單純替換 Header 處的企業身分足以誘發實質的邊際決策移動。

## 2. 這一偏移能否與字元表面變更（表面擾動）明確區分？僅部分成立

**觀察：**
- **亂碼表面對照組**：表面亂碼與字母替換控制組 `name_form_control` 產生平均 $\Delta M = \mathbf{-0.8286}$ nats（95% CI [−1.0607, −0.5892]，5/35 翻轉）。
- **匿名化各條件未超越表面對照**：
  - `anonymous_name`：平均 $\Delta M = -0.1893$ nats；
  - `anonymous_identity`：平均 $\Delta M = -0.5929$ nats；
  - `anonymous_ticker`：平均 $\Delta M = -0.0107$ nats（95% CI [−0.0929, +0.0786]，跨越 0，無顯著效應）。

**解讀：** `same_sector_swap`（1.3857）的效應顯著大於表面控制組（0.8286），具備超越純文字擾動的實質影響；但單純匿名化操作的效應均小於表面控制組，無法排除表面分詞（Tokenization）擾動的影響。

## 3. Discovery 階段形成了何種後續決策與邊界？

**決策與邊界：**
1. **Primary 條件凍結**：僅將 `same_sector_swap` 列為後續校準階段的 Primary 條件；`constructed_identity` 列為次要診斷；`name_form_control` 保留為必要表面控制組。
2. **非完整實體因果**：本實驗僅測量既有證據之上的 Header 邊際敏感度，不外推為模型內部完整的實體因果機制。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| 原始提案與條件設計 | [實驗提案](proposal.md)（V1 discovery）。 |
| Discovery 執行記錄與產物 | run `tech-header-discovery-gpu1-20260831T090100Z`，位於 `artifacts/qwen3.5-4b/technology-header-span-sensitivity/runs/`；245 筆記錄，Bootstrap 2000 次。 |

**本次編輯說明：** 本報告按三項核心問題改寫，明確記錄各條件效應與表面控制組對比；原始協議 `proposal.md` 完整保留。
