# Sector/Context Follow-up：L16 指令狀態具實體敏感性但未達產業特異（B V1 校準負結果）

**狀態：已完成（B V1 校準未通過，`success=false`，held-out test 未執行；A/B/C discovery 完成）。** 模型為 Qwen3.5-4B（bf16）；2026-08-31 完成。原始操作契約與 A/B/C 設計見 [實驗提案](proposal.md)。

**一句話發現：** 在固定負面證據下，跨產業抽換 L16 指令狀態能顯著移動決策 margin（Toward-Source +0.178 nats，勝過 Header 對照），但同產業 peer 控制組的位移更大（|ΔM| = 0.357 > 0.229），未通過產業特異性檢驗，否定產業專屬中介假設。

## 1. 跨產業置換 L16 指令狀態能否轉移決策傾向且勝過 Header 對照？成立

在固定負面財務證據下，我們將目標公司的 L16 `instruction_context` 殘差狀態替換為另一產業來源公司的對應狀態：

**觀察：**
- **Toward-Source 效應顯著**：跨產業 L16 context 抽換的 Toward-Source 平均值為 **+0.17805 nats**（95% CI [+0.08417, +0.27129]，Holm 校正後 $p = 0.004395$），穩定將 target margin 往 source clean margin 方向推動。
- **大幅勝過同層 Header 置換**：L16 Context–Header 對比均值為 **+0.17690 nats**（95% CI [+0.06918, +0.27631]，Holm 校正後 $p = 0.004883$）。
- **通過 10/11 項門檻**：校準集通過包括最小配對數、No-op 契約、雙向證據分層等 10 項判準。

**解讀：** 中期層的 Header 狀態已退化為 No-op，而證據之後的指令上下文狀態保留了充分的決策偏好資訊。

## 2. 該效應是否具備產業特異性（Sector Specificity）？未通過，同產業對照效應更大

**觀察：**
- **未通過 Same-sector Peer 特異性 Gate**：
  - 跨產業 L16 context 抽換的絕對位移量 $|\Delta M| = \mathbf{0.22870}$ nats；
  - 同產業同儕（Same-sector Peer）context 抽換的絕對位移量 $|\Delta M| = \mathbf{0.35662}$ nats。
- 同產業同儕抽換產生的絕對擾動反而比跨產業替換大 **0.12792 nats**。

**解讀：** 依預先凍結規則，Gate 判定為 **FAIL**（`calibration_success = false`，`test_authorized = false`），Held-out 測試依規則未執行。該結果否定了「L16 指令區間存在特定產業專屬表徵通道」的假說，轉而支持更寬泛的公司身分（company-conditioned）敏感度。

## 3. 這項結果對模型表徵解剖的意涵為何？支持實體敏感性，否定產業通道

**觀察與限制：**
1. **實體敏感性成立**：更換指令狀態能顯著移動決策，證明模型在 L16 指令區間保留了公司身分先驗。
2. **否定產業專屬中介**：同產業同儕替換同樣引發大幅移動，現有 controls 無法將其收窄為產業層級機制。
3. **充分性測量邊界**：本實驗僅測量固定負面證據下的狀態抽樣充分性，不證明 L16 狀態是唯一或不可或缺的決策成分。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| 原始提案與 A/B/C 操作契約 | [實驗提案](proposal.md)（含 A 跨產業 header、B 負證據 context、C L16 讀出）。 |
| B V1 校準執行紀錄與產物 | run `cross-sector-context-calibration-20260831T092659Z`，位於 `artifacts/qwen3.5-4b/cross-sector-context-overriding/runs/`；`analyze/confirmation.json`。 |
| A/B/C Discovery 完整報告 | [Discovery 報告](details/report-discovery.md)（含三臂初步掃描結果）。 |

**本次編輯說明：** 本報告按三項核心問題改寫，明確記錄校準門檻未過原因與產業特異性否定結論；原始協議 `proposal.md` 完整保留。
