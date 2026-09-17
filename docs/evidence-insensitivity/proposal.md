# Evidence-insensitivity 研究線入口

**文件定位：全線入口，不是新增協議。** 本頁只導覽研究問題、階段狀態與設計決定；
各 phase 的原始協議在 `details/`，其 frozen 狀態與 gate 結果以該文件為準。本頁不
授予新的 run authorization。

**狀態**：進行中——Phase 1 **completed**（Qwen，2026-09-16，formal run
`phase1-gpu-bf16-01`，G-P1..P4 全過、50 responsive / 453 fixed-sell frozen；
Gemma-4-E2B 同設計 formal 執行中）；Phase 2 **frozen**（Rev 1.1；實作完成、
Qwen run 待啟動、Gemma 待其 Phase 1 formal 完成）；Phase 3 planned。主線模型：
Qwen3.5-4B＋Gemma-4-E2B（用戶指示統一雙模型）。

## 全線問題

在 S&P 500 全人口（2024 名單 503 家）下，以全部公司共用的 company-agnostic
證據控制，分出 buy/sell 輸出不跟隨證據淨極性的公司（evidence-insensitive，
entity prior 強）與輸出跟隨證據極性的公司（evidence-responsive），並研究兩組的
L15 instruction span 狀態差與該差異的上游因果來源。

本線回答的交互項是既有線都沒測過的：**entity × evidence**——同公司翻證據極性，
輸出跟不跟。

## 階段路由

| 階段 | 協議 | 狀態 |
|---|---|---|
| **Phase 1：行為篩選與分組** | [Phase 1 協議](details/proposal-phase1.md) / [Phase 1 報告](details/report-phase1.md) | **Completed（2026-09-16）**：formal run `phase1-gpu-bf16-01`（5336 forwards）G-P1..P4 全過、fallback 未觸發；503 家分組表 frozen（50 evidence-responsive / 453 fixed-sell / 0 fixed-buy / 0 mixed；503/503 零證據 sell）；order-swap 臂顯示項目位置為強、非對稱的決策驅動（recency，正項在尾 69–90% sell→buy）。 |
| **Phase 2：capture-layer 組間對比** | [Phase 2 協議](details/proposal-phase2.md) | **Frozen（Rev 1.1，2026-09-16 批准）；實作完成（unit tests 過）、Qwen run `phase2-gpu-bf16-01` 待啟動**：雙模型統一執行；Qwen capture layer = L15（frozen 錨點）、Gemma 以 16 家子樣本 Step A 定位 L*；discovery split 402 家 × {zero, N15, P15} 純 forward 狀態捕獲（不存 raw states）；stance 軸新 family 重導出；per-company 狀態反應曲線、offset vs gain 分解、2×2 交叉表、sector 控制。 |
| **Phase 3：上游因果定位** | 待 Phase 2 完成後起草 | Planned：within-company 極性狀態差 × span（entity/evidence/instruction）× 層 sweep；transfer patching；hold-out confirmation。 |

## 關鍵設計決定（2026-09-16 brainstorming，用戶批准）

1. **人口**：S&P 500 全名單（2024，503 家），不取 16/64 子集。
2. **證據控制**：全部公司共用同一組 1 正＋1 負的 company-agnostic 證據，淨極性由
   百分比梯子（主項 6/8/10/15%、次項 5%）控制；trial plan 的 per-company 證據不作
   V1 主數據，降為未來 robustness 延伸。
3. **行為端點**：greedy generation 實際產出的 JSON（decision + reason）為 primary；
   decision-position margin 為 auxiliary（回應曲線、tie-break、機制相連續計分）。
4. **分組**：between-company、pre-registered（看內部狀態前凍結規則與 fallback）。
5. **機制相止點**：V1 只做到描述＋上游因果定位；不含干預臂（移除 prior 無
   副作用）——若做，另立 V2 版本，座標依 Phase 3 結果決定。

## 明確不做（本輪）

- 單神經元因果搜尋（三次 prior null：balanced-evidence-gap Phase 3 0/3、
  entity-to-dial Phase D、financial-soundness）。
- 重測 stance 軸／k=8 子空間（已收線結論；Phase 2 只正交化）。
- cross-model 已入主線：Qwen3.5-4B＋Gemma-4-E2B 雙模型統一執行（用戶 2026-09-16
  指示）；qwen3.6-27b 延伸仍待機制確認後（用戶決定）。
- 匹配公司集（sector/marketcap/prior matching）：等本線有結果後作為 Phase 3 的
  refine 選項（用戶決定）。
- 干預臂、非 JSON 答案位置、其他任務、多語言。

## 與其他研究的前後關係

此節為文件導覽，不改動本研究協議。關係定義與全線來源對照見[研究總覽](../README.md)。

**上游**

- [balanced-evidence-gap](../balanced-evidence-gap/report.md)（研究承接）：共享
  平衡證據下的 entity gap 現象（16 家 64/64 sell、~1.5 nats spread）動機化本線；
  其共享證據設計與 identity-stripped 慣例被沿用。
- [jspace-token-experiments](../jspace-token-experiments/proposal.md)（資料／產物
  依賴）：零證據 header-only 模板與強 sell prior 觀察（6 tickers，M −3.39～−4.93）；
  本線把零證據量測擴到 503 家。
- [investment-dial](../investment-dial/report.md)（方法參考）：JSON 格式的 prompt
  family、fixed answer-token margin 慣例、全域立場 prior 的背景結果。
- [entity-concept-decision](../entity-concept-decision/proposal.md)（方法參考）：
  stance 軸 derive 與正交化方法（Phase 2 沿用）；其「決策 = 微小公司間差的高增益
  讀出」結論是 Phase 2 解讀的背景。
- [entity-to-dial](../entity-to-dial/proposal.md)（方法參考）：state difference 與
  span × 層 patching 方法（Phase 3 沿用）。

**輔助診斷**

- [Cross-model probe（Llama-3.2-1B/3B、Gemma-4-E2B）](details/diagnostic-cross-model-probe.md)：
  同模板四模型對照——零證據保守 sell 預設為模板共通底層（prompt 層）；證據
  反應結構與 entity 分化為模型特性（Qwen 分級跟隨＋10% 知名股跟隨；Llama
  全面保守；Gemma「有證據就 buy」反向模式）→ 本線研究的現象是模型特性，
  非模板假象。

**後續（proposed，未立案）**

- selective-intervention V2：Phase 3 定位結果若支持，決定 entity prior 干預臂的
  座標與設計。
- cross-model 延伸（qwen3.6-27b、gemma4-31b 量化版）：機制確認後；三個家族
  （Qwen/Llama/Gemma）probe 已完成（見上）。

`data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv`（427 家 per-company
證據）不作為 V1 數據來源；保留為未來 robustness 延伸（per-company 證據下組結構
是否成立）。

## 閱讀邊界

Phase 1 是行為分組（discovery）；Phase 2 是描述性組間對比；Phase 3 才是因果
定位。各 phase 的 input、output、control 與容差只由 `details/` 對應協議維護；
本頁狀態摘要過時時以 `details/` 為準。
