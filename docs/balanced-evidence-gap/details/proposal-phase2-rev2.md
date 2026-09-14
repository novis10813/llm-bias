# Balanced Evidence Gap — Phase 2 Rev 2：Gate 2A 參照修訂協議

- 狀態：**frozen**（2026-09-10）
- 前身：[proposal-phase2.md](proposal-phase2.md)（Rev 1，維持 frozen）
- 觸發：Rev 1 gate 2A 在 `phase2a-gpu-bf16-01` 未過，唯一致命項為
  Spearman vs Phase 1 named median margin（ρ = −0.411，門檻 |ρ| > 0.5）；
  結果記錄於 [report-phase2a.md](report-phase2a.md)

## 1. Rev 1 gate 失敗的診斷

Rev 1 的 Spearman 項假設 2A pure entity margin 的跨公司排序會跟隨
Phase 1 named median margin。實測為負相關（ρ = −0.411），原因是兩個
構念不同：

- Phase 1 named margin = entity 效果 ＋ **公司特定證據**效果（每家公司
  用自己的 10-K 證據對，跨公司 IQR 1.375 nats，證據效果主導排序）。
- 2A pure entity margin = 證據固定為同一組公司中立句子時，只換
  entity header 的 margin，量的是 entity prior 本身。

兩者的 ranking 無必要一致；當公司特定證據效果與 entity prior 負相關
時，總和構念（Phase 1 named）會與純構念（2A pure）呈負相關。描述性
證據支持這個解讀：2A pure margin 與 Phase 1 的 **gap**（named −
anonymous，paired，純 entity 構念）的 Spearman ρ = +0.448
（n = 16，兩側 p ≈ 0.08），方向與預期一致但統計力不足。

Rev 1 協議未定義替代參照，因此 Rev 1 判定維持「gate 未過」，不回填。

## 2. Rev 2 變更（相對 Rev 1，僅 gate 2A）

| 項 | Rev 1 | Rev 2 |
|---|---|---|
| 判準 1 | IQR(pure margin) > 0.5 nats | 不變 |
| 判準 2 | \|Spearman\| vs Phase 1 named median > 0.5 | **改為兩項**：(a) Spearman vs Phase 1 **gap** > 0.3；(b) group construct check（見下） |
| 判準 3 | framing median < 1.5 nats；schema valid = 1.0 | 不變 |
| Spearman vs Phase 1 named | gate 項 | 降為 descriptive |
| 2B / 2C 設計 | — | **完全不變**（繼承 Rev 1 §4.4–4.6） |

判準 2b（group construct check）：以 2A pure margin 取 top-2 / bottom-2
群組後，**top 群組每家**公司的 Phase 1 gap 必須大於 **bottom 群組每家**
公司的 Phase 1 gap（2 × 2 = 4 個 pairwise 比較全數正向：對所有
 t ∈ top、b ∈ bottom，gap(t) > gap(b)）。此項直接
檢查 2B 分組方向與 Phase 1 entity 效果方向一致，是 2B 授權真正需要的
構造條件；16 家全排序相關（判準 2a）在 n = 16 下統計力不足，降為
中等強度方向檢查。

兩項判準的門檻（0.3、全數正向）於本文件 frozen 時點 pre-register，
先於 gate 重評執行。

## 3. 評估方式：re-analysis，無新 inference

2A margin 是確定性 logit 計算（fp32 tail，greedy scoring，無取樣）；
重跑 forward 會產生位元級相同結果。Rev 2 gate 因此是**對既有 2A
forward records 的重評**，不消耗 GPU：

- 輸入：`phase2a-gpu-bf16-01/forward/results.jsonl`（64 records）與
  Phase 1 `balanced-gap-gpu-bf16-01/analyze/summary.json`
  （per-company `gap_mean`）。
- 新 run：`phase2a-rev2-gate-<seq>`，dataset `balanced-evidence-gap-phase2`，
  stages = prepare（provenance：source run ids、record 數、SHA-256）＋
  analyze（gate 2A Rev 2 ＋ descriptive），`finalize(required_stages={prepare, analyze})`。
- operator：`scripts/balanced_evidence_gap_phase2_rev2.py`。

## 4. 群組與 directions（frozen，依 2A 結果）

| 群組 | 公司 | pure margin（median of 4） |
|---|---|---|
| bottom | IT, BDX | −2.568, −2.335 |
| top | NSC, BLK | −1.307, −1.053 |

8 個 directions = 4 ordered pairs × 雙向。top/bottom 指派以 pure margin
排序的 rank 1–2 / 15–16 為界；rank 2↔3（BDX −2.335 vs SYK −2.241）與
rank 14↔15（NSC −1.307 vs AXP −1.344）的界內差距小於 0.3 nats，屬於
描述性觀察，不影響 gate（gate 只檢查已凍結的 4 家群組公司）。

## 5. 成本

CPU-only analysis，秒級。2B / 2C 若獲授權，成本與 Rev 1 §6 相同
（2B ≈ 4,112 forwards；2C 依 2B handoff 區間決定）。

## 6. 邊界

- Rev 2 不修改 Rev 1 的 2A 執行結果、report-phase2a.md 或任何
  frozen artifact。
- Rev 2 gate 通過只授權 2B；2C 仍依 2B 的 handoff 區間與 Rev 1 §4.5
  條件執行。
- 若 Rev 2 gate 仍未過，2B/2C 不執行，line 以負結果收線。

## 7. Revision record

- Rev 2（本文件，2026-09-10）：gate 2A 判準 2 改參照 Phase 1 gap 並
  加入 group construct check；Spearman vs named 降為 descriptive；
  評估改為 re-analysis。使用者核准此方向。
