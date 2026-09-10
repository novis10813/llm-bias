# Balanced Evidence Gap — Phase 2 報告：Entity-to-Decision Path 定位

## 執行概覽

| 項目 | 值 |
|---|---|
| 模型 | Qwen3.5-4B（`.cache/models/qwen3.5-4b` → `/mnt/f/models/Qwen3.5-4B`），bf16，GPU 0 |
| 協議 | [proposal-phase2.md](proposal-phase2.md)（Rev 1）＋ [proposal-phase2-rev2.md](proposal-phase2-rev2.md)（gate 2A Rev 2） |
| 2A run | `phase2a-gpu-bf16-01`（64 prompts，7 分 01 秒，manifest 6/6） |
| 2A Rev 2 gate | `phase2a-rev2-gate-01`（CPU-only re-analysis，4/4） |
| 2B run | `phase2b-gpu-bf16-01`（8 directions × 32 layers × 4 spans，22 分 41 秒，manifest 5/5） |
| 2C run | `phase2c-gpu-bf16-05`（attention 5 層 × 16 heads ＋ MLP 20 層，1 小時 12 分，manifest 5/5） |
| 2C gate 重評 | `phase2c-gate-reanalysis-01`（CPU-only，gate 實作修正後重算，4/4） |
| run root | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/` |
| 狀態 | **2A / 2B / 2C 全部 completed；gate 2A（Rev 2）與 gate 2C 通過** |

前置：Phase 1（[report-phase1.md](report-phase1.md)）已確認 entity-induced
decision gap 的行為存在（named-vs-anonymous gap +0.432 nats，CI
[+0.350, +0.518]）。2A 詳細結果見 [report-phase2a.md](report-phase2a.md)。

## 核心結論

1. **Entity-specific 的決策相關狀態在早期層（L0–11）的 entity position
   殘差流中**：2B entity span patching 的 normalized transfer 在 L0–5
   ≈ 0.95–1.01，toward-source ΔM 的 bootstrap CI 只在 L0–11 排除 0。
2. **Handoff 在 L12–15**：instruction span transfer 從 L12 上升、L15 達
   峰值（T=+0.464），與 investment-dial 的 L15 和 causal tracing 的
   L16 instruction-context peak 在同一區間。crossover band（context T ≥
   entity T）為 L13–31。
3. **Handoff 區間內的元件歸因（gate 2C）：MLP arm 通過、attention arm
   null**。MLP down-projection 在 L19（n6334，ρ=−0.897）、L20
   （n6520，ρ=+0.894）、L26（n2394，ρ=−0.859）各有 1 個 channel 的
   attribution 與 pure entity margin 的相關勝過全部 10 個 matched
   controls，Holm-adjusted sign-flip p = 0.0418 < 0.05，sector
   agreement 4/4。80 個 attention head（5 個 full-attention 層）的
   entity→final edge zeroing 效果全部近零（max +0.0141 nats，Holm
   p = 1.0）。
4. **Investment-dial coordinate（L15/n8490）不是 entity-specific**：
   2A H4 顯示其 activation 跨公司近乎平坦（−0.032 到 +0.074），與 pure
   entity margin 的 Spearman 僅 +0.159（entity position）/ +0.106
   （final position）。

## 2A：Cross-entity probe（摘要）

64 條 prompt（16 tickers × 2 reverse × 2 order，frozen shared-evidence
template）全數 sell；pure entity margin 從 BLK −1.053 到 IT −2.568，
IQR 0.570 nats。Rev 1 gate 2A 在 Spearman vs Phase 1 named margin 項
失敗（ρ = −0.411，構念不匹配：named margin 混入公司特定證據效果）；
Rev 2 協議改以 Phase 1 gap（named−anon，純 entity 構念）為參照並加入
group construct check，5 項全過（IQR 0.570；Spearman vs gap +0.448；
top NSC/BLK vs bottom IT/BDX 的 4/4 pairwise gap 正向；framing 0.365；
schema 1.0），2B 獲授權。群組：bottom = {IT, BDX}、top = {NSC, BLK}，
8 個 directions。

## 2B：Entity-state layer sweep

8 個 directions（top↔bottom 雙向）× L0–31 × 4 spans；每 (layer, span)
為 8 個 direction 的 toward-source ΔM 與 normalized transfer T 的
mean（CI 僅在 ≥4 directions 時計算）。

| Span | 行為 | 關鍵值 |
|---|---|---|
| entity | L0–5 近乎完整轉移，L7–12 急降，L16 後 ≈ 0 | T: L0 +1.011 → L11 +0.441 → L15 +0.033 → L31 −0.037；ΔM CI 排除 0 止於 L11（L11 [+0.079, +1.108]；L12 [−0.096, +0.810]） |
| evidence（傳播診斷） | L6 起上升，L10–12 平台，L19 後 ≈ 0 | 峰值 L12 T=+0.432 |
| instruction | L0–11 ≈ 0，L12–15 上升，L15 峰值，之後緩降 | 峰值 L15 T=+0.464 |
| final（control） | 全程近零 | ≈ −0.04 |

圖表：`docs/assets/balanced-evidence-gap/phase2b_layer_sweep.{pdf,png}`
（renderer：`scripts/plot_balanced_evidence_gap_phase2b.py`）。

**解讀**：把 source 公司在 entity position 的殘差狀態複製進 target
公司 prompt，只要 patch 點在 L11 之前，target 的決策 margin 就移向
source（L0–5 近乎完整）；L12 之後 entity span 的狀態不再承載決策
影響，同時 instruction span 的 patch 效果在 L12–15 達到峰值。即
entity-specific 狀態在 L0–11 被寫入殘差流，於 L12–15 被讀取並整合進
instruction context，下游層對 entity span 不再敏感。evidence span 在
L6–12 出現 0.43 的轉移，表示 entity 訊號已部分洩漏到共享證據位置的
狀態（與 Rev 1 協議對 evidence span 的定位一致：傳播診斷，非 no-op
control）。

## 2C：Component attribution（handoff 區間 L12–31）

### Attention arm（null）

5 個 full-attention 層（L15/19/23/27/31）× 16 heads，每個 head 對 8
個 directions 做 entity→final edge zeroing，paired difference =
ΔM(entity zeroing) − mean ΔM(10 個 position-matched 隨機位置
zeroing)，Holm 調整跨 80 heads。80 個 head 的 mean effect 介於
−0.01 到 +0.0141 nats（44/80 為正），全部 Holm p = 1.0；reconstruction
error 全部遠低於 fail-closed 門檻（max ≈ 0.003 vs floor 0.001/rel
0.02）。**結論：handoff 區間內沒有單一 attention head 承擔可測的
entity→final 路由**（在此 probe 的粒度下）。

### MLP arm（pass：L19、L20、L26）

20 層（L12–31）× 16 tickers 的 differentiable attribution
（|∂M/∂a|·|a| 於 entity position 的 MLP down-projection 輸入，9216
維，僅持久化 top-20）；每層 top neuron = argmax |Spearman(attribution,
pure entity margin)|，matched controls = 10 個隨機 neuron（seed
42+layer，correlation 於 forward 階段計算）。

| 層 | top neuron | ρ | max control | 原始 p | Holm p | sector agreement |
|---|---|---|---|---|---|---|
| **L19** | 6334 | **−0.897** | 0.618 | 0.0021 | **0.0418** | 4/4 |
| **L20** | 6520 | **+0.894** | 0.635 | 0.0021 | **0.0418** | 4/4 |
| **L26** | 2394 | **−0.859** | 0.538 | 0.0021 | **0.0418** | 4/4 |
| L12（參考） | 701 | +0.918 | 0.682 | 0.0106 | 0.181 | 4/4 |

L12–L30 的每層 top neuron 都勝過該層全部 10 個 controls（|ρ| 0.83–
0.92 vs control 0.43–0.70），但 Holm 調整（跨 20 個被測 neuron）後
只有 L19/L20/L26 通過 p < 0.05。**Gate 2C 依協議的 per-arm 存在性
判斷通過（MLP arm）。** 三個通過層的 ρ 符號不一致（−/+/−），表示
這些 channel 分別以相反方向編碼 margin（buy-leaning 或 sell-leaning
coordinate）。L31 的 attribution 為精確的 0：最終塊之後沒有下游
讀取 entity position 的組件，屬於結構性零，非測量結果。

圖表：`docs/assets/balanced-evidence-gap/phase2c_components.{pdf,png}`
（renderer：`scripts/plot_balanced_evidence_gap_phase2c.py`）。

## Gate 2C 實作修正記錄

2C 正式 run（`phase2c-gpu-bf16-05`）的 analyze stage 判定 gate fail，
原因是 gate 實作偏離 frozen 協議：協議 §4.5 定義「attention 臂與 MLP
臂各自獨立判定：**至少 1 個 head 或 1 個神經元**的 top 效應 mean >
matched control mean，且 pair sign-flip test（Holm 調整）p < 0.05」與
「top 效應的 sector-stratified 符號一致率 ≥ 3/4 sector」，即 per-arm
**存在性**判斷；原實作對 MLP arm 做跨層 min/max 匯總（max top ＝
L12、max control ＝ L24、min sector ＝ L31、max p ＝ L31），把不同層
的統計混在一起，讓結構性全零的 L31 污染判定。修正後（存在性判斷＋
Holm 跨被測層）以 CPU-only re-analysis（`phase2c-gate-reanalysis-01`，
provenance 含 source records 的 SHA-256）重算：attention arm null 不變，
MLP arm 於 L19/L20/L26 通過，**gate 2C pass**。run 05 的原始 summary
維持不變。

## 跨線整合

本線測到的路徑與既有定位線的結果互相咬合：

| 線 | 定位 | 與本線的關係 |
|---|---|---|
| Entity cell | L0–4 fact memory | 2B entity span sufficiency 的起點（L0–5 T≈1.0）與 entity cell 的早期層 fact memory 一致 |
| Causal tracing | L6 evidence 97.56%；L16 instruction-context peak | 2B evidence span 轉移自 L6 起、instruction 峰值 L15（±1 層內） |
| Investment-dial | L15/n8490 model-level stance prior | 2B instruction 峰值同層；2A H4 與本線 2C 均不顯示 entity-specific 成分 |
| J-space V2 prior probe | zero-evidence gap ≤ 0.4% | 與「L15 prior 非 entity-specific」一致 |

綜合：**entity 身份在 L0–11 寫入 entity position 的殘差狀態（承載
決策影響），於 L12–15 被讀取並整合進 instruction context（L15 同時是
model-level stance prior 的層），下游 L19/L20/L26 的個別 MLP channel
持續編碼 entity-specific margin，直到 final readout。** 2C 的 attention
null 表示 entity→final 的路由不靠 handoff 區間內的單一 attention head
（至少不在本 probe 的粒度與層範圍內）。

## 限制

- 16 家公司、8 個 directions；2B 的 CI 基於 8 個 direction 的
  bootstrap，2C 的 sign-flip 基於 16 個 ticker。
- 2C MLP 的 attribution 是 local first-order sensitivity（|∂M/∂a|·|a|），
  不是 causal 干預；「top neuron 相關勝過 controls」是關聯證據。
- Attention arm 的 null 只覆蓋 5 個 full-attention 層 × 16 heads；
  linear-attention 層（24 層）無可索引的 attention weight，未被 probe。
- 2C 的 handoff 區間（L12–31）來自 2B 的 crossover band ±1，範圍寬
  （20 層），MLP arm 的 Holm 調整跨 20 個被測 neuron。
- 0/64 buy（2A 共享證據 net 偏空）：2B/2C 的 contrast 完全在 sell 半邊
  操作；top/bottom 指派依賴 pure margin 的相對排序。
- 2C 的正式執行共 5 次嘗試：run 01/02 因程式 bug 失敗（已修復並
  commit），run 03/04 因系統重開機與 WSL 關閉中斷（manifest 標記
  failed），run 05 完成。

## 再現性

```bash
# 2A（約 7 分鐘 GPU）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \
    --model .cache/models/qwen3.5-4b --run-id phase2a-gpu-bf16-01

# 2A Rev 2 gate（CPU-only）
uv run --no-sync python scripts/balanced_evidence_gap_phase2_rev2.py \
    --model .cache/models/qwen3.5-4b --run-id phase2a-rev2-gate-01

# 2B（約 23 分鐘 GPU；--gate-run 指向 Rev 2 gate run）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/balanced_evidence_gap_phase2_patch.py sweep \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --gate-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-rev2-gate-01 \
    --run-id phase2b-gpu-bf16-01

# 2C（約 72 分鐘 GPU）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/balanced_evidence_gap_phase2_patch.py attribute \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \
    --run-id phase2c-gpu-bf16-05

# 2C gate 重評（CPU-only）
uv run --no-sync python scripts/balanced_evidence_gap_phase2c_gate_reanalysis.py \
    --model .cache/models/qwen3.5-4b \
    --phase2c-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2c-gpu-bf16-05 \
    --run-id phase2c-gate-reanalysis-01

# 圖表
uv run --no-sync python scripts/plot_balanced_evidence_gap_phase2b.py
uv run --no-sync python scripts/plot_balanced_evidence_gap_phase2c.py
```

所有 artifacts 為 compact 派生值（margin、T、ΔM、top-k attribution、
統計量與 provenance），不含 raw activation/residual。

## 狀態與後續

- Phase 2（2A/2B/2C）：**completed**；gate 2A（Rev 2）與 gate 2C 通過。
- 本報告為 discovery 性質；2C 的 MLP channel 是關聯證據，若要 causal
  確認（例如對 L19 n6334 做 targeted intervention）需新版本協議。
- 候選後續線（未授權）：(a) L19/L20/L26 top neuron 的 causal
  validation；(b) 跨模型重複（其他 decoder LLM 上重跑 2B/2C）；
  (c) entity span patching 的 downstream 行為效應（決策翻轉率）。
