# Balanced Evidence Gap — Phase 2A cross-entity probe 報告

## 執行概覽

| 項目 | 值 |
|---|---|
| run ID | `phase2a-gpu-bf16-01` |
| 模型 | Qwen3.5-4B（`.cache/models/qwen3.5-4b` → `/mnt/f/models/Qwen3.5-4B`），bf16，GPU 0 |
| 協議 | [proposal-phase2.md](proposal-phase2.md) Rev 1（2A 段落） |
| operator | `scripts/balanced_evidence_gap_phase2.py` |
| run root | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01` |
| prompts | 64（16 tickers × 2 reverse options × 2 evidence orders，frozen shared-evidence template） |
| 狀態 | complete（manifest 6/6 SHA-256 驗證通過，2026-09-10） |
| 時長 | 7 分 01 秒（含 model loading） |

2A 的設計：16 家公司共享同一組 frozen 證據（2 正 2 負、公司中立），只改變
entity header（ticker ＋ 公司名）。per-company pure entity margin 是 4 個
variant 的 median。gate 2A 的 role 是確認這個 pure entity margin 的跨公司
分布足以支持 2B 的 top/bottom 分組，且其 ranking 與 Phase 1 觀察到的 entity
效果一致。

## Gate 2A：未通過（Spearman 項）

| 判準 | 值 | 門檻 | 結果 |
|---|---|---|---|
| 跨公司 IQR（pure entity margin） | 0.570 nats | > 0.5 | 通過（邊緣） |
| framing stability（reverse pair margin diff 的 median） | 0.365 nats | < 1.5 | 通過 |
| schema valid rate | 1.000 | ≥ 1.0 | 通過 |
| \|Spearman ρ\| vs Phase 1 named median margin | 0.411（ρ = −0.411） | > 0.5 | **未通過** |

依協議，2B（entity-state layer sweep）以 gate 2A 全項通過為授權條件，
**2B 未獲授權，未執行**。

## 核心結果

### 共享證據下全部 64 個 prompt 皆 sell，entity 效果呈現為跨公司分歧

16 家公司的 pure entity margin 全為負（BLK −1.053 到 IT −2.568），0/64 buy。
frozen 證據模板的 2 句負面（毛利率 −300 bps、guidance −6%）在 net 上壓過
2 句正面，因此所有公司都被推向 sell；entity 效果疊加在這個共同基線上，
表現為 0.57 nats IQR 的跨公司分歧。per-ticker pure entity margin（依排序）：

| ticker | sector | pure margin | 4-variant spread | framing Δ（revT − revF） |
|---|---|---|---|---|
| IT | Information Technology | −2.568 | 1.426 | +0.260 |
| BDX | Health Care | −2.335 | 1.356 | +0.349 |
| SYK | Health Care | −2.241 | 0.798 | +0.311 |
| HPE | Information Technology | −2.220 | 1.352 | +0.340 |
| GLW | Information Technology | −2.040 | 1.449 | +0.300 |
| HON | Industrials | −1.869 | 1.216 | +0.381 |
| C | Financials | −1.765 | 1.131 | +0.199 |
| DHR | Health Care | −1.752 | 0.679 | +0.409 |
| ABT | Health Care | −1.673 | 1.288 | +0.475 |
| CSX | Industrials | −1.641 | 0.999 | +0.353 |
| DE | Industrials | −1.554 | 0.861 | +0.305 |
| AMAT | Information Technology | −1.526 | 0.831 | +0.399 |
| GS | Financials | −1.482 | 1.233 | +0.337 |
| AXP | Financials | −1.344 | 1.304 | +0.372 |
| NSC | Industrials | −1.307 | 1.290 | +0.466 |
| BLK | Financials | −1.053 | 1.091 | +0.469 |

sector 平均：Financials −1.411、Industrials −1.593、Health Care −2.000、
Information Technology −2.089。

reverse-option framing 方向與 Phase 1 一致（`"sell" or "buy"` 順序把 margin
往 buy 推，per-ticker Δ +0.199 到 +0.475，median 0.365 nats）。

### Spearman 未過的解讀：兩個構念不同

Phase 1 的 named margin 由每家公司自己的 10-K 證據驅動（公司間證據內容
不同，IQR 1.375 nats）；2A 的 pure entity margin 在證據固定為同一組
公司中立句子的條件下測得，量的是「同一證據下、只換公司名」的 entity
prior。兩者的 ranking 不但沒有一致，而是中程度反向（ρ = −0.411）：
Phase 1 中證據面偏正面的公司（ABT +0.938、SYK +0.812、HPE +0.250）在
2A 排在中後段，Phase 1 中最偏 sell 的 DE（−2.000）在 2A 只排中段
（−1.554）。

描述性（未 pre-register）：2A pure entity margin 與 Phase 1 的
named-vs-anonymous gap（paired，+0.432 nats 的 entity 效果量）的 Spearman
ρ = +0.448。這個方向與 gate 期待的「entity 效果跨 evidence regime 穩定」
一致，但強度未達 0.5，且屬事後觀察，不構成 gate 證據。

## H4（descriptive，無 gate）

investment-dial coordinate L15/n8490 的 down-projection 通道值在 entity
position 的跨公司範圍僅 −0.032 到 +0.074（近零且平），與 pure entity
margin 的 Spearman ρ：entity position +0.159、final position +0.106。
沒有證據顯示 dial activation 隨 entity 而變，與 investment-dial 線的
結論（L15/n8490 是 model-level stance prior，非 entity-specific）一致；
H4 的 entity-specific trap 未被觸發。

## 限制

- 16 家公司、4 個 variant 的公司；IQR 只比門檻高 0.07 nats，分組統計力
  有限。
- Spearman 項以 Phase 1 named median 為參照，該參照混入了公司特定證據
  效果；gate 失敗可能反映構念不匹配，而不只是「entity 效果不穩定」。
  協議未定義替代參照，因此不在此處改判。
- 0/64 buy 意味著 2B 的 toward-source 統計將完全在 sell 半邊操作；
  若未來重啟 2B，top/bottom 對比仍需依賴 pure margin 的相對排序。
- 本 run 不產生 causal 證據；2B/2C 未執行。

## 狀態與後續

- 2A：completed，gate 2A（Rev 1）fail（1/4 項未過）。該判定維持，不回填。
- Gate 2A Rev 2（[proposal-phase2-rev2.md](proposal-phase2-rev2.md)，
  2026-09-10）：對同一 2A forward records 的 CPU-only 重評
  （run `phase2a-rev2-gate-01`），5 項全過：IQR 0.570；Spearman vs
  Phase 1 gap +0.448（> 0.3）；group construct check
  （top NSC/BLK vs bottom IT/BDX，4/4 pairwise 正向）；framing 0.365；
  schema 1.0。**2B 獲授權**。
- 2C：依 2B handoff 區間決定。

## Rev 2 重評記錄

| 判準 | 值 | 門檻 | 結果 |
|---|---|---|---|
| IQR（pure entity margin） | 0.570 nats | > 0.5 | 通過 |
| Spearman vs Phase 1 gap | +0.448 | > 0.3 | 通過 |
| group construct check（top 2 vs bottom 2 的 Phase 1 gap，4 pairwise） | 4/4 正向（NSC>IT +0.391, NSC>BDX +0.422, BLK>IT +0.563, BLK>BDX +0.594） | 全數正向 | 通過 |
| framing stability | 0.365 nats | < 1.5 | 通過 |
| schema valid rate | 1.000 | ≥ 1.0 | 通過 |

描述性：Spearman vs Phase 1 named margin = −0.411（Rev 1 gate 項，
降為 descriptive，見 Rev 2 協議 §1 的構念診斷）。

run：`artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-rev2-gate-01`（prepare ＋ analyze，無 GPU；provenance 含 source records SHA-256）。

## 再現性

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \
    --model .cache/models/qwen3.5-4b \
    --run-id phase2a-gpu-bf16-01
```

smoke（4 tickers × 1 variant，不觸發 gate）：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/balanced_evidence_gap_phase2.py \
    --model .cache/models/qwen3.5-4b --smoke
```

artifacts：`prepare/prompts.jsonl`（64 行）、`forward/results.jsonl`
（margin、decision、dial 通道值）、`analyze/summary.json`（pure entity
margins、gate 2A、H4）、`manifest.json`（6 artifacts，SHA-256 全數驗證）。
所有 artifacts 為 compact 派生值，不含 raw activation/residual。

Rev 2 gate 重評（CPU-only，無 GPU）：

```bash
uv run --no-sync python scripts/balanced_evidence_gap_phase2_rev2.py \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --run-id phase2a-rev2-gate-01
```
