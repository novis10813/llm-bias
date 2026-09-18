# Evidence-insensitivity：證據改變買賣傾向，但受測狀態置換未翻轉生成決策

**範圍與狀態：** Qwen3.5-4B、Gemma-4-E2B-it，bf16；S&P 500 名單 503 家、共用多空證據與 JSON 買賣回答。Phase 1–3 已完成（截至 2026-09-17），全線維持進行中，尚未宣告收線。Phase 2／3 使用 discovery 樣本，未完成 hold-out confirmation；Gemma Phase 2 未通過兩項 gate，只作帶保留的描述。

**一句話發現：** 公司是否跟隨證據極性，不能由受測中間層的一維狀態差解釋；置換最後位置的狀態能改變固定答案的買賣分數，但兩模型的生成決策都未翻轉。

## 1. 同樣的證據不保證同樣的決策，順序也會改變 Qwen 的回答

**觀察：** Phase 1 對所有公司使用同一組正負證據，以百分比控制淨極性。Qwen 的 503 家零證據回答全為 Sell，平均 margin 為 −5.40 nats；匿名提示也偏 Sell。這裡的 margin 是固定答案 Buy 相對 Sell 的 logit 差，正值表示讀出偏 Buy，不等同實際生成的決策。

| Phase 1 全人口分組 | Qwen | Gemma |
|---|---:|---:|
| Evidence-responsive：負面主導選 Sell、正面主導選 Buy | 50 | 395 |
| Fixed-sell：兩種極性均選 Sell | 453 | 0 |
| Fixed-buy：兩種極性均選 Buy | 0 | 108 |

Qwen 每家公司的正負證據 margin contrast 都為正，因此固定選 Sell 不表示模型完全忽略證據。另在 100 家的順序對調測試中，正面主導條件把正項放到末尾後，有 69–90 家由 Sell 改選 Buy。

**解讀與限制：** 行為分組取決於模型及提示條件，不能把 Qwen 的固定 Sell 當成跨模型特性。順序對調支持位置會影響回答；本階段沒有內部干預，也不足以單獨確認 Sell 吸引子機制。

## 2. 受測一維狀態未支持「固定組基線不同或證據反應較弱」

**觀察：** Phase 2 在 402 家 discovery 公司上，比較 Qwen L15、Gemma L18 的 instruction 位置狀態。沿 stance 軸量測的 offset 表示基線位置，gain 表示對證據極性的反應；兩者只描述選定方向，不代表完整狀態。

| 比較 | Qwen | Gemma（描述性） |
|---|---|---|
| Offset 組差 | 未檢出差異，p = 0.573 | 未檢出差異，p = 0.655 |
| Gain 組差 | Fixed-sell 比 responsive 高約 6%，p = 0.009；方向與衰減假說相反 | 未檢出差異，p = 0.180 |
| 狀態反應形狀 | 對證據極性近似線性 | 加入證據的位移約為翻轉極性的 10 倍 |

**解讀與限制：** 本次未支持上述一維 offset／gain 解釋，不代表整個 capture-layer 狀態都不含分組資訊。Qwen 的 gain 差未做多重比較校正，不能寫成「全 null」或獨立機制證據。Gemma 的決定論容差與 stance 軸有效性 gate 均 fail；其 capture 位置亦因 BOS 座標差早一個 token，精確終點重測尚未執行。Phase 3 通過 gate 不會追溯消除這些限制。

## 3. 最後位置的狀態可轉移 margin，組差方向卻隨模型相反

**觀察：** Phase 3 各取 42 家 responsive 與 42 家固定決策公司，把同公司的正面主導狀態換入負面主導 run，並反向置換。每次只換一個 token 的 post-block 狀態，掃描 10 層及 entity、evidence、instruction、prompt_end 四個位置。

| 結果 | Qwen | Gemma |
|---|---|---|
| Prompt_end 的 margin 效應 | L15 至 L19 增加，晚期層持續放大 | L10 至 L15 間出現，L28 附近趨於飽和 |
| 最終層正向置換的組平均 ΔM | Responsive 1.412、fixed-sell 1.205 nats | Responsive 9.33、fixed-buy 10.06 nats |
| Entity 控制 | 精確 0 | 精確 0 |
| Instruction 錨點 | L15 接近 0 | L18 精確 0 |

兩模型的 Phase 3 gates 全過，正向置換在最後層對各 84 家的 margin 都朝 Buy 移動。依 frozen S-vs-R 規則，原始報告均判為 **readout-mediated**：兩組都能轉移正向分數，但讀出響應不同；Qwen 是 responsive 較大，Gemma 則是 fixed-buy 較大。

**解讀與限制：** 這項裁決限於受測 margin 端點，不能推成「不跟隨證據的公司普遍讀出較弱」。按 margin 效果選座標會偏向讀出附近的晚期位置；加上層格間隔，本次沒有辨識精確寫入層，也未排除多位置共同作用。

## 4. 生成決策零翻轉，margin 變動不能代替行為改變

**觀察：** 每模型在四個選定座標、兩個方向共做 672 次 patched generation，決策翻轉均為零。Gemma 尤其顯示端點差異：fixed-buy 組在負面主導條件的 canonical-prefix margin 中位數為 −6.02，實際生成卻全為 Buy。

**解讀與限制：** 零翻轉只表示這次單位置置換未改變生成決策，不表示干預零效應。Gemma 生成含 Markdown fence 的 JSON，與固定 prefix 的 margin 量測條件不同，兩者不能互推；兩模型的 margin 幅度也不直接互比。本線尚未確認能改變生成決策、同時保留證據反應的干預方法。

## 查證入口

| 要查什麼 | 原始紀錄與來源 |
|---|---|
| 行為分組、順序對調 | [Phase 1 報告](details/report-phase1.md) · [協議](details/proposal-phase1.md)；雙模型 run `phase1-gpu-bf16-01`。Gemma 全人口分組見該模型 run 的 `analyze/summary.json`，discovery 分組見 Phase 2 報告。 |
| 一維狀態對比、Gemma gate 與 BOS 限制 | [Phase 2 報告](details/report-phase2.md)（含 §8 erratum） · [協議](details/proposal-phase2.md)；雙模型 run `phase2-gpu-bf16-01`。 |
| 單位置置換、雙端點與原始裁決 | [Phase 3 報告](details/report-phase3.md) · [協議](details/proposal-phase3.md)；雙模型 run `phase3-gpu-bf16-01`。 |
| 跨模型初步診斷 | [Cross-model probe](details/diagnostic-cross-model-probe.md)。初步 probe 不取代全人口結果。 |

正式 run 目錄為 `artifacts/<model-slug>/evidence-insensitivity/runs/<run-id>/`；model slug 分別為 `qwen3.5-4b`、`gemma4-e2b-it`。

**來源差異待查：** Phase 3 原報告將 Gemma 早層 evidence 效應寫為 ≤0.16，但 §5.1 表列 L10 組中位數 0.231／0.194；sector 數亦有 11／12 標籤的不同寫法。本摘要不沿用這些上界或「所有 span 全域零效應」的概括，保留原始紀錄待查核。

**編輯說明：** 合併時依研究問題納入 Phase 2／3，移除重複導覽的頂層 `proposal.md`，各階段協議由上表直達；未改原始階段報告、frozen 判準或 run provenance。
