# C2 427 家條件翻轉：四模型的指令區間峰值在中層，尚非決策層級驗證

**已核實（2026-09-24）：** [Phase 2 v2 協議](../../balanced-evidence-gap/details/proposal-phase2-v2.md)使用 `data/baseline/investment-dial/exploratory-v1.json` 的 **427 家**公司；每家以正／負兩句 shared evidence 組成兩個條件，正→負、負→正各一條 direction，合計 854 條。四個模型的 `phase2b-v2-427-01` 均完成全層 × `entity`／`evidence`／`instruction`／`final` 掃描；這是固定答案 margin 的 patching 指標，**不是生成買賣決策翻轉**。

| 模型／層數 | instruction peak（相對深度，T） | entity span 的最大 T | 427 家 Phase 2B 原始結果 |
|---|---|---|---|
| Qwen3.5-4B／32 | L16（0.52，0.408） | L3，−0.007 | [summary](../../../artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json) |
| Gemma-4-12B／48 | L27（0.57，0.190） | L19，−0.007 | [summary](../../../artifacts/gemma4-12b-it/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json) |
| GLM-4-9B／40 | L19（0.49，0.459） | L0–39 並列，約 −0.021 | [summary](../../../artifacts/glm4-9b-0414/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json) |
| GPT-OSS-20B／24（MoE） | L14（0.61，0.532） | L5，−0.012 | [summary](../../../artifacts/gpt-oss-20b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json) |

相對深度 `L/(n_layers−1)`，`T=(M_patched−M_tgt)/(M_src−M_tgt)`。**不可把 entity 負值說成跨公司效果消失：** 427 家 v2 的 source／target 是**同一公司**、只改證據條件，entity 文字相同；[舊 16 家跨公司版本](../c2-phase2b-16/status.md)的 entity peak 測的是另一個問題。舊版指令峰值依序為 L15／L25／L19／L12；本版 L16／L27／L19／L14 也不是同一 estimand 的重複測量。

**gate 與尚缺證據：** Qwen、GPT-OSS 的 v2 Phase 2A gate 通過；[Gemma](../../../artifacts/gemma4-12b-it/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/pairs/directions.json)與 [GLM](../../../artifacts/glm4-9b-0414/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/pairs/directions.json)以留存理由的 override 續跑，因此跨模型只能作 development 描述。第 5 個 Qwen 27B 的 v2 Phase 2B 完成 artifact 尚未找到；舊 lens registry（`config/pretrained_lenses.json`，已移除，見 git tag `pre-cleanup`）是 `Qwen3.6-27B`，與舊 ledger 所記 `Qwen3.8-27B` 身分不一致，遠端狀態待查。`final` 只指未附 decision prefix 的格式化提示詞尾 token，並非 answer-prefix；仍缺 answer-prefix span、選層規則預先固定、被選層的 greedy decision check。此版本的同公司條件差無法驗證原本「跨公司 entity 在 L0–5 最大」的敘述。

**圖片與可重建入口：** [Qwen 427 家雙向 instruction 圖 PNG](figures/v2_427_instruction_swap.png)／[PDF](figures/v2_427_instruction_swap.pdf)（`scripts/plot_v2_427_instruction_swap.py`，Qwen `phase2b-v2-427-01`）；[跨模型 instruction 圖 PNG](figures/v2_427_crossmodel.png)／[PDF](figures/v2_427_crossmodel.pdf)（`scripts/plot_v2_427_crossmodel.py`，上表四模型 v2 2B）；[舊 16 家與新 427 家並排圖 PNG](figures/crossmodel_span_direction.png)／[PDF](figures/crossmodel_span_direction.pdf)（`scripts/plot_crossmodel_span_direction.py`，上表四個 v2 runs 加[舊版四模型 runs](../c2-phase2b-16/status.md)）。圖只作不同設計的描述性對照，不把兩版曲線合併為一個 effect。
