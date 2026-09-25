# C2 早期跨公司版本：16 家的指令層峰值不是 427 家版本的結果

**狀態（2026-09-24）：** 這是 balanced-evidence-gap Phase 2B **早期跨公司**版本：四個模型各用 16 家公司、8 個 transfer direction，掃描完整逐層的 `entity`、`evidence`、`instruction`、`final` span。最新同公司 427 家條件翻轉另見 [C2 v2](../c2-v2-427/status.md)，兩版方向、prompt 與 gate 語義不同，不能混算峰值。這是固定答案 margin 的定位線索，不是 steering 或生成決策因果效果的確認。原始 C2 主張與既有判定見 [claim-to-evidence ledger](../claim-to-evidence.md)。

| 模型／層數 | entity peak：層、相對深度、T | instruction peak：層、相對深度、T | Phase 2B 結果 |
|---|---|---|---|
| Qwen3.5-4B／32 | L0、0.00、1.011 | L15、0.48、0.464 | [summary](../../../artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01/analyze/summary.json) |
| Gemma-4-12B／48 | L9、0.19、1.418 | L25、0.53、0.401 | [summary](../../../artifacts/gemma4-12b-it/balanced-evidence-gap-phase2/runs/phase2b-crossmodel-01/analyze/summary.json) |
| GLM-4-9B／40 | L13、0.33、0.978 | L19、0.49、0.782 | [summary](../../../artifacts/glm4-9b-0414/balanced-evidence-gap-phase2/runs/phase2b-crossmodel-01/analyze/summary.json) |
| GPT-OSS-20B／24（MoE） | L4、0.17、0.849 | L12、0.52、0.819 | [summary](../../../artifacts/gpt-oss-20b/balanced-evidence-gap-phase2/runs/phase2b-crossmodel-01/analyze/summary.json) |

相對深度為 `L / (n_layers − 1)`；`T=(M_patched−M_tgt)/(M_src−M_tgt)`。四個 instruction peak 都在 0.48–0.53 相對深度，entity peak 較淺；`T>1` 代表超過 source／target 基線差，不能單獨推出決策改變。Gemma 與 GPT-OSS 的 Phase 2B 使用有記錄的 2A gate override，屬描述性比較；GLM 此版本的 2A gate 通過（勿與另一次 `v2-427` 混用）。[Gemma override](../../../artifacts/gemma4-12b-it/balanced-evidence-gap-phase2/runs/phase2b-crossmodel-01/pairs/directions.json)、[GPT-OSS override](../../../artifacts/gpt-oss-20b/balanced-evidence-gap-phase2/runs/phase2b-crossmodel-01/pairs/directions.json)。

## 要支持完整 C2 主張，還缺三件事

1. **確認第 5 模型的身分與結果。** Ledger 記為 `Qwen3.8-27B`、在 idlab 執行中；本地 `config/pretrained_lenses.json`（已移除，見 git tag `pre-cleanup`） 指的是 `Qwen/Qwen3.6-27B`，且未找到對應的 Phase 2B 完成 artifact。兩者不可直接視為同一模型；遠端執行狀態待查。
2. **補 answer-prefix 與生成檢驗。** 現有四種 span 中的 `final` 是未附加 `{"decision": "` 的 formatted prompt 最後位置，**不是** answer-prefix。選定層仍欠同一 protocol 的 greedy decision check；固定答案 margin 不能充當 decision-flip 證據。
3. **預先固定層選擇與解讀範圍。** 目前峰值是 development 掃描的事後描述，未在 steering evaluation 前凍結選層規則；原主張的 entity「L0–5 最大」邊界亦未直接完成檢驗。若要宣稱跨模型可轉移，需各模型重新建方向並分開報告模型內結果。

本版圖：[跨模型逐層圖 PNG](figures/crossmodel_layer_sweep.png)／[PDF](figures/crossmodel_layer_sweep.pdf)（`scripts/plot_crossmodel_phase2b.py`，輸入上表四模型的早期 2B runs）；[Qwen 指令圖 PNG](figures/4b_instruction_swap.png)／[PDF](figures/4b_instruction_swap.pdf)（`scripts/plot_4b_instruction_swap.py`，輸入 Qwen 早期 2B run）。

另見與本次 C2 不同的 [跨模型 Cone 舊層診斷](../crossmodel-cone-pilot-16/status.md)：它把上表的指令峰值層當作候選注入點，**不能反過來當作 C2 選層規則的獨立驗證**。
