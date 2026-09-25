# 舊 C2 選層的 Concept Cone 診斷：僅 GLM 同層觀測可引用

**範圍與進度（2026-09-24）：** 使用 2024 年 S&P 500 共 503 家公司，固定切分為 402 家建構與 101 家受測。每個模型只用自己的建構組排序、提取 4D cone，於 [C2 早期 16 家跨公司版本](../c2-phase2b-16/status.md)的指令峰值層注入（L15／L25／L19／L12）；**並非** [C2 v2 的 427 家條件翻轉層](../c2-v2-427/status.md)（L16／L27／L19／L14）。這是**舊 16 家 C2 選層的探索性診斷**，不是 503 家全部受測，Qwen／Gemma 的舊層結果不納入 [427 家 C2 選層的論文主表](../crossmodel-cone-paper/status.md)。執行條件見 [V1 協議](proposal.md)。

| 模型 | 注入層 | 進度與查證入口 | 尚缺什麼 |
|---|---:|---|---|
| Qwen3.5-4B | L15 | 已完成 402 家排序、101 家 × 6 alpha；[逐公司 alpha 表](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/crossmodel-v1-eval-20260924/result.md)、[compact JSON／provenance](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/crossmodel-v1-eval-20260924/result.json) | 對照、獨立切分與多種子；本次只可描述觀察 |
| Gemma-4-12B | L25 | 已完成 402 家排序、101 家 × 6 alpha；[逐公司 alpha 表](../../../artifacts/gemma4-12b-it/concept-cone-steering/runs/crossmodel-v1-eval-20260924/result.md)、[compact JSON](../../../artifacts/gemma4-12b-it/concept-cone-steering/runs/crossmodel-v1-eval-20260924/result.json) | **0/101 解析為有效 JSON**（多數生成以 Markdown code fence 包裹）；尚無合格 decision-flip 分母，不可用 margin 代替 |
| GLM-4-9B | L19 | 已完成 402 家排序、101 家 × 6 alpha；[逐公司 alpha 表](../../../artifacts/glm4-9b-0414/concept-cone-steering/runs/crossmodel-v1-eval-20260924/result.md)／[compact JSON](../../../artifacts/glm4-9b-0414/concept-cone-steering/runs/crossmodel-v1-eval-20260924/result.json) | 101/101 可解析；各 alpha 翻轉均 0/101。L19 同時是 [C2 427 家選層](../c2-v2-427/status.md)，實際條件核對後可引用於[論文主實驗](../crossmodel-cone-paper/status.md) |
| GPT-OSS-20B（MXFP4） | L12 | 並行載入時 VRAM 接近滿載，已主動停止；沒有舊層結果 | 不重啟舊 L12；另於論文主實驗 L14 執行 |

**Qwen 已完成資料的決策翻轉數：** 以下以每家公司自己的 alpha 0 貪婪生成決策為基準；六個 alpha 各有 101/101 個可解析 JSON。完整 margin 與各公司判定請看上方逐公司表，單純固定答案 margin 位移不計作翻轉。

| 指標 | alpha=0 | alpha=2 | alpha=3 | alpha=4 | alpha=5 | alpha=6 |
|---|---:|---:|---:|---:|---:|---:|
| 翻轉數／可比較公司數 | 0/101 | 0/101 | 1/101 | 4/101 | 5/101 | 8/101 |

**解讀邊界：** Qwen／Gemma／GPT-OSS 的 427 家 C2 選層須另跑；本診斷不得事後改標成新層結果；GLM 因實際 L19 相同且完成核對，按原始 run provenance 引用。Gemma 雖產生文字，仍未遵守「只輸出 JSON」的格式；上述翻轉表只屬 Qwen，Gemma 不能寫成零翻轉。五家舊案例的數字只保存在 [舊探索紀錄](../note.md)，使用的是 200 家建構排序與舊計分方式；不能與這次 503 家母體、固定前綴計分的結果合併。本次預先固定的 101 家受測組與建構組不重疊；四模型相同原始 alpha 不代表相同模型內相對效應。隨機方向對照、多種子、matched-norm 比較與 sector-disjoint 切分未做，完成四模型也不等於滿足 [C3／C10 的必要實驗](../claim-to-evidence.md)。
