# C2 427 家選層的 Concept Cone 論文主實驗：四模型已執行，行為資料仍不完整

**範圍（2026-09-24）：** [427 家 C2 結果](../c2-v2-427/status.md)選出 Qwen L16、Gemma L27、GLM L19、GPT-OSS L14；此處另以 2024 年 S&P 500 的 503 家母體固定切成 **402 家建方向／101 家受測**。不同數字代表不同實驗階段，不可把 427 家當作本次 cone 的建構或受測家數。各模型使用自己的建構組 ranking；詳細方法見[本主實驗協議](proposal.md)。

| 模型 | C2 427 選層 | 目前結果 | 下一步 |
|---|---:|---|---|
| Qwen3.5-4B | L16 | **已完成**：[逐公司 alpha 表](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/c2-guided-paper-20260924/result.md)／[原始 JSON](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/c2-guided-paper-20260924/result.json)；402 家建方向／101 家受測 × 6 alpha，101/101 可解析；alpha 5 翻轉 2/101、alpha 6 翻轉 6/101 | C2 427 summary SHA 與 L16 綁定在 JSON metadata；[舊 L15 診斷](../crossmodel-cone-pilot-16/status.md)不納入主表 |
| Gemma-4-12B | L27 | **已完成**：[原始逐公司 alpha 表](../../../artifacts/gemma4-12b-it/concept-cone-steering/runs/c2-guided-paper-20260924/result.md)／[原始 JSON](../../../artifacts/gemma4-12b-it/concept-cone-steering/runs/c2-guided-paper-20260924/result.json)；402／101 家 × 6 alpha，嚴格 JSON 各 alpha **0/101** | [事後完整物件補充解析逐公司表](../../../artifacts/gemma4-12b-it/concept-cone-steering/runs/c2-guided-paper-20260924/decision_reparse.md)／[衍生 JSON](../../../artifacts/gemma4-12b-it/concept-cone-steering/runs/c2-guided-paper-20260924/decision_reparse.json)；另列補充翻轉數，不能覆蓋原始嚴格結果 |
| GLM-4-9B | L19 | **已核對同層可沿用觀測**：[逐公司 alpha 表](../../../artifacts/glm4-9b-0414/concept-cone-steering/runs/crossmodel-v1-eval-20260924/result.md)／[原始 JSON](../../../artifacts/glm4-9b-0414/concept-cone-steering/runs/crossmodel-v1-eval-20260924/result.json)；402／101 切分、prompt、score、6 個 alpha 均一致，101/101 可解析，alpha 0–6 翻轉皆 0/101 | 源 run 仍記舊診斷 schema、未綁 C2 427 SHA；本頁依 [C2 427 GLM L19](../c2-v2-427/status.md)核對沿用，不改寫原始 provenance |
| GPT-OSS-20B（MXFP4） | L14 | **已完成**：[逐公司 alpha 表](../../../artifacts/gpt-oss-20b/concept-cone-steering/runs/c2-guided-paper-20260924/result.md)／[原始 JSON](../../../artifacts/gpt-oss-20b/concept-cone-steering/runs/c2-guided-paper-20260924/result.json)；native 權重、402／101 家 × 6 alpha、C2 427 L14 SHA 已核對 | 606 次生成均未產生可解析的完整決策 JSON（輸出從 `analysis` 文字開始），嚴格 parse 0/101 各 alpha，**沒有 flip 分母**；[嚴格轉移統計](../../../artifacts/gpt-oss-20b/concept-cone-steering/runs/c2-guided-paper-20260924/decision_transitions_strict.md)。不能從文字片段或 margin 猜 buy/sell；若另設輸出生成方案，需另行定義並驗證 |

**四模型的 alpha 欄位（描述性固定答案讀出）：** 下表為受測 101 家各 alpha 的 buy−sell 固定答案 **margin 中位數（nats）**，不是生成 buy/sell、不是 C2 的 `T`；相同原始 alpha 不能當成跨模型等效注入強度。逐公司的 margin／原始生成狀態見上表各模型 `result.md`。

| 模型／層 | alpha=0 | alpha=2 | alpha=3 | alpha=4 | alpha=5 | alpha=6 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen L16 | −2.655 | −2.146 | −1.873 | −1.613 | −1.317 | −1.053 |
| Gemma L27 | +1.208 | +4.957 | +6.905 | +8.532 | +10.202 | +11.551 |
| GLM L19 | +4.057 | +4.573 | +4.845 | +5.116 | +5.374 | +5.604 |
| GPT-OSS L14 | −2.838 | −2.855 | −2.853 | −2.872 | −2.860 | −2.868 |

**同一 101 家的平均固定答案 margin 位移（nats）：** 每格 `mean(ΔM_alpha) = (1/101)Σ_i[M_i(alpha)−M_i(0)]`，先對**同公司**計算 alpha 與 alpha 0 的差，再平均全部受測公司。即使 GPT-OSS／Gemma 沒有嚴格可解析的生成結果，margin 仍可計算；但位移**不是決策翻轉、不是 C2 的 `T`**，不能拿它填補下方缺失的 flip 分母。數值由上表連結的各模型原始 `result.json` 逐公司 `cone_centroid` margin 計算（四捨五入到 0.001）。

| 模型／層 | alpha=0 | alpha=2 | alpha=3 | alpha=4 | alpha=5 | alpha=6 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen L16 | 0.000 | +0.519 | +0.782 | +1.052 | +1.330 | +1.607 |
| Gemma L27 | 0.000 | +3.157 | +4.755 | +6.281 | +7.696 | +9.051 |
| GLM L19 | 0.000 | +0.529 | +0.799 | +1.060 | +1.314 | +1.535 |
| GPT-OSS L14 | 0.000 | +0.008 | +0.026 | +0.028 | +0.032 | +0.040 |

**生成決策的 alpha 欄位（可解析公司數；相對 alpha 0 翻轉數／有效配對）：** 分母只取同公司 alpha 0 與該 alpha 均可解析者。「—」表示 0 個有效配對，**不是零翻轉**；Gemma 事後補充解析另列，不取代其嚴格解析。

| 模型／解析規則 | alpha=0 | alpha=2 | alpha=3 | alpha=4 | alpha=5 | alpha=6 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 嚴格 JSON | 101；0/101 | 101；0/101 | 101；0/101 | 101；0/101 | 101；2/101 | 101；6/101 |
| Gemma 嚴格 JSON | 0；— | 0；— | 0；— | 0；— | 0；— | 0；— |
| Gemma 事後補充解析 | 101；0/101 | 101；17/101 | 101；20/101 | 101；23/101 | 101；28/101 | 101；33/101 |
| GLM 嚴格 JSON | 101；0/101 | 101；0/101 | 101；0/101 | 101；0/101 | 101；0/101 | 101；0/101 |
| GPT-OSS 嚴格 JSON | 0；— | 0；— | 0；— | 0；— | 0；— | 0；— |

**Gemma L27 的事後補充解析（非原協議主要指標）：** 腳本 `scripts/reparse_concept_cone_decisions.py` 只去除[補充分析規則](proposal.md#原始結果之後的補充解析不是原協議主要指標)允許的完整外層標記，再驗證完整兩欄 JSON。606 次輸出全數符合補充解析（503 完整 code fence、86 `thought` + fence、17 `thought` + 裸物件）；嚴格 JSON 仍為 0/606，原始 JSON 不變。來源 SHA-256 `13c2b4a6f1538a7ab84183e86c7e8ba4fe63decff5a898e5d3251a73eb09c7e7`。翻轉以各公司 alpha 0 的**生成**判定為基準，補充解析分母各為 101；33/101 是 sell→buy，不能將它寫作原協議嚴格 JSON 下的翻轉率。

**基準決策與正向注入的解讀限制：** 上述 alpha 表格中的 alpha 0 數字是固定答案 `buy−sell` **margin**，不是 C2 的 normalized transfer `T`；margin > 0 表示固定答案偏 buy，卻不保證自由生成一定是 buy。Qwen L16 的 alpha 0 生成為 sell 101／buy 0（margin 全為負），alpha 6 轉 buy 6 家。GLM L19 的 alpha 0 生成為 buy 101／sell 0（margin 全為正，最小 +2.34），alpha 6 仍 buy 101 家；對**只沿 buy 方向增加的正 alpha**，本來全 buy 的基準群無法再出現 sell→buy 翻轉，0/101 不能解讀為「干預沒影響」，其 margin 中位數由 +4.06 增至 +5.60。Gemma L27（僅依上述事後補充解析）的 alpha 0 為 buy 58／sell 43，固定答案 margin > 0 為 53／101；若將「接近零」暫定為 |margin| < 1 nat，僅 5／101。alpha 6 的 33 次翻轉全為 sell→buy；原本 58 家 buy 留在 buy 不計翻轉，另有 10 家 sell 留在 sell。margin 符號與自由生成判定並非逐家公司一致，翻轉必須按生成結果計算。這是正向單邊掃描的基準分布限制；沒有負 alpha／適當對照前不能據此判定模型間效應大小。

**逐 alpha、依生成基準分組的方向統計（101 家同一受測組）：** 下表由 `scripts/summarize_concept_cone_decisions.py` 從已保存的生成決策計數，先限定同公司 alpha 0 與該 alpha 均可解析，再分方向報**該基準類別的配對分母**；「—」是該基準類別為 0，方向比率未定義，**不是 0%**。Qwen 與 GLM 用原始嚴格 JSON；Gemma 是看過輸出格式後的補充解析，與前兩者只作描述，不混成主分析。

| 模型／解析方式、方向（alpha 0 分母） | alpha=0 | alpha=2 | alpha=3 | alpha=4 | alpha=5 | alpha=6 |
|---|---:|---:|---:|---:|---:|---:|
| [Qwen 嚴格 JSON](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/c2-guided-paper-20260924/decision_transitions_strict.md)：sell→buy（101） | 0/101 | 0/101 | 0/101 | 0/101 | 2/101 | 6/101 |
| Qwen：buy→sell（0） | — | — | — | — | — | — |
| [GLM 嚴格 JSON](../../../artifacts/glm4-9b-0414/concept-cone-steering/runs/crossmodel-v1-eval-20260924/decision_transitions_strict.md)：buy→sell（101） | 0/101 | 0/101 | 0/101 | 0/101 | 0/101 | 0/101 |
| GLM：sell→buy（0） | — | — | — | — | — | — |
| [Gemma 事後補充解析](../../../artifacts/gemma4-12b-it/concept-cone-steering/runs/c2-guided-paper-20260924/decision_transitions_supplementary.md)：sell→buy（43） | 0/43 | 17/43 | 20/43 | 23/43 | 28/43 | 33/43 |
| Gemma 補充解析：buy→sell（58） | 0/58 | 0/58 | 0/58 | 0/58 | 0/58 | 0/58 |
| [Gemma 原始嚴格 JSON](../../../artifacts/gemma4-12b-it/concept-cone-steering/runs/c2-guided-paper-20260924/decision_transitions_strict.md)：任一方向有效配對（0） | — | — | — | — | — | — |
| [GPT-OSS 原始嚴格 JSON](../../../artifacts/gpt-oss-20b/concept-cone-steering/runs/c2-guided-paper-20260924/decision_transitions_strict.md)：任一方向有效配對（0） | — | — | — | — | — | — |

各模型／解析方式的 alpha 0 基準 buy／sell 分別為 Qwen **0／101**、GLM **101／0**、Gemma 補充解析 **58／43**，有效配對在前六列各為 101／101。個別 alpha 可解析家數、四種轉移（包含維持買／賣）、總翻轉及來源 JSON SHA 見各列的衍生報表。**這些比例的分組依據是生成結果，不是 margin 符號，更不是 C2 的 `T`。**

GLM 選層查核綁定 `phase2b-v2-427-01/analyze/summary.json` SHA-256：`d0b7e41ac21adc1cb858fbdc6a75e34338109cab84f3630bcd6f839119b480a6`；原始 run 的舊 schema 不因此變更。

**尚不能寫成跨模型主結論：** Qwen L16 與 GLM L19 的嚴格可解析決策可作描述；Gemma L27 嚴格 JSON 不合格，上表僅為有來源的**事後**補充分析，不能直接與原協議嚴格翻轉數併作主比較；GPT-OSS L14 的生成停在 `analysis` 文字，**沒有可解析決策**，不能報翻轉率。原協議嚴格 JSON 的 parse rate 與相對 alpha 0 的 flip rate 須獨立報告；補充解析只作事後描述，無法解析不能算「沒有翻轉」。單一 split、無多種子／隨機方向對照與跨模型等效 dose 的限制，不能由本次單獨支持 operator 優勢或跨產業泛化。
