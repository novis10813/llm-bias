# Claim-to-Evidence Ledger

**用途：** 對照 `paper-draft.md` 的主張與可追溯證據；分開記錄「實驗已執行」、「結果符合預先登記判準」和「論文可寫範圍」。

**本次核對基礎：** 2026-09-29 已同步的本地 artifacts；四模型 `confirmation-v1-20260925-full-01`，另參照歷史 pilot、[C2 v2-427](c2-v2-427/status.md) 與 [crossmodel cone paper](crossmodel-cone-paper/status.md)。**本檔是證據審查，不是新的 GPU run，也不取代各版本凍結協議。**

## 先看這裡：一頁結論

四個模型（Qwen、Gemma、GLM、GPT-OSS）的完整實驗都跑完了。跑完不等於每個主張都成立，各主張目前的狀態如下（細節見後面的 Claim matrix）：

| 狀態 | 主張 |
|---|---|
| 成立（限指定條件） | C1 推論時可用 DIM 方向改變 buy/sell 判定；C3 neuron、DIM、cone 三種操作已在同一設定下比較；C4 固定 margin 與真實生成的決策會不一致；C6 匿名身分也能被 steering；C12 方法在四個模型上都跑得動 |
| 部分成立 | C2 層與位置的定位（各模型結果不同）；C5 DIM 勝過隨機方向（只在部分模型與劑量）；C7 對財務證據有反應；C10 跨公司泛化（GPT-OSS 未達門檻） |
| 測試後不成立 | C8 cone 比單一方向更平滑、更不易飽和 |
| 沒做 | C9 cone 各方向對應不同金融概念；C11 rationale 是否忠實 |

## 名詞對照

- **DIM**：用公司好惡的差值算出的單一方向，注入模型內部狀態後觀察判定是否改變。**cone**：由多個方向組成的版本（cone2 兩個、cone4 四個）。
- **α（劑量）**：注入方向的強度，正負代表兩個相反方向。**α0**：不注入的基線。
- **flip（翻轉）**：注入後 buy／sell 判定和 α0 相反。**margin**：模型對 buy 與 sell 的偏好差（log p(buy) − log p(sell)），只代表傾向，不等於真實生成的判定。
- **parse／unparsed／collapse／blocked**：輸出能否被解析成決策。unparsed 是解析不出；collapse 是輸出先崩壞、來不及翻；blocked 是可以解析但沒有翻。
- **ITT**：分母固定為 α0 時屬於該類的全部公司，注入後解析不出的算「沒翻」。
- **complete-object／strict**：兩種解析方式。complete-object 允許決策外面包著 code fence、思考文字等外殼；strict 只接受整段就是乾淨 JSON。
- **patching、peak／band**：把某一層某個位置的狀態換掉，看決策受多大影響。peak 是影響最大的層，band 是影響明顯的一段層。
- **arm**：實驗中的一組條件。**CAL**：校準用的 arm。
- **LOSO**：留一產業測試，訓練時排除某個產業的公司，再測該產業。**company-disjoint**：測試公司不出現在建構方向的公司中。
- **R7、R9、R10**：預先登記的檢查編號，內容分別是生成中 patching（R7）、company-disjoint／LOSO（R9）、sector-disjoint 建構（R10）。

## 判定規則與可比較範圍

- **已執行**只表示協議列出的每組實驗都有完整的輸出檔，不表示主張成立。**符合判準**還要確認各主張的分母、解析與崩壞情況、對照組和預先登記的門檻。**不支持**（測了但沒過）和**沒有可檢驗分母**（根本沒有可比較的樣本）是兩回事，不能混用。
- `confirmation-v1` 的主要決策解析用 complete-object；只接受乾淨 JSON 的 strict 解析（`json.loads`）是次要的對照。所有 flip 都以同一家公司、同一條件下 α0 的貪婪生成為基準（ITT 分母，見名詞對照）。固定前綴 margin 不是生成出來的決策；Gemma 與 GPT-OSS 的固定前綴 margin 還偏離模型實際生成的路徑（off-path）。
- 2026-09-22 的 200 家 pilot、2026-09-24 的 402/101 cone run、2026-09-25/26 的 confirmation run，在方向的建構方式、prompt、層、劑量或解析方式上都不同，**數字不可合併**。confirmation 的 503 家公司依 seed `20260923` 固定分成 402 家建構方向、101 家評估；每個模型各自排序、各自擬合方向。
- 結論只適用於所測的模型、prompt 類型、資料切分與劑量範圍。`reason`（模型給的理由文字）不能當成「理由忠實」的證據，也不是金融概念標籤。

## 證據與完成度入口

| 版本／入口 | 可核對內容 | 用途與限制 |
|---|---|---|
| [confirmation-v1 狀態](confirmation-v1/status.md)／`artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-20260925-full-01/` | 四模型各適用 arm 的 `<arm>/result.json`、`cal/calibration.json`、`invocations.jsonl`；分別為 Qwen 17/17、Gemma 16/16、GLM 15/15、GPT-OSS 15/15 arm（數量含 CAL） | 本次主證據；每個模型的 `job_summary.json` 可能只包含最後一次 invocation，應以各 arm 結果為準；非 Qwen 不跑 `dim_layers`，GLM／GPT-OSS 不跑 `shuffle` 是預設範圍 |
| [C2 v3](c2-v3-steering-prompt/status.md)／同一 confirmation run 的 `c2v3`、`c2v3_gen` | steering prompt 的全層、六 span patching；峰值 ±2 的 prefill patch-under-generation | margin 定位與生成檢查分開判讀；部分模型的生成檢查沒有相反 α0 決策分母 |
| [operator comparison v2](operator-comparison-v2/status.md)／`dim`、`ops`、`random`、`jitter`、適用模型的 `shuffle` | matched-norm／projection dose、neuron、cone2／cone4、五個 random seed 等 | C3、C5、C8 的正式比較；與舊 4D sector-demeaned SVD smoke **不是同一種 cone construction** |
| [evidence sensitivity v1](evidence-sensitivity-v1/status.md)／`evidence`、`anon` | pos／neg／mixed2／zero paired 結果、10 匿名身分 | C6、C7；blocked（仍可解析但不翻）與 collapsed（不可解析）分開 |
| [generalization v1](generalization-v1/status.md)／`loso`、`loso_construction`、`split_seed` | DIM 的 company-disjoint evaluation、LOSO folds、兩個新 split seeds | C10 的 DIM 證據；不能自動延伸到 cone 的 sector-disjoint 效果 |
| [crossmodel cone paper](crossmodel-cone-paper/status.md) | 四模型在 C2 v2 選層的 402/101 cone 評估 | 先前的 cone 行為資料；原始 strict JSON 下 Gemma／GPT-OSS 皆無合格 flip 分母，不可與 confirmation 的 complete-object 解析率混報 |
| [2026-09-22 pilot](../../artifacts/qwen3.5-4b/concept-cone-steering/runs/20260922-p0/03_dim_balanced3_frozen.json)、[上游 ranking](../../artifacts/qwen3.5-4b/entity-to-dial-heldout-transfer/runs/entity-to-dial-heldout-transfer-v1-01/manifest.json)、[舊筆記](note.md) | 舊 200 家來源與三家公司 DIM pilot；筆記另記未完整持久化的曲線 | 歷史探索；不能取代 confirmation compact artifact 或拿舊目標當 held-out |

## Claim matrix：已跑與可寫的不是同一件事

| Claim | 本次結果與判定 | 可以寫／不可寫 |
|---|---|---|
| **C1：推論時 stance 可控** | **已跑；指定條件支持 DIM 的 generated flips。**Qwen balanced α0 為 101 sell，α=+4 有 101/101 sell→buy（101/101 可解析）；GLM α0 為 101 buy，α=−4 有 101/101 buy→sell（101/101 可解析）。Gemma／GPT-OSS 亦有雙方向非零分母與 flips；高劑量須另報崩壞。 | 寫四模型各自擬合 DIM 的 prompt-／dose-specific 結果；Qwen balanced 沒有 buy→sell 的 α0 分母，GLM 沒有 sell→buy 分母，不能寫對稱控制已在每模型證實。 |
| **C2：層與 span 可定位 steering site** | **v2 與 v3 patching、v3 generation 都已跑；原本統一的 L15／entity L0–5 敘述不成立。**v3 teacher-forced steer-suffix peak／band：Qwen L15／L14–17（含注入 L16）；Gemma L27／L26–29（含 L27）；GLM L20／L17–21（含 L19）；GPT-OSS L8／`[1,8]`（**不含注入 L14**）。v3 的 entity peak 亦非四模型都在 L0–5。Gemma 在 L27 steer-suffix 的 patch-under-generation 有 13/40 toward-source flips；GPT-OSS 在 L8 有 7/28；Qwen／GLM 的 R7 相反 α0 配對均為 0，不能寫「0% 翻轉」。 | 可描述模型特定 patching 與可檢驗的生成結果；**不得**宣稱四模型 v3 band 均驗證 v2 所選注入層，或以零分母當否定結果。舊 16 家、427 家與 v3 必須分版報告。27B 模型身分與本地完成 artifact 仍未確認；**因模型過大，現階段暫緩運行、不列待辦**，也不算完成。 |
| **C3：neuron、DIM、cone 可比較** | **已跑，同一模型內有預定 target／renderer／span／layer、劑量表與生成結果。**`ops` 保存 neuron、cone2、cone4、等投影 cone4、DIM⊥random4；另有 DIM 與 random 控制。 | 可報 preregistered 設定下的 operator dose-response 比較，應同列 norm、有效投影、flip 與 parse；不是「相同 raw α 等效」，也不是舊 single-neuron activation dial 的重跑。 |
| **C4：固定 margin 與生成決策可分離** | **已跑，現有生成列直接支持。**舊 pilot 的 CNC α=4 固定 margin +1.035、生成仍 sell；新 run 同時保存 fixed-prefix／realized-path margin、greedy decision、parse、finish。 | 以具體列及路徑說明，不能用正 margin 計算 flips；Gemma／GPT-OSS 的 fixed-prefix 更須註記 off-path。 |
| **C5：DIM 勝過 matched-norm random** | **對照已跑，但非四模型全點通過。**五個 random seeds、三個 jitter seeds；Qwen 有分母的 +4／+32 random bootstrap 下界分別 1.0／0.9505（>0），GLM 有分母的負向點通過，GPT-OSS 四個登記點 random 比較下界 >0；Gemma −0.25 下界 −0.0258、−64 下界 0，未達預定 `>0`。 | 只按有分母的符號與劑量報 DIM 對照；完整 C5 判準還要逐點核對 jitter 與 DIM off-target。**不能**寫四模型全部通過或延伸為 cone 對 random 的勝利。 |
| **C6：匿名身分仍可被 steering** | **已跑 10 個匿名身分 × 四 condition。**例如 Qwen balanced α0 為 10 sell，α=+4 有 10/10 翻轉。 | 「與非特定公司身分的 stance steering 相容」；不宣稱方向不含 entity information。若要比較 named 與 anonymous 的不確定性，另報 identity-level bootstrap（n=10）。 |
| **C7：對財務證據敏感** | **paired conditions 與 flip-dose contrasts 已跑。**Qwen neg vs balanced 的可比 sell→buy 公司 101/101 需要更高 dose；GLM pos vs balanced 的可比 buy→sell 公司 101/101 亦較高。Gemma 的相應可比組主要為 blocked；GPT-OSS 的 adverse 組同時有 blocked 與 collapsed（例如 neg vs balanced：45 blocked、12 collapsed，n=57）。 | 可按模型、方向報 dose delay／可解析範圍內的 blocked；**不能把 collapsed 當成 evidence 阻擋翻轉**，也不能把 `reason` 當 evidence faithfulness。 |
| **C8：cone 更平滑／不易飽和** | **曲線與預先登記的 C8 summary 已跑；四模型皆未同時滿足兩個符號的判準。**判準要求 cone4 在兩符號的 monotone-step fraction 嚴格勝 DIM⊥random4、不低於 DIM，且 reversals 不多於兩者。 | 這是**測試後不支持**，不是尚未實驗；不得再用舊筆記聲稱 cone 已勝 1D。可如實報 dose-response／collapse。 |
| **C9：cone rays 對應不同金融概念** | **未做獨立語意識別。**幾何軸與 rationale 不等於金融概念標籤。 | 不作 main claim；須另有標籤／審核／ray-specific 量測才能升級。 |
| **C10：跨公司／跨產業泛化** | **company-disjoint 402/101、LOSO（R9／R10）和兩個 split-seed 複製已跑（DIM）。**R9 的 `±α_50` 有分母方向 CI 上界：Qwen 0、Gemma 0／0.0909、GLM 0；GPT-OSS +4 為 0.0769，但 **−4 為 0.1875 > 0.10**。R10 所要求 sector 在四模型各有至少一個 on-target flip。 | 可以寫已測 split 的 **DIM company-disjoint** 結果；預定 sector-disjoint 判準在 Qwen／Gemma／GLM 的有分母方向符合，在 GPT-OSS 不符合。不要將 DIM 的 LOSO 判準轉嫁到 cone；也不要以舊 5-company construction-overlap 筆記作 held-out 證據。 |
| **C11：金融可解釋性／faithful rationale** | **未做 faithfulness、證據歸因或專家／反事實解釋驗證。** | `reason` 只當質性示例；不作 primary explainability claim。 |
| **C12：跨 decoder LLM 可執行** | **已跑四模型各自擬合的 DIM／cone protocol**，不能再說「沒有第二個模型的 run」。模型的選層、基線類別、解析外殼、崩壞與效果並不相同；C2 的 GPT-OSS v3 band 與注入層也不一致。 | 可以寫方法在這四個模型上已執行、附模型別 parse／flip；不能寫相同 hidden direction、L15 或 raw α 可直接跨模型移植，也不能聲稱四模型各 claim 均成立。 |

## 最小 confirmation package：執行與成立分開記

| Package | 已執行的證據 | 尚須遵守的結論界線 |
|---|---|---|
| P0 protocol freeze | [confirmation-v1 freeze／狀態](confirmation-v1/status.md)：模型與 renderer、split、K、dose、SHA 與 CAL | 舊 pilot 和新方向不可混為同一 protocol |
| P1 layer localization | [C2 v3](c2-v3-steering-prompt/status.md) 全層六 span、生成檢查 | GPT-OSS v3 band 不含 L14；Qwen／GLM R7 翻轉分母為零 |
| P2 operator comparison | [operator comparison v2](operator-comparison-v2/status.md)：neuron／DIM／cone、random／jitter／部分 shuffle | C8 不成立；C5 只在符合門檻的模型／劑量可寫 |
| P3 decision boundary | `dim`、`ops` 等逐列 margin、greedy decision、parse／collapse | 報 ITT 與 complete-object parse；strict JSON 不得混用 |
| P4 evidence sensitivity | [evidence sensitivity v1](evidence-sensitivity-v1/status.md) 的 pos／neg／mixed2／zero 與匿名配對 | 無分母／blocked／collapsed 分開；不是 rationale faithfulness |
| P5 generalization | [generalization v1](generalization-v1/status.md) 的 402/101、R9–R11 | sector-disjoint 的判準限 DIM；GPT-OSS 不達預定 R9 門檻 |

**接下來的文件工作：**以各 claim 的原始分母、解析外殼與劑量製作論文結果表；更新 `paper-draft.md` 時刪除 C8 優勢、C9／C11 強主張，勿把 C2／C10 的條件式結果寫成四模型普遍成立。27B 本輪不運行。舊 pilot、v2 及 crossmodel cone 的版本狀態保留原文，不回填 confirmation 數字。
