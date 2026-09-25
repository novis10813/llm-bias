# Claim-to-Evidence Ledger

**用途：** 對照 `paper-draft.md` 的論文主張、現有實驗規格與可追溯證據，決定哪些內容可以寫成結果、哪些只能保留為探索觀察，以及下一輪需要補什麼實驗。

**審核日期：** 2026-09-23
**目前範圍：** Qwen3.5-4B、32 layers、hidden size 2560、`entity-to-dial-heldout-transfer-v1-01` 的 200-company cohort。
**文件狀態：** working evidence audit；不是正式結果報告，也不會把 `note.md` 的探索紀錄自動升格成 confirmation result。

## 判定規則

- **Artifact-supported exploratory：** 有 compact artifact、run provenance 與可重現輸入，但研究仍屬 development 或 pilot；可以作為目前方向的證據，不能寫成跨模型或正式泛化結論。
- **Record-only exploratory：** 主要數字只在 `note.md` 或命令輸出紀錄中，沒有對應的完整 compact result artifact；可作為待重跑線索。
- **Incomplete：** 有相關資料，但證據的 estimand、prompt、split 或 control 與論文 claim 不一致。
- **Blocked：** 現有設計無法支持該 claim，必須先改變 split、prompt 或 measurement protocol，再另立／凍結版本。
- **Supported：** 只有在 claim 的 primary outcome、control、provenance 與必要 decision-level gate 都完成後才使用。現階段沒有把任何 cone claim 標成此狀態。

固定答案 token 的 continuation margin 只能表示固定讀出改變。只要 claim 使用「決策翻轉」「改變輸出類別」或其他 generated behavior，必須同時報告真實 greedy generation 的 decision-flip 結果與 parse rate。

## 現有證據入口

| 證據入口 | 已確認內容 | 論文用途 | 目前狀態 |
|---|---|---|---|
| [`entity-to-dial-heldout-transfer-v1-01`](../../artifacts/qwen3.5-4b/entity-to-dial-heldout-transfer/runs/entity-to-dial-heldout-transfer-v1-01/manifest.json) | 200 companies、800 prompts、8,274 forward records、200 selection records；提供 DIM 與 cone 的 construction ranking | construction source、company split provenance、clean margin | Artifact-supported exploratory；這是 direction source，不是 steering confirmation |
| [`phase1-v2-stance-char16-01`](../../artifacts/qwen3.5-4b/entity-concept-layer-scan/runs/phase1-v2-stance-char16-01/analyze/summary.json) | candidate layers 8/12/15/19/20/23/26；L15 stance-direction vs margin `R²=0.605`；run 標為 `development` / `not_evaluated` | L15 是目前的 representation-localization clue | Artifact-supported exploratory；不是完整 span×layer residual-patching map |
| [`03_dim_balanced3_frozen.json`](../../artifacts/qwen3.5-4b/concept-cone-steering/runs/20260922-p0/03_dim_balanced3_frozen.json) | frozen balanced prompt；MO、CNC、FOXA；DIM alpha 0–5；margin、greedy JSON、random control、anonymous prompt | DIM 的第一個 decision-level pilot | Artifact-supported exploratory；有結果，但缺 run metadata、multi-seed 與 full evaluation coverage |
| [`smoke-vdim/provenance.json`](../../artifacts/qwen3.5-4b/concept-cone-steering/directions/smoke-vdim/provenance.json) | token-wise DIM shape `[100, 2560]`、L15、Top/Bottom 10、upstream run、git commit | direction construction provenance | Artifact-supported; smoke direction，不是完整 evaluation |
| [`smoke-cone4d/provenance.json`](../../artifacts/qwen3.5-4b/concept-cone-steering/directions/smoke-cone4d/provenance.json) | sector-demeaned contrastive SVD、Top/Bottom 20、4D cone、L15 upstream；保存 stance cosine | 4D cone construction provenance | Artifact-supported; smoke direction，不是完整 cone behavior result |
| [`dim_smoke.json`](../../artifacts/qwen3.5-4b/concept-cone-steering/runs/smoke-20260922/dim_smoke.json) | alpha 只有 0；沒有 steering dose-response | schema／execution smoke | 不能支持效果 claim |
| [`cone_smoke.json`](../../artifacts/qwen3.5-4b/concept-cone-steering/runs/smoke-20260922/cone_smoke.json) | 4D cone；alpha 只有 0；保存 stance cosine 與 clean output | schema／execution smoke | 不能支持 cone steering claim |
| [`note.md`](note.md) | DIM 情境、cone 曲線、5-company 結果、macro probe 的文字紀錄與命令 | 研究線索與待重跑清單 | Record-only exploratory；數字不能取代 compact result artifact |

## Claim matrix

### C1：Inference-time investment stance is controllable

**論文 claim：** representation steering 可以在不改 prompt 或 model weights 的情況下改變 investment stance。

**目前證據：** `03_dim_balanced3_frozen.json` 在 frozen balanced prompt 上，DIM 對 MO、CNC、FOXA 的 margin 隨 alpha 上升，並分別在 alpha 4、5、4 產生 greedy `buy`。這同時包含 margin 與 generated decision，因此比只報 fixed-token margin 更完整。

**判定：** **Artifact-supported exploratory。**

**可寫範圍：**「在 Qwen3.5-4B 的 frozen balanced pilot 中，DIM 能將三個極端 Sell target 的 margin 往 Buy 方向推動，且在有限 alpha sweep 中觀察到 greedy flips。」不要寫成一般 LLM 性質，也不要以三家公司代表穩健泛化。

**需要補的證據：** 多個 construction-disjoint targets、至少多個 random seeds、完整 run metadata、固定的 prompt renderer 與正式 decision-flip summary。

### C2：A useful steering site can be localized by layer and span

**2026-09-24 版本核對：** C2 的 **427 家同公司條件翻轉**結果與缺項見 [C2 v2 狀態](c2-v2-427/status.md)；下方的 16 家／8 direction 峰值表及 L15 論述是[早期跨公司版本](c2-phase2b-16/status.md)的歷史紀錄，**不可當作 427 家結果**。兩版 direction、prompt 與 2A gate 不同。兩版 summary 都已有 `evidence` span；`final` 卻不是 answer-prefix，因此舊段落「未涵蓋 evidence」不再準確。舊段落第 5 模型 `Qwen3.8-27B` 與本地 registry `Qwen3.6-27B` 命名不一致，遠端狀態待查；以下原文留存，不把它視為已完成。

**論文 claim：** entity-span transfer 在 layers 0--5 最大，instruction-span transfer 在 L15 達峰，因此 L15 是 practical steering site。

**目前證據：** `phase1-v2-stance-char16-01` 顯示 L15 的 stance direction 對 16 家公司 final margin 有 `R²=0.605`，而候選層 profile 在 L15 最高。`docs/proposal/progress-after-investment-dial.md` 另記錄過 L15 instruction transfer peak，以及較早 entity handoff 的描述。

**跨模型 `span × layer` residual patching 峰值**（balanced-evidence-gap Phase 2B sweep，16 家公司、8 個 transfer direction，frozen shared-evidence template；4/5 模型完成）

| 模型 (n_layers) | entity peak 層 (rel depth, T) | instruction peak 層 (rel depth, T) | 2B run-id |
|---|---|---|---|
| Qwen3.5-4B (32) | L0 (0.00, 1.011) | L15 (0.48, 0.464) | `phase2b-gpu-bf16-01` |
| Gemma-4-12B (48) | L9 (0.19, 1.418) | L25 (0.53, 0.401) | `phase2b-crossmodel-01` |
| GLM-4-9B (40) | L13 (0.33, 0.978) | L19 (0.49, 0.782) | `phase2b-crossmodel-01` |
| GPT-OSS-20B\* (24) | L4 (0.17, 0.849) | L12 (0.52, 0.819) | `phase2b-crossmodel-01` |

- T 定義：$T=(M_{\text{patched}}-M_{\text{tgt}})/(M_{\text{src}}-M_{\text{tgt}})$；4B / Gemma 淺層 entity T>1 表示 patch 效果超過 source 自身 baseline margin 差（over-transfer）。
- 4 個模型的 instruction peak 相對深度一致落在 0.48–0.53（±0.02），但絕對層不同（L15 / L25 / L19 / L12）；entity peak 在淺層（rel depth 0–0.33）。
- \*GPT-OSS-20B 為 MoE（使用者要求的例外納入）；gpt-oss-20b 與 gemma4-12b-it 的 2A gate 失敗後以記錄在案的 `--gate-override` 繼續（開發階段描述性比較）。
- 第 5 個模型 Qwen3.8-27B（FP8 checkpoint）在 idlab 上執行中，完成後回填本表。

**判定：** **Incomplete。** `span × layer` residual patching map 現已涵蓋 entity 與 instruction spans 的完整 layer range（4 個模型，見上表）；4B 的 instruction peak 在 L15、entity peak 在最早層，與 claim 方向一致。仍未涵蓋 evidence／answer-prefix spans，被選層沒有 greedy decision check，layer selection rule 未事先固定，「0--5」邊界也未直接證實。

**建議論文 wording：** 暫時改成「在既有 Qwen3.5-4B development characterization 中，L15 是 instruction-span stance readout 的候選高點；本文把 layer selection 視為 model-specific procedure。」等完整 map 後再恢復更精確的 layer claim。

**必要實驗：**

1. 凍結與 steering evaluation 相同的 prompt template、company split、margin scorer。
2. 對完整 layer range 與固定 token spans 執行 residual patching，至少包含 entity、evidence、instruction、answer-prefix／final spans。
3. 以 margin displacement 作 primary localization metric；對被選 layer 另做 greedy decision check。
4. 事先固定 layer selection rule，不能看完 steering flip 後再挑 L15。

### C3：DIM and multi-dimensional cone provide comparable operator families

**論文 claim：** single-neuron scalar intervention、DIM 與 multi-dimensional cone 的 dose-response 可以被直接比較。

**目前證據：** DIM direction 有 persisted `[100,2560]` artifact 與 frozen balanced result。4D cone 有 persisted direction provenance，但現有 cone JSON 只有 alpha 0；完整 4D 曲線目前只在 `note.md`。

**判定：** **Incomplete。** 目前只能說三種 operator 已被定義，不能說三者已完成公平 comparison。

**必要實驗：**

- 同一 model、prompt、target set、intervention span、layer、alpha convention。
- 對每個 operator 報告每-token intervention norm，或使用事先固定的 matched-norm dose；不能只用 raw alpha 比較不同 operator。
- 保留 single-neuron dial 的 neuron identity、sign、layer 與 dose definition。
- 每個 operator 同時輸出 margin、greedy decision、parse rate、decision flips 與 random control。
- 把 DIM 與 cone 的 construction companies 和 evaluation companies 分開。

### C4：Fixed-token margin movement can diverge from generated decision changes

**論文 claim：** margin 變正不保證 greedy JSON decision 立即翻轉。

**目前證據：** `03_dim_balanced3_frozen.json` 已直接支持：CNC 在 alpha 4 的 margin 是 `+1.035`，但 greedy decision 仍為 `sell`；MO 與 FOXA 在正 margin 後於 alpha 4 翻成 `buy`。

**判定：** **Artifact-supported exploratory。**

**可寫範圍：** 這是目前最清楚、最適合成為主結果之一的 distinction。結果應報告 exact margin、decision、alpha，而不是只報「margin threshold」。

**必要補強：** 擴大 targets 與 alpha sweep，預先定義 decision-flip rate；若 prompt 可能生成 invalid JSON，需把 parse failure 分開列出，不能從分析中刪除。

### C5：The learned direction is more effective than a matched-norm random direction

**論文 claim：** learned DIM/cone operator 的 movement 不是任意 residual perturbation 都會產生。

**目前證據：** `03_dim_balanced3_frozen.json` 有 MO、single random seed、alpha 0/2/4/6；DIM movement 明顯大於 random pilot，且 random 沒有 flip。

**判定：** **Artifact-supported exploratory，single-seed。**

**不可直接宣稱：** 「高度方向特異性」或「非隨機擾動所致」作為普遍結論。單一 random seed 只能是 pilot control。

**必要實驗：** 至少固定多個 random seeds，對每個 seed 報告 matched-norm effect、decision flips、confidence interval 或 bootstrap summary；control 應與 learned operator 使用相同 prompt 與 alpha／norm grid。

### C6：The operator is a global stance control rather than an entity identifier

**論文 claim：** DIM 對 identity-stripped／anonymous prompt 也有效，因此它不是只編碼某家公司。

**目前證據：** `03_dim_balanced3_frozen.json` 的 anonymous prompt 在 alpha 4 從 margin `-1.788` 變成 `+1.387`，greedy decision 從 `sell` 變成 `buy`。

**判定：** **Artifact-supported exploratory，single anonymous prompt。**

**可寫範圍：** 「該 pilot 與 global stance control 相容」；不能寫成已證明 operator 完全不含 entity information，也不能說已完成 entity debiasing。

**必要實驗：** 多個 identity-stripped prompts、不同 evidence conditions、不同 targets 的 paired anonymous controls；另外保留 entity-present prompt 的同一 alpha／norm grid。

### C7：Steering remains sensitive to financial evidence

**論文 claim：** strong adverse evidence 可以延後或阻止 decision flip，margin movement 和 evidence-grounded generated decision 可能分離。

**目前證據：** `note.md` 記錄 pure-positive、pure-negative、zero-evidence 與 rationale observations；但該紀錄明確指出 balanced pilot 使用 frozen renderer，而 scenario 數字來自後來改過首行的 custom renderer。`probe_dim_steering.py` 目前的 `custom` renderer 與 frozen template 不同，且 `--prompt-style frozen` 只覆蓋 balanced scenario。

**判定：** **Blocked for paper-level evidence until rerun.**

**必要實驗：**

1. 凍結單一 prompt renderer；positive、negative、mixed、zero-evidence 全部使用同一 protocol version。
2. 對同一 company、同一 operator、同一 alpha／norm grid，先記錄 clean margin／decision，再記錄 steered margin／decision。
3. 至少使用多家公司與兩個 evidence polarity；report margin delta、decision-flip rate、parse rate。
4. generated `reason` 只能作 qualitative example；不能把它當成 evidence faithfulness measurement。

### C8：A multi-dimensional cone is smoother or less saturating than a 1D operator

**論文 claim：** cone 比單一方向提供更平滑、較不易 saturation 的 control surface。

**目前證據：** `smoke-cone4d/provenance.json` 證明 4D sector-demeaned SVD direction 已抽取；`note.md` 記錄 1D、4D、8D 曲線與反折觀察，但沒有對應的完整 evaluation JSON。note 同時混有舊 8D basis 與目前 4D construction，不能直接合併成一個結果。

**判定：** **Record-only exploratory。**

**必要實驗：**

- 同一 frozen prompt、同一 target、同一 matched-norm budget。
- 1D DIM、single-neuron、4D cone，以及 dimension ablation（例如 first-1/2/3/4 axes）同時掃描。
- 事先定義 monotonicity、saturation／reversal 與 greedy-flip metrics；不能只挑一條漂亮曲線。
- 將每條曲線保存到 compact result JSON，並保存 cone dimension、construction set、target split、alpha／norm convention。

### C9：Different cone rays correspond to distinct financial concepts

**論文 claim：** cone axes 或 rays 分別調控不同財務考量面向。

**目前證據：** `note.md` 以 generated rationale 的語意差異作為主要依據；沒有獨立 concept labels、human audit、evidence attribution 或 quantitative ray-specific metric。

**判定：** **Blocked。** 幾何正交、不同 margin 曲線或不同生成理由本身都不能證明 axes 是不同 financial concepts。

**論文處理：** 目前應從 main contribution 和 Introduction 移除。若保留，最多寫成 qualitative observation，並明確標成不具 semantic identification 的 exploratory analysis。

### C10：The same operator generalizes across companies and sectors

**論文 claim：** 一套 4D cone 在多個跨產業極端 Sell 公司上都能翻轉 decision。

**目前證據：** `note.md` 記錄 MO、CNC、FOXA、TSN、BAX 的 5/5 結果，但沒有對應的 persisted output JSON。更重要的是，這些 target 是否同時出現在 Top/Bottom 20 construction set 必須先逐一核對；依目前 200-company ranking 設計，至少部分 target 很可能參與了 cone construction。

**判定：** **Blocked as generalization claim.** construction overlap 會使結果成為 in-construction transfer，不能稱為 held-out generalization。

**必要實驗：** 先凍結 company-disjoint construction／evaluation split；若要宣稱 sector generalization，至少做 leave-one-sector-out 或 sector-disjoint evaluation，並保存 target exclusion provenance。報告 per-company flips，不只報 5/5 aggregate。

### C11：The method supports financial explainability

**論文 claim：** steering 後的 generated rationale 仍然反映金融證據，或提供可解釋的 recommendation。

**目前證據：** 只有 generated reason 摘要與 evidence scenario observations，沒有 rationale faithfulness、evidence attribution、expert evaluation 或 counterfactual explanation test。

**判定：** **Blocked as a primary claim。**

**目前可保留的版本：** 把 financial evidence sensitivity 當作 decision-level boundary condition；把 generated reason 當 qualitative output，不稱為 explanation 或 faithful rationale。

### C12：The protocol transfers across decoder LLMs

**論文 claim：** layer-localized steering procedure 可適用於多個 decoder LLMs。

**目前證據：** 所有現有 cone／DIM artifacts 都指向 Qwen3.5-4B；沒有第二個 model 的 run。

**判定：** **Blocked as an empirical claim。** 可以在方法章說 protocol 以 model-specific layer selection 設計，不能在結果章寫 cross-model evidence。

**必要實驗：** 每個 model 重新 fit direction、重新選 layer、重新報 model identity、hidden size、layer index、prompt tokenizer、construction／evaluation split。不能直接移植 L15 或 hidden-space direction。

## 最小 confirmation package

如果論文維持目前三項 contribution，最小可交付的 confirmation package 是：

| Package | 目的 | 必須保存的結果 |
|---|---|---|
| P0 protocol freeze | 消除 frozen／custom renderer 混用 | prompt hash、token spans、model／commit、construction／evaluation split、alpha 或 matched-norm grid |
| P1 layer localization | 支持 model-specific steering site | 完整 span × layer map、primary margin metric、selected-layer rule、候選 layer greedy check |
| P2 operator comparison | 真正比較 single-neuron、DIM、cone | 同一 target set 的 dose-response、matched-norm random controls、多 seed、parse rate |
| P3 decision boundary | 把 fixed readout 和 behavior 分開 | 每個 condition 的 margin、greedy decision、flip status、parse status、flip rate |
| P4 evidence sensitivity | 支持金融場景的 evidence consideration | frozen template 下的 positive／negative／mixed／zero-evidence paired results |
| P5 generalization | 避免 construction leakage | company-disjoint targets；若宣稱 sector transfer，增加 sector-disjoint split |

若 P2 或 P5 尚未完成，論文仍可寫成 Qwen3.5-4B 的 inference-time steering pilot，但 contribution 需要縮小為 DIM／cone feasibility 與 decision-level boundary observation，不能寫成 multi-dimensional operator 的泛化優勢。

## 目前最需要決定的三件事

1. **Cone 的主張要升級到哪裡：** 目前 artifact 支持 cone construction，不支持完整 cone behavior comparison；要先決定是否重跑 P2，或把 cone 降為方法比較中的 exploratory arm。
2. **Evaluation split 是否 company-disjoint：** 若不另建 split，所有「跨公司泛化」措辭都應移除，並把 target 稱為 construction-cohort evaluation。
3. **金融考量的強度：** 建議保留 evidence sensitivity 作為 decision-level boundary；除非增加 faithfulness measurement，否則不把 financial explainability 寫成 contribution。
