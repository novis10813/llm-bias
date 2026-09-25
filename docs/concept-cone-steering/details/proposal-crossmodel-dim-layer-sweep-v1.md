# 三模型 paper cohort DIM 跨層掃描 V1：協議與實作規格

> **已由 [V2](proposal-crossmodel-dim-layer-sweep-v2.md) 取代（2026-09-25）。**V1 只跑過單家公司 smoke，從未執行 101 家正式評估；其 CLI（`sp500_dim_crossmodel`）與 schema 已自程式移除。以下為凍結原文，不再更新。

**狀態：**事前協議，2026-09-25；尚未執行新的三模型 DIM run。**範圍：**Gemma-4-12B、GLM-4-9B、GPT-OSS-20B 的 2024 S&P 500 母體 503 家（402 建方向／101 受測）；承接 [Qwen DIM V1](proposal-dim-layer-sweep-v1.md)，不更名或改寫[四模型 paper cone 主實驗](../crossmodel-cone-paper/proposal.md)、[Qwen V1 結果](status-dim-layer-sweep-v1.md)或其他舊產物。以下規則在新模型任何受測結果揭露前固定；若方向、選層、主要指標、controls 改變，另立版本。

## 高 C2 `T` 是否對應本模型局部 DIM 的較強效果，須同時報計分和真生成

以各模型**自己的** 402 家乾淨固定答案 margin 排序定 Top／Bottom 各 10 家，每一候選層用該層自己的乾淨殘差建立 Difference-in-Means (DIM) 正向單位方向，再在同層介入 101 家。主要臂是 post-block 共同指令尾段逐 token、只改 prefill，與 paper cone 注入幾何相同；補充臂是 pre-block 單向量、覆蓋 prefill 及 cached decoding 的所有 token，兩臂的單一 alpha **不是等效總 dose**。C2 `T` 是 427 家同公司證據條件翻轉的 normalized transfer，**不是**本協議的 `mean(ΔM)`，也不是模型生成決策翻轉。分組、位置和金融提示是[Arditi et al. (2024)](https://arxiv.org/abs/2406.11717)單向量設計的 task-local 改編，與[Wollschläger et al. (2025)](https://arxiv.org/abs/2502.17420)的梯度 cone 均非原文復現。

### 事前層集合與來源

用三份 [`phase2b-v2-427-01` C2 instruction summary](../c2-v2-427/status.md) 的有限 `mean_normalized_transfer`：先取所有 `T ≥ 0.70 × peak T` 層；**若此集合少於 5 層，才加上 peak±2**（界外層忽略）；最後加 L0／最後一層，升序去重，並核對精確集合如下。Gemma 的高 `T` 集合已有 7 層，因此**不加 L25**：這項明確集合在已核准的設計骨架中固定；骨架寫的無條件「加 peak±2」與其枚舉集合互相矛盾，此處在實作前以本段有條件規則解消，**不更動獲核准的枚舉層**。後續不得依本實驗 101 家結果增刪層。

| 模型 slug／層數 | C2 peak `T` | 高 `T` 層 | 低 `T` 比較層 | 全部候選層（主要臂） | C2 summary SHA-256 |
|---|---:|---|---|---|---|
| `gemma4-12b-it`／48 | L27，0.1903 | L26–L32 | L0、L47 | **0,26,27,28,29,30,31,32,47** | `295170b9c09ca453ce7b285c37a0f128d99cf68fc1a20970f390f084fdd54635` |
| `glm4-9b-0414`／40 | L19，0.4592 | L17–L21 | L0、L39 | **0,17,18,19,20,21,39** | `d0b7e41ac21adc1cb858fbdc6a75e34338109cab84f3630bcd6f839119b480a6` |
| `gpt-oss-20b`／24 | L14，0.5323 | L12–L16 | L0、L23 | **0,12,13,14,15,16,23** | `f8ccb7c24d25048d358a32562ba8fffd799384a482b1e17152d7f0bef543a506` |

每層 C2 `n_directions=854`，不可把低 `T` L0／末層假設為零效應或 matched random controls。尤其末層 post-block 注入在最後 transformer block 之後，對僅改 prefill 的生成路徑沒有後續 attention 混合，結構性的零／弱效應不能解讀成否定 DIM。選層所用 427 家 C2 公司與本次 101 家受測公司**重疊 89 家（三模型皆同）**：受測組僅對本次 DIM 排名和方向擬合留出，**不是獨立公司選層驗證**；[Gemma 與 GLM 的 Phase 2A gate 曾 override](../c2-v2-427/status.md)，跨模型亦只作 development 描述。

## REQ-1：來源、族群與模型身份 fail closed

【已驗證：`scripts/probe_concept_cone.py:split_population,prepare_instruction_suffix`；三份下表只讀原始 JSON】2024 年 S&P 500 的 503 家取自 `data/sp500_constituents_2020_2025.csv`，seed `20260923` 固定 402／101；prompt 為 `_render_frozen_prompt(...,order=0,reverse=False)` + `format_prompt(...,use_chat_template=True,enable_thinking=False)`。對**各模型自己的**完整 ranking 僅取最末 10 Top、最前 10 Bottom；不重新使用 Qwen ranking 或以 101 家分數挑方向。來源須 SHA 精確匹配：

| slug／model dtype | 唯一容許的只讀 ranking source（`artifacts/<slug>/concept-cone-steering/runs/` 下） | SHA-256 | schema／選層／共同尾段 |
|---|---|---|---|
| Gemma／`bf16` | `c2-guided-paper-20260924/result.json` | `13c2b4a6f1538a7ab84183e86c7e8ba4fe63decff5a898e5d3251a73eb09c7e7` | `concept-cone-sp500-paper-v1`／L27／100 tokens |
| GLM／`bf16` | `crossmodel-v1-eval-20260924/result.json` | `025a1b4a76f0a564b8b593a19a6ed2805278e2a23d5799a9eaa50ce574a926be` | **舊** `concept-cone-sp500-v1`／L19／98 tokens |
| GPT-OSS／**`native`** | `c2-guided-paper-20260924/result.json` | `6987e4197e683965018ac5f92347daafe841bc80b7ff41e00fa376d7db7ae393` | `concept-cone-sp500-paper-v1`／L14／99 tokens |

【新設計】檢查 source `mode=evaluation,complete=true`、402 個 `(margin,ticker)` 穩定升序且 finite/無重複、`top_20`／`bottom_20` 與排名一致、402／101 ticker 名單與重建 split 逐項吻合且不交疊；`effective_tokens` 等於表列共同尾段，並重新檢查模型當下 tokenizer 對全部 503 家的實際共同尾段。來源 model 絕對路徑／slug／model config SHA／tokenizer config SHA／checkpoint `name+bytes`／tokenizer identity／population SHA／split SHA／seed／prompt family SHA／renderer／固定答案前綴 `{"decision": "`／注入層／`k_pairs=20,k_cone=4`／alphas `[0,2,3,4,5,6]`／dtype／`max_new_tokens=192` 皆與本地建構值一致。Gemma、GPT 原始來源含 `c2_layer_source`，核對 digest／峰值／854 條；**GLM 舊 schema 缺此欄**；已核對其原始 JSON 的 `alphas=[0,2,3,4,5,6]` 和 `max_new_tokens=192` 與本協議相同，不為舊 schema 放寬劑量或生成長度。以本協議固定 source SHA 和獨立驗證的新 C2 summary SHA 綁定新 DIM provenance，**絕不修改或假稱舊 source 已綁 C2**。GPT-OSS 必須 `--model-dtype native` 以保持 packed MXFP4 權重；其他兩模型 bf16。少欄／hash 改變／層數錯／方向和受測 ticker 相交，立即停止。

## REQ-2：兩臂分層擬合與注入，不保存原始向量

【已驗證：`scripts/probe_concept_cone.py:extract_dim_layer_directions,fit_dim_direction,run_cone_evaluation`；`llm_bias/core/inference/interventions.py` 的 post-／pre-block context hooks】主要 `tokenwise` 臂：對 20 家乾淨 prompt，僅取該模型實際共同 instruction 尾段 `[K,d_model]` 的每層 post-block states，計 `v_l[p]=mean(Top_l[p])−mean(Bottom_l[p])`，單 token 各自 FP32 L2 單位化。固定 `K=100/98/99`（Gemma／GLM／GPT），不是以公司的 eval 長度重算，不能截成任意 100；評估時同層在該公司 instruction 尾段 post-block 只加 `alpha*u_l[p]`，只改 prompt prefill，不改之後 decode token。`alpha=0` 不註冊 hook。

補充 `single_all` 臂：在該層 **block 輸入**取 Top／Bottom 的共同 instruction 尾段**最後一個 token**，計單一 `[d_model]` 差 `w_l`，FP32 單位化；在同一層 block 輸入對**所有** prompt prefill 和 cached decode token 加 `alpha*w_l`；固定答案計分與無前綴生成用同一 hook。這是對 Arditi 設計的金融改編，不是 primary post-block 的劑量對照。

20 家的每個原始 state、mean 和方向 tensor 都只在記憶體；差、輸入及所有 norm 必須 finite、norm `>1e−8`、歸一化誤差 `≤1e−5`，否則**該 arm／層 fail closed**，不更換 extraction token／層或抄其他層方向。保存 `top_10/bottom_10`、`K`、每層 min/median/max norm 和單位向量 SHA，不保存 raw hidden states。若預選 L0 補充臂 norm 退化，獨立 compact `l0_fit_diagnostic.json` 記模型／arm／選層／threshold／實測 norm／C2/source SHA；其完整 sweep 明寫為排除 L0 的預選層子集，不能以 0% 或結果造出 L0 rows。主臂若某層退化，沒有該層結果；不得宣稱完整九／七層。

## REQ-3：解析、固定答案位移與依 α0 生成類別條件化的翻轉分開

【已驗證：`scripts/probe_concept_cone.py:dim_layer_summary`】每層、每臂、每一 101 公司在 α `[0,2,3,4,5,6]` 以固定 `{"decision": "` 後綴取得 buy−sell token log-prob margin（nats），在無後綴原 prompt greedy 生成最多 192 token。只在 `json.loads(整段生成)` 為完整 object 且 `decision∈{buy,sell}` 時計入**嚴格解析**；保留全部 generated text 以供追溯。各 alpha 分報嚴格 parsed `n/101`、`median M`、`mean(ΔM_α)=(1/101) Σ_i[M_i(α)−M_i(0)]`；方向轉移以同**層**、同公司 α0 真生成解析後買／賣為條件，再限定該 alpha 同樣可解析，報 `buy→sell`、`sell→buy` 分子／有效分母與全部有效 paired flips；基準類別或有效分母為 0 時 `null`／文字 `—`，**不是 0%**。固定答案 margin 正負與 C2 `T` 一律不能充作生成翻轉。每個 arm 的各層 α0 必須一致或記錄不一致並停止解釋跨層翻轉。

已知 cone 來源生成 α0：Gemma 嚴格 0/101（往往 fenced 完整物件）、GLM 嚴格 101/101 **buy**、GPT-OSS 嚴格 0/101（截斷 `analysis`）；這是**既有 cone**，新 DIM 必須實際測，不預填任何決策。Gemma **事前規定的補充**格式解析只接受[既有完整物件去殼規則](../../../scripts/reparse_concept_cone_decisions.py)限定的 `bare_json`、`fenced_json`、`thought_bare_json`、`thought_fenced_json`，去殼後完整 JSON 僅 `decision`、`reason` 兩 key、reason 非空；逐 ticker／layer／alpha 另存 derived JSON、來源 SHA／格式計數與**依補充 α0 生成分組**的雙方向有效配對，不改寫 strict `result.json` 或以補充結果替換嚴格主結果。GPT-OSS 的截斷片段絕不以 regex 或 margin 猜出決策；若其新生成碰巧有完整物件，只能按事先允許的同一規則獨立報告。GLM 若本版 α0 全 buy，`sell→buy` 分母未定義，不可解讀成方向無效。

## REQ-4：新 CLI／產物版本與續跑須維持舊行為

【新設計】在現有 CLI 旁新增 `--cohort-mode sp500_dim_crossmodel`，只容許上述三模型及其 model dtype、`--dim-arm tokenwise|single_all`、`--dim-layers` 精確固定升序集合或其子集、`--alphas 0 2 3 4 5 6`、`--split-seed 20260923` 和 balanced evidence；只容許 `--smoke-tickers` 選 101 家內的 ticker，metadata `mode=smoke`，不得把 smoke 當成 101 家正式評估。舊 Qwen 的 `sp500_dim_paper`、legacy／cone CLI、既有 JSON schema、預設和舊 artifacts 不變。新 schema 分別 `dim-tokenwise-crossmodel-v1`、`dim-single-all-crossmodel-v1`；`--output-json` 限制在 `artifacts/<slug>/concept-cone-steering/runs/dim-crossmodel-layer-sweep-v1-<id>/<arm>/result.json`，smoke 用不同 `<id>`，不得覆寫任何完整 run。

結果結構 `targets[ticker]["L{layer}"]["rows"]`，每層每家公司六 alpha 的有限 margin、generated text、strict decision/parse；metadata 包含上述所有 source/C2/config/SHA、model dtype、K、protocol 標籤、方向公司、層集合與實際 target tickers，方向診斷逐層 compact。已完成結果唯讀回報；未完成續跑要 source SHA／方向 SHA／全部 metadata 完全相符、現有每格六 alpha 嚴格驗證且未重複，完成旗標只在本 arm 全部選定層 × 101 × 6 有效 margin／生成記錄後設 true；原子寫入中間結果。不可默默覆蓋既有不完整或跨模式資料，不能保存 raw activations/residuals/gradients/Jacobians/KV。各模型正式 main/supplement arm 使用不同輸出路徑；Gemma 補充解析也只寫相鄰新 derived 檔並拒絕覆蓋已有檔。

## 實作切片、可寫邊界與驗收

| Slice | 可寫檔案 | 獨立驗收 |
|---|---|---|
| 1：選層、來源／rank 驗證及動態尾段（REQ-1、2 的純函式） | `scripts/probe_concept_cone.py`、`tests/test_probe_concept_cone.py` | 三模型 C2 curves 的選層／SHA／n854、來源 allowlist 與 GLM 舊 schema、排序／402+101／dtype、K=98/99/100、退化 norm 和原 Qwen K100 測試通過。`uv run pytest -q tests/test_probe_concept_cone.py`。 |
| 2：新 mode／真模型入口／兩臂續跑（REQ-2–4） | `scripts/probe_concept_cone.py`、`tests/test_probe_concept_cone.py`；若確需改 generic hook，才改 `llm_bias/core/inference/interventions.py`、`tests/test_core_inference.py` | fake-model 三種 n_layers／dtype／K、tokenwise post-block 只 prefill、single_all pre-block 包括 cached decode、α0 no-op、`ticker→layer→rows` 無覆寫、strict parse null 分母、污染續跑拒絕；Qwen V1 測試仍綠。`uv run pytest -q tests/test_probe_concept_cone.py tests/test_core_inference.py && uv run python -m compileall -q scripts llm_bias`。 |
| 3：完整物件衍生解析（REQ-3） | `scripts/reparse_concept_cone_decisions.py`、`tests/test_reparse_concept_cone_decisions.py`；只加 DIM schema 分支且保留 cone 格式，或新增 `scripts/reparse_dim_decisions.py` 與其測試（二擇一） | fake DIM 兩層含不同 α0 基數、合法 wrapper、截斷／多物件／無效 reason；驗證 strict 不被取代、分層 paired 分母、來源 SHA、無重覆覆蓋。`uv run pytest -q tests/test_reparse_concept_cone_decisions.py tests/test_probe_concept_cone.py`（新測試檔則一併加入）。 |
| 4：模型 smoke／全量／結果記錄（REQ-1–4） | 新 run 位於忽略追蹤的 `artifacts/<slug>/concept-cone-steering/runs/dim-crossmodel-layer-sweep-v1-*/`；**完成執行後才新增** `docs/concept-cone-steering/details/status-crossmodel-dim-layer-sweep-v1.md` 並更新 `docs/README.md` 直接相關一列 | 各模型先 1 家×峰值＋末層×兩臂真 smoke、補充 L0 擬合診斷；fit／記憶體／hook 出錯立即停該層或該模型，不用受測數據換層。主臂（Gemma 九層、GLM/GPT 七層）先全量，補充臂只在實際可擬合預定層掃 101 家。核對各 model×arm 的完整矩陣、原始 SHA、α0 重現性／解析率／雙方向分母，逐 alpha 報 mean(ΔM)、嚴格和 Gemma 補充翻轉；未完成不可寫成「已完成」。`uv run pytest -q tests/test_probe_concept_cone.py tests/test_core_inference.py tests/test_reparse_concept_cone_decisions.py && uv lock --check && git diff --check`；完整測試／build 於可行時執行。 |

**執行順序與停止：**先本協議唯讀 preflight；完成切片 1/2/3 的無 GPU 測試，才以 smoke 驗證各模型 checkpoint／hook；真模型長跑用背景命令，可在互不搶 GPU 的設備串行。若所有正式 strict 結果無 α0 解析，仍可報固定答案位移與解析限制，但**不能宣稱沒有翻轉**；所有判斷均以實際產物為準。不要觸碰 `docs/research-scripts.md`、`scripts/probe_operator_comparison.py` 的不相關 dirty changes；不得另建隱式 `scripts` 交叉 import 或編造缺少的歷史設定檔。
