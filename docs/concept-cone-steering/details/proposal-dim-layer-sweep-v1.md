# DIM paper cohort 跨層掃描 V1：協議與實作規格

**狀態：**事前協議；尚未執行。**範圍：**Qwen3.5-4B、2024 S&P 500 的 402 家建構／101 家受測公司。本文同時是實作規格；未經紀錄不得在查看 101 家結果後更改選層、方向、劑量或指標。

## 高 C2 `T` 是否也對應較大的 DIM 影響，須分開檢查計分與生成

在同一層建立與注入該層專屬方向，避免把 L16 的向量直接搬到其他層。C2 的 427 家同公司條件交換，在 instruction span 測得平均 normalized transfer `T`；它不是本實驗的固定答案 margin 位移，也不是生成判定翻轉。根據[原始 C2 summary](../../../artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json)，事先固定 **L0、L14、L15、L16、L17、L18、L31**：高 `T`（≥0.25）的 L14–L18，低 `T` 的 L0／L31。原值分別為 L0 −0.0023、L14 0.2542、L15 0.4004、L16 0.4076、L17 0.3193、L18 0.2866、L31 −0.0070。低 `T` 只表示不同 C2 任務的對照層，不保證 DIM 效果為零；L31 接近輸出尤其可能直接影響答案 logit。

[Arditi et al. (2024)](https://arxiv.org/abs/2406.11717)及[原作者程式](https://github.com/andyrdt/refusal_direction/blob/main/pipeline/utils/hook_utils.py)用兩類提示在單一位置算一條均值差向量，依另立 validation 選方向，並在該層全部 token 加向量。本實驗的金融 Top／Bottom 分組、由獨立 C2 事先選層及逐 token 主要比較，**都是改編，不宣稱原文復現**；[Wollschläger et al. (2025)](https://arxiv.org/abs/2502.17420) 的梯度優化 cone 也不是本 repo 的產業去均值 SVD cone。舊 `scripts/probe_dim_steering.py` 的 L15／200 家 pilot 分組與此 402／101 切分不同、答題前綴計分曾不一致、截斷答案曾被 regex 接受；舊協議、程式與 artifacts 一律不覆寫。

## REQ-1：使用原始 paper cone 排名，但方向只由 402 家建立

【已驗證：`scripts/probe_concept_cone.py:split_population`, `rank_construction`, `run_sp500_v1`】從 `data/sp500_constituents_2020_2025.csv` 選 2024 S&P 500 的 503 家，seed `20260923` 重建 402／101；frozen balanced prompt 是 `_render_frozen_prompt(ticker,name,order=0,reverse=False)` 並用 `format_prompt(...,use_chat_template=True,enable_thinking=False)`。已有完整 [paper cone `result.json`](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/c2-guided-paper-20260924/result.json) 的 402 家 `ranking` 使用 `{"decision": "` 前綴算乾淨 buy−sell margin。

【新設計】只讀取該完整 JSON 並記錄其 SHA-256；檢查 `schema=concept-cone-sp500-paper-v1`、`mode=evaluation`、`complete=true`、`effective_tokens=100`、`layer=16`、模型 slug、dtype `bf16`、人口 hash、split seed/hash、402／101 ticker、tokenizer/model config hash、prompt-family hash、答案前綴與 192-token 設定；檢查 402 個排名 ticker 唯一、完全等於建構組，margin 為有限數且按 `(margin,ticker)` 升序。建構公司的 **Bottom 10＝前 10，Top 10＝後 10**；禁止由舊 DIM 的 200 家、任何 101 家分數或其生成結果選方向。另驗證當前 [C2 summary](../../../artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json) SHA 等於來源所記的 digest、instruction peak=L16、每層 `n_directions=854`，七層 `T` 有限且選層恰為上述固定集合。模型 checkpoint 名稱／大小和執行時 tokenizer identity 也比對來源 metadata；無法證實則停止。

## REQ-2：主要臂只改同層、同 100 個指令 token 的 post-block 殘差

【已驗證：`scripts/probe_concept_cone.py:prepare_instruction_suffix`, `extract_concept_cone_basis`, `run_cone_evaluation`；`llm_bias/core/inference/interventions.py:residual_interventions`】所有 503 家相同的 instruction 尾段為 100 個 token，既有 cone 在 post-block 同位置只改 prefill；其 centroid 每 token norm 約 1。`record_residuals(model,ids,layers)` 回傳 transient post-block `[batch,sequence,d_model]`，不得將它寫入 artifacts。

【新設計】每一層 `l` 在 Top／Bottom 各 10 家乾淨 prompt 的共同尾段取 transient FP32 `[100,d_model]`，算 `v_l[p]=mean(Top[p])−mean(Bottom[p])`（**不做產業去均值、配對 SVD**）；取 `u_l[p]=v_l[p]/||v_l[p]||₂`。必須全部 finite、100 個 token 每一個 FP32 norm `>1e−8`，否則整層 fail closed；正規化後逐 token norm 距 1 不大於 `1e−5`（在 FP32、cast 前）。在同一層的相同指令尾段注入 `alpha*u_l[p]`；其餘 prefill 位置與單 token decode step 皆不改，`alpha=0` 不註冊 hook。方向 tensor 只作即時計算、不寫完整向量或 20 家殘差；只存每層方向摘要、norm min/median/max 與 Top／Bottom ticker。兩個算子均在該位置每 token 注入近似單位長度的 ray；不要據此宣稱不同層或 REQ-3 的總注入量相等。

## REQ-3：單向量、全 token 臂是獨立的原文式補充，不與主要臂混稱

【已驗證：`llm_bias/core/inference/interventions.py:record_block_states` 的 `pre` 與 `post`；Arditi `get_activation_addition_input_pre_hook`】原文加法掛在 decoder block **輸入**，覆蓋 prefill 和 KV-cached 單 token 解碼；C2 與 REQ-2 都是 block **輸出**。層號相同不代表介入點相同。

【新設計】在每層 `l` 的 Top／Bottom 乾淨 prompt，取同一共同 instruction 尾段的**最後一個 token**、**block 輸入**殘差：`w_l=mean(Top_last_pre)−mean(Bottom_last_pre)`；FP32 單位化（finite 且 norm `>1e−8`，誤差 `≤1e−5`），在 `l` 的 block 輸入對每個 prefill token 及每個新生成 token 加 `alpha*w_l`。取向量位置是事前固定的金融格式改編，不是原文 validation 選位。固定答案前綴計分與無前綴生成都使用相同位置規則；`alpha=0` 不註冊 hook。以**獨立 arm/run/schema**存結果，完整七層均可跑；先完成主要臂再啟動。記錄施加位置範圍、向量 norm 和原文差異；不要只因同 alpha 比主要臂效果大就宣稱 1D 較強。

## REQ-4：主要結果是 alpha 0 的配對生成轉移，margin 不得代替翻轉

【已驗證：`scripts/probe_concept_cone.py:run_cone_evaluation`】每層每 arm 評估相同 101 家、alpha `[0,2,3,4,5,6]`。margin 是以實際 tokenization 的 prompt **加**固定答案前綴 `{"decision": "`，由 buy/sell token log-prob 差（nats）取得；生成是無該前綴、temperature 0、最多 192 新 token，`json.loads(整段文字)` 得到完整 JSON 物件且 `decision∈{buy,sell}` 才算 strict parse（即使另有 `reason` key），不可用 regex 接受截斷輸出。產物保留可供重判的**全部已生成文字**（不裁掉長於 1024 字元的尾部），不保存 KV cache 或 raw states。

【新設計】每個 alpha 和層分報 n=101 的 strict parse 數／率、median margin、`mean(ΔM_alpha)=Σ_{i=1}^{101}[M_i(alpha)−M_i(0)]/101`、alpha 0 已解析 buy/sell 的各別基數、`buy→sell` 與 `sell→buy` 的有效配對數／各自分母／率、全部有效配對的 flip 率；基數或有效分母為零顯示 `—` 而不是 0%。未解析者保留 `unparsed`，不得當作未翻轉；同一模型不同層的 alpha 0 應分層留存，以利檢查生成一致性。只用 α0、低 `T` 層作本版控制；**沒有 random-direction 對照／獨立確認樣本，不設行為陽性 gate**：完整報告 flip 數與有效分母即可作描述，不能宣稱層峰值因果確認、方向特異性或 DIM 勝過 cone。若某臂／某層缺 101×6 記錄、缺有限 margin、缺完整基線或缺 SHA，該臂不可標為 complete。

## REQ-5：新版本產物可續跑且不影響已存在的 cone／DIM

【新設計】新增 CLI 模式 `--cohort-mode sp500_dim_paper` 加 `--dim-arm tokenwise|single_all` 與 `--dim-layers`（僅允許上述七層或其子集，輸入排序與去重後等於原定子集；完整報告需七層）；其他模式參數與舊產物 schema **不得更改**。兩臂的新 schema 分別為 `dim-tokenwise-paper-v1`、`dim-single-all-paper-v1`；新結果結構為 `targets[ticker]["L16"]["rows"]=[{"alpha":0.0,"margin":-2.1,"decision":"sell","parse_ok":true,"generated_text":"{...}"},...]`，並附 `metadata.dim_arm`、`metadata.layers`、`complete`、`summary[layer]`。新 evaluator／summary 要以 ticker＋layer 定址，不能把七層共用舊的 `cone_centroid` key 或改寫舊 `run_cone_evaluation`／`alpha_summary` 對既有 cone 產物的讀寫行為。僅 Qwen3.5-4B，本地模型預設參數沿用 cone，明示 `--model .cache/models/qwen3.5-4b --model-dtype bf16 --population-csv data/sp500_constituents_2020_2025.csv --split-seed 20260923 --alphas 0 2 3 4 5 6`。`--output-json` 必須落在 `artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-<run-id>/<arm>/result.json`；若已有完整 run，唯讀回報 complete，絕不覆寫；未完成只允許 metadata、source SHA、alpha grid 和已有 ticker/layer 記錄完全一致時以原子寫入續跑，任何欄位缺失／重複／格式衝突即拒絕。僅 compact derived outputs、模型及 tokenizer/prompt/split/source hashes、實際選層 C2 `T`、版本、top/bottom tickers、每層每 ticker 的 alpha 計分/生成文字/strict parse、彙總 Markdown alpha 欄表；不保存原始殘差、完整 activation 或 KV。CLI 可用 `--smoke-tickers`，但只許指定 101 家中 ticker，輸出標 `smoke` 且不得作正式結果。

## 切片邊界與驗收

| Slice | 可寫檔案 | 交付契約與無 GPU 驗收 |
|---|---|---|
| 1：驗證與方向數學（REQ-1、2、3 純函式） | `scripts/probe_concept_cone.py`、`tests/test_probe_concept_cone.py` | 加入排名／C2／SHA fail-closed 校驗、候選層和兩種方向計算；fake `[20,100,3]` states 測方向正負、每 token 正規化、零 norm／NaN 拒絕、402／101 與單位 norm。`uv run pytest -q tests/test_probe_concept_cone.py`。 |
| 2：主要臂 workflow（REQ-2、4、5） | `scripts/probe_concept_cone.py`、`tests/test_probe_concept_cone.py` | 新 CLI 模式與獨立 schema、重用既有 split／ranking 驗證／固定前綴／greedy 生成流程；新 evaluator 必須以 `ticker→layer→rows` 定址並於續跑時檢查完整 alpha grid，舊 `run_cone_evaluation` 的 `cone_centroid` 與舊 `alpha_summary` 不變。fake model 測兩層不互相覆寫、每層 post-block 注入位置、α0 no-op、完整 JSON、獨立 alpha 欄、原 alpha 0 生成 buy/sell 分母與零分母 `null`／表格 `—`、續跑污染拒絕。`uv run pytest -q tests/test_probe_concept_cone.py && uv run python -m compileall -q scripts`。 |
| 3：補充臂 workflow（REQ-3、4、5） | `scripts/probe_concept_cone.py`、`llm_bias/core/inference/interventions.py`、`tests/test_probe_concept_cone.py`、`tests/test_core_inference.py` | 在 `llm_bias/core/inference/interventions.py` 實作 generic pre-block context hook，`single_all` 用獨立策略在所有 prefill 及 cached decode token 注入；舊 cone evaluator 的 `shape[1]==1` skip 仍保持原樣。fake model 測 extraction/input 同位、prefill＋cached decode 皆注入、α0 no-op、hook 即使錯誤也移除、獨立 schema 與總量警語。`uv run pytest -q tests/test_probe_concept_cone.py tests/test_core_inference.py`。 |
| 4：真模型驗證與結果（REQ-1–5） | 忽略追蹤的 `artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-*/`，完成後新增 `docs/concept-cone-steering/details/status-dim-layer-sweep-v1.md` | 先 1 家×2 層×兩 arm 煙霧測試，再事先指定 7 層／101 家、先主要後補充；只在完成後記實際命令、run path、表與解讀限制。`uv run pytest -q tests/test_probe_concept_cone.py tests/test_core_inference.py && git diff --check`；CUDA 記憶體不足或任一校驗失敗立即停，不把部分 run 寫作完整結果。 |

**實作共同界線：**不新增 `scripts/` 之間的隱式 import、不載大 checkpoint 作 unit test、不碰 `scripts/probe_dim_steering.py`、舊 `result.json`、`docs/research-scripts.md` 或 `scripts/probe_operator_comparison.py` 的現有 dirty changes。新的 shared hook 只放 `core/inference`；prompt/ranking/方向的研究語義保留在 owning experiment CLI。切片逐一以測試綠燈驗收後才進下一片，先唯讀審查本協議與實作接口，再修改程式。

## 2026-09-25 執行後的資料重疊勘誤（保留上文事前原文）

REQ-1 的「101 家不可參與層位選擇」及文中「獨立 C2」若被解作**公司名單互斥**，與實際來源不符。執行後核對 [C2 的 854 條雙向方向清單](../../../artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/pairs/directions.json) 與 [paper cone 原始 101 家名單](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/c2-guided-paper-20260924/result.json) 得出：427 家 C2 公司中有 **89 家**屬於本次 101 家受測組。C2 `T` 曲線及 L14–L18／L0／L31 的選層在 DIM 評估之前已存在，**未以這 101 家的 DIM 結果倒選層**；但 101 家僅相對於本次 DIM ranking／方向擬合留出，並非相對於 C2 選層的獨立公司樣本。原選層與實際執行不回溯改寫；所有「held-out 層峰值確認」宣稱均不成立。本勘誤與完整解讀見[執行紀錄](status-dim-layer-sweep-v1.md)。
