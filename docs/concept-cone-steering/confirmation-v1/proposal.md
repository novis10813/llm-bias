# Concept-cone steering 確認實驗 confirmation-v1：總協議

**狀態：**事前協議，2026-09-25 撰寫，smoke 前凍結（本檔 commit 之後才允許跑 smoke）。**範圍：**Qwen3.5-4B、Gemma-4-12B、GLM-4-9B、GPT-OSS-20B；2024 S&P 500 母體 503 家，seed `20260923` 固定 402 家建方向、101 家受測。**取代：**無；本協議與 [DIM V2](../details/proposal-crossmodel-dim-layer-sweep-v2.md)（下稱 R0）並行，不改寫 V2、Qwen V1 或任何既有產物。分項協議：

| 協議 | 內容 | Claims |
|---|---|---|
| 本檔 | renderer、劑量、CAL、readout、flip 定義、gates、provenance、執行 | C1、C4、C12 |
| [operator-comparison-v2](../operator-comparison-v2/proposal.md) | random／jitter／shuffle、single neuron、cone、DIM⊥rand4、C5／C8 判準 | C3、C5、C8 |
| [evidence-sensitivity-v1](../evidence-sensitivity-v1/proposal.md) | pos／neg／zero／mixed2、anonymous、flip dose、C6／C7 判準 | C6、C7 |
| [c2-v3-steering-prompt](../c2-v3-steering-prompt/proposal.md) | 跨公司 span×layer patching（R6）與生成驗證（R7）、A2 | C2 |
| [generalization-v1](../generalization-v1/proposal.md) | LOSO（R9／R10）、split seed 複製（R11）、C10 用詞規則 | C10 |

以下規則在任何 101 家結果揭露前固定；方向來源、主要指標、controls 或 gate 改變時另立版本。

## 使用者已決定的範圍（2026-09-25）

- 原樣 commit V2（`d9f96be`）；新 code 在獨立 worktree／venv（branch `confirmation-v1`）開發，主工作樹在 R0 完成前不動。
- zero-evidence 用空 body；加入 mixed2；tier2 要跑，但 **depth sweep（原 R8）刪除**。
- single-neuron arm **每個模型都做**（論文主線：single neuron → single vector → cone）。
- α0 parse gate 只記錄、不自動降級（先看結果再討論）；CAL 由預先登記規則自動產生網格，不再人工審核（只設自動 abort）。
- 最佳層 random（R12）、Qwen single_all（R2b）與第 5 個模型不做。

## 模型登錄（`llm_bias/core/steering/protocol.py:MODEL_REGISTRY`）

| slug | 層數 | 主要層（C2-427 instruction peak） | C2-427 band | K | dtype | MLP |
|---|---:|---:|---|---:|---|---|
| `qwen3.5-4b` | 32 | 16 | 14–18 | 100 | bf16 | dense |
| `gemma4-12b-it` | 48 | 27 | 26–32 | 100 | bf16 | dense＋post-FFN norm |
| `glm4-9b-0414` | 40 | 19 | 17–21 | 98 | bf16 | dense＋post-MLP norm |
| `gpt-oss-20b` | 24 | 14 | 12–16 | 99 | native（MXFP4） | MoE（32 experts） |

主要層由 C2-427 summary 的 SHA-256 綁定（與 V2 相同）。GPT-OSS L14 是在 `medium` reasoning、固定前綴 readout 下選出的（off-path），只作描述性定位。「最佳 DIM 層」只作探索，不重選主要層。

## Prompt 與模板

- 五種 evidence condition（`prompts.py:render_decision_prompt`）：`balanced`（P1,P2,N1,N2；與 `_render_frozen_prompt(order=0, reverse=False)` 逐 byte 相同）、`pos`（P1,P2）、`neg`（N1,N2；兩者與 `build_prompt_v2(..., reverse=False)` 逐 byte 相同）、`zero`（保留 `— Evidence —` 與收尾 `—`，body 為空，evidence span 明確為空）、`mixed2`（P1 後接 N1）。
- 10 個預先登記的匿名身分（`ANON_IDENTITIES`）：`TICKER / Company X` 加 9 個長度與格式各異的虛構 placeholder；freeze 時對 `data/*constituents*.csv` 的所有 ticker 與公司名做撞名檢查。
- Chat template：thinking 關閉；GPT-OSS 傳 `reasoning_effort="low"`；模板呼叫 `strftime_now` 者（GPT-OSS）以渲染變數固定日期為 `2026-09-25`（與 R0 相同機制），metadata 記 `chat_template_date`。
- K：四個 tokenizer 下，503 家 × 5 condition ＋ 10 個匿名身分 × 5 condition 的共同 instruction 尾段長度必須等於上表，且尾段 token ids 在各 family 間完全相同（已於 CPU 實測通過）。
- Gemma 的 chat template 自帶 `<bos>`，loader 又強制加 BOS（double-BOS）；與所有既有 Gemma run 相同，只記錄、不修改。

## 劑量

- 注入點：主要層 block 輸出（post-block residual），只在 prompt 的 steer suffix（instruction 最後 K 個 token）且只在 prefill；decode token 不改（與 V2 `tokenwise` 相同）。
- 以 raw DIM 差 `d[p] = mean(Top10 h[p]) − mean(Bottom10 h[p])`（fp32，balanced 乾淨 prompt）為錨：operator 的 α=1 劑量 `B[p]` 為等範數 `‖d[p]‖·u[p]` 或等投影（在 `d̂[p]` 上的投影等於 `‖d[p]‖`）；注入 `α·B[p]`。DIM 本身 `B = d`，與 V2 完全相同。
- 每個 operator 記錄 dose 表：注入 norm（median／min／max）、`cos(B, d̂)`、在 `d̂` 上的投影，以及 `‖B‖ / median‖h‖`（同層、Top/Bottom 20 家 suffix 狀態）。跨模型表不用 raw α 當欄，改用 `α/α_50`、flip 率與 ΔM。
- Top/Bottom10 由每個模型自己在 402 家 construction 上的乾淨固定前綴 margin 排序（`(margin, ticker)` 升序）；ranking arm 同時算 503 家，供 CAL、LOSO 與 split seed 使用。

## CAL（R1）：construction-only 劑量校準

- 公司：三個 split seed（20260923／24／25）的 construction 交集，扣除本模型在任一 seed 下的 Top/Bottom10；以 seed `20260926` 打亂後，依序生成 α0（balanced），在前 120 家內湊 12 家 buy＋12 家 sell，某類不足時由另一類補滿 24 家。這些公司永遠不在任何 eval split 中。
- 梯度：±{0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64}，DIM、主要層、balanced。同一公司同一符號連續 2 點 unparsed 即停止往上（未跑的點視為 unparsed、未翻）。
- 某方向的 α0 起始類別少於 6 家時：sell→buy 改用同批公司的 `neg` 條件、buy→sell 改用 `pos` 條件，只跑該符號的半邊梯度，標記 cross-condition。
- 規則（`summary.py:cal_rule`）：
  - `α_flip50(dir)`：該方向 ITT on-target flip 率 ≥ 50% 的最小梯度點；到 64 都沒有則右截斷。`α_flip90` 同理。
  - `α_50`：兩方向 `α_flip50` 幾何平均最近的梯度點（只有一方向有值就用它）；`α_90` 同理，無值取 `α_hi`。
  - `α_hi`：兩個符號在 24 家上 primary parse 率都 ≥ 90% 的最大梯度點（truncated 也算不可讀）。
  - `α_lo`：從 0.0625 往上、兩方向 flip 率都 ≤ 10% 的連續區段上緣；0.0625 已超過 10% 則左截斷（`α_lo = 0.0625`）。
  - Full 網格 `{0, ±2} ∪ ±(梯度 ∩ [α_lo, α_hi]) ∪ {±α_next}`（`α_next` 為 `α_hi` 上一點，≤ 64），非零點上限 14，超過時從 `< α_50/2` 的低端每隔一點刪。縮減網格 `G_red = {0, ±α_50, ±α_hi}`；random 網格 `{±α_50, ±α_90, ±α_hi}`。
  - 另記 collapse dose 與 truncation dose。
- 自動 abort（eval arm 拒絕啟動，只保存 CAL）：無 `α_hi`、無 `α_50`、`α_hi < α_50`，或 `calibration.json` 記錄的程式 SHA 與目前不同。

## Readout 與 flip

- **Primary：**生成文字的 complete-object 解析（`decision_parsing.py` 的 6 種外殼，與 V2 相同）。**Secondary：**整段 `json.loads` 的 strict 解析、固定前綴 margin（`{"decision": "` 後 `log p(buy) − log p(sell)`，fp32 head 分塊計算）。GPT-OSS 與 Gemma 的固定前綴 margin 在報告中標為 off-path。
- 每列另存 `finish`（eos／max_new_tokens）、`n_new_tokens`、`path_class`（direct_json／brace_newline／fenced／thought／harmony_final／harmony_analysis／other）、`unparsed_kind`（`collapsed` = eos 但 malformed；`truncated` = 撞到 192 token），以及 **realized-path margin**：同一次 greedy 生成中，輸出決策值那一步的原始 logits 上 `log p(buy-token) − log p(sell-token)`（counterfactual token 為原 token 字串中 buy↔sell 互換；值被拆成多個 token 或互換後不是單一 token 時記 status、不猜）。α0 列另存生成 token ids，供 teacher-forced readout。
- **Flip：**以同一 prompt、同一 condition 的 α0 生成決策為條件。α>0 的 on-target 為 sell→buy、α<0 為 buy→sell；off-target 另列。Primary 分母為 ITT（該 α0 類別全部公司，steered unparsed 記為未翻）；兩端可解析的條件式 flip 率為 secondary。分母為 0 記 `None`，表格記「—」。
- α0 每個 prompt × condition 只生成一次（`alpha0/result.json`），所有 arm 共用。α0 parse gate（≥ 90%）只記錄。

## Gates 與結構零層

- `gates` arm（smoke 失敗即停，full 只記錄）：ABNB α0 重跑逐字相同；零位移 hook（主要層、α=2、`B=0`）與 α0 逐字相同；最後一層注入（`B=d`、α=8）與 α0 逐字相同；ABNB α0 全文與 V2 smoke-01（Gemma/GLM/GPT-OSS）或 Qwen V1 paper-01 相同；主要層 `difference_sha256` 與 V2 smoke-01 相同（Qwen V1 只存單位向量，不比）。
- `dim` arm 另跑最後一層 10 家 × ±α_hi 的結構零層診斷；margin 差 > 1e-6 或文字不同時，summary 記 `structural_zero_ok=false` 並停止解釋該模型的翻轉，直到找出原因。
- `dim` arm 與 R0 在 ±0.5／±1／±2 的共同點逐字比對生成（決定性檢查，不作沿用）。

## Arms 與 tier

| Arm | 內容（原計畫 ID） | Qwen | Gemma | GLM | GPT-OSS |
|---|---|---|---|---|---|
| gates, ranking, alpha0, cal | 身分、503 家排序、α0、CAL（R1） | T1 | T1 | T1 | T1 |
| dim | 主要層 DIM，full 網格（R3a 的 DIM arm、C1） | T1 | T1 | T1 | T1 |
| dim_layers | Qwen L0、L14、L15、L17、L18 full 網格＋L31 診斷（R2） | T1 | — | — | — |
| random, jitter | 5 seed 共用 random；3 seed jitter（R3a） | T1 | T1 | T1 | T1 |
| ops | neuron、cone2、cone4、cone4 等投影、DIM⊥rand4（R3b/R3c） | T1 | T1 | T2 | T2 |
| shuffle | shuffled-label DIM 3 seed（R3d） | T2 | T2 | — | — |
| evidence, anon | R4、R5 | T1 | T1 | T1 | T1 |
| c2v3, c2v3_gen | R6、R7 | T1 | T1 | T1 | T1 |
| loso | R9（101 家 eval） | T1 | T1 | T1 | T1 |
| loso_construction | R10（construction 公司） | T1 | T2 | T1 | T2 |
| split_seed | R11 | T1 | T1 | T1 | T1 |

與 2026-09-25 整合計畫的差異：R8 刪除（使用者決定）；neuron 擴及四模型（使用者決定）；DIM arm 不沿用 R0 列（R0 列缺 finish／realized 欄位），改為全部重算並逐字比對作決定性檢查。

## 產物、續跑與 provenance

- 輸出：`artifacts/<slug>/concept-cone-steering/runs/<run-id>/<arm>/result.json`（CAL 為 `cal/calibration.json`），run id 為 `confirmation-v1-<date>-{smoke,full}-NN`；tier1 與 tier2 使用同一 run id（共用 ranking、α0 與 CAL）。
- 每完成一個 (operator, prompt) 單位原子寫入；續跑時 arm metadata（含程式 SHA、direction SHA、CAL 網格）必須完全相同，且每列的決策欄位必須能從全文與 finish 重新推導，否則拒絕。`--max-rows` 用於測試中斷後續跑，結果需與不中斷逐 byte 相同（unit test 已驗證）。
- metadata：checkpoint identity、tokenizer、template SHA／kwargs／日期、split／population／五個 family 的 prompt SHA、K、主要層與 C2 source SHA、程式檔 SHA-256（程式身分以此為準）；每次呼叫另記 `invocations.jsonl`（git commit、scoped dirty flag、torch／transformers／jlens 版本、GPU）。git commit 不放進 arm metadata，所以 tier1 與 tier2 之間只改文件的 commit 不會讓共用的 ranking、α0 與 CAL 失效。full 模式在 `scripts/`、`llm_bias/`、`tests/`、`pyproject.toml`、`uv.lock` 有未 commit 修改時拒絕執行。
- 不保存 hidden states、殘差或梯度；只保存列、dose 統計與 SHA。
- Bootstrap：company（或 identity）層級重抽，B=2000，seed `20260925`，95% percentile。

## 執行

```bash
# CPU（worktree 內）
uv run python scripts/freeze_steering_protocol.py --run-id confirmation-v1-freeze-20260926
uv run python scripts/analyze_decision_boundary.py --run-id audit-v1-20260926
uv run python scripts/summarize_c2_overlap_exclusion.py --run-id audit-v1-20260926
# smoke：每個模型一個 process，只 load 一次；smoke 用固定劑量 {0, ±2, ±8}、2 家 CAL 公司
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
  uv run python scripts/probe_steering_confirmation.py --model .cache/models/gemma4-12b-it \
  --phase smoke --run-id confirmation-v1-20260926-smoke-01 --arms all
# full：先 tier1，再 tier2（同 run id）
uv run python scripts/probe_steering_confirmation.py --model .cache/models/<slug> \
  --phase full --run-id confirmation-v1-20260926-full-01 --arms tier1
```

Smoke 先以 `--max-rows 40` 中斷一次再不設上限續跑，確認 GPU 上的續跑與決定性。Smoke 通過條件：`gates` 全部為 true 或 None（None 僅限參考 run 不存在）；各 arm `complete=true`；結構零層與 self-patch 為 no-op；operator dose 表符合構造（cone4 `cos=0.5`、等投影的投影量等於 DIM norm、random 與 DIM 等範數）；`c2v3` self-patch ≤ 1e-6；人工一次性審核 ABNB／AEP 的生成文字與 path class。smoke 通過後直接排入 full 佇列。

## 限制

- 101 家中 89 家同時在 C2-427 選層集合內（A2 另報排除後的 band）；Gemma 與 GLM 的 Phase 2A gate 曾 override。
- GPT-OSS 在 `low` 下仍可能撞到 192 token 上限；截斷率列為限制。
- 同一 α 在不同 operator 間是等範數或等投影，不是等效果；跨 operator 比較以 dose 表與 α_eff 為準。
- 結論只對這四個模型、這個 prompt family 與這個 split 成立。
