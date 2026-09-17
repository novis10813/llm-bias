# Evidence-insensitivity Phase 3：組差的上游因果定位（protocol proposal）

**狀態**：frozen（2026-09-17 用戶批准；run ID `phase3-gpu-bf16-01`，雙模型）
**對象模型**：Qwen3.5-4B（32 層）＋ Gemma-4-E2B-it（text tower 35 層，0–34；
Rev 1.1 誤書 42 層，見 Rev 1.2）——雙模型統一執行
**研究線定位**：機制相第二階段（因果定位）。Phase 2 已排除 capture layer
（Qwen L15 / Gemma L18）1D stance 狀態的 offset/gain 承載；本 phase 以
**within-company 極性 transfer patching** 量測「行為分組的因果中介位置」：
在哪些（層 × span 位置）把 P15 run 的狀態搬進 N15 run（或反向）會改變
決策／margin。本 phase 是線內第一個（也是唯一）干預臂。

---

## 1. 核心假說與文獻邊界

**研究問題**：evidence-responsive 與 evidence-insensitive 組在 P15 的決策差
（Qwen：buy vs sell；Gemma：buy vs buy 中的固定 buy）的因果中介在哪個
（層，位置）？具體地拆成兩個可判別子問題：

- **S（state-mediated）**：pro-buy 訊號只在 responsive 組的 P15 狀態裡
  → T1 patch（P15 狀態 → N15 run）只在 R 組產生效應，FS 組無。
- **R（readout-mediated）**：兩組 P15 狀態都含 pro-buy 訊號，但 insensitive
  組的讀出不用 → T1 在兩組都有效應（組差在 patch 位置之後／讀出段）。

Phase 2 的「狀態軌跡幾乎重合＋單一共享方向」使 R 先驗較不可忽視；但
selective-intervention V1 已證明 L15 k=8 entity-difference 子空間的移除
能縮小行為 group gap（1.27→0.59 nats）——L15 附近確實有因果內容，只是
不在 1D stance 軸上。本 phase 不預設答案，量完整（層×位置）效應地圖。

**上游依據**：

- [Phase 1 報告](report-phase1.md)：分組表（Qwen 42R/360FS、Gemma 317R/85FB
  discovery）；行為端點（greedy JSON decision）與 decision-position margin。
- [Phase 2 報告](report-phase2.md)：capture layer 狀態不承載組差（offset/
  gain 皆 null）；狀態反應形狀為模型特性。
- [Activation-patching causal tracing](../../activation-patching-causal-tracing/proposal.md)：
  層×span resample patching 方法（source forward 記錄狀態 → target forward
  帶 hook 替換 → 端點量測）；本 phase 端點從 margin 換成本線的
  generation decision（primary）＋decision-position margin（auxiliary）。
- [Selective-intervention V1](../../selective-intervention/report.md)：L15
  k=8 子空間移除有效（efficacy 陽性）——patching 機制在本模型族有效。
- [Core intervention machinery](../../shared-experiment-core.md)：
  `core/inference/interventions.py` 的 `record_block_states`（pre/mid/post
  全序列狀態，transient）與 `residual_interventions`（post-block 替換）
  直接重用；不新增 shared mechanics 的本地副本。

## 2. 術語定義

- **source / target run**：同一公司在同一 prompt 條件下的完整 prompt
  forward。T1：source = P15、target = N15。T2：source = N15、target = P15。
- **patch 座標 (L, p)**：層 L（post-block 狀態，與 Phase 2 capture 同一
  語義）× 位置 p（token 索引，0-indexed）。
- **四位置**（raw formatted prompt 空間，span 終點前一 token）：
  `entity`（`Stock Name: Y\n` 終點）、`evidence`（evidence span 終點）、
  `instruction`（instruction span 終點＝Phase 2 capture position）、
  `prompt_end`（prompt 最終 token）。
- **ΔM（Stage 1 端點）**：patched target 在 decision position
  （`prompt + DECISION_PREFIX`，Phase 1 Rev 1.2 定義）的 buy/sell margin
  減去 baseline（unpatched target）同位置 margin。Stage 1 的序列為
  `raw_prompt + DECISION_PREFIX`；前三位置索引與 raw prompt 相同，
  `prompt_end` 取該序列最終 token（記為 `score_end`）。
- **flip（Stage 2 端點）**：patched target 的 greedy 生成決策
  （128 tokens，Phase 1 parse 規則）≠ baseline 決策。net direction =
  （sell→buy 數）−（buy→sell 數）。
- **active 座標**：Stage 1 中 median |ΔM| 前三的 (L, p) 組合
  （兩方向合併、全樣本），與 `(L_anchor, instruction)`（Qwen L15 /
  Gemma L18，Phase 2 座標，必測）之併集，去重後 ≤ 4 個。
- **樣本**：Qwen = 42 家 discovery R 全取＋42 家 discovery FS
  （seed 20260916 排序後 shuffle 取 42）；Gemma = 42 家 discovery R
  （自 317，同 seed 規則）＋42 家 discovery FB（自 85，同 seed 規則）。
  每模型 84 家。
- **層格（Stage 1 scan）**：Qwen `{0,4,8,12,15,19,23,26,30,31}`；
  Gemma `{0,5,10,15,18,23,28,32,33,34}`（Rev 1.2 修正；原 `{0,5,10,15,18,23,28,32,36,41}`
  的 36/41 超出實際層數）。各 10 層，含各自 capture 錨點與最終層。

## 3. 預設 Input / Output 契約

### 3.1 Input

| 項目 | 來源 | 契約 |
|---|---|---|
| 分組表／span／prompt 文字 | 各模型 Phase 1 `phase1-gpu-bf16-01`（prompts.jsonl primary arm ＋ summary groups） | 與 Phase 2 相同來源；span 終點 bounds 檢查 fail-closed |
| Phase 2 capture layer | Qwen L15 / Gemma L18（各自 Phase 2 run metadata） | `L_anchor` |
| model | `.cache/models/qwen3.5-4b` / `.cache/models/gemma4-e2b-it`（bf16） | identity 驗證 |

### 3.2 Output（compact，不存 raw states——`record_block_states` 的 tensor
僅在 run 生命週期內存在於 GPU/CPU memory）

`artifacts/<model>/evidence-insensitivity/runs/phase3-gpu-bf16-01/`：

- `prepare/selection.json`：樣本 84 家×組別、層格、位置表（4 位置×模型
  的 token 索引）、provenance（seed、來源 run、span bounds 驗證結果）。
- `forward/scan_records.jsonl`：每（公司×L×p×direction）一筆：baseline_M、
  patched_M、delta_M（finite）、baseline_decision、parse 失敗標記。
  84×10×4×2 = 6,720 筆/模型。
- `forward/focus_combos.json`：active 座標清單＋選取規則輸出（median |ΔM|
  排序表）。
- `forward/generation_records.jsonl`：每（公司×active(L,p)×direction）：
  baseline_decision、patched_decision、flip、net_direction、baseline_M、
  patched_M（finite）。84×(≤4)×2 ≤ 672 筆/模型。
- `forward/metadata.json`：model identity、layer/position 表、determinism
  （20 筆 unpatched re-run：margin max Δ 與 decision mismatch 數）、
  G-3B efficacy probe 結果、forward 計數。
- `analyze/summary.json`：gates、效應地圖（flip rate 與 median ΔM 的
  層×位置×方向×組別表）、S-vs-R 裁決統計、sector 敏感性、雙模型對照
  所需的結構表。
- `manifest.json`：`ArtifactRun` lifecycle（prepare/forward/analyze）。

## 4. 凍結常量

| 項目 | 值 | 來源 |
|---|---|---|
| 方向 | T1（P15→N15）、T2（N15→P15） | 本提案 |
| 位置 | entity / evidence / instruction / prompt_end（§2） | 本提案（span 邊界＝Phase 1 契約） |
| 層格 | §2 凍結（10 層/模型） | 本提案（含 Phase 2 錨點） |
| Stage 2 規則 | median |ΔM| top-3 ＋ (L_anchor, instruction)，去重 ≤4 | 本提案 |
| 端點 | Stage 1：decision-position margin（FP32）；Stage 2：greedy 128 tokens ＋ Phase 1 parse 規則 | Phase 1 frozen 端點 |
| 樣本 seed | 20260916 | Phase 1/2 慣例 |
| 容差 | margin determinism 0.01（對齊 Phase 1 G-P3 bit-exact 慣例的 re-run 檢查）；decision 0 mismatch | Phase 1 慣例 |

## 5. 預先註冊分析

1. **效應地圖**：median ΔM（Stage 1）與 flip rate（Stage 2）的
   層×位置×方向表，分組別（R vs FS/FB）。
2. **S-vs-R 裁決**：在每個 active 座標，T1 的 R 組效應 vs FS 組效應
   （margin：Welch；flip：Fisher exact）。全 active 座標 FS 效應 ≈ 0
   且 R 效應 > 0 → state-mediated；兩組皆 > 0 → readout-mediated
   （組差在座標之後）。
3. **T1 vs T2 對照**：sufficiency（T1）vs necessity（T2，R 組 P15 run
   被 N15 狀態抹除 buy）的層結構是否一致。
4. **位置控制**：entity 位置是天然控制（entity 內容跨條件相同——
   patch 不應效應）；若 entity 位置出現效應，視為 span 邊界或位置
   混淆，需查證後解釋。
5. **sector 敏感性**：active 座標的 flip rate 分 sector（Qwen 42 R 集中
   IT 的背景）。
6. **雙模型對照**：同分析的結構對照（不比絕對幅度）。

## 6. Gates（fail-closed）

| Gate | 定義 | 門檻 |
|---|---|---|
| G-3A baseline integrity | 20 筆 unpatched re-run（seeded 分層）：margin max Δ 與 decision mismatch | Δ ≤ 0.01 且 0 mismatch |
| G-3B intervention efficacy | (L_final, score_end) T1 probe：N15 target 被 P15 終態 patch 後，全樣本 median ΔM 應被推向 sell（P15 的 pro-buy 終態進 N15 run……方向定義：ΔM = M_patched − M_N15_baseline，P15 狀態較 pro-buy → 預期 ΔM > 0） | median ΔM > 0（≤0 即機制失效，fail-closed 停止） |
| G-3C Stage 2 解析功效 | generation 記錄的 parse 成功率；每組×active 座標 n | parse ≥ 0.95 且每格 n ≥ 30 |

（G-3B 的方向直覺：把更 pro-buy 的 P15 終態搬進 N15 run，decision-position
margin 應上移；若下移或不動，patch 管線有問題。）

## 7. Run 結構

`prepare → forward（scan pass → 規則選 active → focus pass）→ analyze`，
每模型單一 run `phase3-gpu-bf16-01`（`ArtifactRun` lifecycle）。

| 階段 | 內容 | 量級 |
|---|---|---|
| prepare | 分組表 fail-closed、樣本、位置表（entity 邊界從 frozen 模板推導＋span bounds 驗證）、L_anchor 讀取 | CPU |
| forward | source 記錄：84×2 條件 × 1 forward（全層 `record_block_states`，+prefix 序列）；scan：84×80 patched forward＋margin；active 選取（規則）；focus：≤672 patched forward＋128-token generation；20 筆 determinism re-run | 每模型 ≈ 7,600 forwards＋≤672 generation（~1h，GPU） |
| analyze | gates、§5 全分析、summary | CPU |

狀態 memory 峰值：單公司全層 × 2 條件 × 全序列 ≈ 120 MB（bf16，CPU
offload），逐公司處理。

## 8. 邊界與限制

- patching 識別的是**被測 contrast 的因果中介**（T1/T2 的極性狀態差），
  不是 entity 內容的因果（entity 跨條件相同，不作 source 差異）。
- discovery split only；hold-out（Qwen 8R/93FS、Gemma 78R/23FB）保留給
  confirmation（Phase 3 後續版本，若 discovery 結果值得）。
- 單位置替換（每座標 1 token）；不 patch 多位置組合（V2 選項）。
- 不存 raw states；換層/換位置/換座標規則皆需重跑 forward。
- Gemma 的 T2 baseline 為 100% buy（P15），flip 定義為相對各自 baseline。
- order-swap（recency）不屬本 phase；主臂 only。
- bf16 forward；margin FP32 scoring（Phase 1 慣例）。

## 9. 版本分立觸發條件

任一變更即建新版本：方向（T1/T2 定義）、位置集、層格、端點（generation/
margin）、active 座標選取規則、gate 門檻、樣本（公司集/seed）。

## 10. 實作與驗證

- 新 module `llm_bias/evidence_insensitivity/phase3.py`＋ operator
  subcommands `phase3-prepare/forward/analyze`（`--model`/`--model-slug`/
  `--phase1-run-id`/`--phase2-run-id`）。
- 重用：`core/inference/interventions`（record_block_states＋
  residual_interventions）、`core/inference/continuation_scoring`
  （margin）、`core/inference/generation`、`evidence_insensitivity/
  screening.parse_decision`、Phase 1 span/分組。不 import 其他
  experiment package。
- 測試（fake model，無 GPU）：entity 邊界推導與 fail-closed、位置表
  bounds、active 座標規則（top-3＋anchor、tie、去重）、flip/net
  direction 計數、G-3A/B/C 門檻邊界、scan→focus 全流程 fake lifecycle
  （含 intervention hook 的 fake block 替換驗證：patch 確實改變 fake
  margin）、summary 過 core `_RAW` guard。
- 驗證：`uv run pytest -q`、`compileall`、`uv build`、`uv lock --check`。
- 每模型 1-prompt real-model preflight（source 記錄＋1 patch＋margin
  finite；G-3B probe 方向）。

## 11. Revision history

- **Rev 1（2026-09-17）**：初稿。動機＝Phase 2 雙模型 state-level null
  （報告 §1/§7）；S-vs-R 二分法為本 phase 的核心可判別問題；方法移植
  activation-patching causal tracing 的 patching 協議（端點換成本線
  generation decision＋margin 雙讀數）；兩階段（margin scan → generation
  focus）以控制 generation 成本。
- **Rev 1.1（2026-09-17，frozen）**：用戶批准 formal run。
- **Rev 1.2（2026-09-17，frozen 常量修正）**：Gemma preflight 發現 text tower
  實際為 35 層（`text_config.num_hidden_layers = 35`；Phase 2 Step A sweep
  artifacts 為 0–34），Rev 1.1 層格的 36/41 不存在（`record_block_states`
  fail-closed 拒跑）。此為 frozen 前事實錯誤（Gemma 尚無任何 forward），
  依「保留所有有效層；無效層換最近未用有效層；必含最終層」規則修正為
  `{0,5,10,15,18,23,28,32,33,34}`（36→34、41→33；10 層中 8 層不變，
  L18 錨點不變）。Qwen 層格驗證無誤，不變。Phase 2 proposal 同處「42 層」
  書面誤記，另立 Rev 1.3 erratum。
- **Rev 1.3（2026-09-17，implementation alignment，不改任何 frozen
  門檻／樣本／座標定義）**：(a) Gemma coordinate-space 修正：`load_model`
  以 `jlens.from_hf(..., force_bos=True)` 包住模型，對有 `bos_token_id`
  的 tokenizer 就地設 `add_bos_token=True`（Gemma 預設 False → 推理時序列
  多 1 個 BOS token）；prepare 與 forward 分屬兩個程序，若 prepare 用未
  修正的 tokenizer 導出位置表，全部 patch 座標位移 1 token（Gemma
  preflight 測得全 coordinate ΔM=0，G-3B probe 會 fail-closed；未有任何
  Gemma formal forward 在錯誤座標下執行）。修正：`core/model.py` 新增
  `load_tokenizer_for_inference`（複現 force_bos 的 in-place 修正），
  prepare 改用它；三 span 位置改由 frozen template 的 char 範圍在實際
  序列內重新導出，並對 Phase 1 存檔 token span 做 fail-closed cross-check
  （須精確一致或統一 +1 BOS offset）；forward 開頭加 coordinate guard
  （存檔 `prompt_end` 須等於 forward 時編碼長度 −1）。provenance：
  `selection.json` 增 `stored_span_offset`（Qwen 0、Gemma 1）。Qwen
  `bos_token_id=None`（force_bos no-op），其 formal run 座標有效、不需
  重跑。(b) 實作對齊註記：Stage 1 scan 記錄為 margin-only（Stage 1 不做
  generation；§3.2 所稱 scan 行的 baseline_decision／parse 標記實際由
  generation_records 承載）；`net_direction` 可由每筆 generation record 的
  （baseline_decision, patched_decision）導出，未另存欄位。
