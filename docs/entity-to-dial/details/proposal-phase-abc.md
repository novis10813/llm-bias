# Entity-to-Dial 路徑解剖實驗協議

**狀態**：discovery completed（Rev 1.3：三段 formal run 完成，2026-09-11；Gate A1 / B / C 皆 fail；見 [report.md](report-phase-abc.md)）  
**對象模型**：Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構，32 層，full-attention 層為 L3/7/11/15/19/23/27/31）  
**研究線定位**：承接 balanced-evidence-gap Phase 2B 的 entity span 承載帶定位結果（L0–11）與 investment-dial 線的 L15/N8490 model-level stance prior，設計三段式因果解剖，找出 entity identity 在 L0–11 寫入殘差流後，如何傳遞至決策 dial 座標、以及是否繞過 dial 走另一條路徑。本協議不重開 balanced-evidence-gap Phase 1–3 的任何 gate，也不修改 entity-cell 或 investment-dial 的既有結論。

---

## 1. 背景：三條線的已知邊界與中間缺口

### 1.1 已知的兩端座標

| 線 | 座標 | 已確認的因果效應 | 限制 |
|---|---|---|---|
| Entity Cell | L0–4，JNJ/BAC/CAT/PLTR 各一顆 MLP 神經元 | 壓制 → 事實失憶（最大 −8.51 nats） | 決策翻轉 0 次（與決策路徑功能解離） |
| Investment Dial | L15/N8490 | ±4s 推注 → margin ±1.0–1.1 nats，決策翻轉 83/199 | 對所有公司一致施力，不認識特定 entity |

### 1.2 balanced-evidence-gap Phase 2B 的路徑定位

- Entity span 在 L0–11 承載 entity-specific 決策影響（L0–5 normalized transfer T ≈ 1.0，L12 後 CI 包含 0）。
- L12–15 為 handoff 區間，instruction span T 在 L15 達峰值 +0.464。
- Phase 3 對 L19/L20/L26 的 single-neuron mlp_addition 全部 null（|mean ΔM| ≤ 0.012 nats，低於 control noise floor 0.025–0.051 nats）。

### 1.3 缺口

entity cell（L0–4 事實記憶）→ entity span 承載帶（L0–11）→ handoff（L12–15）→ dial（L15/N8490）→ decision 這條路徑中，以下問題未解：

1. L0–11 的 entity span 承載，哪些 token（ticker vs. company name）和哪幾層是充分的？
2. L12–15 handoff 是靠 MLP block 還是 attention block 完成的（Phase 2C 用 first-order attribution 無法回答，Phase 3 已排除 single-neuron 版本）？
3. Entity bias 是否流經 L15/N8490 dial 座標？Entity 改變決策的機制是「修改 dial 的激活值」，還是走 dial 旁邊另一條路？

### 1.4 方法與文獻邊界

- **方法譜系**：entity-state patching 的 patching 契約（bidirectional、FP32 tail-logit margin、self-source no-op 驗證）沿用 [activation-patching-causal-tracing 協議](../../activation-patching-causal-tracing/proposal.md#patching-protocol) 與 [balanced-evidence-gap Phase 2](../../balanced-evidence-gap/details/proposal-phase2.md)；dial 座標定義與 `mlp_addition` 推注語義（all-position、native units）來自 investment-dial 線（Park et al. 2026 方法復現）。
- **本協議新增的干預**：block-level activation patch（§4.3）在本 repo 無先例，語義由 §4.3 定義。

| 文獻／線上游原始設定 | 本協議適應性修改 | 邊界與風險 |
|---|---|---|
| causal tracing span patch：post-block residual 替換，span 級 | Phase A 把 span 細分為 token-group；hook 點與映射契約不變 | 無新假設；token 粒度為 tokenizer BPE 單位（見 §8） |
| causal tracing span patch：post-block residual 替換 | Phase B 改為 block 貢獻替換（mid 與 post 兩點干預） | 一階向量替換，非 circuit-level 分離；L12–14 的 attention block 是 Gated DeltaNet 線性注意力子層，不只 full attention |
| investment-dial `mlp_addition`：all-position、δ 以 s 倍數給定 | Phase C 改用 measured δ = a_named − a_anon（per company） | all-position 語義繼承（[Phase 3 report §限制](../../balanced-evidence-gap/details/report-phase3.md)同記）；dial 讀取點（entity span 最後 token）是近似（§8） |
| — | — | 結論限於 16 家公司 population 與 Qwen3.5-4B；不主張外推（§8） |

---

## 2. 定義

本協議新增以下術語。未覆蓋的定義沿用 [balanced-evidence-gap proposal-phase2 §2](../../balanced-evidence-gap/details/proposal-phase2.md#2-定義本协议新增術語)。

- **token-group patch**：在殘差流 resample patching 中，把 patch 範圍縮小到 entity span 內部的特定 token 子集（ticker tokens 或 company-name tokens），而非整個 entity span。patch 語義與 2B 的 entity-state patching 相同（source 公司的殘差狀態替換 target 公司的對應位置）。
- **block-level activation patch**：把某一層 MLP block 或 attention block 的整體貢獻（block 對殘差流的向量貢獻）從 source 公司替換進 target 公司的 residual stream，位置限定在 entity span。精確語義見 §4.3。
- **block state（pre / mid / post）**：層 l 的三個殘差流狀態：`pre` = block 輸入（前一 block 輸出）；`mid` = post-attention residual（`post_attention_layernorm` 的輸入）；`post` = block 輸出（residual stream 在該層之後的狀態）。
- **block contribution**：MLP block 貢獻 = `post − mid`（逐 token position 的向量差）；attention block 貢獻 = `mid − pre`。
- **mid-residual hook**：掛在層 `post_attention_layernorm` 輸入端的 hook，可捕獲（capture）或替換（intervention）mid 狀態。本協議在 shared core 新增（§5）。
- **ticker-group**：entity span 中屬於 ticker symbol 的所有 token（例如 `NSC` 的 token 集合；token 粒度為 tokenizer 單位，方括號 token 若與 symbol 合併則一併計入）。
- **name-group**：entity span 中屬於 company name 的所有 token（例如 `Norfolk`、`Southern` 的 token 集合）。
- **dial activation**：L15/N8490 MLP down-projection 輸入通道（intermediate activation，native units）在特定 prompt 的特定 token position 的激活值；與 investment-dial 線的定義一致（`mlp_coordinates` 讀取點）。
- **dial-transfer experiment**：把 named entity 在 L15/N8490 的 dial activation 讀取後，對匿名 prompt（entity 換成 `[Company X]`）施加 `mlp_addition(layer=15, neuron=8490, delta=δ)`，使其 dial channel 激活值對齊 named entity 的值，再量測 decision margin 是否向 named entity 的 margin 移動。
- **anonymous prompt**：把 named prompt 的 entity header 兩行方括號內容替換為字面 placeholder `Stock Ticker: [TICKER]` 與 `Stock Name: [Company X]`、其餘文字 byte-identical 的 cross-entity probe prompt。header 行排版（兩行之間的空行）與 named prompt 相同（Rev 0 §4.1 的單換行記法為排版疏漏，Rev 1 定稿為排版繼承）。
- **group gap**：top 群組（NSC、BLK）pure entity margin 的最小值，減去 bottom 群組（IT、BDX）pure entity margin 的最大值。
- **fp32 block arithmetic**：block-level patch 的貢獻差（`post − mid`、`mid − pre`）與替換和（`x_mid + Δ`、`x_pre + Δ`）一律先在 FP32 計算、再 cast 回 model dtype（bf16）。pipeline 的 self no-op 檢查用直接複製（clone/assign、bit-exact by construction）；該算術的精確性由 unit test 另行驗證（§4.3、§5、§12）。

---

## 3. 假說

- **H1（token 充分性）**：在 L0–11 的 entity span patching 中，ticker-group 單獨 patch 在 L0–5 就已充分（toward-source ΔM 的 95% CI 排除 0），company-name-group 的增量貢獻在 L6–11 才顯現。
- **H2（block 分工）**：L12–15 中，MLP block-level patch 的 toward-source ΔM 顯著大於 attention block-level patch（entity 訊號主要透過 MLP block 完成 handoff）；或兩者均顯著（並行路徑）。
- **H3（dial 路徑）**：entity A 的 dial activation 推注進匿名 prompt 後，decision margin 移動量的 95% CI 方向與 named-vs-anonymous gap 同號，且解釋比例 ≥ 0.25。若 CI 不含 gap，表示 entity bias 還走了 dial 以外的路徑。

H1、H2 各自有 formal gate；H3 為研究線的核心因果問題，gate 設計見 §4.4。

---

## 4. 實驗設計

### 4.1 公司與 prompt 族

沿用 balanced-evidence-gap Phase 2A 的 16 家公司（test split，4 sector × 4 家）和 frozen shared-evidence template。Phase 2A 的 pure entity margin 分組維持：bottom = {IT, BDX}，top = {NSC, BLK}（frozen 於 `phase2a-rev2-gate-01`）。8 個 directions 直接讀取 `phase2b-gpu-bf16-01` 的 frozen pairs（NSC↔IT、NSC↔BDX、BLK↔IT、BLK↔BDX，雙向），prepare 階段核對集合相等，不等則 fail-closed。

匿名 prompt：named prompt 文字做 byte-level 單一替換——header 的 `[{ticker}]` 換成 `[TICKER]`、`[{name}]` 換成 `[Company X]`，其餘（evidence、instruction、行排版）不變；替換必須恰好命中一次，否則 fail-closed。每家公司對應一條匿名 prompt（canonical variant：reverse=False, order=0）。共 **16 條 named + 16 條 anonymous = 32 條 clean prompt**，供 Phase C 的 dial activation 讀取。

**Smoke pre-check（Phase A 和 Phase B 執行前必須通過）**：

1. **Pre-check 1（group gap，data check）**：group gap ≥ 0.5 nats。margin 定義為 2A 各公司 4 variants 的中位數，prepare 階段從 2A `forward/results.jsonl` 重算，並與 2A `analyze/summary.json` 的 `pure_entity_margin_median` 交叉核對（不一致 fail-closed）。2A 存檔實測（2026-09-11 驗證）：group gap = 1.028 nats。
2. **Pre-check 2（self-source no-op，forward check）**：self-source patch（source = target）的 |ΔM| ≤ 1e-12（clone/assign 語義下為 bit-exact，Rev 0 的 1e-4 門檻被此收斂）。於 smoke 與每個 formal run 的 forward stage 執行（§4.2/§4.3 的 no-op 紀律）。

### 4.2 Phase A：Token-group × Layer 充分性地圖

**問題**：ticker-group 和 name-group 各在哪幾層成為充分的？

**token-group 切分**（`spans.py`，在 2A 存檔的 prompt 文字上直接計算，不引入新 tokenization 假設）：

- ticker-group = ticker 行方括號內 symbol 字元區間對應的 token 集合（core `token_span` 的 intersect 語義）；name-group = name 行方括號內 symbol 字元區間對應的 token 集合。
- 不變式（prepare 階段 fail-closed 檢查）：兩組皆為非空；`ticker-group ∪ name-group ⊆ entity_span`（2A 存檔值）；`ticker-group ∩ name-group = ∅`。
- token 粒度為 tokenizer BPE 單位：若方括號與 symbol 合併成同一 token，該 token 整體計入對應 group（patch 以 token 為原子單位）。

**操作**：

- Patch 範圍：entity span 拆成 ticker-group 和 name-group 兩個子集，分別 patch（不合併 patch）。
- Span 映射：source group span → target group span 用 2B 的 nearest-position 映射契約（等長退化为 identity；name-group 不等長時用 nearest 映射）。
- Layers：L0–11（12 層）。
- Directions：8 個（2B frozen pairs）。
- 每個 (layer, token-group, direction) 跑一次 patch forward，量 toward-source ΔM 與 normalized transfer T。
- 整個 entity span patch（ticker + name 合併）作為 upper-bound reference，直接複用 2B 的存檔 `sweep/records.jsonl`（entity span 行，不重跑），per-layer mean T 嵌入 Phase A analyze summary（附 2B records 檔 SHA-256 provenance）。

**Forward 協議（per direction，8 個 direction 合計）**：

1. Source 狀態 forward：`record_residuals(source, L0–11)`（1 次）。
2. Target clean forward：`record_residuals(target, L0–11 + final layer)`（1 次），final-layer residual 走 FP32 tail 得 live target margin。
3. Group patch forwards：12 layers × 2 token-groups = 24 次（post-block transform，clone/assign 語義，位置限定 group span）。
4. Self no-op forwards：12 layers × 1（ticker+name 合併的 full-entity self-patch）= 12 次；|ΔM − live target margin| 必須 ≤ 1e-12，否則中止（fail-closed，2B 紀律）。

共 **8 × (2 + 24 + 12) = 304 次 forwards**。

**Gate A**（pre-registered，在 Phase A analyze stage 評估）：

| 判準 | 門檻 | 說明 |
|---|---|---|
| A1：ticker-group 在 L0–5 的 toward-source ΔM | 8 directions 的 per-direction mean（L0–5 平均）bootstrap 95% CI（n=2000, seed=42）排除 0 | ticker token 在早期層即為充分 |

Gate A pass = A1 通過。若 A1 fail，Phase B/C 仍繼續（Phase A gate 不 block 後續），但報告中標記 A1 未過。name-group 在 L0–5 的 CI 是否排除 0 為描述性（不 gate）。

### 4.3 Phase B：Handoff 區間 Block-level Activation Patch

**問題**：L12–15 的 entity signal 由 MLP block 還是 attention block 傳遞？

**Hook 點與干預語義**（Qwen3.5 decoder layer 的殘差流順序：`pre → input_layernorm → attention → +residual → mid → post_attention_layernorm → MLP → +residual → post`；full-attention 與 Gated DeltaNet 兩類層同構）：

| 狀態 | 捕獲點 | 干預點 |
|---|---|---|
| pre | block pre-hook（forward 第一參數） | — |
| mid | `post_attention_layernorm` pre-hook（其輸入） | `post_attention_layernorm` pre-hook（替換其輸入） |
| post | block post-hook（block 輸出） | block post-hook（替換 block 輸出；複用 `residual_interventions`） |

- **MLP block patch**（層 L）：`post[entity positions] ← mid_target[entity positions] + (post_source − mid_source)[mapped source positions]`，以 post-block transform 實施（closure 使用 target clean forward 捕獲的 mid）。
- **Attention block patch**（層 L）：`mid[entity positions] ← pre_target[entity positions] + (mid_source − pre_source)[mapped source positions]`，以 mid-residual pre-hook 替換實施（closure 使用 target clean forward 捕獲的 pre）。
- 位置限制：只 patch entity span 位置（整個 entity span 的所有 token，不拆 ticker/name）；span 映射用 2B 的 nearest-position 契約；非 entity 位置不變。
- 層內傳播：attention 計算在干預點（mid）之前已完成，使用的都是未干預的 pre 狀態，故 attention patch 只作用於 entity position 的 mid（同層其他位置不受影響）；MLP patch 同理只作用於 entity position 的 post。兩者皆向後續層傳播——與 2B post-block span patch 的 position-local 語義一致。
- **fp32 block arithmetic**：`(post − mid)`、`(mid − pre)` 與替換和先在 FP32 計算、再 cast 回 model dtype（bf16）。精確性性質：結果在 FP32 下與原狀態精確相等；cast 回 bf16 後與原值相差至多 1 ulp（mid 與 post 每 channel 相差小於 2 倍的區間由 Sterbenz 引理保證 bit 相同）。pipeline 的 self no-op 檢查不依賴此算術，而用直接複製（clone/assign、bit-exact by construction，與 2B 語義一致）；算術精確性由 unit test 驗證（§5 Tests）。
- 狀態捕獲的正確性前提：單一 (layer, component) patch forward 中，層 L 之前的 block 未受干預，故 L 處的 pre/mid 與 clean forward 相同；此前提由 self no-op 檢查（|ΔM| ≤ 1e-12）封閉驗證。

**Forward 協議（per direction，8 個 direction 合計）**：

1. Source 狀態 forward：`record_block_states(source, L12–15)`（1 次）。
2. Target clean forward：`record_block_states(target, L12–15 + final layer)`（1 次），final-layer post 走 FP32 tail 得 live target margin。
3. Patch forwards：4 layers × 2 components = 8 次。
4. Self no-op forwards：4 layers × 2 components = 8 次；|ΔM − live target margin| 必須 ≤ 1e-12，否則中止。

共 **8 × (2 + 8 + 8) = 144 次 forwards**。Source/target 狀態捕獲為 transient（GPU 記憶體），不跨 run 持久化（§8 raw-data 禁令）。

**Gate B**（pre-registered，在 Phase B analyze stage 評估）：

| 判準 | 門檻 | 說明 |
|---|---|---|
| B1：L12–15 中至少 1 層的 MLP block patch toward-source ΔM 的 bootstrap 95% CI（8 directions, n=2000, seed=42）排除 0 | — | MLP 路徑存在 |
| B3：B1 通過的最強層（qualifying layers 中 |mean ΔM| 最大者），4 sector 的 mean ΔM 同號 | 4/4 | MLP 效應跨 sector 一致 |

Gate B pass = B1 + B3 通過。B3 的 sector mean 定義：4 家公司各屬一個 sector；某 sector 的 mean ΔM = 該公司作為 source 或 target 參與的 4 個 direction 的 toward-source ΔM 平均；4 個 sector mean 必須嚴格同號（mean = 0 視為非一致）。B1 通過層的 MLP block ΔM 與 attention block ΔM 的大小比較為描述性（不 gate）。若 Gate B fail，記為 null result，Phase C 仍執行。

### 4.4 Phase C：Dial 路徑干涉實驗

**問題**：entity bias 是否透過 L15/N8490 dial 座標傳遞？

**Forward 協議（單一 forward 同時量測）**：每條 clean prompt 的 forward 同時取得 (a) FP32 tail decision margin 與 (b) L15/N8490 down-projection 輸入通道在兩個位置的 dial activation——`entity_position`（entity span 最後 token，2A 慣例）與 `final_position`（formatted prompt 最後 token，2A 慣例）。package-local hook 一次 forward 記錄兩位置（`capture_mlp_channel` 模式的雙位置擴充），避免每位置一次 forward。

**Step C1（觀測 dial activation 的 entity 特異性，descriptive）**：

對 16 條 named prompt 和 16 條 anonymous prompt（32 條，canonical variant），forward 後計算 named vs. anonymous 的 dial activation 差（`dial_delta = named_dial − anon_dial`，entity position 為準），對 16 家公司做 Spearman ρ（`dial_delta` vs. pure entity margin from 2A median）。final position 的 `dial_delta` ρ 為第二描述性指標。Step C1 為描述性，不 gate；若 entity-position ρ < 0.3，在 Step C2 的 analyze 中記錄 warning。

**Step C2（Dial 推注實驗）**：

- 對象：16 條 anonymous prompt（entity = `[TICKER]` / `[Company X]`）。
- 干預：對每條 anonymous prompt，計算 `δ = a_named − a_anon`（entity position 的 dial activation），用 `mlp_addition(layer=15, neuron=8490, delta=δ)` 推注。`mlp_addition` 為 **all-position 語義**（與 investment-dial 線與 Phase 3 慣例一致）；entity-position-only 推注不受 core 支援，列為本線限制（§8）。
- δ = 0 的 company 不跳過：照常跑 mlp_addition(δ=0) forward（數值精確 no-op），維持 records 形狀一致。
- 量 decision margin，與 anonymous 的 clean margin 比較（`ΔM_dial = M_pushed_anon − M_clean_anon`）。
- 同時量 named 的 clean margin（`M_named`，本 run 重測），計算 `gap = M_named − M_clean_anon`（per company）。
- **2A cross-check（provenance hygiene，非 gate）**：16 條 named canonical prompt 的本 run margin 與 2A 存檔 margin 逐公司比較；max |Δ| > 0.1 nats（約 2× Phase 3 實測 bf16 jitter 帶 0.05 nats）時記錄 warning 於 analyze metadata（不 fail-closed，兩 run 各自 bf16，jitter 預期內）；gate 判定一律以本 run 值為準。

**Gate C**（pre-registered）：

| 判準 | 門檻 | 說明 |
|---|---|---|
| C1：dial 推注後 mean ΔM_dial 的 bootstrap 95% CI（16 companies, n=2000, seed=42） | CI 兩端與 mean gap 同號（即 CI 排除 0 且位於 mean gap 的同一側） | dial 推注方向正確 |
| C2：mean \|ΔM_dial\| / mean \|gap\|（16 companies 算術平均） | ≥ 0.25 | dial 路徑至少解釋 25% 的 entity gap |

（C1 的 CI 端點同號判讀為 Rev 1 對 Rev 0「CI 方向與 gap 同號」的明文化；gate 門檻 0.25 與 Rev 0 相同。）

Gate C pass = C1 + C2 通過，表示「entity bias 有一部分流經 dial 座標」。Gate C fail 表示「entity bias 主要走 dial 以外的路徑」，兩種結果都是有效的科學發現。mean gap = 0 時 C1 fail-closed（符號未定義）。

**Step C3（unexplained gap，descriptive）**：  
`gap − ΔM_dial`（per company 與 16 公司平均）為 dial 路徑無法解釋的 entity gap 剩餘（unexplained gap；artifact 欄位名不用 `residual_gap`，因 core serializer 將 `residual` 列為 raw-state 保留字），報告其大小和方向，供後續研究線使用。

**Forward 合計**：32（clean）+ 16（pushed）+ 1（δ=0 no-op 驗證，任取 1 條 anonymous prompt）= **49 次 forwards**。

---

## 5. 實現與 Artifact

**Package**：`llm_bias/entity_to_dial/`（新實驗 package，不 import 其他 experiment package；shared mechanics 走 `llm_bias/core/`）。

子模組規劃：

- `template.py`：常數（dataset slug `entity-to-dial`、schema version `entity-to-dial-v1`、dial 座標 L15/N8490、anonymous header 字面文字、pre-check 門檻、smoke grid）。
- `spans.py`：token-group 定義（ticker-group、name-group）的 character→token 切分（在 2A 存檔 prompt 文字上計算）；anonymous prompt 的 byte-level 構造（單一命中斷言）。
- `block_patch.py`：MLP/attention block-level patch 的 transform 構造（貢獻差、fp32 arithmetic、entity-position 限制、self no-op）；research 語義層，hook lifecycle 用 core。
- `dial_probe.py`：單一 forward 的 margin + 雙位置 dial activation 讀取；`mlp_addition` 推注包裝。
- `analysis.py`：Gate A/B/C 的統計計算（toward-source ΔM、normalized transfer、bootstrap CI、Spearman ρ、sector agreement）；gate 函式與 2B `analysis.py` 同構（純函式、fake-model test 覆蓋）。
- `pipeline.py`：`run_phase_a` / `run_phase_b` / `run_phase_c`，各走 `prepare → forward → analyze` + `ArtifactRun.finalize`。

**Shared core 新增**（`llm_bias/core/inference/interventions.py`，additive，不改既有 API）：

- `record_block_states(model, input_ids, layers)` → `{layer: {"pre": T, "mid": T, "post": T}}`（transient，no_grad；pre = block pre-hook 輸入，mid = `post_attention_layernorm` 輸入，post = block 輸出；缺 `post_attention_layernorm` fail-closed）。
- `mid_residual_interventions(model, transforms)`：mid 點的 `residual_interventions` 同構 context manager（transform 契約、shape 檢查、exception-safe hook 移除）。

歸入 core 的依據：forward-execution hook mechanics 屬 shared mechanics（與 `residual_interventions`、`record_residuals` 同類）；`core/` 現無 mid-layer 捕獲先例；未來的 block-level 因果線（activation-patching 線延伸）可複用同一 primitive。core 層只含 hook lifecycle，無研究語義。

**Operators**（`scripts/`，follow balanced-evidence-gap 線的 operator 慣例；不新增 `pyproject.toml` entry point）：

- `scripts/entity_to_dial_phase_a.py`（Phase A token-group patch sweep）
- `scripts/entity_to_dial_phase_b.py`（Phase B block-level patch）
- `scripts/entity_to_dial_phase_c.py`（Phase C dial 實驗）

CLI 契約（1:1 綁定；正式 run 與 smoke 各一條命令，不混裝跨階段參數）：

```bash
# Phase A（smoke / formal 同參數，--smoke 開關）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/entity_to_dial_phase_a.py \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \
    --run-id entity-to-dial-a-01            # smoke：加 --smoke（run-id 預設 entity-to-dial-a-smoke-<timestamp>）

# Phase B（smoke / formal 同形）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/entity_to_dial_phase_b.py \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --phase2b-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01 \
    --run-id entity-to-dial-b-01

# Phase C（只需要 Phase 2A run）
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/entity_to_dial_phase_c.py \
    --model .cache/models/qwen3.5-4b \
    --phase2a-run artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01 \
    --run-id entity-to-dial-c-01
```

授權前置檢查（operator 層級，fail-closed）：formal run 執行前，operator 重新驗證 pre-check 1（group gap，data check）與 upstream provenance；pre-check 2 由 forward stage 的 self no-op 紀律內建執行。Phase B/C 的 formal run 不要求 Phase A 的 gate 通過（§7），但要求對應 phase 的 smoke 曾通過（§14）。

**Run root**：`artifacts/qwen3.5-4b/entity-to-dial/runs/<RUN_ID>/`（dataset slug `entity-to-dial`）；每個 Phase 獨立 run ID（`entity-to-dial-a-<seq>`、`entity-to-dial-b-<seq>`、`entity-to-dial-c-<seq>`）；smoke run ID 為 `entity-to-dial-<phase>-smoke-<UTC timestamp>`（2B smoke 慣例）。

**Stages（每個 Phase）**：`prepare` → `forward` → `analyze` + finalize（`ArtifactRun`）。

- `prepare`：provenance（§10 upstream 檔的 SHA-256 + record 數 + upstream manifest `status=complete` 核對；frozen groups 與 2B directions 集合核對；group gap 計算與記錄）+ `rows.jsonl`（named 16 行：prompt/formatted/prompt_ids/entity_span/ticker_span/name_span/entity_position/final_position/pure_entity_margin；Phase C 另加 anonymous 16 行）。token-group 不變式檢查（§4.2）。
- `forward`：依 §4.2–§4.4 的 forward 協議；所有 margin 經 FP32 tail；self no-op 檢查 fail-closed；transient 狀態不落地。
- `analyze`：curves + gate 判定（gate 結果只記錄、不作 abort 依據——gate fail 是有效結果）；Phase C 含 C1 描述性 ρ、warning 記錄與 2A cross-check。
- finalize：`run.finalize(required_stages={"prepare", "forward", "analyze"})`，manifest 標 `complete`。

**Artifact schema（compact，per phase）**：

- Phase A：
  - `prepare/rows.jsonl`：上述 16 行 + `prepare/provenance.json`（§10）。
  - `forward/records.jsonl`：`{phase: "a", direction, layer, token_group ∈ {ticker, name, self_noop}, patched_margin, toward_source_delta_m, normalized_transfer, m_source, m_target, live_target_margin}`；`self_noop` 行的 `toward_source_delta_m` 記錄實測值（預期 0）。
  - `analyze/summary.json`：per (token_group, layer) 的 `mean_toward_source_delta_m`、`mean_normalized_transfer`、`delta_m_ci_95`、`n_directions`；`gate_a`（A1 pass/fail + ticker L0–5 mean/CI）；`upper_bound_reference`（2B entity-span per-layer mean T，L0–11）；`descriptive`（name-group L0–5 CI）；`smoke` 布林。
- Phase B：
  - `forward/records.jsonl`：`{phase: "b", direction, layer, component ∈ {mlp, attn, self_noop}, patched_margin, toward_source_delta_m, normalized_transfer, m_source, m_target, live_target_margin}`。
  - `analyze/summary.json`：per (component, layer) curves + CI；`gate_b`（B1 qualifying layers、strongest layer、B3 per-sector means + pass）；`descriptive`（attn vs. mlp 大小比較）；`smoke` 布林。
- Phase C：
  - `forward/results.jsonl`：clean 行 `{ticker, prompt_type ∈ {named, anon}, prompt, formatted, prompt_ids, entity_span, entity_position, final_position, margin, dial_entity, dial_final}`；push 行 `{ticker, delta, pushed_margin, clean_margin_anon, delta_m_dial}`。
  - `analyze/summary.json`：`c1_descriptive`（entity/final position ρ、warning）；per-company `gap`、`delta_m_dial`、`unexplained_gap`；`gate_c`（C1 mean/CI/pass、C2 ratio/pass）；`c3_unexplained_gap`；`cross_check_2a`（per-company |Δ|、max、warning）；`smoke` 布林。

**持久化**：只存 compact 派生值（margin、ΔM、T、CI、ρ、dial activation 標量、provenance）；不存 raw activations、residuals 或 hidden states（core serializer 拒收 tensor payload）。所有數值欄位必須 finite（serializer 拒收 non-finite）。

**Tests**（`tests/`，fake model / mocked forward / temporary directories，不載入大型 checkpoint；follow `tests/test_balanced_evidence_gap_phase2.py` 與 `tests/test_mlp_addition.py` 的 fake-model 慣例）：

- `tests/test_entity_to_dial_spans.py`：token-group 切分（fake tokenizer + offset_mapping）：不變式（非空、disjoint、⊆ entity span）、anonymous prompt 的 byte-level 構造（單一命中斷言、排版繼承）、fail-closed 路徑（缺行、空 group）。
- `tests/test_entity_to_dial_block_patch.py`：小型真實 `Qwen3_5ForCausalLM`（2–4 層、mixed layer_types）：block state 關係（`post − mid` 等於 `mlp(post_attention_layernorm(mid))` 的 allclose 驗證）、self no-op bit-exact（兩 component、直接複製語義）、block 算術精確性（純函數 fp32 精確相等；bf16 cast 後 1 ulp 內）、patch 只改 entity positions、hook lifecycle（exception 後移除）。
- `tests/test_entity_to_dial_dial_probe.py`：dial 讀取 = down-proj 輸入通道值；`mlp_addition(δ)` 後同一通道值精確 +δ（all positions）；單一 forward 同時取得 margin 與雙位置 dial。
- `tests/test_entity_to_dial_analysis.py`：Gate A1/B1/B3/C1/C2 邏輯與 edge cases（CI 跨 0、ratio 邊界 0.25、mean gap = 0、缺 direction、per-sector mean 含 0）。
- `tests/test_entity_to_dial_pipeline.py`：monkeypatched `load_model`/`load_tokenizer` + fake model 的端到端 smoke 路徑（temp dir）：manifest `complete`、no-op enforcement（違反時 fail-closed）、artifact schema 欄位齊全。
- core 端擴充 `tests/test_core_inference.py`：`record_block_states`（三狀態形狀/內容、tuple block output、缺 module fail-closed）與 `mid_residual_interventions`（transform 施加、shape 檢查、exception-safe 移除）。

Gate 邏輯必須通過 fake-model test、smoke 通過後，才授權正式 run（§14）。

---

## 6. 成本估算

| Phase | Forwards | 明細（per direction / 合計） | 估算（4B bf16，1× GPU） |
|---|---|---|---|
| A（token-group patch） | 304 | 8 × (2 capture + 24 patch + 12 no-op) | ~8–10 min |
| B（block-level patch） | 144 | 8 × (2 capture + 8 patch + 8 no-op) | ~5 min |
| C（dial 觀測 + 推注） | 49 | 32 clean + 16 pushed + 1 no-op | ~3 min |
| **合計** | **497** | 三個獨立 run（各自載入一次模型） | **~20 min** |

（Rev 0 的 304 forwards 估算未含 no-op 紀律與狀態捕獲；Rev 1 以實際 forward 協議計數。）

---

## 7. Gate 授權鏈

```
smoke + pre-check（§14）
       ↓
   Phase A  ──────────────────────┐
   (Gate A，不 block 後續)         │
       ↓                          │
   Phase B  ←─────────────────────┘
   (Gate B，不 block Phase C)
       ↓
   Phase C
   (Gate C)
```

所有 Gate 判準和門檻在本文件 frozen 後不得修改；若需調整，建新版本協議（不回填）。Phase A smoke 必須通過（§4.1 pre-check 驗收），才授權 Phase B/C 的正式 run。Phase B/C 不以 Phase A gate 通過作為授權條件，因為三個問題獨立且 gate fail 本身有科學價值。Gate 判定只記錄於 analyze summary，不作為 run 的 abort 條件（gate fail = 有效結果，run 仍 finalize 為 complete）。

---

## 8. 邊界與非目標

- 本線不主張找到 entity bias 的「唯一路徑」。三個 Phase 分別確認某類路徑的存在性，不排除並行路徑。
- Phase A 的 token-group 切分基於 tokenizer 的 BPE 單位（方括號 token 若與 symbol 合併則整體計入該 group），不引入新的 tokenization 假設。
- Phase B 只覆蓋 L12–15（2B handoff 區間）；L16 以後的 block 貢獻不在本線範圍。L12–14 的 attention block 是 Gated DeltaNet 線性注意力子層；「attention block」在本協議指該子層對殘差流的貢獻（含其遞迴狀態的層內效果），非僅 full-attention 層。
- Phase B 的 block-level patch 是一階向量替換（把 source 的 block 貢獻搬進 target 的流），不是 circuit-level 分離；它回答「哪個 block 的貢獻搬運後足以轉移 margin」，不排除 block 內部的更細結構。
- Phase C 的 dial 推注用 entity span 最後一個 token position 的 dial activation 作為 named entity 的代表值。若 entity signal 分佈在多個 token position，此讀取點是近似，報告中標記此限制。
- Phase C 的 `mlp_addition` 為 all-position 語義（與 investment-dial 線與 Phase 3 一致）；entity-position-only 推注是未排除的替代設計（[Phase 3 report §限制](../../balanced-evidence-gap/details/report-phase3.md)同記），若 Gate C 邊界性 fail，優先考慮此替代（建新版本）。
- margin 量測下限：bf16 forward 的 jitter 帶約 0.05 nats（Phase 3 實測）；FP32 tail 與 fp32 block arithmetic 消除 instrument 端的系統性誤差，但不消除 forward 端的 bf16 jitter。任何 |ΔM| < 0.05 nats 的效應解釋必須標記此下限。
- 16 家公司（2A population）；不主張結果外推至 427 家宇宙（investment-dial dataset）。
- 模型與語言單一（Qwen3.5-4B、英文 prompt）；不做跨模型推論。
- Gate C pass 不等於「dial 是唯一路徑」；它只確認「dial 路徑對 entity gap 有可偵測的貢獻」。
- 不持久化任何 transient 狀態（residual、block state、dial 向量）；不跨 run 复用狀態（每個 phase 在自己的 forward stage 重新捕獲 source/target 狀態）。

---

## 9. 預期知識收益

| Phase | Gate pass | Gate fail |
|---|---|---|
| A | Ticker token 在 L0–5 是充分的 entity signal 載體；company name 為補充 | Entity signal 需要 ticker + name 合力，或需要更深的層積累 |
| B | MLP block 在 L12–15 承擔 handoff；量化各層貢獻比例 | Entity handoff 不靠單一 MLP block，可能是分散式的；attention block ΔM 提供對照 |
| C | Entity bias 部分流經 dial 座標；`ΔM_dial / gap` 量化比例 | Entity bias 繞過 dial；需在 L12–15 block 殘差中尋找另一條路 |

Phase B fail + Phase C fail 的組合意味著 entity bias 的傳遞不在 MLP block 粒度上可以因果抓到，需要轉向更粗（整層 residual）或更細（circuit-level）的方法。

---

## 10. 上游 Run 依賴

| 上游 Run | 路徑 | 用途 | 驗證方式 |
|---|---|---|---|
| `phase2a-gpu-bf16-01` | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01` | 16 家公司 canonical prompt（`prepare/prompts.jsonl`）、pure entity margin 與 dial 存檔值（`forward/results.jsonl`）、median 交叉核對（`analyze/summary.json`） | prepare 階段 SHA-256 比對 + record 數核對 + manifest `status=complete` |
| `phase2a-rev2-gate-01` | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-rev2-gate-01` | top/bottom 群組凍結（NSC/BLK vs IT/BDX）的 provenance（`analyze/summary.json`） | prepare 階段 SHA-256 比對 + manifest `status=complete` |
| `phase2b-gpu-bf16-01` | `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01` | frozen 8 directions（`pairs/directions.json`）；Phase A upper-bound reference（`sweep/records.jsonl` entity 行，descriptive，不重跑） | prepare 階段 SHA-256 比對 + directions 集合相等核對 + manifest `status=complete` |

每個 phase 的 consumed 檔案：Phase A/B = 上表全部；Phase C = 僅 `phase2a-gpu-bf16-01`。上游 run 路徑以 `--phase2a-run`、`--phase2b-run` 參數傳入各 operator，prepare stage 驗證 SHA-256 並 fail-closed（任何一項不符則不進入 forward stage，run manifest 標 `failed`）。

---

## 11. Input/Output 契約

### 11.1 Input

| 來源 | 檔案 | 預期欄位／型別 | 用途 |
|---|---|---|---|
| `phase2a-gpu-bf16-01` | `prepare/prompts.jsonl`（64 行） | `id, ticker, name, sector, reverse, order, prompt (str), formatted (str), prompt_ids (list[int]), entity_span (list[int]×2), evidence_span, instruction_span, final_position (int), entity_position (int)`；取 `reverse=False, order=0` 的 16 行 | prompt 文字、entity span、dial 讀取位置 |
| `phase2a-gpu-bf16-01` | `forward/results.jsonl`（64 行） | 上述欄位 + `margin (float), decision (str), dial_channel_entity (float), dial_channel_final (float)` | pure entity margin（4-variant median）、2A cross-check 參照、dial 描述性參照 |
| `phase2a-gpu-bf16-01` | `analyze/summary.json` | `pure_entity_margin_median (dict[ticker, float])` | median 交叉核對（重算 vs. 存檔，不符 fail-closed） |
| `phase2a-rev2-gate-01` | `analyze/summary.json` | `gate_2a_rev2 (dict, pass=true)`、`descriptive.groups.top/bottom (list[ticker])` | frozen 群組 provenance |
| `phase2b-gpu-bf16-01` | `pairs/directions.json` | `directions (list[[source, target]]×8)` | 8 directions（集合相等核對） |
| `phase2b-gpu-bf16-01` | `sweep/records.jsonl`（1024 行） | `direction, layer, span, patched_margin, toward_source_delta_m, normalized_transfer` | Phase A upper-bound reference（entity 行） |

Tokenizer 條件：與 2A 相同（Qwen3.5-4B 本地 checkpoint 的 tokenizer；chat template 與 `DECISION_PREFIX = '{"decision": "'` 固定）。模型：`.cache/models/qwen3.5-4b`（bf16）；模型 identity 記錄於 manifest `model` 欄位（2A/2B 線慣例）。

### 11.2 Output

見 §5 的 artifact schema。補充規則：

- 所有數值欄位必須 finite float；serializer 拒收 non-finite、tensor、numpy array。
- 禁止保存未聚合的 raw tensor/activation/KV cache；dial activation 以 per-position 標量保存（2 位置 × per prompt）。
- 每個 run 必有 `manifest.json`（schema version 1，全部 output artifact 的 SHA-256 + record counts + stage lifecycle）。
- `smoke=true` 的 run 與 formal run 同 schema；gate 欄位照算，但 smoke run 的 gate 不作為授權依據。

### 11.3 CLI 契約

見 §5 的三條命令（每條命令對應一個 phase 的一個 operator，無跨階段條件分支）。

---

## 12. 邊界情況與防禦性行為

| # | 邊界情況 | 防禦行為 |
|---|---|---|
| 1 | upstream 檔案缺失、record 數不符、SHA-256 不符、upstream manifest 非 `complete` | prepare fail-closed（raise；run manifest 標 `failed`；不進入 forward） |
| 2 | 2B `directions.json` 的 8 directions 集合與 frozen pairs 不等 | prepare fail-closed |
| 3 | group gap（pre-check 1）< 0.5 nats | formal run 於 prepare 中止（Rev 0：abort）；smoke 同樣中止並報錯 |
| 4 | ticker/name token span 為空、缺失、或 ⊄ entity span、或兩組相交 | prepare fail-closed（per-row 錯誤訊息含 ticker 與 group 名） |
| 5 | anonymous prompt 的 header 替換命中數 ≠ 1 | prepare fail-closed |
| 6 | self no-op 檢查 |ΔM| > 1e-12（Phase A full-entity、Phase B 兩 component、Phase C δ=0） | forward stage 中止（fail-closed，run 標 `failed`） |
| 7 | margin / ΔM / T / dial 值 non-finite | 立即 raise（serializer 層另有拒收） |
| 8 | mean gap = 0（Phase C） | Gate C1 fail-closed（符號未定義），summary 記錄 `degenerate_gap: true` |
| 9 | bootstrap CI 不可算（units < 4） | 本協議 units 固定為 8（directions）或 16（companies），正常不可發生；若發生則 gate fail-closed 並記錄 |
| 10 | 2A cross-check max |Δ| > 0.1 nats | 記錄 warning（不 fail-closed），gate 一律以本 run 值為準 |
| 11 | C1 描述性 ρ < 0.3 | Step C2 analyze 記錄 warning（不 gate） |
| 12 | bf16 forward jitter | FP32 tail margin + fp32 block arithmetic 消除 instrument 系統誤差；|ΔM| < 0.05 nats 的效應在報告中標記 jitter 下限（§8） |
| 13 | forward 中途中斷（OOM、GPU 錯誤） | `ArtifactRun` 標 `failed`；transient 狀態隨 process 釋放（無部分持久化）；重跑用新 run ID |
| 14 | 層缺 `post_attention_layernorm`（模型結構異常） | core hook fail-closed（TypeError），不靜默跳過 |

數值容差匯總：self no-op ≤ 1e-12（bit-exact 預期）；2A cross-check warning 0.1 nats；group gap 0.5 nats；C2 ratio 0.25；bf16 jitter 標記下限 0.05 nats。

---

## 13. 版本分立觸發條件

以下任一要素變更時，必須建立新版本協議（`proposal-vN.md`）與對應 report，嚴禁原地修改本文件：

1. **Prompt 構造**：anonymous header 的 placeholder 文字或排版、named prompt family（evidence/instruction template）、company population 變更。
2. **Estimand / direction source**：toward-source ΔM 或 normalized transfer T 的定義、directions 來源（不再用 2B frozen pairs）、dial 座標（非 L15/N8490）。
3. **Gates / thresholds**：A1/B1/B3/C1/C2 的判定規則、CI 方法（bootstrap n、seed、units）、0.25 ratio、0.5 nats group gap、1e-12 no-op 容差。
4. **Controls / patch 語義**：token-group 切分規則、span 映射契約（nearest-position）、block 貢獻定義（pre/mid/post 的取點）、fp32 block arithmetic、all-position vs. entity-position-only 的 `mlp_addition` 語義。
5. **模型**：非 Qwen3.5-4B 或 dtype 變更（bf16 → 其他）。

---

## 14. 強制端到端 Preflight（smoke）

任何 formal run 前，對應 phase 的 `--smoke` 必須先以真實模型（`.cache/models/qwen3.5-4b`）與真實 tokenizer 通過。Smoke 為小型真實 run（非 mock），run ID 帶 timestamp，metadata 標 `smoke: true`；smoke run 不是正式 output（不作為任何報告的數據來源），但完整走 `prepare → forward → analyze → finalize` 以驗證全 pipeline。

| Phase | Smoke grid | 涵蓋 | Forwards |
|---|---|---|---|
| A | directions {NSC→IT, IT→NSC} × layers {0, 3, 5, 9} × groups {ticker, name} + 12 層中該 4 層的 full-entity no-op | token-group 切分、span 映射、residual 捕獲、post-block transform、no-op 紀律、pre-check 1（data）、pre-check 2（forward）、analyze/manifest | 2 × (2 + 8 + 4) = 28 |
| B | directions {NSC→IT, IT→NSC} × layers {12, 15} × components {mlp, attn} + 4 (layer, component) no-op | block state 捕獲（pre/mid/post）、mid-residual hook、post-block transform、fp32 arithmetic、no-op 檢查（直接複製） | 2 × (2 + 4 + 4) = 20 |
| C | tickers {NSC, IT}：4 clean + 2 pushed + 1 δ=0 no-op | 雙位置 dial 讀取、anonymous prompt forward、mlp_addition 推注、margin pipeline | 7 |

Smoke pass 條件：pipeline 無例外完成、manifest `complete`、全部 no-op 檢查通過、pre-check 1 通過（A/B）、所有輸出欄位 finite 且符合 §11.2 schema。Smoke fail 則不授權該 phase 的 formal run；修復後重跑 smoke。

**Smoke 狀態（2026-09-11，`.cache/models/qwen3.5-4b` GPU bf16）**：三段皆 pass——A `entity-to-dial-a-smoke-20260911T062930Z`（24 筆 forward records、no-op ΔM = 0.0）、B `entity-to-dial-b-smoke-20260911T063501Z`（16 筆、no-op ΔM = 0.0）、C `entity-to-dial-c-smoke-20260911T064217Z`（6 筆、δ=0 no-op ΔM = 0.0、2A cross-check max |Δ| = 0.0000 nats）。

---

## 15. Revision Record

| Rev | 日期 | 狀態 | 說明 |
|---|---|---|---|
| 0（Draft） | 2026-09-10 | proposed | 初稿：三段式設計（Phase A token-group、Phase B block-level、Phase C dial 路徑）。 |
| 1 | 2026-09-11 | proposed（實作設計定稿） | 實作設計定稿：(a) 對照現行程式驗證可行性——core 無 mid-layer hook 先例，定稿 `record_block_states` + `mid_residual_interventions` 為 additive core 新增（§5）；(b) Phase B 定稿 hook 點表、block 貢獻定義、fp32 block arithmetic（self no-op bit-exact）、per-direction forward 協議（§4.3）；(c) Phase A/C 定稿 forward 協議與 no-op 紀律，成本重計為 497 forwards（§4.2/§4.4/§6）；(d) 明文化 Rev 0 含糊處：Gate C1 的 CI 端點同號判讀（§4.4）、Gate B3 的 per-sector mean 定義（§4.3）、anonymous prompt 排版繼承 named header（§2/§4.1）；(e) pre-check 1 以 2A 存檔驗證 group gap = 1.028 ≥ 0.5（§4.1）；(f) 補 docs/AGENTS.md 必備章節：Input/Output 契約（§11）、邊界情況與防禦性行為（§12）、版本分立觸發（§13）、強制 preflight（§14）；(g) 確認三個 upstream run 皆存在且 manifest complete（§10）。Gate 門檻（0.25、0.5 nats、CI 方法）與 Rev 0 相同。 |
| 1.1 | 2026-09-11 | proposed（實作中精化） | 實作期數值精化：fp32 block arithmetic 的 self-source patch 在 non-Sterbenz 區間下 cast 回 bf16 可能差 1 ulp（非 bit-exact），故 Phase B 的 self no-op 檢查改用直接複製（clone/assign、bit-exact by construction，與 2B 語義一致）；fp32 算術精確性改為 unit test 驗證（fp32 精確相等、bf16 1-ulp 界）。無 gate、direction source、primary outcome 或 control 變更。 |
| 1.2 | 2026-09-11 | proposed（實作完成，smoke 通過） | 實作完成：core 新增 `record_block_states`/`mid_residual_interventions`（additive）、`llm_bias.entity_to_dial` package、3 個 operators、fake-model tests（spans/block_patch/dial_probe/analysis/pipeline）。Phase C Step C3 輸出欄位由 `residual_gap` 改名為 `unexplained_gap`（core serializer 將 `residual` 列為 raw-state 保留字）。三段 smoke 於真實模型通過（§14）。Formal run 仍未授權。 |
| 1.3 | 2026-09-11 | discovery completed | 三段 formal run 完成（`entity-to-dial-{a,b,c}-01`，2026-09-11）：Gate A1 fail（ticker-group L0–5 CI 跨 0；name-group 為顯著承載者，T ≈ 0.7）、Gate B fail（L12–15 兩 block 皆 null）、Gate C fail（dial 解釋 gap 約 5%，C2 ratio 0.051）。全部 no-op 檢查通過（ΔM = 0.0）、C 的 2A cross-check max abs(Δ) = 5.96e-08 nats。結果與解讀見 [report.md](report-phase-abc.md)。 |