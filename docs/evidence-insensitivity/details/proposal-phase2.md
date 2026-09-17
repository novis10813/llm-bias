# Evidence-insensitivity Phase 2：L15/L* 組間狀態對比（protocol proposal）

**狀態**：frozen（2026-09-16 用戶批准；run ID `phase2-gpu-bf16-01`，雙模型）
**對象模型**：Qwen3.5-4B（bf16，32 層，hidden 2560）＋ Gemma-4-E2B-it（bf16，
42 層，hidden 2560，MatFormer E2B）——雙模型統一執行
**研究線定位**：機制相第一階段（描述性）。在 Phase 1 的 frozen 行為分組上，
量測每家公司在證據極性條件下 L 層 instruction span 的狀態結構，找證據
不靈敏度的狀態層簽章。本 phase 不介入、不建立 causal claim（因果定位在
Phase 3）。

---

## 1. 核心假說與文獻邊界

**研究問題**：evidence-insensitive 公司與 evidence-responsive 公司在
decision-formation 層（Qwen L15；Gemma 依 Step A 定位的 L*）的
instruction-span 狀態上有什麼差異？差異是 offset（基線 stance 偏移，閾值
故事）還是 gain（狀態對證據極性的敏感度，承載衰減故事）？

**上游依據**：

- [Phase 1 報告](report-phase1.md)：Qwen 50 responsive / 453 fixed-sell；
  503/503 零證據 sell；503/503 家 C_c > 0（margin 全人口跟隨極性）；
  order-swap recency 效應（本 phase 不處理，已於 Phase 1 描述）。
- [entity-concept-decision](../../entity-concept-decision/report.md)：L15
  stance 1D 方向解釋 margin 變異 60%；L15 狀態 99.5% 跨公司共享、公司間差
  0.5% norm；決策是微小公司間差的高增益非線性讀出。
- [entity-to-dial](../../entity-to-dial/report.md)：決策形成定位 L15
  instruction span；L15 k=8 殘差子空間承載 entity swap 98.3%。
- [balanced-evidence-gap Phase 2B](../../balanced-evidence-gap/details/report-phase2.md)：
  entity span L0–11 承載、L12–15 handoff、instruction 峰值 L15（層定位方法
  沿用）。
- [Cross-model diagnostic](diagnostic-cross-model-probe.md)：零證據 sell
  預設為模板共通底層；證據反應結構與 entity 分化為模型特性（Qwen 分級
  跟隨；Gemma「有證據就 buy」反向模式＋entity 效應方向不同）——雙模型
  對比預期看到**不同方向的狀態簽章**，而非同一狀態差。

**文獻原始設定 vs 本線適應（Adaptations）**：

| 來源 | 原始設定 | 本線適應 | 代價 |
|---|---|---|---|
| entity-concept-decision stance 軸 | 16 家、balanced-evidence family 上 derive | 於本線 prompt family（共用證據＋極性梯級）503 家重新 derive；polarity 對照即 N15 vs P15 狀態差 | 舊方向不可直接 transfer（僅報告 cos 作描述） |
| BEG 2B 層定位 | 16 家 instruction-span readout layer sweep | Gemma 用 16 家子樣本做 Step A 定位（Qwen 沿用 L15 frozen 錨點） | Gemma 錨點為 discovery 性質 |
| 狀態持久化 | 各線皆不存 raw states | 同：in-memory 計算，只輸出 compact 派生量（投影值、PCA 摘要、群組均值） | 不可事後重切層/位置 |

## 2. 術語定義

- **capture layer（捕獲層）**：Qwen 固定 **L15**（上游線 frozen 錨點）；
  Gemma 由 Step A 預先註冊規則定位的 **L\***。
- **capture position（捕獲位置）**：`instruction_span` 的最終 token（與
  Phase 1 prepare 的 span 邊界一致；不等於 decision position）。
- **state s(c, v)**：公司 c 在條件 v（zero / N15 / P15）下，capture layer
  於 capture position 的 residual state（bf16 forward，in-memory，不持久化）。
- **stance 軸（本 family 重新 derive）**：
  `d_stance = normalize( mean_{c ∈ discovery} [ s(c, P15) − s(c, N15) ] )`。
  描述性附加：報告 d_stance 與 (a) 公司間均值差方向、(b) 舊線 stance
  方向（若可載入）的 cos。
- **stance 投影**：`r(c, v) = ⟨ s(c, v), d_stance ⟩`（每公司×條件一個
  finite float）。
- **offset / gain**：per-company 線性拟合 `r(c, v) = offset_c + gain_c ·
  v/4`（v ∈ {0, −4, +4} 三點，零截距不假設；三點兩參數，殘差記為
  nonlinearity_c）。
- **發現/確認 split**：沿用 Phase 1 seed 20260916 stratified split（模型
  無關，兩模型同一批公司）；Qwen discovery = 42 responsive / 360
  fixed-sell，hold-out = 8 / 93。Gemma discovery 組規模以其 Phase 1
  formal 結果為準（同 402 家 discovery 公司）。

## 3. 預設 Input / Output 契約

### 3.1 Input

| 項目 | 路徑／值 | 契約 |
|---|---|---|
| 分組表 | `artifacts/<model>/evidence-insensitivity/runs/phase1-gpu-bf16-01/analyze/summary.json` | 取 `groups`（503 家 × group/split）；Qwen 已 complete；Gemma formal 完成後 |
| prompts | 同 run `prepare/prompts.jsonl` | 複用 Phase 1 的 prompt 文字與 span（不重構）；只取 discovery 公司 × {zero, N15, P15} |
| model | `.cache/models/qwen3.5-4b` / `.cache/models/gemma4-e2b-it`（bf16） | forward 前驗證 model identity |
| 層定位子樣本（Gemma Step A 用） | seed 20260916 自 discovery 抽 16 家（stratified by group，responsive 與 fixed 各半；若 Gemma 某組 <8 家則該組全取＋另一組補滿） | × {zero, N15, P15} |
| capture 容差 | projection 尺度 0.05（對齊 balanced-evidence-gap Phase 3 的 bf16 jitter 帶校準） | determinism 檢查用 |

### 3.2 Output（compact，不存 raw states / residuals / KV cache）

`artifacts/<model>/evidence-insensitivity/runs/phase2-gpu-bf16-01/`：

- `prepare/selection.json`：capture 公司清單（discovery × 3 條件）、Step A
  子樣本、provenance（分組表 SHA、prompt 來源 run、seeds）。
- `forward/records.jsonl`：每（公司×條件）一筆：`ticker`、`condition`、
  `r_stance`（stance 投影）、`norm_state`（state L2 norm，finite）、
  `top5_component_loadings`（對 in-memory PCA 前 5 成分的載荷，finite
  floats，compact）、`n_tokens_span`。
- `forward/layer_sweep.json`（Gemma 僅）：42 層 × 3 條件的 readout-margin
  correlation 表（finite floats）。
- `forward/metadata.json`：generation 參數（無 generation，純 forward）、
  model identity、capture layer/position、determinism 結果（20 筆 re-run
  的 max Δr_stance 與 mismatch 數）。
- `analyze/summary.json`：gates、stance 軸描述、per-company 表（group /
  sector / split / r(zero,N15,P15) / offset / gain / nonlinearity）、
  組間對比、2×2 交叉表、sector 控制結果。所有數值 finite float。
- `manifest.json`：`ArtifactRun` lifecycle。

## 4. 凍結常量

| 項目 | 值 | 來源 |
|---|---|---|
| Qwen capture layer | L15 | entity-to-dial / BEG 2B frozen 錨點 |
| Gemma Step A 定位規則 | `L* = argmax_L |corr( mean_c [s(c,P15)−s(c,N15)] 在 L 的 readout , final margin 差 C_c )|`，16 家子樣本；tie 取較淺層；Step A 完成後 freeze 再跑 Step B | 本提案（BEG 2B 方法移植） |
| capture position | instruction_span 最終 token | Phase 1 span 契約 |
| 條件集 | zero / N15 / P15 | Phase 1 分組條件（offset/gain 三點拟合的最小集） |
| stance 軸定義 | 見 §2 | 本提案（entity-concept-decision 方法移植） |
| split seed | 20260916（與 Phase 1 同） | Phase 1 frozen |

## 5. 預先註冊分析（全部 descriptive，non-causal）

1. **狀態反應曲線**：per-company `r_stance` × {zero, N15, P15}；分組
   疊圖＋分布。
2. **offset vs gain 分解**：per-company 拟合（§2）；組間 offset 差與 gain
   差（Welch t＋Cohen's d，sector-stratified 敏感性）。解讀對照：
   offset 差大＝閾值故事；gain 差大＝承載衰減故事；兩者皆小＝狀態層無
   組差（決策差在更後段讀出）。
3. **2×2 交叉表**：行為組 × P15 狀態方向（r(P15) 高於/低於 0）；四格
   計數與 sector 結構。
4. **狀態共享/分化結構**：PCA 前 5 成分的變異解釋率、組×成分載荷表；
   state norm 的組差（對照舊線「99.5% 共享、0.5% 公司間差」背景）。
5. **stance 軸有效性**：d_stance 對 per-company 極性狀態差變異的解釋率；
   與 margin C_c 的相關。
6. **雙模型對照**：同分析的 Qwen vs Gemma 並排（預期：不同方向/結構的
   簽章；Gemma 的「有證據就 buy」應反映在其 offset/gain 形狀）。

## 6. Gates（fail-closed sanity gates）

| Gate | 定義 | 門檻 |
|---|---|---|
| G-2A determinism | 20 筆 re-run 的 max \|Δr_stance\| 與 text/record mismatch | Δ ≤ 0.05 且 0 mismatch |
| G-2B stance 軸有效性 | d_stance 對極性狀態差變異的解釋率（R²） | ≥ 0.30（低於則 stance 投影降級為描述，組間對比改以 offset/gain 直接於原始投影差進行，並記錄） |
| G-2C 分組功效 | discovery split 兩組家數 | 各 ≥ 10（Qwen 42/360 已知通過；Gemma 以其 formal 分組表為準——若某組 <10，該模型只做 §5.1/5.2/5.4 的 within-model 描述，不做組間對比宣告） |

## 7. Run 結構

`prepare → forward → analyze`（每模型單一 run，`ArtifactRun` lifecycle）。

| 階段 | 內容 | 產出 |
|---|---|---|
| prepare | 分組表載入＋fail-closed（503 家、group 非空比例）、discovery 公司×3 條件 selection、Step A 子樣本（Gemma） | `prepare/*` |
| forward | Qwen：402×3 = 1206 forwards（純 forward，無 generation）。Gemma：Step A 16×3×42 層 readout（同一批 forward 內逐層 readout，非 42 次 forward）＋ 402×3 = 1206 forwards；各 20 筆 determinism re-run | `forward/*` |
| analyze | gates、§5 全分析、雙模型對照表 | `analyze/summary.json`、`manifest.json` |

Gemma Step A 的 42 層 readout 在同一次 forward 的 hook 內逐層計算
（16×3 = 48 forwards，非 48×42）。

## 8. 邊界與限制

- 描述性；無介入、無 causal claim（Phase 3 才做）。
- 狀態捕獲限 discovery split；hold-out 留 Phase 3 confirmation。
- 不持久化 raw states：層/位置/軸若需變更，必須重跑 forward。
- Gemma L* 為 16 家 discovery 子樣本定位（discovery 性質）；其組規模小時
  依 G-2C 降級為描述。
- 雙模型 hidden 皆 2560 但 tokenizer/模板不同；跨模型對照只比**結構**
  （offset/gain 形狀、2×2 格局），不比絕對數值。
- bf16 forward；jitter 帶 0.05（G-2A）。

## 9. 版本分立觸發條件

任一變更即建立新版本文件，禁止原地修改：

1. capture layer/position 或條件集（zero/N15/P15）；
2. stance 軸定義或 offset/gain 拟合規則；
3. gate 門檻；
4. split（公司集、seed）；
5. 加入 generation 端點或介入臂（後者屬 V2 版本，不在本 phase）。

## 10. 實作與驗證

- 新 module `llm_bias/evidence_insensitivity/phase2.py`（state capture
  hook、層 sweep、offset/gain 拟合、組間分析）＋ operator 新增 subcommand
  `phase2`（`prepare` / `forward` / `analyze`，`--model` / `--model-slug`）。
- 重用 `core/prompt_input`（span）、`core/inference`（forward hook）、
  `core/artifacts`；不 import 其他 experiment package。
- 測試（fake model，無 GPU）：selection 計數（1206）、層 sweep 規則
  （argmax＋tie-break）、offset/gain 三點拟合邊界（共線/零方差 fail-closed）、
  G-2A/B/C 門檻邊界值、summary 過 core `_RAW` 序列化 guard、pipeline
  lifecycle smoke。
- 驗證：`uv run pytest -q`、`uv run python -m compileall -q llm_bias`、
  `uv build`、`uv lock --check`。
- 每模型 1-prompt real-model preflight（capture hook 輸出 finite、span
  邊界對齊）。

## 11. Revision history

- **Rev 1（2026-09-16）**：初稿。雙模型統一設計（用戶指示：主線統一使用
  Qwen＋Gemma）；Gemma 層錨點以 Step A 定位（其 42 層無 L15 對應）；
  offset/gain 分解與 2×2 交叉表來自 2026-09-16 用戶討論；Gemma 組規模
  未定，G-2C 含降級分支。
- **Rev 1.1（2026-09-16，frozen）**：用戶批准 formal run。實作對齊註記
  （不改變任何 pre-registered 統計量或 gate）：(a) capture position 在 prepare 逐筆驗證
  `1 ≤ instruction_span 終點 ≤ 序列長`（越界即 fail-closed），捕獲於
  終點前一 token；Qwen chat template 在 user turn 後附加 assistant
  前綴＋空 think block（9 token），故 span 終點一般不等於 prompt 最終
  token——以 span 為準（§2 定義）；(b) Step A 的
  per-company readout 精確化：`x_{c,L} = ⟨s(c,P15)−s(c,N15), d_L⟩`，
  `d_L` 為該層子樣本均值極性差方向；(c) offset/gain 拟合為 v/4 ∈ {−1,0,+1}
  的最小二乘（offset＝截距、gain＝斜率、nonlinearity＝最大殘差）；
  (d) determinism 20 筆為 seed 20260916 分層抽取（7 N15 / 7 P15 / 6 zero）；
  (e) per-company C_c 取自 Phase 1 分組表 `contrast_c` 欄位；(f) prompts
  直接複用 Phase 1 `prepare/prompts.jsonl`（primary arm，byte-identical）。
- **Rev 1.2（2026-09-17，實作對齊註記，不改任何 frozen 門檻）**：Gemma
  formal run 的 G-2A 以絕對容差 0.05 fail（max Δr_stance 0.129）。兩模型
  **相對** jitter 相同（Qwen 0.0197 / norm 10.34 ≈ 0.19%；Gemma 0.129 /
  norm 68.79 ≈ 0.19%）——絕對容差未隨 state norm 縮放是校準限制。frozen
  gate 結果維持（不溯及）；後續跨 model 的 G-2A 應先實測 jitter 帶再定
  絕對容差。另：2×2 交叉表因 stance 軸符號在兩模型皆退化（402/402 同側），
  無分辨力——記錄為設計教訓（若需狀態方向交叉表，應改用 r(zero) 或
  r(N15)）。結果見 [Phase 2 報告](report-phase2.md)。
