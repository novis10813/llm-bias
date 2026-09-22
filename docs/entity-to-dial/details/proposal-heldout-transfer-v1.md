# Entity-to-Dial Held-out Transfer V1：全 sector external entity evaluation

**狀態**：completed；formal run `entity-to-dial-heldout-transfer-v1-01` 完成於 2026-09-22。  
**對象模型**：Qwen3.5-4B（bf16；模型、tokenizer、chat template 與 revision 必須與 `entity-to-dial-e-01` basis provenance 相符）。  
**上游依賴**：[Entity-to-Dial Phase E](proposal-phase-e.md) formal run `entity-to-dial-e-01` 的 L15 top-8 right-singular-vector basis \(V_8\)。  
**研究問題**：不重新 fitting 的前提下，L15 \(V_8\) 是否在一個未參與 construction 的 200-company cohort 上，仍能恢復 source→target **fixed-answer margin** state transfer？

本版本建立 C3 的 held-out entity evaluation。它不改寫 Phase E 的 construction-set result，不是 Selective-intervention M6 removal 的重跑，也不量測或宣稱 generated decision change。

## 1. Estimand 與邊界

對同一 prompt variant 的 source \(s\) 與 target \(t\)，令 \(P\) 是 L15 instruction-span 的 target-complete offset mapping，\(h_{c,p}\) 是 L15 post-block residual state。主臂為既有 frozen basis 的 pair-specific projected transplant：

\[
h'_{t,p}=h_{t,p}+V_8V_8^\top(h_{s,p}-h_{t,p}),\qquad p\in P.
\]

它不是把同一個 universal vector 加到所有 prompts；每條 transfer 仍使用該 source-target pair 的 observed state difference。reference full swap 為：

\[
h'_{t,p}=h_{s,p}.
\]

每個 prompt 的 score 為既有 fixed-answer continuation margin：

\[
M=\log p(\text{buy})-\log p(\text{sell}).
\]

對每個 variant \(v\)、有方向 transfer \(s\to t\)、arm \(a\)，定義 toward-source shift：

\[
D_{a,v}=\operatorname{sign}(M_{s,v}-M_{t,v})\,[M_{a,v}-M_{t,v}].
\]

主分析先在三個 evaluation variants 對每條 directed transfer 取中位數：

\[
\widetilde D_a=\operatorname{median}_{v\in E}(D_{a,v}),
\qquad
R_{V_8}=\frac{\widetilde D_{V_8}}{\widetilde D_{\mathrm{full}}}.
\]

只有 \(|\widetilde D_{\mathrm{full}}|\geq 0.2\) nats 的 transfer 進入 ratio summary。這沿用 Phase E 的 denominator threshold；eligibility 只由 full-swap reference 決定，絕不依 \(V_8\) 或 random-control 結果篩選。所有 excluded transfer、其 stratum、方向與 full-swap magnitude 必須完整輸出。

margin transfer 是測試的唯一 outcome。它證明或限制受測 internal readout 的 state-transfer sufficiency，**不能單獨代表 generated Buy/Sell decision、完整 entity circuit、universal entity subspace 或跨模型結論**。

## 2. Frozen inputs 與 fail-closed provenance

### 2.1 Basis 與 intervention

下列值在任何 held-out model forward 前確認並凍結：

- `entity-to-dial-e-01/analyze/summary.json` 的 SHA-256、上游 manifest status 與其 `pca_basis_vectors` / `pca_singular_values`；
- `load_pca_basis(..., expected_k=16)` 對 basis shape、finite values、奇異值順序與 orthonormality 的既有 fail-closed checks；僅取前 8 columns 為 \(V_8\)；
- L15 post-block hook、instruction-span scope、FP32 delta arithmetic 與 cast-back semantics；
- shared-evidence template、answer prefix、`buy`/`sell` answer-token scoring；
- model/tokenizer/chat-template identity 與 upstream provenance 一致。

不得以 held-out cohort 的 states、margins 或 transfers 重新 fit SVD、修改 \(k\)、選 layer、改 span、調整 intervention strength，或更換 basis source run。

### 2.2 Population

母體為 `data/all_constituents_2020_2025.csv` 中 `index_name == "S&P 500"` 且 `year == "2024"` 的 canonical rows。它必須通過既有 `load_population` contract：503 rows、ticker unique、name/sector non-empty。

排除名單是下列兩集合的 union：

1. Phase E \(V_8\) construction 的 16 家 benchmark entities：`AMAT, GLW, HPE, IT, AXP, BLK, C, GS, ABT, BDX, DHR, SYK, CSX, DE, HON, NSC`；
2. frozen M6-V2 external-removal manifest `companies` 的 12 家 selected entities。

M6 的 12 家雖未參與 \(V_8\) fitting，仍排除以維持本 cohort 對既有 external intervention evaluation 的完全新穎性。prepare 必須驗證 M6 manifest 的 schema、其自身 artifact hash、12 個 entries 的 required fields與 ticker uniqueness，並驗證其與 construction 16 家無 overlap。M6 manifest 的 historical source-file hash 只記錄其原始 selection provenance，**不要求**等於本協議 `data/all_constituents_2020_2025.csv` 的 hash；M6 ticker 若不在 2024 503-row population 中，從 exclusion set 的 intersection 自然不產生效果，非錯誤。只有 manifest schema／entries缺失、manifest hash不符、construction overlap、非 503-row population 或本協議 population file hash mismatch 才 fail-closed。

### 2.3 Cohort selection

從 eligible population 作 **200-company proportional GICS-sector stratified sample**：

- selection seed：`20260930`；
- 各 sector quota 以 `n × sector_count / eligible_count` 計算，先取 floor，剩餘名額依 fractional remainder 由大至小分配；remainder 相同時 sector name lexicographic ascending；
- 每個 sector 中，使用 SHA-256-derived local seed `sha256("20260930:{sector_index}:{sector}")[:8]` 的 deterministic shuffle，取 quota 前列；`sector_index` 是 sector-name lexicographic order 的 zero-based index；
- final cohort 按 ticker sort；不掃描 seed、不比較多個 cohort、不使用任何 model output 選人。

在 sampling 前，以 tokenizer 建立四個 frozen prompt variants 並驗證 company name/ticker、answer prefix、instruction span 非空且四 variants 均可完成 source/target offset mapping。這是結構性 eligibility check，不得執行 model forward。若 selected 200 任一 ticker 未通過，formal run fail-closed；不得以候補 ticker 替換。

prepare artifact 必須在任何 model forward 前寫入 `cohort_manifest.json`，保存 population CSV SHA-256、全部 eligible tickers、exclusions 及其來源、per-sector eligible counts/quotas、seed、selection algorithm identifier、selected rows與 selected-ticker digest。

本 cohort 對應 2024 S&P 500 的 eligible ticker population，不對全體上市公司、其他年份、其他模型或真實投資情境外推。

## 3. Prompts、selection/evaluation split 與 pair graphs

每個 company 均使用已凍結 shared-evidence prompt family 的四 variants：

| Label | `reverse` | `order` | Role |
|---|---:|---:|---|
| `selection` | `false` | `0` | 只用於 pairing graph construction |
| `eval_forward_order_reverse` | `true` | `0` | evaluation |
| `eval_reverse_evidence` | `false` | `1` | evaluation |
| `eval_reverse_both` | `true` | `1` | evaluation |

`selection` clean margin 只形成 pair graph；不得進任何 primary ratio、random-control comparison 或 bootstrap summary。evaluation variants 不能被用來增加、刪除或重配 pairs。

### 3.1 Sector-within graph

對每個 GICS sector，將 cohort 內 tickers 依 `(selection_margin ascending, ticker ascending)` 排序。令該 sector 有 \(n\) entities，對每個 \(j=0,\ldots,\lfloor n/2\rfloor-1\)，建立：

\[
(\mathrm{high},\mathrm{low})=(rows[n-1-j], rows[j]).
\]

奇數 \(n\) 的中央 entity 不進 sector-within graph，仍可進 cross-sector graph。每個 pair 評估 high→low 與 low→high 兩個 directed transfers。

此 stratum 問的是：固定 sector 時，frozen \(V_8\) 是否仍恢復 entity-specific state transfer？

### 3.2 Cross-sector graph

將全 cohort 依 `(selection_margin ascending, ticker ascending)` 排序，底部 100 家為 targets set \(L\)，頂部 100 家為 sources set \(H\)。只允許 \(s,t\) 滿足：

\[
s\in H,\quad t\in L,\quad sector(s)\ne sector(t).
\]

在這個 bipartite graph 上，以 deterministic maximum-cardinality matching 形成 100 個 pairs：依 \(L\) 的 `(selection_margin ascending, ticker ascending)` 順序跑 depth-first augmenting paths；每個 target 的 candidate sources 依 `(selection_margin descending, ticker ascending)` order 考慮。matching 最大 cardinality 必須為 100；否則 fail-closed，不以相同 sector pair、不同 100/100 split、另一個 seed 或事後人工選配替代。

每個 cross-sector pair 也評估 high→low 與 low→high。此 stratum 問的是：frozen \(V_8\) 是否可跨 GICS-sector context transfer？

pair graphs、selection margins、algorithm version、ties、unmatched within-sector tickers、每條 pair 的 sector labels與 SHA-256 必須在 **任何 evaluation-arm forward 前**寫入 `forward/pair_graphs.json`。此時 graph 是 frozen；不得根據 full swap、\(V_8\)、random arm 或任何 evaluation result 修改。

兩個 graph 都使用同一 200-company cohort，但在每一個 stratum 內各 entity 至多屬於一個 undirected pair。兩個 stratum 分開估計與解讀，絕不 pooled。

## 4. Intervention arms、controls 與 real-model smoke

對每個 graph、undirected pair、兩個 direction 及三個 evaluation variants，記錄：

1. **clean**：source/target margins及 transient L15 post states；
2. **full**：source post state替換 target instruction span；是 \(R\) 的唯一 reference denominator；
3. **V8**：frozen \(V_8\) projected pair-specific transplant；
4. **random-8D-0..3**：四個 independent random orthonormal 8D bases 的同一 projected transplant；
5. **full self-source no-op** 與 **V8 self-source no-op**：對每個 cohort company及每個 evaluation variant 檢驗 transform semantics。

random bases 在 prepare 一次建立：對 `j ∈ {0,1,2,3}`，以 CPU `torch.Generator` seed `20260930 + j` 抽取 \(2560\times8\) FP64 iid standard-normal matrix，thin QR orthonormalize，並以每個 column 最大 absolute entry 為正的 sign convention 固定 QR sign。只保存每 basis 的 seed、shape、orthonormality diagnostic 與 SHA-256；不保存 matrix。四個 controls 與 \(V_8\) 都是 8D orthonormal projector，且投影同一 pair-specific \(h_s-h_t\) delta。

不強制將 random projected delta 的 norm 縮放為 \(V_8\) projected delta 的 norm：兩者投影後的 norm 本來就是要比較的 geometry outcome；額外 scaling 會改變 estimand。random arm 只檢驗「任意同維度 orthonormal residual subspace」是否同樣恢復 full transfer。

no-op contract：每一條 self-source full / \(V_8\) intervention 都必須 finite，且 `abs(patched_margin - clean_margin) <= 1e-12` nats。任何不符停止 formal analysis。

### Smoke

formal run 前必須完成一次 real-model smoke。它使用由 cohort manifest ticker-sorted 前兩家公司組成的 fixed pair及 `eval_forward_order_reverse` variant；這個 pair 不根據 margin 選取，smoke outputs 不進 formal records或 pair graph。smoke 必須驗證：

- frozen basis/source manifests、population/cohort manifests、prompt spans與 offset mapping；
- clean、full、\(V_8\)、random-8D-0 finite；
- full與 \(V_8\) self-source no-op 為 bit-exact；
- hook各 forward 恰好 fire 一次；
- 不保存 source/target residuals、deltas或 caches；
- `prepare → forward → analyze → finalize` artifact lifecycle 和 compact schema 完整。

smoke 不設 effect-size expectation；full swap或 \(V_8\) 小效應不構成 smoke failure。formal run 必須在新 run root重新執行所有 selection、pairing與evaluation forwards。

## 5. Analysis

### 5.1 Per-edge outcomes

對每條 directed transfer，三個 evaluation variants先個別計算 \(D_{\rm full}\)、\(D_{V_8}\)與每個 \(D_{\rm random,j}\)，再對 variant取中位數。full eligibility rule套用於 \(\widetilde D_{\rm full}\)。對 eligible edges：

\[
R_{\mathrm{random},j}=\frac{\widetilde D_{\mathrm{random},j}}{\widetilde D_{\mathrm{full}}},
\qquad
A=R_{V_8}-\operatorname{median}_{j=0}^{3}R_{\mathrm{random},j}.
\]

每個 stratum 分別輸出：

- \(R_{V_8}\) distribution、median、full-swap magnitude與 eligibility rate；
- four random-control distributions、其 across-basis median，及 paired advantage \(A\)；
- high→low 和 low→high 方向分開的結果；
- sector-within 的 sector breakdown，以及 cross-sector 的 source-sector × target-sector breakdown；
- variant-level records和三-variant aggregation diagnostics。

如 full transfer在 held-out prompt上不能穩定恢復 source margin，這是結果的一部分，不能把 failed full arm 從報告中隱去或改用 \(V_8\) arm作 denominator。

### 5.2 Uncertainty and predeclared interpretation

每一 stratum 的 independent resampling unit 是 **undirected pair bundle**：一個 bundle含該 pair的 high→low、low→high records。因為每個 graph內 entity不重複，這等價於該 stratum 的 entity-clustered resampling，並保留同一 pair兩方向的依賴。

以 10,000 replicate、seed `20260930` 的 paired-bundle percentile bootstrap，重抽有效 pair bundles並重算所有 reported median summaries。95% CI 不跨 stratum比較，也不將兩個 graph的重複 entities視為獨立樣本。

每個 stratum的預先定義讀法：

| Condition | Interpretation |
|---|---|
| median \(R_{V_8}\geq0.8\)，且 95% CI lower bound \(\geq0.8\)，並且 median \(A>0\) 且其 95% CI lower bound \(>0\) | 受測 stratum 支持 frozen \(V_8\) 的 high-recovery, subspace-specific fixed-margin state transfer。 |
| median \(R_{V_8}\geq0.8\)，但任一 CI criterion未過 | point estimate與 construction result一致，但 precision或random specificity未確認；記為 unresolved。 |
| median \(R_{V_8}<0.8\) | 不支持在該 stratum維持 Phase E 所採用的 high-recovery target；仍報告估計值與CI，不把結果稱為 zero effect。 |

\(0.8\) 是 Phase E E2a 的既有 low-dimensional recovery target，作為 held-out high-recovery interpretation threshold；它不宣稱自然界存在二元 gate。random specificity條件失敗時，即使 \(R_{V_8}\) 很高，也只能說有 transfer，不可歸為 \(V_8\)-specific evidence。

不補 sample、不換 ticker、不改 graph或 bootstrap method來縮窄 CI。任何變更 cohort、exclusion、seed、selection variant、pair algorithm、layer/span, \(k\)、denominator threshold、random control、evaluation outcome或interpretation threshold，均建立 Held-out Transfer V2，不回填本版本。

## 6. Workflow、artifacts 與資料政策

Package workflow必須使用 shared `llm_bias.core` lifecycle：`prepare → forward → analyze → finalize`；不得讓 Entity-to-Dial package新增對另一實驗 package的 import。

run root：

```text
artifacts/qwen3.5-4b/entity-to-dial-heldout-transfer/runs/<run-id>/
```

compact outputs：

- `prepare/cohort_manifest.json`：population/cohort provenance、sector counts、exclusions、hashes、seeds、basis/random-basis metadata及 model identity；
- `prepare/prompts.jsonl`：200 × 4 prompt metadata、ticker/name/sector、variant、instruction spans與 input provenance；
- `forward/selection.jsonl`：200 selection-variant clean margins；
- `forward/pair_graphs.json`：frozen within/cross graphs、selection margins、matching diagnostics、graph hashes；
- `forward/records.jsonl`：graph/pair/direction/variant/arm、clean source/target margin、patched margin、toward-source shift、full eligibility fields、hook/no-op checks；
- `forward/noop_records.jsonl`：每個 company × evaluation variant × {full, V8} 的 bit-exact verification；
- `analyze/summary.json`：每 stratum/方向的 ratios、random comparisons、pair-bootstrap CI、exclusions、sector breakdown、interpretation、record counts與 `raw_runtime_payloads: false`；
- `manifest.json`：lifecycle、input/output SHA-256、record counts與 provenance。

不得持久化 raw activations、residual states、state deltas、hidden states、gradients、Jacobians、KV caches、random basis matrices或完整 model outputs。只有 compact margins、derived ratios、hashes、IDs和metadata可以輸出。

## 7. Required tests and acceptance criteria

以 fake model、monkeypatch和temporary directories的 deterministic tests覆蓋：

1. 503-row population filter、construction/M6 exclusions、比例 sector quota、deterministic 200-company manifest及任何 selected invalid prompt的 fail-closed；
2. sector-within graph的排序、ticker tie-break、奇數 sector leftover；
3. cross-sector perfect matching的 determinism、no same-sector edge、以及無 full matching時 fail-closed；
4. selection variant不進 evaluation summary；evaluation records不改 graph；
5. source/target offset mapping、full和projected transform、四個 random bases的 shape/orthonormality/digest、random zero-delta no-op；
6. full-reference-only denominator rule、variant median aggregation、random paired advantage與excluded records完整保留；
7. pair-bundle bootstrap保留兩方向依賴；三段 interpretation thresholds；
8. compact artifact schema、raw-state禁止、manifest lifecycle及 source provenance mismatch fail-closed；
9. real-model smoke的 finite/hook/no-op acceptance，僅在實作完成後執行。

實作前不得執行 formal model run。實作完成後，先跑新測試、既有 Entity-to-Dial tests和 workflow-boundary tests；只在上述 smoke全過後，才可授權 formal 200-company execution。

## 8. Formal result

正式 run `entity-to-dial-heldout-transfer-v1-01` 依本協議執行，run root 為：

```text
artifacts/qwen3.5-4b/entity-to-dial-heldout-transfer/runs/entity-to-dial-heldout-transfer-v1-01/
```

run 從 503 家 2024 S&P 500 constituents 排除 16 家 construction 與 12 家 M6 entities，對 476 家 eligible population 按 11 個 GICS sector 與 1 個 `Unspecified` label 比例抽取 200 家。sector-within graph 形成 97 pairs、留 6 個奇數-sector entities；cross-sector graph 形成 100 pairs。394 directed transfers 各有 three evaluation variants × seven arms；1,200 self-source no-op records 均為 `delta_m = 0.0`。

| Stratum | Eligible / excluded directed transfers | $V_8$ median recovery（95% pair-bundle bootstrap CI） | $V_8$ minus median random-8D ratio（95% CI） | Status |
|---|---:|---:|---:|---|
| Sector-within | 124 / 70 | 0.6759 [0.6420, 0.7033] | 0.6656 [0.6450, 0.6929] | `not_supported` |
| Cross-sector | 140 / 60 | 0.6596 [0.6343, 0.7089] | 0.6629 [0.6321, 0.6982] | `not_supported` |

兩個 stratum 都保留正的 random-control advantage，表示 frozen $V_8$ 比同維 random projector 更保留 fixed-margin transfer；但兩個 median recovery 與 CI lower bound 都低於 0.8 high-recovery target。因此本版本不支持把 Phase E 的 98.3% construction-set recovery 外推為 held-out entity generalisation。結果只涉及 fixed-answer margin，不提供 generated decision conclusion。

## 9. Non-goals

- 不學習 RDO/RCO 型 gradient-optimised direction或 cone；
- 不以 held-out entities重估 \(V_8\) 或更新 Phase E result；
- 不重跑、rescue或重解釋 Selective-intervention M6-V2 removal result；
- 不將 margin transfer寫成 generated decision flip、selective intervention或完整 causal circuit；
- 不在本版本新增 cross-model、cross-task或 normative bias claim。

## 10. Revision history

| Rev | Status | Summary |
|---|---|---|
| V1 | proposed | 從 2024 S&P 500 eligible population按全 GICS sector比例抽200家，排除 Phase E construction 16家及 M6 12家；以 selection/evaluation variant split建立 sector-within和cross-sector graphs，對 frozen L15 \(V_8\) source→target projected transplant作 fixed-margin held-out evaluation。 |
| V1 formal | completed | Run `entity-to-dial-heldout-transfer-v1-01` 完成；sector-within 124/194、cross-sector 140/200 directed transfers eligible，兩 stratum $V_8$ median recovery 0.6759／0.6596 與 CI lower bound 皆低於 0.8 target，status 皆為 `not_supported`；random-8D advantage 皆為正。 |
