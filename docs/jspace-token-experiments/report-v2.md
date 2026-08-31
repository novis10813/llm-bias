# J-space outcome-conditioned decision-flip V2：實驗報告

**Formal pipeline verdict：`success=false`.** Buy steering 通過，但 sell-direction
Holm gate 因 eligible test tickers 數量不足而失敗。Frozen protocol、controls、gates，
以及 prior probe / direction decode 定義見 [V2 提案](proposal-v2.md)。

### First probe run（`outcome-flip-prior-probe-20260828T110726Z`，2026-08-28）

第一次正式 probe run，12 個 frozen conditions（6 tickers × 2 sector labels），
model `.cache/models/qwen3.5-4b`，direction 用 frozen discovery identity
（`evidence_item_end`，L10–L30 全部 21 層 hash 驗證通過，
`direction_verified=true`）。scoring prompt 固定 91 tokens；residual norm 範圍
6.6–56.7。Run 位置：`artifacts/qwen3.5-4b/jspace-outcome-direction-flip/runs/
outcome-flip-prior-probe-20260828T110726Z/`（worktree branch
`experiment/header-only-prior` 的 ignored `artifacts/`；status complete）。

主要數值結果（association-only）：

- **Margin**：12 個 conditions 全部 clean-sell，M ∈ [−4.93, −3.39]。Sector
  label means：Technology −4.021、Financial Services −4.372；ticker means：
  JPM −3.854、NVDA −3.895、AAPL −3.878、WFC −4.228、INTC −4.494、MS −4.828；
  ticker-group means：Technology tickers −4.089、Financial Services tickers
  −4.303。Zero-evidence template 在該 model 下有強 sell prior。
- **Sector-label contrast（Technology − Financial Services，per-ticker paired
  bootstrap，n=6）**：M point +0.351，95% CI [−0.019, +0.736]（含 0，弱且
  未決）；per-layer projection_relative 全部 |point| ≤ 0.0035，只有 L18 的 CI
  不含 0（+0.0029 至 +0.0042），magnitude 可忽略。
- **NVDA − JPM contrast**：M point −0.042（≈0）；per-layer
  projection_relative |point| ≤ 0.001。
- **Projection 的 template-shared 結構**（12 個 conditions 幾乎一致，非
  identity effect）：L10 ≈ +0.046、L15–L18 負（最深 ≈ −0.049）、L23–L25 正
  （最高 ≈ +0.043）、L30 ≈ −0.133（相對值）。L30（最後 decoder layer，緊接
  final norm + unembedding）的強負 projection 與 sell-leaning margin 一致。

解讀：在受測 template/model/frozen direction 下，header 的 ticker 與 sector
identity 在 decision position 的 frozen V2 outcome axis 上沒有可量測的
association（所有 per-ticker contrast 的 relative projection 皆 ≤ 0.4% 的
residual norm）；每個 condition 的大讀數（sell-leaning margin 與逐層
projection pattern）是 12 個 conditions 共有的 template-level 結構，不是
entity-specific effect。本 run 不構成任何 entity 或 sector 的 causal claim；
若未來要检验 header identity 的 causal role，需要另建 intervention 設計
（新 experiment version）。此 run 為 first formal probe run；重跑或改設計前
必須先記錄於 Revision record。

### Direction decode 結果

**Run**：`outcome-decode-tech-20260828T113010Z`（diagnostic status；run root：
`artifacts/qwen3.5-4b/jspace-outcome-direction-decode/runs/outcome-decode-tech-20260828T113010Z/`，
本 worktree 的 ignored `artifacts/`）。模型 `.cache/models/qwen3.5-4b`
（`CUDA_VISIBLE_DEVICES=0`），canonical lens 以絕對路徑唯讀載入（主 repo
`artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt`）。

**Binding 與 recompute**：direction 依 V2 no-persistence protocol 在記憶體
deterministic recompute（35 個 discovery tickers、21 個 fitted layers、雙
position rule），layer-wise direction hash 與
`outcome-flip-tech-discovery-det-20260828T015605Z` 的 identity 全部 21 層 × 2
rules 一致（fail-closed 未觸發）；`z(-d) = -z(d)` self-check max abs deviation
= 0.0（bit-exact）。Selection：L10–30、`evidence_item_end`、r=0.4（dose 只
記錄於 provenance，不參與解碼）。Buy/Sell answer token：`buy`=19180、
`sell`=33680。

**逐層 buy/sell 軌跡**（`+d` = Buy steering；logit margin = z(buy)−z(sell)）：

| layer | +d logit margin | +d buy rank | −d sell rank | −d sell probability |
|---|---|---|---|---|
| 10 | −0.16 | 67660 | 190822 | 8.8e-07 |
| 15 | 2.89 | 65513 | 13162 | 2.5e-06 |
| 18 | 4.66 | 47679 | 3411 | 2.4e-05 |
| 21 | 11.84 | 118 | 13 | 1.6e-03 |
| 24 | 21.84 | 1 | 1 | 0.658 |
| 26 | 23.37 | 5 | 1 | 0.822 |
| 28 | 19.48 | 36 | 1 | 0.265 |
| 30 | 7.94 | 3108 | 17 | 3.1e-04 |

（完整 21 層 × 雙 sign 的 top-k、rank、logit、probability 在
`forward/direction_decode.jsonl`；band scope 在
`analyze/direction_decode_summary.json`。）

**代表性 top tokens**：

- `+d`：L15「驚喜 / inspiring / seamlessly」、L18「uplifting / positive /
  boost」、L24「buy / Life」、L26「 life / Life / Life」；band（L10–30 等權
  full-softmax 平均）top-6 為「 life / 「 / positive / Life / _card / buy」，
  buy 平均 probability 1.05e-3（rank 6）、sell 2.8e-7。
- `−d`：L15「failed / fails / worse」、L18「枯萎 / 凋零 / 负面」、L24
  「sell / Sell / sell」（sell probability 0.66）、L26「sell / Sell / Sell」
  （0.82）；band top-1 即「sell」（平均 probability 0.202，rank 1），其次
  failed / doomed / fails / worse / futile / failure / fatal。

**觀察（非結論）**：answer token 在 L21–L29 中層帶進入 top-30（buy 於 `+d`
L23–27、L29；sell 於 `−d` L21–30），L24–26 最尖銳；L10–12 與 L30 以
format/低資訊 token 為主，margin 明顯較小。`−d` 的 sell 解碼強度（峰
probability ≈0.82）遠高於 `+d` 的 buy（峰 ≈0.005），與 V2 calibration 中 sell
方向 flip 數（8）高於 buy（4）的觀察同向。全 21 層中 sell 從未進入任何 `+d`
top-30、buy 從未進入任何 `−d` top-30（sign separation）。

**解讀限制**：Direction decode 是 transported representation readout of a
fitted aggregate axis，不是 chain-of-thought、discrete reasoning path 或
standalone causal claim；top tokens 只描述該層 axis 指向哪些詞彙，不證明任何
token 因果驅動 buy/sell 偏好，也不能把 V2 的 steering 效果歸因到解碼出的詞彙。
V2 Test run 1 已顯示 position specificity 不成立；本 decode 不補強、不推翻
任何 causal 宣稱，也不消耗 V2 的 run-once 性質。

Regression tests：`tests/test_jspace_outcome_decode.py`（fake model/lens、無 GPU）。

## Revision record

- **Probe protocol（Zero-Evidence Header-Only Prior Probe）**：在 V2 文件新增
  readout-only prior probe 協議：frozen template version 1（header 的
  `Stock Ticker:`/`Sector:` + 單一中性 evidence item `No evidence provided.`）、
  frozen discovery direction identity 的 in-memory recompute + hash
  verification（fail closed）、final-position projection（unit direction 的
  FP32 內積）與 V2 local-scale relative projection、V2 同函數 margin scorer、
  12-condition matched design（6 tickers × 2 sector labels）、descriptive
  analysis（group means、sector-label / pair / ticker-group contrasts、paired
  bootstrap CI）與 interpretation limits（association-only）。新增
  `prepare-prior-probe-config` / `run-prior-probe` CLI、`outcome_prior_probe_*`
  artifacts 與 `tests/test_jspace_prior_probe.py`。不修改任何 Draft 1 凍結值、
  V2 estimator 或既有 artifact schema。
- **Probe run 1（第一次正式 probe run，2026-08-28）**：
  `outcome-flip-prior-probe-20260828T110726Z`（12 conditions，direction
  verification 通過）。結果：全部 12 個 conditions clean-sell（M ∈ [−4.93,
  −3.39]）；sector-label contrast M +0.351（95% CI 含 0）、NVDA − JPM
  M −0.042；header identity 對 frozen outcome axis 的 per-ticker projection
  contrast 全部 |relative| ≤ 0.0035，無可量測的 identity association；大的
  逐層 readout（含 L30 ≈ −0.133 relative）為 12 個 conditions 共有的
  template-level 結構。解讀上限為受測 template/model/direction 下的
  association/prior evidence，非 entity causal proof。
- **Draft 0**：將 V1 vocabulary-direction screen 與 V2 outcome-gradient decision-flip
  protocol 分開；把 Buy/Sell decision flip 設為 primary outcome，full generation 設為
  必要 behavioral validation。
- **Draft 1**：凍結 tie rule（`M=0` 排除出 eligible set）、local scale
  （residual norm，floor 1.0）、dose grid `r ∈ {0.05, 0.10, 0.20, 0.40}` 與 safety
  bound `r ≤ 0.50`、candidate layer bands `{L14–L26, L12–L28, L10–L30}`、candidate
  position rules（evidence item 結尾 / evidence span 全 tokens）、clean margin strata、
  calibration selection rule 與 fail-closed criterion、最低實用 flip rate 10%、
  reverse-flip gate、full-generation gates（parse success ≥ 90%、decision agreement
  ≥ 70%）與 multiplicity family（2 primary estimands，Holm）。
- **Implementation 1**：依 Draft 1 實作 V2 CLI、config schema、artifact
  schema 與 regression tests；未修改任何 Draft 1 凍結值。Implementation clarifications：
  decision margin 重用既有 single-token FP32 scorer（於 prompt 最後 position 計分），
  fitting target 用同一 tail 的可微分版本；delivered dose 改以實際 applied
  perturbation 量測；reverse gate 對空 reverse-eligible pool 視為 rate 0。第一次正式
  discovery run 尚未執行。
- **Implementation 2**：第一次 discovery smoke run（real
  `HFLensModel`）暴露 correctness bug：wrapper 在建構時 freeze 所有 params，frozen
  leaves 的 forward 不建 autograd graph，`fit_prompt_layer_gradients` 因此拿到沒有
  `grad_fn` 的 layer output 而 fail closed（`element 0 of tensors does not require
  grad`）。修復方式是在第一個 fitted layer 的輸出 `requires_grad_(True)` re-root
  （同 jlens `ActivationRecorder.start_graph_at` 慣例），retained graph 恰為 fitted
  layers；fake-model 測試的 params 同步 freeze 以把此路徑鎖進 regression tests（移除
  修復後 5 個 fitting/pipeline 測試 fail）。不修改任何 Draft 1 凍結值、estimator 或
  artifact schema。
- **Implementation 3（current）**：第一次 discovery 正式 run
  （`outcome-flip-tech-discovery-20260828T011535Z`）完成後，第一次 calibration
  嘗試（`outcome-flip-tech-calibration-20260828T011744Z`）fail closed：recomputed
  direction hash 與 discovery identity 不一致（`evidence_item_end` layer 10）。
  Diagnosis：default bf16 GEMM backward 非 deterministic，discovery identity
  本身不可 bit-exact 重現（實測同一 process 重複 fitting 的 raw gradient max abs
  差異約 5e-3，且集中在 L10–L26）。修復：`run_outcome_flip_pipeline` 在首次 CUDA
  使用前啟用 `torch.use_deterministic_algorithms(True)` 與
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`（涵蓋 fitting、scoring、generation）；啟用後
  gradient hash 實測同 process 與跨 process bit-identical，新增 regression test
  `test_enable_deterministic_gpu_sets_cublas_workspace_and_flag`。舊 discovery
  identity 因以 non-deterministic mode 計算而不可 verify，必須以 deterministic
  mode 重跑 discovery 重新 freeze。不修改任何 Draft 1 凍結值、estimator 或
  artifact schema。
- **Test run 1（第一次正式 full pipeline，2026-08-28）**：discovery
  （`outcome-flip-tech-discovery-det-20260828T015605Z`）→ calibration
  （`outcome-flip-tech-calibration-det-20260828T015659Z`，selection：L10–30、
  `evidence_item_end`、r=0.4，24 個候選組合全部 parse 100% 且雙方向有 flip）→ test
  （`outcome-flip-tech-test-20260828T051624Z`，11 個 held-out Technology tickers）。
  Verdict：**`success=false`**。Buy 方向（9 個 clean-Sell tickers）outcome 9/9、
  matched-random 0/9、label-permutation 0/9（reverse 9/9），p_one_sided=0.0015、
  Holm-adjusted 0.003，通過；sell 方向（僅 2 個 clean-Buy tickers）outcome 2/2、
  controls 0/2，但 n=2 時 exact paired test 的最小 p_one_sided 為 0.25，結構上
  無法通過 0.05 gate——失敗原因為 test split 的 sell 方向 power 不足，非 effect
  缺席。其餘 gates 全過：specificity CI [1,1]（雙方向）、min flip rate 100% ≥ 10%、
  net specificity 正、reverse not excess、controls < 10%、generation parse 100%、
  agreement 100%、safety max 0.402 ≤ 0.50。重要診斷發現：final-position control
  在 calibration 與 test 皆雙方向 100% flip（買 9/9、sell 2/2），效果可被
  answer-position injection 完全重現，position specificity 不成立；V2 的解釋上限為
  outcome-axis steering of the decision，非 evidence-position 因果。輔助診斷：
  per-(layer, position) attribution screen（
  `scripts/jspace_outcome_token_attribution.py`，
  `artifacts/qwen3.5-4b/jspace-outcome-token-attribution/`）顯示 evidence item-end
  的 per-token 一階敏感度集中在 L0–L14、L15–L18 斷崖，與 V2 band 大部分在斷崖下方
  一致；但 aggregate direction 的跨 ticker 對齊（pre_unit_norm）在 L15–L26 最高，
  故 band 選擇與 attribution 圖不矛盾。run-once 性質已消耗；若要以更大或更平衡的
  test split 重驗 sell 方向，依 experiment versioning 開新版本。
- **Direction decode 1（2026-08-28）**：新增輔助診斷 workflow
  `decode-outcome-direction`（dataset slug `jspace-outcome-direction-decode`）：以
  canonical Jacobian lens 把 frozen $d_l$ 運送到 final-layer basis，FP32 final norm +
  unembedding 解出 transported direction logit 與完整詞彙 softmax，逐層輸出 ±d 的
  compact top-k 與 buy/sell rank/margin；direction 依 V2 no-persistence protocol
  deterministic recompute 並驗證 direction identity，lens 只讀。不修改任何 Draft 1
  凍結值、V2 estimator、gate 或 artifact schema；第一次正式 run 見
  「Direction decode 結果」。
