# `tests/` scope

這個目錄放置 entity-bias experiment 的 regression tests；上層規則見 root
`AGENTS.md`。Counterfactual 線測試已隨程式移至 `archive/tests/`。

## 目前主要 regression tests

- `test_jspace_intervention.py` / `test_jspace_intervention_workflow.py`：
  coordinate swap/gain math、live dose diagnostics、dose-matched controls、
  hook cleanup 與 artifact schema 相容性。
- `test_jspace_activation_patching.py`：semantic span resolution、unequal span
  position mapping、hook-driven margin changes and cleanup、compact patching
  records with flips、frozen confirmation matrix 的 equal-ticker aggregation、
  bootstrap/exact/Holm gates、artifact/CLI 與 fail-closed completeness；全部使用
  deterministic fake model/tokenizer、無 GPU。
- `test_cross_sector_patching.py`：A V1 cross-sector header patching 的 deterministic
  preparation（balanced byte-matched pairing、smoke pair、unequal-count truncation）、
  target-complete header mapping、self-source no-op、compact pipeline 與 CLI；全部
  fake model、無 GPU。
- `test_context_overriding.py`：B V1 context overriding 的 negative-evidence filter、
  `instruction_context` 排除 final position、control/direction/span ID 唯一性、
  self-source no-op、equal-pair means 與 paired contrasts、pipeline 與 CLI；全部
  fake model、無 GPU。
- `test_context_readout.py`：C V1 L16 readout 的 frozen config（duplicate 處理、atomic
  寫入、tamper 拒絕）、average-before-top-k 與 span exclusion、family mass/rank
  compact serialization、artifact 不含 raw/full vectors 與 CLI；全部 fake model/lens、
  無 GPU。
- `test_entity_cell_e2.py`、`test_entity_cell_e3.py` 與同族 `test_entity_cell_*.py`
  （E1 V1/V2 pipeline、E2 donor contract、E3 record）：entity-cell workflow 的
  deterministic preparation、fail-closed gates、compact artifacts 與 CLI；
  `test_entity_cell_v3_fact_gate.py` 另覆蓋 E1 V3 fact-level amnesia stage
  （fake model 的 gold 生成/own/cross rows、classification 六態、
  gold verification 記錄與 tamper fail-closed）與 `verify-fact-gold` CLI；
  全部 fake model、無 GPU。
- `test_jspace_valence_readout.py`：valence pair 驗證/選擇/rendering、
  full-softmax average-before-top-k contract、contrast 分數 closed form、
  frozen candidate 候選規則與上限、fake model/lens 的 workflow 與 CLI 行為。
- `test_jspace_token_screen.py`：V1 token causal screen 的 config 驗證（對稱
  劑量、token_id 唯一、answer word、SHA）、fake model 的 dose math / scale
  floor / side invariance / matched-random 同範數 / hook cleanup、
  symmetric-slope 分析（equal-ticker 聚合、specificity、Holm、shortlist
  上限）、pipeline lifecycle（preflight 不建 run、SHA tamper 拒絕、
  determinism）與 CLI 行為。
- `test_jspace_outcome_flip.py`：V2 outcome-conditioned decision flip 的 config
  驗證、decision-position margin 計分位置語義、direction fitting（gradient
  確定性/ uphill / hook cleanup、label-permutation signs）、paired arms 的 dose
  matching 與 flip role、ticker-level estimands（exact paired test、strata、
  gates、calibration selection 與 fail-closed）、pipeline lifecycle（identity
  tamper 拒絕、missing input 拒絕、frozen combination 綁定）與 CLI 行為；
  全部用 fake model、無 GPU。同族的 `test_jspace_prior_probe.py` 覆蓋 V2
  zero-evidence header-only prior probe 的 frozen template、config schema、
  projection normalization、fail-closed direction verification、完整 lifecycle
  與 CLI（fake model、無 GPU）。
- `test_entity_to_dial_attribution.py`：entity-to-dial Phase D 的 per-position signed 梯度 hook（shape/position 選擇/strict single-fire/exception-safe 移除/frozen 與 requires-grad 雙路徑）、vs. `mlp_summed_derivatives` 的 sign-agreement 機件、`top_channel_stats`（Spearman argmax 同分最低 index、sector agreement、matched control seed 規則、degenerate 與 invalid input）、monkeypatched `run_phase_d` smoke pipeline（D1+D2 串行、manifest complete、no-op enforcement、final-layer structural-zero sign check、上游與 span 等長 fail-closed）；全部 fake model、無 GPU。同族的 `test_entity_to_dial_pipeline.py` / `test_entity_to_dial_analysis.py` / `test_entity_to_dial_block_patch.py` / `test_entity_to_dial_dial_probe.py` / `test_entity_to_dial_spans.py` 覆蓋 Phase A/B/C 的 pipeline、gate 統計、block transform、dial probe 與 token-group span。
- `test_entity_to_dial_joint.py`：entity-to-dial Phase E（`joint_patch.py`）的 joint/full/projected transform 算術（FP32 鏈、self-source bit-exact、full − joint == pre 差項）、PCA 機件（右奇異向量正交歸一/決定性、投影完整性與殘差正交）、`dial_value_capture`（完整 capture、fail-closed、hook 清理）與 `dial_channel_transplant`（position-restriction 以 pre-hook 觀察 down-proj 輸入、strict single-fire、空 delta 不註冊 hook）、`dial_footprint_direction`（權重欄向量）、`evaluate_gate_e` 的 R1 denominator rule（邊界 ≥ 0.2 納入、缺臂 fail-closed、falsifier 區間）與 monkeypatched `run_phase_e` smoke pipeline（E1+E2 串行、2B full-reference 用 fake model 重算後 in-run bit-exact 對上、reference 被污染 fail-closed、span 等長 fail-closed）；全部 fake model、無 GPU。
- `test_entity_to_dial_phase_f.py`：entity-to-dial Phase F 的 `directional_push_transform`（constant-across-positions、P 外 position 不變、零 push bit-exact no-op）、`make_projected_transplant`（與 E2 `project_delta` 路徑 bit-exact 等價、零 delta no-op、投影冪等性）、`dual_hook_interventions`（全 no-op bit-exact clean、strict single-fire、exception-safe hook 清理）、`load_pca_basis`（in-memory [d, k] 約定 + orthonormal/遞減/finite fail-closed）、`classify_f2_push` 四分類（含 |ΔM| = 0.05 邊界歸 jitter-band）、`evaluate_gate_f`（R1 分母規則、0.85 邊界、secondary 分組 median、consistency、F2 dose grid fail-closed）與 monkeypatched `run_phase_f` smoke pipeline（F1+F2 串行、fabricated e-01 以 fake model 重算 L15 reference 後 consistency 0.0、dual-hook 組成性質 bit-exact、anonymous 16/16 identity + m_anon 帶、anonymous header 損壞 fail-closed）；全部 fake model、無 GPU。
- `test_selective_intervention_subspace.py`：selective-intervention V1 的 `subspace_removal_transform`（α=0 回傳同 tensor 的 structural no-op、P 外 position bit-exact、pure in-subspace 分量於 α=1 精確移除、partial strength 縮放、bf16 dtype 保持、forward 形狀/寬度/position 範圍 guard、construction fail-closed：non-finite/negative α、空/重複 position、center keys 不匹配、non-orthonormal basis）、`random_orthonormal_basis`（orthonormal＋seed 決定性）、`tensor_sha256`、`nearest_position_mapping`/`align_grid`（等長 offset identity、degenerate fail-closed）與 `load_e01_basis`（roundtrip＋六種 fail-closed mutations）；純 unit、無 model。
- `test_selective_intervention_analysis.py`：V1 的 `group_gap`/`iqr`/`per_ticker_margins`/`decision_flips` 基礎指標與 `evaluate_gates` 的預先註冊門檻（G1a/G1b 0.5 縮放對、G2 25% 縮減邊界與 main-arm 零縮減時 not-evaluable（pass=null）、G3 0.15/G4 0.10 邊界、degenerate clean gap fail-closed）；純 unit、無 model。
- `test_selective_intervention_pipeline.py`：V1 monkeypatched `run_selective_intervention_v1` fake smoke pipeline（fabricated 2A 以 fake model 重算 64 margins 作 bit-exact 參照、fabricated e-01 16×32 basis、16 層 fake model L15=final layer、122 arm records 臂矩陣精確計數、clean 臂 bit-exact 對照、forward metadata digests、summary schema 過 core 序列化器保留字遞迴 guard、manifest 三階段 complete）與 fail-closed 路徑（corrupt archive margin → bit-exact 中斷＋manifest failed、non-orthonormal basis、缺 e-01 manifest、2A population 不足 64）；全部 fake model、無 GPU。
- `test_selective_intervention_m6_{analysis,manifest,pipeline}.py`：M6 外部 12 家 manifest 的 seed／產業分層／排除驗證、company-level IQR spread、paired bootstrap 三級判定、specificity／G3′／G4′與 generation diagnostics，以及以 fake model 重建 V1/V2 center digest、48 題 external arm matrix、V2 current-runtime provenance 與 compact summary 的 pipeline smoke；無大型 checkpoint、無 GPU。
- `test_jspace_outcome_geometry.py`：V2 outcome direction geometric projection
  的 dot projection 公式（general coefficient、normalization invariance、
  orthogonal/antiparallel、degenerate 與 invalid input）、fake model sector state
  累積對照 closed form、analyze compact metrics 與 TF-IDF 段、pipeline lifecycle
  （direction identity tamper fail-closed、missing input 與 bad binding 拒絕、
  TF-IDF arm 與 fake lens、logodds prototype 拒絕）與 CLI 行為；全部用 fake
  model、無 GPU。
- `test_jspace_outcome_decode.py`：V2 direction decode（Jacobian lens 解碼
  frozen $d_l$）的 fp32 tail（`fp32_next_token_logits` 與
  `fp32_next_token_log_probs` 一致性）、transport 反對稱與非保範數、
  per-(layer, sign) 記錄與 antisymmetry self-check fail-closed、answer token
  解析（multi-token / 跨 record 不一致 fail-closed）、band average-before-top-k
  聚合、pipeline lifecycle（identity/selection 綁定、band 候選集驗證、tamper
  拒絕、artifact 不含 raw payload 掃描）與 CLI 行為；全部用 fake
  model/lens、無 GPU。
- `test_investment_dial.py` / `test_mlp_addition.py`：JSON 格式、帶符號導數聚合、
  全 token 加法、A/B 校準、獨立公司隔離、來源雜湊與失敗清理；另以小型隨機
  Qwen3.5 混合層模型測有限差分與 cached generation，不載入大型 checkpoint。
- `test_baseline_trial_pipeline.py`：baseline trial workflow 行為。
- `test_span_sensitivity.py`：header condition rendering、paired estimands、artifact lifecycle
  與 CLI。
- `test_prompt_input.py`、`test_core_inference.py`、`test_core_analysis.py`、
  `test_core_artifacts.py`：shared workflow 核心子套件。
- `test_entity_concept_materials.py`、`test_entity_concept_directions.py`：Phase 1
  基礎工具的材料驗證、等家族權重方向 fitting 與退化處理；無模型、無 GPU。
- `test_entity_concept_development.py`：Phase 1 V2 Slice 2 的 caller-owned development runner；使用至少 17 層 deterministic fake model、真實 core residual recording、compact artifacts、random reproducibility、preflight rejection、failed manifest 與 hook cleanup；不載 checkpoint、不需 GPU。
- `test_continuation_scoring.py`：buy/sell margin 與 KL 計算。
- `test_lens_artifacts.py`、`test_lens_install.py`、`test_lens_loader.py`、
  `test_lens_promotion.py`、`test_lens_registry.py`：canonical lens 安裝、存取與
  promotion 規則。
- `test_prompt_readout.py`、`test_attribution_validation.py`、
  `test_generation_artifact.py`：prompt-analysis readout、generation 與 attribution
  artifact contracts。
- `test_workflow_boundaries.py`：root guidance contract、CLI 與 experiment import 邊界。

這份清單按 workflow 選列；新增或刪除 test file 時不要求在此枚舉整個目錄，但新的
獨立 workflow 必須加入一行入口。

## Instruction Index

目前 `tests/` 沒有含 `AGENTS.md` 的直接子目錄。Integration/GPU tests 若日後移入獨立
子目錄並形成專屬 fixture/runtime 規則，再新增局部 `AGENTS.md` 並更新本節。

## 新增測試原則

- 純函式與資料格式優先使用 deterministic unit tests、fake model、monkeypatch
  與 temporary directories；不要為了單元測試載入大 checkpoint
  （例如 Qwen3.5-4B 約 8.7GB BF16）。
- 任何 patch hook 或輸出 schema 改動，都要補上 regression test。
- 模型載入與 GPU inference 屬於 smoke/integration test，應明確標示，避免讓
  `uv run pytest -q` 依賴下載或特定 GPU。
- `tests/conftest.py` 的 autouse fixture 在 unit-test suite 將
  `torch.cuda.is_available()` 設為 false，避免真實 GPU 主機上的 deterministic flag
  汙染後續測試。測試 CUDA-available 分支時，在該 test 內明確 monkeypatch 為 true。

## 局部驗證

```bash
uv run pytest -q tests/<target_test_file>.py
```

修改 shared workflow contract 或 import boundary 時，另執行
`uv run pytest -q tests/test_workflow_boundaries.py`。
