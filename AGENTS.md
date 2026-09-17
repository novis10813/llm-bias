# Repository guidance

這份文件是 repository 的 AI 協作入口。先讀本檔；工作落在子目錄時，再讀該目錄的
`AGENTS.md`。若祖先目錄有多份 instruction files，越靠近目標檔案者優先；下層只
補充局部規則，未覆蓋的規則繼續沿用上層。詳細分工與維護方式見
[Documentation and instruction system](docs/documentation-system.md)。

## Repository Scope

本 repo 研究 decoder LLM 的 entity-sensitive 與 sector-sensitive representation、
Jacobian-lens transported readout，以及 residual/J-space intervention 對固定答案分布
的影響。Active experiments 使用 `data/baseline/`，目前包含 baseline trial、prompt
analysis、span sensitivity 與 J-space intervention。

本 repo 不是 production trading system，也不把 lens readout 當成 chain-of-thought、
離散 reasoning path 或單獨的 causal proof。Counterfactual、synthetic 與 10-K 線已
frozen；還原方式見 [`archive/README.md`](archive/README.md)。

研究目錄、狀態與實驗前後關係先見 [研究總覽](docs/README.md)。各實驗頂層只保留
`proposal.md` 與 `report.md`；版本／階段協議與中間紀錄放 `details/`，原始協議仍為
對應版本的 source of truth。編排規則見 [Documentation system](docs/documentation-system.md)。

全線入口：[J-space token V1/V2](docs/jspace-token-experiments/proposal.md)、
[Entity Cell](docs/entity-cell-localization/proposal.md)、[Investment-dial](docs/investment-dial/proposal.md)、
[Balanced Evidence Gap](docs/balanced-evidence-gap/proposal.md)、[Entity-to-Dial](docs/entity-to-dial/proposal.md)、
[Evidence-insensitivity（S&P 500 證據不敏感性與 entity prior）](docs/evidence-insensitivity/proposal.md)（Phase 1 **completed** 2026-09-16：formal run `phase1-gpu-bf16-01` 5336 forwards、G-P1..P4 全過、503 家分組表 frozen——50 evidence-responsive / 453 fixed-sell、503/503 零證據 sell；order-swap 臂量出強 recency 效應（正項在尾 69–90% sell→buy）；[報告](docs/evidence-insensitivity/details/report-phase1.md)；[cross-model 診斷](docs/evidence-insensitivity/details/diagnostic-cross-model-probe.md)：零證據 sell 預設為模板底層、證據反應結構為模型特性；Phase 2 **completed** 2026-09-17（雙模型 Qwen3.5-4B＋Gemma-4-E2B；[報告](docs/evidence-insensitivity/details/report-phase2.md)：capture-layer 狀態不承載行為分組——offset/gain 組差皆 null、Qwen gain 差顯著但反向且極小、狀態反應形狀為模型特性；Gemma G-2A/G-2B fail 帶保留）；Phase 3 **completed** 2026-09-17（雙模型 `phase3-gpu-bf16-01`，gates 全過；[報告](docs/evidence-insensitivity/details/report-phase3.md)：span 位置全域 null、極性 contrast 在 final-position 狀態（capture 層附近寫入）、S-vs-R 皆 readout-mediated 且組差方向跨模型相反、0 flip、Gemma 特有 margin/決策端點分離；線內第一個干預臂））、
[公司身分的中間概念](docs/entity-concept-decision/proposal.md)（收線 2026-09-15：[收線報告](docs/entity-concept-decision/report.md)。可分離公司概念前提被多輪真實模型開發否決（L15 k=8 與 L8–L26、4/16 家一致 null）；研究問題曾轉向 stance 通道並取得收斂描述——1D stance 方向 L15 解釋 margin 變異 60%（≠ entity signal、因果探查為弱槓桿）、軸分解顯示 L15 狀態 99.5% 共享、決策是微小（0.5%）公司間差的高增益非線性讀出，不存在低維局部線性因果軸。原三階段 pipeline 未實作／formal 未授權）。
以下特定版本連結保留直達原始協議／證據，方便查核：

- [Baseline trial proposal](docs/baseline-trial/proposal.md) / [prompt-analysis reproducibility report](docs/baseline-trial/report.md)
- [Qwen Jacobian-lens selection proposal](docs/jacobian-lens-selection/proposal.md) / [selection report](docs/jacobian-lens-selection/report.md)
- [Interactive prompt-lens dashboard](docs/interactive-prompt-lens-dashboard.md)
- [J-space sector intervention proposal](docs/jspace-sector-intervention/proposal.md) / [held-out report](docs/jspace-sector-intervention/report.md)
- [J-space valence vocabulary readout proposal](docs/jspace-valence-readout/proposal.md) / [Technology discovery report](docs/jspace-valence-readout/report.md)
- [J-space token experiment versions](docs/jspace-token-experiments/proposal.md)
  - [V1 proposal](docs/jspace-token-experiments/details/proposal-v1.md) / [report](docs/jspace-token-experiments/details/report-v1.md)（completed；shortlist empty）
  - [V2 proposal](docs/jspace-token-experiments/details/proposal-v2.md) / [report](docs/jspace-token-experiments/report.md)（Draft 1 已實作；第一次正式 pipeline 的 verdict 為 `success=false`；另含 zero-evidence header-only prior probe 與 direction decode，兩者皆有 first formal run）
  - [V2 outcome direction 幾何投影分解](docs/jspace-token-experiments/details/report-v2-geometry.md)（輔助診斷；描述性幾何，非 causal）
- [Activation patching proposal](docs/activation-patching-causal-tracing/proposal.md) / [report](docs/activation-patching-causal-tracing/report.md)（Draft 1 completed；discovery 找到 evidence→instruction context→final position 的 layer-dependent sufficiency 轉移；unchanged held-out confirmation `success=true`）
- [Sector and context follow-up proposal](docs/sector-context-followup/proposal.md) / [discovery report](docs/sector-context-followup/details/report-discovery.md) / [B V1 confirmation report](docs/sector-context-followup/report.md)（A cross-sector header-state patching、B negative-evidence context overriding、C L16 instruction-context readout；A/B/C discovery 完成；B V1 calibration `success=false`，same-sector peer specificity gate 未通過，held-out test 未執行）
- [Shared experiment core](docs/shared-experiment-core.md)
- [Technology header-span sensitivity proposal](docs/span-sensitivity/proposal.md) / [discovery report](docs/span-sensitivity/report.md)（V1 discovery completed；`same_sector_swap` 為 calibration primary condition，calibration/test protocol 未凍結）
- [Entity cell localization and downstream attribution versions](docs/entity-cell-localization/proposal.md)（[proposal V1](docs/entity-cell-localization/details/proposal-v1.md) / [report V1](docs/entity-cell-localization/details/report-v1.md)；[proposal V2](docs/entity-cell-localization/details/proposal-v2.md) / [report V2](docs/entity-cell-localization/details/report-v2.md)；[proposal V3](docs/entity-cell-localization/details/proposal-v3.md) frozen（4 entity cells，calibration + hold-out 雙驗證）；[收線報告](docs/entity-cell-localization/report.md)；[E4 proposal](docs/entity-cell-localization/details/proposal-e4.md) proposed（residual stream 壓抑 readout 比較，[報告](docs/entity-cell-localization/details/report-e4-readout-delta.md)））
- [財務穩健判斷相關神經元定位](docs/financial-soundness-localization/proposal.md) / [因果驗證](docs/financial-soundness-causal-validation/proposal.md)（探索版 V1；formal run 尚未授權）
- [Balanced Evidence Gap（entity-induced decision gap 行為確認）](docs/balanced-evidence-gap/details/proposal-phase1.md) / [Phase 1 報告](docs/balanced-evidence-gap/details/report-phase1.md)（Phase 1 completed；三判準全過）/ [Phase 2 proposal](docs/balanced-evidence-gap/details/proposal-phase2.md)（Rev 1）/ [Phase 2A 報告](docs/balanced-evidence-gap/details/report-phase2a.md)（2A run `phase2a-gpu-bf16-01` completed；Rev 1 gate 2A fail：Spearman vs Phase 1 named margin ρ = −0.411；64/64 sell；H4 descriptive：dial 非 entity-specific）/ [Phase 2 Rev 2 proposal](docs/balanced-evidence-gap/details/proposal-phase2-rev2.md)（gate 2A 參照改 Phase 1 gap ＋ group construct check；run `phase2a-rev2-gate-01` 5 項全過，2B 獲授權）/ [Phase 2 報告](docs/balanced-evidence-gap/details/report-phase2.md)（2B `phase2b-gpu-bf16-01`：entity span L0–11 承載、L12–15 handoff、instruction 峰值 L15；2C `phase2c-gpu-bf16-05`＋ gate 重評 `phase2c-gate-reanalysis-01`：MLP arm pass（L19/L20/L26 top neuron 勝過 controls，Holm p=0.0418）、attention arm null；gate 2C pass。Phase 2 completed，discovery 性質）/ [Phase 3 proposal](docs/balanced-evidence-gap/details/proposal-phase3.md)（Rev 2 frozen：2C 三 coordinate 的 causal validation；Rev 2 將 2C 重導驗證容差校準至實測 bf16 jitter 帶 0.05）/ [Phase 3 報告](docs/balanced-evidence-gap/details/report-phase3.md)（formal run `phase3-gpu-bf16-02` completed；**gate 3A fail（0/3 confirmed）**：三 coordinate 的 additive 介入效應 ≤ matched controls、零 flip；L19 極性符合 2C 預測、L26 相反、L20 無；dial 對照 ±1.0 nats。Phase 3 以 null 收線）/ [收線報告](docs/balanced-evidence-gap/report.md)（線已正式收線：行為確認通過、表徵層帶定位明確、晚期單神經元因果收口宣告虛無）/ [J-lens 結構診斷](docs/balanced-evidence-gap/details/diagnostic-jlens-neurons.md)（收線後輔助診斷：三 entity 通道＋dial 的 transported direction readout；entity 通道無可辨識結構簽章、與隨機 control 不可區分，與 Phase 3 null 一致）
- [Entity-to-Dial 路徑解剖（entity signal 從 L0–11 承載帶到 L15/N8490 dial 座標的因果路徑）](docs/entity-to-dial/details/proposal-phase-abc.md) / [discovery 報告](docs/entity-to-dial/details/report-phase-abc.md)（Rev 1.3；Phase A/B/C formal run 完成，Gate A1/B/C 皆 fail；entity signal 在 L12 後不走 entity position 也不走 dial，主體路徑在 instruction span）/ [Phase D proposal](docs/entity-to-dial/details/proposal-phase-d.md)（Rev 1.3：formal run `entity-to-dial-d-02` 完成，2026-09-11；Gate D fail——D1 無合格層，raw 效應一致 sell 方向（erasure 簽章在 block 層級重現）、toward-source 8 方向均值 ≈ 0；D2 描述統計：19/19 非 final 層 top channel |ρ|（0.95–0.99）勝過 matched controls 但 sector agreement 多數 0/4。首次 formal run d-01 的 D2 因 rows 只含 4 家無效（Rev 1.2））/ [Phase E proposal](docs/entity-to-dial/details/proposal-phase-e.md)（Rev 1.3，completed：formal run `entity-to-dial-e-01` 完成，2026-09-12；**Gate E1 pass**（L15 joint ratio median 0.575，H_joint 勝出 H_carryover）、**Gate E2b fail**（dial ratio 0.434，未 falsified）；E2a：k=8 效果比值 0.983，state difference 低維集中；full arm 與 2B 存檔 56/56 bit-exact；事後幾何：cos(v₁, dial footprint) = −0.020，v₁ 為分散式 residual 方向（max channel |cos| 0.298），per-direction 顯示 stance transfer asymmetry（bottom→top 強、top→bottom 弱/反向，sell attractor 待確認））/ [Phase F proposal](docs/entity-to-dial/details/proposal-phase-f.md)（Rev 1.4，completed：formal run `entity-to-dial-f-02` 完成，2026-09-13，**Gate F1 fail**（additivity ratio median 0.7329 < 0.85；4 個 bottom→top 方向 additive residual 全負 −0.49 至 −0.55，v₁ 與 dial 幾何獨立但下游匯流/飽和）；F2 兩臂皆 context_dependent_or_null（8 判定點全在 ±0.05 jitter 帶內，v₁ 無可偵測本征 signed loading，stance 軸描述降級為 state-difference 主方向）；consistency vs e-01 diff 0.0；依 §7 fallback 以 k=8 子空間（0.983）作 L15 段完整描述收線。首次 formal f-01 因 summary key 觸發 core 保留字 fail-closed（保留為偏差記錄，key 更名 interaction_delta_m））/ [收線報告](docs/entity-to-dial/report.md)（線已正式收線：L15 機制定位完成，雙通道緊湊表示否決，採納 k=8 殘差子空間 0.983 完整描述）
- [Investment-dial（Park et al. 2026 單神經元 investment-bias dial 之方法復現線）](docs/investment-dial/proposal.md)（V1 exploratory calibration、[fine-a diagnostic](docs/investment-dial/details/diagnostic-fine-a.md) / [報告](docs/investment-dial/details/report-fine-a.md)、[V2 proposal](docs/investment-dial/details/proposal-v2.md) / [報告](docs/investment-dial/details/report-v2.md)；V2 formal run gate `pass`；[收線報告](docs/investment-dial/report.md)——線已收線，非 numeric replication）
- [Selective-intervention（M5 entity-specific selective intervention：推論期 entity-bias 控制）](docs/selective-intervention/proposal.md)（[V1 proposal](docs/selective-intervention/details/proposal-v1.md) frozen Rev 1.5 / [V1 報告](docs/selective-intervention/report.md) completed：L15 k=8 entity-difference subspace removal、dose grid {0.25, 0.5, 0.75, 1.0} × 64 prompts；formal `selective-intervention-v1-gpu-bf16-01` gate `fail`：G1a/G1b/G2 pass（group gap 1.2697→0.5863、IQR 半減、random 對照 0.49% 縮減）、G3/G4 fail（full-strength mean shift +0.330、anon shift −0.269）→ 依 frozen 決策表 full-strength 負結果；efficacy＋specificity 陽性、L15 層特異性成立、dial 通道不受損）
- [Entity-bias proposal and roadmap](docs/proposal/README.md)
- [Artifact identity and run manifest contract](docs/artifact-contract.md)

J-space evaluation 位於 [`docs/j-space-evaluation/proposal.md`](docs/j-space-evaluation/proposal.md)。它是
optional、proposed、non-runnable auxiliary preflight，只評估 synthetic task-local
J-space-candidate evidence；它不建立 global workspace 結論，也不 gate active
experiment milestones。可執行的 sector intervention 位於
`llm_bias/jspace_intervention/`（CLI `jspace-intervention`），與該 synthetic preflight
及 archived entity-only patching protocol 分開。

## Shared experiment workflow contract

The shared experiment workflow is `prepare → forward → analyze → finalize`. Reuse the four core subpackages—`llm_bias/core/prompt_input`, `llm_bias/core/inference`, `llm_bias/core/analysis`, and `llm_bias/core/artifacts`—for cross-experiment workflow mechanics. Experiment packages must not sink shared prompt preparation, model forward execution, common analysis, artifact serialization, manifest/provenance, or lifecycle finalization into local copies; keep research-specific semantics and presentation in the owning experiment package. Compatibility rules are mandatory: preserve existing public CLI/API behavior and artifact schemas unless a canonical workflow document explicitly versions a change; experiment packages (`baseline_trial`, `jspace_intervention`, `prompt_analysis`, `span_sensitivity`) must not import each other, and shared infrastructure must not import any experiment package. Experiment workflows consume an existing validated canonical lens and must not fit, mutate, or replace one implicitly. Never persist raw activations, residuals, hidden states, gradients, Jacobians, or KV caches; emit only compact derived outputs with provenance.

目前 `baseline_trial` 仍直接重用部分 `prompt_analysis` modules，屬於待收斂的 legacy
compatibility exception；不要新增同類依賴。新 shared mechanics 必須放進 `core/`。

## Research semantic boundaries

- 不保存完整 raw activations；只輸出 compact top-k、rank、統計量、token IDs/text、probabilities 與 provenance。
- 不要把不同 token 的 top-1 probability 差直接當成 causal effect；使用固定答案 token probability、logit margin 或明確定義的 normalized transfer。
- prompt readout 的 aggregate 必須先平均每個 condition 的完整 vocabulary softmax，再選 top-k。Attribution 是 local first-order sensitivity，不是 attention map 或 standalone causal claim。
- Jacobian lens 是 transported representation readout，不是 chain-of-thought、離散 reasoning path 或 standalone causal evidence。
- Counterfactual 線的 Pair/span-mapping/control-patch/bias-specific pair 研究語義隨程式一併移至 [`archive/README.md`](archive/README.md)。

## 設定與檔案放置

- Python 3.13 與 workspace 依賴由 `.python-version`、`pyproject.toml`、`uv.lock`
  定義；使用 `uv sync` 建環境，新增套件使用 `uv add`。
- Pinned lens registry 放在 `config/pretrained_lenses.json`；修改 model identity、revision
  或 SHA-256 時，依 [Qwen Jacobian-lens selection](docs/jacobian-lens-selection/proposal.md)
  重新驗證。
- 可追蹤的詳細政策與 workflow 放 `docs/`；script ownership map 見
  [Research scripts reference](docs/research-scripts.md)。
- Input/provenance 放 `data/`；模型與 Hugging Face cache 放 `.cache/`；run outputs 與
  lens artifacts 放 `artifacts/`。這些大型或 generated 內容遵守 `.gitignore`，不要
  加入 root Git。
- `.pi/` 保存本地 agent runtime，`graphify-out/` 保存 generated repository diagrams，
  `.worktrees/` 保存本地 Git worktrees；它們不是 main source tree，也不要加入 root Git。
- `third_party/jacobian-lens` 與 `third_party/jspace-viz` 是 editable workspace members，但整個 `third_party/` 被 `.gitignore` 忽略。
- 新環境請依照 `README.md` clone 兩個外部 repo 後再執行 `uv sync`。
- 每個 model 只有一個 active、完整逐層的 canonical lens：
  `artifacts/<model-slug>/jacobian-lens/jacobian_lens.pt`。一般 partial/stride fitting
  checkpoint 放 `artifacts/archive/<model-slug>/jacobian-lens/checkpoints/`；受控
  candidate-selection workflow 可依
  [Qwen Jacobian-lens selection](docs/jacobian-lens-selection/proposal.md) 使用
  `artifacts/<model-slug>/jacobian-lens/candidates/` 的 candidate-adjacent digest
  checkpoints，但不得把 candidate 當 active lens。

## Instruction Index

以下只列 root 直接子目錄中的 instruction files：

- [`archive/AGENTS.md`](archive/AGENTS.md)：frozen code、還原邊界與 archive 內入口。
- [`docs/AGENTS.md`](docs/AGENTS.md)：canonical 文件分類、引用與狀態維護。
- [`llm_bias/AGENTS.md`](llm_bias/AGENTS.md)：active Python packages 的 ownership 與局部驗證。
- [`scripts/AGENTS.md`](scripts/AGENTS.md)：research operators、diagnostics 與 renderers 的慣例。
- [`tests/AGENTS.md`](tests/AGENTS.md)：regression test 地圖與 fake-model 測試規則。

目前 `config/` 與 `data/` 是未來候選：registry schema 或 dataset-specific provenance
規則變得無法用一兩句覆蓋時，再在該目錄新增 `AGENTS.md`。其他目錄也採同一門檻。
新增後，只更新最近一層祖先 `AGENTS.md` 的 Instruction Index，不在 root 枚舉更深層
檔案。

## Working Rules

- 維持既有 package ownership、public CLI/API 與 artifact schema；需要版本變更時，
  先更新對應 canonical workflow 文件。
- 只改任務要求的範圍，不做順手重構，也不覆蓋不相干的 dirty changes。開始前先看
  `git status` 與相關 diff。
- 文件中的命令、path、run 狀態與架構描述要能對上 code、config、tests 或 artifact
  provenance；不確定的內容標成 proposed 或 note。
- 實驗術語的定義必須在文件中：提及概念時只使用 repo 文件已定義的英文術語（原文
  照用）；沒有術語的概念用完整中文描述句。需要為新概念命名時，先將定義寫入 owning
  experiment 的 canonical workflow 文件再使用；不得引入任何文件中查無出處的英文複合詞，
  也不得在回覆中直接使用未定義的新詞。
- 新增詳細規則時先更新 `docs/`，再讓 `AGENTS.md` 連結該文件，避免兩處維護完整副本。
- 討論 J-space token 實驗時必須標明 V1 或 V2。若 direction source、primary outcome、
  controls 或 gate 改變，依
  [experiment versioning](docs/documentation-system.md#experiment-versioning) 建新版本，
  不把新設計回填成舊版本結果。

## Verification

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
node --check llm_bias/static/prompt_readout.js
node --check llm_bias/static/attribution_dashboard.js
```

測試應優先使用 deterministic unit tests、fake model、monkeypatch 與 temporary directories；不要為一般 unit test 載入大型 checkpoint。模型/GPU inference 應明確視為 smoke 或 integration test。
