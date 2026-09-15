# A 方案的共通契約與實作驗證（Draft 1）

**proposed／non-runnable。**本附件規範 [Phase 1](proposal-phase1.md)、[Phase 2](proposal-phase2.md)、[Phase 3](proposal-phase3.md)；三者分別凍結與批准，不合併為單一執行協議。以下編號是未來實作需求，不是現在授權的程式變更。

## 1. 方法來源與適應性修改

| 方法來源 | 原始設定 | 本線修改及風險 |
|---|---|---|
| Kim et al. (2018), [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://proceedings.mlr.press/v80/kim18d.html) | 以概念範例建立向量，用方向導數衡量分類敏感度；主要展示影像分類 | 以成對文字的平均狀態差作候選方向，限制到固定 L15 k=8，另做實際置換；不是 TCAV 演算法或數值復現。文字差可能混入語法、立場或語域 |
| Zhang & Nanda, [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042) | 比較 patching 的量測與 corruption 選擇如何影響定位 | 用同證據跨公司自然狀態差、固定答案 margin、雙方向對照；混合狀態可能不自然，不直接套用自然中介效應的識別假設 |
| Repo Entity-to-Dial E/F、Selective-intervention V1 | 固定配對的低維投影、移除與雙 hook | 增加獨立概念定義及早層置換後的中間分量恢復；不得把幾何正交當因果獨立 |

閱讀文獻不等於驗證本方法；本線不宣稱完成文獻全面回顧。

## 2. 已核實的程式與產物可重用，但有邊界

### 2.1 固定輸入與 shape

路徑以 repository root 為基準：

| 輸入 | 欄位／契約 | 核對結果 |
|---|---|---|
| `artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01/analyze/summary.json` | `pca_basis_vectors: list[list[float]]` [16,2560]；`pca_singular_values: list[float]` [16]；只取前 8 行，轉成列基底 Q∈R^(2560×8) | 檔案存在；SHA-256 `ce3c3a9cd443358894691a9681f04f6c1218a840724ebfb2bc98d529e3f5e358` |
| `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01/prepare/prompts.jsonl` | 64 rows；`id,ticker,name,sector,prompt,formatted` strings，`prompt_ids` integer list，`reverse` bool，`order` int，`entity_span,evidence_span,instruction_span` 兩個整數端點，`entity_position,final_position` int | SHA-256 `713e393a1da0c157f7019e672d3496f4c15a4bfe7793a39ca2f162f0b6f7647f` |
| 同一 2A run 的 `forward/results.jsonl` 及兩個上游 `manifest.json` | 上游完成狀態、margin 參照與登記的 output refs | 執行前須另核 SHA、schema、counts；不得只相信本頁 digest |
| `config/pretrained_lenses.json` 及 `artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt` | 完整逐層 canonical lens、metadata、model/revision/hash | 由既有 registry validator 驗證；不 fitting、不替換 |

2A 實際 16 家：ABT、AMAT、AXP、BDX、BLK、C、CSX、DE、DHR、GLW、GS、HON、HPE、IT、NSC、SYK。E 的 8 方向來自 BDX/IT 與 BLK/NSC 的雙向配對；k sweep 為 {1,3,8,16}。不得把 64 prompts 當 64 獨立公司。

保存的 16 個 singular values 不是完整 spectrum；其平方比只能描述「已保存 16 維內」的比例。基底向量已保存，但逐 prompt 狀態、逐位置係數及中心未保存；新方向介入或語義量測必須另做 forward。只憑 basis 的正負詞彙不能恢復公司在其上的實際 loading。

### 2.2 Ownership 與最小新增範圍

| 已有位置 | 可重用部分 | 尚需新增／驗證 |
|---|---|---|
| `llm_bias/core/inference/interventions.py` | `residual_interventions(model, transforms)` 已支援多層字典；`record_block_states` 可取 transient pre/mid/post 狀態 | 不新增整套 hook engine；`record_block_states` 自行執行一次 forward，不是 live capture context。clean／U-only 參照各用獨立 recording forward；組合 hook 讀當下 output 與 caller 傳入的 transient 參照。補 prevalidation、single-fire、部分註冊失敗清理與 capture 順序測試 |
| `llm_bias/core/inference/forward.py`、`core/continuation_scoring.py` | `record_residuals`、`fp32_next_token_log_probs` | decoder bf16，只有 final norm/unembedding/log-prob 為 FP32；不是整個 forward FP32 |
| `llm_bias/core/analysis/transport.py`、`distributions.py`、`records.py` | transport、完整 vocabulary 分布平均、top-k records | 真實狀態 readout 與單獨方向的診斷必須分欄標記 |
| `llm_bias/entity_to_dial/joint_patch.py`、`llm_bias/selective_intervention/subspace.py` | 數學與載入器可作參照 | 新 package 不 import 舊實驗；只在確有跨實驗需求時提取共通投影／basis 驗證到 core，舊 API 保留相容 wrapper |
| `llm_bias/core/prompt_input/`、`core/analysis/statistics.py`、`core/artifacts/` | tokenizer/span、統計、SHA、lifecycle | 實驗定義與 gate 留在新 owning package |

未來 owning package 建議 `llm_bias/entity_concept_decision/`，此路徑目前**不存在**。新增檔邊界建議：`concepts.py`（Phase 1 材料與概念定義）、`intervention.py`（Phase 2 研究操作）、`path.py`（Phase 3 組合與判據）、各階段獨立 runner；shared mechanics 不落入 local copies。實作 spec 須在每階段凍結後另寫，不把這張表當可平行 batch。

## 3. 資料分割不洩漏，但不能靠程式保證語義

1. 舊 16 公司及所有已見模板只作 discovery/calibration，不能稱全新 held-out。`split_manifest` 須記 `company_id, template_family_id, paraphrase_group_id, split, prior_exposure`。
2. 新公司依事先列定的非 outcome 條件選取；列出在 repo 既有研究中的使用紀錄。不能看新公司 margin 後挑極端公司，再稱不受選擇影響。
3. Phase 1 的概念 fitting、語義 validation、最終 audit 材料家族互斥；同一句改寫不可跨 split。概念正負號只由人工定義決定，不按 margin 翻號。
4. Phase 2 development 可用舊公司；confirmation 須保留公司及整個模板家族。分開報告新公司／舊模板、舊公司／新模板、兩者皆新，不能以 pooled pass 掩蓋某格失敗。
5. Phase 3 的上游層選擇不能看 Phase 3 confirmation。Phase 2 已看過的公司若重用，只能稱路徑 discovery；需另留公司或在 Phase 2 前共同凍結兩階段確認集。
6. 不把概念標籤、解釋或候選排序注入決策 prompt。概念材料當然可以表達概念；不能用「所有概念詞均不得出現」的字串檢查代替人工審查。固定證據自然提及相關概念也不自動構成洩漏。

## 4. 位置、數值與防禦性契約

- L15 指 **0-based post-block**；模型 hidden dimension 必須由 config 驗證為 2560，不能誤用 MLP 9216 或其他模型 4096。只有 residual hook，不假設所有層都有普通 attention heads。
- instruction token interval 使用半開區間 [start,end)；主要比較要求對應 token IDs 完全一致，採相對 offset 一一映射。改寫模板只在同模板公司對內配對，不跨模板硬對齊。不得以 nearest mapping 掩蓋長度／token 不相等。
- entity、evidence、instruction、answer prefix 與其餘位置以原子 token partition 驗證無重疊且覆蓋整個序列；介入只取 owning span。跨邊界 token 無法唯一指派時中止，不能默默截斷。
- final prefix 與 buy/sell continuation 需 tokenizer 實測為不同 single token；使用共同 scorer，先分別轉 Python float 再相減以保持舊計分契約。
- 投影 FP32、只在最後 cast 回 bf16。QᵀQ 的載入容差 1e-6；維度、finite、正交、source/target 映射必驗。近零概念方向、零 random norm、缺公司／缺對照、未授權增補樣本一律 fail-closed 或標 `inconclusive`，不得替換方向。
- `alpha=0`／self-source 採 structural no-op，要求 bit-exact；真正的投影減去再加回只要求已校準容差，不要求 bf16 代數可逆。
- 每個新階段至少以真實 tokenizer/model 跑一個**完整公司 pair，兩方向**的端到端 smoke，涵蓋每個 hook、control、analyze、finalize；Phase 1 加一組概念 pair。這比 repo 的至少一筆要求更嚴格。smoke 不是正式證據。

### 4.1 容差需要獨立 calibration，不能沿用 0.05 當萬用答案

令 J_M 為獨立 calibration 上重複 clean、等價計分路徑與可逆對照的最大 margin 誤差；候選數值容差 τ_M=max(0.05 nats, 2J_M)。0.05 是舊 bf16 經驗底限，不是本線已驗證值。概念係數容差 τ_a=max(1e-6,2J_a)，J_a 同法量測，單位不同不能套 margin 帶。

在獨立 calibration 前先定可接受上限 τ_M,max、τ_a,max；超出即中止修工程／精度，不得靠擴大容差過 gate。最小科學效應 δ_M 必須 >2τ_M，δ_a>2τ_a；若合理效應低於可辨識精度，結果為不足以判定，不默默換 scorer。所有容差與實質門檻均在 confirmation 前固定。

## 5. Compact artifacts 與 CLI 不可虛構完成

共通輸入還包括 config JSON：`schema_version:int, phase:int, protocol_version:str, model_ref, tokenizer_ref, lens_ref, upstream_refs`，各 ref 含 path/hash；`splits_ref, candidates_ref, thresholds, seeds, arm_registry, approved_run_mode`。未填 freeze blockers 禁止 formal。

建議 run root `artifacts/qwen3.5-4b/entity-concept-decision-phase<N>/runs/<run-id>/`，由 core path helpers 產生，不手工混用 slug。各階段：

- `prepare/metadata.json`：prompt/config/material hashes、count、span 與 token ID、介入公式版本、arm registry、split lineage。
- `forward/records.jsonl`：`id, pair_id, company_id, template_id, split, arm, layer, concept_id, seed`，`margin, concept_score, patch_norm, raw_delta_m, toward_source_delta_m` 等 compact finite 數值，適用欄位依 stage schema 固定。未定義 ratio 用 null 加 `reason`，不用 NaN；字詞輸出為 top-k/rank/probability。
- `analyze/summary.json`：每 gate 的 eligible count、效果、CI、調整後 p、status；`complete` 與科學 `pass/fail/inconclusive` 分開。
- 候選檔只保存人工定義、最多 3 個 2560 維聚合方向、8 維座標、量測門檻、hash 與 split provenance。依既有 compact basis 先例明列 schema；不保存逐句或逐位置全部投影係數、raw states、中心、KV caches、gradient/Jacobian。
- 原始高維 states 僅在 RAM、同一必要階段內使用，做完即釋放。不修改上游 manifest。

遵守 [artifact contract](../../artifact-contract.md)。任何 JSON keys 必須通過 core reserved-key guard；例如 `interaction_delta_m`，不沿用含保留字的名稱。不能只用「無 raw」旗標規避檢查。

**CLI 尚未實作。**規劃每階段一個專屬 subcommand，但目前不刊登可執行命令；實作時需把 parser、help、config schema、對應 phase 協議 1:1 補齊並凍結。測試指令見下，不是實驗執行指令。

## 6. 實作順序與三類驗證各自回答不同問題

### 6.1 現有 regression（命令存在）

```bash
uv run pytest -q tests/test_workflow_boundaries.py tests/test_core_artifacts.py tests/test_continuation_scoring.py
uv run pytest -q tests/test_selective_intervention_subspace.py tests/test_selective_intervention_analysis.py tests/test_selective_intervention_pipeline.py
uv run pytest -q tests/test_entity_to_dial_joint.py tests/test_entity_to_dial_phase_f.py
uv run pytest -q tests/test_lens_artifacts.py tests/test_lens_registry.py
```

### 6.2 必須新增的 deterministic tests（以下是規格，尚無檔案）

1. **投影**：P_C²=P_C、P_K-P_C 正交剩餘、基底 sign/rotation 不改 projector；scale、dtype、非目標位置不變；full replacement 與分解重建 FP32 誤差界。
2. **動態恢復**：用 hook 當下 h 計算 `(h_t-h)P_C`，不能用預先固定的 clean source-target delta 假裝恢復；兩層 hook 先後、單次 firing、前後 capture、所有 exit path 清理及 invalid second layer 的部分註冊 rollback。
3. **fake model**：至少 16 層之後還有非線性下游運算；線性模型只驗代數，不能替代真實 L15 後續結構。設計已知有／無概念路徑兩種模型，驗 gates 拒絕全局推注與抹除。只驗控制流程與算式，不能從 fake model 校準真實效應門檻。
4. **資料**：fit/audit 公司及模板家族互斥、同義改寫歸組、候選檔 freeze hash、正負號不按 outcome 選；材料品質另由人審。
5. **統計**：raw 與 toward-source 符號、雙向不當獨立樣本、共享公司配對的相依性、Holm、null ratio、所有排除原因、fail 與 inconclusive；同方向推 sell 不應通過雙向 gate。
6. **schema/lifecycle**：no raw tensors、finite、各 arm coverage、invalid upstream hash、partial run、已存在 run 拒絕覆寫、schema 與 stage 配對。

### 6.3 數值驗證與科學驗證不能以測試通過替代

數值：tokenizer、canonical lens、hook、計分、誤差帶及舊 artifact 重現。
科學：未見材料能否辨識概念、自然公司是否有 loading 差、介入是否勝 controls、上游效應能否被特異恢復。模板不變性或跨公司泛化是實驗結果，不是 unit test 可保证的前提。

每階段先跑以上 regression 與新增 tests，再經批准跑真實 smoke，再批准 calibration／formal。最終 code 變更另跑 root AGENTS 的 lock/pytest/compileall/build/JS checks。

## 7. 版本變更與授權邊界

三階段的 prompt family、direction source、estimand、split/freeze sequence、control family 或 gate 實質改變，依 [experiment versioning](../../documentation-system.md#experiment-versioning) 建新版本。formal 前 Draft 可修訂但保留變更紀錄；formal 後不可因結果不佳放寬。候選為空也應 finalize 為完整負結果，不啟動後續階段。
