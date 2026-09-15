# Phase 1 首批實作規格：只建立無模型的驗證與計算基礎

**狀態：Draft 1 已獲實作批准；Slices 1–3 已實作並各自驗收。**三階段研究規劃已獲批准，本 spec 只涵蓋 [Phase 1](proposal-phase1.md) 的基礎程式，不代表完整 Phase 1 pipeline 已定案。研究語義依原始協議與[共通契約](design-and-validation.md)；此檔是下列 slices 的唯一實作指引。

## 1. 目標、範圍與禁止事項

**目標**：不載入模型，驗證來源及材料，並以合成向量證明概念方向計算符合已批准公式。三個 slices 依序執行、各自驗收，不採並行 batch。

In-scope：新增不破壞既有 API 的 core helpers；新實驗 package 的材料 schema 與方向計算；temporary fixtures 的 CPU tests。

Non-goals：不新增 CLI／runner／ArtifactRun、不生成真實概念材料、不讀 checkpoint／tokenizer／lens、不推論、不存逐句向量、不作 semantic pass/fail、無 Phase 2/3 hook。不得修改舊實驗 package、既有 artifacts、pyproject／依賴／lockfile、CLAUDE.md 或其失敗測試。`CLAUDE.md` 缺失仍須另案處理。

完成本 spec 只可稱「Phase 1 基礎工具可測試」，不能稱 implemented Phase 1。模型 preflight、讀出聚合、正式候選檔、統計與 CLI 須另寫後續 spec，待材料／gate 契約凍結。此範圍不替任何 gate 補上任意數字。

## 2. 編號需求與精確契約

### REQ-1：共用上游驗證須對照登記內容，不只算新 hash

新增 `llm_bias/core/artifacts/verification.py`，不修改或取代 `registered.verified_run`：

`verify_completed_inputs(run_root: str | Path, *, dataset: str, model: str, run_id: str, expected_manifest_sha256: str, required: Mapping[str, tuple[str, int | None]]) -> dict[str, dict[str, str | int]]`

- `required` key 是 POSIX run-relative 檔案路徑；tuple 為 `(expected_file_sha256, expected_record_count)`。JSON 的 expected count 必須 None，ref.record_count 必須 absent／None；JSONL 的 expected 與 ref count 必須正整數；只支援這兩副檔名。required 不得空。
- 用 `core.artifact_paths.file_sha256` 驗證 manifest 外部提供 digest；再檢查 schema_version==1、完整 run status、dataset/model/run_id 精確相等。bool 不能當 schema/count integer。
- 只讀 top-level `output_refs`（現行 RunManifest schema），不把 `artifacts` list 或未知 nested schema 當 fallback。所有 ref path 都須為非空規範相對路徑：拒絕 absolute、`..`、`.`、反斜線、重複斜線；resolve 後不可離開 run root，包括 symlink。重複路徑拒絕。
- 每個 required 檔案須恰有一個 ref，ref `role=output,status=complete`，stage 非空且 manifest.stages[stage].status==complete。expected digest、ref digest、實際 bytes digest 三者相同。所有 digest 嚴格 lowercase 64 hex。
- JSONL 用 `core.artifacts.io.read_jsonl` 驗證 object rows，count 與 ref.record_count 及 required count 一致；JSON parse 須為 object。manifest、JSON 與每個 JSONL object 都須遞迴檢查 finite float（包含解碼成 Inf 的 1e400），不能假設既有 read_jsonl 已拒絕 NaN/Infinity。額外 outputs 不開檔不驗 hash，明確只保證 required inputs，不能宣稱全部 run 產物驗過。
- 結果以 required path 為 key，value 為 `path`（resolved absolute string）、`sha256`、`stage`；JSONL 加 `record_count`。不返回 manifest 或資料 payload、不建立檔案。
- 缺檔保留 FileNotFoundError；格式、digest、identity、逃逸或狀態錯誤 ValueError，錯誤訊息帶 offending path／field。

例（hash strings 在測試中由 fixture 真實 bytes 計算）：

```python
required = {
    "analyze/summary.json": (summary_digest, None),
    "prepare/rows.jsonl": (rows_digest, 2),
}
# result["prepare/rows.jsonl"]:
# {"path": "/tmp/run/prepare/rows.jsonl", "sha256": rows_digest,
#  "stage": "prepare", "record_count": 2}
```

真實 upstream 預期身份：E `model=qwen3.5-4b,dataset=entity-to-dial,run_id=entity-to-dial-e-01`；2A `model=qwen3.5-4b,dataset=balanced-evidence-gap-phase2,run_id=phase2a-gpu-bf16-01`。本批只測 fabricated run，不將本機 artifacts 作測試 fixture。

### REQ-2：基底解析與投影是無研究語義的共用純函式

新增 `llm_bias/core/analysis/subspaces.py`：

- `basis_from_rows(rows: Sequence[Sequence[float]], *, dimension: int, rank: int, atol: float = 1e-6) -> torch.Tensor`：嚴格二維 shape [rank,dimension]（含 rank=dimension=1 時仍為 [1,1]；dimension/rank 必須 int 且非 bool）、finite real numbers（拒 bool/string）、0<rank≤dimension；Gram 以 CPU FP64、`rtol=0` 驗 `rows @ rows.T≈I`；返回 CPU FP32 contiguous [dimension,rank]。atol 必須 finite 0<atol≤1e-3。不讀檔、不要求 singular values，不做重新正交化。
- `project_onto_basis(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor`：兩者 CPU floating tensors，values shape [d] 或 [n,d]、n>0；basis [d,k]、1≤k≤d、finite、Gram FP64 atol=1e-6/rtol=0。FP32 計算 `(values @ basis) @ basis.T`，返回同 shape FP32 CPU、不修改輸入。拒 complex/int、NaN/Inf、錯維度／非正交。返回值亦須 finite，FP32 overflow 拒絕。

不移動既有 loaders、不改其 tolerant 行為；新實驗後續 loader 可讀 `pca_basis_vectors` 再呼叫此函式。SVD 的正值及排序驗證留給未來 upstream schema integration；此 helper 不宣稱驗過 e-01。

### REQ-3：概念材料須有可稽核的人工定義與家族分割

新增 `llm_bias/entity_concept_decision/materials.py` 與空白／docstring-only `__init__.py`。以 frozen dataclasses 定義下列公開型別：

- `ConceptDefinition(concept_id: str, definition: str, positive_pole: str, negative_pole: str, excluded_interpretations: tuple[str,...])`。
- `ConceptPair(id: str, concept_id: str, family_id: str, split: str, text_positive: str, text_negative: str, review_status: str, confound_tags: tuple[str,...])`。
- `MaterialBundle(concepts: tuple[ConceptDefinition,...], pairs: tuple[ConceptPair,...], source_sha256: str)`。

`load_material_bundle(path: str | Path, *, expected_sha256: str) -> MaterialBundle`：本批明確採單一 JSON object，內有 `schema_version:1, concepts:[...], pairs:[...]`；不與 future run schema 混用。未知／缺欄、重複 JSON keys（各 nesting 層）、NaN/Inf 均拒絕。先核 SHA 再 parse；不寫檔。

所有字串須非空且首尾無空白，不偷偷 trim 改動輸入。概念 1–3 個；ids 唯一、pair.concept_id 存在；excluded_interpretations 非空、tags 可空，兩種 lists 均 unique strings。positive/negative poles 不同、pair 兩句不同。review_status 僅 `approved`，否則 ValueError；split 僅 `fit,validation,audit`。每概念至少各有一個家族於三 splits：這是結構最低限，不是協議建議的 6/3/3 預算或功效保證。

family_id 為 bundle-global：同一家族可有多個概念／改寫，但只能落一個 split。另用 `unicodedata.normalize('NFKC', text).casefold()` 後 collapse whitespace 作 duplicate key；同一正／負句不可跨 splits，正負句正規化後相同亦拒絕。相同字串同 split 可留存以表示共用控制，id 仍 unique。這只能抓直接文字洩漏，不能辨識所有語義同義或證明人工審查可信。

資料例（完整合法 fixture 需再加 validation/audit 兩家族）：

```json
{
  "schema_version": 1,
  "concepts": [{
    "concept_id": "c1",
    "definition": "合成測試的地形差異，非真實候選",
    "positive_pole": "mountain",
    "negative_pole": "coast",
    "excluded_interpretations": ["general sentiment"]
  }],
  "pairs": [{
    "id": "p1", "concept_id": "c1", "family_id": "f1", "split": "fit",
    "text_positive": "The setting is mountainous.",
    "text_negative": "The setting is coastal.",
    "review_status": "approved", "confound_tags": []
  }]
}
```

### REQ-4：每個家族等權，投影後近零須明確退化

新增 `llm_bias/entity_concept_decision/concepts.py`：

`fit_concept_direction(bundle: MaterialBundle, concept_id: str, *, positive: torch.Tensor, negative: torch.Tensor, pair_ids: Sequence[str], basis: torch.Tensor, min_norm: float) -> ConceptFit`

- positive/negative 是 RAM 內已平均 instruction positions 的 [n,d] CPU floating tensors；此函式不負責 capture。兩者有限同 shape，n>0；pair_ids 長度 n、unique，集合必須恰等於該 concept 所有 `fit` rows 的 IDs，不接受 validation/audit、缺列或混概念。不同輸入順序依 ID 正確對應。
- basis 使用 REQ-2 的契約，不硬編碼 d=2560 於純函式。production dimension 驗證由後續 loader 負責。
- 先對每列求 positive−negative（FP32），再在各 family 內平均，最後家族等權求 d_c；家族以 `sorted(family_ids)` 排序，各家族內依 pair ID 排序再 reduction，以固定輸入排列不影響計算。不能讓有較多改寫的家族權重更大。
- `u_c = project_onto_basis(d_c,basis)`。min_norm 是 caller 提供 finite >0，無 default 科學門檻；||d_c|| 或 ||u_c||≤min_norm 返回退化，不用 eps 強行正規化。溢位非 finite 是 ValueError，不算退化。
- `ConceptFit` frozen dataclass 欄位：`concept_id:str,status:str,reason:str|None,direction:torch.Tensor|None,coordinates:torch.Tensor|None,source_norm:float,projected_norm:float,retained_fraction:float|None,n_pairs:int,n_families:int,materials_sha256:str`。
- status=`ok` 時 direction=u/||u|| [d]、coordinates=basis.T@direction [k]，reason=None；retained_fraction=projected_norm/source_norm（不默默 clamp）。status=`degenerate` 時 direction/coordinates=None，reason=`source_norm_below_minimum` 或 `projected_norm_below_minimum`；包括 degenerate 狀態：source_norm==0.0 時 fraction=None，source_norm>0.0 時 fraction=float(projected_norm/source_norm)，不以 min_norm 作分母。無 gate `pass` 欄位。
- `concept_scores(values: torch.Tensor, direction: torch.Tensor) -> torch.Tensor`：values CPU floating [n,d]、n>0，direction [d] unit（FP64 norm atol=1e-6,rtol=0），均 finite，返回 FP32 [n] `values @ direction`；不 center、不定 threshold、不讀 margin。

此批不序列化 ConceptFit 中的 tensors。資料 reader 只經 core hash helpers；不可新增 local artifact serializer。

### REQ-5：所有新 tests 無模型，所有既有行為保持

不 import experiment siblings、不觸發 model/lens/tokenizer loader、不寫真 artifacts。core helpers 沒有實驗詞彙。test suite 不依赖網路/GPU，缺 repo datasets 不影響 tests。所有新 API 的 input mutation tests 與 fail-closed tests 必備。

## 3. Sequential slices 與驗收

### Slice 1：上游驗證 helper

- Covers：REQ-1、REQ-5。
- Create：`llm_bias/core/artifacts/verification.py`、`tests/test_core_input_verification.py`。
- 不修改其他 code；不 re-export 到 core `__init__`。
- Consumes：既有 file_sha256、read_jsonl；Produces：verify_completed_inputs。
- Fixture：tmp run，schema v1，prepare/analyze complete，1 個 JSON object、2-row JSONL，實際 hashes 登記 output_refs；不同 fixture 測 absolute/path traversal/symlink escape、duplicate ref、bad stage/status、wrong expected/ref/actual hash、count mismatch、bool count、non-object JSON/JSONL、non-finite JSON、extra missing non-required output 不阻擋。
- Steps：先建立正例與 mutation tests；實作 strict verify；確認無新增 output。
- Gate：`uv run pytest -q tests/test_core_input_verification.py tests/test_core_artifacts.py tests/test_artifact_manifest.py tests/test_artifact_paths.py`。

### Slice 2：投影數學 helper

- Covers：REQ-2、REQ-5。
- Create：`llm_bias/core/analysis/subspaces.py`、`tests/test_core_subspaces.py`。
- Consumes：torch；Produces：basis_from_rows、project_onto_basis。無 Slice 1 import 依賴，但仍依序驗收。
- Fixture：d=4、k=2 的 e1/e2 基底；[1,2,3,4] 投影為 [1,2,0,0]；45° rotation 與 sign flip projector 等價；nonorthogonal、ragged、bool、超大值、empty shapes 等拒絕。FP64 Gram 檢驗與 FP32 返回分開斷言。
- Steps：實作解析／projection、輸入不變、idempotence 誤差界；不改既有 projection。
- Gate：`uv run pytest -q tests/test_core_subspaces.py tests/test_core_analysis.py tests/test_selective_intervention_subspace.py tests/test_entity_to_dial_joint.py`。

### Slice 3：材料與概念 fitting 純邏輯

- Covers：REQ-3、REQ-4、REQ-5；依賴 Slice 2。
- Create：`llm_bias/entity_concept_decision/{__init__,materials,concepts}.py`、`tests/test_entity_concept_materials.py`、`tests/test_entity_concept_directions.py`。
- Modify：`llm_bias/AGENTS.md` 加新 package 狀態「Phase 1 基礎工具；無 CLI／runner」與 proposal link；`tests/AGENTS.md` 加兩測試入口一行。不更新已列的全局 CLI 名單。
- Fixture：一個概念，fit 兩家族（其一有兩個改寫）、validation/audit 各一家族；fitting 差向量設為 f1=[2,0,0,0]（兩列相同）、f2=[0,2,0,0]（一列），期望 d=[1,1,0,0]，v=[1/√2,1/√2,0,0]，不得變 [4/3,2/3,0,0]。正負 label 交換時方向反號，pair 順序不改方向。
- Tests：duplicate keys/id、same family cross split、normalized duplicate cross split、未知／未批准材料、missing split、wrong SHA、fit 與 audit 混用、少／多 pair ids、NaN、zero source、方向全落 K 外、norm 恰等於 threshold、unit scoring、tensors 不被序列化、不改輸入。
- Gate：`uv run pytest -q tests/test_entity_concept_materials.py tests/test_entity_concept_directions.py tests/test_core_subspaces.py tests/test_prompt_input.py`。

## 4. 全域驗收與已知 blocker

每 slice 先由 general 實作，再由主流程親自跑 gate；失敗未解決不標 completed。文件審查不等於程式驗收。

全批另跑：

```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
uv build
node --check llm_bias/static/prompt_readout.js
node --check llm_bias/static/attribution_dashboard.js
```

目前已知 `tests/test_workflow_boundaries.py::test_root_guidance_shares_exact_workflow_contract` 因 HEAD 缺 CLAUDE.md 失敗；完整 suite 仍執行並如實報告，不 skip、不 xfail、不寫臨時檔騙過。新 slice tests 必須全綠；若全套仍此失敗，只能稱 bounded slices verified，不能稱全線 verification pass。其他新失敗要查原因。

## 5. 首批以外的依賴不以 placeholder 冒充完成

本 spec 所列函式都有可執行驗收條件；故意不提供 model runner 或「尚未決定的 gate」欄位。後續 spec 必須明確定義：概念材料正式檔、common suffix render/token partition、lens 完整 softmax 聚合、upstream P1 integration、校準／audit 執行順序、數值與語義判準、run schema、CLI 與真實 smoke。未批准後續 spec 不啟動這些工作。

## 6. 本 spec 的 preflight 紀錄

唯讀 explorer 已對照 codebase 與 Phase 1 協議審查。已明確化 JSON ref count、rank=1 的二維 shape、退化比例與 family/pair reduction 順序；主流程另補 JSONL 的非有限值檢查，避免誤信既有 reader。七份本線 Markdown 的相對檔案目標存在，spec 引用的 8 個既有測試路徑存在、4 個新增測試明確標為尚未實作；`git diff --check` 通過。spec 審查輪未重新跑 pytest，當時四份新增 tests 尚未實作。後續實作與實際驗證另見[首批交付紀錄](implementation-phase1-foundations-results.md)，不將本節歷史 preflight 當作程式測試結果。
