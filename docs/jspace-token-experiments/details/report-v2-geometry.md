# J-space outcome direction 幾何投影分解

## Status and scope

**Status：Draft 1 已實作（2026-08-28）；第一次正式 run 見下方 Run record。**

本文件是 **outcome direction geometric projection** 的 canonical workflow 文件。
它是一個 **輔助、描述性、非 causal 的幾何診斷**，綁定 V2 的 frozen direction
identity，不改變 V2 protocol 的任何 direction source、primary outcome、split/freeze
sequence、control family 或 success gate，因此不是 V2 的新版本（版本入口見
[J-space token experiment versions](README.md)）。它的地位與
V2 文件記載的 per-(layer, position) attribution screen 相同：讀取既有 frozen
artifacts 與 model/lens，只輸出 compact derived metrics。

它回答的問題是：對每一個 V2 fitted layer \(l\)，Technology 與 Financial Services
discovery prompts 的 **sector state difference** \(\Delta h_{sector,l}\)（定義見下）
在 V2 **outcome direction** \(d_l\) 上的 **dot projection** 多大？把 \(\Delta
h_{sector,l}\) 分解為與 \(d_l\) 平行的 **parallel component** 與垂直的
**perpendicular component** 後，兩者的 norm、angle 與能量占比如何？若提供 repo
既有的 frozen **contrastive TF-IDF sector prototype**（sector intervention 的
TF-IDF direction），同層同方向下它的 parallel/perpendicular 分解為何？

**Interpretation limits（固定，不隨結果改變）：**

- 全部輸出是 representation geometry 的描述：projection coefficient、norm、cosine
  angle、energy fraction。幾何對齊 **不是** causal evidence；不得從本 run 宣稱
  sector difference「造成」outcome steering，也不得宣稱 TF-IDF prototype direction
  與 outcome direction 的 correlation 有任何 causal 意涵。
- Sector state difference 使用 discovery split，與 V2 direction fitting 的 prompt
  population 同源（Technology 側完全相同），因此觀察到的任何對齊都是
  in-sample、描述性的，不構成 held-out evidence。
- 本 workflow 沒有 success gate、沒有 dose、沒有 intervention；run 成功僅表示
  pipeline 完成且 direction identity 通過 hash 驗證。

## 定義與公式

### Sector state difference

對 fixed position set \(P_i\)（prompt \(i\) 內的 token positions）、layer \(l\)、
sector \(s\)，定義

\[
\bar h_{i,l,P}=\frac{1}{|P_i|}\sum_{p\in P_i} h_{i,l,p},
\qquad
H_{s,l,P}=\frac{1}{|S_s|}\sum_{i\in S_s}\bar h_{i,l,P},
\]

\[
\Delta h_{l,P}=H_{source,l,P}-H_{contrast,l,P}.
\]

\(h_{i,l,p}\) 是 clean forward 中 layer \(l\) 的 residual state（model dtype
capture 後轉 float32 累積，永不落盤）。固定協議選擇（frozen in Draft 1）：

- split：**discovery**（direction fitting 的 population）；
- sectors：`source_sector` = V2 config 的 `source_sector`（Technology），
  `contrast_sector` = 預設 Financial Services（CLI 可改，但必須是 split manifest
  中存在且 discovery 非空的 sector）；
- prompt：每個 ticker 一個 prompt，預設 column
  `prompt_with_context_attribute_0`（V2 同一 default）；
- position sets：V2 frozen position rules（`evidence_item_end`、
  `evidence_span_all`）加上 `final_position`（scoring prompt 的最後 token，即
  V2 decision margin 的計分位置，文件術語 final position）；
- layers：V2 config 的 `fitted_layers`（\(d_l\) 存在的層）。

### Dot projection 與 parallel/perpendicular 分解

對 vector \(\Delta h\) 與 direction \(d\)，**不假設** \(d\) 已 unit-normalized
（V2 direction 依定義是 unit，但 formula 對任意 norm 成立，且 run 會報告
\(\|d_l\|\) 與 hash 以證明）：

\[
c=\frac{\langle \Delta h,\, d\rangle}{\|d\|^{2}},
\qquad
\Delta h_{\parallel}=c\,d,
\qquad
\Delta h_{\perp}=\Delta h-\Delta h_{\parallel}.
\]

\(c\) 是 **projection coefficient**（帶符號：\(c>0\) 表示 sector difference 指向
\(+d\)，即 V2 的 Buy steering 方向）。派生 compact metrics：

\[
\|\Delta h_{\parallel}\|=|c|\,\|d\|,
\qquad
\cos\theta=\frac{\langle \Delta h,d\rangle}{\|\Delta h\|\,\|d\|},
\qquad
\theta=\arccos(\cos\theta)\in[0^\circ,180^\circ],
\qquad
f_{\parallel}=\frac{\|\Delta h_{\parallel}\|^{2}}{\|\Delta h\|^{2}}.
\]

\(\|\Delta h\|=0\) 時 cosine 與 angle 記 null（degenerate）。所有點積在 float64
上計算；另報 Pythagoras error
\(\bigl|\|\Delta h\|^{2}-\|\Delta h_{\parallel}\|^{2}-\|\Delta h_{\perp}\|^{2}\bigr|
/\|\Delta h\|^{2}\) 作為 numeric sanity diagnostic（應在 float64 round-off 量級）。

因為 \(d_l\) 依 position rule 而異（`evidence_item_end` 與 `evidence_span_all`
各 fitting 一條 axis），每個 \((P, l)\) 的 projection 對 **每一條** frozen rule
的 \(d_l\) 各報一次，不做預設偏好。

### TF-IDF sector prototype geometry（optional input）

Repo 既有 TF-IDF direction 是 sector intervention 的 **contrastive TF-IDF sector
prototype**：以 frozen discovery TF-IDF contrastive scores 選出每個 sector 的
single-token concepts，每層方向為

\[
v_{s,l}=\text{unit}\Bigl(\sum_{t\in s} w_t\,\frac{W_U J_l[t]}{\|W_U J_l[t]\|}\Bigr)
\]

（`layer_prototypes` 的既有實作：canonical Jacobian-lens token directions 的
weighted unit mean，依 `sector_prototype` 歸一化）。提供
`--tfidf-config`（frozen `jspace_intervention_config`，且兩個 prototype 的
`score_type` 必須是 `contrastive_tfidf`，否則 fail closed）時，對每個 sector
prototype \(v_{s,l}\) 與每條 rule 的 \(d_l\) 做同一套 dot projection 分解，另報
\(\cos(v_{source,l},v_{contrast,l})\) 與 \(\cos(v_{s,l},\Delta h_{l,P})\)
（每個 position set）。這些是 geometry/correlation 描述，**不是** causal proof
（見上方 Interpretation limits）。

## Inputs 與 fail-closed bindings

全部輸入以 SHA-256 綁定；任一不一致即 fail closed（run 標記 failed）：

| Input | 要求 |
|---|---|
| baseline prompt CSV | `input_sha256` 與 direction identity 冻结值一致 |
| split manifest | `split_manifest_sha256` 與 config 及 identity 冻结值一致 |
| V2 `outcome_flip_config` | `artifact_type=outcome_flip_config`；`model` 一致；`split_manifest_sha256` 一致 |
| V2 `outcome_flip_direction_identity` | `split=discovery`；`model`/input/split/config hashes 一致；`fitted_layers` 與 `position_rules` 與 config 一致；identity 冻结的 record 集合仍可在目前 input 重導出 |
| optional `jspace_intervention_config`（TF-IDF） | `artifact_type=jspace_intervention_config`；`model` 一致；`source.sector` = config `source_sector`、`target.sector` = `contrast_sector`；兩者 `score_type=contrastive_tfidf`；其 `layers` 與 `fitted_layers` 交集非空（只在交集上計算） |
| canonical lens（僅 TF-IDF 路徑） | 既有 validated canonical lens，只讀，不 fitting、不取代 |

Direction 本身永不落盤：本 run 在 deterministic mode（`torch.
use_deterministic_algorithms(True)` 與 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，同 V2
Implementation 3）內 recompute V2 direction，並對 identity 的每條 rule、每個
fitted layer 做 bit-exact SHA-256 驗證（含 label-permutation directions），
mismatch 即 fail closed。

## Implementation

- **Package**：`llm_bias/jspace_intervention/outcome_geometry.py`（V2 同 owner；
  複用 `verify_direction_identity`、`record_residuals`、`layer_prototypes`、
  `sector_prototype` 與 core artifact lifecycle）。
- **CLI**（`jspace-intervention`）：`run-outcome-geometry`

  ```bash
  uv run jspace-intervention run-outcome-geometry \
    --input data/baseline/paper-local-qwen36-27b/trial_plan_prompts.csv \
    --split-manifest artifacts/qwen3.5-4b/jspace-intervention/splits.json \
    --config artifacts/qwen3.5-4b/jspace-outcome-direction-flip/config-technology-draft1.json \
    --direction-identity artifacts/qwen3.5-4b/jspace-outcome-direction-flip/runs/<discovery-run>/forward/direction_identity.json \
    --model .cache/models/qwen3.5-4b \
    --run-id outcome-geometry-tech-<ts>Z \
    --contrast-sector "Financial Services" \
    --lens artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt \
    --tfidf-config artifacts/qwen3.5-4b/jspace-intervention/config-v2-tfidf-tech-to-financial.json
  ```

  `--lens` 與 `--tfidf-config` 必須同時提供（TF-IDF geometry 需要 lens 與 frozen
  prototype spec）；省略兩者時只跑 sector state difference 分解。`--max-seq-len`
  預設 1024；`--prompt-column` 可重複，預設
  `prompt_with_context_attribute_0`。
- **Lifecycle**：`prepare → forward → analyze → finalize`
  - prepare：驗證全部 bindings、preflight prompt 長度、落
    `prepare/prompt_records.jsonl` 與 prepare metadata；
  - forward：deterministic mode recompute + verify V2 direction；clean forward
    逐 prompt 流式累積 per-sector mean state（記憶體中，raw activations 即算即
    棄）；TF-IDF 路徑以 canonical lens 建構 sector prototype directions；
  - analyze：sector state difference、dot projection、parallel/perpendicular
    分解與全部 compact derived metrics；
  - finalize：只註冊 compact outputs 與 provenance，postcheck 後 complete。
- **Dataset slug**：`jspace-outcome-direction-geometry`；runs 位於
  `artifacts/<model-slug>/jspace-outcome-direction-geometry/runs/<run-id>/`。
- **Artifact types（schema version 1）**：
  - `outcome_geometry_prompt_record`（prepare/prompt_records.jsonl）：
    record_id、ticker、name、sector、marketcap、prompt_column、split、prompt；
  - `outcome_geometry_prepare_metadata`：bindings、sector 與 prompt counts、
    position sets、layers、contrast sector；
  - `outcome_geometry_sector_state`（forward/sector_state_statistics.jsonl）：
    每 `(sector, position_set, layer)` 一行：`state_mean_norm`（sector mean
    state vector 的 L2 norm）、`state_mean_sha256`（float32 bytes hash）、
    `mean_record_state_norm`（per-record position-mean norm 的 sector 平均）、
    `record_count`。**不**落盤任何 vector；
  - `outcome_geometry_metadata`（forward/metadata.json）：direction identity
    sha、per-rule per-layer recomputed hash 與 norm（驗證記錄）、deterministic
    mode、lens/TF-IDF provenance、record counts；
  - `outcome_geometry_analysis`（analyze/outcome_geometry_analysis.json）：
    per-layer `direction`（rule → norm/hash）、per-layer per-position-set
    `delta_state`（norm、sha256、對每條 rule 的 projection：coefficient、
    parallel/perpendicular norm、cosine、angle_deg、parallel_fraction、
    pythagoras_error）、optional `tfidf_prototype` 段（同分解 +
    cosine_source_contrast + cosine_to_delta_state）、provenance 與
    interpretation limits；
  - `outcome_geometry_analysis_metadata`：interpretation =
    `descriptive_geometry`、counts。

## Run record

**正式 run 1（2026-08-28，discovery split，deterministic mode）**：

- Run：`artifacts/qwen3.5-4b/jspace-outcome-direction-geometry/runs/outcome-geometry-tech-20260828T101549Z`
  （worktree branch `experiment/geometric-projection` 的 ignored `artifacts/`）。
- Inputs（SHA 綁定）：V2 config `config-technology-draft1.json`；direction identity
  `outcome-flip-tech-discovery-det-20260828T015605Z`（重算後 2 條 rule × 21 層全部
  hash 一致，`‖d_l‖ = 1.00000`）；TF-IDF 方向：
  `config-v2-tfidf-tech-to-financial.json` 的 contrastive TF-IDF prototypes
  （L14–26，canonical lens）。
- Population：35 個 Technology discovery tickers vs 36 個 Financial Services
  discovery tickers，各 1 prompt（`prompt_with_context_attribute_0`）；layers L10–30；
  position sets `evidence_item_end`、`evidence_span_all`、`final_position`。

主要結果（對 `evidence_item_end` rule 的 d_l；全部 126 個
`(layer, position set, rule)` 投影皆 angle ∈ [82°, 92°]）：

| 觀察 | 數值 |
|---|---|
| \|\|Δh\|\| 隨層深單調增大 | 0.47（L10, item_end）→ 4.88（L30, item_end） |
| projection coefficient（帶符號） | 全部 \|c\| ≤ 0.125；符號逐層交錯（L11–15 item_end 微正、L20–30 微負），無一致 signed alignment |
| parallel energy fraction \|\|∥\|\|²/\|\|Δh\|\|² | 全部 ≤ 0.019；最大為 L16 `final_position`（0.019），多數 < 0.005 |
| angle(Δh, d_l) | 82°–92°（近 orthogonal） |
| TF-IDF prototypes vT/vF 對 d_l | angle 89°–92°，\|c\| ≤ 0.036（L14–26） |
| cos(vT, vF) | 0.51（L14）→ 0.31（L26），sector prototype directions 隨層深趨散 |

解讀（限於 Interpretation limits）：在 discovery population 上，兩產業平均
residual state 差（sector state difference）在每個 fitted layer 幾乎全部落在
outcome direction 的垂直方向；幾何上不支持「sector difference 沿 outcome axis」
的描述性假設。這是 in-sample、描述性幾何，不是 causal evidence，也不能推翻
V2 Test run 1 的 behavioral 結果（final-position control 已顯示 outcome steering
可由 answer-position injection 重現）。

## Verification

```bash
uv run pytest -q tests/test_jspace_outcome_geometry.py
uv run python -m compileall -q llm_bias
```
