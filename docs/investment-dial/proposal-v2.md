# Investment-dial calibration V2 proposal（fine-grid 重校準與 B reevaluation）

Status: proposed；未實作；formal run 尚未授權。
本文件是 investment-dial 線的第 2 版 calibration 協議。它不修改 V1（protocol
`investment-dial-local-v1`）與 fine-a diagnostic（protocol
`investment-dial-fine-a-v1`）的任何結果；V1 的 run artifacts 保持不變。Paper
provenance 見 [README](README.md)。

## 1. 核心假說與文獻邊界（Scientific Question & Literature Boundary）

### Scientific question

V1 calibration 在 Qwen3.5-4B 的 selected coordinate（L15/n8490）上對 target grid
{−0.3, 0, +0.3} 的 B reevaluation RMSE 為 0.5757：target −0.3 達成（π_B=−0.382），
但 neutral（量到 +0.717）與 +0.3（量到 +0.988）各偏約 0.7。fine-a diagnostic
（run `fine-a-gpu-bf16-01`）顯示 A transfer curve 是 front-loaded 的 steep
transition：π 的變化 98% 在 δ=1.5 前完成，而 V1 的四個 coarse curve points {−8, −4, +4, +8} 沒有取樣 0–2 區間。

本版回答一個明確問題：**把 A transfer curve 在 0–2（與 −0.5–0）細化取樣後重新
inversion，所得的 Δ̂(t) 能否在 B reevaluation 上以可接受的 RMSE 重現 target grid？**
具體地，V1 的失敗是否與 coarse interpolation 的取樣解析度有關（fine-a 支持但未
單獨證明），仍由本版的 B reevaluation 結果描述；本版不預先聲稱 intermediate
stance 具有或不具有某種本質限制。

### 文獻出處

- Park, S., Park, S., Lee, H., Kwon, G., Ahn, W., Choi, J., Lopez-Lira, A., Kim,
  Y., Choi, C., Kong, H., and Lee, Y. (2026). *Your AI, On a Dial: Controlling
  Investment Bias in LLMs with a Single Neuron*. arXiv:2608.22852。
- 本協議直接對應：equation (5)（Δ̂ = Inv(π_A, t)）、equation (6)（RMSE on
  S_B）、Algorithm 1 的 step 4–7（A transfer curve + B RMSE）、Table 2/3 的
  報告格式。

### 文獻原始設定 vs 本專案適應性修改（Adaptations）

| 項目 | 論文原始設定 | 本協議設定 | 理論風險 / 邊界 |
|---|---|---|---|
| Models | 5 個 open-weight LLM（含 Qwen3-8B，L23/F4099，RMSE 0.038） | 僅 Qwen3.5-4B（非論文模型） | 不可做跨模型 generalization claim；RMSE 不與論文絕對值直接比較 |
| 宇宙規模 | 427 tickers，每 setting ~8,500 decisions | A 85 + B 85 companies，每 setting 340 decisions | iid 近似下 per-point π 的 SE 約 ±0.054（n=340、中段）；四筆 prompt/公司結構使
實際不確定性另受 company dependence 影響，statistical power 低於論文 |
| Candidate selection | screening + 多 candidate RMSE ranking | coordinate 固定為 V1 selected（15, 8490），不重選 | 若 coordinate 本身次優，v2 失敗會被誤歸因為 grid 問題（confound，見 §6 解釋約束） |
| 最終設定 | 選定後在 full universe 重估（Algorithm 1 step 9） | A-only inversion（沿用 V1 的 A_only_no_full_universe_refit） | A→B transfer 誤差可能偏大或偏小；與論文最終報告口徑不同 |
| Transfer curve Δ 範圍 | 各 model 的 reachable range | 組裝 15 點：V1 raw 的 {−8, −4, +4, +8}、fine-a 的 9 點 {0, 0.25, …, 2}、以及新量測 {−0.5, −0.25} | 負側局部斜率與兩端飽和由 compact curve points 描述；不可推論跨 coordinate 或跨 model 的可達範圍 |
| Target grid | {−0.3, 0, +0.3} | 相同 | — |
| Intervention | 單 MLP down-projection coordinate、all token positions（含 generated）、additive Δ | 相同（original intervention placement） | — |
| Decoding / budget | greedy、256 tokens | greedy、256 tokens、bf16 | — |

### Pre-registered hypotheses

- **H1（grid artifact）**：v2 gate 通過（RMSE ≤ 0.15）；Δ̂(0) ∈ [0.10, 0.30]、
  Δ̂(+0.3) ∈ [0.25, 0.55]（依 fine-a curve 的線性插值預期區間）。這些是
  pre-registered descriptive expectations，不是額外 gate。
- **H2（reevaluation miss）**：gate 失敗，尤其 neutral target 仍偏 >0.25；結果
  只能描述為本 coordinate 在本 B population 上未達 frozen tolerances，不能單獨
  證明 intermediate stance 的 inherent impossibility。
- 兩種結果都有資訊量；任何結果都不允許 in-scope iteration（失敗即 v3 或終止，
  見 §4）。

## 2. 預期的 Input / Output 契約

### Input

三個 mandatory 輸入，皆 fail-closed（無 fallback 到其他 model、subset 或
neuron）：

1. `--model`：本地 checkpoint 目錄。`model_identity`（checkpoint 檔 SHA-256 +
   tokenizer identity，hash-based，同 V1/fine-a 的 `local_model_identity`）必須
   同時匹配兩個 parent run 記錄的 identity。
2. `--v1-run`：完整 V1 calibration run
   （`artifacts/qwen3.5-4b/investment-dial-calibration/runs/exploratory-v1-20260907-07-gpu0-calibration`）。
   - Manifest 驗證 + 全部 registered outputs 的 hash 驗證
     （`verified_run`）。Parent manifest SHA-256：
     `173cc0cf29c6cd4c2980a87ea95bb0eb72bb467824644687c33c6dddcb6d5ff3`。
   - `analyze/result.json` 的 selected coordinate 必須為 (15, 8490)。
   - `prepare/trials.json`：split A 與 split B 各 340 筆、各 85 家
     （`positive_count == 2`），field 含 `id, ticker, split, positive_count,
     reverse_options, prompt, prompt_ids` 等。
   - `forward/effects.jsonl`：phase `A_curve` 中 (15, 8490) 的 δ ∈ {−8, −4, +4, +8}
     四點各恰好 340 筆 raw records；V1 的其他 coordinate、phase 與 delta records
     必須被排除，不得混入 assembled curve。
3. `--fine-a-run`：完整 fine-a run
   （`artifacts/qwen3.5-4b/investment-dial-fine-a/runs/fine-a-gpu-bf16-01`）。
   - Manifest 驗證 + hash 驗證。Parent manifest SHA-256：
     `ab8a6cb19a594584381218793c245d856e1f4ed7c601f4d6ede5f8983bc61957`。
   - status `complete`、protocol version `investment-dial-fine-a-v1`、
     coordinate (15, 8490)、9 個 `forward/delta-*.jsonl` 各 340 筆。
   - 兩個 parent 的 runtime（torch / transformers / dtype /
     chat_template_sha256）必須與本機 runtime 完全一致（device 差異允許並
     記錄，同 fine-a 契約）。

**A 數據重用規則（frozen）**：fine-a 的 A records（δ ∈ [0, 2]，9 × 340）直接
作為 v2 A curve 的 positive 側。理由：與 v2 的 B 測試同機、same runtime、
same model identity、tokenization 已驗證、greedy/bf16/256 設定一致；其 δ=0
與 V1 A baseline 的差異（1 筆 prompt）已記錄為 descriptive 差異。V1 raw
records 提供 δ ∈ {−8, −4, +4, +8}。新量測只有兩處：**A 的 δ ∈ {−0.5, −0.25}**
（680 次 generation，補負側局部斜率，讓 target −0.3 的 bracket 落在短區間
內）與**全部 B 點**（見下）。

**V1 數據品質註記**：V1 `analyze/result.json` 的 stored `A_curve` 在 δ=8 點
（buy=338, sell=0, n=340, stored pi=0.994117）與 raw records 經現行
`summary()` 重算的結果（valid=338 → π=+1.0）不一致；同陣列其他點與現行
formula 一致。因此本協議的 frozen rule：**v2 永不採用任何 stored summary
數值，所有重用點一律從 raw records 以現行 `summary()` 重算**（δ=−8, −4, +4, +8 來自 V1 `effects.jsonl`；δ=0…2 來自 fine-a jsonl——fine-a
本身即由現行 formula 產生）。重算後的 V1 粗點為 (−8, −1.0)、(−4, −1.0)、
(+4, +1.0)、(+8, +1.0)，δ=8 點保留在 assembled curve 中、參與 monotonicity
pre-check。該 stored-summary 差異的成因未知，本版不將它歸因於特定程式版本。

### B 測試設計（frozen）

- B population：V1 trials 中 split B 且 `positive_count == 2` 的 340 筆（85
  家），不新增、不移除、不替換。B 曾被 V1 用於 candidate ranking，因此本版
  將它稱為 **B reevaluation**，不是 untouched independent holdout；population 與
  validity gates 原樣保留。
- 量測點：δ=0 baseline（340）＋ Δ̂(−0.3)、Δ̂(0)、Δ̂(+0.3)（各 340）。共
  1,360 次 generation。
- 總 generation budget：680（A 負側）+ 1,360（B）= 2,040（RTX 3060 約
  4.5 小時）。

### Output

Run root：`artifacts/qwen3.5-4b/investment-dial-calibration-v2/runs/<run-id>/`
（`--artifact-root` 預設 `artifacts`）。

- `prepare/protocol.json`：schema_version 1、protocol_version
  `investment-dial-calibration-v2`、兩個 parent 的 path 與 manifest SHA-256、
  coordinate (15, 8490)、targets、A curve assembly rule（每點的 source：
  `v1-raw` / `fine-a` / `v2-new`）、inversion method（指向
  `llm_bias.investment_dial.analysis.inverse_curve`，不允許替換）、gates
  （§3 的數值）、max_new_tokens 256、runtime、source_identity。
- `prepare/trials.json`：B rows（340）。
- `forward/a_curve.json`：新 A 負側量測完成後組裝的 15-point curve（每點：delta、
  source、n、buy、sell、valid、pi）＋ monotonicity pre-check 結果；它屬 forward，
  因為 inversion 必須等待新 A inference 完成。
- `forward/a-negative.jsonl`：680 筆（δ ∈ {−0.5, −0.25}，含 layer/neuron/
  delta/decision/validity 欄位，格式同 fine-a records）。
- `forward/b-baseline.jsonl`（340，δ=0）、`forward/b-target-minus-0-3.jsonl`、
  `forward/b-target-0.jsonl`、`forward/b-target-plus-0-3.jsonl`（各 340）。
  每個檔在其 grid point 完成後註冊（同 fine-a 的逐檔註冊）。
- `analyze/result.json`：每個 B 點的 finite counts/rates 與 pi（pi 在 0 個
  valid 時為 null）；Δ̂ 三個值（full-precision float，與 target 成對）；
  RMSE、per-target `target_errors`（π_B(Δ̂) − target）與 max error；gate verdict
  （`pass` / `fail` / `not_evaluable`）；degraded 標記；`certified`（bool）；
  monotonicity pre-check 記錄。
- `manifest.json`：shared prepare → forward → analyze → finalize lifecycle。
- 不保存 raw activations/residuals/gradients；所有數值欄位為 finite float
  （或契約允許的 null）。

### CLI 契約（1:1 綁定；proposed，隨本 proposal 實作）

新 operator `scripts/investment_dial_calibration_v2.py`（重用
`llm_bias.investment_dial` pipeline 的 `verified_run`、`run_context`、
`_decisions`、`frozen_eval`、`summary`、`inverse_curve` 等既有機制；不新增
shared mechanics 到本地）。沒有 automatic resume；中斷後用新 run ID。

Formal run（GPU）：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
uv run --no-sync python scripts/investment_dial_calibration_v2.py \
  --model .cache/models/qwen3.5-4b \
  --v1-run artifacts/qwen3.5-4b/investment-dial-calibration/runs/exploratory-v1-20260907-07-gpu0-calibration \
  --fine-a-run artifacts/qwen3.5-4b/investment-dial-fine-a/runs/fine-a-gpu-bf16-01 \
  --run-id calib-v2-gpu-bf16-01 --device cuda
```

Preflight：同參數加 `--smoke`（定義見 §5）。

## 3. 邊界情況與防禦性行為（Edge Cases & Fail-Safe Policies）

### 退化條件（fail-closed，無 fallback）

- 任一 parent 的 manifest/hash/identity/runtime/tokenization 檢查失敗。
- B population ≠ 340 筆或 distinct tickers ≠ 85。
- V1 `effects.jsonl` 中 (15, 8490) 的 δ ∈ {−8, −4, +4, +8} 任一筆數 ≠ 340。
- **Monotonicity pre-check**：組裝曲線（15 點，delta 遞增排序）必須
  non-decreasing（`inverse_curve` 的既有條件）；違反即中止，不允許靜默剔除
  任何點。
- **Inversion**：使用現行 `analysis.inverse_curve`（strict、無 extrapolation；
  target 超出 measured range 會 raise `target unreachable; extrapolation
  forbidden`）。任何 exception → 中止，不報告部分校準。
- 任一 B target 點 valid_decision_rate = 0（pi null）→ gate 未定義，
  `certified=false`，`gate_verdict="not_evaluable"`，描述性數值仍寫入。

### Degraded 點與 gate（pre-registered，run 前凍結）

- 任一 B target 點 `valid_decision_rate < 0.90` 或 `schema_rate < 0.90`
  （沿用 V1 的 minimum_rate=0.9 語義）→ 該點標記 degraded；任一 target 點
  degraded → `gate_verdict="not_evaluable"`、`certified=false`。
- **Primary gate**（全部 B target 點非 degraded 時判定）：
  - `RMSE = sqrt(mean_t (π_B(Δ̂(t)) − t)²) ≤ 0.15`，且
  - `max_t |π_B(Δ̂(t)) − t| ≤ 0.25`。
  - 兩者皆滿足 → `gate_verdict="pass"`、`certified=true`；否則
    `gate_verdict="fail"`、`certified=false`。
- Gate 定標理由（run 前冻结，非事後挑選）：iid 近似下 per-point π 的 SE 約 0.054
  （n=340 中段）；四筆 prompt/公司結構使實際不確定性另受 company dependence 影響。
  0.15 與 0.25 是 pragmatic frozen tolerances，不把 interpolation error 拆成獨立
  上限，也不宣稱相對 V1 或論文數字有固定倍數改善（論文 n 約 25 倍）。max-residual
  0.25 防止 V1 型態（一個 target 好、兩個 target 各偏 0.7，RMSE 仍可能被平均掉）
  單獨靠 RMSE 過關。
- B baseline π(0) 與三個 B 點的 monotone ordering 檢查為**描述性**，不進入
  gate（iid 近似下 per-point 雜訊約 ±0.054；四筆 prompt/公司結構另受 company dependence 影響，小幅 ordering 違反屬預期）。

### 控制組缺失

v2 不含 control-neuron arm（那是獨立 evaluation stage 的設計，不屬本
calibration 契約）。B population 的完整性由上述 population check 保證；缺失
即 fail-closed，不以 A 或其他 split 頂替。

### 序列覆蓋

本協議無 sequence position split（intervention 依論文 equation (3) 施加於 all
token positions，含 prompt 與 generated tokens，與 V1/fine-a 的 original
intervention placement 相同），此項 N/A。

### 數值容差

- bf16 weights/activations；Δ̂ 以 full-precision float 存檔，於 hook 時加入
  （與 V1/fine-a 相同），不做 bf16 量化。
- 使用 greedy、bf16 與固定 hook arithmetic；這些設定不保證跨硬體或重複 run 的
  bitwise/effect 等價，hardware/device 差異必須記錄，並不得把重複 greedy 結果寫成
  determinism guarantee。
- 除結構性檢查（筆數、identity hashes、population）外，不使用 tolerance
  等價判斷。
- A δ=0 不自量測（重用 fine-a）；fine-a δ=0 與 V1 A baseline 的 1 筆差異已
  記錄為 descriptive，不設 gate。

## 4. 版本分立觸發條件（Version Break Triggers）

以下任一變更 → 建立 `proposal-v3.md` / `report-v3.md`，禁止原地修改本文件：

1. Prompt family 變更（JSON decision/reason prompt、evidence 構造、
   `positive_count` 規則、option counterbalancing）。
2. Estimand 或 direction source 變更：π 定義（buy/sell 計數與分母）、target
   grid {−0.3, 0, +0.3}、coordinate (15, 8490)。
3. Gate 或 method 變更：RMSE 0.15、max-residual 0.25、minimum_rate 0.9、
   inversion method（`analysis.inverse_curve`）、A curve assembly rule（點的
   source、monotonicity pre-check、重用規則）。
4. B population 變更（rows、companies、選擇方式）。

註：coordinate 的重新 screening/selection 是獨立研究線（新 line 或新 version
index），不屬本协议的 v3。

## 5. 強制端到端 Preflight 要求

Formal run 前必須先執行 `--smoke`（真實 model + 真實 tokenizer，至少每側 1
筆 prompt 的完整路徑），全部通過（未拋例外、輸出符合 schema）才可啟動
formal run：

1. 全部 §2 的 parent/model/runtime/tokenization 檢查（與 formal 相同，
   fail-closed）。
2. 從 V1 `effects.jsonl` 重算 (15, 8490) 的 δ ∈ {−8, −4, +4, +8} 四點（現行
   `summary()`），驗證各 340 筆。
3. 組裝 A curve 預覽（負側缺口明確標示），不執行 inversion。
4. 生成 2 筆：1 筆 A row 於 δ=−0.25、1 筆 B row 於 δ=0（完整 hook、greedy、
   256 tokens、bf16）。
5. 兩筆輸出通過 `parse_response` 的 JSON/schema 檢查（`json_object` 與
   `schema_valid` 皆 true）。
6. 印出 PASS/FAIL 與 A curve 預覽；不寫入任何 run 目錄。

## 6. 解釋約束與報告計畫

- **Primary outcome**：B reevaluation 上三個 target 的 RMSE 與 per-target residuals。
- Gate 通過時，報告（`report-v2.md`）以論文 Table 2/3 的格式呈現
  （Δ̂、π_B(Δ̂)、target error、validity/parse/schema rates），並明寫 reproduction
  邊界：不同 model（Qwen3.5-4B 非論文模型）、n=340 vs 427-ticker 宇宙、
  RMSE 不與論文絕對值直接比較。
- Gate 失敗時，報告記錄 `fail` 與 pre-registered H2（reevaluation miss），並明寫 confound：
  coordinate 未重選，失敗不能唯一歸因於 grid；後續若要做，屬 v3 或新線。
- 任何結果都不得回填修改 V1/fine-a 的 artifacts 或文件；calibration 或 gate
  的變更必須走 §4。
- 報告完成後更新 [README](README.md) 的 version 矩陣與
  [research-scripts.md](../research-scripts.md) 的 operator 登記。

## 驗證

實作完成後（formal run 前）：

```bash
uv run pytest -q tests/test_investment_dial_calibration_v2.py tests/test_investment_dial.py
uv run python -m compileall -q llm_bias
```

Tests 使用 mocked inference；不載入完整 checkpoint。
