# Phase 1 V2 開發 smoke：L15 k=8 內客戶集中度可讀出，景氣敏感性不可

**狀態：development／model_smoke，非正式 audit，`scientific_status=not_evaluated`。**run `phase1-v2-development-smoke-01`，manifest SHA-256 `eb56316616717fbd390be90e8138999774717ff213f5ef9d41cac2b56f5edc17`，三階段 complete。65 次 forward（60 材料＋4 公司＋首筆重複），48.9s，GPU 峰值 11.7GB，重複前向 margin 差 0.0、mean 差 0.0（bf16 確定性）。模型 `.cache/models/qwen3.5-4b`（config `ddc63e1c…`），層 15，k=8 基底 `[2560,8]` 取自 e-01，seed 1729。材料來源 SHA `ff28cc9b…`／`ab20ca9d…`。

> **已被 smoke-03 修正：** 本 run 判讀「C 具概念特異性」是通用 stance 混淆的假陽性。新增 stance 正交化後，C 方向 88% 是 stance 軸、公司間差異消失。見 [smoke-03 修正紀錄](report-phase1-v2-development-smoke-03.md)。

## 1. 量測內容

固定 e-01 k=8 子空間 Q，對每筆 L15 instruction mean 求候選方向分數（`scalar`），對 final residual 求 buy−sell margin（FP32）。候選方向只用 primary 行、按群等權 fitting；對照行只測反應、不進 fitting。LO-group：用「去掉一個改寫群」後的方向去分那被拿掉的群。random：Q 內 16 個 seed 單位方向對同比較。全部為描述性，非校準 gate。

## 2. 客戶集中度（C）：真實且具特異性的 k=8 方向

- **LO-group 一致勝 random**：方向只在 C2 fit 時，C1 五對中四對 held-out delta 超過 random q95（+0.035 至 +0.058）；只在 C1 fit 時，C2 三對全勝（+0.025 至 +0.053）。群均值 +0.0315／+0.0407。唯一落帶的是 C-V01（累積帳本表述，−0.0158）。
- **概念特異性**：固定評價時概念差仍為正（E01 `concept_at_positive` +0.0605、`concept_at_negative` +0.0273，皆勝 q95）；無關競爭內容（X01 員工數）落在 random 帶內（+0.0037）。
- **須誠實記錄的混淆**：(a) 該方向同時對一般正負評價有帶外反應（`evaluation_at_*` −0.054 至 −0.092），即含一股通用 stance 成分；(b) 數值對照 L01（75% vs 15%）帶外 −0.0268，存在數字表面捷徑。故 C 是「客戶集中度＋部分通用立場／數字」的混合方向，尚非乾淨的公司相關概念。
- 保留比例（k=8 內）：C 全資料 0.329；C1 0.297、C2 0.295，兩群相似 0.99。
- 四家旧公司 C 分數（未中心化、只看相對）：IT 0.088、BDX 0.083、NSC 0.060、BLK 0.031。

## 3. 景氣敏感性（S）：未達可讀出

- 群內 held-out 群均值僅 +0.0063（S1，5/7 正）與 +0.0127（S2，2/3 正），大多落在 random q05–q95 帶內；S2 held-out 幾乎不勝 random。
- 保留比例：S 全資料 0.198；S1 0.147、S2 0.228，S1↔S2 相似僅 0.719（兩群方向分歧，互相抵消）。
- 四家旧公司 S 分數全部強烈負且差異小（−0.558 至 −0.615），像被共同負偏移主導，不含可分公司的景氣訊號。
- 判讀：在現行材料與 k=8 內，S 未呈現超越 random 的獨立概念方向；不宣稱 S 不存在於其他層／維度。

## 4. 能與不能宣稱

- 能：在固定 e-01 L15 k=8、本批 60 筆 AI 合成材料內，存在一個對「客戶收入集中度」具 held-out 一致性、超過 random、且部分概念特異的方向。這是開發級正號，值得投入清理。
- 不能：(1) 非正式 audit，`n_independent_families=None`、未達統計功效、random 僅 16 個，勝 q95 是弱基準非校準門檻；(2) C 含通用 stance 與數值捷徑混淆，未與一般評價解耦；(3) 材料為 AI 合成、未經獨立人審；(4) 未測決策因果（Phase 2 內容）；(5) 未做 J-lens 詞彙讀出，不稱完成 P1a 詞表提名。

## 5. 下一步（依此結果收斂）

1. 以 C 為主線，擴充**held-out** 改寫群＋**評價正交**對照（讓正負極的通用評價配平），把「集中度」與「通用立場／數字」分離；S 降為候選或撤換。
2. 在 C 方向上，用未見公司 prompt 查自然狀態是否呈現同樣集中度差，而非只理解明示材料。
3. 補 independent 人工盲審與更大 random/label-permutation 集，才考慮凍結任何正式門檻。

本文件不設 gate、不存高維 vector；所有數字可自 run 的 `analyze/summary.json`、`analyze/comparisons.jsonl`、`forward/records.jsonl` 重現。
