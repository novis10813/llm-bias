# Selective-intervention V1 report（L15 k=8 entity-contrast subspace removal）

**檔案狀態**：本報告對應 protocol
[`docs/selective-intervention/proposal-v1.md`](proposal-v1.md)（Rev 1 frozen；
Rev 1.1–1.5 實作對齊筆記與 run 記錄）的第一次 formal run
（`selective-intervention-v1-gpu-bf16-01`，2026-09-14，local）。**結論：gate
verdict 為 `fail`——G1a/G1b/G2 pass（efficacy 與 subspace specificity 成立）、
G3/G4 fail（full strength 有全局副作用）；依 frozen 決策表，V1 於 full
strength 為負結果（不預先註冊 dose 救援），但完整 dose-response 曲線與
specificity 對照均已記錄。** Paper provenance 與版本矩陣見 [README](README.md)。

## 摘要

- Run `selective-intervention-v1-gpu-bf16-01` 一次完成 663 forwards（646 arm＋
  17 calibration），6 個 registered artifacts 全部 `complete`，manifest
  `status=complete`；clean 臂 64/64 對 2A 存檔 bit-exact（max |ΔM| = 0.0；
  另以 artifacts 獨立重算 0 mismatches 驗證）。
- 主臂（L15 instruction span、k=8 entity-contrast subspace、cloud
  centering、α=1.0）把 frozen group gap 由 1.2697 縮減至 0.5863（−53.8%）、
  entity-contrast IQR 由 0.5640 縮減至 0.2800（−50.2%）→ **G1a、G1b
  pass**。
- Random 8 維正交子空間對照只縮減 group gap 0.49%（1.2697→1.2635，mean
  shift +0.0016 nats）→ **G2 pass**：縮減高度特異於 entity-contrast
  subspace。
- Full strength 把 64-prompt mean margin 由 −1.7500 推至 −1.4201（+0.3299
  nats > 0.15）→ **G3 fail**；anonymous prompt 的 margin 由 −3.2280 移至
  −3.4968（Δ −0.2689，|·| > 0.10）→ **G4 fail**。
- 依 frozen 決策表「G3 或 G4 fail → full-strength 有全局副作用；報告 dose
  曲線，V1 於 full strength 為負結果（不預先註冊 dose 救援）」：V1 收口為
  **full-strength 負結果＋efficacy/specificity 陽性**。

## Run 與 provenance

- **Run ID**：`selective-intervention-v1-gpu-bf16-01`
- **Run root**：`artifacts/qwen3.5-4b/selective-intervention/runs/selective-intervention-v1-gpu-bf16-01`
- **Protocol**：`docs/selective-intervention/proposal-v1.md`（protocol rev 1，frozen）
- **Operator**：`scripts/selective_intervention_v1.py`（SHA-256
  `8d02de859a5741cdc30d0ac9e82eac5879fda9e7fb95f5e3c08a66f2f9050dff`）
- **Source identity**：git commit `c360b011975694a2405802f4ea6ca48c1d456322`
  （本 run 執行時 head；run 期間無 source 變更）
- **Upstream runs**（prepare fail-closed 驗證通過）：
  - 2A population 與 archived margins：
    `artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01`
    （64 = 16 公司 × 2 reverse × 2 order）
  - Basis source：`artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01`
    （`pca_basis_vectors` 前 8 rows；basis SHA-256
    `cc048c3bfb1859dc76a8418efca9a1756f7c774124e737afe5a32150c534b21c`；
    singular range [3.5065, 1.1830]）
- **Random control basis**：frozen seed 20260913 的 8 維正交子空間（SHA-256
  `47ebbd3fc113a6d057b3150ad324b7eac0274fdb087b6e9032912211685a5f27`）
- **Frozen groups**：TOP = {BLK, NSC}、BOTTOM = {BDX, IT}（group gap 定義於
  此 4 家公司共 8 prompts；其餘 12 家為 population 背景）
- **Anonymous prompt**：SHA-256
  `53b658a85a105e25bc539e8f6207eb91c2f9c70228b84816440b1cc98bdabeae`
  （與 entity-to-dial 線同一 identity-stripped string）
- **Model / runtime**：`.cache/models/qwen3.5-4b`（bf16）、torch
  bfloat16、cuda:0（RTX 3060）；runtime 1130 s（~19 min，快於 formal 預算
  65 min）
- **Smoke preflight**：`selective-intervention-v1-smoke-20260913T060259Z`
  PASS（127 forwards，clean 16/16 bit-exact；前兩次 smoke 被 fail-closed
  預檢擋下——margin 減法 float 語義與 transform 構造的 device 不匹配——皆
  為實作 bug，已修並附回歸測試，見 proposal Rev 1.1 B5 與本次 Rev 1.5 附
  錄）
- **Forward 筆數**（與 protocol §5 budget 一致）：clean 64＋dose 256＋
  centering 128＋scope 64＋random 64＋layer 64＋anon 2＋dial 4 = 646 arm
  forwards＋17 calibration forwards（16 named＋1 anonymous full grid）
- 沒有 automatic resume；本 run 一次完成，中斷重跑需新 run ID

## Gate 結果

| Gate | 定義（frozen） | 值 | 門檻 | 判定 |
|---|---|---|---|---|
| G1a | group gap 半減 | 0.5863 | ≤ 0.6348 | **pass** |
| G1b | entity-contrast IQR 半減 | 0.2800 | ≤ 0.2820 | **pass** |
| G2 | random 縮減 ≤ 25% 主臂縮減 | 0.0049（主臂 0.5382） | ≤ 0.1346 | **pass** |
| G3 | 無全局推注 | +0.3299 nats | ≤ 0.15 | **fail** |
| G4 | anonymous 穩定 | −0.2689 nats | ≤ 0.10 | **fail** |
| **verdict** | | | | **fail** |

參考值：`group_gap_clean = 1.2697`、`group_gap_intervention = 0.5863`、
`iqr_clean = 0.5640`、`iqr_intervention = 0.2800`、
`mean_clean = −1.7500`、`mean_intervention = −1.4201`、
`anon_clean = −3.2280`、`anon_intervention = −3.4968`。

G1b 貼邊通過（0.2800 vs 0.2820，99.3% 於門檻）。G3 與 G4 的 fail 方向一致：
子空間移除同時帶走了 entity-contrast 與一個全局 stance 分量。

## Dose-response 曲線（primary centering，L15 instruction span）

| α | group gap | entity-contrast IQR | mean margin | mean shift | sell→buy flips |
|---|---|---|---|---|---|
| 0（clean） | 1.2697 | 0.5640 | −1.7500 | — | 0（64/64 sell） |
| 0.25 | 1.0784（−15.1%） | 0.4840 | −1.6582 | +0.0918 | 0 |
| 0.50 | 0.9120（−28.2%） | 0.4040 | −1.5793 | +0.1707 | 0 |
| 0.75 | 0.7394（−41.8%） | 0.3489 | −1.4969 | +0.2531 | 0 |
| 1.00 | 0.5863（−53.8%） | 0.2800 | −1.4201 | +0.3299 | 0 |

曲線在四個 level 上對 group gap、IQR 與 mean shift 皆單調且近線性；無任何
decision flip（全 population 保持 sell）。描述性觀察（非 pre-registered
rescue）：G3 門檻 0.15 介於 α=0.25（+0.0918）與 α=0.50（+0.1707）之間。

## Controls 與 probes

- **Specificity（G2 數據）**：random 8 維子空間 group gap 1.2635（縮減
  0.49%）、mean shift +0.0016 nats——幾乎無效，主臂縮減為 subspace 特異。
- **Layer control**（同 16-prompt canonical subset，α=1，L15-fitted basis）：
  L13 1.0960、L14 0.8945、**L15 0.7661**、L16 1.0100、L17 1.1622。L15 為
  五層最小（最有效），與 entity-to-dial 收線的 L15 定位一致；L14 次低、
  其餘層接近 clean 1.2697。
- **Centering 對照**（α=1）：cloud 0.5863（shift +0.3299）vs zero 0.5783
  （+1.0948）vs anon 0.5789（+1.0611）——三種 centering 的 gap 縮減幾乎
  相同（≈0.58），但 cloud centering 的 mean shift 只有 zero/anon 的三成，
  支持 entity-contrast center 為正確 primary。
- **Position scope 對照**（α=1）：instruction 0.5863（shift +0.3299）vs
  full sequence 0.1236（+0.3875）——full scope 把 group gap 壓得更深但
  mean shift 略大；instruction span 定位（entity-to-dial 收線）足以承載
  大部分效果。
- **Anonymous probe**（α=1，reference row）：−3.2280 → −3.4968（Δ −0.2689）
  → G4 fail 數據；identity-stripped prompt 也被推注，確認全局分量存在。
- **Dial probe**（reference row，L15/N8490 mlp_addition）：+4 push ΔM
  clean +6.9148 / 介入後 +7.0302；−4 push ΔM clean −2.9580 / 介入後
  −2.8402——介入後投資-dial 旋鈕仍完全可控（差異 < 0.12 nats），subspace
  removal 不損傷 MLP dial 通道。
- **Order consistency**：per-company `(M̄_ord1 − M̄_ord0)` 於 clean vs
  α=1 的最大位移 0.4314 nats（描述性）。

## Per-company 表（clean vs α=1 primary 臂）

| ticker | sector | clean mean | int mean | shift | clean contrast | int contrast | flips |
|---|---|---|---|---|---|---|---|
| IT | Information Technology | −2.5154 | −1.7905 | +0.7248 | 0.7126 | 1.4375 | 0 |
| HPE | Information Technology | −2.1883 | −1.5855 | +0.6028 | 1.0397 | 1.6425 | 0 |
| SYK | Health Care | −2.2143 | −1.6297 | +0.5846 | 1.0137 | 1.5983 | 0 |
| BDX | Health Care | −2.3188 | −1.7364 | +0.5824 | 0.9092 | 1.4915 | 0 |
| GLW | Information Technology | −2.0105 | −1.5657 | +0.4449 | 1.2175 | 1.6623 | 0 |
| C | Financials | −1.7751 | −1.3474 | +0.4277 | 1.4529 | 1.8806 | 0 |
| HON | Industrials | −1.8580 | −1.4529 | +0.4051 | 1.3700 | 1.7751 | 0 |
| CSX | Industrials | −1.6591 | −1.3457 | +0.3134 | 1.5689 | 1.8823 | 0 |
| DE | Industrials | −1.5199 | −1.2316 | +0.2884 | 1.7081 | 1.9964 | 0 |
| DHR | Health Care | −1.6867 | −1.4056 | +0.2811 | 1.5413 | 1.8224 | 0 |
| ABT | Health Care | −1.6741 | −1.4184 | +0.2557 | 1.5539 | 1.8096 | 0 |
| GS | Financials | −1.5001 | −1.3142 | +0.1859 | 1.7279 | 1.9138 | 0 |
| AMAT | Information Technology | −1.4637 | −1.3096 | +0.1542 | 1.7643 | 1.9184 | 0 |
| NSC | Industrials | −1.2814 | −1.1684 | +0.1130 | 1.9466 | 2.0595 | 0 |
| AXP | Financials | −1.3216 | −1.2339 | +0.0877 | 1.9064 | 1.9941 | 0 |
| BLK | Financials | −1.0134 | −1.1859 | −0.1725 | 2.2146 | 2.0421 | 0 |

（按 shift 降序排列；contrast = 每公司 4 個 variant margins 的 max−min。）

兩個模式：(1) shift 大小與 clean sell-bias 強度單調相關——最 sell-biased
的 IT/HPE/SYK/BDX/GLW 被推高最多（+0.44～+0.72），最不偏的 AXP/NSC 幾乎
不動、BLK（clean 最不偏）唯一被往下壓（−0.1725）；(2) per-company
entity contrast 幾乎全數增大（尤其最偏的幾家翻倍：IT 0.71→1.44、HPE
1.04→1.64）——組間 group gap 縮小同時，組內對 evidence 排列的敏感度
上升，這是 G3/G4 之外的第二個全局效應。

## 解讀

1. **Efficacy 與 specificity 成立（M5 主要宣稱的核心部分）**：L15
   instruction span 的 k=8 entity-contrast subspace removal 在 frozen
   population 內把 group gap 縮減 53.8%、IQR 半減，而 random 子空間
   對照幾乎無效（0.49%）。這與 entity-to-dial 收線（k=8 子空間承載 98.3%
   的 entity 因果 transfer、L15 為因果層）互相印證：entity-induced group
   gap 的因果載體是一個低維、層特異、位置特異的子空間。
2. **Full strength 帶全局副作用（G3/G4 fail）**：同一子空間同時承載一個
   全局 stance 分量——移除後 64-prompt mean margin 上推 +0.3299 nats、
   anonymous prompt 也被推注 −0.2689 nats。cloud centering 已把
   center 對準 entity-contrast 的 company cloud，但仍不能完整隔離該
   全局分量（zero/anon centering 的 shift 更大：+1.09/+1.06，方向一致）。
3. **依 frozen 決策表收口**：「G3 或 G4 fail → full-strength 有全局副作用；
   報告 dose 曲線，V1 於 full strength 為負結果（不預先註冊 dose
   rescue）」。dose 曲線完整記錄於上：效果與副作用皆隨 α 單調近線性
   縮放，α=0.25 的 mean shift（+0.0918）落在 G3 門檻內但 gap 縮減也較小
   （15.1%）——描述性記錄，非 V1 結論。
4. **Per-company 模式**：shift 與 clean sell-bias 單調相關（最偏者修正
   最多），方向上是 de-biasing 簽章；但 entity contrast 普遍增大是未
   預期的第二效應，限制「只動 entity 分量」的解釋。
5. **與 dial 的互動**：L15/N8490 MLP dial 通道在介入後仍可控（ΔM 差異
   < 0.12 nats）；subspace removal 與 dial 通道（Phase F 已證實幾何
   獨立：cos ≈ −0.02）互不干擾。

## 邊界與限制

- 單模型（Qwen3.5-4B）、單一 frozen 64-prompt population；group gap 定義
  於 4 家極端公司（frozen groups），12 家背景公司的 per-company 效應見
  上表但不入 gate。
- 計分為 fixed answer-token margin（不生成）；不宣稱生成端行為。
- G3/G4 門檻為 frozen pre-registered 值，本報告不做門檻重校或 dose
  rescue；任何低劑量或 side-effect-aware 設計屬 V2 提案範圍。
- Layer control 的 population 為 16-prompt canonical subset（frozen 設計，
  §3 arm matrix），與主臂 64-prompt 不同 population，用於層特異性方向性
  比較。
- 全部 64 筆 clean margin 與 2A 存檔 bit-exact 驗證於 run 內 fail-closed
  執行；run 後另以 artifacts 獨立重算 0 mismatches 驗證（bit-exact 計數
  自本次起記錄於 forward metadata，見 proposal Rev 1.5 B6）。

## Artifact 索引

- Run：`artifacts/qwen3.5-4b/selective-intervention/runs/selective-intervention-v1-gpu-bf16-01/`
  - `prepare/rows.jsonl`（65 = 64 named＋1 anonymous；SHA-256
    `07452568922a5a85ecb967eabbd4868334306758b7217f3ee72d8250e3282a5e`）
  - `prepare/provenance.json`（basis/random basis/anonymous/groups 指紋）
  - `forward/records.jsonl`（646 records；SHA-256
    `932c05f7d7e4e2bc5b528ce256e9ad667a98dd6126d8f6d33255b527649b944f`）
  - `forward/metadata.json`（calibration digests、m_anon、forward 計數）
  - `analyze/summary.json`（gates、dose、controls、per-company；SHA-256
    `c7064c33dd11a803da425083412ebba8f400d5dc4d100a2c0423865e08653aab`）
  - `manifest.json`（status complete）
- Smoke：`artifacts/qwen3.5-4b/selective-intervention/runs/selective-intervention-v1-smoke-20260913T060259Z/`
  （PASS；前兩次 smoke run 為 fail-closed 預檢擋下的偏差記錄，保留原狀）

## 附錄：Run 期間的實作修正

- **B5（proposal Rev 1.1）**：`margin_from_log_probs` 的減法由 FP32 張量
  減法改為 float64（先轉 Python float），與 2A 參照機制
  `score_single_token_margin_fp32` bit-for-bit 一致；首次 real-model smoke
  被 clean 預檢以 5e-8 差異擋下。附回歸測試。
- **B6（本次 Rev 1.5）**：forward 構造時 orthonormal 檢查的 `torch.eye`
  device 不匹配（GPU-only）修正；forward metadata 新增
  `n_clean_bit_exact` / `clean_max_abs_delta_m` 審計欄位；summary 新增
  `random_control` 描述欄位。皆為 additive 實作修正，不改變 frozen 設計
  或 gate 邏輯。
