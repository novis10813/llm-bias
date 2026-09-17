# Evidence-insensitivity Phase 3 報告：組差的上游因果定位（Qwen＋Gemma 雙模型）

**日期**：2026-09-17
**Runs**：`qwen3.5-4b/phase3-gpu-bf16-01`、`gemma4-e2b-it/phase3-gpu-bf16-01`
（各 84 家：42 discovery responsive ＋ 42 discovery fixed，seed 20260916）
**協議**：[Phase 3 協議](proposal-phase3.md)（frozen Rev 1.1；Rev 1.2 層格修正、
Rev 1.3 座標系修正，見 §9）
**性質**：線內第一個（也是唯一）干預臂。within-company 極性 transfer
patching：T1（P15 post-block 狀態 → N15 run）、T2（反向），10 層 ×
4 span 位置（entity / evidence / instruction / prompt_end），雙讀數
（Stage 1 decision-position margin scan，6,720 patched forwards/模型；
Stage 2 greedy 128-token generation 於規則選取的 active 座標，
≤672 patched generations/模型）。

---

## 1. 結論（TL;DR）

1. **極性 contrast 的因果中介不在 span 位置的 residual stream 狀態**：
   兩模型、兩方向、兩組別下，entity 位置效應皆**精確 0.0**（天然控制，
   干預管線特異性確認）；evidence 位置 max |ΔM| ≤ 0.022（Qwen）/
   ≤ 0.16 且 L15 後精確 0（Gemma）；instruction 位置 ≤ 0.007（Qwen）/
   精確 0（Gemma）。把 P15 的 evidence/instruction 位置狀態搬進 N15 run，
   任何層都不改變決策讀出。
2. **contrast 整合進 final-position（prompt_end）狀態，且發生在 capture
   層附近**：Qwen 的 T1 ΔM 自 L15（0.041）經 L19（0.540）漸升到 L31
   （1.386）；Gemma 在 L10（≈0）→ L15（1.68）之間**跳變**出現，L28
   飽和於 ~9.6–10.3 nats。T2（必要性方向）兩模型皆為 T1 的近乎精確鏡像
   （Qwen L31：−1.386/−1.200；Gemma L34：−9.691/−10.302）。
3. **S-vs-R 裁決（frozen 規則 §5.2）：兩模型皆為 readout-mediated（R）**——
   两组的 P15 最終狀態都含 pro-buy 訊號（T1 在 84/84 家、兩組皆正向）。
   組差在讀出段，且**方向跨模型相反**：Qwen 的 responsive 組讀出響應
   較大（L31：1.412 vs 1.205，Welch t = 4.54，cohen d = 0.99）；Gemma 的
   fixed-buy 組較大（L34：9.33 vs 10.06，t = −2.75，d = −0.60）。
   L15/L18 instruction 錨點座標兩模型皆 null。
4. **行為端點（greedy decision）零 flip**：四個 active 座標 × 兩方向，
   84 家 × 84 家全部無決策翻轉（Qwen、Gemma 同）。margin 端點在最終層
   移動 +1.2～+2.0 nats（Qwen）／+5～+12 nats（Gemma）卻未跨越任何一家
   的生成決策邊界——生成決策端點對 final-state 注入比 1-token margin
   讀出更穩健（Qwen 45/84 家 N15 margin 已在邊界 1.5 nats 內）。
5. **效應跨全 sector**：active 座標的 T1 效應在 12 個 GICS sector 皆有
   （Qwen）／11 個皆有（Gemma），無 IT-only _artifact。
6. **Gemma 特有**：fixed-buy 組在 N15 的 canonical-prefix margin 中位
   −6.02（sell 側）卻 100% 生成 buy——Gemma 的 fenced-JSON 生成路徑與
   canonical 決策位置讀出是**不同的讀出路徑**，FB 組的 buy 決策不由
   canonical 決策位置讀出承載（§5.3）。

## 2. Gates

| Gate | 定義 | 門檻 | Qwen | Gemma |
|---|---|---|---|---|
| G-3A baseline integrity | 20 筆 unpatched re-run | ΔM ≤ 0.01 且 0 mismatch | **pass**（0.0 / 0） | **pass**（0.0 / 0） |
| G-3B intervention efficacy | (L_final, prompt_end) T1 probe | median ΔM > 0 | **pass**（+1.278） | **pass**（+9.838） |
| G-3C Stage 2 解析功效 | parse 率、每格 n | ≥ 0.95 且 n ≥ 30 | **pass**（1.0 / 84） | **pass**（1.0 / 84） |
| **gates_passed** | | | **True** | **True** |

兩模型 determinism 皆 bit-exact（margin re-run max ΔM = 0.0、decision
re-run 0 mismatch），bf16 forward 下干預管線完全確定。

## 3. 方法執行概要

- 樣本：Qwen 42 家 discovery R（全取）＋ 42 家 FS（seeded）；Gemma 42 家
  R（自 317，seeded）＋ 42 家 FB（自 85，seeded）。
- 層格：Qwen `{0,4,8,12,15,19,23,26,30,31}`；Gemma `{0,5,10,15,18,23,28,32,33,34}`
  （Rev 1.2 修正）。
- Stage 1：每（公司×L×p×direction）一筆 ΔM（6,720 筆/模型）。
- active 座標規則（frozen）：median |ΔM| top-3 ＋ (L_anchor, instruction)。
  Qwen → `[(31,prompt_end),(30,prompt_end),(26,prompt_end),(15,instruction)]`；
  Gemma → `[(34,prompt_end),(28,prompt_end),(33,prompt_end),(18,instruction)]`。
  規則輸出由 |ΔM| 排序決定，prompt_end 座標依规则入选（§7 討論其讀出近旁性）。
- Stage 2：active 座標 × 兩方向的 patched greedy generation（672 筆/模型）
  ＋ 168 筆 baseline generation。

## 4. Qwen3.5-4B（gates 全過）

### 4.1 效應地圖（T1 = P15 狀態 → N15 run，median ΔM，n = 42/組）

| layer | R evidence | R instruction | **R prompt_end** | FS evidence | FS instruction | **FS prompt_end** |
|---|---|---|---|---|---|---|
| 0 | −0.000 | 0.005 | −0.004 | 0.009 | 0.006 | 0.003 |
| 4 | 0.012 | 0.003 | −0.000 | −0.001 | 0.003 | 0.004 |
| 8 | 0.004 | −0.002 | −0.003 | 0.005 | 0.003 | 0.002 |
| 12 | 0.006 | 0.005 | 0.024 | 0.009 | 0.007 | 0.028 |
| 15 | 0.004 | −0.001 | 0.041 | 0.006 | −0.001 | 0.034 |
| 19 | 0.000 | 0.001 | **0.540** | −0.002 | 0.002 | **0.514** |
| 23 | 0.004 | 0.006 | **0.762** | 0.001 | 0.003 | **0.720** |
| 26 | 0.004 | −0.006 | **0.842** | −0.001 | 0.001 | **0.768** |
| 30 | −0.000 | 0.000 | **1.250** | 0.001 | 0.001 | **1.127** |
| 31 | 0.000 | 0.000 | **1.386** | 0.000 | 0.000 | **1.200** |

- **span 位置全域 null**：evidence max |ΔM| = 0.022、instruction 0.007、
  entity **精確 0.0**（T1/T2 皆然）。
- **prompt_end 漸升**：L15 → L19 之間跳升（0.041 → 0.540），L19–L31 持續
  放大。T2 為鏡像（L31：R −1.386、FS −1.200；全層同構）。
- 逐公司分布（T1@31:prompt_end）：R 42/42 正（[1.01, 2.04]）、
  FS 42/42 正（[0.83, 1.62]）——非邊緣公司驅動。

### 4.2 S-vs-R 裁決（§5.2 frozen 規則）

| active 座標 | T1 margin（R vs FS） | Welch t | cohen d | flip（R/FS） |
|---|---|---|---|---|
| (31, prompt_end) | 1.412 / 1.205 | 4.54 | **0.99** | 0 / 0 |
| (30, prompt_end) | 1.277 / 1.129 | 3.53 | 0.77 | 0 / 0 |
| (26, prompt_end) | 0.875 / 0.781 | 2.94 | 0.64 | 0 / 0 |
| (15, instruction) | −0.001 / −0.0004 | −0.10 | −0.02 | 0 / 0 |

兩組皆顯著正向 → **readout-mediated（R）**：pro-buy 訊號在兩組的 P15
最終狀態中都在；組差（R > FS，d 0.64–0.99）在最終狀態之後的讀出響應
幅度。L15 instruction 錨點 null——與 Phase 2「L15 狀態不承載組差」一致。

### 4.3 Stage 2 生成讀出：0 flip 及其意義

84 家 × 4 active 座標 × 2 方向全部無 flip。T1 baseline（N15 run）margin
84/84 為負（R median −1.13、FS −1.86），其中 45/84 在 0 邊界
1.5 nats 內，而最終層注入的 +1.2～+2.0 nat margin 位移仍未改變任何生成
決策。解讀：Qwen 的 greedy JSON 決策（`{ "decision": "X"` 直接生成）
對 prompt_end 最終狀態的單座標替換是**穩健的**——決策不是由單一位置的
最終狀態 margin 直接讀出，而是更分散/更後段的過程；或決策邊界距離大於
可注入的 contrast 幅度。margin 端點（frozen auxiliary）承載了全部結構。

### 4.4 控制與 sector

- entity 控制精確 0.0 → 干預特異於極性 contrast（非一般狀態擾動）。
- T1@active 座標的效應存在於全部 12 個 sector 標籤（11 個 GICS sector
  ＋ Unspecified；部分 cell 為單公司），無 IT 集中 artifact。

## 5. Gemma-4-E2B-it（gates 全過）

### 5.1 效應地圖（T1，median ΔM，n = 42/組）

| layer | R evidence | R instruction | **R prompt_end** | FB evidence | FB instruction | **FB prompt_end** |
|---|---|---|---|---|---|---|
| 0 | −0.016 | 0.054 | 0.040 | 0.005 | −0.050 | −0.013 |
| 5 | −0.103 | 0.049 | 0.044 | −0.158 | −0.037 | −0.109 |
| 10 | 0.231 | −0.093 | 0.045 | 0.194 | −0.114 | −0.012 |
| 15 | 0.000 | 0.000 | **1.675** | 0.000 | 0.000 | **1.871** |
| 18 | 0.000 | 0.000 | **5.382** | 0.000 | 0.000 | **6.334** |
| 23 | 0.000 | 0.000 | **9.105** | 0.000 | 0.000 | **9.749** |
| 28 | 0.000 | 0.000 | **9.608** | 0.000 | 0.000 | **10.174** |
| 32 | 0.000 | 0.000 | 9.583 | 0.000 | 0.000 | 10.176 |
| 33 | 0.000 | 0.000 | 9.587 | 0.000 | 0.000 | 10.215 |
| 34 | 0.000 | 0.000 | **9.691** | 0.000 | 0.000 | **10.302** |

- **L10 → L15 跳變出現**（~0 → 1.68/1.87），L18–L28 快速飽和（~10 nats）。
  注意跳變發生在 **L15**——與 Qwen 的 capture 錨點同層，且低於本線
  Phase 2 為 Gemma 定位的 L18 capture 層（L18 時 contrast 已成形但未飽和）。
- **span 位置**：entity 精確 0.0；evidence/instruction 在 L15 後**精確
  0.0**（bf16 解析度下位元一致），L0–L10 僅 ±0.16 量級雜訊。
- T2 鏡像（L34：R −9.691、FB −10.302）。
- 逐公司（T1@34:prompt_end）：R 42/42 正（[5.32, 11.90]）、
  FB 42/42 正（[8.09, 12.24]）。

### 5.2 S-vs-R 裁決（§5.2 frozen 規則）

| active 座標 | T1 margin（R vs FB） | Welch t | cohen d | flip（R/FB） |
|---|---|---|---|---|
| (34, prompt_end) | 9.33 / 10.06 | −2.75 | **−0.60** | 0 / 0 |
| (33, prompt_end) | 9.27 / 10.00 | −2.77 | −0.60 | 0 / 0 |
| (28, prompt_end) | 9.25 / 9.99 | −2.80 | −0.61 | 0 / 0 |
| (18, instruction) | 0 / 0（零變異） | — | — | 0 / 0 |

兩組皆強正向 → **readout-mediated（R）**；組差方向與 Qwen **相反**
（fixed-buy 組讀出響應較大，d ≈ −0.60）。(18, instruction) 錨點：兩組
皆精確 0（零變異，Welch 未定義，按 null 記錄）。

### 5.3 Gemma 特有：margin 端點與生成決策的分離

| 組 | T1（N15）baseline margin median | T1 baseline decision | T2（P15）baseline margin median | T2 baseline decision |
|---|---|---|---|---|
| responsive | −8.03 | sell（42/42） | +1.52 | buy（42/42） |
| fixed-buy | **−6.02** | **buy（42/42）** | +4.43 | buy（42/42） |

Gemma 生成 fenced JSON（` ```json\n{...`），其決策 token 的條件化包含
fence token，與 canonical-prefix margin 讀出（`{ "decision": "` 後一位）
是**不同的讀出路徑**。fixed-buy 組在 N15 的 canonical 決策位置讀出偏向
sell（median −6.02）卻 100% 生成 buy——FB 組的 buy 吸引子不由 canonical
決策位置讀出承載。更極端的是：Gemma T1 baseline 84/84 家距離 margin
邊界 ≥ 2.1 nats，最終層注入位移 +5～+12 nats（在 margin 尺度上足以跨越
邊界），生成決策卻 0 flip——fence 決策路徑與 canonical margin 路徑幾乎
完全解耦。故 Gemma 的 margin 端點讀數（含 §5.1/§5.2 的幅度與
組差）應理解為 canonical-prefix 讀出路徑上的 state response，與 Phase 1
行為端點（生成決策）不互推。此分離本身是 FB 組機制的候選簽章。

### 5.4 控制與 sector

- entity 控制精確 0.0；sector 表：active 座標效應存在於全部 12 個 sector
  標籤（11 個 GICS ＋ Unspecified；部分 cell 單公司），無 sector artifact。

## 6. 雙模型結構對照（§5.6，比結構不比幅度）

| 結構特徵 | Qwen | Gemma |
|---|---|---|
| span 位置因果中介 | null（entity 精確 0） | null（entity 精確 0） |
| contrast 所在位置 | prompt_end（final-position 狀態） | prompt_end |
| 出現層 | L15 → L19 漸升 | L10 → L15 跳變（L28 飽和） |
| T2/T1 對稱性 | 鏡像（±1.386/1.200 @31） | 鏡像（±9.69/10.30 @34） |
| S-vs-R 裁決 | readout-mediated（R > FS，d ≈ +1.0） | readout-mediated（FB > R，d ≈ −0.6） |
| capture 錨點（instruction 位置） | null | null |
| 生成決策 flip | 0/84 | 0/84 |
| sector 覆蓋 | 全 12 sector | 全 11 sector |

**共性**：極性 contrast 不在 span 位置的 residual 狀態，而在整合後
final-position 狀態（capture 層附近成形）；兩組狀態都帶 pro-buy 訊號
（R 裁決）；行為端點對單座標 final-state 注入穩健。
**差異**：成形動力學（漸升 vs 跳變）、飽和幅度（模型 margin 尺度
特性）、組差方向相反（讀出差是模型特性，非「不靈敏組讀出較弱」的
通用性質）。

## 7. 解讀與文脈

1. **對 Phase 2 null 的補全**：Phase 2 找到 capture layer 的 1D stance
   狀態不承載組差；本 phase 找到極性 contrast 的因果內容在
   final-position 狀態（capture 層附近寫入、晚期層放大）。兩結果一致：
   contrast 不在 span 局部狀態、也不在 1D stance 軸，而在整合後的
   final-position 表示——與舊線「決策是微小公司間差的高增益非線性讀出」
   （entity-concept 收線）及 selective-intervention「L15 k=8
   entity-difference 子空間移除有效」共同指向：**晚期 final-position
   狀態 → 讀出映射**是 entity-bias 行為的分岔點，而組間差在讀出映射
   （幅度/方向皆模型特性）。
2. **active 座標的讀出近旁性**（誠實討論）：frozen 規則按 median |ΔM|
   選 active 座標，prompt_end 座標因「patch 越靠近讀出、剩餘下游處理越
   少」而系統性佔優，Stage 2 的 generation 讀出因此落在讀出近旁座標。
   該座標的 T1 效應部分反映 final 狀態 contrast 本身的大小（L31/L34
   即 contrast 的終值），不完全是「上游因果位置」。上游 span 座標已在
   Stage 1 全格量測為 null——這是本 phase 的主要因果定位結果；Stage 2
   的 0 flip 則是行為端點穩健性的負結果。若未來版本要以行為 flip 為
   端點，需重新設計座標規則/端點（§9 版本觸發）。
3. **Gemma L15 跳變**：Gemma 的極性 contrast 在 L15（非其 Phase 2
   capture 層 L18）一步成形——L15 對兩模型都是 contrast 的「寫入層」
   候選，值得在後續 confirmation 中針對性檢查。

## 8. 限制

- discovery split only（Qwen 42R/42FS 自 402 discovery；Gemma 42R/42FB
  自 402 discovery）；hold-out（Qwen 8R/93FS、Gemma 78R/23FB）保留給
  confirmation。
- 單位置替換（每座標 1 token）；未測多位置組合（V2 選項）。
- sector cell n = 1–8，sector 表僅描述性。
- bf16 forward：低於 bf16 解析度的效應記 0.0（Gemma L15+ span 位置的
  精確 0 含此可能；Qwen 同）。
- Gemma margin 端點為 canonical-prefix 讀出路徑（§5.3），與生成決策
  端點分離；雙模型幅度不可互比。
- 層格間隔（Qwen 4–5 層、Gemma 5 層）：L15–L19（Qwen）與 L10–L15
  （Gemma）的成形動力學在格間隔內，精確寫入層未分辨。

## 9. 偏差與 errata 記錄

- **層格（Rev 1.2）**：Gemma text tower 實為 35 層（Rev 1.1 誤書 42）；
  frozen 層格 36/41 不存在，`record_block_states` fail-closed 拒跑（Gemma
  尚無任何 forward）。依規則修正為 `{0,5,10,15,18,23,28,32,33,34}`。
  Phase 2 proposal 同處誤記，另立 Rev 1.3 erratum。
- **BOS 座標系（Rev 1.3）**：jlens `force_bos=True` 就地把 Gemma
  tokenizer 設 `add_bos_token=True`，prepare（未修正 tokenizer）與
  forward（修正後）編碼差 1 token，全部 patch 座標位移 1（preflight 測得
  全 coordinate ΔM = 0，G-3B 將 fail-closed）。修正：
  `core/model.load_tokenizer_for_inference`＋char-range 重導出＋
  fail-closed cross-check＋forward 時 coordinate guard。三個無效 attempt
  目錄保留為記錄：`phase3-gpu-bf16-01.attempt-old-grid`、
  `.attempt-bos-drift`、`.attempt-partial-bos`（皆無有效 Gemma forward）。
  Qwen `bos_token_id=None`（force_bos no-op），不受影響、未重跑。
- **Phase 2 Gemma capture 位置**：受同一 BOS 問題影響早 1 token
  （instruction span 次末 token）；見 [Phase 2 報告 §8 erratum](report-phase2.md)。
- **impl 對齊**：Stage 1 scan 記錄 margin-only（§3.2 的 scan 行
  baseline_decision 由 generation_records 承載）；net_direction 未另存欄位
  （可由 record 導出）。

## 10. 對後續的輸入

1. **Confirmation（若執行）**：hold-out 上重測 active 座標
   （Qwen 8R/93FS、Gemma 78R/23FB），檢定 (a) span-null、(b)
   prompt_end 層結構、(c) S-vs-R 組差方向是否保持。建議同時加測
   Gemma L15 寫入層假說（L13–L16 密格）。
2. **端點 V2（若做）**：margin-crossing 行為端點（注入量級 × 邊界距離
   配對的 flip 設計）或多位置 patch；任一改變皆依 §9（協議）版本分立。
3. **與 selective-intervention 合流**：L15 區域（entity-difference
   子空間 vs 極性 contrast final 狀態）的因果內容重疊度值得一次
   對照實驗（entity-dial 線候選）。
