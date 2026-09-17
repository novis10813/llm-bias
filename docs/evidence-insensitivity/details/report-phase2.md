# Evidence-insensitivity Phase 2 報告：capture-layer 組間狀態對比（Qwen＋Gemma 雙模型）

**日期**：2026-09-17
**Runs**：`qwen3.5-4b/phase2-gpu-bf16-01`（capture layer L15，frozen 錨點）、
`gemma4-e2b-it/phase2-gpu-bf16-01`（capture layer L18，Step A 定位）
**協議**：[Phase 2 協議](proposal-phase2.md)（frozen Rev 1.1）
**性質**：描述性（discovery split，無介入、無 causal claim；因果定位在 Phase 3）
**偏差記錄**：Qwen 首兩次 forward 嘗試分別因 device bug（`56c5654` 修）與
bf16→numpy 轉換 bug（`5a95ee9` 修）失敗；第三次 forward 的 analyze 完成但
manifest 被先前 crash 的 stale failed 狀態擋住 finalize——該目錄保留為
`phase2-gpu-bf16-01.attempt-bf16-crash`，正式 run 為其後的重跑，兩版
summary 逐位元一致（4020/4020 per-company 值相同，determinism 成立）。

---

## 1. 結論（TL;DR）

1. **兩模型在 capture layer 的狀態空間皆為單一共享方向**（Qwen L15 PC1
   解釋 88.8%、Gemma L18 PC1 解釋 98.4% 變異），與舊線「L15 狀態 99.5%
   跨公司共享」的背景一致。
2. **閾值故事（offset 差）被拒絕**：Qwen offset 差 d = +0.094（p = 0.57）、
   Gemma d = −0.053（p = 0.66）——evidence-insensitive 組的基線 stance 位置
   與 responsive 組無可測差異。
3. **承載衰減故事（gain 差）也不成立**：Gemma gain 差 d = −0.165（p = 0.18，
   null）；Qwen gain 差 d = −0.425（**p = 0.009，顯著但極小且方向相反**）——
   fixed-sell 組的狀態極性敏感度比 responsive 組**高**約 6%（0.047 vs 0.044，
   state norm ≈ 10.3 的尺度上差 0.003），不是衰減。
4. **狀態層的極性敏感度與行為 contrast 無關**：corr(r_diff, C_c) Qwen
   −0.035、Gemma +0.121。
5. **狀態反應形狀是模型特性**（與 Phase 1 行為差異鏡像）：Qwen 的狀態反應
   近似線性且極性對稱（zero→N15 +0.139、N15→P15 +0.092）；Gemma 是
   **證據存在驅動**（zero→N15 +2.646 vs N15→P15 +0.258，~10× 非對稱，
   nonlinearity 中位數 1.6）——「有任何證據就 buy」在狀態層有對應簽章。
6. **解讀**：行為分組（跟隨 vs 不跟隨）不由 capture layer 的 1D stance
   狀態 offset/gain 承載。結合舊線「決策是微小公司間差的高增益非線性讀出」
   的背景，組差更可能在 state→decision 讀出段（更後層／讀出映射）——這是
   Phase 3 上游因果定位的動機。

## 2. Gates

| Gate | 定義 | 門檻 | Qwen | Gemma |
|---|---|---|---|---|
| G-2A determinism | 20 筆 re-run max \|Δr_stance\| ≤ 0.05 且 0 mismatch | 0.05 | **pass**（0.0197） | **fail**（0.129） |
| G-2B stance 軸有效性 | r2_stance ≥ 0.30 | 0.30 | **pass**（0.331） | **fail**（0.117） |
| G-2C 分組功效 | 各組 ≥ 10（discovery） | 10 | **pass**（42/360） | **pass**（317/85） |
| **gates_passed** | | | **True** | **False** |

**G-2A（Gemma）fail 的解讀**：絕對容差 0.05 是 Qwen 尺度校準的（承載自
balanced-evidence-gap Phase 3 的 bf16 jitter 帶）。兩模型相對 jitter 其實
相同：Qwen 0.0197 / norm 10.34 ≈ **0.19%**；Gemma 0.129 / norm 68.79 ≈
**0.19%**。絕對門檻未隨 state norm 縮放是本次校準限制；frozen gate 結果
維持 fail（不溯及修改），後續 Gemma run 應先測 jitter 帶再定容差（見
協議 Rev 1.2 註記）。依 fail-closed，Gemma Phase 2 的組間對比**不作
gate-pass 宣告**，以下數字為帶此保留的描述。

**G-2B（Gemma）fail**：stance 軸只解釋 11.7% 的極性狀態差變異——Gemma
的狀態反應主要是證據存在驅動（§4.2），1D 極性軸天然低效；Gemma 的
對比結果依協議降級為描述。

## 3. Qwen3.5-4B（L15，gates 全過）

**樣本**：discovery 402 家（42 evidence-responsive / 360 fixed-sell），
3 條件（zero / N15 / P15）× 402 = 1206 狀態 forward。

### 3.1 狀態結構

| 量 | 值 |
|---|---|
| state norm（全記錄） | 中位 10.34（範圍 10.20–10.45，極緊） |
| mean r_stance | zero +0.744 / N15 +0.887 / P15 +0.980 |
| 中位跳躍 | zero→N15 **+0.139**、N15→P15 **+0.092**（極性雙向、近對稱） |
| nonlinearity（3 點拟合最大殘差） | 兩組中位數 ≈ 0.069（狀態反應近線性） |
| stance 軸 R² | 0.331（33% 極性狀態差變異沿均值方向；舊線同類量 60%，新 prompt family 較弱） |
| corr(r_diff, C_c) | −0.035 |
| PCA | PC1 88.8%、PC2 2.5%、PC3–5 各 ~0.4%；兩組 PC1 載荷 −0.94 / −0.97（單一共享方向） |

### 3.2 組間對比（pre-registered）

| 量 | Welch t-test | 組均值（R / FS） | 解讀 |
|---|---|---|---|
| **offset** | d = +0.094，p = 0.573 | 0.872 / 0.870 | **null**——無閾值差 |
| **gain** | d = −0.425，**p = 0.0090** | 0.044 / 0.047 | 顯著但極小（Δ0.003，相對 ~6%）且**方向與衰減假說相反**：fixed-sell 組狀態跟隨極性略強 |
| r(P15) | d = −0.208，p = 0.199 | 0.975 / 0.980 | null |
| r(zero) | d = +0.268，p = 0.109 | 0.754 / 0.743 | null |

Sector 敏感性（gain 的 per-sector Cohen's d）：全部 sector d ≤ 0（fixed-sell
≥ responsive），Information Technology ≈ 0（−0.025，n 14/38，即
responsive 集中的 IT 內無差），Consumer Discretionary 最大（−1.06，
n 5/30）。沒有 sector 顯示反向結構。

**2×2 交叉表退化**：402/402 家 r(P15) ≥ 0（全 positive 格）——stance 軸由
P15−N15 均值差定義，P15 狀態天然投在正側，該交叉表在此設計下無分辨力
（記錄為設計教訓；若需 2×2 應改以 r(zero) 或 r(N15) 為狀態方向）。

### 3.3 解讀

L15 上，42 responsive 與 360 fixed-sell 的**狀態軌跡幾乎重合**（offset
Δ ≈ 0.002，相對 norm 0.02%；gain 差 6% 且方向相反）。行為上「跟不跟隨證據」
的分岔不在 L15 的 1D stance 狀態裡。唯一的顯著效應（gain p = 0.009）方向
與「不靈敏組承載衰減」相反、量級極小，列為描述性發現，不單獨解讀。

## 4. Gemma-4-E2B-it（L18，gates_passed = False）

**樣本**：discovery 402 家（317 evidence-responsive / 85 fixed-buy），
Step A 層定位 16 家 × 3 條件 + 1206 狀態 forward。

### 4.1 Step A 層定位

`L* = argmax_L |pearson(x_L, C_c)|`（16 家子樣本）：**L18（|corr| 0.575）**；
鄰域 L14–22 皆有 0.30–0.56 結構（14: −0.461、19: −0.559、21: −0.543、
20: −0.499、22: −0.490），L34 有孤立 bump（+0.35）。定位在晚期層群，與
Qwen L15/32（47% 深度）的相對位置（18/42 = 43%）大致可比。

### 4.2 狀態結構：證據存在驅動

| 量 | 值 |
|---|---|
| state norm | 中位 68.79（範圍 68.35–71.54） |
| mean r_stance | zero **−13.453** / N15 −10.810 / P15 −10.554 |
| 中位跳躍 | zero→N15 **+2.646**、N15→P15 **+0.258**（**~10× 非對稱**） |
| nonlinearity | 中位數 ≈ 1.6（相對 gain ~0.03 很大） |
| stance 軸 R² | 0.117（G-2B fail） |
| corr(r_diff, C_c) | +0.121 |
| PCA | PC1 98.4%（比 Qwen 更極端的單一方向） |

狀態反應的形狀與 Phase 1 行為直接對應：**加不加證據**移動狀態 2.6 units，
**證據正負**只移動 0.26 units——「有任何證據就 buy」（Phase 1：N6 97%
buy、N15 21% buy、P* 100% buy）在 L18 狀態層有其簽章。

### 4.3 組間對比（描述性，帶 G-2A/G-2B 保留）

| 量 | Welch t-test | 組均值（R / FB） |
|---|---|---|
| offset | d = −0.053，p = 0.655 | −11.607 / −11.601 |
| gain | d = −0.165，p = 0.180 | 0.126 / 0.133 |
| r(P15) | d = −0.090，p = 0.444 | −10.556 / −10.547 |
| r(zero) | d = −0.069，p = 0.563 | −13.455 / −13.443 |

全 null。2×2 交叉表 402/402 negative（軸符號與 Qwen 相反——均值差方向指
向負側；同 §3.2 的退化）。

## 5. 雙模型對照

| 維度 | Qwen（L15） | Gemma（L18） |
|---|---|---|
| 狀態共享結構 | PC1 88.8% | PC1 98.4% |
| 狀態反應形狀 | 極性線性、對稱（+0.14 / +0.09） | 證據存在驅動、10× 非對稱（+2.65 / +0.26） |
| offset 組差（閾值故事） | null（p = 0.57） | null（p = 0.66） |
| gain 組差（承載衰減故事） | 顯著但反向、極小（p = 0.009，FS 高 6%） | null（p = 0.18） |
| 狀態敏感度 × 行為 contrast | −0.035 | +0.121 |
| gates | 全過 | G-2A/G-2B fail（容差校準＋軸有效性） |

**一致結論**：兩個家族、兩種相反的行為結構（Qwen 分級跟隨＋10% 跟隨群；
Gemma 有證據就 buy＋17% 固定 buy），在 capture layer 的 1D stance 狀態上
**都不承載行為分組**。狀態層看到的是各自的「反應形狀」差異（模型特性），
而不是「誰跟隨誰不跟隨」的差異。

## 6. 邊界與限制

- 描述性、discovery split、無介入；causal 定位在 Phase 3。
- Gemma gates_passed = False：其對比不作 gate-pass 宣告（G-2A 容差校準
  限制見 §2；G-2B 軸低效見 §4.2）。
- 2×2 交叉表在兩模型皆退化（軸符號使然），無分辨力——記錄為設計教訓。
- Qwen gain 差（p = 0.009）未做多重比較校正（4 個 contrast 中唯一的
  顯著值）；量級 6%，不作機制宣告。
- 狀態只捕獲 instruction span 終點單位置；其他 span（entity / evidence）
  與更後層不在本 phase（Phase 3 scope）。
- 正式 run 與 crash 版本逐位元一致（determinism 旁證）。

## 7. 對 Phase 3 的輸入

1. **動機**：組差不在 capture layer 狀態 → Phase 3 的 upstream causal
   localization 應聚焦 (a) 更後層（L15 之後，Qwen）的 state→decision
   讀出段、(b) 三 span（entity / evidence / instruction）× 層的 transfer
   patching，找組差被因果寫入的位置。
2. **Qwen gain 差方向**（FS 略高）提示：若 Phase 3 在後層找到組差，其
   機制的候選不是「不靈敏組狀態承載弱」，而是「同樣（或略強）的狀態
   反應經過不同的讀出映射」。
3. **Gemma 容差校準**：後續 Gemma run 先做 jitter 帶實測（相對 ~0.19%
   為本次實測值）再定 G-2A 容差。

## 8. Erratum（2026-09-17，Phase 3 preflight 發現）

Gemma 推理管線（`load_model` → `jlens.from_hf(..., force_bos=True)`）對
有 `bos_token_id` 的 tokenizer 就地設 `add_bos_token=True`（Gemma 預設
False）——推理時序列首部多 1 個 BOS token。Phase 1 存檔 span（含本 phase
capture position＝`instruction_span 終點 −1`）以未含 BOS 的 tokenizer
導出，故 **Gemma run 的實際 capture position 比 frozen 定義早 1 個
token**（instruction span 的次末 token，仍在 span 內）。Qwen
（`bos_token_id=None`，force_bos no-op）不受影響。影響評估：Gemma 全部
state-level 結果（Step A 定位 L18、G-2A/B/C、offset/gain 對比、asymmetric
jump）量測於 instruction span 次末 token——span 邊界內 1 token 的位移，
不改變本報告任何 null 結論（皆為組間對比，span 內相鄰 token 狀態連續）。
若需 confirmation，可用精確終點 token 重測（未執行）；Phase 3 已改用
inference-matched tokenizer 導出座標（見 Phase 3 protocol Rev 1.3）。
