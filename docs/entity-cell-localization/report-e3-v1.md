# Entity Cell Localization: Report E3 V1 (Suppression Discovery)

**Status:** E3 V1 Discovery run complete (`entity-cell-e3-discovery-v3`).
Causal entity specificity confirmed on candidate `(L0, N104)` for FTNT
against within-ticker and same-collision cross-ticker controls. Downstream
identity source attenuation confirmed with 100% evidence preservation.
Calibration and held-out test not run. Version index: [README](README.md).

> **Instrument note:** the v3 margins above were computed with the v1 shared-core FP32 tail, which was wrong for Qwen3.5's final norm. Re-verification under the corrected v2 instrument (`entity-cell-e3-discovery-v4`) confirms all frozen gates — see [§6](#6-instrument-revision-re-verification-v2-instrument-2026-09-03).

**Protocol:** [proposal-e3-v1](proposal-e3-v1.md).

---

## 1. Runs on Record

All runs located under `artifacts/qwen3.5-4b/entity-cell-localization/runs/`:

| run | state | note |
|---|---|---|
| `entity-cell-e3-discovery-v1` | failed (preserved) | timed out at 600s in bash before completion |
| `entity-cell-e3-discovery-v2` | failed (preserved) | `e3-upstream` completed (1,116 rows); failed at `e3-downstream` due to `random_subset` position coverage bug, fixed in `4af77f9` |
| **`entity-cell-e3-discovery-v3`** | **complete** | **Formal E3 V1 discovery run** (324 upstream suppression rows, 60 downstream attenuation rows, 22 analysis groups; GPU 0 bf16, 6m21s) |

---

## 2. E3-A Upstream Suppression Results

Evaluated across doses $\alpha \in \{1.0, 0.5, 0.0, -1.0, -2.0, -3.0\}$ on 3 financial prompts per ticker.

### A. Target Ticker (FTNT) Within-Ticker Contrasts

| Condition | Scope | Cell | Clean Margin | Intervened Margin ($\alpha=-3$) | Anonymous Progress $A_p(-3)$ | Decision Flips |
|---|---|---|---|---|---|---|
| **Target** | `all_positions` | **(0, 104)** | 0.7939 | 0.7068 | **+0.0758** | 0/3 |
| **Wrong Entity** | `all_positions` | (0, 5101) [FTV] | 0.7939 | 0.7752 | **-0.0025** | 0/3 |
| **Matched Random** | `all_positions` | (0, 500) | 0.7939 | 0.8167 | **-0.0501** | 0/3 |
| **Target** | `header_only` | **(0, 104)** | 0.7939 | 0.6932 | **+0.0969** | 0/3 |
| **Wrong Entity** | `header_only` | (0, 5101) [FTV] | 0.7939 | 0.8088 | **-0.0412** | 0/3 |
| **Matched Random** | `header_only` | (0, 500) | 0.7939 | 0.7947 | **-0.0244** | 0/3 |

**Observation**: FTNT target suppression moved the margin towards the Anonymous baseline ($A_p > 0$), while both the non-degenerate wrong-cell control and matched-random control produced flat or negative progress.

---

### B. Cross-Ticker Specificity Contrasts (Same-Collision & Non-Collision Peers)

Suppressing the **exact same neuron `(0, 104)`** across peer tickers:

| Ticker | Role in E1 V2 Localization | Clean Margin | Intervened Margin ($\alpha=-3$) | Anonymous Progress $A_p(-3)$ | Specificity Contrast vs FTNT $\Delta A_p$ |
|---|---|---|---|---|---|
| **FTNT** | **Target** (sole trusted candidate) | 0.7939 | 0.7068 | **+0.0758** | — |
| **ADI** | Peer (shared top-1 at N104, failed form-robust) | -0.3426 | -0.4325 | **-0.3416** | **+0.4173** |
| **MU** | Peer (shared top-1 at N104, failed form-robust) | -0.2704 | -0.2527 | **-0.3321** | **+0.4079** |
| **FTV** | Peer (different top-1 at N5101) | -1.2477 | -1.3366 | **+0.0965** | **-0.0207** |

#### Mean Peer Contrast ($\mathcal{P} = \{\text{ADI}, \text{MU}\}$):
$$\Delta A_p^{\text{specificity}}(-3.0) = A_p^{\text{FTNT}}(-3.0) - \frac{A_p^{\text{ADI}}(-3.0) + A_p^{\text{MU}}(-3.0)}{2} = +0.0758 - (-0.3369) = \mathbf{+0.4127}$$

**Observation**: When `(0, 104)` is suppressed on ADI and MU, their margins do **not** move toward their respective Anonymous baselines; instead, they move in the opposite direction ($A_p \approx -0.34$). The specific contrast $\Delta A_p = +0.4127$ exceeds the pre-registered interpretation threshold ($> 0.10$) by 4-fold.

---

## 3. E3-B Downstream Attenuation Results (FTNT)

Evaluated at $\beta = 0.0$ (complete attenuation of the reconstructed component) on the 5 selected E2 heads (`(31, 0)`, `(31, 1)`, `(31, 3)`, `(19, 4)`, `(27, 6)`):

| Mode | Intervened Margin ($\beta=0$) | Anonymous Progress $A_p(0)$ | Identity DLA Delta $\Delta D_{\text{id}}$ | Evidence DLA Delta $\Delta D_{\text{evid}}$ | Evidence Preservation |
|---|---|---|---|---|---|
| **`identity`** | 0.7729 | +0.0016 | **-0.0104** | **0.0000** | **100.0% preserved** |
| **`evidence`** (norm-matched) | 0.7694 | +0.0058 | 0.0000 | +0.0007 | — |
| **`random_subset`** (norm-matched) | 0.7816 | -0.0089 | 0.0000 | 0.0000 | 100.0% preserved |
| **`whole_head`** (side-effect upper bound) | 0.7656 | +0.0100 | -0.0110 | -0.0025 | Damaged (-0.0025) |

**Observation**: Attenuating the reconstructed identity-source component specifically reduced identity DLA by $-0.0104$ while leaving evidence DLA completely unchanged ($\Delta D_{\text{evid}} = 0.0000$). In contrast, whole-head attenuation caused collateral damage to the evidence DLA ($-0.0025$).

---

## 4. Scientific Verdict & Conclusions

Against the pre-registered criteria in `proposal-e3-v1.md`:

1. **Within-Ticker Cell Specificity**: **SUPPORTED**  
   Suppressing `(0, 104)` moved FTNT toward anonymity ($A_p = +0.0758$), whereas the same-layer wrong-entity cell `(0, 5101)` produced $A_p = -0.0025$ and the matched-random neuron produced $A_p = -0.0501$.
2. **Cross-Ticker Specificity vs Generic Slot Hypothesis**: **SUPPORTED (Generic Slot Disproved)**  
   The hypothesis that `(0, 104)` is a generic company entity-slot neuron acting uniformly across companies is rejected. Suppressing `(0, 104)` on same-collision peers (ADI and MU) shifted their decisions away from anonymity ($A_p \approx -0.34$), producing a net specificity delta of $+0.4127$ (threshold $> 0.10$).
3. **Downstream Surgical Routing**: **SUPPORTED**  
   Attenuating the routed identity vector in the 5 full-attention heads isolated identity DLA reduction with zero evidence leakage ($\Delta D_{\text{evidence}} = 0.0000$).

---

## 5. Next Steps

1. **Formalization for Milestone M5**:  
   This establishes the first causal evidence in Qwen3.5-4B that:
   - An early MLP neuron `(L0, N104)` exhibits genuine identity specificity for a specific corporate entity (FTNT).
   - Downstream attention attenuation can surgically decouple identity influence from evidence reasoning.
2. **Calibration Freeze**:  
   Any confirmatory calibration/test pipeline can now freeze `(0, 104)` and the 5 selected heads with explicit pre-registered gates.

---

## 6. Instrument Revision Re-verification (v2 instrument, 2026-09-03)

**Erratum context.** The v3 run above was scored with the shared core FP32 tail under its v1 definition, which re-implemented the final norm as `norm(x)·w` and was therefore wrong for Qwen3.5's `Qwen3_5RMSNorm` (`norm(x)·(1+w)`)（見 [`docs/shared-experiment-core.md`](../shared-experiment-core.md) 測量變更記錄 v2，修正於 commit `101e44e`）。本節是同一凍結 E3 V1 設計在 v2 儀器下的重驗，不改寫上方原始記錄。

**Re-verification run:** `entity-cell-e3-discovery-v4`（324 upstream rows、60 downstream rows、22 analysis groups；CPU fp32；同 v3 的 frozen 參數：同 prepared dir、同 E1 v2 trusted 集合、同 E2 v5 selected heads、peer tickers ADI/MU/FTV）。

### 6.1 決策語義的修正（本節最重要的變化）

v1 儀器下 FTNT 的「decision conflict」敘事（clean Buy、匿名 Sell，公司名把決策推過 0 點）是計分錯誤的假象。真決策下：

| Prompt | clean（真） | 匿名（真） | 真決策語義 |
|---|---|---|---|
| `record_26ba8dca` | +1.9470 | +0.6034 | 匿名基線亦為 Buy |
| `record_76eee4cd` | +1.7098 | +0.4383 | 匿名基線亦為 Buy |
| `record_8147b61f` | +1.5643 | +0.3928 | 匿名基線亦為 Buy |

公司名效應是 **Buy 偏好放大概 +1.17~+1.34 log-odds**，不是決策反轉。全部 324 條 upstream records 的 `flip` 皆為 false——任何 dose、任何對照、任何 ticker 都沒有 Buy/Sell 決策翻轉。

### 6.2 Frozen gates 在 v2 儀器下的重驗（均用 3-prompt 均值）

| Gate（`proposal-e3-v1.md` §判定） | v3（舊儀器） | v4（真儀器） | 判定 |
|---|---|---|---|
| $A_p^{FTNT}(-3) > 0$ | +0.0758 | **+0.0971** | pass |
| $A_p^{FTNT} - A_p^{wrong(0,5101)} > 0$ | +0.0783 | **+0.0925**（對照 +0.0046） | pass |
| $A_p^{FTNT} - A_p^{random(0,500)} > 0$ | +0.1259 | **+0.1053**（對照 −0.0082） | pass |
| $A_p^{FTNT} - A_p^{ADI} > 0.10$ | +0.4173 | **+0.4104**（peer −0.3133） | pass |
| $A_p^{FTNT} - A_p^{MU} > 0.10$ | +0.4079 | **+0.3932**（peer −0.2961） | pass |

**所有 frozen gates 在真決策下全部通過。**同撞車 peer 的「反向移動」訊號（ADI/MU 的 $A_p \approx -0.30$）在真儀器下不變，通用槽位假說仍被拒絕。

**逐 prompt 補充（均值背後的細節）**：target 的 $A_p(-3)$ 在三個 prompt 上分別是 −0.0373（`26ba8dca`，margin 反向上升 +1.947→+1.997）、+0.1004（`76eee4cd`）、+0.2281（`8147b61f`）。即 2/3 prompts 朝匿名方向移動、1/3 反向；frozen gate 定義於均值上，判定不受影響，但個別 prompt 的效應方向不一致，解讀時應以均值為準。

### 6.3 E3-B 下游衰減（β=0，FTNT 3-prompt 均值）

| Mode | Δ identity DLA（v3→v4） | Δ evidence DLA（v3→v4） | evidence preservation |
|---|---|---|---|
| `identity` | −0.0208 → **−0.0291** | 0.0000 → **0.0000** | 100% preserved |
| `evidence`（norm-matched） | 0.0000 → 0.0000 | +0.0015 → +0.0020 | — |
| `random_subset`（norm-matched） | 0.0000 → 0.0000 | 0.0000 → 0.0000 | 100% preserved |
| `whole_head`（side-effect bound） | −0.0227 → −0.0316 | −0.0052 → −0.0070 | damaged |

identity-source 衰減仍精確隔離 identity DLA（且幅值在真儀器下更大），evidence DLA 保持 100% 不變；whole-head 仍有證據損傷上界。downstream margin 影響在兩種儀器下皆近乎 0（$A_p \approx 0$）。

### 6.4 修正後的解讀

1. **特異性結論維持**：`(L0, N104)` 對 FTNT 的 causal 特異性（within-ticker 與 cross-ticker）在真決策下全部維持，frozen gates 全數通過。原 §4 的三項 verdict（SUPPORTED）不需要推翻。
2. **決策語義修正**：該神經元的因果效應是「把 FTNT 的 Buy 偏好向匿名基線拉回（逐 prompt 的 margin 移動：P1 +0.050 反向、P2 −0.128、P3 −0.267，均值約 0.15 log-odds，$A_p$ 均值 +0.097），並對同撞車 peer 產生方向相反的效應」，**不是決策翻轉，也不是決策的主驅動**。這與梯度決策歸因的發現一致（`(0,104)` 在 true-margin first-order 決策貢獻排名中很靠後，且決策分散在大量神經元上）。
3. **儀器註記**：v4 run 在 CPU fp32 執行（GPU 被占；fp32 數值精度高於 v3 的 GPU bf16），run manifest 與 load log 記載 device。上方 §2~§5 的所有數值為 v1 儀器下的記錄，保留原樣；後續引用 E3 V1 數值時必須標明儀器版本。
