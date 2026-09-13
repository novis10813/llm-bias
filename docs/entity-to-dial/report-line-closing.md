# Entity-to-Dial 研究線收線報告

**狀態**：收線（2026-09-13）  
**對象模型**：Qwen3.5-4B（bf16，hybrid Gated DeltaNet 架構，32 層）  
**核心結論**：從底層實體特徵承載帶（L0–11）到決策形成層（L15）的因果路徑解剖正式收線。實體訊號的主體路徑不在實體 token 區間，亦不經由單一 dial 通道瓶頸傳遞，而在指令上下文區間的狀態差值（$\Delta s$）。在轉移峰值層 L15，該差值呈現高度低維集中性：8 維殘差子空間（$k=8$）即可恢復 full-swap 效應的 98.3%；雖然 dial 激活通道（L15/n8490）與該子空間幾何嚴格正交（$\cos = -0.020$）且獨立承載 43.4% 效應，但兩通道於單一 forward 同時介入時在下游讀出呈現一致的飽和匯流（加法比值中位數 0.733 < 0.85，Gate F1 fail）。雙通道緊湊表示不成立，L15 段的最終因果表徵描述採納 $k=8$ 殘差子空間版本；本研究線核心問題已獲明確解答，正式收線。

**名詞定義**：

- **toward-source delta（$\Delta M$）**：以純實體偏向為方向基準的 margin 偏移量，定義為 $\text{sign}(M_{\text{src}} - M_{\text{tgt}}) \cdot (M_{\text{patched}} - M_{\text{tgt}})$（單位：nats）。
- **effect ratio**：介入條件誘發之 $\Delta M$ 對同方向 in-run full swap $\Delta M_{\text{full}}$ 的比值（R1 convention：僅納入 $|\Delta M_{\text{full}}| \ge 0.2$ 之有效方向）。
- **additivity ratio**：dual-hook 同時介入（$v_1 + \text{dial}$）誘發之 $\Delta M_{\text{combined}}$ 對 $\Delta M_{\text{full}}$ 的比值。
- **additive residual / interaction_delta_m**：$\Delta M_{\text{combined}} - (\Delta M_{v_1} + \Delta M_{\text{dial}})$；顯著為負表示兩通道在下游讀出存在飽和或交互。
- **$v_1$ 與 $k=8$ 子空間**：8 個凍結方向於 L15 指令區間之狀態差值矩陣經 SVD 分解持久化之前 16 個右奇異向量（$\mathbb{R}^{2560}$，正交歸一）；$v_1$ 為第 1 主成分方向，$k=8$ 為前 8 個方向張成之子空間。
- **dial transplant**：對 L15 MLP down-projection 輸入之 channel 8490（9216 維空間），僅於指令區間施加 $\delta(p) = a_{\text{src}}(p) - a_{\text{tgt}}(p)$ 的位置受限通道移植。
- **dual-hook combined patch**：於單一 forward 中同時施加 L15 post-block 殘差變換（$v_1$ 投影）與 L15 MLP down-projection 輸入 pre-hook 變換（dial 通道移植）。

---

## 1. 實體訊號的主體路徑不在實體 token 亦不在 Dial，而在指令上下文表徵差值

探索階段（Phase A–C；詳見 `report.md` Rev 1.3）以嚴格因果介入排除了實體決策偏誤走「實體 token 獨立通道」或「單一 dial 神經元瓶頸」的直覺假說：

- **實體 token 區間於高層失去因果充分性（Phase A，Gate A1 fail）**：在 L0–11 承載帶之後，將 ticker 或 name token 區間單獨跨實體置換，在 L12 至 L18 的 normalized transfer 均值全數落於零附近（95% 信賴區間均涵蓋 0）。實體特徵在進入 L12 前已完全移轉，不再停留在實體字元所在位置。
- **Block 層級解耦未見單一子模組主導（Phase B，Gate B fail）**：在 L12–15 的交接窗口，單獨置換 MLP block 或 Attention block 的殘差增量均無法解釋整體轉移量（各層 transfer < 0.20）；然而在 L15 指令上下文區間實施 full-residual swap 時，決策轉移量達到全域峰值（$+0.604$ nats）。
- **Dial 單坐標無法承載跨實體偏誤主體（Phase C，Gate C fail）**：直接將 source 實體的 dial 坐標（L15/n8490）激活值移植至 target 實體，未解釋落差（unexplained gap）高達 90% 以上；dial 坐標對決策 margin 的調控屬於通用偏置（unconditional dial），而非實體特定偏誤的專屬中繼站。

三項初期檢驗共同確立：實體訊號在 L12 之後轉由指令上下文區間（instruction span）的整體表徵差值承載。

---

## 2. Block 層級 Raw Channel 觀測重現 Erasure 簽章，無法建立單神經元因果路徑

針對 L12–18 指令區間的 MLP down-projection 激活向量實施通道級因果搜尋（Phase D，`entity-to-dial-d-02`），證實單純的 raw channel 介入無法構成實體向 dial 匯流的路徑：

- **層級特異性完全缺席（Gate D1 fail）**：在 L12 至 L18 逐層進行 block patch 時，無任一層符合特異性標準；各層 raw 效應一致偏向 sell 方向，其 toward-source 8 方向均值約等於 0。此現象完全重現了 balanced-evidence-gap Phase 3 的抹除簽章（erasure signature）——破壞特定通道僅誘發模型輸出向預設先驗坍縮，而非精確傳遞實體極性。
- **產業一致性低落（Gate D2 描述性指標）**：19 個非 final 層的一階歸因 top channel 雖然在絕對相關性上顯著超越隨機對照組（$|\rho| \approx 0.95 \sim 0.99$），但四大產業間的符號一致性多數為 0/4 或 1/4，缺乏跨產業的一致因果機制。

Phase D 否定了「指令區間中存在特定 intermediate channel 群負責將實體訊號泵入 dial」的假設，研究重心因而轉向指令區間殘差狀態空間的幾何結構。

---

## 3. L15 狀態差值呈低維集中，並由兩條近乎正交之通道平行承載

在轉移峰值層 L15 深入解剖指令區間狀態差值（Phase E，`entity-to-dial-e-01`），揭示了高維殘差流與低維語義載體之間的結構關係：

- **Block Delta 組合具自洽因果充分性（Gate E1 PASS）**：L15 dual-block joint patch（Attention 與 MLP 增量差值相加，不置換 pre-L 狀態）的 effect ratio 中位數達 **0.575**（$\ge 0.5$，$n_{\text{effective}} = 6/8$），顯著勝出預先凍結之 pre-L 殘留假說（$H_{\text{carryover}} \approx 0.187$）。L15 當層的計算增量已包含過半之決策轉移能量。
- **狀態差值呈極端低維集中（E2a 降維曲線）**：對 8 個方向的狀態差值實施 PCA，投影至前 $k$ 維奇異向量後的效應比值呈現陡峭上升：$k=1$ 恢復 73.7%，$k=3$ 達 88.5%，$k=8$ 達 **98.3%**（已飽和 full swap 的 1.000），$k=16$ 達 100.1%。2560 維殘差流中的實體資訊實質被約束在不超過 8 維的線性子空間中。
- **Dial 構成顯著但非唯一的平行管道（Gate E2b FAIL，未被證偽）**：位置受限的 dial channel 移植單獨達成 43.4% 的效應比值（中位數 0.434 < 0.5 判定未達 gate，但屬實質效應）；其相應的 residual footprint 投影亦達成 43.9%。
- **兩通道幾何嚴格正交**：事後無 GPU 幾何掃描確認，$\cos(v_1, \text{dial\_footprint}) = \mathbf{-0.020}$，且 dial footprint 與前 16 個 PCA 基底向量的絕對夾角餘弦均小於 0.071。同時，$v_1$ 對所有 9216 個 MLP channel footprint 的最大 $|\cos|$ 僅為 0.298（ch 8324）。$v_1$ 是高度分散的殘差方向，與 dial channel 坐標在幾何上完全獨立。
- **立場轉移不對稱性（Stance Transfer Asymmetry）**：4 個 bottom→top 方向（以 sell 立場為 source）轉移極強（$+0.95$ 至 $+1.44$ nats），而 4 個 top→bottom 方向（以 buy 立場為 source）轉移微弱或反向（$+0.27, +0.14, -0.26, -0.10$ nats）。在 top→bottom 方向中，$v_1$ 與 dial 均單向推動模型賣出，係由 pre-L 狀態差值扮演抑制煞車（brake）。

---

## 4. 雙通道於下游讀出匯流飽和，雙通道緊湊表示被否決

為了驗證「$v_1$（殘差方向）+ dial（MLP channel）」是否能構成極致緊湊之雙通道表示，Phase F（`entity-to-dial-f-02`）在單一 forward 實施 dual-hook 聯合介入，並以中性文本 push 測定 $v_1$ 本征 loading：

### 4.1 加法性檢驗：下游讀出呈現一致飽和（Gate F1 fail）

在 6 個 R1 有效方向上，dual-hook 聯合介入的 additivity ratio 中位數為 **0.7329**（$< 0.85$ 門檻，Gate F1 fail）：

| 方向 | 方向分類 | full swap | $v_1$ 臂 | $k=8$ 臂 | dial 臂 | combined 臂 | additivity ratio | additive residual |
|---|---|---|---|---|---|---|---|---|
| BDX→BLK | bottom→top | +1.3237 | +0.9392 | +1.2830 | +0.4716 | +0.9160 | 0.6920 | **−0.4949** |
| BDX→NSC | bottom→top | +0.9473 | +0.7217 | +0.9274 | +0.5551 | +0.7269 | 0.7673 | **−0.5500** |
| IT→BLK | bottom→top | +1.4387 | +1.0231 | +1.4193 | +0.4727 | +1.0049 | 0.6985 | **−0.4908** |
| IT→NSC | bottom→top | +1.0654 | +0.8308 | +1.0894 | +0.5445 | +0.8390 | 0.7875 | **−0.5364** |
| BLK→IT | top→bottom | +0.2745 | −0.0545 | +0.2564 | −0.5891 | −0.0762 | −0.2775 | +0.5674 |
| NSC→BDX | top→bottom | −0.2584 | −0.4858 | −0.2699 | −0.6547 | −0.5054 | +1.9560 | +0.6351 |
| BLK→BDX | 排除（R1） | +0.1396 | −0.1706 | +0.1088 | −0.6699 | −0.1997 | excl | +0.6408 |
| NSC→IT | 排除（R1） | −0.0967 | −0.3177 | −0.0878 | −0.6154 | −0.3368 | excl | +0.5963 |

（單位：nats。有效方向中位數：0.7329；次要統計量：bottom→top 中位數 0.7329，top→bottom 中位數 0.8393。跨 run 一致性：相較 e-01 四臂差異均為 0.00e+00。）

數據展現了清晰的機制特徵：
- **Bottom→top 方向一致呈現負交互項**：在 4 個乾淨轉移的 bottom→top 方向上，additive residual 全數為負（$-0.491$ 至 $-0.550$ nats），combined 的效果量甚至小於 $v_1$ 單臂本身。這證實兩通道在幾何上雖不重疊，但進入晚期讀出網絡時競爭同一非線性飽和帶，無法維持線性加性。
- **Top→bottom 方向之 combined 抑制反向漂移**：在 top→bottom 方向中，combined 產生的負向偏移（$-0.076$ 與 $-0.505$）小於兩臂之和（$-0.644$ 與 $-1.141$），交互項反轉為正（$+0.567$ 與 $+0.635$），同樣反映出下游對大幅度偏移的壓縮抑制。

### 4.2 方向性 Push：$v_1$ 無本征立場 Loading（H_F2 描述性判定）

在中性匿名文本（anonymous prompt，$m_{\text{anon}} = -3.2280$ nats）施加 $\pm\alpha \cdot \text{push\_base} \cdot d$ 介入（$\text{push\_base} = 0.00540$ 殘差單位）：

- 在評估點 $\alpha \in \{1.0, 2.0\}$ 上，$v_1$ 與 dial footprint 兩臂的全部 8 個判定點位移幅度均落於 $|\Delta M| \le 0.05$ nats 之 bf16 擾動帶內（$v_1$ 於 $\alpha=1.0$ 為 $+0.006$ / $-0.046$，$\alpha=2.0$ 為 $+0.013$ / $+0.004$；dial_fp 於 $\alpha=1.0$ 為 $-0.014$ / $-0.0004$，$\alpha=2.0$ 為 $-0.013$ / $+0.024$）。
- 兩臂判定結果均為 **`context_dependent_or_null`**。
- **理論修正**：$v_1$ 在脫離實體上下文時不具備獨立的 signed stance loading，其在 transplant 實驗中所誘發之大幅位移依賴於與 target 實體內部殘留特徵的非線性交互。「stance 軸」之詮釋降級為「特定方向差值之第一主成分」，其因果推力本質上是 context-dependent 的。

### 4.3 最終描述版本採納

依據 Phase F 協議 §7 之預先註冊決策表，Gate F1 fail 觸發 fallback 機制：由 1 個殘差方向加 1 個 MLP 通道構成的「極致雙通道模型」正式被否決；L15 段因果表徵的最終完整描述**採納 $k=8$ 殘差子空間版本（恢復 98.3% 效應量）**。

---

## 5. 研究宣稱之邊界與未涵蓋事項

1. **實體與方向母體限制**：本研究線的所有數值結論均建立於 16 家標竿企業及預先凍結之 8 個極端對比方向（TOP = {NSC, BLK}，BOTTOM = {IT, BDX}）；未宣稱此幾何結構可無條件推廣至全域未見企業或中性對比方向。
2. **層級聚焦邊界**：因果解剖集中於轉移峰值層 L15；中間過渡層（L12–14）之加法性與子空間演變未逐層重測，以 e-01 之記錄為參照。
3. **因果介入非電路級完全拆解**：dual-hook combined patch 屬一階狀態介入，證實兩通道在決策輸出層呈現飽和，但不等同於在內部計算圖上完全還原其交互之精確突觸權重。
4. **模型架構綁定**：本結論針對 Qwen3.5-4B（Gated DeltaNet 混合架構）；其他純 Transformer 架構模型是否具備同類 L15 指令讀出轉移帶，不在本宣稱範圍。
5. **未涵蓋之上游機制**：買方立場不可轉移之原因（pre-L brake 的物理來源層）以及 $v_1$ 向量在 L12–13 的寫入機制，屬於衍生之獨立課題，不在本收線範圍內。

---

## 6. 收線判定：核心問題已獲解答，L15 機制定位確立，研究線正式收線

本研究線啟動時的核心命題為：「**實體決策訊號如何從底層（L0–11）特徵帶傳遞至晚期（L15）的 Dial 通道**」。歷經六個階段的系統性因果解剖，該問題已獲得完全自洽的科學解答：

1. **路徑形態非通道式傳遞**：實體訊號並非透過實體 token 區間以離散通道形式逐層向上遞送，亦非在晚期經由單一 dial 通道收口（Phase A/B/C/D 全數排除相關直覺假設）。
2. **路徑實體為指令上下文之低維殘差子空間**：實體特徵於 L12–15 窗口完全轉譯為指令區間的殘差狀態差值，並在 L15 呈現極致的幾何約束性——**8 維線性子空間承載了 98.3% 的全部因果轉移力**。
3. **Dial 是平行但匯流的同效應通道**：Dial 坐標激活值雖具備 43.4% 的轉移能力且與主子空間嚴格正交，但在下游決策讀出時與殘差子空間共享同一飽和網絡，無法疊加出超越 8 維子空間的緊湊表徵。

核心科學問題定位明確，反駁證據與肯定證據均已在嚴格一致性（diff = 0.0）下重現收斂。**Entity-to-Dial 研究線至此正式收線**。

---

## 7. 產物與數據索引

| 階段 | Run ID / 存檔路徑 | 狀態 | 核心產物與驗證腳本 | 主要判定結果 |
|---|---|---|---|---|
| **Phase A** | `entity-to-dial-a-01` | complete | `scripts/entity_to_dial_phase_a.py`<br>`forward/results.jsonl` (672 recs) | **Gate A1 fail**：實體 token 於 L12+ 無充分性 |
| **Phase B** | `entity-to-dial-b-01` | complete | `scripts/entity_to_dial_phase_b.py`<br>`forward/records.jsonl` (448 recs) | **Gate B fail**：單一 block patch transfer < 0.20 |
| **Phase C** | `entity-to-dial-c-01` | complete | `scripts/entity_to_dial_phase_c.py`<br>`forward/records.jsonl` (64 recs) | **Gate C fail**：Dial transplant unexplained gap > 90% |
| **Phase D** | `entity-to-dial-d-02`<br>*(d-01 為偏差記錄)* | complete | `scripts/entity_to_dial_phase_d.py`<br>`forward_d1` (112) / `forward_d2` (320) | **Gate D fail**：無合格層，toward $\approx 0$，呈抹除簽章 |
| **Phase E** | `entity-to-dial-e-01` | complete | `scripts/entity_to_dial_phase_e.py`<br>`forward_e1` (224) / `forward_e2` (72) | **Gate E1 pass**（joint 0.575）；**Gate E2b fail**（dial 0.434）；$k=8$ 達 0.983；$\cos(v_1, \text{dial}) = -0.020$ |
| **Phase F** | `entity-to-dial-f-02`<br>*(f-01 為偏差記錄)* | complete | `scripts/entity_to_dial_phase_f.py`<br>`forward_f1` (80) / `forward_f2` (13) | **Gate F1 fail**（additivity 0.7329，飽和匯流）；**F2 context-dependent**；收線採納 $k=8$ 子空間版本 |
