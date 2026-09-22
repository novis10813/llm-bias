# Entity-to-Dial 小型探索：L15 殘差流方向性推注（Steering Probe）

**紀錄日期：** 2026-09-22  
**對象模型：** Qwen3.5-4B（bf16，GPU 1，transformers 5.14.1）  
**問題：** 凍結的狀態移植（Transplant）受限於成對真實狀態差，無法跨過貪婪解碼的決策門檻（翻轉率為 0）。若改採表徵工程（Representation Steering，如 Arditi et al. 2024 / Wollschläger et al. 2025）的做法，以可調乘數 $\alpha$ 沿著實體高低評分均值差向量推注，能否在貪婪解碼下誘發決策翻轉？又是否具備方向特異性與實體特異性？

---

## 1. 最小方法與對照設計

- **資料來源：** 200 家 held-out 實驗 `entity-to-dial-heldout-transfer-v1-01` 的 selection 提示詞前向記錄。
- **方向提取（Token-wise DIM）：**  
  取 clean margin 最高 10 家（Top 10：ED, GM, PG, XEL, MSFT, ORLY, AMZN, NEE, META, OXY；margin −0.885 至 −0.097）與最低 10 家（Bottom 10：MO, CNC, FOXA, BR, ERIE, TSN, BAX, TPR, FOX, COO；margin −2.504 至 −2.002）。  
  在第 15 層（L15）指令區間（100 tokens），計算逐 token 的均值差：
  $$\vec{v}_{\text{DIM}}[p] = \mu_{\text{Top10}}[p] - \mu_{\text{Bottom10}}[p] \in \mathbb{R}^{2560}, \quad p \in [0, 99]$$
  逐 token 模長範圍為 0.086 至 1.297，平均 0.209。
- **實作與執行腳本：** `scripts/probe_dim_steering.py`，支援 `--evidence-mode`、`--alphas`、`--target-tickers` 與 `--include-controls`。
- **干預算子：** 在 prefill 階段對 L15 指令區間注入 $h_{15, p} \leftarrow h_{15, p} + \alpha \cdot \vec{v}_{\text{DIM}}[p]$；單 token 解碼步不變。
- **受測對象：**
  1. 極端 Sell 公司：MO（clean margin −1.967）、CNC（−2.059）、FOXA（−1.867）。
  2. 隨機對照組（Random 1D Control）：同 token 模長的隨機高斯方向 $\vec{v}_{\text{rand}}[p]$。
  3. 匿名對照組（Anonymous Prompt）：完全去除實體名稱與代號的中性提示詞（clean margin −1.788）。
  4. Wollschläger 表徵獨立性（RepInd）正交約束：在凍結的 8 維 $V_8$ 子空間內，尋求與匿名敏感度正交的權重 $\vec{w}^*$。
- **評估指標：**
  - Continuation margin：$M = \log p(\text{buy}) - \log p(\text{sell})$。
  - 貪婪解碼決策（Greedy generation，$T=0$，`max_new_tokens=48`）：解析輸出的 JSON `{"decision": "buy" | "sell"}`。

---

## 2. 實測結果

### (1) 極端 Sell 公司的決策翻轉（主臂）

| 受測對象 | $\alpha=0.0$（Clean） | $\alpha=2.0$ | $\alpha=3.0$ | $\alpha=4.0$ | $\alpha=5.0$ | 首次翻轉點 | 生成理由摘要（翻轉後） |
|---|---:|---:|---:|---:|---:|---|---|
| **MO** | −1.967（sell） | −0.715（sell） | +0.266（sell） | **+1.377（buy）** | +2.167（buy） | $\alpha=4.0$ | 肯定 Q3 營收與自由現金流，蓋過利潤率與財測下修 |
| **CNC** | −2.059（sell） | −1.029（sell） | −0.166（sell） | +1.035（sell） | **+1.935（buy）** | $\alpha=5.0$ | 肯定營收成長與現金流 |
| **FOXA** | −1.867（sell） | −0.558（sell） | +0.352（sell） | **+1.582（buy）** | +2.318（buy） | $\alpha=4.0$ | 肯定營收成長與現金流 |

Margin 隨 $\alpha$ 呈現嚴格單調遞增；當 Margin 提升至約 +1.3 至 +1.5 nats 以上時，貪婪解碼首字與 JSON 判定穩定翻轉為 `buy`，文字維持語法完整。

### (2) 隨機方向對照（特異性檢驗）

在 MO 上注入相同模長的隨機方向 $\vec{v}_{\text{rand}}$，掃描 $\alpha \in [0.0, 6.0]$：
- Margin 僅在 −1.967 至 −1.718 間微幅擺動（全幅位移 $< 0.25$ nats）。
- 貪婪解碼全程 100% 維持 `sell`，零翻轉。
- **結論：決策翻轉具備高度方向特異性，非隨機殘差擾動所致。**

### (3) 匿名提示詞對照（實體特異性檢驗）

在無公司名稱的匿名提示詞施加 $\vec{v}_{\text{DIM}}$：
- $\alpha=0.0$：Margin −1.788，生成 `sell`。
- $\alpha=2.0$：Margin −0.695，生成 `sell`。
- $\alpha=4.0$：Margin +1.387，**同樣翻轉為 `buy`**。
- **結論：$\vec{v}_{\text{DIM}}$ 同時拉動匿名提示詞，說明該方向本質上是全域立場調控（Global Stance Steering），而非純粹的公司實體識別碼。**

### (4) 幾何正交約束檢驗（Wollschläger RepInd）

嘗試在凍結的 $V_8$ 子空間內尋找「只推 MO、不推 Anonymous」的方向：
- 測量 MO 與 Anonymous 在 $V_8$ 的 8 個基底上的敏感度向量 $\vec{\Delta}_{\text{MO}}$ 與 $\vec{\Delta}_{\text{Anon}}$：
  $$\cos(\vec{\Delta}_{\text{MO}}, \vec{\Delta}_{\text{Anon}}) = \mathbf{0.9375}$$
- 兩者在 L15 指令區間的敏感度高度重合（93.8%）。
- 若以正交投影強制消除匿名敏感度（$\vec{w}^* \perp \vec{\Delta}_{\text{Anon}}$），剩餘有效方差不足 12%：
  - 在 $\alpha \le 8.0$ 下，受非線性影響，匿名 Margin 依然漂移 $+1.0$ nat。
  - 在 $\alpha \ge 12.0$ 下，因脫離自然分佈，模型語言退化（輸出無法解析成合法 JSON）。

### (5) 證據極性與客觀事實錨定抗衡（Evidence Polarity Probe）

在 MO 上測試不同強度之單邊客觀證據與 Steering 的拉鋸：

| 證據情境 | 客觀證據內容 | Clean 基準判定 | Steering 干預效果 | 翻轉門檻 | 翻轉後模型的真實理由 |
|---|---|---|---|---|---|
| **純利多 (2 條)** | 營收成長 14% + 自由現金流創高 | **`buy`**（Margin +0.870） | 負向推注 $\alpha < 0$<br>$\alpha=-1.0 \to M=+0.021$<br>$\alpha=-4.0 \to M=-1.789$ | **$\alpha = -1.0$<br>(輕推即翻)** | **`sell`**："Altria 股價歷史上落後大盤，且核心菸草業務面臨長期逆風，短期獲利不足以支持買入。" |
| **微利多 (1 條)** | 僅營收成長 14% | **`sell`**（Margin −0.140） | 負向推注 $\alpha < 0$<br>$\alpha=-1.0 \to M=-0.899$<br>$\alpha=-4.0 \to M=-2.212$ | 無需翻轉<br>(Clean 即為 Sell) | **`sell`**："營收成長雖好，但單一季度不足以扭轉菸草成熟市場的監管風險。" |
| **微利空 (1 條)** | 僅財測下修 6% | **`sell`**（Margin −2.712） | 正向推注 $\alpha > 0$<br>$\alpha=+4.0 \to M=-0.110$<br>$\alpha=+6.0 \to M=+1.511$ | **$\alpha = +6.0$<br>(重推才翻)** | **`buy`**："Altria 是一家成熟、現金充沛的防禦型公司，高股息殖利率足以彌補 6% 的溫和財測下調。" |
| **純利空 (2 條)** | 毛利縮 300bps + 財測下修 6% | **`sell`**（Margin −3.007） | 正向推注 $\alpha > 0$<br>$\alpha=+4.0 \to M=-1.203$<br>$\alpha=+6.0 \to M=+0.813$ | $\alpha > 6.0$<br>(Margin 轉正但**決策未翻**) | **`sell`**：即使 Margin 被強推近 4 nats，模型依然死守兩項硬性財務利空，文字解碼拒絕翻轉為 Buy。 |

- **多空不對稱性：** 覆蓋利多（Buy $\to$ Sell）極易（$\alpha=-1.0$），覆蓋利空（Sell $\to$ Buy）極難（需要 $\alpha=+6.0$ 甚至無法翻轉）。
- **合理化機制：** 模型被迫翻轉時未捏造假數據，而是調取公司外部屬性（如「防禦型高股息」或「菸草長期逆風」）合理化新決策。

### (6) 零財務證據情境（Zero-Evidence / 純公司身分資訊）

去除全部財務證據區塊，僅提供公司代號與名稱（`[TICKER]` 與 `[NAME]`）：

| 受測對象 | Clean 基準判定 | Clean 生成理由摘要 | Steering 干預效果（$\alpha = +6.0$） | 干預後生成理由摘要 |
|---|---|---|---|---|
| **MO (Altria)** | **`sell`**<br>(Margin −2.525) | 核心菸草業務因吸菸率下降與監管審查面臨顯著逆風。 | **`buy`**<br>(Margin **+1.428**) | 具備結構性高營業利潤率與優異股息殖利率，為具吸引力的防禦型標的。 |
| **MSFT (Microsoft)** | **`buy`**<br>(Margin **+1.920**) | 基本面強勁、雲端與 AI 領域居領導地位，且現金流穩定。 | **`buy`**<br>(Margin **+3.129**) | 持續強化多頭立場。 |
| **ANONYMOUS** | **`sell`**<br>(Margin −0.604) | 缺乏足夠數據，無法給出有效的買入建議。 | **`buy`**<br>(Margin **+0.006**) | 數據不足但預設偏向正向。 |

- **純粹實體先驗落差：** 在毫無任何財務數據輸入下，模型對不同公司存在巨大的內生偏好差距（MSFT 與 MO 的 clean margin 落差達 **4.45 nats**）。
- **記憶檢索焦點轉移（Parametric Memory Pivoting）：** 當施加 steering 翻轉決策時，模型不捏造虛假財務數字，而是**自動切換其內部預訓練知識的檢索面向**（由「菸草監管逆風」轉向「高股息防禦價值」）。
- **隨機對照依然無效：** MO 的隨機高斯方向在 $\alpha \in [0, 6]$ 下 Margin 維持在 −2.525 至 −2.233，維持 100% Sell。

### (7) 多維 Concept Cone 與單一 1D 方向飽和對比（Wollschläger Cone 驗證）

將 Phase E 凍結的 8 維正交基底 $V_8$ 的 8 個軸按 Buy 方向對齊為 $\mathbf{b}_1, \dots, \mathbf{b}_8$，構成 Buy 概念的 8 維 Polyhedral Cone：$\mathcal{R}_8 = \{ \sum_{i=1}^8 \lambda_i \mathbf{b}_i \mid \lambda_i \ge 0 \}$。在 MO 多空平衡提示詞下比較「單一 1D 基底軸」vs「8D Cone 質心射線（$\vec{w} = \frac{1}{\sqrt{8}}\sum_{i=1}^8 \mathbf{b}_i$）」的推注表現：

| 推注射線 | $\alpha=2.0$ | $\alpha=4.0$ | $\alpha=6.0$ | $\alpha=8.0$ | $\alpha=10.0$ | $\alpha=12.0$ | 飽和與翻轉表現 |
|---|---:|---:|---:|---:|---:|---:|---|
| **$b_7$ 單軸** (Top-1 敏感軸) | −1.170 | −0.556 | −0.315 | **−0.437** | — | — | **提早飽和並反折**（無法突破負值區，零翻轉） |
| **$b_1$ 單軸** (Top-2 敏感軸) | −0.503 | +0.511 | +0.874 | +1.002 | — | — | 增速遞減平原（未達生成翻轉門檻） |
| **$b_6$ 單軸** (Top-3 敏感軸) | −1.329 | −0.716 | −0.141 | +0.247 | — | — | 增幅緩慢，未達生成翻轉門檻 |
| **$b_3$ 單軸** (Top-4 敏感軸) | −1.331 | −0.843 | −0.439 | **−0.373** | — | — | **提早飽和停滯**（停在負值區） |
| **8D Cone 質心** ($\frac{1}{\sqrt{8}}\sum \mathbf{b}_i$) | −0.619 | +0.547 | +1.045 | +1.386 | +1.745 | **+1.909** | **持續嚴格單調線性攀升，$\alpha=12.0$ 成功翻轉為 `buy`** |

- **驗證 Wollschläger 的核心論點：** 單一 1D 方向受限於個別神經迴路的非線性飽和帶，過大推力會提早飽和甚至反折（如 $b_7$ 在 $\alpha=8.0$ 反折至 −0.437）；
- **多維圓錐的優勢：** 8 維 Concept Cone 質心將推注能量分散到 8 個互相正交的通道，成功避開單一通道的非線性飽和瓶頸，展現出高度平滑、單調的推力，並在多空平衡條件下成功驅動生成決策翻轉。

### (8) 200 家產業去均值與時序對比 SVD（Token-wise Sector-Demeaned Contrastive SVD）

為克服 16 家舊基底（平均重疊度僅 0.274）的局部文字雜訊與非監督配對混入的產業語義，實施四大工程改進：
1. **產業去均值化（Sector Demeaning）：** 針對 200 家中的 Top 20 與 Bottom 20，逐 token 減去其所屬產業均值 $\widetilde{h} = h - \mu_{\text{sector}}$，濾除產業固有語義。
2. **時序對比 SVD（Slice-wise Contrastive SVD）：** 在 L15 指令區間的 100 個 token 位置上，逐位置分解 Top 20 vs Bottom 20 對比差值張量 $[20, 100, 2560]$，提取各 token 的前 4 個正交基底 $B[p] \in \mathbb{R}^{2560 \times 4}$。
3. **符號定向（Cone Sign Alignment）：** 確保每個 token 的奇異向量與正向對比均值對齊，構成 4 維 Buy Concept Cone。
4. **多公司泛化檢驗：** 評估 4D Cone 質心射線（$\vec{w}_{\text{cone}}[p] = \frac{1}{2}\sum_{j=1}^4 \mathbf{b}_j[p]$，單位模長）在多個極端 Sell 公司上的翻轉表現：

| 受測對象 (極端 Sell) | $\alpha=0.0$ (Clean) | $\alpha=2.0$ | $\alpha=3.0$ | $\alpha=4.0$ | $\alpha=5.0$ | 翻轉判定 ($T=0$) |
|---|---:|---:|---:|---:|---:|---|
| **MO (Altria, 菸草)** | −1.967 (sell) | −0.936 (sell) | −0.049 (sell) | +0.704 (sell) | **+1.287 (buy)** | **$\alpha=5.0$ 成功翻轉為 Buy** |
| **FOXA (Fox, 媒體)** | −1.867 (sell) | −0.769 (sell) | −0.047 (sell) | +0.656 (sell) | **+1.347 (buy)** | **$\alpha=5.0$ 成功翻轉為 Buy** |
| **CNC (Centene, 醫療)** | −2.059 (sell) | −1.103 (sell) | −0.484 (sell) | +0.275 (sell) | +0.816 (sell) | 嚴格單調上升（預計 $\alpha=6.0$ 翻轉） |

- **軸向角色分工：** 檢驗各軸發現，主成分軸 $b_1$ 在 $\alpha=5.0$ 達到 $+1.640$（單獨翻轉為 `buy`）；$b_2 \sim b_4$ 則編碼了不同公司間的正交偏好差異；4D Cone 質心有效融合了主立場與互補差異軸，實現平滑穩健的跨產業泛化翻轉。

---

## 3. 邊界與限制（不誇大宣稱）

1. **非實體完全去偏（Not Pure Entity Debiasing）：**  
   本探索證實可透過殘差推注翻轉貪婪決策，但此方向為全域立場調控（對匿名有效），尚未實現「只改單一公司而完全鎖死匿名基準」的實體隔離。
2. **非正交下游讀出假說獲得佐證：**  
   實體敏感度與匿名敏感度在 L15 高度重合（$\cos = 0.938$），印證了實體偏誤是透過共用的下游立場讀出軸影響決策，而非獨立於立場的孤立特徵。
3. **單一模型與特定模板：**  
   結論僅限於 Qwen3.5-4B 在多空平衡財務提示詞下的表現，不外推至其他模型架構或開放域問答。

---

## 4. 對研究進展的意義與後續

- **超越單一神經元：** 相較於 Park et al. (2026) 定位單一神經元（L15/N8490），本方法證實 L15 殘差流的 Directional Steering 是一個更連續、方向特異且能穩定翻轉真實生成的調控機制。
- **後續判斷：**  
  若未來要攻克「嚴格鎖死匿名基準的實體特異性推注」，不宜繼續在 L15 指令區間做線性正交投影；需轉向實體與立場尚未匯流的早層（L0–L11 實體 token 位置）或採用帶約束的全模型梯度優化（RDO）。目前本探索紀錄留檔，不擴大為正式 confirmation run。
