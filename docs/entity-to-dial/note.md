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
