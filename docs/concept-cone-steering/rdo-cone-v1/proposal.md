# rdo-cone-v1：梯度訓練的方向與 cone（Wollschläger et al. 的做法），只跑 Qwen3.5-4B

**狀態：**事前協議，2026-09-29，跑之前凍結。**性質：**探索性的快速版，單一模型、單一訓練 seed；結果不併入 confirmation-v1 的數字，也不取代 [operator-comparison-v2](../operator-comparison-v2/status.md)。

## 為什麼做

operator-comparison-v2 的 cone 是 PCA 建構：軸 1 是 DIM，軸 2–4 是扣掉 DIM 之後的主成分，彼此正交，但**沒有任何一條軸被要求單獨能改變 buy/sell**。[Wollschläger et al.](https://arxiv.org/abs/2502.17420)（ICML 2025）的 cone 不一樣：基底是用梯度下降訓練出來的，錐內每個方向都必須單獨有效。因此 C8（cone 較平滑）為負，只說明「PCA 建構的多維方向沒有優勢」，並沒有檢驗論文的 cone。本版用最接近論文的做法檢驗這件事。

## 設定

- 模型 Qwen3.5-4B，主要層 L16（`MODEL_REGISTRY` 的 `peak`），balanced prompt，steer suffix（K=100）prefill 注入，與 confirmation-v1 同一套 prompt／解析（complete-object）。
- 切分沿用 confirmation-v1：seed `20260923`，402 家建構、101 家評估。**訓練只用 402 家建構公司**，評估組不參與。
- 劑量沿用等範數慣例：注入 `α · ‖d[p]‖ · u`，`d[p]` 是 DIM 的逐 token 差，`u` 是被學習的單位向量（所有 token 共用，與 `random` 對照相同）。DIM 的 Top/Bottom10 取自 confirmation-v1 的 Qwen ranking（只用建構公司）。

## 訓練（與論文的對應與差異）

| 項目 | 論文 | 本版 |
|---|---|---|
| 演算法 | RDO（單向）與 RCO（cone）：梯度下降，每步後 Gram–Schmidt | 相同；cone 基底維度 n=4（與 PCA cone4 同維度） |
| cone | `{Σλᵢbᵢ, λᵢ ≥ 0}`，每步同時對「錐內抽樣的點」與「基底本身」算損失 | 相同；抽樣係數為單位球正象限上的均勻樣本 |
| addition 損失 | 加到無害指令上，引發拒絕 | 加到 balanced prompt 上，讓固定答案 margin 往 buy 走：`softplus(-M)`，`M = log p(buy) − log p(sell)` |
| ablation 損失 | 有 | **省略**（沒有對應的第二種行為） |
| retain 損失 | 拿掉方向後，輸出分布不變（KL） | 注入後，答案前綴位置上「buy／sell 以外」的 token 分布相對未注入的 KL，用來限制格式被破壞；權重 `λ_ret` |
| 訓練劑量 | α 取 DIM 的範數 | `α_train = 1`（單位為 DIM 範數，與論文相同） |
| 初始化 | 隨機 | 隨機（不從 `d̂` 開始，避免偏向 DIM） |
| 方向 | 單一向量 | 單一向量，所有 suffix token 共用（與 `random` 對照相同）；DIM 仍為逐 token |

只訓練 sell→buy：Qwen 的 balanced prompt 在全部 503 家公司上都是 sell（ranking 的 margin 全為負），沒有 buy→sell 的分母，所以這一版不談雙向。

## 評估（101 家評估公司，balanced，greedy，complete-object 解析）

方向與劑量（單位為 DIM 範數的倍數）：

| 名稱 | 說明 | 劑量 |
|---|---|---|
| `dim` | 同一程式路徑重跑的 DIM（對照，可與 confirmation-v1 的 `dim` arm 比對） | 1, 2, 4, 8 |
| `rand1` | 隨機單位向量（seed 0），等範數對照 | 1, 2, 4, 8 |
| `rdo1` | 單向 RDO | 1, 2, 4, 8 |
| `rco_b1`…`rco_b4` | RCO 的四條基底，各自單獨注入 | 1, 2, 4 |
| `rco_centroid` | 四條基底的歸一化和 | 1, 2, 4 |
| `rco_s1`…`rco_s8` | 從錐裡抽的 8 個方向（固定 seed） | 1, 2, 4 |

## 預先登記的判準（探索性，僅描述，不做顯著性檢定）

主要指標是 sell→buy 的 ITT flip 率（分母為 α0 為 sell 的公司，注入後解析不出算未翻）與 parse 率，都以 101 家為單位。

- **Q1（梯度方向比 DIM 有效率嗎）：**`rdo1` 第一次 flip 率 ≥ 0.5 的劑量，是否不高於 DIM 的一半（DIM 在 confirmation-v1 為 4，故門檻為 2）。
- **Q2（cone 基底是否各自有效）：**`rco_b1`…`rco_b4` 在 α ≤ 4 是否都達到 flip 率 ≥ 0.5。任一條沒有達到，就記為「基底不全有效」。這是論文對 cone 的定義性質，PCA cone 沒有檢查過。
- **Q3（cone 有沒有用）：**（a）8 個抽樣方向的 flip 率分布（各劑量的最小／中位數／最大）與 `rdo1` 比較；（b）best-of-8：每家公司只要 8 個方向中任一個能翻就算翻，與抽樣方向的平均 flip 率比較（對應論文的 best-of-N）；（c）各方向的崩壞劑量（parse 率 < 90% 的最小劑量）與 `rdo1`、`dim` 比較。
- **Q4（reasoning 是否不同）：**每個方向保存完整 `generated_text`。本版只做描述：對同一批公司、同樣翻成 buy 的列，並排看 `dim`、`rdo1` 與 cone 樣本的 `reason` 是否在內容上不同，不設定量判準，也不據此宣稱概念對應（C9／C11 仍未做）。

## 解讀界線

- 單一模型、單一訓練 seed、單一符號、單一次隨機抽樣；沒有訓練 seed 的變異估計。任何「cone 優於／不優於」都只算探索性線索。
- 若 Q2 為否（基底不全有效），優先檢查訓練是否收斂（記錄的損失曲線），再解讀 cone 的結果。
- 結果不改寫 operator-comparison-v2 與 claim ledger 的 C8。若要升級為論文主張，須另開協議：多個訓練 seed、多個模型，以及雙向的 α0 分母。

## 實作與執行

- 腳本：`scripts/probe_rdo_cone.py`；測試：`tests/test_probe_rdo_cone.py`。
- 輸出：`artifacts/qwen3.5-4b/concept-cone-steering/runs/<run-id>/`（`training.json`、`directions.json`、`result.json`），只存緊湊統計、學到的方向與 SHA，不存 hidden states。
- 超參數（訓練步數、batch、學習率、`λ_ret`）在 smoke 中先確認損失下降，之後在 full run 的 metadata 中固定並記錄。
