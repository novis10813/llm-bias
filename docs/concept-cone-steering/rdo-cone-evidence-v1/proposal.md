# rdo-cone-evidence-v1：梯度訓練的方向對證據有反應嗎

**狀態：**事前協議，2026-09-29，跑之前凍結。**上層：**[rdo-cone-v1](../rdo-cone-v1/status.md)（使用其已存的方向，不重新訓練）。**性質：**探索性，Qwen3.5-4B，單一訓練 seed，只有 sell→buy。

## 為什麼做

`rdo1` 是直接對「balanced prompt 上翻成 buy」訓練的，且與 DIM 幾乎正交。它可能是一個不管證據為何都把答案推向 buy 的開關。DIM 對證據有反應：[evidence-sensitivity-v1](../evidence-sensitivity-v1/status.md) 中，Qwen 在負面證據（neg）下要更高的劑量才會翻。若 `rdo1` 在負面證據下照樣用低劑量翻，就不能稱為「投資立場」的控制。

## 設定

- 方向取自 `rdo-cone-v1-20260929-full-01/directions.json`（逐一驗證 SHA-256）：`rdo1`、`rco_b4`（最弱的正交基底），另以同一程式路徑重跑 `dim` 作對照。
- prompt 條件：`neg`（兩則負面證據）、`mixed2`（一正一負）、`zero`（沒有證據）。各條件有自己的 α0（confirmation-v1 中 Qwen 的 neg 為 101 sell，mixed2 為 98 sell／3 buy，zero 為 96 sell／5 buy）。
- 劑量沿用等範數慣例：`α ∈ {1, 2, 4, 8, 16, 32}`（DIM 範數的倍數），只跑正號（sell→buy）。
- 101 家評估公司，greedy，complete-object 解析，ITT 分母為該條件下 α0 為 sell 的公司。

## 預先登記的判準（僅描述，不做顯著性檢定）

比較基準是同一方向在 balanced 的翻一半劑量（rdo-cone-v1 結果：`dim` 4、`rdo1` 1、`rco_b4` 4；confirmation-v1 中 DIM 在 neg 的翻一半劑量為 8）。

- **E1（負面證據下的反應）：**`rdo1` 在 `neg` 的翻一半劑量，除以它在 balanced 的翻一半劑量（1）。比值 ≥ 2（與 DIM 的 8/4 相當）記為「證據會延後翻轉」；比值 = 1 且 α≤2 已達 0.5 記為「負面證據下照翻」（開關式）。介於之間只描述。`rco_b4` 與 `dim` 同樣計算。
- **E2（沒有證據時的基線）：**`zero` 條件下各方向的翻一半劑量與 balanced 相比，只描述。
- **E3（格式與崩壞）：**各條件、方向的 parse 率與 parse 率 < 0.9 的最小劑量。
- **E4（理由文字）：**在 `neg` 下已翻成 buy 的列，看理由是否提到負面證據（只做描述並列出例子；不設定量判準，也不宣稱理由忠實，C11 仍未做）。

## 解讀界線

- 探索性，單一訓練 seed，只有 Qwen 與一個符號。
- 「延後翻轉」只表示需要更高劑量，不等於對證據做了推理；`reason` 不是 evidence faithfulness 的證據。
- 若 `rdo1` 對負面證據不敏感，只表示這一個訓練目標得到的方向是開關式的，不能推論梯度訓練的方向一律如此。

## 實作與執行

- 腳本：`scripts/probe_rdo_evidence.py`（重用 `scripts/probe_rdo_cone.py` 的 `Context`，不修改該檔）；測試：`tests/test_probe_rdo_evidence.py`。
- run id 使用 `rdo-cone-v1-evidence-<date>-<phase>-NN`，輸出在 `artifacts/qwen3.5-4b/concept-cone-steering/runs/<run-id>/result.json`。
- 成本：3 個方向 × 3 條件 × 6 個劑量 × 101 家，加 3 個條件的 α0，約 5,700 次生成，估計約 1.5 小時（估算，尚未實測）。
