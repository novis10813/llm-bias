# 論文主實驗：由 C2 的 427 家定位結果選擇 Concept Cone 注入層

**設計狀態：** 這是以 [C2 427 家同公司條件翻轉結果](../c2-v2-427/status.md)為選層證據的新主實驗，不是把[舊 16 家 C2 選層診斷](../crossmodel-cone-pilot-16/status.md)改名。源實驗的 direction／prompt 與本實驗不同：C2 只提供 model-specific 候選層，不提供這次 cone 的方向或 decision-flip 證據。本次 paper-intended 執行完成前不得寫作跨模型確認；結果需原樣保留失敗與缺項。

## 固定方法

1. **選層來源：** Qwen3.5-4B L16、Gemma-4-12B L27、GLM-4-9B L19、GPT-OSS-20B L14；各對應 `artifacts/<model>/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/analyze/summary.json` 的 `instruction` span 最大 mean normalized transfer。每次執行保存來源 summary 的 SHA-256 並核對峰值；不依 cone 的 flip 結果另選層。
2. **建構／受測與模型：** 沿用 2024 年 `data/sp500_constituents_2020_2025.csv` 的 503 家母體、seed 20260923 的固定 402 家建構／101 家受測切分；各模型在自己的 402 家 balanced prompt 上計算 clean fixed-prefix buy−sell margin，取 Top／Bottom 各 20 家，於其選定層建 sector-demeaned token-wise 4D SVD cone，中心射線每 token 正規化。101 家不參與建構。GPT-OSS 模型權重保持 MXFP4 native，不強轉 BF16。
3. **受測：** 在各模型自己的層掃 alpha 0／2／3／4／5／6，只有 frozen balanced prompt；固定答案 margin 使用真正追加 `{"decision": "` 前綴的 forward，貪婪生成自未附前綴的 prompt 開始，最多 192 新 tokens。只把完整有效 JSON 內 buy/sell 視為可解析 decision；code fence、thought prefix 與截斷均列 parse failure，不可偷偷從未解析文字推斷 flip。逐公司逐 alpha 留 compact margin、decision、parse status、有限長度生成文字與來源；計算有效配對的 flip rate 及完整 101 家的 parse rate。另用**同一受測組的全部 101 家**計算每個 alpha 的平均固定答案 margin 位移 `mean(ΔM_alpha) = (1/101)Σ_i[M_i(alpha)−M_i(0)]`（nats），與生成決策的翻轉率並列但不可互相替代；它不是 C2 的 normalized transfer `T`，也不用於生成買賣分組。
4. **輸出與失敗界線：** 本實驗使用 `scripts/probe_concept_cone.py --cohort-mode sp500_paper`；產物在 `artifacts/<model>/concept-cone-steering/runs/c2-guided-paper-*/result.json` 與同目錄 `result.md`，後者以 alpha 為欄。metadata 綁定模型、tokenizer、母體／切分 SHA、C2 427 summary SHA、選層、prompt family、計分方式與 alpha grid；逐公司中斷後可按同 metadata 續跑。不得把舊診斷的 L15/L25/L12 結果改標成論文主實驗；GLM 的 L19 同層舊 run 可以作核對參考，是否列正式同一批結果必須附其實際 provenance。

**逐 alpha 決策轉移的統計定義：** 以每家公司 alpha 0 的**自由生成** `decision` 為基準；只在基準與受測 alpha 都能依同一指定解析規則讀出 `buy` 或 `sell` 時納入有效配對。各 alpha 分開列出有效配對數、`buy→buy`、`buy→sell`、`sell→buy`、`sell→sell`；總翻轉率為 `(buy→sell + sell→buy) / 有效配對數`，方向別翻轉率分別為 `buy→sell / 有效配對中基準 buy 家數` 與 `sell→buy / 有效配對中基準 sell 家數`。基準類別為零時該方向比率**未定義**，不報成 0%。另外列出每個 alpha 可解析的家數與所有基準 buy／sell 家數，避免解析失敗造成的分母誤導。固定答案 buy−sell margin 及另一實驗 C2 的 normalized transfer `T` 都**不能決定這裡的生成 buy/sell 分組或翻轉**。原始嚴格解析與事後補充解析要各自列一張表，不能混合其分母；具來源 SHA 的衍生統計由 `scripts/summarize_concept_cone_decisions.py` 產生，不改寫原始 run。

## 原始結果之後的補充解析（不是原協議主要指標）

Gemma 已完成的生成包含完整 JSON，但外面常帶 ` ```json\n...\n``` ` 或精確的 `thought\n` 開頭；原協議依 `json.loads(整段輸出)` 把這些記為不合格。為區分格式遵循與是否能從**完整物件**讀到決策，另以 `scripts/reparse_concept_cone_decisions.py` 對原始結果做**事後補充分析**，不改寫上述原始嚴格解析規則與產物。補充解析只接受四種完整形式：裸 JSON 物件、完整 ` ```json\n...\n``` `、精確 `thought\n` + 裸物件、精確 `thought\n` + 完整 fenced 物件；外層不得有其他字元。物件必須僅有 `decision`（精確 `buy`／`sell`）與非空字串 `reason` 兩欄，不能從未解析的文字以 regex 擷取片段或猜答案。每個 alpha 與同公司 alpha 0 均解析成功才列入翻轉分母；其他失敗原樣報數。

腳本以唯讀 `result.json` 為輸入，在同一 run 旁新增 `decision_reparse.json`（來源 SHA-256、格式類別、逐公司決策及分母／翻轉數）與 `decision_reparse.md`（alpha 為欄的逐公司表）。此為看過 Gemma 格式後的補充結果，不能事後替代原設計的嚴格 JSON parse rate；若用於論文主要行為宣稱，須預先固定解析規則並對所有模型統一驗證。

## 仍須在論文宣稱前補足

這一批只有單一固定切分、無多種子 matched-norm random control、沒有 sector-disjoint evaluation，且沒有預先定義跨模型等效 dose 或成功 gate。四模型全部跑完只能報描述性 per-model margin／parse／flip，不能由 margin gate 直接宣稱行為改變，也不能將任意 α 作模型間等效強度比較。若要確認 C3／C8／C10 的比較或泛化主張，須另有控制組和獨立驗證；不可事後修改本次原始結果。
