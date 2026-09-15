# Phase 1 V2：64 筆初稿已完成，家族分割尚不可凍結

**文件製作紀錄，非實驗結果；無 run ID。**依[材料協議](proposal-phase1-v2.md)製作[待審材料](materials-phase1-v2-draft.md)。兩個候選各32筆，共64筆描述、28個暫定家族、40個預定開發比較；不含 audit，所有材料 pending。

## 審查結果

AI 起草後另由 read-only explorer 檢查計數、概念定義、對照及家族重疊。此為 AI 預審，不能替代獨立人工審查。

- 修正 S-V04 正極只表達相關性、未明示幅度的問題，補明擴張上升／收縮下降幅度。
- 保留並標記 C-V02 的抽樣描述與 C-V04 的抽象定義；不因語句生硬就換成 fitting 同義句。可讀性仍待人審。
- 不採納把 S-F04 全改成歷史敘述的建議：同一 pair 內語氣匹配即可，不要求所有家族完全同句型；假設式材料的適用性另記。
- C-L01 續約率含評價干擾，不能當作純數值對照；S-L01 辦公室面積對照的語境距離也待審。不得由文字預審推定 tokenizer／attention 異常。
- 記錄跨擬 fitting／validation 的家族合併候選。**F/V 分割未生效，不能凍結或進行驗證。**若合併後不足，降為未分割開發材料，另提改寫方案，不換構念湊數。

本次只完成約定的初稿製作，不宣稱已有6個獨立 fitting 與4個獨立 validation 家族。統計獨立性不能從 ID、語义重疊或 AI 意見直接判定。

## 文件驗證與未執行項目

以無模型 Python 檢查唯一 ID、筆數、每家族成員、正規化全文重複及比較計數；檢查 Markdown 相對文件路徑與 `git diff --check`。未改 production code、tests、舊協議的 gates、dependencies 或 artifacts；未重跑 pytest，先前缺檔 failures 不因此消失。

後續第二輪[逐家族 AI 預審](review-materials-phase1-v2.md)建議排除 C-V02/V04、修正7筆S描述，主要材料保守合併成4個工作改寫群。方案待使用者決策，原64筆未覆寫；60筆是採納後預算，尚未成為新的材料檔案。沒有將AI預審當作獨立人工審查。

後續先人工標記 retain/revise/reject 與改寫群，才制定正式 schema、hash 及批准材料。現有 reader 缺 audit 與對照契約，不能直接載入本稿；沒有建立 `approved` JSON，未執行 tokenizer/model/lens 或 GPU。
