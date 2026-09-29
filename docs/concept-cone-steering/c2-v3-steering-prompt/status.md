# c2-v3-steering-prompt：四模型 patching 已完成，生成驗證有零分母與選層不一致

**狀態：**依 [v3 事前協議](proposal.md)，四模型的 R6 `c2v3` 全層六 span、R7 `c2v3_gen` peak／peak±2 生成檢查，已在 `confirmation-v1-20260925-full-01` 完成；執行／同步紀錄見 [confirmation-v1/status.md](../confirmation-v1/status.md)。這是 steering prompt 上的 construction-only **跨公司** patching，與 [舊 16 家跨公司](../c2-phase2b-16/status.md)及 [427 家同公司條件翻轉](../c2-v2-427/status.md)不能合併為單一曲線。

`artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-20260925-full-01/{c2v3,c2v3_gen}/result.json` 四模型均標為 `complete=true`。R6 保存 entity、evidence、instruction、final、answer_prefix、steer_suffix × 全層的 teacher-forced clean-path primary 與 fixed-prefix secondary T、direction bootstrap CI；self-patch 最大 margin 差皆 0。R7 只 patch prefill，記 parsed、eligible pairs、toward-source flips 和 realized-path shift；10 家 self-patch 生成皆與 α0 相同。

| 模型 | v2-427 所選注入層 | v3 teacher-forced steer-suffix peak／70% band | R7 peak 的 steer-suffix toward-source flips／可比配對 |
|---|---:|---|---:|
| Qwen3.5-4B | L16 | L15／L14–17 | 0／**0** |
| Gemma-4-12B | L27 | L27／L26–29 | 13／40 |
| GLM-4-9B | L19 | L20／L17–21 | 0／**0** |
| GPT-OSS-20B | L14 | L8／`[1,8]` | 7／28 |

**判讀：**Qwen／GLM 的 construction pairs α0 決策都相同，R7 的 toward-source flip 無檢驗分母；`0/0` **不是 0% 效果**，雖有 margin shift 也不能當作 generated flip。Gemma、GPT-OSS 有可比配對與生成檢查，但 GPT-OSS 的 v3 band **不含**先前 v2-427 選的 L14，因此不能宣稱四模型 steering site 均被 v3 重現。v3 entity peak（Qwen L4、Gemma L12、GLM L0、GPT-OSS L3）也不支持跨模型一律「entity L0–5」。

**27B 決定：**[v2-427 狀態](../c2-v2-427/status.md)所記的 Qwen 27B 模型命名和本地完成 artifact 尚未核實；因模型過大，使用者決定**目前暫不運行**。這不是完成結果，也不列為當前待補實驗。報告按四個實際完成模型，不再寫第 5 模型「正在執行」。
