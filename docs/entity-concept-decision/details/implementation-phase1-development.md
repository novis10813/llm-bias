# Phase 1 V2 開發流程實作規格

**使用者已授權自行實作、迭代與小規模真實執行。**不再要求逐句批准；材料標記 AI 審查，非獨立人工審查。此規格限定 development／smoke，不執行正式 audit、Phase 2/3 或語義 success gate。原始協議與待審材料保留，工程完成不等於 Phase 1 科學驗證完成。

## 1. 本輪要取得的證據

將[原64筆](materials-phase1-v2-draft.md)依[AI修訂表](review-materials-phase1-v2.md)撤出4筆、替換7筆，得到60筆開發描述。保守C1/C2/S1/S2分組。使用相同指令後綴測 L15 instruction mean，檢查平均方向、排除整個改寫群後的辨識、評價與詞面對照；4個既有公司只作自然狀態描述。每個概念僅兩個主要群，不能計算或宣稱正式audit pass。

## 2. Slice 1：材料與 token preparation

檔案邊界：新增 `llm_bias/entity_concept_decision/development_materials.py`、`tests/test_entity_concept_development_materials.py`。不得更改原 `materials.py`/`concepts.py`、core 或舊實驗。

REQ-1 `build_development_materials(source_path, review_path) -> dict`：從原稿Markdown表抽取64唯一ID、review表7 replacement，明確剔除C-V02-P/N、C-V04-P/N。嚴格完整64原ID矩陣、7 replacement精確ID集、replacement非空且不同；拒絕重複ID、空字串、normalize重複全文、含投資建議buy/sell（word boundary）。不解析任意新材料，這是該版明確有界的來源轉換。輸出schema_version=1、protocol='phase1-v2-development'、review_status='ai_reviewed_development'、兩份path/SHA256、60 rows。每row含id,concept_id(C/S),family_id原ID前綴,group_id(C1/C2/S1/S2或對照原family),role(primary/evaluation/lexical/competitor),pole(P/N/PH/PL/NH/NL/A/B),text。group對應review表：C1 F01/F04/F05/V01/V03；C2 F02/F03/F06；S1 F01/F02/F03/F05/F06/V02/V04；S2 F04/V01/V03。E01/E02各保留family，另source_group C2/S1。L/X不偽標P/N。返回排序rows與38 comparisons（20原primary減2=18 +16 evaluation +4 other），每comparison id,concept_id,family_id,kind(primary/concept_at_positive_evaluation/concept_at_negative_evaluation/evaluation_at_positive_concept/evaluation_at_negative_concept/lexical/competitor),positive_id,negative_id。無fit/validation/audit split、無approved標記。

REQ-2 `prepare_development_prompt(tokenizer, text, *, instruction_suffix, answer_prefix, use_chat_template=True) -> dict`：前文固定 `Refer to the company description below to make a final investment decision.\n\nCompany description:\n{text}\n\n—\n\n` 再接suffix。core format_prompt + input_ids(add_special_tokens=False)，append answer_prefix至chat formatted。不得自行新增tokenization／forward engine。instruction suffix必須恰好出現一次，排除chat包裝與answer prefix；使用core token_span解析，再以offset檢查邊界token無跨instruction邊界，token IDs與offset encoding相符；跨邊界拒絕。出raw prompt/formatted/input_ids/instruction_span/instruction_ids/final_position與全部序列之partition：before_instruction/instruction/after_instruction半開互斥覆蓋。這是development描述布局，不假裝擁有company entity span。tokenizer無offset/空span/非法輸入 fail-closed。

驗收：`uv run pytest -q tests/test_entity_concept_development_materials.py tests/test_entity_concept_materials.py tests/test_entity_concept_directions.py tests/test_prompt_input.py`。真實來源60/38 counts，temp malformed檔，字元fake tokenizer、跨界token、重複suffix、exact partition測試；無模型載入。

## 3. Slice 2：caller-owned模型量測與development分析

檔案邊界：新增 `llm_bias/entity_concept_decision/development.py`、`tests/test_entity_concept_development.py`；可更新package與tests AGENTS的入口。不得改core、dependencies、既有helpers或其他實驗。

REQ-3 `run_development(*, model, tokenizer, device, basis, materials, instruction_suffix, answer_prefix, company_prompts, run_id, model_name, artifact_root, provenance, layer=15, min_norm=1e-6, random_seed=1729, random_count=16) -> Path`：caller擁有已載入model/tokenizer、CPU orthonormal Q[d,k]及來源驗證。支持tests d小值，real wrapper驗2560×8、32blocks。材料來源為Slice1 builder，runner仍驗rows/comparisons的ID完整性、role/pole、兩concept、預期群、所有pair coverage，防止變更後誤執行。只容development；provenance需mode(fake_smoke/model_smoke/development)、upstream/lens references與review_status，無任意formal通道。公司列表可空供單元測試，real wrapper必含BDX/IT/BLK/NSC canonical各一條formatted/ids/instruction_span來源。公司prepared record不得改寫。

REQ-4 prepare全部材料並驗每row instruction_ids一致；company prompts也驗同一instruction_ids、ids為正確tokenize(formatted)、partition合法（公司的既有span沿用，其餘位置不介入）。preparation/輸入檢查在建run前。使用core ArtifactRun與write_json/jsonl以及register_artifact；三stage嚴格prepare→forward→analyze，finalize為postcheck。拒覆寫，任何stage error留下failed manifest。provenance、sources hash、seed、dtype、layer、shape、min_norm寫metadata。已有upstream/lens實體path可register role input/lens；metadata不能偽造未驗證digest。

REQ-5 使用core inference.forward.record_residuals(model, torch.tensor([ids], dtype=torch.long, device=device), layers=[layer,last])取得postblock（不得呼叫會另做forward的record_block_states當context）。batch=1,use_cache遵循adapter，立即把instruction mean以FP32降為CPU[d]，final residual取captured[last][:,-1,:]維持[1,d]，透過core continuation_scoring.fp32_next_token_log_probs取得buy−sell margin（候選由core prompt_input.encoding.continuation_token_ids返回list，驗len==1且兩者不同）。只transient保留mean矩陣供計算，不輸出高維states；沿用核心hook lifecycle。每樣本一次forward。首個材料再重複一次，記mean最大absolute差與margin差，不校準正式τ、不據此放寬門檻。返回維度/finite/missing capture fail-closed。若FP32 head顯存不足，報失敗，不能改用bf16 scorer偷偷過關。

REQ-6 分析只用primary rows。全development概念方向與每次排除一整群之方向重用既有fit_concept_direction；可在RAM建立MaterialBundle/ConceptPair作數學adapter（split='fit'明列只是選中的計算集合，review_status='ai_reviewed_development'，無audit/approved，不呼叫或放寬原strict loader）。等群權重（family_id以group_id）；pair以原family識別。同concept LO-group-out：fit另一群，score被排除群，報pair差與群均值，標descriptive/non-independent-claim，不假裝validation gate。全development方向上報全部38比較與4公司scalar scores，primary全資料比較明標in_sample。degenerate reason傳播且分數null有reason；不拋棄失敗候選、不norm放大。

REQ-7 以seed在Q內產16個random unit directions，對相同primary及control比較報每random比較delta或其compact分位數（清楚aggregation）；不按margin挑方向。報各主要群的source norm/projected norm/retained fraction及全資料方向相似性，檢查群平均抵消。不存在可宣稱audit pass的數字；summary scientific_status='not_evaluated', purpose='development', n_independent_families=null，LO-group結果與in-sample分開。company margins僅描述、不作方向選擇。

REQ-8 輸出prepare/materials.json、prepare/prompts.jsonl、forward/records.jsonl(id,kind,margin與scalar重複診斷，不含states)、analyze/comparisons.jsonl、analyze/summary.json。不存任何高維向量，連聚合direction本輪也不持久化；將來正式candidate schema另作。每項register hash/count，run最後核counts、finite、所有比較coverage。memory states/directions只在run內RAM。未實作J-lens vocabulary readout，不宣稱P1a完成；real wrapper仍驗validated canonical lens供身份一致性。

驗收：`uv run pytest -q tests/test_entity_concept_development.py tests/test_entity_concept_development_materials.py tests/test_entity_concept_directions.py tests/test_core_subspaces.py tests/test_core_artifacts.py`。deterministic fake model至少17層、L15後非線性、真正core recording（不能全monkeypatch走過場）；測closed-form或可控signal、LO-group排除、controls不能參與fit、random可重現、all-degenerate可finalize、數值錯誤failed manifest、輸入错run前拒絕、exception hook cleanup、重跑run-id拒覆寫、輸出無tensor arrays。測試不載checkpoint。

## 4. 真實執行邊界（主流程負責，非上述delegation）

使用者已授權本agent執行bounded smoke/development。先用現有registry/core loader核canonical lens，strict completed-input verifier核e-01 Q與2A prompts，model `.cache/models/qwen3.5-4b`，GPU bf16 decoder，FP32計分。上游缺失不得補造。

第一個真實run使用完整60材料＋4個舊公司canonical prompts＋首筆重複=65次forward，batch=1。run mode=model_smoke；無介入、無formal語義gate、無audit；它不是原協議包含所有patch arms的完整Phase1端到端smoke。若需追加，限修復工程失敗的重跑與同一development材料診斷，不自行擴大公司／層／候選搜索。記每forward耗時、總耗時與GPU峰值，資源不足停止，不下載／換模型。

真實operator、準確命令及run結果在實作後新增script與報告，不在尚無程式時刊登虛構命令。對Phase1 V2的status同步為development可執行、正式pipeline未完成。先前未授權聲明是歷史節點，本次授權只覆蓋此節。
