# 重建以生成決策為主的 layer、neuron、DIM 與 cone 實驗

先固定完整 S&P 500 母體、兩個 evidence slot 與 schema decoding，再重跑定位和三種 steering 方法。現有結果保留為各自協議下的歷史觀測；不能只修改解析器或結果表，就把它們當成新實驗。

本清單是 2026-09-30 的審查與新設計，尚未實作，也未啟動 GPU run。審查基準為工作樹及本地 compact artifacts，HEAD 為 `b27500a51123435792acba2817bcc35f07fb1dac`。開始時已有使用者修改，本次不改那些檔案。

使用者指定的 `docs/concept-cone-steering/crossmodel-cone-paper/paper-draft.tex` 不存在。實際讀到的草稿是 [paper-draft.tex](docs/concept-cone-steering/paper-draft.tex)，其路徑在 index 中已 staged deletion，但檔案仍在磁碟。下面的論文行號指這份磁碟版本，不從舊 tag 還原。

## 1. 新論文先檢驗關係，再決定結論

論文的主線應是「替換何處會改變生成決策，三種介入如何與這些位置及彼此相關」。以下都是待檢驗假說，不能先放進摘要當結果。

| 問題 | 必須量到的證據 | 不足以回答問題的結果 |
|---|---|---|
| A：早層 entity、中後層 instruction 是否最有效？ | 全層、各 span 與 token 位置的 residual replacement，實際生成 buy/sell；模型別 effect、分母與不確定性 | fixed-prefix／teacher-forced margin peak，或只在 peak 附近驗生成 |
| B1：哪些 neuron 能改變偏好，是否集中在 A 找到的層？ | 真正的 MLP activation edit、多個獨立選出的 neuron、生成 dose-response 與 layer 分布 | 固定 L15/N8490，或把 DIM 最接近的 write vector 加到 residual |
| B2：單一 DIM 是否與有效 neuron 的 residual 作用相關？ | 同模型、同層、同 hook site 的有效 residual delta 與獨立 DIM 比較，再做移除分量的行為測試 | 用 DIM cosine 選 neuron 後再宣稱兩者相關 |
| B3：獨立 cone 是否與 DIM 相關，且能改變理由的內容？ | 獨立學得的 cone、每條 basis／抽樣 ray／組合的生成效果，DIM 對 cone 的投影，以及盲評的 reason 比較 | 把 DIM 放成第一軸，或只注入一次固定 centroid，或展示幾句不同措辭 |

幾何正交不代表行為獨立，非正交也不代表同一因果機制。若新實驗不支持預期的互補關係，就報告不支持；不調整建構方式來強迫關係成立。

## 2. 已確認的差異必須分成協議落差與程式問題

下表區分原論文方法、本 repo 的實作及本次研究目標。移植到金融任務是 adaptation；符合原作者的操作與目標後，才能進一步討論 replication 的範圍。

| 項目 | 原方法／本次目標 | 現在實作與證據 | 新版要做的事 |
|---|---|---|---|
| 母體 | 所有正式研究來自同一份完整 S&P 名單 | formal confirmation 已驗證 2024 年 503 個 ticker，402/101 split；`core/steering/protocol.py` L81-99。舊 DIM／operator 入口仍用 200 家來源與少量預設目標 | 保留完整母體驗證，移除新入口的替代名單與少量研究預設 |
| 選層資料 | selection 與最終 evaluation 隔離 | C2 runner 用 427 家 exploratory list；其中 89 家在後續 101 家 evaluation 中。`run_v2_427_crossmodel.sh` L24-28、`summarize_c2_overlap_exclusion.py` L1-6 | 先切角色，再在 development 角色做新定位；舊排除重算只補充歷史敏感度分析 |
| evidence | `[company_info][evidence1][evidence2][instruction]` | `core/steering/prompts.py` L20-25 的 balanced 是 P1/P2/N1/N2；其他條件是 0 或 2 則。L130-138 只提供合併 evidence span | 正式條件一律兩個非空 slot，獨立 spans；固定骨架與 instruction |
| 輸出 | 解碼時約束 schema，兩種 decision 都可生成 | `core/inference/generation.py` L11-42 直接呼叫 HF generate，沒有 grammar constraint；`decision_parsing.py` 是事後解析 | 新增 schema constrained greedy decoding；posthoc validation 仍保留 |
| 主要指標 | 實際生成的方向與 flip | confirmation 已用 generated ITT；legacy 報表有 paired-only 分母。`core/steering/summary.py` L34-64 仍依 steered rows 決定公司集合 | 所有新版評估從 frozen planned IDs 開始；缺列不能縮分母或產生正式完成表 |
| A：定位 | 生成決策上的全層／位置效果 | `probe_steering_confirmation.py` L988-1077 主要算 margin transfer；L1080-1138 生成只測 peak ±2、兩個 span | 新 prompt 上全層生成定位；補 evidence1/evidence2 和 token 位置，不由 margin 選層 |
| Park neuron | 在 MLP down projection 前，對 neuron activation 加 scalar；原文包含 prompt 和 decode | legacy `probe_operator_comparison.py` L200-214 有真正 prehook；confirmation L721-748 改成 DIM 選出的 write-direction residual injection | 獨立 discovery、多 neuron 生成驗證；原方法的 scope 與 matched-scope 比較分開 |
| Arditi DIM | 按 layer／position 建候選，再選一條 shared `[d_model]` 向量；addition／ablation 分開 | `core/steering/directions.py` L27-57 算 Top10-Bottom10 的 `[K,d_model]`；Top/Bottom 由固定前綴 margin 排序 | 依 generated labels／預先定義的 contrast 建候選，用 validation 選單一 shared vector；tokenwise 版本另命名 |
| Wollschläger cone | RDO/RCO 的 sequence addition、ablation 與 retain 目標，個別 basis 和 cone rays 都要驗 | `core/steering/directions.py` L116-161 是 DIM 第一軸加正交 eigendirections，再取固定等權合成向量；其 DIM cosine 是建構公式 | 主方法改為獨立 learned cone；DIM-anchored eigenspace 只作有明確名稱的 control |
| learned RDO/RCO | 金融 adaptation 要明列 source loss 到 buy/sell 的映射 | 新版確有梯度訓練，但 `probe_rdo_cone.py` L78-90、L219-254 是固定前綴二元 margin loss＋單步 masked KL，沒有 sequence ablation loss | 重新定義 sequence training objective 與 target responses，生成驗證決定模型有效性 |
| RDO validation | 上游 fit 不使用 validation；新 acceptance 依生成結果 | v2 L181-187 的 DIM anchor 使用全部 construction，包含其 validation；L214-227 用 `M>0`；L276-284 改成 sample/basis 合併平均 | train-only fit 所有 anchor／scale；恢復或明列原文分組 loss 權重；改生成 gate |
| 跨產業 | metadata 完整、每種方法有自己的 hold-out 證據 | 主 CSV 有 29 個 `Unspecified`；C2 prepare 有 411/427 blank sector；現有 LOSO 主證據限 DIM | 新 metadata 另版補來源；未解決的 sector 保留並報 coverage，不能替 cone／neuron 宣稱泛化 |

其中 CAL 並非已知 test leakage：四模型正式 CAL 各 24 家，與主 evaluation 無重疊。不要把合法的 construction 使用與選層污染混為一談。

RDO v2 的 gate 還有兩種語意：[proposal.md](docs/concept-cone-steering/rdo-cone-v2/proposal.md) L41 規定四組調參都不通過就不進 full；L59 又允許 full 階段未通過的 seed 做有標記的描述性評估。程式 L424-426 沒有自動驗證階段一通行資格，但「未收斂 seed 有評估」本身不違反 L59。現有 status 記錄沒有進 full。新版要把 pipeline 通行資格、seed acceptance 和 diagnostic evaluation 明確分開，不能聲稱目前已產生違規 full 結果。

### 2.1 論文中的可核對錯誤先列為文件修正

- [ ] 改寫摘要 L35-40、contributions L100-106、結論 L550-555：目前的定位是 readout transfer，neuron 是 write-direction control，cone 是 anchored centroid。不能用現有數字回答新版 A/B1/B2/B3。
- [ ] 修正 appendix L704 的「所有模型 self-patches exact no-op」。Gemma `confirmation-v1-20260925-full-01/c2v3_gen/result.json` 的 `self_patch_identical=false`：ALGN/BX/CHRW 的 reason 文字不同，7/10 完全相同，10/10 decision 相同。先分清 text identity 與 decision identity，不猜測差異成因。
- [ ] 修正 appendix L716/L725 的 GPT-OSS band。實際 artifact 是離散層 `[1,8]`，不能寫成連續 L1-L8。
- [ ] 更正新選層工作對 band 的解讀。[layer-check-v1/proposal.md](docs/concept-cone-steering/layer-check-v1/proposal.md) L7/L13 把 L4 當作 L1-L8 的 band 內部，但實際 `[1,8]` 不含 L4。標記此候選理由已被 audit 推翻，再另版定義候選，不改已完成 run。
- [ ] 修正 appendix L672 的跨 tokenizer token identity 說法。相同 suffix IDs 只在各模型內成立；四模型 K 是 100/100/98/99，沒有跨模型同 token IDs 的證據。
- [ ] 逐項標示數值的 run、解析器、分母、dose 選擇及分析時間。C8 的 margin 版、事後 generated monotonicity 版，以及新 learned cone 是不同測試。
- [ ] 核對三篇原文的作者／版本／方法描述；例如本地 bibliography 的 Paleka 名字需對原文核對。不要把引用年份或 method 名稱當作 replication 證明。

這些修正是後續任務；本次只新增 TODO，不覆寫正在刪除／重整中的 paper 檔案。

## 3. 新檔案架構以一份契約串起所有入口

以下全是 proposed 路徑，尚未建立。共用契約、生成、metrics 與幾何統計放 `llm_bias/core/`；一次性方法與訓練 loop 留在 script，不先建一套通用實驗框架。

| Proposed 路徑 | 職責與輸入／輸出 |
|---|---|
| `configs/concept-cone-steering/rebuild-v1/protocol.json` | 母體 snapshot、split policy、evidence 條件、schema、主要指標、seeds、controls、acceptance；freeze 後以 hash 綁定所有 run |
| `configs/concept-cone-steering/rebuild-v1/models/<slug>.json` | checkpoint／tokenizer／模板／precision／reasoning policy／hook 能力；不預填舊 peak 或 K 作新版結論 |
| `configs/concept-cone-steering/rebuild-v1/decision.schema.json` | 唯一 output schema；grammar backend 與版本記於 protocol |
| `data/concept-cone-steering/rebuild-v1/population.json` | 從現有 CSV 產生完整 membership、source hash、issuer/share-class mapping；大型／研究輸入仍不進 git |
| `data/concept-cone-steering/rebuild-v1/evidence_pairs.jsonl` | 全母體 evidence IDs、文字、來源、正負標籤、兩個 slot、order 與 hashes；不得依資料可用性取 intersection |
| `llm_bias/core/population.py`、`experiment_contract.py` | 唯一母體 loader、角色分派、planned row keys、完整性與 resume 相容性驗證；重用既有 manifest/path utilities |
| `llm_bias/core/prompt_input/decision_prompt.py` | 統一 renderer，回傳 generation IDs 與 entity/evidence1/evidence2/instruction spans；不強制帶 scoring prefix 或 buy/sell token IDs |
| `llm_bias/core/inference/structured_output.py` | schema compiler／grammar processor 與本地 HF generation 整合；保留既有 residual／MLP hooks |
| `llm_bias/core/decision_metrics.py` | 全部 primary outcomes、ITT、coverage、schema completion、失敗類型及公司／issuer cluster bootstrap |
| `llm_bias/core/steering/geometry.py` | 同座標系 cosine、span projection、positive-cone distance、compact induced-delta 統計 |
| `scripts/stance_baseline.py`、`stance_localize.py` | 完整母體 baseline；全層／span／token replacement，產出 selection 結果與 held-out 檢驗 |
| `scripts/stance_neurons.py`、`stance_dim.py`、`stance_cone.py` | 各自獨立建構／選擇 operator，明列原文 adaptation；輸出 derived operator、fit roles 與 provenance |
| `scripts/stance_compare.py`、`stance_geometry.py`、`stance_reasons.py` | common evaluation、非循環幾何／分量移除測試、reason 盲評與 evidence counterfactual |
| `scripts/stance_summarize.py` | 只讀合格新 run，產生主表／圖／claim rows；不能重新選 layer、dose、ray 或 parser |
| `tests/test_stance_contract.py`、`test_structured_output.py`、`test_decision_metrics.py`、`test_stance_interventions.py` | fake model＋temporary fixtures，驗研究契約與 hook 語意；不載入真 checkpoint |
| `docs/concept-cone-steering/rebuild-v1/{proposal,status,method-audit}.md` | 新協議、目前結論與原文實作對照；結果未跑前 status 標 proposed |
| `docs/concept-cone-steering/rebuild-v1/paper/{paper-draft.tex,custom.bib,figures/}` | 新主張與新結果的唯一論文入口；舊 draft 保留其歷史身分 |
| `artifacts/<slug>/concept-cone-steering/runs/<new-run-id>/` | 按 baseline/localization/neuron/dim/cone/calibration/comparison/geometry/reason 階段分組；每階段有 parent hashes 和 coverage，永不覆蓋舊 run |
| 同 run 的 `localization/selected-sites.json` | development/validation 決定的 layer／span／pooling、角色 IDs、source/protocol hashes；方法入口驗此產物，不從 model config 取預填 peak |

### 3.1 舊入口先停用於新主表，再安排遷移

| 現有入口／模組 | 新版處理 |
|---|---|
| `probe_dim_steering.py`、`probe_operator_comparison.py`、`probe_concept_cone.py` 的 legacy mode | 從新執行清單排除，標示 200 家／少量目標／歷史探索。不能直接沿用其 default operator |
| `balanced_evidence_gap_phase2*.py`、`entity_to_dial_heldout_transfer.py` | 保留凍結上游；新版不再 import 它們的實驗常數、ranking 或 margin 函式作主流程 |
| `probe_steering_confirmation.py`、`core/steering/{protocol,prompts,directions,evaluate,summary}.py` | 保留 confirmation-v1 的可重現行為；新契約另版，generic hook／manifest 工具可以重用 |
| `probe_rdo_cone*.py`、`probe_rdo_evidence.py` | 保留 margin-objective 歷史版本；新版 cone 入口重寫目標與驗收，不把舊方向改名冒充新方法 |
| `probe_evidence_scan.py`、`build_evidence_pool.py` | 留作 427 家、四則 evidence 的探索來源；新 evidence compiler 要完整兩-slot coverage |
| 舊 `plot_*`、`summarize_*`、`reparse_*` | 只對各自舊 schema 工作。新主表不能自動掃入任意 `result.json` |

- [ ] 建立 current-entrypoint 索引，只有新版入口可以產出新 paper 的 eligible result。
- [ ] 確認 imports、歷史命令與 provenance 之後，再決定是否搬入 `scripts/legacy/`；不能先大量搬檔造成引用失效。
- [ ] 保留舊 `proposal.md`、`status.md`、artifacts 與數值。新結果另開 `rebuild-v1`，不回填舊版。

## 4. P0 先修契約與 no-op，再容許研究 run

### P0.1 母體完整與 split 隔離必須可檢查

- [ ] 暫以現有 2024 S&P snapshot 的 503 個 ticker 為新母體，不悄悄換成當年最新成分股。來源 CSV SHA-256 為 `27c454d250ce513fda2016b43e7009ccbe0f5381e6c7648d3efe0d418c501a38`；另保存所選 membership 的 canonical hash。
- [ ] 固定角色：完整母體先切 development/evaluation；development 再切 fit/validation/calibration。layer、neuron、DIM、cone 的 fitting 不用 evaluation；validation 不進入 upstream anchor／scale fit；CAL 用獨立 calibration。
- [ ] 明列 ticker 與 issuer 的分析單位。GOOG/GOOGL、FOX/FOXA 等同 issuer 的 share class 不可跨 train/test 後宣稱 company-disjoint；保留 503 筆，不任意去重，角色按 issuer grouping 分派並記實際家數。
- [ ] formal baseline 覆蓋全部 503；steering 主檢驗用 frozen evaluation，完整角色聯集仍為 503。若論文要「503 家皆 held-out」的效果，另登記 issuer-grouped cross-fitting，不能在所有 503 上 fit 再 test。
- [ ] 新研究 CLI 不提供替代 smoke list 或任意 `--target-tickers`。可 shard／batch／resume，但它們只是同一完整 plan 的分工，合併後必須恰好等於 planned cohort；缺 evidence／錯 ticker／重複列直接失敗，不取 intersection。
- [ ] 29 個 `Unspecified` sector 要有缺失政策；確認資料來源後另版補 metadata，未補齊不能宣稱完整 sector-disjoint 結果。

驗收：錯 snapshot、漏公司、交叉 issuer、角色越界、缺 evidence、重複 shard 與相容性 hash 改變都會拒收。tiny fixtures 只用於 fake-model unit tests，不成為研究資料集。

### P0.2 所有條件維持兩個 evidence slot

- [ ] 固定公司資訊欄位及唯一 renderer；公司資訊後接 evidence1、evidence2，最後才是 instruction。模型必需的 system/chat wrapper 另記，不插入第二份任務指令。
- [ ] 新主條件 proposed 為 `(+,+)`、`(+,-)`、`(-,+)`、`(-,-)`；每筆恰好兩個非空 evidence item，item 可多句。instruction 和公司資訊保持相同。
- [ ] 把「極性比較」與「順序比較」分成配對 contrasts；測順序時交換同一對 evidence，不同時換內容。正負數量相同不代表說服強度相同，需事前標記來源、數字、強度與品質審核規則。
- [ ] entity、evidence1、evidence2、instruction 各自存 char/token spans、token IDs hash 與角色；chat-template／schema 改變後重算 K，不沿用舊 100/98/99。
- [ ] 先完成 full coverage 的共享證據配對，作為只變公司身分的主測試。公司專屬 evidence 可另版檢驗，但不能和主測試混合後歸因於 entity。
- [ ] 不把舊四則 evidence 截掉兩則後繼續使用舊 ranking／operator。外部 corpus 缺失、價格計算或標籤未驗證時，明列資料重建工作，不補造來源。

驗收：所有正式 prompt 符合四段順序及兩個 slot；只有測試 fixture 可以故意不合規。zero-evidence 等 ablation 若另設，仍保留兩個 slot 並明列 ablation 內容，不能偽裝成主條件。

### P0.3 schema 必須參與解碼，且不替模型決定 buy/sell

Proposed schema 如下；field order 另外由 grammar 固定為 decision 在前、reason 在後，JSON Schema 本身不保證順序。

```json
{
  "type": "object",
  "properties": {
    "decision": {"type": "string", "enum": ["buy", "sell"]},
    "reason": {"type": "string", "minLength": 1}
  },
  "required": ["decision", "reason"],
  "additionalProperties": false
}
```

- [ ] 在現有本地 HF `generate` 加 grammar／logits processor，所有方法、baseline、controls 共用。候選 XGrammar 的可用介面見文末官方文件；目前未安裝或驗證四模型相容性。
- [ ] `buy` 和 `sell` 都合法，只約束結構；不得用 margin 指定 enum，不 repair 出決策後算模型生成。
- [ ] grammar 限制閉合前 EOS、重複 key、額外 key；semantic validator 拒絕空白-only reason。固定 token budget、stopping、grammar version、thinking／channel policy 與 hashes。
- [ ] 對四模型分別驗 tokenizer/head vocabulary、special tokens、precision、reasoning channel 和 hooks。GPT-OSS 的 `low` reasoning 不等於已關閉 analysis；需要 channel-aware 方案的模型另記能力與 budget，不能靠任意剝字串混成同一生成契約。
- [ ] 主要成功要求完整 schema-valid object；已吐出 decision 但 reason 截斷只作 diagnostic，不用 partial JSON regex 挽救 flip。
- [ ] 分別記 `decision_complete`、`schema_complete`、reason validity、finish reason、timeout、exception、無合法 token。schema 約束不保證有限 budget 一定完成，也不保證 reason 合理。

驗收：用 fake logits 測非法 token 的分數最高時仍只能生成合法結構，buy/sell 均可被模型選出；覆蓋多 token 值、escape、EOS、截斷、duplicate keys 與缺 reason。

### P0.4 主要 evaluator 不再需要 margin

- [ ] 新 generation row、prompt contract、metrics 完全移除必填的 `score_ids`、buy/sell token pair、`margin`、`realized_margin`；舊 readout 僅留獨立歷史診斷入口，不加入新模型選擇或 loss。
- [ ] `sell→buy` 分母是該條件下 planned evaluation 中 α0 schema-valid 且 decision=sell 的全部單位；`buy→sell` 對稱。介入後 invalid／truncated 算未翻，分母不變；來源類別為 0 時是 NA。
- [ ] 同列 `N_planned`、baseline buy/sell/unknown、方向分母、flip count、schema completion、off-target、保持率和失敗類型。另報可觀察 flip 除以 `N_planned` 的保守 coverage 指標；baseline unknown 不猜類別。
- [ ] primary 不使用兩端皆解析成功的 conditional 分母。多 evidence trial 先按公司聚合，再按公司／issuer cluster 做 paired bootstrap，不能把同公司 trial 或雙向 pair 當獨立樣本。
- [ ] 缺 planned row 的 run 只能顯示進度，不能發布 complete summary。由完成生成得到的 failure row 保留；未執行的列不能偷偷消失。
- [ ] CAL、layer／neuron／checkpoint 選擇、收斂 acceptance 一律用 generated outcomes 與 schema／reason gates。門檻、seeds、dose grid 與失敗判讀先寫進新 proposal，不能看 evaluation 再選最好點。

### P0.5 no-op、hook 語意和環境先有證據

- [ ] 釐清 Gemma self-patch text 不同的原因：同 checkpoint、相同輸入重跑 α0、zero hook、self replacement，比較 decision、完整 token sequence 和 compact numerical diagnostics。先排除重複 baseline 差異、capture 設定、cache／kernel／precision 差異，再判定 hook 是否有錯。
- [ ] 事前固定 no-op／reproducibility 必須符合的條件；full 和其他研究 phase 一律執行。失敗時留下 diagnostic artifact，但禁止產生 eligible 主結果，不能像舊 `c2v3_gen` L1139-1141 只在 smoke 擋下失敗，full 卻仍記 `complete=true`。
- [ ] 新 intervention 明列 hook site、prompt 絕對 positions、prefill/decode scope。現在 `seq_len==1` 略過 decode 的 heuristic 不能描述 `use_cache=False` 的完整重算；chunked prefill 與 cache policy 也需獨立驗證。
- [ ] reuse 真正的 `core/inference/mlp_addition.py`，測 zero scalar、特定 neuron／position 變化、decode scope、exception 後 hook cleanup。Gemma/GLM 的 normalization 和 GPT-OSS expert routing 不用靜態 weight×gain 近似冒充真正 activation edit。
- [ ] 不保存 raw activations、residuals、gradients 或 KV cache；只保留 compact stats、generated text／decisions、derived operators 和 provenance。
- [ ] 依 README 重建缺少的 editable `third_party/jacobian-lens`，再 `uv sync`；不是改掉 workspace 宣告來掩蓋缺 dependency。

本次實際執行 `uv lock --check`、`uv run pytest -q`，兩者都在啟動前失敗：`jlens` 在 `tool.uv.sources` 指定 workspace，但本地不是可用 workspace member。沒有測試通過的證據。本次未安裝環境；TODO 文件另做 path／格式檢查。

## 5. P1 重跑定位及三種方法，避免在建構時預設答案

### P1.1 A 用全層生成 replacement 找位置

- [ ] 全 503 母體、全部登記的兩-slot 條件先生成 baseline，fit／validation／calibration／evaluation 各角色皆有自己的原始輸出。
- [ ] 在 development 上分開做：相同 evidence 的跨公司 replacement，以及同公司跨 evidence replacement。前者檢驗 entity-context，後者檢驗 evidence effect；不混合成一條「偏好傳遞」曲線。
- [ ] 全層測 entity、evidence1、evidence2、instruction replacement，並做預先登記的單 token／局部 token 位置測試。source capture 只用原 prompt，不把已生成答案 prefix 當成主 causal localization 的 donor。
- [ ] 跨 layer／span 比較使用同一組 donor-target pairs，記 span token 數與實際 residual delta norm。全 span replacement 是主測試；另做相同 token 數／可比 perturbation 大小的控制，避免長 instruction 只因改動較多位置就被解讀為特定 site 更有效。
- [ ] source／target 長度不同時，把 token alignment 規則與控制寫定；instruction 的共用 token 用 exact mapping，entity 的 mapping 敏感度另報，不默認 nearest mapping 就代表相同語意位置。
- [ ] 主要量是相反 clean decisions 的 toward-source generated flip，附 schema rate、n 與 cluster CI；相同 clean decision 的 pair 另報保持／reason 變化。沒有相反 pair 時不能靠 margin 造分母或宣稱成功定位。
- [ ] 用 development 定 peak／band 與候選層，再在 frozen evaluation 檢驗早層 entity、中後層 instruction。報模型特定 depth，允許假說不成立。GPT-OSS 歷史生成在 L8 的 entity 是 12/28，suffix 是 7/28，不能先宣稱後段 instruction 普遍更有效。
- [ ] patching 層與 additive steering 最佳層分別驗；A 的 peak 只是 B 的候選，不假設兩種操作必然同層。

### P1.2 Park 先找出多個有效 neuron，再研究其 layer 與方向

- [ ] 主方法對 MLP intermediate activation 真正加 scalar；在全部 layers 或事前固定的跨層分層 candidate panel 中 discovery，報每層搜尋 coverage。不能只搜尋 A 已選出的那一層再宣稱 neuron 與定位層相關；不固定 L15/N8490，也不按 DIM cosine 挑選。
- [ ] 原 Park 的 derivative screening 與最終 generated-prior/RMSE 選擇分開記錄。若保留原文 logit-gradient screening，只能作明標原文對照的 secondary replication；新版主 discovery 以 generated intervention response 篩選，不讓固定前綴 margin 作候選驗收。
- [ ] 事前固定候選範圍、dose/search budget 與 multiplicity 處理；每個有效候選報 held-out flip 曲線、兩方向分母、失敗率、layer 與 effect 分布，才能支持「很多 neuron，但幅度不同」。
- [ ] 原方法 prompt＋decode scope 和共同 prompt-only／instruction-only scope 都明確標示。方法忠實度比較與 scope-matched 比較各自報表，不能混成 operator 優劣。
- [ ] dense neuron 用相同 residual site 的實際 induced delta 比較，分別量小 scalar 的局部方向與正式 dose 的有限改變；normalized/MoE 架構納入 context、dose、normalization 與 routing。GPT-OSS 若尚無真正 routed neuron hook，標 unsupported，不能以 residual write control 代替。

### P1.3 Arditi 版本要產生真正的單一向量

- [ ] 先由 construction 的 schema-generated buy/sell 建 contrast groups；資料不足時停止該 estimand。不得用 margin extremes 把全 sell 的 baseline 假裝成 buy/sell 兩群。
- [ ] 優先檢驗相同 mixed evidence、不同公司產生的兩類；若改成同公司正／負 evidence contrast，明標這是 evidence-response direction，不能自動解讀為 entity preference direction。
- [ ] 按預定 layer／position 或明列 pooling 建 `[d_model]` DIM 候選，validation 用生成結果選一條；tokenwise `[K,d_model]` 另作 adapted control。
- [ ] addition 與 projection ablation 各有 scope、符號、dose 與假說。單方向有效果不代表雙向皆有分母；不同模型各自 fit，不能共用 Qwen ranking。

### P1.4 Wollschläger 版本要訓練並測試整個 cone 的多個方向

- [ ] 寫 source-to-finance loss 對照：sequence addition CE、ablation CE、retain sequence KL 的資料、目標、scope、權重和正負類；不以固定 JSON prefix 的 `softplus(-M)` 取代整段目標。
- [ ] sequence targets 只由 construction 建立，明列回應來源、schema format、evidence/label 審核，不用 evaluation reason 做 teacher target。train-only fit normalization／anchor；不再讓 validation 進入 DIM scale fit。
- [ ] CE／KL 是明列的可微訓練 surrogate，不是 primary outcome；greedy decision 不可直接反向傳播。若改用離散 generated reward／derivative-free optimization，另登記為 objective adaptation，不冒充原作者 RCO。
- [ ] checkpoint／hyperparameter acceptance 用 validation 真實生成的 flip、schema completion 與 evidence/retention 檢查，不用 `M>0`。階段一是否通行要可由程式驗證；未通過者不能當作已驗證 operator，但所有嘗試 seeds、失敗數與原因都進方法完成度／訓練成功率主表，不只保留成功 seeds。事前固定指定 seed／聚合政策；全數失敗仍是該方法的結果。
- [ ] basis 用獨立初始化／objective，不能固定 DIM 為第一軸或強制與 DIM 相關。核對 sample loss 與 basis loss 各自平均再相加的權重，不把所有方向合併平均後稱同一方法。
- [ ] 每條 basis、centroid、預先固定正係數 ray panel 都測兩方向生成 curves，並做多 training seed。記中位數、較差 rays、schema 失敗與共同 dose 範圍，不只報最好 ray。
- [ ] 正係數組合定義的是正向 cone `C`；負 dose 測的是反向 `-C`，屬雙向金融 adaptation。兩者各自報效果與分母，正向 ray coverage 不能驗證反向 cone；projection ablation 也是另種操作。
- [ ] best-of-N 若報，標為 oracle 上界，對照相同嘗試次數／compute budget 的獨立 1D directions；部署用 ray／權重只能在 validation 選。有限 ray panel 的成功不是「整個連續 cone 都有效」的證明。

## 6. P2 統一比較、獨立幾何和 reason 內容

- [ ] 每個 method 使用相同公司、兩-slot evidence、schema、template、decision metrics 和 evaluation plan；scope 差異另外拆 arm。
- [ ] dose 至少報 native scalar、實際 residual perturbation norm、相對 clean residual norm、層／位置／作用時長。共同 norm 比較與 projection-matched 比較分開，後者只能作刻意條件化的 secondary control；raw α 不代表相同強度。
- [ ] controls 包含 α0、zero/self patch、matched-norm random、多個 random seed、random neuron、label-shuffle、1D learned direction、DIM-anchored cone 和獨立 random subspace。每項按相同來源類別算 on-target flip；any-flip 和目標 flip 不混成同一 primary 差值。
- [ ] 在相同模型／layer／residual site 上測 independently selected neurons→DIM cosine 分布、DIM→cone basis／ray cosine、DIM 投影到 cone span 的 norm ratio，以及 positive-cone 非負投影距離。只在 span 內但需負係數，不算在 positive cone 內。
- [ ] 幾何統計納入隨機／選擇偏差對照；移除 neuron-induced direction 的 DIM component、移除 cone 的 DIM-aligned component，再重測生成。從這些行為效果評估共享機制，不從角度直接推因果。
- [ ] 分量移除同時報原始未重縮放效果及移除後重新匹配 norm 的效果，避免把幅度減少誤認成共享機制被移除；不跨模型或不同 residual site 直接算 coordinate cosine。
- [ ] reason 實驗配對相同公司／evidence／decision／可比 perturbation strength，比較 basis／ray 與單 DIM；先排除「決策不同所以理由不同」和單純改寫措辭。
- [ ] 相同 decision 的篩選屬介入後條件化，另報無條件 paired 結果、保留／排除數及 coverage。若需 stance-matched dose，只在 validation 校準後固定，不在 evaluation 逐列調 dose 直到得到所要決策。
- [ ] 事前固定 blind annotation rubric：引用了哪個 evidence、偏重的風險／成長因素、是否增加未提供的公司知識、矛盾／重複／無內容；標註者不看 method/ray 名稱，報一致性及配對不確定性。
- [ ] 用 evidence slot 替換／順序反轉的 counterfactual 檢驗 reason 的反應；不增加第三個 evidence。文字多樣、schema 合法和 flip 各自成指標，不等同 faithful reasoning 或金融概念識別。
- [ ] company/issuer-disjoint、split seeds 與 sector-disjoint 若保留為論文主張，對 neuron、DIM、cone 各自重做。DIM 的歷史 LOSO 不能替其他方法背書；27B 不在本輪模型範圍。

## 7. 重跑清單按依賴排序，舊結果只能回答舊協議

P0 通過後，依序跑 R0→R1→R2/R3/R4→R5→R6/R7→R9；R8 依保留的主張決定。R2/R3/R4 可以獨立執行，但不得用彼此結果強迫方向相關。每個模型都需要自己的 baseline、定位、fit、calibration 與 evaluation。

| 新 run | 優先級／是否必須 | 要取代或補上的現有實驗 | 必須重跑的原因與完成證據 |
|---|---|---|---|
| R0：full baseline＋generated labels | P0，必須 | confirmation alpha0/ranking、427 evidence scan | 兩-slot＋schema 改變生成／分類；完整 503 各條件 coverage、未知類別與角色 manifests |
| R1：全層／span／token generated localization | P1，必須 | 16-company C2、427 C2、c2v3/c2v3_gen、舊 layer-check 候選 | 舊 peak 依 margin／不同 prompt／混入 evaluation；新 source/target 分母、no-op、每層生成效應 |
| R2：many-neuron discovery＋真正 activation intervention | P1，必須 | L15/N8490 pilot、confirmation neuron comparator | 舊 comparator 不是 neuron edit；新獨立候選、layer 分布、dose-response 和雙向分母／untestable 說明 |
| R3：single-vector DIM extraction＋validation | P1，必須 | tokenwise DIM、crossmodel DIM layer sweeps | 舊 ranking／方向／K 全部改變；shared vector、fit provenance 與 held-out generation |
| R4：independent learned cone/RDO/RCO | P1，必須 | SVD cone、anchored cone、rdo-cone-v1/v2/evidence | 舊 centroid 不測 cone coverage；舊 learned loss/gate 目標不同；新 sequence objective、accepted seeds、basis/ray panel |
| R5：各方法 CAL＋matched operator/control evaluation | P2，必須 | confirmation cal/dim/ops/random/jitter/shuffle、舊 C8 | 新 operator 和 decoding 需新 dose calibration；schema rate、ITT、off-target、共同 norm/scope 曲線 |
| R6：non-circular geometry＋分量移除生成 | P2，必須 | 目前 DIM-picked neuron／DIM-first-axis 幾何描述 | 現有相關由建構指定；新版同座標幾何＋行為控制才可檢驗互補假說 |
| R7：ray-specific reason／two-slot evidence contrasts | P2，必須，對應使用者的 reason 目標 | evidence/anon arm、RDO 三家公司 reason 觀察 | 需同 decision 盲評、證據採納、反事實一致性；不只找不同句子 |
| R8：額外匿名／產業／多 split robustness | P2，有相應論文主張才必須；多 training seed 已包含於 R4 | generalization-v1、anon、split_seed、loso_construction | 新 data/scope/outcome 不能沿用舊泛化；逐 method 做 coverage，缺 sector 另報 |
| R9：新表／圖／claim ledger／paper | 依賴上述完成，必須 | 現有草稿與 paper figures | 只取新 protocol、同解析器與分母的 eligible rows；圖表可由 compact artifacts 重建，保留負結果與 NA |

### 7.1 可以 CPU 重算的工作不能冒充新實驗

- [ ] 從舊 generated text 重算各自舊 protocol 的 transition／ITT／conditional 比率，補分母與來源 SHA；這是 audit，不是 schema run。
- [ ] 重算／核對歷史 C2 evaluation overlap exclusion、GPT band、自我替換與 margin-vs-generation 差異；原 artifact 不變，新衍生 audit 有自己的 run ID。
- [ ] 保留已有真實 generated flip、RDO 的單向效果及未收斂紀錄；不能因新版目的不同就宣稱那些觀測完全無效。
- [ ] 重生成兩-slot/schema baseline 和 operator，不能重解析四-slot輸出、換標籤或重命名 loss 來省掉 R0-R5。

## 8. 完成標準是契約可驗、主張可追溯

- [ ] 所有正式入口吃同一完整 membership 和兩-slot prompt，沒有別的公司名單或默認少量 research target。
- [ ] fake tests 覆蓋 dataset/role leakage、schema 兩類皆可生成、ITT invalid／zero denominator、planned row completeness、resume hashes、actual neuron hook、cache/scope、exception cleanup、independent geometry 與 acceptance。不以單純鏡像公式的測試代替研究契約。
- [ ] 環境修復後執行 `uv lock --check`、`uv run pytest -q`；checkpoint 相容性／GPU 驗證依正式完整 manifest 開始並可分批續跑，不另建立縮小公司清單。
- [ ] 每個 figure/table 的母體、fit/evaluation IDs、schema/template/model/code hashes、primary outcome 與 source artifact 均可追溯；missing、unsupported、untestable 和 negative result 各自標示。
- [ ] paper 圍繞 A→B1/B2/B3→共同生成／幾何／reason 檢驗；舊 cone smoothness 和 margin discrepancy 可作歷史背景，不替新假說提供結論。

## 9. 審查來源及新設計仍需凍結的選擇

原始方法以實際查閱版本為準：[Park et al., Your AI, On a Dial, 2608.22852v1](https://arxiv.org/html/2608.22852v1)、[Arditi et al., 2406.11717v2](https://arxiv.org/html/2406.11717v2)及[作者程式](https://github.com/andyrdt/refusal_direction)、[Wollschläger et al., 2502.17420v2](https://arxiv.org/html/2502.17420v2)及[作者程式](https://github.com/wollschlager/geometry-of-refusal)。Park 原文的 427 ticker／其他模型不等同本次 503 ticker／Qwen3.5-4B；不能沿用原 neuron 座標。

本地證據入口：[claim-to-evidence.md](docs/concept-cone-steering/claim-to-evidence.md)、[confirmation-v1](docs/concept-cone-steering/confirmation-v1/proposal.md)、[C2 v3](docs/concept-cone-steering/c2-v3-steering-prompt/proposal.md)、[RDO v1](docs/concept-cone-steering/rdo-cone-v1/proposal.md)、[RDO v2](docs/concept-cone-steering/rdo-cone-v2/proposal.md)、[evidence-scan](docs/concept-cone-steering/evidence-scan-v1/proposal.md)。具體程式證據的表中，`core/...` 是 `llm_bias/core/...`，腳本 basename 是 `scripts/...`。

Schema 整合的能力依據是 [HF generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)、[XGrammar quick start](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html)及[compiler API](https://xgrammar.mlc.ai/docs/latest/api/python/grammar_compiler.html)。若採 XGrammar，須鎖版本並驗 ordered required/unique keys，不能從文件能力推定本 repo 四模型已兼容。

新版 protocol 在第一個研究 run 前仍需凍結：issuer mapping／角色實際家數、完整 evidence pairs 及品質規則、schema backend／各模型 channel policy、generated neuron search budget、DIM estimand／pooling、sequence targets／loss weights、cone dimension／rays／seeds、acceptance 門檻及 dose normalization。這些是明列的新設計項目，不是宣稱已實作的設定。

文件的組織骨架是：先建立共用契約，接著驗 A 的位置，再獨立驗三種方法，最後用共同生成評估、幾何和 reason 測互補假說。前三項是主結論必要內容；sector robustness 與歷史 margin 診斷按論文保留的主張決定範圍。
