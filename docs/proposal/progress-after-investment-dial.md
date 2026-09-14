# AI on a Dial 方法復現後：公司身分決策差異的定位與介入

**報告用途：**研究進度回顧，供指導教授與未讀過程式庫的研究同儕閱讀。  
**涵蓋範圍：**Investment-dial 方法復現，以及後續 Balanced Evidence Gap、Entity-to-Dial A–F、Selective-intervention V1；截至 2026-09-14 的 formal 結果。  
**共同模型：**Qwen3.5-4B，主要以 bf16 精度執行。本文是跨實驗研究進度摘要，不取代各線的 canonical protocol/report，不新增實驗授權，也不修改既有判準。

## 一、復現建立了全域立場調節能力，後續轉問公司差異從何而來

**目前已確認公司身分造成的決策差異，並找到能改變差異的內部狀態，但尚未建立沒有全局副作用的選擇性控制方法。**研究依序從行為確認、候選神經元驗證，推進到殘差狀態的定位與移除。

起點是 Park et al. 的 *Your AI, On a Dial* 方法復現：對 L15 的 MLP 神經元 N8490 加上偏移，即可調節模型整體的多空立場。這裡 L15 是從 0 起算的層編號，N8490 是該層 MLP 內部通道的編號；MLP 是每層中對文字位置的表示進行非線性變換的模組。Investment-dial V2 在 A 組校準劑量，再於 B 組評估三個目標立場，RMSE 為 **0.0579**，通過凍結門檻。[1] 這是方法復現，非原論文數值重製；B 組也曾用於先前候選排序，不能視為全新樣本確認。

後續問題因此改為：**若財務證據不變，公司名稱本身會不會改變判斷？這種差異是否也能由一個神經元控制？**主要量測採固定答案的 **margin**，即 buy 與 sell 的 logit 分數差；數值越高代表相對越偏向 buy，但增加不一定造成買賣翻轉。本文的 nats 是這類分數差使用的單位，不是機率百分點；gate 則指實驗前訂定的通過門檻。

## 二、公司名稱會影響決策，但三個晚期候選神經元未通過因果驗證

**Balanced Evidence Gap 先確認了行為差異，再把定位與因果驗證分開。**Phase 1 對 16 家公司的財務證據分別製作具名、匿名版本，共 256 prompts；具名相對匿名的平均 margin 增加 **0.432 nats**，95% CI 為 [0.350, 0.518]，128 對中有 **16 對決策翻轉**。[2] 這裡同公司配對保持證據一致，但不同公司仍使用各自的證據。

Phase 2A 因而改用所有公司共用的證據，產生 64 prompts。雖然答案全為 sell，margin 仍隨公司而異。接著使用 **activation patching**，把一個公司 prompt 在指定位置與層的內部狀態，置換進另一個 prompt，測量輸出變化。「位置」指模型切分文字後的 token 位置；entity span 是公司識別文字所在的區間，instruction span 則是要求模型作投資判斷的指令文字區間。兩者是同一份輸入中的不同部分，並非兩個模型。Phase 2B 支持早期公司名稱區間的 L0–11 承載帶，以及 L12–15 向指令區間的交接；指令區間的轉移指標在 L15 達峰值。[3]

然而，Phase 2C 以一階敏感度找到的 L19、L20、L26 三個候選神經元，在 Phase 3 加性介入時全部不勝過同層對照，**0/3 通過、零翻轉**；已知 dial 對照仍能產生明顯位移。[4] 因此，候選與決策相關，尚不足以證明候選是可操縱決策的因果通道。

## 三、路徑解剖支持低維狀態差，未建立單一 dial 瓶頸

**Entity-to-Dial A–F 將研究重心由單一通道轉向指令區間的整體狀態。**A–D 分別測試股票代碼／公司名稱、MLP／attention block 與 dial activation，預先指定的 gates 均未通過；部分 patch 雖有位移，卻一致朝 sell，不能直接解讀為成功複製來源公司的立場。[5]

### 子空間描述的是整體狀態的變化方向，不是另一組神經元

**模型在每個文字位置的殘差狀態，可以理解為一張由 2,560 個數值組成的內部狀態表。**這些數值共同表示模型處理到該層時的資訊，後續層會繼續讀取與更新它們。「殘差流」就是這份向量表示在層與層之間傳遞、累加更新的過程。這裡的 2,560 維與 dial 所在的 MLP 內部通道不是同一個空間，因此 N8490 並不是這張狀態表的第 8,490 格。

**比較兩個公司 prompt 的對應位置，就能得到一個 2,560 維的「狀態差」。**例如，其餘證據相同時，來源公司與目標公司的內部數值可能在許多格同時不同。把整個來源狀態換進目標，稱為完整狀態置換（full swap）；也可只將兩者差值的一部分加到目標，檢查這部分是否足以重現輸出變化。

**「子空間」是選定若干方向後，由這些方向的所有線性組合構成的範圍。**可以先想像三維空間中的一張通過原點的平面：平面內的移動只需要兩個座標描述，所以平面是一個二維子空間。同理，在 2,560 維的內部表示中，也可能只需少量方向就能描述與某組實驗有關的主要變化；每個方向本身仍由 2,560 個數值組成。

### 八個維度是八種可組合的變化方式，不是八個已命名的概念

**本研究的 k=8，表示選用八個彼此正交的基底方向。**沿著每個方向移動多少，由一個係數決定；八個係數加上固定的八個基底，就能重建子空間內的一個向量。這不是挑出八個神經元，也不是已找到「聲譽、風險、成長」等八種語義，更不是把整個模型壓縮成八個數字。

研究先收集凍結公司配對在 L15 各指令位置的狀態差，將它們堆疊後做奇異值分解（SVD），取得主要變化方向；協議將這組分析稱為 PCA sweep。「第一主成分」指排序第一、捕捉這批差值最多平方量的方向，不保證它有單獨可命名的語義，也不保證其介入效果最大。實驗比較 k=1、3、8、16 與完整差值，**八維是實際測過的一個選擇，並未證明八是唯一或最小的足夠維度**。

另須區分「八個公司轉移方向」與「八維基底」：前者是來源公司→目標公司的八種配對操作；後者是從多個配對、多個文字位置的差值向量中取出的八個幾何方向，兩者不一一對應。

### 0.983 表示保留八維分量後，能近乎重現完整置換的輸出效應

**投影的操作，是保留狀態差沿這八個方向的分量，捨去其餘方向，再用保留的差值介入目標 prompt。**如同把三維位移分成平面內與垂直平面的兩部分，投影只留下平面內的部分；此處只是把平面換成八維子空間。

Phase E 在 L15 測得：k=1、3、8 的介入效應相對完整狀態置換的比值中位數，分別為 0.737、0.885、**0.983**。[6] 例如，假設某個配對的 full swap 使 margin 增加 1.0 nats，八維投影介入使它增加 0.98 nats，該配對的比值就是 0.98；這只是計算示例，正式的 0.983 是六個符合分母門檻的方向之比值中位數。

這支持「所測狀態差的決策介入效果可以集中在少數方向」，**不是說八維保存了模型 98.3% 的知識、解釋了 98.3% 的所有公司偏見，或有 98.3% 的答案翻轉**。同階段也將兩家公司在指令位置的 dial activation 真實差值移植進模型，得到比值 **0.434**：未達門檻，但不能說 dial 毫無作用。

Phase F 再嘗試以「第一主成分方向＋dial」取代八維描述，聯合介入比值只有 **0.7329**，未達 0.85；匿名情境的方向推注也沒有超出數值擾動判定帶。[7] 因此，現階段保留八維子空間作為所測條件的描述，不把第一主成分稱為通用立場旋鈕。

## 四、子空間移除能把公司差異減半，但正式控制判準仍未通過

**Selective-intervention V1 問的是：既然這些方向能傳遞公司差異，拿掉它們能否讓公司間的判斷更接近？**這與 Phase E 的操作不同：Phase E 把另一家公司的狀態差放進來；V1 則不指定來源公司，而是減掉目前狀態在該子空間中的分量。

主臂先以 16 家公司在對齊位置的平均狀態作為中心，再將每份 prompt「偏離中心的向量」投影到八維子空間，減去這個投影。α 是移除比例：α=0 不動，α=0.5 減去一半，α=1 即 full strength，減去全部投影。**移除的是相對中心、沿八個方向的偏離，不是刪除八個神經元、把整份狀態清零，或從模型權重中刪掉公司知識。**其他方向在這一步不主動改動，但後續非線性運算仍可能改變輸出，因此需要副作用對照。

**結果是有效性陽性、整體判定陰性。**在 16 公司、64 prompts 上移除 L15 指令區間的八維分量，full strength 使預先固定的高低群組差異縮減 **53.8%**，公司間 IQR（四分位距，即中間 50% 公司分數的分散範圍）約減半；同維度隨機子空間只縮減 0.49%，通過有效性與 random 對照判準。[8]

副作用仍超標：具名 prompts 平均 margin 位移 **+0.3299 nats**，匿名 prompt 位移 **−0.2689 nats**，分別超過 0.15 與絕對值 0.10 的門檻。因此，**V1 formal verdict 為 fail**，且所有主臂樣本仍為 sell。這說明我們已能縮小公司間的分數差異，但尚未做到只調整公司相關部分而維持其他輸出穩定；低劑量結果不能事後拿來改判 V1 成功。

## 五、本階段完成機制探索，跨情境確認仍是後續工作

**本階段的研究產出是行為確認、狀態定位，以及對簡化控制假說的否證。**目前結果限於單模型、有限公司與英文財務模板；八維效應比值不是「解釋 98.3% 的所有公司偏見」，名稱效應也不直接等同有害偏見。

後續若保留上述發現作為主要研究主張，需在新公司、不同 prompt、模型與任務上獨立確認。若繼續研究選擇性控制，應另立版本，同時要求差異縮減與非目標輸出穩定，不放寬 V1 已凍結的門檻。

## 附錄 A：各階段完成了哪些檢驗，結果如何？

以下區分 **completed**（流程執行完成）、**pass/fail**（是否通過預先指定判準）與 **descriptive/discovery**（描述性或探索性證據）。完成不等於假說通過。

| 階段 | 研究問題與設計 | 主要結果及判定 |
|---|---|---|
| Investment-dial V2 | 單神經元加性介入能否校準整體立場？A、B 各 85 家公司，每組 340 prompts。 | 三目標誤差約 0.0824、0.0529、0.0215；RMSE 0.0579，gate pass。屬 B 組 reevaluation。 |
| Balanced Evidence Gap Phase 1 | 同一份公司別證據，具名與匿名是否不同？ | 256 prompts；平均 gap +0.432 nats；16/128 配對翻轉；三項描述性判準通過。 |
| Phase 2A | 改用共用證據後，公司差異是否仍存在？ | 64/64 sell，但公司 margin 不同；Rev 1 gate fail。Rev 2 以 Phase 1 named–anonymous gap 作參照並加入群組檢查後，五項通過。 |
| Phase 2B | 哪些位置與層的狀態置換能影響公司間差異？ | 8 個方向 × 32 層 × 4 個區間；支持 L0–11 entity span 與 L12–15 handoff，instruction span 的 normalized transfer 在 L15 為 0.464。Discovery，非完整電路還原。 |
| Phase 2C | 能否以敏感度提名組件？ | MLP 提名 L19/n6334、L20/n6520、L26/n2394；Holm 校正 p=0.0418。被測 5 個 full-attention 層、共 80 heads 的 attention arm 為 null；未涵蓋其他線性注意力層。 |
| Phase 3 | 提名神經元是否勝過 matched controls？ | 三個候選 gate 點平均 margin 位移絕對值約 0.0015–0.0122 nats，均不勝過對照；0/3 confirmed、零翻轉。 |
| 收線後 J-lens 診斷 | 候選方向經 Jacobian lens 轉為詞彙讀出後，有無特殊結構？ | 三候選的讀出落於同層隨機 controls 範圍；屬描述性表徵讀出，不另構成因果證據。 |
| Entity-to-Dial A | 早期效果是否主要由股票代碼承載？ | Ticker gate fail；name-group patch 效應較強，但 raw 位移一致向 sell、接近匿名基準。是否屬抹除而非來源立場轉移，仍需專用對照。 |
| Entity-to-Dial B | L12–15 的 entity span 是否由單一 block 承載？ | 單獨置換 MLP 或 attention block 貢獻均未找到合格層；Gate B fail。 |
| Entity-to-Dial C | 用具名／匿名的 dial activation 差做全位置推注，可解釋多少 gap？ | mean ΔM 約 +0.013 nats、CI 跨 0；mean absolute effect 相對 gap 的比值 0.051，低於 0.25；Gate C fail。 |
| Entity-to-Dial D | 將 block 搜尋移到 instruction span，能否找到合格層？ | L12–31 無合格層；8 方向的 toward-source 平均效應接近零，raw 效應偏 sell。D2 多層 top channel 的相關性很高，但產業一致性多數不足；Gate D fail。 |
| Entity-to-Dial E1 | 當層 attention 與 MLP 聯合增量是否足以重現一半完整置換效應？ | L15 joint ratio 中位數 0.575，門檻 0.5，Gate E1 pass。 |
| Entity-to-Dial E2a | 狀態差能否用少量主成分描述？ | k=1、3、8、16 的 effect ratio 中位數依序 0.737、0.885、0.983、1.001；低維集中為 descriptive 結果。 |
| Entity-to-Dial E2b | 在 instruction positions 移植真實 dial activation 差是否有效？ | Ratio 0.434，未達 0.5，Gate E2b fail；但未低於 0.05 falsifier，不是 dial 路徑的零效應結論。 |
| Entity-to-Dial F1 | 第一主成分加 dial 能否構成足夠的雙通道描述？ | 聯合臂／full-swap 比值中位數 0.7329 < 0.85；Gate F1 fail。部分方向存在負交互項，不能把兩個單臂效應直接相加。 |
| Entity-to-Dial F2 | 匿名情境中，第一主成分能否獨立推動立場？ | 第一主成分與 dial residual footprint 兩臂共 8 個判定點皆在 ±0.05 nats 帶內；均為 `context_dependent_or_null`。只涵蓋所測情境與劑量。 |
| Selective-intervention V1 | 移除 L15 k=8 分量，能否縮小差異而不改動非目標輸出？ | 663 forwards，含校準與對照；G1a/G1b/G2 pass，G3/G4 fail；full-strength 負結果。 |

## 附錄 B：Selective-intervention V1 的有效性與副作用需分開判讀

| 指標 | 未介入 | Full strength | 判定 |
|---|---:|---:|---|
| Frozen group gap | 1.2697 | 0.5863 | 縮減 53.8%，G1a pass |
| 公司間 entity-contrast IQR | 0.5640 | 0.2800 | 縮減 50.2%，G1b pass；接近半減門檻 0.2820 |
| Random 子空間的 group gap | 1.2697 | 1.2635 | 僅縮減 0.49%，G2 pass |
| 64-prompt mean margin | −1.7500 | −1.4201 | 位移 +0.3299，G3 fail |
| Anonymous margin | −3.2280 | −3.4968 | 位移 −0.2689，G4 fail |
| Sell→buy 翻轉 | — | 0/64 | 分數差縮小，未造成買賣翻轉 |

**群組差異與整體分散度是不同指標。**G1a 的 TOP={BLK, NSC}、BOTTOM={BDX, IT} 在實驗前固定；G1b 則衡量 16 家公司聚合分數的四分位距。表內 group gap、IQR 與 margin 均以 nats 計，不可與 Investment-dial 的離散立場指標或 RMSE 直接比較。

**控制實驗支持所測子空間有效，尚不足以宣稱一般能力保留。**Layer control 在五個被測層中以 L15 效果最強，但使用的是 16-prompt 子集；不同 centering 都能縮小 gap，16 公司中心的 mean shift 較小。Dial probe 在移除後仍有明顯反應，但此 probe 不能取代其他任務的能力評估。

**低劑量只呈現效果與副作用的取捨。**α=0.25 時，group gap 縮減 15.1%、mean shift +0.0918 nats；尚未達主要的半減目標，也沒有據此完成新的匿名穩定性驗證。V1 的 full-strength fail 不因這項觀察改判。

## 附錄 C：修訂與解讀邊界保留在研究紀錄中

1. **Phase 2A 的 Rev 1 失敗不被 Rev 2 覆蓋。**Rev 1 與 Phase 1「具名 margin」的 Spearman ρ=−0.411；該參照混入公司別證據效果。Rev 2 改用同證據配對的 named–anonymous gap，ρ=+0.448，並加入群組檢查後通過。此處採用階段報告與 Rev 2 協議的數字，不沿用收線摘要中不一致的相關係數。
2. **Phase 2C 的 pass 來自明確記錄的 gate 重評。**首次 formal 分析判 fail；修正 gate 實作後，對同一份存檔重新分析才得到 MLP arm pass。候選仍屬 discovery，Phase 3 的獨立介入驗證沒有通過。
3. **Phase 3 的方向判據有已記錄的內部矛盾，但不改變 0/3。**依字面判據或自洽的符號讀法，部分候選的方向判斷會改變；一致性與勝過 controls 的條件仍未通過。Null 限於三個候選、所測劑量與全位置加性操作。
4. **Phase C 的 0.051 與 E2b 的 0.434 不是同一實驗量。**前者把具名／匿名的 activation 差作全位置推注；後者跨公司、逐 instruction position 移植 activation 差，且以 full swap 為分母。不能把兩者當作 dial 作用比例隨研究時間增加。
5. **0.983 是介入效應比值，不是變異解釋率或全域偏見比例。**Phase E 預先要求完整置換的 |ΔM|≥0.2 nats 才納入比值；8 個方向中有 6 個有效，報告其中位數。基底與效應評估來自同一組研究方向，未建立新樣本泛化。
6. **近乎正交不等於因果獨立。**第一主成分與 dial residual footprint 的 cosine 約 −0.020；這是幾何描述。Phase F 的聯合介入顯示不能用兩臂線性和完整解釋輸出，但未定位交互作用的全部下游計算，也未證明其唯一來源是飽和。

## 附錄 D：主要 run 與來源索引

歷史 artifacts 與其 provenance 保持原狀。本文彙整 canonical 文件，未重跑模型或重新計算所有原始產物。

| 引用 | 主要 run IDs | 文件 |
|---|---|---|
| [1] 方法復現 | `calib-v2-gpu-bf16-01` | [收線報告](../investment-dial/report.md)；[V2 報告](../investment-dial/details/report-v2.md) |
| [2] 行為確認 | `balanced-gap-gpu-bf16-01` | [Phase 1 報告](../balanced-evidence-gap/details/report-phase1.md) |
| [3] 跨公司 probe、層帶與組件定位 | `phase2a-gpu-bf16-01`；`phase2a-rev2-gate-01`；`phase2b-gpu-bf16-01`；`phase2c-gpu-bf16-05`；`phase2c-gate-reanalysis-01` | [Phase 2 報告](../balanced-evidence-gap/details/report-phase2.md)；[Rev 2 協議](../balanced-evidence-gap/details/proposal-phase2-rev2.md) |
| [4] 三候選因果驗證 | `phase3-gpu-bf16-02` | [Phase 3 報告](../balanced-evidence-gap/details/report-phase3.md) |
| 輔助 J-lens 診斷 | `phase3-jlens-neurons-03` | [J-lens 診斷](../balanced-evidence-gap/details/diagnostic-jlens-neurons.md) |
| [5] 路徑解剖 A–D | `entity-to-dial-a-01`；`entity-to-dial-b-01`；`entity-to-dial-c-01`；`entity-to-dial-d-02` | [A–C 報告](../entity-to-dial/details/report-phase-abc.md)；[D 協議與 formal 結果](../entity-to-dial/details/proposal-phase-d.md) |
| [6] 聯合增量與低維狀態差 | `entity-to-dial-e-01` | [E 協議與 formal 結果](../entity-to-dial/details/proposal-phase-e.md) |
| [7] 雙通道與匿名方向推注 | `entity-to-dial-f-02` | [F 協議與 formal 結果](../entity-to-dial/details/proposal-phase-f.md)；[收線報告](../entity-to-dial/report.md) |
| [8] 選擇性介入 V1 | `selective-intervention-v1-gpu-bf16-01` | [V1 報告](../selective-intervention/report.md)；[V1 協議](../selective-intervention/details/proposal-v1.md) |

[1]: ../investment-dial/report.md
[2]: ../balanced-evidence-gap/details/report-phase1.md
[3]: ../balanced-evidence-gap/details/report-phase2.md
[4]: ../balanced-evidence-gap/details/report-phase3.md
[5]: ../entity-to-dial/details/report-phase-abc.md
[6]: ../entity-to-dial/details/proposal-phase-e.md
[7]: ../entity-to-dial/details/proposal-phase-f.md
[8]: ../selective-intervention/report.md
