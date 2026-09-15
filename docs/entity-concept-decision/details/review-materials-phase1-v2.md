# Phase 1 V2 材料預審：建議排除4筆、修正7筆，暫不分割

**AI 預審建議，待使用者決策；不是獨立人工盲審，沒有模型結果。**來源為[原始64筆](materials-phase1-v2-draft.md)及[V2 協議](proposal-phase1-v2.md)。原稿不覆寫；以下 replacement 與分組尚未採用，沒有 approved inputs。

## 1. 結論與計數

建議保留原稿作歷史，排除 C-V02、C-V04 共4筆，另修正 S 的7筆文字。若採納，工作材料為60筆：36筆主要描述、24筆對照；原始家族標記剩26個，比較由40減為38。這些數目不等於獨立家族或檢定數。

主要描述先保守歸成4個改寫群：C兩群、S兩群。群定義見§3；它們是限制跨 split 的工作分組，不是對獨立樣本數的實測。所有材料暫留開發用途，取消原擬F/V分配的效力。不以新措辭、新ID或模型結果拆群。

本建議不更換概念、control family 或 gate。若後續增加固定目標狀態的對照、重新配置 fit/validation/audit 或減少正式預算，另修訂協議，不由本審查表暗中生效。

## 2. 28個原始家族逐項裁定建議

retain/revise/reject 僅是AI建議。retain 不表示獨立或已通過語義審核。

| 原始家族 | 建議 | 主要理由與用途 |
|---|---|---|
| C-F01 | retain | 百分比清楚；併入C1，不靠此單例排除數值大小 |
| C-F02 | retain | 直接收入分布；併入C2 |
| C-F03 | retain | 固定100客戶有助排除客戶數變化；併入C2 |
| C-F04 | retain | 同時固定收入與客戶數是優點；算術負擔另記，併入C1 |
| C-F05 | retain | 內容等價於大客戶收入占比；替換情境可能引入損失評價，併入C1，只作開發改寫 |
| C-F06 | retain | 是文字圖形描述，非視覺輸入；不能無模型就宣稱只量到圖形語義，併入C2 |
| C-V01 | retain | 累積占比與F01/F04共用推理關係；併入C1，不作獨立validation |
| C-V02 | reject | 「likely」範圍不明，抽樣敘述對一般讀者不自然；撤出主材料，不宣稱數學上錯誤 |
| C-V03 | retain | 業務經理只是大客戶占比的敘事變化；併入C1，管理職務干擾另記 |
| C-V04 | reject | 有效客戶數缺參照界，抽象術語理解可能主導；不藉改寫成占比補回獨立家族 |
| C-E01 | retain | 2×2比較完整；與C2共同來源，不獨立於其內容 |
| C-E02 | retain | 與E01是評價改寫，不多算一個獨立內容家族 |
| C-L01 | retain | 數值＋續約評價混合診斷，非純數值對照、非已知固定集中度 |
| C-X01 | retain | 規模競爭解釋診斷；目標未指定，不當作集中度負例 |
| S-F01 | revise | 正例未明確配對擴張上升／收縮下降；§4修正，併入S1 |
| S-F02 | revise | 差異大小未限定方向；正負皆改成較高多少，併入S1 |
| S-F03 | retain | 同向與幅度都有表達；併入S1 |
| S-F04 | retain | 同一pair均用假設語氣，無需全改成歷史；併入S2 |
| S-F05 | retain | 明確擴張高、收縮低；保留數值診斷限制，併入S1 |
| S-F06 | retain | 是S1的文字圖表改寫，不另算独立家族 |
| S-V01 | revise | 預測改善不等於變動幅度；改成情境收入差，併入S2 |
| S-V02 | retain | much higher已含幅度與方向，不採納「只有相關性」的指摘；併入S1 |
| S-V03 | retain | 與V01/F04共享情境比較，併入S2 |
| S-V04 | retain | 原稿已補明幅度；去趨勢敘述是S1延伸，不宣稱獨立 |
| S-E01 | revise | 正例只有大幅變動，可能包含反景氣；固定好壞後綴，只補同向性 |
| S-E02 | retain | 正例已指定復甦上升／放緩下降；與S1及E01共享來源 |
| S-L01 | retain | 僅檢查large/small在非收入內容的反應；語境距離待人工判斷 |
| S-X01 | retain | 當季增減不足以標記跨景氣敏感性；保留為未指定目標的診斷 |

## 3. 建議的主要材料改寫群

| 群 | 原始家族成員 | 共同描述方式 |
|---|---|---|
| C1 | C-F01, C-F04, C-F05, C-V01, C-V03 | 少數客戶占總收入多少，以及累積／替換／經理敘事的衍生表達 |
| C2 | C-F02, C-F03, C-F06 | 客戶收入分布均勻或集中，以及文字圖形改寫 |
| S1 | S-F01, S-F02, S-F03, S-F05, S-F06, S-V02, S-V04 | 已描述的跨景氣收入差，包含數值、圖形、去趨勢改寫 |
| S2 | S-F04, S-V01, S-V03 | 在擴張／收縮情境下假設或預估收入差 |

C1與C2、S1與S2仍可能相依；不能因分成兩群就指定一群fit、一群validation並宣稱確認。相同構念不自動等於改寫，但本批缺乏獨立作者與來源紀錄，採保守合併。

交叉對照的內容也必須記來源：C-E01/E02連到C2；S-E01/E02連到S1；兩候選共用評價後綴。正式分割時不能把這些控制單獨分配到另一split以假裝新內容。詞面/競爭內容控制另記比較角色，不加入概念方向fitting。

## 4. 七筆建議替換文字

以下ID表示原稿同ID的replacement，不是新增樣本。原稿P/N及PH/PL含義不變。

| ID | 建議英文描述 |
|---|---|
| S-F01-P | Across past economic expansions and contractions, the company's revenue rose substantially during expansions and fell substantially during contractions. |
| S-F02-P | The company's revenue has been much higher in economic upturns than in downturns. |
| S-F02-N | The company's revenue has been only slightly higher in economic upturns than in downturns. |
| S-V01-P | Forecasts for economic expansions put the company's revenue substantially above forecasts for economic contractions. |
| S-V01-N | Forecasts for economic expansions put the company's revenue only slightly above forecasts for economic contractions. |
| S-E01-PH | Company revenue rises sharply during economic expansions and falls sharply during economic contractions. An observer describes the overall business outlook as favorable. |
| S-E01-PL | Company revenue rises sharply during economic expansions and falls sharply during economic contractions. An observer describes the overall business outlook as unfavorable. |

F02與V01負例是「輕微同向」，符合較不敏感，不是收入絕對不動；兩極都使用higher/above，避免單靠方向詞辨識。仍可能只分辨much/slightly，需其他家族及對照，不把改寫後可讀當作已通過檢驗。

## 5. 未採納的審查建議與理由

read-only explorer 提供第二意見，但其意見不能直接當作已證實缺陷：

- 固定100客戶或固定總收入不是confound本身；變動分布而固定其他量是控制。算術負擔與目標混淆需分開。
- 不把圖表文字直接判定為只能辨識視覺語義，也不從語境生硬推定tokenizer／attention异常。
- 不用「average customer base」「normal sector averages」「typical baseline」取代未指定目標。這些措辭沒有固定集中度或景氣敏感性，反而會加入模糊的中性標籤。
- 當前L/X的對照角色已限定為未明示目標時的反應。若有反應，可能是詞面、評價或合理先驗，不足以單獨否定概念特異性；若無反應，也不足以证明已隔離混淆。
- 不以audit模型結果或相關矩陣重新分群來證明獨立性；材料來源與改寫關係須先定，統計假設另查。

## 6. 人工決策與後續邊界

這輪需要批准的是「排除兩組、七筆替換、保守合併、全數開發用途」的修訂方案，不是64筆或60筆已通過人工盲審。

若採納，先將60筆整理成新的待審稿並保留原始ID lineage。由人工逐筆判概念、評價與歧義，再制定新家族的補寫與正式分割。現階段沒有達到每候選6 fitting／4 validation家族的已批准目標，不因湊足筆數放寬。

本輪無程式／schema修改、無模型／tokenizer／lens載入、無GPU、無正式audit、未commit。
