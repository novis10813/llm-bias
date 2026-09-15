# Phase 3：早層公司置換的決策作用是否經候選概念傳遞

**狀態：proposed，Draft 1，依賴 Phase 2，未實作／未授權。**本階段有獨立批准與 run；不因 Phase 2 pass 自動開跑。文獻原始設定與 adaptations、Input/Output 基本 schema、CLI 非 runnable 狀態、數值／span／防禦與版本政策依[共通契約](design-and-validation.md) §1–7，為本協議的一部分。

## 1. 只檢驗指定中間分量的路徑參與

H3：早層公司區間狀態置換會改變 L15 候選概念分量及決策；將此分量恢復至 target 原值時，轉移作用減弱；只補入上游誘發的同一分量時，部分作用可重現。

這是受控內部介入，不等於已識別自然中介效應。即使成立，也不證明早層 Entity Cell 必要、概念唯一或訊號必經 dial；它們是另案問題。

## 2. 先固定上游，再使用獨立資料確認

development 沿用舊公司，但 Phase 1/2 的 concept 與 L15 位置已凍結。建議只檢查 L4、L8、L11 post-block entity span，代表早層／中段／承載帶末端；這是 proposed 搜尋集合，不是已知最好層。

對每層做跨公司整個 entity span 置換。此 span 不可用 instruction 的相同 token 要求：公司名自然不同。primary 僅使用 source/target **相同 entity token 數量**的 pair，採相對 offset 映射；不夠 pair 就停止／另立版本處理變長映射，不以 nearest mapping 靜默補齊。記錄周邊相同 prefix 與非介入位置。

只在 development 依上游 toward-source transfer 的预定門檻與固定 tie-break（合格者最早層）選一層，不能按「恢復效果最大」選層。若三層皆不合格，停止本版本；不推論所有早層公司表徵無作用。

confirmation 使用共通 §3 的独立公司與模板；Phase 2 看過的確認集不能再稱全新 Phase 3 confirmation。可在兩階段前共同凍結新公司分割，或另找新集合。

## 3. 恢復必須使用 hook 當下的狀態

同一 target prompt 記 h_t 為 clean L15 狀態，早層置換 U 後自然到達 L15 的狀態為 h_U。P_C 為 Phase 2 通過概念的 projector；位置均指 target instruction span。

| arm | 操作 | 判讀 |
|---|---|---|
| clean | 無介入 | M₀ |
| U | source 早層 entity 狀態置換至 target | M_U；先確認上游本身有效 |
| U_restore_C | 早層 U；L15 hook 當下 h←h+P_C(h_t−h) | M_UR；將候選恢復原值，不動其餘分量 |
| inject_U_C | 無 U；L15 h_t←h_t+P_C(h_U−h_t) | M_IC；只補入上游誘發的候選變化 |
| U_restore_R | 同 U，但恢復 K 内 frozen 隨機 rank-1 分量，另加同 norm 匹配版本 | 排除任意局部擾動造成衰減；自然恢復與 norm-matched 兩版本不可混稱 |
| U_restore_K | U 後恢復整個 k=8 分量 | L15 已知子空間參照 |
| U_restore_all_instruction | U 後恢復 L15 instruction 整體狀態 | 上界診斷，其他位置／歷史狀態仍可能保留 U |
| self-source／no-op | 相同 source/target、α=0；restore-to-current | 結構與 hook 對照 |

**不能用固定的 −P_C(h_s−h_t) 代替動態恢復**：上游 U 到 L15 的效應不是完整 source 狀態差。h_U 需另做 U-only capture，在 RAM 保存必要位置，完成該 pair 即釋放。restore hook 內讀 live h；測试要驗證 capture registration order 沒有讀錯 pre/post-transform 狀態。

Qwen3.5-4B 含 recurrence／hybrid attention；此法只介入 residual，不能還原其他 token 或內部歷史狀態。full instruction restore 不回 baseline 不必然是 bug，也不能藉此判定全部路徑已排除。

## 4. 三個條件缺一即不支持本次路徑主張

令 s=sign(M_s−M_t)，上游量 T_U=s(M_U−M₀)，恢復衰減 A_C=s(M_U−M_UR)，單分量重現 I_C=s(M_IC−M₀)。同時記 raw quantities 與概念係數變動，不把「朝來源方向」寫成「更接近來源」：過度位移仍須另看 |M−M_s|。

1. **上游連結存在**：T_U 超過 δ_M 且概念 loading 在 U 下改變超過 δ_a；eligible pairs 數與雙向皆符合凍結規則。
2. **特異性恢復**：A_C 大於 matched random 的恢復衰減，CI 下界超過預定效應界；U_restore_C 的概念分量確實回到 clean target 容差內。只見 margin 減弱卻未恢復概念，不支持操作有效。
3. **互補重現**：I_C 的符號與 T_U 一致、效果超過誤差與 controls；不要求 A_C=I_C，也不要求兩臂線性可加。

以不重疊公司 pair 為統計單位，合併 pair 內方向／模板前先個別報告；候選家族 Holm p<0.05，加預先凍結的實質效果界。δ_M/δ_a、上游 eligible 門檻、CI 精確度、random seeds、雙向最低樣本量及剩餘誤差上限需在 confirmation 前完成。小分母恢復比例只作 secondary，使用共通數值保護，不回報「百分之幾的全部 Bias」。

若只 1 成立：有公司資訊到候選表徵的證據，但未建決策作用。若 1、2 成立而 3 不成立：可報恢復操作在此情境下干擾了上游作用，不能按本版本稱完整支持。三者成立也僅支持指定層、概念、位置、材料下的路徑參與。未見效應與精度不足分開標 fail／inconclusive。

## 5. Input/Output、防禦與完整 preflight

Input：Phase 2 通過候選的 completed manifest/hash；development 上游層選擇檔；獨立 split/pair manifest；共通 model/tokenizer/basis refs。basis、方向與 source layers 不能在確認過程更新。

Output：共通 records 加 `early_layer, early_scope, restore_scope, upstream_effect, attenuation, injection_effect, score_change, score_restoration_error`，逐 pair／方向／模板／control seed 保留 compact finite 數值；summary 有三個 gate、eligibility、CI 與各模式 interpretation。不保存 h_t/h_U、逐位置全部係數、raw cache。

fail-safe：非同長 entity span、缺 early/later firing、非 finite、partial hook 註冊或清理失敗、缺 control、上游接近零皆拒絕或 inconclusive。fake model 需有已知兩條路徑以驗證「full instruction restore 不必然歸零」的合理情況。

正式前獨立批准真實 smoke，至少一個 pair 兩方向跑上表全部 arms、兩層 hooks、controls、analysis/finalize；檢查 live projection restoration 誤差與 caller exception cleanup。CLI 仍未實作，未來專屬 Phase 3 subcommand 不容許與 Phase 2 混用 mutually exclusive flags。任何核心更動遵守共通 §7 version policy。
