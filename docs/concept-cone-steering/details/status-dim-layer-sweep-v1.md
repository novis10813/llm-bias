# Qwen3.5-4B DIM paper cohort V1：L16 不是唯一能改變生成判定的層

**範圍與狀態（2026-09-25）：**完成 Qwen3.5-4B、2024 S&P 500 的 402 家建構／101 家受測、固定七層的逐 token DIM 掃描；另完成六層的單向量補充掃描，L0 在其事前指定的 block-input 提取位置沒有可正規化的方向。兩臂都依[事前協議與執行後勘誤](proposal-dim-layer-sweep-v1.md)分開解讀。**主要發現：**逐 token DIM 在 C2 高 `T` 的 L14–L18 都使固定答案 margin 上移，並有可解析的真實生成 sell→buy；但 L15 的早期翻轉與 alpha 6 平均位移高於 C2 `T` 峰 L16，故不能說只有 L16 有此能力。這是預先指定層的描述性比較，不是獨立選層確認或方向特異性證據。

## 逐 token DIM：L15 比 L16 更早翻轉，L14–L16 高強度時都達 101/101

主要臂在**各層建立自己的** Top10−Bottom10 方向，對同一 100-token 指令尾段的 post-block 殘差逐 token 單位化並注入同層，僅改 prefill。每層 alpha 0 的 101 家都是嚴格 JSON `sell`，且計分及生成文字逐公司與[原 paper cone alpha 0](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/c2-guided-paper-20260924/result.json)完全相同。以下 `sell→buy` 是以**該公司 alpha 0 實際生成判定**為基準、且兩端都嚴格可解析的有效配對；「—」表示該 alpha 沒有有效配對，**不是 0%**。所有 `buy→sell` 的 alpha 0 基數為 0，該方向比率各層各 alpha 均未定義。

| 同層建／注方向：sell→buy／有效 sell 配對 | α0 | α2 | α3 | α4 | α5 | α6 |
|---|---:|---:|---:|---:|---:|---:|
| L0（C2 `T` −0.002） | 0/101 | 0/101 | — | — | — | — |
| L14（0.254） | 0/101 | 12/101 | 50/101 | 86/101 | 101/101 | 101/101 |
| L15（0.400） | 0/101 | 29/101 | 98/101 | 101/101 | 101/101 | 101/101 |
| L16（0.408） | 0/101 | 3/101 | 15/101 | 50/101 | 99/101 | 101/101 |
| L17（0.319） | 0/101 | 0/101 | 7/101 | 15/101 | 43/101 | 85/101 |
| L18（0.287） | 0/101 | 0/101 | 4/101 | 7/101 | 14/101 | 21/101 |
| L31（−0.007） | 0/101 | 0/101 | 0/101 | 0/101 | 0/101 | 0/101 |

相同 101 家的平均配對固定答案 margin 位移，`mean(ΔM_alpha)=Σ_i[M_i(alpha)−M_i(0)]/101`，單位 nats，**不是** C2 的 normalized transfer `T`；位移不能填補未解析的生成分母。

| 同層建／注方向：mean(ΔM) | α0 | α2 | α3 | α4 | α5 | α6 |
|---|---:|---:|---:|---:|---:|---:|
| L0 | +0.000 | −2.715 | +0.168 | +3.099 | +3.234 | +3.123 |
| L14 | +0.000 | +1.994 | +2.550 | +2.862 | +3.147 | +3.481 |
| L15 | +0.000 | +2.253 | +3.314 | +4.151 | +4.756 | +5.201 |
| L16 | +0.000 | +1.496 | +2.187 | +2.892 | +3.586 | +4.233 |
| L17 | +0.000 | +1.310 | +1.851 | +2.376 | +2.882 | +3.362 |
| L18 | +0.000 | +1.190 | +1.641 | +2.019 | +2.348 | +2.621 |
| L31 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

**嚴格 parse 邊界決定能否宣稱翻轉：**L0 的 α0／2 為 101/101，α3／4／5／6 均為 **0/101**（生成以說明文字開頭，不是完整 JSON）；其他六層每個 alpha 均為 **101/101**。因此 L0 在 α6 的 +3.123 nats **不能**當作生成判定翻轉。L31 的 post-block 只改已完成的指令 token，後面沒有混合位置的 decoder block，全部 101 家計分位移精確為零且無翻轉；這個注入位置有結構性限制，**不是**可一般化的「低 `T` 層都無法 steering」證據。[原 4D cone L16](../crossmodel-cone-paper/status.md) 的 alpha 6 為 6/101 sell→buy、平均 +1.607 nats；其方向建構用 Top／Bottom 各 20、產業去均值 SVD，而此處用各 10 的均值差，故這個同位置／近似同每 token 劑量的描述差**不能**單獨歸於 1D／4D，也不能當優劣證明。

## 單向量全 token 臂：margin 有位移，但六層皆無生成判定翻轉

補充臂在**各層自身的 block 輸入**、共同指令尾段最後一個 token 擷取單一 Top10−Bottom10 均值差方向，對該層所有 prefill 和 decode token 加同一單位向量。此法接近 Arditi 的加法**機制**，但金融分組／選層並非原文復現；其位置與 token 數皆不同於主要臂及 C2 的 post-block，數字相同的 alpha 不代表總注入量相同。L0 在建構階段[實測 `||v_0||₂ = 0`、兩組狀態各只有一種值](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-paper-01/single_all/l0_fit_diagnostic.json)，小於事前固定的 `1e−8` 門檻，依協議 fail closed，未對 L0 生成任何決策；其餘六層各 alpha 嚴格 parse 都是 **101/101**、alpha 0 全為 sell、`sell→buy` 均 **0/101**；`buy→sell` 基數為 0，方向比率未定義。

| 單向量全 token：mean(ΔM)，nats | α0 | α2 | α3 | α4 | α5 | α6 |
|---|---:|---:|---:|---:|---:|---:|
| L14 | +0.000 | +0.205 | +0.473 | +0.602 | +0.648 | +0.619 |
| L15 | +0.000 | −0.720 | −0.777 | −0.668 | −0.539 | −0.477 |
| L16 | +0.000 | +0.279 | +0.494 | +0.697 | +0.879 | +1.030 |
| L17 | +0.000 | +0.502 | +0.711 | +0.901 | +1.057 | +1.225 |
| L18 | +0.000 | +0.101 | +0.244 | +0.425 | +0.627 | +0.843 |
| L31 | +0.000 | −0.032 | −0.047 | −0.063 | −0.079 | −0.092 |

## 本次比較不構成獨立的 L16 選層或方向特異性確認

**公司重疊（執行後核對）：**C2 427 家的[方向名單](../../../artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-v2-427-01/pairs/directions.json)與本次 101 家有 **89 家交集**。101 家沒有用於 DIM clean ranking、方向建構，也沒有根據其 DIM 結果重選層；然而 C2 的選層本身不是公司名單獨立的留出資料。上方協議的「101 家不可參與層位選擇」已於[協議末尾勘誤](proposal-dim-layer-sweep-v1.md)標明與實際來源不符；不能把「建方向留出」寫成「整個選層流程獨立」。

**對照限制：**只有 α0 及低 `T` 層作對照，沒有 matched random directions、不同資料的確認組，且 L0／L31 各受格式或介入位置限制。選定的七層不足以估計全部 32 層的 `T`—DIM 強度相關或宣稱 layer-local causal peak；與單向量臂比較也改變注入位置／範圍。對於是否只有 L16 真正影響生成，本版可回答「**不是：L14、L15、L17、L18 在本設定也有嚴格解析的翻轉**」，不能回答其他方向、所有模型或未知控制下是否仍有此模式。

**查證與重建：**主要臂 [逐 alpha 表](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-paper-01/tokenwise/result.md)／[JSON](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-paper-01/tokenwise/result.json)（SHA-256 `5a086da0b4c6f57564b59a86f7afe2bd597f7a2d7dd61908567f9b93bfc3ae82`）；補充臂 [逐 alpha 表](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-paper-01/single_all/result.md)／[JSON](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-paper-01/single_all/result.json)（SHA-256 `c89422bdc085aa0f2c4005f3e3e5788ecb9fc0b8af33e225070afef6f59a996b`）。兩臂都綁定 paper ranking 來源 SHA `6aea1dada88f4ac58c727b0699878f17bf2e28dc9b40c3b6d379646bdf065cc8` 與 C2 summary SHA `5f5da88738c10f784357fbd26994235400cc1ccd72f46495314748738e1733f7`，完整 prompt/split/model/tokenizer 身分見各 JSON metadata。先通過 [tokenwise ABNB L0/L16 smoke](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-smoke-01/tokenwise/result.json) 與 [single_all ABNB L16 smoke](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-smoke-02/single_all/result.json)，才跑本次兩組完整受測；單向量 ABNB L0 嚴格擬合失敗，沒有生成決策 artifact；[只保留新算出的緊湊建構診斷](../../../artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-paper-01/single_all/l0_fit_diagnostic.json)（SHA-256 `4c9c8658ba1d080f0f7c842ce09b8fe38dca7b3c083ff1cdf96ee0ac0fb72fec`），不含任何原始狀態。

從 repo root 執行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/probe_concept_cone.py --cohort-mode sp500_dim_paper --model .cache/models/qwen3.5-4b --model-dtype bf16 --population-csv data/sp500_constituents_2020_2025.csv --split-seed 20260923 --dim-arm tokenwise --dim-layers 0 14 15 16 17 18 31 --alphas 0 2 3 4 5 6 --output-json artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-paper-01/tokenwise/result.json
CUDA_VISIBLE_DEVICES=1 uv run python scripts/probe_concept_cone.py --cohort-mode sp500_dim_paper --model .cache/models/qwen3.5-4b --model-dtype bf16 --population-csv data/sp500_constituents_2020_2025.csv --split-seed 20260923 --dim-arm single_all --dim-layers 14 15 16 17 18 31 --alphas 0 2 3 4 5 6 --output-json artifacts/qwen3.5-4b/concept-cone-steering/runs/dim-layer-sweep-v1-20260925-paper-01/single_all/result.json
```
