# Selective-intervention 研究線

**工作流名稱**：Entity-specific selective intervention（roadmap M5 工作流）  
**狀態**：V1 completed（formal gate `fail`：efficacy＋specificity 陽性、
full-strength 全局副作用；見 [report.md](report.md)）

本線承接原始 research proposal §4.4 候選形式 #3（low-rank
projection/subtraction from an entity-difference subspace）與原始 roadmap M5。
目前 [research program](../proposal/entity-bias-research-proposal.md) 與
[roadmap](../proposal/entity-bias-roadmap.md#selective-intervention-remains-a-separate-research-line)
將本線列為獨立後續研究，不作本輪完成條件：
以 entity-to-dial 線收線時確定的 L15 8 維 entity-difference subspace 為介入方向，
在推論期移除該子空間分量，評估 entity-induced decision gap 的縮減、specificity 與
task preservation。

## 版本索引

| 版本 | 狀態 | 協議 | 報告 |
|---|---|---|---|
| V1 | completed（Rev 1.5） | [proposal-v1.md](details/proposal-v1.md)（L15 k=8 subspace removal，16 公司 frozen balanced-evidence population） | [report.md](report.md)（gate fail：G1a/G1b/G2 pass、G3/G4 fail；full-strength 負結果） |

## 上游依賴（皆已收線）

- [Balanced Evidence Gap](../balanced-evidence-gap/report.md)：16 公司
  balanced-evidence gap 行為確認；2A population（64 prompts）與 group gap 1.028 nats。
- [Entity-to-Dial](../entity-to-dial/report.md)：L15 指令區間 state
  difference 的 k=8 子空間恢復 full-swap 效應 98.3%（run `entity-to-dial-e-01`）。
- [Investment-Dial](../investment-dial/report.md)：L15/n8490 dial 坐標
  與 ±4 native-unit push 機件（`mlp_addition`）。

## 後續研究與總覽

V1 full-strength 結果依 frozen 決策表為負結果；若改 direction source、controls
或 gates，須另立版本，本入口不授權新 run。研究前後關係見 [研究總覽](../README.md)。
