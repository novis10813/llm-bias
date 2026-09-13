# Research program and planning

This directory defines the cross-experiment questions, expected contributions and completion requirements. Experiment protocols, gates, run results and artifact contracts remain in their owning experiment directories.

## This round targets behaviour and mechanisms; generalisation is still required

- [Behavioural Confirmation and Mechanistic Analysis of Entity Influence](entity-bias-research-proposal.md): revised objectives, RQ1–RQ5, evidence roles and expected contributions. Selective inference-time control is an independent follow-up, not a required success outcome.
- [Research roadmap](entity-bias-roadmap.md): original M1–M6 versus the actual research path, current evidence and remaining work. Cross-model **and** cross-task validation remain required and incomplete.
- [Research directory and experiment relationships](../README.md): all experiment entrances and upstream/downstream relationships. Each experiment keeps `proposal.md` and `report.md` at the top level; `details/` retains phase protocols and intermediate records.

This revision does not mark the original 8-K milestones as completed, promote discovery to confirmation, or rewrite frozen protocols. Closed Qwen3.5-4B lines do not imply that the overall program is complete. The original NAACL 2027 schedule remains historical planning context, not a verified deadline.

## Evidence is organised by the question it answers

| Question | Canonical sources |
|---|---|
| Inputs, instrument provenance and behavioural sensitivity | [Baseline trial](../baseline-trial/proposal.md), [lens selection](../jacobian-lens-selection/proposal.md), [span sensitivity](../span-sensitivity/proposal.md) |
| Evidence representations, steering and context-state effects | [Valence readout](../jspace-valence-readout/proposal.md), [sector intervention](../jspace-sector-intervention/proposal.md), [J-space token V1/V2](../jspace-token-experiments/proposal.md), [activation patching](../activation-patching-causal-tracing/proposal.md), [sector/context follow-up](../sector-context-followup/proposal.md) |
| Factual memory and financial-judgment probes | [Entity Cell](../entity-cell-localization/proposal.md), [financial-soundness localisation](../financial-soundness-localization/proposal.md), [causal validation](../financial-soundness-causal-validation/proposal.md) |
| Global stance, entity gaps and their causal paths | [Investment-dial](../investment-dial/proposal.md), [Balanced Evidence Gap](../balanced-evidence-gap/proposal.md), [Entity-to-Dial](../entity-to-dial/proposal.md) |

Cross-model/task protocols still need to be defined. [J-space evaluation](../j-space-evaluation/proposal.md) remains optional, proposed and non-runnable; it does not gate their execution. Archived 8-K/counterfactual workflows remain under the [archive documentation](../archive/README.md), not the active execution path.

## Operational contracts stay outside the research plan

See [shared core](../shared-experiment-core.md), [artifact identity](../artifact-contract.md), [research scripts](../research-scripts.md), [interactive dashboard](../interactive-prompt-lens-dashboard.md), [documentation rules](../documentation-system.md) and [repository guidance](../../AGENTS.md). The program documents do not duplicate CLI commands, schemas or per-run numerical tables.
