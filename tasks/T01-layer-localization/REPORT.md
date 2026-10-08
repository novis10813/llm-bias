# T01 layer localization

The evidence-driven buy/sell margin gap is carried most strongly by the instruction span at relative depth
0.49–0.61 in all four models. Those layers are the injection layers for T02 and T03. On the steering prompt
itself, patching the steer suffix peaks within one layer of the injection layer in Qwen, GLM and Gemma, but not
in GPT-OSS.

## Setup

- **Models:** Qwen3.5-4B (32 layers), GLM-4-9B (40), Gemma-4-12B (48), GPT-OSS-20B (24, MoE).
- **Stage 1, evidence-flip patching:** 427 companies. Each company gets two prompts, `pos` (P1, P2) and `neg`
  (N1, N2), which gives 854 directions (pos→neg and neg→pos). At every layer, the residual states of one span
  (entity, evidence, instruction, final) in the target prompt are replaced with the source prompt's states.
  The readout is the fixed-prefix margin `log p(buy) − log p(sell)`. Transfer is
  `T = (m_patched − m_target) / (m_source − m_target)`, averaged over directions. The instruction-span peak sets
  the injection layer.
- **Overlap check:** 89 of the 427 companies are in the 101-company evaluation set. Stage 1 is recomputed on the
  676 directions of the other 338 companies.
- **Steering-prompt patching:** construction companies only. The 20 highest-margin and 20 lowest-margin
  companies are paired by rank in both directions, giving 40 directions on the balanced prompt. Six spans are
  patched at every layer (entity, evidence, instruction, final, answer prefix, steer suffix). The readout is the
  margin at the decision token with the target's own unsteered generation teacher-forced. The band is every
  layer with `T ≥ 0.7 × peak` on the steer-suffix curve.
- **Patch under generation:** at the steer-suffix peak and ±2 layers, the steer suffix is patched during
  prefill only, and the greedy decision is compared with the source company's unsteered decision. Only pairs
  whose unsteered decisions differ can show a flip.

## Results

Table 1. Stage 1 instruction-span peak.

| Model | Layers | Peak | Depth | T | Peak and T without evaluation companies |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 32 | L16 | 0.52 | 0.408 | L16, 0.407 |
| GLM-4-9B | 40 | L19 | 0.49 | 0.459 | L19, 0.460 |
| Gemma-4-12B | 48 | L27 | 0.57 | 0.190 | L27, 0.190 |
| GPT-OSS-20B | 24 | L14 | 0.61 | 0.532 | L14, 0.532 |

Removing the evaluation companies changes no peak and moves T by at most 0.001.

Table 2. Steering-prompt patching and patch under generation.

| Model | Injection layer | Steer-suffix peak (T) | 70% band | Injection in band | Generated flips at peak | Entity peak (depth) |
|---|---:|---:|---|---|---:|---:|
| Qwen3.5-4B | L16 | L15 (0.458) | L14–17 | yes | n/a (0 comparable pairs) | L4 (0.13) |
| GLM-4-9B | L19 | L20 (0.564) | L17–21 | yes | n/a (0 comparable pairs) | L0 (0.00) |
| Gemma-4-12B | L27 | L27 (0.383) | L26–29 | yes | 13/40 | L12 (0.26) |
| GPT-OSS-20B | L14 | L8 (0.103) | L1, L8 | no | 7/28 | L3 (0.13) |

Self-patches change no margin at any layer or span (maximum absolute difference 0) and leave all ten checked
generations unchanged.

## Comparisons

- Qwen, GLM and Gemma: the steering-prompt band contains the Stage 1 injection layer, so both analyses point
  to the same layers.
- GPT-OSS: the steer-suffix curve is weak (peak T 0.103), and its band is layers 1 and 8 only. It does not
  contain the injection layer L14.
- Qwen and GLM give the same unsteered decision for all 40 construction companies, so patch under generation
  has no comparable pairs. This is missing evidence, not a zero effect.
- Entity-span peaks come earlier than the steer-suffix peaks in every model (depth 0.00–0.26), but at no common
  layer.

## Limitations

- Stage 1 was a development run. Its protocol and layer-selection rule were committed after the runs, and
  Gemma and GLM continued past a failed Phase 2A gate by override.
- GPT-OSS's injection layer was selected under `medium` reasoning, while every steering run uses `low`.
- Stage 1 uses a same-company evidence flip, and the steering-prompt patching uses cross-company pairs. They
  measure different things and are not merged into one curve.
- These are model-specific measures of where patching moves the readout. They do not show that the layers are
  a common site across models or the origin of entity bias.
