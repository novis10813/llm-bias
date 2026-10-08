# T03 control limits

DIM steering is not an artifact of the readout, of a generic perturbation or of firm recognition, but it acts as
a prior rather than a switch. It beats matched-norm random directions at moderate and large doses and works on
invented companies. Against opposing evidence, no company flips up to |α| = 16 in GLM, Gemma or GPT-OSS. Qwen
needs 4 times its balanced dose, and GLM flips at 8 to 16 times its balanced dose. Fixed-prefix margins
misreport steering at transition doses.

## Setup

- **Steering:** DIM at the T01 injection layer, as in T02, on the 101 evaluation companies. Flip rates are ITT
  on generated decisions, with 95% company bootstrap CIs (2,000 resamples).
- **Readout check:** among parsed outputs, the share whose fixed-prefix margin
  (`log p(buy) − log p(sell)` after `{"decision": "`) has the opposite sign of the generated decision.
- **Random control:** five random unit directions, scaled to `‖d[p]‖` at every token. Rand is the largest
  share, over the five seeds, of all companies whose decision changes in either direction. Gain = DIM − Rand.
  The two rates have different denominators (source class for DIM, all companies for Rand).
- **Opposing evidence:** sell→buy is steered on a prompt with only N1, N2, and buy→sell on a prompt with only
  P1, P2. The direction is still the balanced-prompt DIM.
- **Anonymous identities:** ten invented ticker and name pairs, checked against every constituent.
- **Supplement:** the Qwen, GLM and GPT-OSS ±0.25 cells and most Random cells come from the 2026-10-06
  supplement run. Gemma's supplement Random cells (†) are scaled to a DIM re-estimated on newer GPU hardware,
  whose median norm is about 14% below the original.

## Results

Table 1. DIM validation by dose. Rand: largest share, over five random seeds, of companies whose decision changes in either direction. †: random rescaled to a re-estimated DIM.

| Model | α | DIM | Readout disagreement | Rand | Gain | Opposing | Anonymous |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | +0.25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Qwen3.5-4B | +2 | 0.11 [0.05, 0.17] | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | 0.11 [0.05, 0.17] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| Qwen3.5-4B | +4 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| Qwen3.5-4B | +8 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 0.88 [0.81, 0.94] | 1.00 [1.00, 1.00] |
| Qwen3.5-4B | +16 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.07 [0.03, 0.12] | 0.93 [0.88, 0.97] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| GLM-4-9B | -0.25 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| GLM-4-9B | -2 | 0.78 [0.70, 0.86] | 0.48 [0.38, 0.57] | 0.00 [0.00, 0.00] | 0.78 [0.70, 0.86] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| GLM-4-9B | -4 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.03] | 0.99 [0.97, 1.00] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| GLM-4-9B | -8 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.20 [0.12, 0.29] | 0.80 [0.71, 0.88] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| GLM-4-9B | -16 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.19 [0.11, 0.27] | 0.81 [0.73, 0.89] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| Gemma-4-12B | +0.25 | 0.49 [0.35, 0.65] | 0.04 [0.01, 0.08] | 0.23 [0.19, 0.32] | 0.26 [0.09, 0.38] | 0.00 [0.00, 0.00] | 0.40 [0.10, 0.70] |
| Gemma-4-12B | +2 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.43† [0.34, 0.52] | 0.57 [0.48, 0.66] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| Gemma-4-12B | +4 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.43† [0.34, 0.52] | 0.57 [0.48, 0.66] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| Gemma-4-12B | +8 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.05† [0.02, 0.10] | 0.95 [0.90, 0.98] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| Gemma-4-12B | +16 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.07† [0.03, 0.13] | 0.93 [0.87, 0.97] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| Gemma-4-12B | -0.25 | 0.43 [0.29, 0.57] | 0.06 [0.02, 0.11] | 0.35 [0.26, 0.44] | 0.08 [-0.03, 0.20] | 0.00 [0.00, 0.00] | — |
| Gemma-4-12B | -2 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.43† [0.34, 0.52] | 0.57 [0.48, 0.66] | 0.00 [0.00, 0.00] | — |
| Gemma-4-12B | -4 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.39† [0.30, 0.49] | 0.61 [0.51, 0.70] | 0.00 [0.00, 0.00] | — |
| Gemma-4-12B | -8 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.08† [0.03, 0.14] | 0.92 [0.86, 0.97] | 0.00 [0.00, 0.00] | — |
| Gemma-4-12B | -16 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.30† [0.21, 0.39] | 0.70 [0.61, 0.79] | 0.00 [0.00, 0.00] | — |
| GPT-OSS-20B | +0.25 | 0.09 [0.02, 0.18] | 0.45 [0.35, 0.53] | 0.09 [0.06, 0.16] | -0.00 [-0.09, 0.06] | 0.00 [0.00, 0.00] | 0.11 [0.00, 0.33] |
| GPT-OSS-20B | +2 | 0.60 [0.47, 0.72] | 0.68 [0.59, 0.77] | 0.27 [0.22, 0.37] | 0.33 [0.17, 0.44] | 0.00 [0.00, 0.00] | 0.11 [0.00, 0.33] |
| GPT-OSS-20B | +4 | 0.91 [0.84, 0.98] | 0.04 [0.01, 0.08] | 0.44 [0.38, 0.53] | 0.48 [0.35, 0.55] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| GPT-OSS-20B | +8 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.49 [0.46, 0.60] | 0.51 [0.40, 0.54] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| GPT-OSS-20B | +16 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.54 [0.49, 0.65] | 0.46 [0.35, 0.51] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| GPT-OSS-20B | -0.25 | 0.11 [0.02, 0.20] | 0.48 [0.38, 0.57] | 0.13 [0.08, 0.20] | -0.02 [-0.11, 0.07] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| GPT-OSS-20B | -2 | 0.43 [0.30, 0.57] | 0.32 [0.23, 0.41] | 0.32 [0.27, 0.42] | 0.11 [-0.07, 0.26] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| GPT-OSS-20B | -4 | 0.77 [0.64, 0.89] | 0.15 [0.09, 0.23] | 0.41 [0.37, 0.50] | 0.37 [0.19, 0.48] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| GPT-OSS-20B | -8 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.59 [0.52, 0.69] | 0.41 [0.31, 0.48] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| GPT-OSS-20B | -16 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.56 [0.49, 0.66] | 0.44 [0.34, 0.51] | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |

Table 2. Random directions at |α| ≥ 8: parse rate of the seed that sets Rand, and seeds parsing ≥ 90%.

| Model | α | DIM parse | Rand | Parse of that seed | Seeds ≥ 0.90 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | +8 | 1.00 | 0.00 | 1.00 | 5/5 |
| Qwen3.5-4B | +16 | 1.00 | 0.07 | 1.00 | 5/5 |
| GLM-4-9B | -8 | 1.00 | 0.20 | 1.00 | 5/5 |
| GLM-4-9B | -16 | 1.00 | 0.19 | 1.00 | 5/5 |
| Gemma-4-12B | +8 | 1.00 | 0.05 | 0.11 | 0/5 |
| Gemma-4-12B | +16 | 1.00 | 0.07 | 0.30 | 0/5 |
| Gemma-4-12B | -8 | 1.00 | 0.08 | 0.73 | 0/5 |
| Gemma-4-12B | -16 | 1.00 | 0.30 | 0.81 | 0/5 |
| GPT-OSS-20B | +8 | 1.00 | 0.49 | 1.00 | 5/5 |
| GPT-OSS-20B | +16 | 1.00 | 0.54 | 1.00 | 4/5 |
| GPT-OSS-20B | -8 | 1.00 | 0.59 | 1.00 | 5/5 |
| GPT-OSS-20B | -16 | 1.00 | 0.56 | 1.00 | 4/5 |

Table 3. Opposing evidence over the full calibrated grid (flip rate / parse rate). Sell→buy uses only N1, N2; buy→sell uses only P1, P2.

- Qwen3.5-4B, sell→buy: +0.25: 0.00 / 1.00; +1: 0.00 / 1.00; +2: 0.00 / 1.00; +4: 0.00 / 1.00; +8: 0.88 / 1.00; +16: 1.00 / 1.00; +32: 0.00 / 0.00; +64: 0.00 / 0.00
- GLM-4-9B, buy→sell: -0.25: 0.00 / 1.00; -1: 0.00 / 1.00; -2: 0.00 / 1.00; -4: 0.00 / 1.00; -8: 0.00 / 1.00; -16: 0.00 / 1.00; -32: 0.74 / 1.00; -64: 1.00 / 1.00
- Gemma-4-12B, sell→buy: +0.25: 0.00 / 1.00; +2: 0.00 / 1.00; +4: 0.00 / 1.00; +8: 0.00 / 1.00; +16: 0.00 / 1.00; +32: 0.00 / 1.00; +64: 0.00 / 1.00
- Gemma-4-12B, buy→sell: -0.25: 0.00 / 1.00; -2: 0.00 / 1.00; -4: 0.00 / 1.00; -8: 0.00 / 1.00; -16: 0.00 / 1.00; -32: 0.00 / 1.00; -64: 0.00 / 1.00
- GPT-OSS-20B, sell→buy: +0.125: 0.00 / 1.00; +0.25: 0.00 / 1.00; +0.5: 0.00 / 1.00; +2: 0.00 / 1.00; +4: 0.00 / 1.00; +8: 0.00 / 1.00; +16: 0.00 / 1.00; +32: 0.00 / 0.72
- GPT-OSS-20B, buy→sell: -0.125: 0.00 / 1.00; -0.25: 0.00 / 1.00; -0.5: 0.00 / 1.00; -2: 0.00 / 1.00; -4: 0.00 / 1.00; -8: 0.00 / 1.00; -16: 0.00 / 1.00; -32: 0.00 / 0.02

Table 4. Pre-registered flip-dose contrasts (C7): companies whose unsteered decision is the source class under both conditions.

| Model | Contrast | Comparable | Higher dose | Equal | Lower | Blocked | Collapsed |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B | neg vs balanced (sell->buy) | 101 | 101 | 0 | 0 | 0 | 0 |
| Qwen3.5-4B | neg vs mixed2 (sell->buy) | 98 | 98 | 0 | 0 | 0 | 0 |
| Qwen3.5-4B | pos vs mixed2 (buy->sell) | 3 | 3 | 0 | 0 | 0 | 0 |
| GLM-4-9B | pos vs balanced (buy->sell) | 101 | 101 | 0 | 0 | 0 | 0 |
| GLM-4-9B | pos vs mixed2 (buy->sell) | 101 | 33 | 20 | 48 | 0 | 0 |
| Gemma-4-12B | neg vs balanced (sell->buy) | 43 | 0 | 0 | 0 | 43 | 0 |
| Gemma-4-12B | pos vs balanced (buy->sell) | 58 | 0 | 0 | 0 | 58 | 0 |
| Gemma-4-12B | pos vs mixed2 (buy->sell) | 101 | 0 | 0 | 0 | 101 | 0 |
| GPT-OSS-20B | neg vs balanced (sell->buy) | 57 | 0 | 0 | 0 | 45 | 12 |
| GPT-OSS-20B | neg vs mixed2 (sell->buy) | 91 | 0 | 0 | 0 | 67 | 24 |
| GPT-OSS-20B | pos vs balanced (buy->sell) | 44 | 0 | 0 | 0 | 1 | 43 |
| GPT-OSS-20B | pos vs mixed2 (buy->sell) | 10 | 0 | 0 | 0 | 0 | 10 |

Table 5. Anonymous identities: unsteered decisions on the balanced prompt (n = 10).

| Model | buy | sell |
|---|---:|---:|
| Qwen3.5-4B | 0 | 10 |
| GLM-4-9B | 10 | 0 |
| Gemma-4-12B | 0 | 10 |
| GPT-OSS-20B | 1 | 9 |

Table 6. Pre-registered C5 rule at ±α_50 and ±α_hi, and fixed-prefix readout disagreement over all parsed rows.

| Model | C5 points (α: Gain lower bound) | C5 holds | Readout disagreement, C4 arms | All generated rows |
|---|---|---|---:|---:|
| Qwen3.5-4B | +4: 1.000; +32: 0.950 | yes | 158/11757 (0.013) | 159/13250 (0.012) |
| GLM-4-9B | -8: 0.713; -64: 1.000 | yes | 296/13911 (0.021) | 324/16079 (0.020) |
| Gemma-4-12B | +0.25: 0.091; -0.25: -0.026; +64: 1.000; -64: 0.000 | no | 202/11325 (0.018) | 298/13785 (0.022) |
| GPT-OSS-20B | +4: 0.350; -4: 0.189; +16: 0.347; -16: 0.337 | yes | 2841/12519 (0.227) | 3578/14845 (0.241) |

All models, C4 arms (alpha0, dim, ops, evidence): 3497/49512 (0.071) parsed rows disagree with their fixed-prefix margin.

All models, all generated rows: 4359/57959 (0.075) parsed rows disagree with their fixed-prefix margin.

## Comparisons

- **Readout:** disagreement is at most 0.01 in Qwen and 0.06 in Gemma and is 0 for |α| ≥ 8 in every model. At
  transition doses it is large: 0.48 in GLM at −2 (where DIM flips 0.78) and 0.68 in GPT-OSS at +2. Over all
  parsed rows, 22.7% of GPT-OSS outputs disagree with their margin, against 1.3–2.1% in the other models and
  7.1% overall.
- **Random:** random directions flip at most 0.20 of companies in Qwen and GLM. In Gemma and GPT-OSS they flip
  up to 0.43 and 0.59, and DIM still exceeds them at moderate and large doses (Gain 0.57 in Gemma at ±2, 0.51
  in GPT-OSS at +8). At the smallest doses in Gemma buy→sell and GPT-OSS, the Gain interval includes zero.
  The pre-registered C5 rule holds in Qwen, GLM and GPT-OSS and fails in Gemma (lower bound −0.026 at −0.25).
- **Random at large doses:** in Gemma at |α| ≥ 8 no random seed parses 90% of outputs, and the seed that sets
  Rand parses 0.11–0.81. Unparsed outputs count as not flipped, so Rand is biased down and Gain up there. In
  GPT-OSS the seed that sets Rand parses fully.
- **Opposing evidence:** in GLM, Gemma and GPT-OSS no dose up to |α| = 16 flips any company, while the same
  doses flip every company on the balanced prompt. Beyond that, GLM flips 0.74 at −32 and 1.00 at −64, Gemma
  never flips up to ±64 with every output parsed, and GPT-OSS outputs break at ±32 (parse 0.72 and 0.02). Qwen
  needs +16 for full flipping, against +4 on the balanced prompt.
- **Flip-dose contrasts:** every comparable Qwen and GLM company needs a higher dose under opposing evidence
  than on the balanced prompt. Gemma's comparable companies are all blocked (parsed but never flipped). In
  GPT-OSS, buy→sell under P1, P2 is mostly collapsed (43 of 44), so it is format loss rather than evidence
  resisting the steering.
- **Anonymous identities:** in every direction with at least nine source-class identities, DIM reaches 1.00 no
  later than the dose at which real companies saturate. Gemma has no buy identity, so its buy→sell rate is undefined. GPT-OSS
  buy→sell rests on one identity. Near threshold, GPT-OSS sell→buy moves anonymous identities less readily
  (0.11 vs 0.60 at +2).

## Limitations

- The opposing-evidence test uses two facts in one direction per model, and the grid shown in the paper stops
  at |α| = 16.
- Ten anonymous identities give wide intervals.
- Rand is the maximum over five seeds of an any-direction flip rate. Unparsed random outputs count as not
  flipped, so Gain is an upper bound wherever random directions break the format.
- Gemma's Random cells from the supplement are scaled about 14% below the original DIM norm.
- GPT-OSS evidence rows are not fully reproducible: regenerating one stored row at −32 changes its decision.
