# T02 steering operators

At the T01 injection layer, a one-dimensional DIM direction flips every testable generated decision by
|α| = 8 in all four models. The best-aligned single MLP neuron is unreliable, and no tested cone point gives a
smoother control surface than DIM.

## Setup

- **Evaluation:** 101 held-out S&P 500 companies (seed `20260923`; the other 402 build the operators), the
  balanced prompt (P1, P2, N1, N2), greedy generation up to 192 tokens. A flip is counted on the generated
  decision. The denominator is every company whose unsteered decision is the source class, and an unparsable
  output counts as not flipped (ITT).
- **Injection:** `h ← h + α·B[p]` at the output of the injection layer's block, on the last K instruction
  tokens (the steer suffix), during prefill only.
- **Operators:** all are anchored to the raw DIM difference `d[p]` between the mean states of the ten
  highest-margin and ten lowest-margin construction companies.
  - DIM: `B = d[p]`.
  - Single neuron: the MLP write vector with the largest cosine to `mean_p d̂[p]`, scaled to `‖d[p]‖` at every
    token. Gemma and GLM write vectors include the post-MLP norm gain. GPT-OSS candidates are expert neurons.
  - Cone k ∈ {2, 4}: `‖d[p]‖/√k · (d̂[p] + v_2 + … + v_k)`. The extra axes are the leading eigenvectors of the
    100 top-minus-bottom differences with their DIM component removed, pooled over all tokens and shared
    across them, then orthogonalized per token against `d̂[p]`.
  - 4D cone (Proj): the 4D cone without the `1/√k` factor, so its projection on `d̂` equals DIM's.
  - 4D cone (RandAxes): the 4D cone with random orthogonal extra axes.
- **Dose grid:** shown at |α| ∈ {0.25, 2, 4, 8, 16}. The calibrated grids of the full run reach ±64 (GPT-OSS
  ±32). Qwen, GLM and GPT-OSS have no ±0.25 in that grid, so those cells come from a later supplement run.
- **CIs:** 95% percentile bootstrap over companies, 2,000 resamples.

## Results

Table 1. Injection layer, steer-suffix length and unsteered decisions.

| Model | Injection layer | K | buy | sell |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | L16 | 100 | 0 | 101 |
| GLM-4-9B | L19 | 98 | 101 | 0 |
| Gemma-4-12B | L27 | 100 | 58 | 43 |
| GPT-OSS-20B | L14 | 99 | 44 | 57 |

Qwen has no testable buy→sell direction and GLM no testable sell→buy direction.

Table 2. Flip rate by operator and dose. ‡: parse rate below 90%.

Qwen3.5-4B, sell→buy

| Operator | +0.25 | +2 | +4 | +8 | +16 |
|---|---:|---:|---:|---:|---:|
| Single neuron | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00‡ [0.00, 0.00] |
| DIM | 0.00 [0.00, 0.00] | 0.11 [0.05, 0.17] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 2D cone | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.03] | 0.44 [0.34, 0.52] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 4D cone | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.07] | 0.33 [0.24, 0.42] | 1.00 [1.00, 1.00] |
| 4D cone (Proj) | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.07] | 0.33 [0.24, 0.42] | 1.00 [1.00, 1.00] | 0.00‡ [0.00, 0.00] |
| 4D cone (RandAxes) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.02 [0.00, 0.05] | 0.38 [0.29, 0.47] | 0.91 [0.85, 0.96] |

GLM-4-9B, buy→sell

| Operator | −0.25 | −2 | −4 | −8 | −16 |
|---|---:|---:|---:|---:|---:|
| Single neuron | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.02 [0.00, 0.05] | 0.31 [0.22, 0.41] |
| DIM | 0.00 [0.00, 0.00] | 0.78 [0.70, 0.86] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 2D cone | 0.00 [0.00, 0.00] | 0.26 [0.18, 0.35] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 4D cone | 0.00 [0.00, 0.00] | 0.22 [0.14, 0.30] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 4D cone (Proj) | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 4D cone (RandAxes) | 0.00 [0.00, 0.00] | 0.13 [0.07, 0.20] | 0.95 [0.90, 0.99] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |

Gemma-4-12B, sell→buy

| Operator | +0.25 | +2 | +4 | +8 | +16 |
|---|---:|---:|---:|---:|---:|
| Single neuron | 0.02 [0.00, 0.07] | 0.14 [0.05, 0.26] | 0.30 [0.16, 0.44] | 0.72 [0.58, 0.84] | 0.98 [0.93, 1.00] |
| DIM | 0.49 [0.35, 0.65] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 2D cone | 0.40 [0.26, 0.56] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 4D cone | 0.28 [0.14, 0.42] | 0.95 [0.88, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 4D cone (Proj) | 0.42 [0.28, 0.58] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 4D cone (RandAxes) | 0.47 [0.33, 0.63] | 0.88 [0.79, 0.98] | 0.00‡ [0.00, 0.00] | 0.00‡ [0.00, 0.00] | 0.00‡ [0.00, 0.00] |

Gemma-4-12B, buy→sell

| Operator | −0.25 | −2 | −4 | −8 | −16 |
|---|---:|---:|---:|---:|---:|
| Single neuron | 0.02 [0.00, 0.05] | 0.12 [0.05, 0.21] | 0.33 [0.21, 0.45] | 0.47 [0.34, 0.59] | 0.00 [0.00, 0.00] |
| DIM | 0.43 [0.29, 0.57] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 2D cone | 0.29 [0.17, 0.41] | 0.98 [0.95, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 4D cone | 0.14 [0.05, 0.22] | 0.93 [0.86, 0.98] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.84‡ [0.74, 0.93] |
| 4D cone (Proj) | 0.40 [0.28, 0.53] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.81‡ [0.71, 0.91] | 0.00‡ [0.00, 0.00] |
| 4D cone (RandAxes) | 0.52 [0.38, 0.64] | 0.00 [0.00, 0.00] | 0.00‡ [0.00, 0.00] | 0.00‡ [0.00, 0.00] | 0.00‡ [0.00, 0.00] |

GPT-OSS-20B, sell→buy

| Operator | +0.25 | +2 | +4 | +8 | +16 |
|---|---:|---:|---:|---:|---:|
| Single neuron | 0.11 [0.04, 0.19] | 0.26 [0.16, 0.39] | 0.18 [0.09, 0.28] | 0.26 [0.16, 0.37] | 0.44‡ [0.32, 0.56] |
| DIM | 0.09 [0.02, 0.18] | 0.60 [0.47, 0.72] | 0.91 [0.84, 0.98] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 2D cone | 0.07 [0.02, 0.14] | 0.54 [0.40, 0.67] | 0.81 [0.68, 0.89] | 1.00 [1.00, 1.00] | 0.98 [0.95, 1.00] |
| 4D cone | 0.16 [0.07, 0.26] | 0.75 [0.65, 0.86] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 4D cone (Proj) | 0.26 [0.16, 0.39] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.67‡ [0.54, 0.79] |
| 4D cone (RandAxes) | 0.09 [0.02, 0.16] | 0.25 [0.14, 0.37] | 0.42 [0.28, 0.54] | 0.89 [0.81, 0.96] | 0.98 [0.95, 1.00] |

GPT-OSS-20B, buy→sell

| Operator | −0.25 | −2 | −4 | −8 | −16 |
|---|---:|---:|---:|---:|---:|
| Single neuron | 0.11 [0.02, 0.20] | 0.30 [0.16, 0.43] | 0.41 [0.27, 0.55] | 0.34 [0.20, 0.48] | 0.39 [0.25, 0.52] |
| DIM | 0.11 [0.02, 0.20] | 0.43 [0.30, 0.57] | 0.77 [0.64, 0.89] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| 2D cone | 0.07 [0.00, 0.16] | 0.48 [0.34, 0.64] | 0.80 [0.66, 0.91] | 0.91 [0.82, 0.98] | 0.82 [0.70, 0.93] |
| 4D cone | 0.07 [0.00, 0.14] | 0.70 [0.57, 0.82] | 0.89 [0.77, 0.98] | 0.89 [0.80, 0.98] | 0.11‡ [0.02, 0.23] |
| 4D cone (Proj) | 0.23 [0.11, 0.36] | 0.86 [0.75, 0.95] | 0.89 [0.80, 0.98] | 0.14‡ [0.05, 0.25] | 0.00‡ [0.00, 0.00] |
| 4D cone (RandAxes) | 0.07 [0.00, 0.16] | 0.25 [0.11, 0.39] | 0.52 [0.36, 0.66] | 0.39 [0.25, 0.52] | 0.73 [0.59, 0.86] |

Table 3. Output status of every ‡ cell, as shares of the 101 companies. Flip among parsed counts source-class
companies with a parsable output.

| Model | Operator | α | Parsed | Truncated | Collapsed | Flip among parsed |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | 4D cone (Proj) | +16 | 0.00 | 0.00 | 1.00 | — |
| Qwen3.5-4B | Single neuron | +16 | 0.00 | 0.00 | 1.00 | — |
| Gemma-4-12B | 4D cone | −16 | 0.89 | 0.11 | 0.00 | 49/52 |
| Gemma-4-12B | 4D cone (Proj) | −16 | 0.81 | 0.00 | 0.19 | 0/48 |
| Gemma-4-12B | 4D cone (Proj) | −8 | 0.87 | 0.13 | 0.00 | 47/50 |
| Gemma-4-12B | 4D cone (RandAxes) | −16 | 0.00 | 0.40 | 0.60 | — |
| Gemma-4-12B | 4D cone (RandAxes) | −8 | 0.81 | 0.04 | 0.15 | 0/48 |
| Gemma-4-12B | 4D cone (RandAxes) | −4 | 0.87 | 0.00 | 0.13 | 0/52 |
| Gemma-4-12B | 4D cone (RandAxes) | +4 | 0.00 | 0.61 | 0.39 | — |
| Gemma-4-12B | 4D cone (RandAxes) | +8 | 0.00 | 0.77 | 0.23 | — |
| Gemma-4-12B | 4D cone (RandAxes) | +16 | 0.00 | 0.34 | 0.66 | — |
| GPT-OSS-20B | 4D cone | −16 | 0.19 | 0.34 | 0.48 | 5/11 |
| GPT-OSS-20B | 4D cone (Proj) | −16 | 0.00 | 1.00 | 0.00 | — |
| GPT-OSS-20B | 4D cone (Proj) | −8 | 0.20 | 0.35 | 0.46 | 6/11 |
| GPT-OSS-20B | 4D cone (Proj) | +16 | 0.64 | 0.36 | 0.00 | 38/38 |
| GPT-OSS-20B | Single neuron | +16 | 0.87 | 0.13 | 0.00 | 25/50 |

Table 4. DIM beyond |α| = 16 (flip rate / parse rate). The other doses match Table 2 with parse rate 1.00.

| Model | Testable direction | ±32 | ±64 |
|---|---|---:|---:|
| Qwen3.5-4B | sell→buy | 0.98 / 0.98 | 0.00 / 0.00 |
| GLM-4-9B | buy→sell | 1.00 / 1.00 | 1.00 / 1.00 |
| Gemma-4-12B | sell→buy | 1.00 / 1.00 | 1.00 / 1.00 |
| Gemma-4-12B | buy→sell | 1.00 / 1.00 | 0.00 / 1.00 |
| GPT-OSS-20B | sell→buy | 0.60 / 0.66 | not run |
| GPT-OSS-20B | buy→sell | 0.02 / 0.02 | not run |

Table 5. Operator geometry and the pre-registered C8 smoothness rule. C8 holds for a sign when the 4D cone's
share of monotone margin steps is above RandAxes' and not below DIM's, with no more reversals than either.

| Model | Selected neuron of candidates (cos to `mean_p d̂[p]`, runner-up) | Median over tokens of cos(B[p], d̂[p]): neuron / cone2 / cone4 / Proj / RandAxes | C8 holds (+ / −) |
|---|---|---|---|
| Qwen3.5-4B | 3168 of 9216 (0.083, 0.082) | 0.020 / 0.707 / 0.500 / 0.500 / 0.500 | no / no |
| GLM-4-9B | 6618 of 13696 (0.243, 0.191) | 0.091 / 0.707 / 0.500 / 0.500 / 0.500 | no / no |
| Gemma-4-12B | 12185 of 15360 (0.259, 0.237) | 0.065 / 0.707 / 0.500 / 0.500 / 0.500 | yes / no |
| GPT-OSS-20B | expert 22, neuron 1617 of 92160 (0.144, 0.109) | 0.034 / 0.707 / 0.500 / 0.500 / 0.500 | yes / no |

Gates pass in all four models: a repeated unsteered generation is identical, a zero-displacement hook and an
injection at the final layer leave the output unchanged, and the final-layer structural check passes.

## Comparisons

- **DIM:** reaches 1.00 in every testable direction by |α| = 2 (Gemma), 4 (Qwen, GLM) or 8 (GPT-OSS), with every
  output parsable up to |α| = 16. Gemma is the most sensitive: 0.49 sell→buy and 0.43 buy→sell at |α| = 0.25.
- **Single neuron:** never flips Qwen, stays at or below 0.44 in GLM and GPT-OSS, and is non-monotone in Gemma
  buy→sell (0.47 at −8, 0.00 at −16). The selected neuron is barely aligned with DIM: the median per-token cosine
  is 0.020–0.091, and in Qwen the runner-up is within 0.002 of it.
- **Cones:** the dose–response curves shift right of DIM's rather than flatten. In Qwen the 4D cone at α matches
  Proj at α/2 (0.33 at +8 and +4), which the `1/√k` factor explains. At matched projection, Proj is weaker than
  DIM in Qwen (0.33 vs 1.00 at +4) and stronger in GLM (1.00 vs 0.78 at −2). The 4D cone is at or above
  RandAxes at every dose only in GLM and GPT-OSS sell→buy, and the gap is clear only in GPT-OSS sell→buy
  (0.75 vs 0.25 at +2, 1.00 vs 0.42 at +4).
- **C8:** the pre-registered rule fails in all four models. It holds for sell→buy only, in Gemma and GPT-OSS.
- **Output format:** of the 16 ‡ cells, three are lost steering with outputs mostly parsable (Gemma Proj −16,
  RandAxes −4 and −8: 0 of 48–52 parsed buy companies flip). The rest are format loss.

## Limitations

- One prompt template, four synthetic facts and a forced buy/sell choice.
- The ±0.25 cells of Qwen, GLM and GPT-OSS come from a supplement run whose doses were chosen after the full
  results were known.
- Beyond |α| = 16, DIM is not monotone and not always parsable (Table 4).
- The single neuron is chosen by one rule (alignment with DIM) and steers by adding its write vector, not by
  editing its activation. Other neurons or rules are not ruled out.
- Only one equal-weight point of each cone is tested, with k ∈ {2, 4}.
