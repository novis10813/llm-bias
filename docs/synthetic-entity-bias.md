# Synthetic entity-bias pilot contract

`synthetic-entity-bias validate|run` is the reproducible Qwen entity-bias pilot. It accepts only the explicit S&P 500, Russell 1000, and Russell 2000 constituent CSV paths; `all_constituents_2020_2025.csv` is their derived concatenation and must not be supplied as a fourth input. Input schema is `index_name,year,ticker,company_name,gics_sector` (the index may also be inferred from the explicit filename). Rows are restricted to 2020–2025, deduplicated by complete source-row identity, then aggregated by normalized ticker (`trim`, uppercase, `.` to `-`). The current three inputs produce 3,045 entities and 9,138 preflight prompts. The pool preserves `years`, `memberships`, `membership_years`, `sectors`, source row count, highest tier, and anomaly flags. Source files may contain current snapshots copied across years; preserved years are source provenance, not independently verified historical membership evidence. Input SHA-256 and source row counts belong in run provenance.

## Immutable protocol

The baseline is exactly `The company`. Templates are exactly the three constants in `llm_bias.synthetic_entity_bias.spec`; each rendered user content is sentence, blank line, then the fixed scoring instruction. Qwen chat formatting uses one user turn, generation prompt, and `enable_thinking=False`. Labels `0` through `8` are nine single-token continuations and map to scores `-4` through `+4`. Preflight is untruncated, verifies exact continuation prefix, one suffix token per label, shared IDs across every entity and baseline prompt, exact entity offsets, contiguous span, and entity-before-answer boundaries. Any mismatch fails before model forward.

The final readout is a restricted nine-logit softmax only. Every row retains nine entity and baseline probabilities, expected score, entropy in nats, effective temperature, and signed `delta_expected_score = E_entity - E_baseline`. Effective temperature is the reciprocal norm of the final-normalized transported answer-position residual.

## Localization

For every model layer and template, localization uses answer-position entity-minus-baseline residual. Non-final layers are transported through the complete canonical Jacobian lens in float32; the final layer is direct. Ticker-level deterministic stable-hash split is tier-stratified 80/20, and all templates for a ticker share a split. Train high/low groups are signed final-layer delta-E q75/q25; the normalized high-minus-low direction is fit online. Eval reports mean cosine, Pearson r, Spearman rho, intercept linear R², counts, quantiles, and machine-readable degeneracy flags. Constant targets, insufficient groups, near-zero directions, and non-finite values are errors or explicit flags, never silent zeroes.

## Artifacts and lifecycle

Canonical root is `artifacts/qwen3.5-4b/synthetic-entity-bias-2020-2025/runs/<run-id>/` (or the supplied artifact root/model/dataset). Required files and schemas are:

- `manifest.json`: lifecycle status, stages, input/lens/output references, hashes, and counts.
- `config.json`: exact model/tokenizer/lens identity, input hashes, seed/split, immutable hashes, and score mapping.
- `entity_pool.csv`: one row per unique ticker with pool provenance and anomaly columns.
- `tokenization_validation.json`: label IDs, decoded text, prompt count, and anomalies.
- `no_entity_baselines.csv`: exactly 3 rows (`template,entity,probabilities,expected_score,entropy_nats,effective_temperature`).
- `raw_entity_template_results.csv`: exactly `pool_count × 3` rows with entity/baseline nine-point distributions, scalars, delta-E, split/tier, span, and answer position.
- `layer_template_localization.csv`: one row per layer/template with the statistics above.
- `README.md`: exact model/tokenizer/lens provenance, hashes, labels, templates, pool limitations, definitions, command, and limitations.

Probabilities must be finite, non-negative, and sum to one; entropy is non-negative and temperature positive. Writers use explicit allowlists and reject tensor/ndarray values and keys containing activation, residual, hidden-state, or gradient. No per-example residual, hidden state, activation, or gradient artifact is permitted. The manifest moves only created→running→complete/failed; all stages and post-count checks must pass before complete, and every exception marks a non-terminal run failed.

Example Qwen3.5-4B command (the shell option is intentional):

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 uv run synthetic-entity-bias run --constituents data/sp500_constituents_2020_2025.csv --constituents data/russell1000_constituents_2020_2025.csv --constituents data/russell2000_constituents_2020_2025.csv --model .cache/models/qwen3.5-4b --lens artifacts/qwen3.5-4b/jacobian-lens/jacobian_lens.pt --artifact-root artifacts --dataset synthetic-entity-bias-2020-2025 --run-id pretrained-wikitext-4b --batch-size 16 2>&1 | tee artifacts/synthetic-entity-bias-qwen3.5-4b.log
```

Qwen3.6-27B follows the same protocol with its model-specific canonical lens. Run the models sequentially. Try one 96 GB GPU first with `--batch-size 1`; if that produces CUDA OOM, use the explicit GPU-only two-device map rather than automatic CPU/disk offload or quantization.

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 uv run synthetic-entity-bias run --constituents data/sp500_constituents_2020_2025.csv --constituents data/russell1000_constituents_2020_2025.csv --constituents data/russell2000_constituents_2020_2025.csv --model .cache/models/qwen3.6-27b --lens artifacts/qwen3.6-27b/jacobian-lens/jacobian_lens.pt --artifact-root artifacts --dataset synthetic-entity-bias-2020-2025 --run-id pretrained-wikitext-27b --batch-size 1 2>&1 | tee artifacts/synthetic-entity-bias-qwen3.6-27b.log
```

The pinned Neuronpedia lenses were calibrated on Salesforce/WikiText. Treat lens source as an experimental condition; these artifacts are not the local bilingual candidate-selection winner.

## Artifact-only visualization

A completed run can be summarized and visualized without loading the model, tokenizer, or Jacobian lens:

```bash
uv run synthetic-entity-bias visualize \
  --run-root artifacts/qwen3.5-4b/synthetic-entity-bias-2020-2025/runs/pretrained-wikitext-20260812
```

The visualizer accepts only a source run whose schema-version-1 manifest and all four stages are complete. Before writing output it verifies the required output references, relative paths, SHA-256 digests, CSV schemas, record counts, immutable template/label hashes, ticker/template/layer coverage, probability normalization, numeric domains, and cross-file entity/baseline identity. It rejects failed or partial runs, stale hashes, mixed artifacts, malformed distributions, and incomplete grids. The source `manifest.json` is never modified. The derived visualization bundle currently uses its own schema version 3.

The default output is `<run-root>/visualization/`; an existing non-empty bundle is refused unless `--replace-existing` is explicit. `--output-dir` may select another destination. The default is paper-first and contains:

```text
visualization/
├── visualization_metadata.json
├── figures/
│   ├── entity_effect_distribution.{png,svg,pdf}
│   ├── entity_effect_tail_diagnostics.{png,svg,pdf}
│   ├── baseline_entity_movement.{png,svg,pdf}
│   ├── temperature_null_diagnostics.{png,svg,pdf}
│   ├── entity_effect_by_tier.{png,svg,pdf}
│   ├── template_relationships.{png,svg,pdf}
│   ├── localization_profiles.{png,svg,pdf}
│   ├── sector_effects.{png,svg,pdf}
│   ├── entity_halo_vs_sensitivity.{png,svg,pdf}
│   ├── tier_sector_sentiment_spread.{png,svg,pdf}
│   └── layer_localization_ribbon.{png,svg,pdf}
├── captions/
│   └── <one publication caption per figure>.md
└── tables/
    ├── template_summary.csv
    ├── entity_effect_tail_diagnostics.csv
    ├── baseline_entity_movement.csv
    ├── temperature_null_diagnostics.csv
    ├── localization_transition_diagnostics.csv
    ├── familiarity_tier_summary.csv
    ├── sector_summary.csv
    ├── ticker_template_effects.csv
    ├── localization_summary.csv
    ├── ticker_halo_sensitivity.csv
    ├── tier_sector_sentiment_summary.csv
    └── layer_sentiment_ribbon.csv
```

Each static figure is designed to remain interpretable outside the repository: it includes a publication title, metric definition, sample size, panel labels, reference annotations, complete legend, and concise figure note. PNG is a high-resolution raster export, while SVG and PDF preserve vector marks and text. Each deterministic Markdown caption identifies its figure stem and supporting table and includes the relevant interpretation limits.

The metadata records the source run identity, manifest and artifact hashes/counts, validation checks, aggregation definitions, paper formats/layout, caption linkage, lens condition, output hashes, and the explicit fact that no model loading occurred. Sector summaries explode the pipe-delimited source sector memberships; missing sectors are reported as `Unknown`, and the plot applies a documented minimum group count. Localization plots use normalized layer depth so runs with different layer counts remain visually comparable, but this command is a single-run report: it does not perform a statistical 4B-versus-27B comparison.

An interactive dashboard is optional:

```bash
uv run synthetic-entity-bias visualize \
  --run-root <completed-run> \
  --with-dashboard
```

The auxiliary dashboard renders inline SVG without a CDN or server dependency and links to the paper figures in `figures/`. Embedded ticker/company values are serialized as script-safe JSON and inserted into the DOM with `textContent`, not data-driven HTML.

All figures are descriptive. `delta_expected_score` is the entity expected score minus its matched no-entity baseline under the restricted nine-label distribution; it is not by itself a standalone causal effect. Localization remains Jacobian-transported representation evidence, not chain-of-thought, and lens calibration remains an experimental condition. No activation, residual, hidden state, or gradient payload is written.

## Artifact-only statistical analysis

A completed run can also be converted into CSV tables containing descriptive statistics, confidence intervals, effect sizes, and statistical tests:

```bash
uv run synthetic-entity-bias analyze \
  --run-root artifacts/qwen3.5-4b/synthetic-entity-bias-2020-2025/runs/pretrained-wikitext-20260812
```

The default destination is `<run-root>/analysis/`. Use `--output-dir` to select another directory and `--replace-existing` to replace an existing non-empty analysis directory. Like `visualize`, this command first applies the strict completed-run artifact validation, does not load the model, tokenizer, or lens, and never modifies the source `manifest.json`.

The output is specific to this experiment:

```text
analysis/
├── template_statistics.csv
├── template_pairwise_tests.csv
├── familiarity_tier_statistics.csv
├── familiarity_tier_pairwise_tests.csv
├── sector_statistics.csv
├── localization_statistics.csv
├── baseline_statistics.csv
├── entity_distribution_diagnostics.csv
├── temperature_null_diagnostics.csv
└── localization_transition_diagnostics.csv
```

- `template_statistics.csv` reports entity-level count, mean, sample standard deviation, SEM, median, quartiles, IQR, range, sign fractions, deterministic percentile-bootstrap 95% intervals, one-sample t-tests, Wilcoxon signed-rank tests, Cohen's dz, and Holm-adjusted p-values for each prompt context.
- `template_pairwise_tests.csv` aligns contexts by ticker and reports `template_a - template_b` paired differences, bootstrap intervals, paired t/Wilcoxon tests, Cohen's dz, Pearson/Spearman association, sign agreement/reversal, and Holm-adjusted p-values.
- `familiarity_tier_statistics.csv` reports the same descriptive summaries by context and deterministic familiarity tier, together with a Kruskal–Wallis omnibus result for that context.
- `familiarity_tier_pairwise_tests.csv` reports tier-pair mean and median differences, bootstrap intervals, Welch t-tests, Mann–Whitney U tests, Hedges' g, and Holm-adjusted p-values.
- `sector_statistics.csv` explodes pipe-delimited sector memberships exactly as the visualizer does. All groups remain in the table, but one-sample tests are run only for sector–context groups with `n >= 20`; excluded groups contain empty test values and a machine-readable reason.
- `localization_statistics.csv` summarizes each context and localization metric by absolute peak layer/depth, final-layer value, layer-profile mean/median, signed and absolute trapezoidal AUC, zero crossings, and degeneracy count.
- `baseline_statistics.csv` records the three no-entity baseline scores, entropy, artifact effective temperature, entity score mean/median and their movement from baseline, including score-zero crossings.
- `entity_distribution_diagnostics.csv` reports q05/q95, bias-corrected skewness and excess kurtosis, mean-minus-median, central means, and fixed positive/negative tail counts at absolute delta-E thresholds 0.2, 0.5, 0.75, and 1.0.
- `temperature_null_diagnostics.csv` fits the post-hoc probability null `p_i(T) ∝ p0_i^(1/T)` over the fixed interval `[0.25, 4.0]` and reports expected-score and entropy differences from the null, errors, and template-specific R-squared values.
- `localization_transition_diagnostics.csv` reports metric peaks, final values, sign-changing layer pairs, and the largest adjacent-layer jump for every context and localization metric.

The analysis uses fixed two-sided tests, a fixed deterministic bootstrap seed, 2,000 bootstrap resamples, 95% percentile intervals, and Holm family-wise correction within each reported test family. Empty test fields mean that a test was not valid for the observed sample; the adjacent status and reason fields distinguish insufficient, degenerate, and excluded cases. P-values are not effect magnitudes. Familiarity tiers and sectors are observational deterministic groupings rather than randomized treatments, and multi-sector tickers make sector groups non-independent. The persisted localization artifact contains layer-level aggregates rather than entity-level localization observations, so `localization_statistics.csv` is descriptive and does not invent layer-wise confidence intervals or p-values.

## Completed experiment results (2026-08-12)

The completed pretrained-lens runs are:

```text
artifacts/qwen3.5-4b/synthetic-entity-bias-2020-2025/
  runs/pretrained-wikitext-20260812/
artifacts/qwen3.6-27b/synthetic-entity-bias-2020-2025/
  runs/pretrained-wikitext-20260812/
```

Both runs use the same 3,045-entity pool and three prompt contexts, producing 9,135 entity-template result rows. The familiarity groups contain 605 S&P 500, 481 Russell 1000, and 1,959 Russell 2000 entities. The model-specific pretrained lenses are the pinned Neuronpedia/Salesforce-WikiText artifacts described above. All results in this section are conditional on this entity pool, baseline, restricted nine-label readout, templates, model checkpoints, and lens condition.

### Template-level entity effects

The primary quantity is signed `delta_expected_score = E_entity - E_baseline`. Positive values move the restricted expected score upward relative to `The company`; negative values move it downward.

| Model | Context | Mean delta-E | Bootstrap 95% CI | Cohen's dz | Holm-adjusted t-test p |
|---|---|---:|---:|---:|---:|
| Qwen3.5-4B | negative | +0.2808 | [+0.2705, +0.2914] | +0.9976 | effectively 0 at stored precision |
| Qwen3.5-4B | positive | -0.2102 | [-0.2146, -0.2058] | -1.6416 | effectively 0 at stored precision |
| Qwen3.5-4B | neutral | +0.2849 | [+0.2706, +0.2991] | +0.7144 | 5.60e-275 |
| Qwen3.6-27B | negative | -0.0020 | [-0.0064, +0.0021] | -0.0165 | 0.3625 |
| Qwen3.6-27B | positive | -0.2962 | [-0.3009, -0.2917] | -2.2208 | effectively 0 at stored precision |
| Qwen3.6-27B | neutral | -0.1504 | [-0.1551, -0.1452] | -1.0706 | effectively 0 at stored precision |

The 4B run therefore has positive aggregate shifts in the negative and neutral contexts and a negative shift in the positive context. The 27B run has strong negative shifts in the positive and neutral contexts, while its negative-context mean is practically zero. For 27B negative, the Holm-adjusted Wilcoxon p-value is `1.09e-7` despite the mean CI crossing zero and the t-test being non-significant. This difference should be reported as distributional asymmetry or a nonzero-rank location result, not as evidence for a stable nonzero mean.

These model differences are descriptive. The experiment does not provide a formal cross-model test, and the checkpoints differ in residual width, layer count, training, and model-specific lens artifact. The observed direction changes therefore do not establish a parameter-count scaling law.

### Baselines, distribution shape, and tail diagnostics

The matched no-entity baselines are:

| Model | Context | Baseline expected score | Entropy (nats) | Artifact effective temperature |
|---|---|---:|---:|---:|
| Qwen3.5-4B | negative | -0.1969 | 1.6842 | 0.006554 |
| Qwen3.5-4B | positive | +3.2347 | 0.9477 | 0.006680 |
| Qwen3.5-4B | neutral | -0.6553 | 1.1647 | 0.006516 |
| Qwen3.6-27B | negative | -2.2340 | 1.1675 | 0.007096 |
| Qwen3.6-27B | positive | +3.7738 | 0.5785 | 0.007100 |
| Qwen3.6-27B | neutral | -0.0404 | 0.1042 | 0.007453 |

For 4B negative, the entity mean score moves from `-0.1969` to `+0.0839`, crossing the restricted score midpoint. The 27B negative entity mean is `-2.2360`, almost unchanged from its `-2.2340` baseline; its entity median is `-2.2168`. The two models therefore do not show the same aggregate crossing/convergence pattern.

The 27B negative delta-E distribution explains why its mean-based and rank-based tests disagree. Its mean is `-0.0020`, median is `+0.0172`, skewness is `-2.5988`, excess kurtosis is `18.4252`, minimum is `-1.4961`, and maximum is `+0.4953`. There are 151 negative versus 26 positive observations with absolute delta-E at least `0.2`; 17 observations are at most `-0.5`, while none are at least `+0.5`. Removing the 177 observations with absolute delta-E at least `0.2` leaves a central mean of `+0.0137`.

This supports a single central mass with a slight positive location shift and a sparse, heavy negative tail. It does not establish a bimodal mixture: histogram peaks are bin-dependent, and the current diagnostics do not include a preregistered dip test or a bootstrap-stable mixture analysis. Sector-associated cancellation also contributes to the pooled mean, but substantial within-sector variation remains.

### Baseline-centred probability-temperature null

The persisted artifacts contain the complete restricted nine-point probabilities but not the original model logits. A post-hoc mathematical null therefore starts from each matched baseline distribution `p0` and allows only one-dimensional sharpening or flattening:

```text
p_i(T) = p0_i^(1/T) / sum_j p0_j^(1/T),  T in [0.25, 4.0]
```

For each entity, `T` is fitted by deterministic least-squares distance in probability space. The fitted quantity is called **fitted probability temperature** and is not the artifact `effective_temperature`, not a recovered full-vocabulary model-logit temperature, and not a new inference run.

| Model | Context | Observed mean delta-E | Null mean delta-E | Mean difference from null | Expected-score R-squared | Entropy R-squared |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | negative | +0.2808 | -0.0201 | +0.3009 | -1.111 | 0.781 |
| Qwen3.5-4B | positive | -0.2102 | -0.1556 | -0.0547 | 0.729 | 0.795 |
| Qwen3.5-4B | neutral | +0.2849 | +0.0800 | +0.2049 | -0.099 | 0.648 |
| Qwen3.6-27B | negative | -0.0020 | +0.0096 | -0.0116 | -0.828 | 0.166 |
| Qwen3.6-27B | positive | -0.2962 | -0.4189 | +0.1227 | -0.753 | -0.148 |
| Qwen3.6-27B | neutral | -0.1504 | -0.1205 | -0.0299 | 0.345 | 0.972 |

The null explains substantial entropy variation in several contexts and partially explains expected-score movement for 4B positive and 27B neutral. Negative expected-score R-squared for 4B negative/neutral and 27B negative/positive means that the fitted one-dimensional concentration trajectory performs worse than a constant template-mean predictor. Most observed effects therefore require label-specific probability-mass relocation beyond simple baseline sharpening or flattening. This remains a post-hoc readout diagnostic and does not identify a causal semantic mechanism.

### Context dependence within each model

Ticker-aligned paired comparisons show that the entity effect is not well described by one context-invariant company score.

| Model | Paired contrast | Mean difference | Bootstrap 95% CI | Cohen's dz | Pearson r | Spearman rho |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B | negative - positive | +0.4910 | [+0.4807, +0.5008] | +1.7129 | +0.1865 | +0.1913 |
| Qwen3.5-4B | negative - neutral | -0.0041 | [-0.0205, +0.0127] | -0.0085 | +0.0180 | +0.0126 |
| Qwen3.5-4B | positive - neutral | -0.4951 | [-0.5096, -0.4801] | -1.1502 | -0.0967 | -0.0802 |
| Qwen3.6-27B | negative - positive | +0.2942 | [+0.2869, +0.3019] | +1.4275 | -0.3236 | -0.3154 |
| Qwen3.6-27B | negative - neutral | +0.1484 | [+0.1427, +0.1541] | +0.9000 | +0.2047 | +0.2846 |
| Qwen3.6-27B | positive - neutral | -0.1458 | [-0.1526, -0.1393] | -0.7926 | +0.0985 | +0.1460 |

For 4B, negative and neutral have nearly identical aggregate means but almost zero ticker-level association. The similar means therefore arise from different entity-level response patterns rather than stable responses to the same companies. For 27B, negative and positive are negatively associated across tickers. Together, these observations support a context-dependent `entity × prompt` interaction and argue against interpreting the output as a single stored reputation scalar.

### Familiarity-tier associations

Mean delta-E by deterministic index-membership tier is:

| Model | Context | S&P 500 | Russell 1000 | Russell 2000 |
|---|---|---:|---:|---:|
| Qwen3.5-4B | negative | +0.1313 | +0.1461 | +0.3600 |
| Qwen3.5-4B | positive | -0.2543 | -0.2488 | -0.1871 |
| Qwen3.5-4B | neutral | +0.6217 | +0.4131 | +0.1494 |
| Qwen3.6-27B | negative | +0.0132 | -0.0103 | -0.0046 |
| Qwen3.6-27B | positive | -0.3985 | -0.3578 | -0.2495 |
| Qwen3.6-27B | neutral | -0.2121 | -0.1850 | -0.1229 |

The largest 4B tier contrast is neutral S&P 500 minus Russell 2000: `+0.4723`, 95% CI `[+0.4386, +0.5059]`, Hedges' `g = +1.4743`. The 4B negative context has the opposite ordering: Russell 2000 is more positive than S&P 500 by `0.2287`. In 27B, the negative-context tier differences are statistically detectable but small; S&P 500 minus Russell 2000 is `+0.0178`, with `g = +0.1498`. The 27B positive S&P 500 minus Russell 2000 contrast is `-0.1490`, with `g = -1.3390`, and the neutral contrast is `-0.0892`, with `g = -0.7674`.

These are observational associations, not a causal estimate of familiarity or training exposure. Index tier is confounded with company size, sector composition, public prominence, likely corpus frequency, tokenization, and other entity attributes. The source years are also provenance rather than independently verified historical index membership.

### Sector-associated heterogeneity

Sector summaries show that an aggregate mean can hide opposing subgroup patterns. The clearest example is the 27B negative context: its overall mean is `-0.0020`, but Energy is `+0.0644` (`dz = +0.9801`), Utilities is `+0.0606` (`dz = +0.8511`), Information Technology is `-0.0564` (`dz = -0.4521`), and Health Care is `-0.0444` (`dz = -0.3841`). Thus, an aggregate value near zero does not imply that entity-level or subgroup effects are uniformly near zero.

The 4B sector effects are more directionally uniform within each context but still differ materially in magnitude. For example, neutral-context means range from `+0.1542` for Energy to `+0.6993` for Communication Services. In 27B, positive and neutral primary-sector groups are consistently negative, while negative-context sectors include both signs.

Only sector-context groups with `n >= 20` receive primary one-sample tests. Sector membership is not randomized, some entities preserve multiple sector memberships, and sector composition overlaps with familiarity and other company characteristics. These results establish sector-associated heterogeneity, not sector-level causal bias.

### Jacobian-transported localization profiles

The correlation and linear-fit metrics generally peak in the later part of each model:

- Qwen3.5-4B: Pearson, Spearman, and linear R-squared peaks occur around normalized depth `0.68` to `0.81` (layers 21 to 25 of the persisted grid).
- Qwen3.6-27B: the same metrics peak around normalized depth `0.76` to `0.87` (layers 48 to 55).

Representative strong profiles include:

- 4B neutral: peak Pearson `0.8990`, peak Spearman `0.9191`, peak linear R-squared `0.8082`; final-layer values are `0.8358`, `0.8554`, and `0.6985`.
- 27B positive: peak Pearson `0.8860`, peak Spearman `0.8897`, peak linear R-squared `0.7850`; final-layer values are `0.6748`, `0.6720`, and `0.4553`.

Mean cosine does not consistently agree with correlation or R-squared. In 4B it often peaks at layers 3 to 4; in 27B it can change sign repeatedly or remain negative in later layers even when rank and linear associations are positive. For 4B, mean cosine has 0, 2, and 2 sign changes in the negative, positive, and neutral contexts; the largest adjacent jumps are `0.2448` into layer 3, `0.2226` into layer 4, and `0.4523` into layer 11. For 27B, the corresponding sign-change counts are 12, 5, and 1, with largest adjacent jumps `0.3437` into layer 30, `-0.2693` into layer 14, and `-0.4814` into layer 17. This metric dependence and the local jumps are consistent with repeated geometric re-expression across layers, rather than a fixed direction accumulating or rotating smoothly and monotonically.

The localization evidence is descriptive evidence about a Jacobian-transported representation and its association with the fitted task-local direction. It does not identify a unique storage layer, reveal chain-of-thought or a discrete reasoning path, or establish a causal mechanism.

### Supported interpretation and remaining limits

The completed runs support the following bounded interpretation:

1. Named-company prompts systematically change the restricted nine-label answer distribution relative to the matched generic baseline.
2. The direction and magnitude of that change depend strongly on prompt context and model checkpoint.
3. Weak, absent, or negative cross-context ticker correlations indicate that the response is not a single context-invariant company preference score.
4. Familiarity-tier and sector groupings reveal structured heterogeneity, including subgroup cancellation hidden by aggregate means.
5. Jacobian-transported correlation and linear-fit evidence is usually strongest in the model's later half, while directional cosine evidence is less geometrically stable.

The runs do **not** establish that a company name has a standalone causal effect, that the models hold a stable real-world financial belief, that index familiarity or sector causes the observed differences, that 27B is globally more or less biased than 4B, or that the localization peak is a reasoning location. The most direct follow-up for stronger causal claims is the repository's entity-only counterfactual pairing and residual activation patching protocol with matched contexts and fixed outcome options.
