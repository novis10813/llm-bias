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
│   ├── entity_effect_by_tier.{png,svg,pdf}
│   ├── template_relationships.{png,svg,pdf}
│   ├── localization_profiles.{png,svg,pdf}
│   └── sector_effects.{png,svg,pdf}
├── captions/
│   └── <one publication caption per figure>.md
└── tables/
    ├── template_summary.csv
    ├── familiarity_tier_summary.csv
    ├── sector_summary.csv
    ├── ticker_template_effects.csv
    └── localization_summary.csv
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
