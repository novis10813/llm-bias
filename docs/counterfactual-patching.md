# Counterfactual residual activation patching

**Status:** implemented workflow. This document is the canonical operational
source for `counterfactual-patching`; the research design and dataset protocol
are documented separately.

The workflow records source and target residual streams, replaces a selected
source representation with the target representation, and compares the final
answer distribution. It supports both factual smoke/generalisation pairs and
reviewed 8-K entity-bias pairs. These pair types have different metrics and must
not be mixed in interpretation.

## Scope and boundaries

- The experiment performs residual activation patching across transformer
  layers.
- A Jacobian lens can provide transported representation readouts for compact
  diagnostics. It is not a chain-of-thought trace, discrete reasoning path,
  attention map, or standalone causal proof.
- A patch result is one piece of evidence. Entity-bias claims additionally
  require entity-only pairs, matched controls, paired statistics, and the review
  gates described in the [counterfactual dataset protocol](counterfactual-dataset-generation.md).
- This workflow never writes complete raw activation tensors to the result
  artifacts.

## Required model, lens, and pair alignment

A run must use one consistent set of:

1. model weights and tokenizer;
2. model-specific pair artifact;
3. model-specific complete Jacobian lens, when a lens readout is requested; and
4. output directory and provenance metadata.

The default lens path is:

```text
artifacts/lenses/<model>/jacobian_lens.pt
```

For a model-specific rendered pair file, use an explicit `--pairs` path rather
than relying on the generic default. Token IDs and entity spans are tokenizer
dependent; a pair rendered for Llama must not be reused for Qwen, and vice versa.

The model loader and lens validator should be treated as the runtime authority
for model/lens compatibility. The command examples below use shell variables to
make accidental model/pair mismatches visible:

```bash
MODEL=.cache/models/llama-3.2-1b-instruct
LENS=artifacts/lenses/llama-3.2-1b-instruct/jacobian_lens.pt
RUN_ROOT=artifacts/counterfactual_patching/llama-3.2-1b-instruct/smoke
PAIRS="$RUN_ROOT/pairs.jsonl"
RESULTS="$RUN_ROOT/patch_results.jsonl"
```

## CLI surface

The implemented entry point is:

```bash
uv run counterfactual-patching <command> [options]
```

It has five subcommands:

| Command | Purpose |
|---|---|
| `prepare-data` | Tokenize the built-in factual smoke/generalisation pair source and save aligned pairs. |
| `run` | Run source, target, patched, and control forwards across patch layers. |
| `summarize` | Aggregate patch rows into `layer_summary.csv` and the factual transfer plot. |
| `visualize` | Rebuild the summary and write patch/readout heatmaps. |
| `serve` | Start the interactive local dashboard for one model, lens, and pair file. |

Run `uv run counterfactual-patching <command> --help` for the exact parser help.
The command options currently include:

```text
prepare-data: --model --output --max-pairs
run:          --model --pairs --lens --output --max-pairs
summarize:    --input --output-dir
visualize:   --input --output-dir
serve:       --model --lens --pairs --host --port
```

`run --lens` is required by the CLI. `serve --lens` is optional and defaults to
the canonical model lens path.

## Factual smoke/generalisation pairs

`prepare-data` reads the repository's built-in factual pair source, tokenizes the
prompts with the selected model, and saves only pairs that survive alignment.
It is useful for checking the patching implementation and producing compact
smoke artifacts. It is not the reviewed 8-K entity-bias dataset.

A small model-scoped smoke run is:

```bash
MODEL=.cache/models/llama-3.2-1b-instruct
LENS=artifacts/lenses/llama-3.2-1b-instruct/jacobian_lens.pt
RUN_ROOT=artifacts/counterfactual_patching/llama-3.2-1b-instruct/smoke

uv run counterfactual-patching prepare-data \
  --model "$MODEL" \
  --output "$RUN_ROOT/pairs.jsonl" \
  --max-pairs 4

uv run counterfactual-patching run \
  --model "$MODEL" \
  --pairs "$RUN_ROOT/pairs.jsonl" \
  --lens "$LENS" \
  --output "$RUN_ROOT/patch_results.jsonl" \
  --max-pairs 4

uv run counterfactual-patching summarize \
  --input "$RUN_ROOT/patch_results.jsonl" \
  --output-dir "$RUN_ROOT"

uv run counterfactual-patching visualize \
  --input "$RUN_ROOT/patch_results.jsonl" \
  --output-dir "$RUN_ROOT"
```

`visualize` calls `summarize` internally before creating the heatmaps. The
separate `summarize` command is useful when only tabular aggregation is needed.

## Reviewed 8-K entity-bias pairs

The entity-bias data workflow is separate:

```text
entities -> sample -> annotate -> review-bundle -> promote
  -> build-pairs -> render --model <model> -> validate
```

Use the [counterfactual dataset protocol](counterfactual-dataset-generation.md)
for source data, review/promotion, pairing strategies, and tokenizer-specific
rendering. After promotion and rendering, use the model-specific file produced
under:

```text
artifacts/counterfactual_data/8k_earnings_v1/rendered/<model-slug>/pairs.jsonl
```

The patching commands then use the same model, lens, pair file, and output root:

```bash
MODEL=.cache/models/qwen3.5-4b
LENS=artifacts/lenses/qwen3.5-4b/jacobian_lens.pt
PAIRS=artifacts/counterfactual_data/8k_earnings_v1/rendered/qwen3.5-4b/pairs.jsonl
RUN_ROOT=artifacts/counterfactual_patching/qwen3.5-4b/entity_bias

uv run counterfactual-patching run \
  --model "$MODEL" \
  --pairs "$PAIRS" \
  --lens "$LENS" \
  --output "$RUN_ROOT/patch_results.jsonl"

uv run counterfactual-patching summarize \
  --input "$RUN_ROOT/patch_results.jsonl" \
  --output-dir "$RUN_ROOT"

uv run counterfactual-patching visualize \
  --input "$RUN_ROOT/patch_results.jsonl" \
  --output-dir "$RUN_ROOT"
```

Do not treat the existence of a rendered pair file as proof that the dataset is
research-ready. The content must have passed the review/promotion gate, and the
result must still include controls and uncertainty before supporting a primary
entity-bias claim.

## Variable-length span patching

Pairs retain entity token start/end positions and complete token-ID spans. When
source and target spans have different lengths, the batch patcher uses
normalized span-internal token centers and maps each source position to the
nearest target position.

The mapping is deterministic and:

- does not insert or delete sequence tokens;
- does not average or synthesize activations;
- copies the selected target residual into the source span; and
- records `source_entity_span`, `target_entity_span`, `span_mapping`, and
  `span_mapping_strategy="normalized_nearest"` in each result row.

Source and target prompts may have different sequence lengths in the batch
runner. Answer logits are read at each prompt's own final position. Control
patches use each side's final non-entity position. The interactive dashboard is
more restrictive: it currently rejects source and target prompts whose encoded
tensor shapes differ with an HTTP 400 validation error.

## Metrics and interpretation

### Factual pairs

For factual pairs, the runner defines a fixed answer-token margin and reports
normalized transfer:

```text
transfer = (patched_margin - source_margin)
           / (target_margin - source_margin)
```

The source condition is interpreted as zero and the target condition as one. If
the denominator is effectively zero, `transfer` is `null`. `control_transfer`
is calculated using the corresponding non-entity control patch.

### Entity-bias pairs

For `task_type=entity_bias`, the runner uses the fixed outcome margin:

```text
margin = logit(positive) - logit(negative)
```

It writes:

```text
direct_entity_effect = target_margin - source_margin
causal_patch_effect  = patched_margin - source_margin
control_patch_effect  = control_margin - source_margin
```

It also writes source, target, and patched margins oriented toward the gold
`expected_outcome`. Factual normalized transfer is set to `null` for these rows;
it must not be used as the entity-bias effect metric.

A difference between unrelated token top-1 probabilities is not a causal effect.
Use fixed outcome-token probabilities, fixed logit margins, or a predefined
normalized transfer for the factual task only.

### Jacobian readouts

When a compatible lens is supplied, result rows may include compact readout
contrasts for the entity span and answer position. These are transported
representation diagnostics. They do not independently establish a causal path,
entity bias, or a hidden reasoning trace.

## Artifacts

The default artifact root is `artifacts/counterfactual_patching/`; model-scoped
runs should use a subdirectory to avoid mixing tokenizers and lenses.

| Command | Main outputs |
|---|---|
| `prepare-data` | `pairs.jsonl` |
| `run` | `patch_results.jsonl`, one row per pair and patch layer |
| `summarize` | `layer_summary.csv`, `transfer_by_layer.png` when plotting dependencies are available |
| `visualize` | `patch_transfer_heatmap.png`, and `jacobian_readout_heatmap.png` when readout columns exist |
| `serve` | runtime dashboard plus `artifacts/visualization/dependencies.json` for ignored checkout revisions |

The pair artifact retains prompts, token IDs/text, entity spans, task metadata,
and provenance. The batch `patch_results.jsonl` rows retain pair/task metadata,
source/target margins and ranks, patch/control margins, spans, mappings, and
readout contrasts when a lens is available. The interactive API computes
probabilities, top-k readouts, and summary statistics at request time. Full raw
activations, model weights, and lens binaries remain outside tracked Git
artifacts.

The current aggregate plotting code is primarily factual-transfer oriented. For
entity-bias rows, inspect the per-row fixed-margin effect fields and do not read
`transfer_by_layer.png` as a complete entity-bias result figure until a
bias-specific summary is implemented.

## Interactive dashboard

Start the local dashboard with a model-specific pair file:

```bash
uv run counterfactual-patching serve \
  --model "$MODEL" \
  --lens "$LENS" \
  --pairs "$PAIRS" \
  --host 127.0.0.1 \
  --port 8321
```

It serves `http://127.0.0.1:8321` and exposes:

- `GET /api/info` for model, lens, layer, and device metadata;
- `GET /api/pairs` for available pair summaries; and
- `POST /api/counterfactual` for a source/target/patched forward comparison.

The dashboard keeps the model resident and computes the three forwards for the
selected pair. Its grid shows top-1 readouts by token position and layer, while
hover data contains the requested top-k probabilities and summary statistics.
The backend keeps the complete requested top-k response; the frontend chooses
what to display.

The current dashboard API always computes the factual normalized transfer field
from answer-token margins and labels it `normalized transfer`. For an
`entity_bias` pair this field is not a valid bias metric. Use the batch
`patch_results.jsonl` effect fields for entity-bias analysis, and treat dashboard
support for entity-bias visualization as an explicit future implementation task.

## Verification

For the deterministic span mapping regression:

```bash
uv run pytest -q \
  tests/test_interventions.py::test_normalized_span_mapping_uses_nearest_target_centers
```

For repository-wide verification, follow the commands in
[`CLAUDE.md`](../CLAUDE.md). This document is the canonical source for patching
workflow commands and semantics; the proposal and dataset documents should link
here rather than duplicate this operational detail.
