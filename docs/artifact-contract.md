# Artifact identity and run manifest contract

This contract describes the canonical run manifest emitted by the prompt-analysis
runner. Producers use the helpers in `llm_bias.core.artifact_paths` and
`llm_bias.core.artifact_manifest` directly. The synthetic entity-bias workflow has
an additional normative artifact contract in [synthetic-entity-bias.md](synthetic-entity-bias.md);
its compact schemas, no-raw rules, and lifecycle postchecks apply to synthetic runs.

## Canonical layout

For a model identity, dataset identity, and run ID, artifacts use:

```text
artifacts/
└── <model-slug>/
    ├── jacobian-lens/
    │   └── jacobian_lens.pt
    └── <dataset-slug>/
        └── runs/
            └── <run-id>/
                ├── manifest.json
                └── ... run artifacts ...
```

`model_slug` follows the existing lens convention: a Hub ID such as
`Qwen/Qwen3.5-4B` becomes `Qwen--Qwen3.5-4B`, while a local model path uses its
final component. `dataset_slug` uses the same safe spelling. Runner dataset identity
is the input filename stem by default, and both the run-root segment and manifest
`dataset_slug` are derived from the core `dataset_slug()` helper. The path helpers
accept an alternate artifact root for tests, but the default is `artifacts/`.
Dataset slug is part of the run root, so two datasets cannot overwrite a run
with the same model and run ID.

Use `jacobian_lens_path(model)` for the model-specific lens and
`run_root(model, dataset, run_id)` for a run. Run IDs are directory names, not
opaque paths.

## Stable identity and hashes

`stable_record_id(*parts)` canonicalizes JSON with sorted object keys and returns
`record_<24 lowercase hex characters>` (a 96-bit SHA-256 prefix). Passing separate
identity components keeps boundaries explicit. `sha256_json`, `sha256_bytes`, and
`sha256_file` provide complete lowercase SHA-256 digests for input and output
provenance.

`atomic_write_json` and `atomic_write_jsonl` write a temporary file beside the
destination, flush it, and atomically replace the destination. JSONL writes
return the number of records. JSONL records should be compact, serializable
objects; `count_jsonl_records` validates and counts non-empty lines.

## Manifest schema and lifecycle

`RunManifest` emits schema version `1` at
`artifacts/<model-slug>/<dataset-slug>/runs/<run-id>/manifest.json`. The canonical
object contains:

- `schema_version`, `model`, `model_slug`, `dataset`, `dataset_slug`, `run_id`, and
  `run_root`;
- `status` (`created`, `running`, `complete`, or `failed`), `created_at`, `updated_at`,
  and optional `error`;
- `artifacts`, with role-indexed `input_refs`, `lens_refs`, and `output_refs`;
- `record_counts`, keyed by artifact type; and
- `stages`, keyed by enabled stage name, with status and lifecycle timestamps.

The prompt-analysis runner registers the input CSV and configured Jacobian lens when it
initializes. Each completed stage registers the files it produced. An artifact reference
contains `artifact_type`, `stage`, `status`, `role`, a path, a lowercase SHA-256 digest,
and, for JSONL, an inferred `record_count`; small producer metadata may also be present.
The manifest groups the same references in the role arrays and totals registered counts
in `record_counts`.

The runner lifecycle is `created` → `running` → `complete` or `failed`. The runner
marks each enabled stage `complete` after its command succeeds and its outputs are registered;
a stage failure marks the run failed and records an error. The stored hashes and counts make
independent completion checks possible without relying on stale file existence alone.

The canonical stage tree is:

```text
<run-root>/
├── manifest.json
├── readout/   # RUN_READOUT=1
├── forward/   # RUN_GENERATION=1
└── backward/  # RUN_ATTRIBUTION=1
```

`attribute-generated` metadata records the model identity, parent forward path/hash,
output hash, record counts, and generated-token coverage. It checks model identity,
parent hash, and per-record coverage; it does not claim a same-dataset or same-run binding.
Raw activation and gradient artifact types are rejected. Store only compact top-k, rank,
probability, summary, token, generation, and provenance data.

## Multi-run forward sampling contract

Multi-run price sampling is a separate artifact family, not a `RunManifest` lifecycle run.
A sampling root contains `sampling_manifest.json` and one forward directory per run:

```text
<sampling-root>/
├── sampling_manifest.json
├── run_000/forward/generated_outputs.jsonl
├── run_000/forward/metadata.json
└── run_001/forward/generated_outputs.jsonl
```

The sampling manifest uses `artifact_type: generated_output_sampling`, records the model/input
provenance, selected dates and condition counts, shared generation configuration, and each
run's forward path, SHA-256, record count, and seed. `plot-price-distributions` consumes only
these forward generated-output files; it never uses generated-token backward attribution.
The sampling manifest and lifecycle `manifest.json` are distinct schemas and must not be
interchanged.
