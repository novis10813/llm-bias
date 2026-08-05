# Artifact identity and run manifest contract

This contract is the shared foundation for experiment producers. It does not
change existing producers, migrate ignored runtime output, or provide a
compatibility fallback. New workflow units should import the helpers in
`llm_bias.core.artifact_paths` and `llm_bias.core.artifact_manifest` directly.

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
final component. `dataset_slug` uses the same safe spelling. The path helpers
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

`create_run_manifest(model, dataset, run_id)` creates and persists a manifest
with `schema_version: 1` and status `created`. A normal run calls:

```python
manifest.start()
manifest.start_stage("prompt-generation")
manifest.register_artifact(
    input_path,
    artifact_type="prompt_input",
    stage="prompt-generation",
    role="input",
)
manifest.register_artifact(
    lens_path,
    artifact_type="jacobian_lens",
    stage="prompt-generation",
    role="lens",
)
manifest.register_artifact(
    output_path,
    artifact_type="prompt_output",
    stage="prompt-generation",
    role="output",
    record_count=record_count,
)
manifest.finish_stage("prompt-generation", record_count=record_count)
manifest.complete()
manifest.save()
```

`register_artifact` hashes an existing file. For JSONL files, omitted
`record_count` is inferred and validated automatically. A registration contains
only compact provenance:

- `artifact_type`, `stage`, `status`, and `role` (`input`, `lens`, or `output`);
- a path reference (relative to the run when possible);
- the file SHA-256 and optional record count; and
- optional small metadata supplied by the producer.

The manifest exposes the same references in `artifacts` and the role-specific
`input_refs`, `lens_refs`, and `output_refs` arrays. `record_counts` totals
registered counts by artifact type. `stages` records stage status and optional
counts. A run ends with `complete()` or `fail(message)`; both require `save()` to
persist the new state. `RunManifest.load(path)` validates and rehydrates a
manifest.

Prompt generation and optional backward/generated-token attribution are separate
versioned artifacts and should use distinct `artifact_type` and `stage` values.
An attribution artifact must never be represented as a fallback for a missing
prompt-generation artifact. Raw activation and raw gradient artifact types are
rejected by the registration API. Store only compact top-k, rank, probability,
summary, token, and provenance data in experiment outputs.
