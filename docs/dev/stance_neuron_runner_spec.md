# Native GLM discovery runner is a screening stage, not completed R2

## D1: Fixed runnable plan

`scripts/run_stance_neuron_discovery.py` accepts only `--model`, `--inputs`,
`--parent`, `--output-dir`, `--shard-index`, and `--num-shards`. It validates the
approved full503/2012 inputs and completed original baseline using accepted
loaders, then verifies the original checkpoint metadata, backend, grammar,
tokenizer, template, dtype and cache settings as localization does. Only GLM,
40 actual dense layers, 16 independent native coordinates per layer, seed
20261003, plain JSON and inherited 512-token/180-second/cached generation are
supported. It does not contact existing jobs.

The full four-role `NeuronDiscoveryPlan` stays bound to parent/input/model and
protocol hashes. This execution slice expands only its fit keys at `+-`,
exactly 302 unique tickers, across -2/+2 and `prompt_and_decode`. Each shard
owns layers whose index modulo shard count equals its index. The full budget
is 386560 interventions plus two clean and one zero-native-hook generation
per assigned layer. No ticker, layer, coordinate, dose or policy override is
available. Internal helper fixtures may be tiny, never public research plans.

## D2: Native gate and immutable execution

Before any intervention at a layer, generate the first lexical fit key twice
clean and once through the actual native down-projection input hook at scalar
zero, on that layer's first panel coordinate. All three must be schema-valid
and match the complete parent output including token IDs, text, finish and
payload. A fresh absolute-position tracker surrounds each native execution. The runtime
inspects the complete accepted panel once and applies the accepted
`scoped_mlp_addition` directly after arm membership validation, avoiding a
full40-layer SHA re-ranking for every intervention.
Gate failures are written before aborting and remain failed on resume.

Registration binds the full panel/roles/parent plus runtime source hashes and
shard settings. Write-once records bind the configuration hash and immutable
cell key (panel/layer/neuron/ticker/condition/trial/dose/scope). Each envelope
has a canonical payload SHA. Resume imports and validates every existing
record and its generation/parent provenance before invoking any callbacks.
Unknown files, staging leftovers, symlinks, changed registration, invalid
hashes, missing gates and summaries with missing cells are rejected. Executed
failure outcomes are retained and never retried. Missing cells without a
summary are resumable work, not a complete result. No raw tensors are written.

## D3: Outputs and limits

Each generated cell stores only structured generation and compact native hook
coordinates. Per-candidate/dose summaries preserve all planned fit denominators,
baseline buy/sell/unknown, directional flips, any flip, schema completion and
failure counts. Directional rates with zero denominator are null. Initial plan
and per-cell progress are emitted with executed/missing counts. Stable summary
accounting includes summed recorded generation seconds and parent runtime
resource metadata. Completion means this shard has every planned cell and
passed layer gate, not experimental acceptance. `research_eligible=false`
always. Multi-shard output is never marked globally complete by one shard.

There is no discovery-success threshold, validation, dose calibration, control
comparison or selection implementation. Subsequent selection is fixed top8 by
fit company-first any-flip response, ties `(layer, neuron)`. Those subsequent
stages and central reviewer approval/GPU submission remain dependencies.
