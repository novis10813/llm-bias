# Compile fit-only full-response cone teachers before training

This engineering slice implements the fixed Wave2 cross-evidence finance target
adaptation. It does not certify truth, evidence faithfulness, trained operators,
or generation acceptance. Main reviews the commit before a real CPU compilation.

## REQ-1. Consume verified full inputs and baseline boundaries

Verified APIs: `load_baseline_inputs`, `load_completed_baseline`, and
`load_merged_baseline` provide immutable full-population snapshots. The compiler
rebuilds `build_baseline_plan(inputs, parent.plan.identity)` and requires exact
plan equality and 2012 rows. Direct Python construction of these capabilities is
trusted caller input, as documented by `CompletedBaseline`; the CLI always uses
the loaders. It accepts original `--parent` plus optional `--recovery`, never a
caller-authored teacher or ticker subset. Merged rows retain per-row source and
policy. Harmony produces `unsupported_harmony_full_channel_teacher_policy`.

## REQ-2. Fixed pairing and coverage

New API: `compile_cone_teachers(inputs, parent, tokenizer)` returns a JSON-ready
pack with manifest, source/review documents, teacher records and unavailable
cells. It plans three purposes for each authoritative fit ticker (302): addition
uses -- input and same-ticker ++ buy target, ablation uses ++ input and -- sell
target, retain uses +- input and its primary response. No validation, calibration,
or evaluation response becomes a target. Invalid or wrong-class required sources
produce unavailable cells. Any unavailable cell suppresses the entire accepted
teacher list. Successful coverage is exactly 906, 302 per purpose.

## REQ-3. Verify actual tokenizer, grammar and entire response

Verified APIs: `compile_decision_grammar`, `render_decision_prompt`,
`validate_decision_payload`, and `TeacherResponse`. Recompile CPU grammar with
recorded head vocabulary and declared stop IDs. Compare tokenizer, schema,
grammar, byte policy, compiler settings and generation policy provenance. Render
source and input prompts with the recorded wrapper and verify body/wrapper/schema
hashes. The CLI loads only `AutoTokenizer` from the exact recorded local checkpoint
and verifies its metadata files against the parent's model binding. No weights,
GPU, raw tensors or model forward pass.

Require HF decoded bytes to equal backend token bytes for the whole continuation
and target payload. Remove only explicitly recognized terminal stop IDs, record
those IDs and finish reason, and preserve the remaining entire JSON including
grammar-accepted internal whitespace and escapes. The pinned grammar rejects
outer whitespace. The compiler reports it unavailable rather than trimming. Reject embedded stops, special/blocked tokens, duplicate
keys, wrong order, truncated JSON and grammar-invalid sequences. No re-encoding,
reason trimming, header stripping or inferred channel extraction. Mask every
response token true and every input prompt token false. `TeacherResponse` supplies
the accepted causal-shift API, not logit-position masks.

## REQ-4. Immutable provenance and reporting

New CLI: `scripts/compile_stance_cone_teachers.py --inputs PATH --parent PATH
--model PATH --output-dir FRESH_PATH [--recovery PATH]`. Write a fresh directory
only, with config, source manifest, machine review manifest, teachers JSONL,
unavailable JSON and final manifest containing file hashes and purpose counts.
Source documents include the complete consumed parent's raw file hashes and
selected full generation exports, keys and effective per-row policies. Review
binds source hash, fixed pairing, label, boundaries and adaptation name. This is
machine contract review, not human expert annotation. Pin compiler/dependency code
and lock hashes. Unavailable reports remain diagnostics and CLI exits 2. Errors
reject rather than fabricate coverage. Reusing any existing output path fails.

## S1. One commit verifies the compiler independently

Create only the module, script, tests and this spec. Tests first exercise the
actual fast tokenizer and installed grammar with full503 fixtures, including
906 coverage, masks, fit isolation, missing class, byte mismatch, malformed JSON,
terminal stops, template/provenance mismatches, Harmony refusal and fresh output.
Run focused pytest with worktree PYTHONPATH and `uv run --no-sync`, followed by
`uv lock --check` and `git diff --check`. Main owns full regression and real
compilation. Training and validation selection are separate dependencies.

Verification evidence: tests began with an expected missing-module collection
failure, then passed 45 focused compiler/objective tests. The synthetic effective
fixture exercises `row_source_for` and the accepted `row_policy_for` policy-only
interface. The production merged loader remains Harmony-only and therefore
explicitly unsupported for teachers. No actual GLM pack has been compiled here.
