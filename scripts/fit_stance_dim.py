"""Fit-only DIM capture on a declared five-depth post-block panel.

No validation selection, dose calibration, margin or efficacy claim. Sources and
per-row span means remain in memory. Every invocation requires a fresh run path.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.run_stance_baseline import runtime_metadata
from scripts.smoke_stance_checkpoint import generation_adapter, compile_smoke_grammar
from scripts.recover_stance_baseline_truncations import (
    publish, check_prompt, check_capability, bind_relocated_tokenizer,
)
from scripts.run_stance_localization import _equal, _provenance
from llm_bias.core.artifact_paths import sha256_json, sha256_bytes, canonical_json_bytes
from llm_bias.core.model import load_model
from llm_bias.core.stance_baseline_inputs import load_baseline_inputs
from llm_bias.core.stance_baseline_parent import load_completed_baseline
from llm_bias.core.stance_baseline_merged import load_merged_baseline
from llm_bias.core.stance_dim_extraction import extract_dim_candidates, pool_original_span
from llm_bias.core.prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from llm_bias.core.inference.structured_output import StructuredGenerationPolicy, generate_structured
from llm_bias.core.inference.harmony_generation import generate_harmony_structured
from llm_bias.core.inference.stance_localization_execution import _same_full_output, _bind_expected
from llm_bias.core.inference.stance_interventions import GenerationPositionTracker
from llm_bias.core.inference.stance_transient_capture import capture_prompt_residual

SPANS = ('entity', 'evidence1', 'evidence2', 'instruction')


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('inputs', 'parent', 'model', 'output-dir'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--recovery', type=Path)
    return p


def depth_panel(count):
    if type(count) is not int or count <= 0:
        raise ValueError('actual layer count must be a positive integer')
    return tuple(sorted({i * (count - 1) // 4 for i in range(5)}))


def fit_rows(inputs, parent):
    tickers = set(inputs.roles['roles']['fit'])
    return tuple(sorted((row for row in parent.rows if row.key.ticker in tickers),
                        key=lambda row: row.key))


def descriptor(inputs, parent, count, d_model, bindings):
    panel = dict(layers=list(depth_panel(count)), spans=list(SPANS), hook_site='post',
                 rule='floor_i_times_L_minus_1_over_4', actual_layer_count=count)
    capture_policy = dict(original_prompt_only=True, pooling='arithmetic_span_mean',
        full_parent_output_match=True, simultaneous_layers=True, bindings=bindings)
    return dict(kind='stance_dim_fit_v1', research_eligible=False,
        parent_sha256=parent.parent_sha256, plan_hash=parent.plan.plan_hash,
        inputs_manifest_sha256=inputs.manifest_sha256,
        full_plan_identity=parent.plan.identity.to_dict(), execution_roles=['fit'],
        planned_fit_rows=len(fit_rows(inputs, parent)), d_model=d_model, panel=panel,
        candidate_panel_sha256=sha256_json(panel), capture_policy=capture_policy,
        capture_policy_sha256=sha256_json(capture_policy))


def fresh_output(directory, desc):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'records').mkdir()
    publish(directory, 'config.json', dict(descriptor=desc, config_sha256=sha256_json(desc)))


def capture_row(model, tokenizer, prompt, capability, policy, expected, layers):
    """One generation, simultaneous sources, mean only after parent authentication."""
    means = {}
    actual = None
    try:
        weight = model.hf_model.get_input_embeddings().weight
        ids = torch.tensor([prompt.inference_token_ids], device=weight.device, dtype=torch.long)
        tracker = GenerationPositionTracker(ids.shape[1], policy.use_cache)
        with ExitStack() as stack:
            stack.enter_context(tracker.track(model))
            holders = {layer: stack.enter_context(capture_prompt_residual(model,
                tracker=tracker, layer=layer, hook_site='post')) for layer in layers}
            driver = (generate_harmony_structured if policy.channel_policy == 'harmony_no_tools'
                      else generate_structured)
            actual = driver(model, tokenizer, ids, capability, policy=policy)
            if actual.failure_type is not None:
                return actual, {}, 'generation_failed'
            if not _same_full_output(actual, expected):
                return actual, {}, 'parent_mismatch'
            try:
                _provenance(actual, expected)
                for name in ('model_binding', 'backend', 'backend_version', 'byte_policy',
                             'compiler_policy', 'grammar_correction', 'channel_contract'):
                    _equal(actual.provenance.get(name), expected.provenance.get(name),
                           'generation configuration differs')
            except ValueError:
                return actual, {}, 'configuration_mismatch'
            for layer, holder in holders.items():
                # The accepted holder proves first complete ordered prompt authority.
                source = holder.require_source()[0].detach().to(device='cpu', dtype=torch.float64).numpy()
                for span in SPANS:
                    record = getattr(prompt, span + '_span')
                    means[layer, span] = pool_original_span(source,
                        range(record.token_start, record.token_end),
                        original_prompt_length=ids.shape[1], d_model=weight.shape[1])
                del source
        return actual, means, 'matched'
    except Exception:
        # Exception messages may embed tensors. Never serialize them here.
        means.clear()
        return actual, {}, 'capture_unsupported'


def fit(directory, inputs, parent, desc, capture):
    """Internal fake-friendly execution boundary. Public caller authenticates inputs."""
    started = time.monotonic()
    rows = fit_rows(inputs, parent)
    sites = [(l, s) for l in desc['panel']['layers'] for s in SPANS]
    pooled = {site: {} for site in sites}
    attempted = matched = 0
    status = 'completed_capture'
    config_hash = sha256_json(desc)
    try:
        for row in rows:
            attempted += 1
            try:
                actual, means, row_status = capture(row.key)
            except Exception:
                actual, means, row_status = None, {}, 'capture_unsupported'
            if row_status == 'matched' and set(means) != set(sites):
                row_status = 'capture_unsupported'
            publish(directory / 'records', sha256_json(row.key.to_dict()) + '.json',
                dict(key=row.key.to_dict(), config_sha256=config_hash, status=row_status,
                     full_output_match=(None if actual is None else
                         _same_full_output(actual, parent.generation_for(row.key))),
                     configuration_match=(row_status == 'matched'),
                     failure_type=None if actual is None else actual.failure_type,
                     actual_token_sha256=None if actual is None else actual.generated_token_sha256))
            if row_status != 'matched':
                means.clear()
                status = row_status
                # No partially captured fit can export an apparently supported operator.
                for values in pooled.values(): values.clear()
                break
            matched += 1
            for site in sites: pooled[site][row.key] = means[site]
            means.clear()
        extractions = []
        for layer, span in sites:
            extractions.append(extract_dim_candidates(plan=parent.plan,
                members=inputs.members, roles=inputs.roles['roles'], rows=rows,
                pooled_residuals=pooled[layer, span], layer=layer, span=span,
                d_model=desc['d_model'], parent_sha256=parent.parent_sha256,
                candidate_panel_sha256=desc['candidate_panel_sha256'],
                capture_policy_sha256=desc['capture_policy_sha256']).to_dict())
            pooled[layer, span].clear()
        publish(directory, 'operators.json', dict(config_sha256=config_hash, extractions=extractions))
        report = dict(kind=desc['kind'], config_sha256=config_hash, research_eligible=False,
            status=status, complete_capture=status == 'completed_capture' and matched == len(rows),
            planned_rows=len(rows), attempted_rows=attempted, matched_rows=matched,
            missing_rows=len(rows) - matched, retained_capture_rows=(matched if status == 'completed_capture' else 0),
            site_families=len(sites), condition_candidates=4 * len(sites),
            elapsed_seconds=time.monotonic() - started)
        publish(directory, 'summary.json', report)
        return report
    finally:
        for values in pooled.values(): values.clear()


def run(args):
    # Reserve a fresh path before any runtime work. Never reuse an interrupted run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'records').mkdir()
    try:
        inputs = load_baseline_inputs(args.inputs)
        parent = (load_completed_baseline(args.parent, inputs=inputs) if args.recovery is None
                  else load_merged_baseline(args.parent, args.recovery, inputs=inputs))
        parent_metadata = parent.metadata
        bindings = parent_metadata.get('original_metadata', parent_metadata)['bindings']
        checkpoint = args.model.resolve(strict=True)
        metadata = runtime_metadata(checkpoint)
        if args.recovery is None:
            _equal(metadata['model'], bindings['model'], 'original checkpoint path/metadata differs')
        else:
            _equal(metadata['model']['metadata_file_sha256'],
                   bindings['model']['metadata_file_sha256'], 'effective checkpoint metadata differs')
        for name in ('torch', 'transformers', 'xgrammar', 'jlens', 'cuda', 'kernel_policy',
                     'cudnn', 'deterministic_algorithms'):
            _equal(metadata['backend'][name], bindings['backend'][name], 'parent backend differs')
        metadata['code']['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha256_bytes(Path(__file__).read_bytes())
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        loaded, _, device = load_model(str(checkpoint), device_map=None,
            dtype='native' if bindings['backend']['requested_dtype'] == 'native' else torch.bfloat16)
        model = generation_adapter(loaded)
        model.hf_model.eval()
        tokenizer = model.tokenizer
        if args.recovery is not None:
            metadata['tokenizer_relocation'] = bind_relocated_tokenizer(
                tokenizer, metadata['model'], bindings['model'])
        head = model.hf_model.get_output_embeddings().weight
        embedding = model.hf_model.get_input_embeddings().weight
        if any(torch.device(d).type != 'cuda' for d in (device, head.device, embedding.device)):
            raise ValueError('actual model must be on CUDA')
        for value, name in ((head.dtype, 'head_dtype'), (embedding.dtype, 'embedding_dtype')):
            _equal(str(value), bindings['backend'][name], 'actual dtype differs')
        _equal(getattr(model.hf_model.config, '_attn_implementation', None),
               bindings['backend']['attention_implementation'], 'attention policy differs')
        metadata['backend'].update(device=str(device), head_dtype=str(head.dtype),
            embedding_dtype=str(embedding.dtype),
            requested_dtype=bindings['backend']['requested_dtype'],
            attention_implementation=getattr(model.hf_model.config, '_attn_implementation', None),
            gpu_name=torch.cuda.get_device_name(embedding.device))
        wrapper_record = bindings['template']['actual_wrapper_record']
        wrapper = WrapperPolicy(**{k: v for k, v in wrapper_record.items() if k != 'tokenizer_chat_template'})
        members = {m.ticker: m for m in inputs.members}
        rows = fit_rows(inputs, parent)
        prompts, capabilities, policies, sources = {}, {}, {}, {}
        for row in rows:
            key = row.key
            expected = parent.generation_for(key)
            policy = StructuredGenerationPolicy(**expected.provenance['generation_policy'])
            policy_hash = sha256_json(asdict(policy))
            if policy_hash not in capabilities:
                capabilities[policy_hash] = compile_smoke_grammar(tokenizer, head.shape[0],
                    expected.provenance['stop_token_ids'], policy.channel_policy)
            capability = capabilities[policy_hash]
            check_capability(capability, expected.provenance)
            if expected.failure_type is None:
                _bind_expected(expected, getattr(capability, 'json_capability', capability), policy, 'parent')
            prompt = render_decision_prompt(tokenizer, members[key.ticker],
                inputs.pair_for(key.ticker, key.condition, key.trial_id), wrapper_policy=wrapper)
            check_prompt(prompt, bindings)
            if prompt.schema_sha256 != parent.plan.identity.schema_sha256:
                raise ValueError('prompt schema differs')
            prompts[key], policies[key] = prompt, policy
            sources[sha256_json(key.to_dict())] = dict(key=key.to_dict(),
                source=parent.row_source_for(key) if hasattr(parent, 'row_source_for') else 'original',
                policy=parent.row_policy_for(key) if hasattr(parent, 'row_policy_for') else asdict(policy))
        desc = descriptor(inputs, parent, len(model.layers), embedding.shape[1],
            dict(runtime=metadata, template=bindings['template'], row_sources=sources,
                 grammar=parent.generation_for(rows[0].key).provenance))
        publish(args.output_dir, 'config.json', dict(descriptor=desc, config_sha256=sha256_json(desc)))
        def capture(key):
            policy = policies[key]
            return capture_row(model, tokenizer, prompts[key], capabilities[sha256_json(asdict(policy))],
                policy, parent.generation_for(key), tuple(desc['panel']['layers']))
        report = fit(args.output_dir, inputs, parent, desc, capture)
        print(canonical_json_bytes(report).decode(), flush=True)
        return 0 if report['complete_capture'] else 1
    except Exception as exc:
        if not (args.output_dir / 'summary.json').exists():
            publish(args.output_dir, 'summary.json', dict(kind='stance_dim_fit_v1',
                research_eligible=False, complete_capture=False, status='preflight_or_fit_failure',
                error_type=type(exc).__name__))
        raise


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except Exception as exc:
        print(canonical_json_bytes(dict(kind='stance_dim_fit_failure', research_eligible=False,
            error_type=type(exc).__name__)).decode(), file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
