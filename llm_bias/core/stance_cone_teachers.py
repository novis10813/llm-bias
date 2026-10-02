"""CPU full-response compiler for a declared cross-evidence finance adaptation.

Production callers load complete parent/input capabilities before calling this
module. Tokens and text are allowed exports. Model tensors are never consumed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json

import xgrammar as xgr

from .artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from .stance_baseline_inputs import BaselineInputs
from .stance_baseline_parent import CompletedBaseline
from .stance_baseline_plan import build_baseline_plan
from .stance_cone_objectives import TeacherResponse
from .prompt_input.decision_prompt import WrapperPolicy, render_decision_prompt
from .inference.structured_output import (
    compile_decision_grammar, validate_decision_payload, _check_capability,
)

ADAPTATION = 'cross_evidence_finance_targets_v1'
PAIRINGS = (('addition', '--', '++', 'buy'),
            ('ablation', '++', '--', 'sell'), ('retain', '+-', '+-', None))


def _binding(capability, provenance, generation_policy):
    for name in ('schema_sha256', 'schema_bytes_sha256', 'tokenizer_sha256',
                 'tokenizer_info_sha256', 'head_vocab_size', 'grammar_sha256',
                 'byte_policy'):
        if provenance.get(name) != getattr(capability, name):
            raise ValueError('teacher grammar/tokenizer binding differs: ' + name)
    expected = dict(backend='xgrammar', backend_version=capability.backend_version,
        stop_token_ids=list(capability.stop_token_ids),
        compiler_policy=dict(strict_mode=True, any_order=False, any_whitespace=True),
        grammar_correction='xgrammar-0.2.8-minLength-json-escapes-v1',
        generation_policy=generation_policy['policy'], hf_controls=generation_policy['hf_controls'],
        generation_policy_sha256=sha256_json(generation_policy))
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise ValueError('teacher source policy binding differs: ' + name)


def _response(generation, tokenizer, capability, decoded_vocab, expected_decision):
    if (generation.failure_type is not None or generation.decode_error is not None
            or generation.error_message is not None or not generation.schema_complete
            or not generation.reason_valid or not generation.decision_complete
            or generation.decision not in ('buy', 'sell')):
        raise ValueError('invalid_needed_primary_source')
    if expected_decision is not None and generation.decision != expected_decision:
        raise ValueError('wrong_class_needed_source')
    ids = generation.generated_token_ids
    if not ids or generation.generated_token_sha256 != sha256_json(list(ids)):
        raise ValueError('invalid_source_token_hash')

    def decode(tokens):
        if any(type(t) is not int or not 0 <= t < len(decoded_vocab) for t in tokens):
            raise ValueError('source_token_outside_vocabulary')
        exact = b''.join(decoded_vocab[t] for t in tokens).decode('utf8', errors='strict')
        hf = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if hf.encode('utf8', errors='strict') != exact.encode('utf8'):
            raise ValueError('HF_backend_byte_identity_differs')
        return exact

    if decode(ids) != generation.generated_text:
        raise ValueError('source_full_text_byte_identity_differs')
    end = len(ids)
    while end and ids[end - 1] in capability.stop_token_ids:
        end -= 1
    response, removed = ids[:end], ids[end:]
    if removed and generation.finish_reason != 'eos':
        raise ValueError('terminal_stop_without_explicit_eos_finish')
    if generation.finish_reason not in ('eos', 'schema_complete'):
        raise ValueError('noncomplete_source_finish')
    if (not response or any(t in capability.stop_token_ids or t in capability.blocked_token_ids
                            or t in capability.special_token_ids for t in response)):
        raise ValueError('embedded_stop_or_control_in_response')
    payload = decode(response)
    if payload != generation.json_payload:
        raise ValueError('source_payload_byte_identity_differs')
    parsed = validate_decision_payload(payload)
    if (not parsed.schema_complete or not parsed.reason_valid
            or parsed.decision != generation.decision or parsed.reason != generation.reason):
        raise ValueError('strict_payload_primary_differs')
    matcher = xgr.GrammarMatcher(capability.compiled_grammar)
    if not all(matcher.accept_token(t) for t in response) or not matcher.is_completed():
        raise ValueError('response_not_full_ordered_grammar')
    return response, removed


def compile_cone_teachers(inputs: BaselineInputs, parent: CompletedBaseline, tokenizer) -> dict:
    """Compile a full verified capability or explicit coverage-unavailable report.

    Directly constructed capability objects are trusted caller inputs, matching
    the parent API. The CLI always uses strict raw filesystem loaders instead.
    Global binding mismatches raise. Per-cell invalid source responses preserve
    planned coverage and suppress all accepted teachers.
    """
    if type(inputs) is not BaselineInputs or not isinstance(parent, CompletedBaseline):
        raise ValueError('require approved inputs and completed baseline capabilities')
    rebuilt = build_baseline_plan(inputs, parent.plan.identity)
    if rebuilt.to_json() != parent.plan.to_json() or len(parent.rows) != 2012:
        raise ValueError('parent must match the full approved 2012-key baseline')
    if {r.key for r in parent.rows} != set(parent.plan.keys):
        raise ValueError('parent full row coverage differs')
    roles = inputs.roles['assignments']
    fit = sorted(t for t, role in roles.items() if role == 'fit')
    if len(fit) != 302:
        raise ValueError('expected 302 authoritative fit tickers')
    parent_metadata = parent.metadata
    bindings = parent_metadata.get('original_metadata', parent_metadata)['bindings']
    gp = bindings['generation_policy']
    manifest = dict(kind='stance_cone_teacher_pack_v1', adaptation=ADAPTATION,
        parent_sha256=parent.parent_sha256, inputs_manifest_sha256=inputs.manifest_sha256,
        roles_sha256=inputs.roles_sha256, plan_hash=parent.plan.plan_hash,
        planned_count=906, accepted=False, training_completed=False,
        research_eligible=False, purpose_counts=dict.fromkeys(('addition', 'ablation', 'retain'), 0))
    sources = dict(parent_sha256=parent.parent_sha256, parent_file_sha256=parent.file_sha256,
                   parent_metadata=parent.metadata, selected_records=[])
    review = dict(adaptation=ADAPTATION, review_kind='machine_contract_review_not_human_annotation',
        faithfulness_label=False, target_matches_input_evidence_claim=False,
        full_response_policy='exact_tokens_and_HF_backend_bytes;terminal_declared_EOS_only', cells=[])
    pack = dict(manifest=manifest, sources=sources, review=review, teachers=[], unavailable=[])
    if gp['policy']['channel_policy'] != 'plain_json':
        manifest['status'] = 'unsupported_harmony_full_channel_teacher_policy'
        pack['unavailable'] = [dict(ticker=t, purpose=p, reason=manifest['status'])
                               for t in fit for p, *_ in PAIRINGS]
        review['source_sha256'] = sha256_json(sources)
        manifest.update(source_sha256=sha256_json(sources), review_sha256=sha256_json(review),
                        unavailable_count=906)
        return pack
    reference = parent.generation_for(parent.plan.keys[0]).provenance
    capability = compile_decision_grammar(tokenizer, reference['head_vocab_size'], reference['stop_token_ids'])
    _binding(capability, reference, gp)
    if capability.schema_sha256 != parent.plan.identity.schema_sha256:
        raise ValueError('parent schema binding differs')
    decoded_vocab = _check_capability(capability)
    wrapper_record = bindings['template']['actual_wrapper_record']
    wrapper = WrapperPolicy(**{k: v for k, v in wrapper_record.items() if k != 'tokenizer_chat_template'})
    members = {m.ticker: m for m in inputs.members}
    keys = {(k.ticker, k.condition): k for k in parent.plan.keys}
    if len(keys) != 2012:
        raise ValueError('one fixed trial required per ticker/condition')
    prepared = {}
    for ticker in fit:
        for condition in ('++', '--', '+-'):
            key = keys[ticker, condition]
            g = parent.generation_for(key)
            policy = (dict(policy=parent.row_policy_for(key),
                           hf_controls=g.provenance['hf_controls'])
                      if hasattr(parent, 'row_policy_for') else gp)
            # Merged consumers must bind each source's effective policy, not just the original.
            _binding(capability, g.provenance, policy)
            prompt = render_decision_prompt(tokenizer, members[ticker],
                inputs.pair_for(ticker, condition, key.trial_id), wrapper_policy=wrapper)
            if (prompt.wrapper_policy != wrapper_record
                    or prompt.template_sha256 != bindings['template']['body_template_sha256']
                    or prompt.schema_sha256 != capability.schema_sha256):
                raise ValueError('teacher prompt wrapper/template/schema binding differs')
            source = (parent.row_source_for(key) if hasattr(parent, 'row_source_for') else
                      dict(parent_sha256=parent.parent_sha256, key=key.to_dict()))
            sources['selected_records'].append(dict(key=key.to_dict(), source=source,
                policy=policy, generation=g.to_dict(), prompt_token_ids=list(prompt.inference_token_ids),
                prompt_token_sha256=sha256_json(list(prompt.inference_token_ids)),
                rendered_prompt_sha256=sha256_bytes(prompt.rendered_text.encode('utf8'))))
            prepared[ticker, condition] = (key, g, prompt)
    candidates = []
    for ticker in fit:
        for purpose, input_condition, target_condition, target_class in PAIRINGS:
            input_key, _, prompt = prepared[ticker, input_condition]
            target_key, g, _ = prepared[ticker, target_condition]
            cell = dict(ticker=ticker, role='fit', purpose=purpose, input_key=input_key.to_dict(),
                        target_key=target_key.to_dict(), target_decision=g.decision)
            try:
                response, removed = _response(g, tokenizer, capability, decoded_vocab, target_class)
            except (ValueError, UnicodeError) as exc:
                pack['unavailable'].append(cell | dict(reason=str(exc)))
                review['cells'].append(cell | dict(available=False, reason=str(exc)))
                continue
            prompt_ids = prompt.inference_token_ids
            review['cells'].append(cell | dict(available=True, prompt_token_count=len(prompt_ids),
                response_token_count=len(response), response_token_sha256=sha256_json(list(response)),
                response_bytes_sha256=sha256_bytes(g.json_payload.encode('utf8')),
                removed_terminal_stop_ids=list(removed), source_finish_reason=g.finish_reason))
            candidates.append((ticker, purpose, prompt_ids + response,
                               (False,) * len(prompt_ids) + (True,) * len(response), g.decision))
    source_hash = sha256_json(sources)
    review['source_sha256'] = source_hash
    review_hash = sha256_json(review)
    manifest.update(source_sha256=source_hash, review_sha256=review_hash,
                    unavailable_count=len(pack['unavailable']))
    if not pack['unavailable']:
        pack['teachers'] = [json.loads(canonical_json_bytes(asdict(TeacherResponse(t, 'fit', p, tokens, mask, source_hash,
            capability.tokenizer_sha256, capability.schema_sha256, sha256_json(list(tokens)),
            'construction_generated', decision, review_hash))))
            for t, p, tokens, mask, decision in candidates]
        manifest.update(accepted=True, status='accepted_teacher_coverage',
                        purpose_counts=dict(Counter(r['purpose'] for r in pack['teachers'])))
    else:
        manifest['status'] = 'coverage_unavailable'
    return pack


def write_teacher_pack(directory: str | Path, pack: dict, config: dict) -> None:
    """Publish only into a fresh path. Interrupted output is never resumed/reused."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files = {
        'config.json': canonical_json_bytes(config) + b'\n',
        'source_manifest.json': canonical_json_bytes(pack['sources']) + b'\n',
        'review_manifest.json': canonical_json_bytes(pack['review']) + b'\n',
        'teachers.jsonl': b''.join(canonical_json_bytes(r) + b'\n' for r in pack['teachers']),
        'unavailable.json': canonical_json_bytes(pack['unavailable']) + b'\n',
    }
    manifest = pack['manifest'] | dict(file_sha256={name: sha256_bytes(data) for name, data in files.items()})
    files['manifest.json'] = canonical_json_bytes(manifest) + b'\n'
    for name, data in files.items():
        with (directory / name).open('xb') as stream:
            stream.write(data)
