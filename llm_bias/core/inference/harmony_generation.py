"""Factory-bound native no-tool Harmony generation in one greedy HF call.

Only the verified final body reaches xgrammar. Analysis and headers remain
model-generated; CPU engineering acceptance does not certify any checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any
from weakref import WeakKeyDictionary

import torch
from transformers import (
    GenerationConfig as HFGenerationConfig, LogitsProcessor, LogitsProcessorList,
    StoppingCriteria, StoppingCriteriaList,
)
import xgrammar as xgr

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_json
from .harmony_channels import HarmonyBoundaryTracker, HarmonyTokenContract, UnsupportedHarmonyChannel
from .structured_output import (
    CompiledDecisionGrammar, FiniteLegalTokenGuard, NoLegalTokenError,
    StructuredGenerationPolicy, StructuredGenerationResult, UnsupportedTokenizerError,
    VerifiedModelBinding, _COMPILER_POLICY, _GenerationTimeout, _bind_model,
    _check_capability, _compact_error, _declared_tokens, _fast_backend, _hf_controls,
    _lossless_vocab, _tokenizer_identity, _validate_policy, compile_decision_grammar,
    validate_decision_payload,
)


@dataclass(frozen=True, eq=False)
class CompiledHarmonyDecisionGrammar:
    """Process-local factory identity, never reconstructed from provenance."""
    json_capability: CompiledDecisionGrammar
    contract: HarmonyTokenContract
    contract_sha256: str

    def new_processor(self, *, prompt_length: int, deadline: float | None) -> HarmonyDecisionProcessor:
        return HarmonyDecisionProcessor(self, prompt_length=prompt_length, deadline=deadline)


_FACTORY_CAPABILITIES: WeakKeyDictionary = WeakKeyDictionary()


def _check_harmony_capability(capability: CompiledHarmonyDecisionGrammar) -> tuple[bytes, ...]:
    if not isinstance(capability, CompiledHarmonyDecisionGrammar):
        raise ValueError('Harmony capability must be factory-compiled')
    owned = _FACTORY_CAPABILITIES.get(capability)
    if (owned is None or owned[0] is not capability.json_capability
            or owned[1] is not capability.contract or owned[2] != capability.contract_sha256
            or owned[3] != canonical_json_bytes(asdict(capability.contract))):
        raise ValueError('Harmony capability differs from its factory-owned nested identity')
    return _check_capability(capability.json_capability)


def compile_harmony_decision_grammar(
    tokenizer: Any, head_vocab_size: int, contract: HarmonyTokenContract,
) -> CompiledHarmonyDecisionGrammar:
    """Verify pinned IDs, native spellings and independent bytes before hashing."""
    if not isinstance(contract, HarmonyTokenContract):
        raise ValueError('contract must be a HarmonyTokenContract')
    if type(head_vocab_size) is not int or head_vocab_size <= 0:
        raise ValueError('head_vocab_size must be a positive integer')
    vocab_ids = set(tokenizer.get_vocab().values())
    record = asdict(contract)
    contract_ids = tuple(i for value in record.values()
                         for i in (value if isinstance(value, tuple) else (value,)))
    if any(type(i) is not int or not 0 <= i < head_vocab_size or i not in vocab_ids for i in contract_ids):
        raise ValueError('every contract ID must be inside tokenizer vocabulary and head bounds')
    backend = _fast_backend(tokenizer)
    added, declared = _declared_tokens(tokenizer, backend, head_vocab_size)
    controls = {contract.message_end_id, contract.final_stop_id,
                contract.initial_analysis_header_ids[0], contract.initial_analysis_header_ids[-1],
                contract.assistant_restart_ids[0], *contract.forbidden_control_ids}
    if not controls.issubset(declared):
        raise ValueError('contract controls must be declared special vocabulary IDs')
    nested = compile_decision_grammar(tokenizer, head_vocab_size, [contract.final_stop_id])
    # Reuse the accepted adapter, not the plain blocked set (which also includes
    # safe special controls). Declaring a special never excuses unsafe bytes.
    decoded_vocab, mismatched = _lossless_vocab(
        tokenizer, nested.compiled_grammar.tokenizer_info, added, nested.byte_policy,
    )
    HarmonyBoundaryTracker(contract, special_token_ids=nested.special_token_ids)
    literals = (
        (contract.prompt_suffix_ids, '<|start|>assistant'),
        (contract.assistant_restart_ids, '<|start|>assistant'),
        (contract.initial_analysis_header_ids, '<|channel|>analysis<|message|>'),
        (contract.initial_final_header_ids, '<|channel|>final<|message|>'),
        ((contract.message_end_id,), '<|end|>'),
        ((contract.final_stop_id,), '<|return|>'),
    )
    for tokens, literal in literals:
        if set(tokens) & mismatched:
            raise UnsupportedTokenizerError('admitted Harmony token bytes disagree with native/backend bytes')
        exact = b''.join(decoded_vocab[i] for i in tokens)
        native = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if exact != literal.encode('utf-8') or native != literal:
            raise UnsupportedTokenizerError('Harmony suffix/header/control literal bytes do not match')
    contract_hash = sha256_json(record)
    capability = CompiledHarmonyDecisionGrammar(nested, contract, contract_hash)
    _FACTORY_CAPABILITIES[capability] = (nested, contract, contract_hash, canonical_json_bytes(record))
    return capability


class HarmonyDecisionProcessor(LogitsProcessor):
    """Single-use tracker plus matcher; raw observations survive protocol errors."""
    def __init__(self, capability: CompiledHarmonyDecisionGrammar, *, prompt_length: int,
                 deadline: float | None):
        _check_harmony_capability(capability)
        if type(prompt_length) is not int or prompt_length <= 0:
            raise ValueError('prompt_length must be a positive integer')
        if deadline is not None and (isinstance(deadline, bool)
                or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
            raise ValueError('deadline must be None or a finite real nonbool number')
        self.capability = capability
        self.prompt_length = prompt_length
        self.deadline = deadline
        nested = capability.json_capability
        self._boundary_tracker = HarmonyBoundaryTracker(capability.contract, special_token_ids=nested.special_token_ids)
        self.matcher = xgr.GrammarMatcher(nested.compiled_grammar)
        self.bitmask = xgr.allocate_token_bitmask(1, nested.head_vocab_size)
        self.generated_token_ids: tuple[int, ...] = ()
        self._unsafe_ids = frozenset(nested.blocked_token_ids) - set(nested.special_token_ids)
        self._analysis_blocked_ids = tuple(sorted(
            (set(nested.blocked_token_ids) - {capability.contract.message_end_id})
            | {capability.contract.final_stop_id}))

    @property
    def boundary_tracker(self) -> HarmonyBoundaryTracker:
        return self._boundary_tracker

    def observe(self, input_ids: torch.Tensor) -> None:
        if (not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2
                or input_ids.shape[0] != 1 or input_ids.shape[1] < self.prompt_length
                or input_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Harmony observation requires batch-one integer prompt plus continuation')
        tokens = tuple(input_ids[0, self.prompt_length:].tolist())
        old = self.generated_token_ids
        if len(tokens) < len(old) or tokens[:len(old)] != old:
            raise ValueError('generation changed a previously observed continuation')
        self.generated_token_ids = tokens
        nested = self.capability.json_capability
        tracker = self.boundary_tracker
        for index in range(len(tracker.observed_token_ids), len(tokens)):
            token = tokens[index]
            if not 0 <= token < nested.head_vocab_size:
                raise ValueError('generated token ID outside the bound head range')
            if token in self._unsafe_ids:
                raise ValueError('generated byte-mismatched/gap/padded token ID')
            phase = tracker.phase
            if phase == 'final':
                if token == self.capability.contract.final_stop_id:
                    # The boundary alone cannot certify JSON completion.
                    if not self.matcher.is_completed():
                        raise UnsupportedHarmonyChannel('premature final-stop before complete decision grammar')
                elif token in nested.special_token_ids:
                    raise UnsupportedHarmonyChannel('unsupported special token in final body')
                if not self.matcher.accept_token(token):
                    raise ValueError(f'generated final token {token} rejected by decision grammar')
            tracker.observe(tokens[:index + 1])

    def check_timeout(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _GenerationTimeout('Harmony generation exceeded monotonic deadline')

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.observe(input_ids)
        self.check_timeout()
        nested = self.capability.json_capability
        if scores.ndim != 2 or scores.shape != (1, nested.head_vocab_size):
            raise ValueError('logits head does not match the compiled head vocabulary')
        tracker = self.boundary_tracker
        allowed = tracker.allowed_next_ids()
        if allowed is not None:
            blocked = torch.ones(nested.head_vocab_size, dtype=torch.bool, device=scores.device)
            blocked[list(allowed)] = False
            scores.masked_fill_(blocked.unsqueeze(0), -float('inf'))
        elif tracker.phase == 'analysis':
            scores[:, list(self._analysis_blocked_ids)] = -float('inf')
        else:
            self.matcher.fill_next_token_bitmask(self.bitmask)
            xgr.apply_token_bitmask_inplace(
                scores, self.bitmask.to(scores.device), vocab_size=nested.head_vocab_size,
                backend='cpu' if scores.device.type == 'cpu' else 'auto',
            )
            scores[:, list(nested.blocked_token_ids)] = -float('inf')
        return scores


class _ObserveAndTimeout(StoppingCriteria):
    def __init__(self, processor: HarmonyDecisionProcessor):
        self.processor = processor

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.processor.observe(input_ids)
        self.processor.check_timeout()
        return torch.tensor([self.processor.boundary_tracker.is_terminated], device=input_ids.device)


def generate_harmony_structured(
    model: Any, tokenizer: Any, prompt_ids: torch.Tensor,
    capability: CompiledHarmonyDecisionGrammar, *, policy: StructuredGenerationPolicy,
    verified_binding: VerifiedModelBinding | None = None,
) -> StructuredGenerationResult:
    """One native HF call; external scoped interventions cover the whole turn."""
    decoded_vocab = _check_harmony_capability(capability)
    nested, contract = capability.json_capability, capability.contract
    _validate_policy(policy, nested.head_vocab_size)
    if policy.channel_policy != 'harmony_no_tools':
        raise ValueError('Harmony generation requires harmony_no_tools policy')
    if (not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2
            or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0):
        raise ValueError('Harmony generation requires a nonempty batch-one prompt')
    if (prompt_ids.dtype not in (torch.int32, torch.int64)
            or (prompt_ids < 0).any() or (prompt_ids >= nested.head_vocab_size).any()):
        raise ValueError('prompt IDs must be integer tokens inside the head range')
    suffix = contract.prompt_suffix_ids
    if tuple(prompt_ids[0, -len(suffix):].tolist()) != suffix:
        raise ValueError('prompt must end exactly in bound Harmony prompt_suffix_ids')
    if _tokenizer_identity(tokenizer) != nested.tokenizer_sha256:
        raise ValueError('tokenizer identity differs from compiled Harmony grammar')
    binding = _bind_model(model, nested, verified_binding)
    start = time.monotonic()
    controls = _hf_controls(policy, nested.stop_token_ids)
    # Freeze identity and controls before an arbitrary generate can mutate kwargs.
    provenance_bytes = canonical_json_bytes({
        'backend': 'xgrammar', 'backend_version': nested.backend_version,
        'model_binding': binding, 'byte_policy': nested.byte_policy,
        'schema_sha256': nested.schema_sha256, 'schema_bytes_sha256': nested.schema_bytes_sha256,
        'tokenizer_sha256': nested.tokenizer_sha256, 'tokenizer_info_sha256': nested.tokenizer_info_sha256,
        'head_vocab_size': nested.head_vocab_size, 'stop_token_ids': list(nested.stop_token_ids),
        'compiler_policy': dict(_COMPILER_POLICY),
        'grammar_correction': 'xgrammar-0.2.8-minLength-json-escapes-v1',
        'grammar_sha256': nested.grammar_sha256,
        'mask_backend': 'cpu' if prompt_ids.device.type == 'cpu' else 'auto',
        'generation_policy': asdict(policy), 'hf_controls': controls,
        'generation_policy_sha256': sha256_json({'policy': asdict(policy), 'hf_controls': controls}),
        'channel_contract': asdict(contract), 'channel_contract_sha256': capability.contract_sha256,
        'channel_policy_sha256': sha256_json({'policy': 'harmony_no_tools', 'contract_sha256': capability.contract_sha256}),
    })
    processor = capability.new_processor(prompt_length=prompt_ids.shape[1], deadline=start + policy.timeout_seconds)
    failure = error = None
    tokens: tuple[int, ...] = ()
    try:
        with torch.no_grad():
            output = model.hf_model.generate(
                prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                generation_config=HFGenerationConfig(**controls), **controls,
                logits_processor=LogitsProcessorList([processor, FiniteLegalTokenGuard()]),
                stopping_criteria=StoppingCriteriaList([_ObserveAndTimeout(processor)]),
            )
            sequences = getattr(output, 'sequences', output)
            if (not isinstance(sequences, torch.Tensor) or sequences.ndim != 2
                    or sequences.shape[0] != 1 or sequences.shape[1] < prompt_ids.shape[1]
                    or sequences.dtype not in (torch.int32, torch.int64)):
                raise ValueError('generate must return unchanged prompt plus integer continuation')
            tokens = tuple(sequences[0, prompt_ids.shape[1]:].tolist())
            if not torch.equal(sequences[:, :prompt_ids.shape[1]], prompt_ids):
                raise ValueError('generate changed the prompt prefix')
            if len(tokens) > policy.max_new_tokens:
                raise ValueError('generate returned more continuation tokens than max_new_tokens')
            processor.observe(sequences)
            processor.check_timeout()
            if len(tokens) < policy.max_new_tokens and not processor.matcher.is_completed():
                raise ValueError('generate terminated early without complete final schema or explained stop')
        finish = 'eos' if processor.boundary_tracker.is_terminated else 'token_budget'
    except _GenerationTimeout as exc:
        failure, finish, error = 'timeout', 'timeout', _compact_error(exc)
    except NoLegalTokenError as exc:
        failure, finish, error = 'no_legal_token', 'no_legal_token', _compact_error(exc)
    except UnsupportedHarmonyChannel as exc:
        failure, finish, error = 'unsupported_channel', 'unsupported', _compact_error(exc)
    except Exception as exc:
        failure, finish, error = 'exception', 'exception', _compact_error(exc)
    if not tokens:
        tokens = processor.generated_token_ids
    tracker = processor.boundary_tracker
    final_start, final_end = tracker.final_content_start, tracker.final_content_end
    payload_ids = () if final_start is None else tokens[final_start:final_end]
    text = payload = ''
    decode_error = None
    try:
        text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        payload = tokenizer.decode(payload_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for ids, decoded in ((tokens, text), (payload_ids, payload)):
            if any(type(i) is not int or not 0 <= i < len(decoded_vocab) for i in ids):
                raise ValueError('cannot decode IDs outside the bound vocabulary')
            exact = b''.join(decoded_vocab[i] for i in ids).decode('utf-8', errors='strict')
            decoded.encode('utf-8', errors='strict')
            if exact != decoded:
                raise UnsupportedTokenizerError('HF decoded text differs from exact backend token bytes')
    except Exception as exc:
        decode_error = _compact_error(exc)
        for name, decoded in (('text', text), ('payload', payload)):
            try:
                decoded.encode('utf-8', errors='strict')
            except (UnicodeError, AttributeError):
                if name == 'text':
                    text = ''
                else:
                    payload = ''
        if failure is None:
            failure, finish = (('unsupported_tokenizer', 'unsupported') if isinstance(exc, UnsupportedTokenizerError)
                               else ('exception', 'exception'))
            error = decode_error
    parsed = validate_decision_payload(payload if decode_error is None else '')
    if failure is None:
        if parsed.failure_type is None:
            if finish != 'eos':
                finish = 'schema_complete'
        elif len(tokens) == policy.max_new_tokens and not parsed.schema_complete:
            failure, finish = 'truncated', 'token_budget'
        else:
            failure = parsed.failure_type
            if parsed.schema_complete and finish != 'eos':
                finish = 'schema_complete'
    provenance = json.loads(provenance_bytes)
    provenance.update(final_content_start=final_start, final_content_end=final_end,
                      analysis_segments=[list(span) for span in tracker.analysis_segments],
                      generated_token_count=len(tokens), final_token_count=len(payload_ids))
    return StructuredGenerationResult(
        text, tokens, sha256_json(tokens), payload,
        parsed.decision if failure is None else None, parsed.reason if failure is None else None,
        parsed.decision_complete, parsed.schema_complete, parsed.reason_valid,
        finish, failure, time.monotonic() - start, canonical_json_bytes(provenance), error, decode_error,
    )


__all__ = ['CompiledHarmonyDecisionGrammar', 'HarmonyDecisionProcessor',
           'compile_harmony_decision_grammar', 'generate_harmony_structured']
