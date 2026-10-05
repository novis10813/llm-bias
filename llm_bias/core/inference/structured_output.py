"""Batch-one plain-JSON greedy decoding, independent of historical parsers.

The pinned xgrammar compiler constrains syntax; the strict validator gates the
primary decision. This CPU-tested capability does not certify real checkpoints
or Harmony/other channel protocols. All generation policies are explicit.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib.metadata import version
import json
import math
import re
from pathlib import Path
import time
from typing import Any, Literal
from weakref import WeakKeyDictionary

import torch
from transformers import (
    GenerationConfig as HFGenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
)
import xgrammar as xgr

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_json
from .interventions import ResidualTransform, residual_interventions

CANONICAL_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / 'configs/concept-cone-steering/rebuild-v1/decision.schema.json'
)
FinishReason = Literal['eos', 'schema_complete', 'token_budget', 'timeout',
                       'exception', 'no_legal_token', 'unsupported']
FailureType = Literal['truncated', 'timeout', 'exception', 'no_legal_token',
                      'invalid_json', 'invalid_schema', 'invalid_reason', 'unsupported_channel',
                      'unsupported_tokenizer']
_COMPILER_POLICY = {'strict_mode': True, 'any_order': False, 'any_whitespace': True}
# Exact canonical-schema output of the pinned compiler, not a substring patch.
_NATIVE_GRAMMAR_SHA256 = '263f38e55c51d02c97f81ff7078269ff2d344728f61bd96302fb755554fb99e8'


class UnsupportedTokenizerError(ValueError):
    """Lossless backend/HF decoded bytes cannot be established for this tokenizer."""


class UnsupportedModelBindingError(ValueError):
    """Missing model declarations require an explicit verified-binding attestation."""


class _DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f'non-JSON constant: {value}')


def _strict_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


@lru_cache(maxsize=None)
def _schema_record(path: Path) -> tuple[bytes, str, str]:
    raw = path.read_bytes()
    try:
        schema = _strict_json(raw.decode('utf-8'))
        valid = (
            isinstance(schema, dict)
            and set(schema) == {'type', 'properties', 'required', 'additionalProperties'}
            and schema['type'] == 'object'
            and isinstance(schema['properties'], dict)
            and list(schema['properties']) == ['decision', 'reason']
            and canonical_json_bytes(schema['properties']['decision'])
            == canonical_json_bytes({'type': 'string', 'enum': ['buy', 'sell']})
            and canonical_json_bytes(schema['properties']['reason'])
            == canonical_json_bytes({'type': 'string', 'minLength': 1})
            and schema['required'] == ['decision', 'reason']
            and schema['additionalProperties'] is False
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('schema must exactly match the canonical decision schema') from exc
    if not valid:
        raise ValueError('schema must exactly match the canonical decision schema')
    # Same semantic identity as the renderer, with a separate exact-byte hash.
    return raw, sha256_bytes(raw), sha256_json({'schema': schema, 'key_order': ['decision', 'reason']})


def load_decision_schema(path: str | Path = CANONICAL_SCHEMA_PATH) -> dict[str, Any]:
    """Read/validate once per resolved path; return a fresh, unshared JSON copy."""
    raw, _, _ = _schema_record(Path(path).resolve())
    return json.loads(raw)


def _tokenizer_identity(tokenizer: Any) -> str:
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    return sha256_json({
        'class': f'{type(tokenizer).__module__}.{type(tokenizer).__qualname__}',
        'vocab': tokenizer.get_vocab(),
        'backend': backend.to_str() if backend is not None else None,
        'special_tokens_map': tokenizer.special_tokens_map,
        'special_token_ids': list(tokenizer.all_special_ids),
        'chat_template': getattr(tokenizer, 'chat_template', None),
        'name_or_path': getattr(tokenizer, 'name_or_path', None),
    })


@dataclass(frozen=True)
class StructuredGenerationPolicy:
    """Required, explicit controls; no slug-derived channel policy or sampling."""
    max_new_tokens: int
    use_cache: bool
    pad_token_id: int
    timeout_seconds: float
    channel_policy: Literal['plain_json', 'unsupported', 'harmony_no_tools']


@dataclass(frozen=True)
class DecisionPayloadValidation:
    decision: Literal['buy', 'sell'] | None
    reason: str | None
    decision_complete: bool
    schema_complete: bool
    reason_valid: bool
    failure_type: FailureType | None


def validate_decision_payload(payload: str) -> DecisionPayloadValidation:
    """Parse the entire ordered object. No regex, fences, or partial salvage.

    A complete legal enum can be diagnostic on a complete invalid-reason object,
    but no failing payload supplies a primary decision or reason. Partial text
    never establishes decision completion.
    """
    load_decision_schema()
    try:
        obj = _strict_json(payload)
    except _DuplicateKeyError:
        return DecisionPayloadValidation(None, None, False, False, False, 'invalid_schema')
    except (ValueError, TypeError, RecursionError):
        return DecisionPayloadValidation(None, None, False, False, False, 'invalid_json')
    if (not isinstance(obj, dict) or list(obj) != ['decision', 'reason']
            or obj['decision'] not in ('buy', 'sell') or not isinstance(obj['reason'], str)):
        return DecisionPayloadValidation(None, None, False, False, False, 'invalid_schema')
    reason = obj['reason']
    if not reason.strip() or any(0xd800 <= ord(char) <= 0xdfff for char in reason):
        return DecisionPayloadValidation(None, None, True, len(reason) >= 1, False, 'invalid_reason')
    return DecisionPayloadValidation(obj['decision'], reason, True, True, True, None)


@dataclass(frozen=True)
class StructuredGenerationResult:
    """Tensor-free generated-output record; failure always clears primary values.

    generated_text preserves special tokens; json_payload removes only recognized
    stop IDs at the terminal suffix. elapsed_seconds uses a monotonic clock.
    provenance exports a fresh copy of immutable canonical JSON bytes containing
    schema/backend/tokenizer/head/stops/cache/policy identity. Use to_dict() for
    persistence (dataclasses.asdict includes private bytes). decode_error defaults
    to None and records a compact secondary text/byte decoding error without
    replacing the first execution error_message or finish/failure classification.
    No logits, hidden states, residuals, gradients, or KV cache are stored.
    """
    generated_text: str
    generated_token_ids: tuple[int, ...]
    generated_token_sha256: str
    json_payload: str
    decision: Literal['buy', 'sell'] | None
    reason: str | None
    decision_complete: bool
    schema_complete: bool
    reason_valid: bool
    finish_reason: FinishReason
    failure_type: FailureType | None
    elapsed_seconds: float
    _provenance_bytes: bytes = field(repr=False)
    error_message: str | None = None
    decode_error: str | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        """A defensive export of privately stored immutable canonical JSON bytes."""
        return json.loads(self._provenance_bytes)

    def to_dict(self) -> dict[str, Any]:
        """Tensor-free, JSON-persistable export; nested provenance is never shared."""
        record = asdict(self)
        record.pop('_provenance_bytes')
        record['provenance'] = self.provenance
        return record


@dataclass(frozen=True, eq=False)
class CompiledDecisionGrammar:
    """Reusable compiled capability, bound to tokenizer/schema/backend identity.

    new_processor always allocates a new matcher and full-head bitmask. Callers
    must not reuse a mutable processor across generation calls.
    """
    compiled_grammar: xgr.CompiledGrammar
    head_vocab_size: int
    stop_token_ids: tuple[int, ...]
    blocked_token_ids: tuple[int, ...]
    special_token_ids: tuple[int, ...]
    schema_sha256: str
    schema_bytes_sha256: str
    tokenizer_sha256: str
    tokenizer_info_sha256: str
    backend_version: str
    grammar_sha256: str
    byte_policy: str

    def new_processor(self, *, prompt_length: int, deadline: float | None) -> DecisionGrammarProcessor:
        return DecisionGrammarProcessor(self, prompt_length=prompt_length, deadline=deadline)


def _correct_reason_string_rule(compiler: xgr.GrammarCompiler, compiled: xgr.CompiledGrammar) -> xgr.CompiledGrammar:
    """Correct the pinned minLength rule, preserving compiler-enforced key order.

    xgrammar 0.2.8 converts minLength=1 to a raw-character-only rule: it rejects
    JSON escapes and admits C0 bytes. Replace ONLY those two known reason rules
    with RFC 8259 characters/escapes, still requiring at least one character.
    Fail closed if the pinned compiler's output differs; never weaken the schema.
    This correction and the final grammar hash are explicit result provenance.
    """
    grammar = str(compiled.grammar)
    if (version('xgrammar') != '0.2.8'
            or sha256_bytes(grammar.encode('utf-8')) != _NATIVE_GRAMMAR_SHA256):
        raise ValueError('unsupported xgrammar minLength conversion; cannot safely compile canonical schema')
    old_tail = r'root_prop_1_1 ::= ("" | ([^\"\\\r\n] root_prop_1_1))'
    old_first = r'root_prop_1_2 ::= (([^\"\\\r\n] root_prop_1_1)) (=("\""))'
    if grammar.count(old_tail) != 1 or grammar.count(old_first) != 1:
        raise ValueError('unsupported xgrammar minLength conversion; cannot safely compile canonical schema')
    grammar = grammar.replace(old_tail, 'root_prop_1_1 ::= ("" | (decision_reason_character root_prop_1_1))')
    grammar = grammar.replace(old_first, r'root_prop_1_2 ::= ((decision_reason_character root_prop_1_1)) (=("\""))')
    grammar += r'''
decision_reason_character ::= ([^\"\\\0-\x1f] | ("\\" decision_reason_escape))
decision_reason_escape ::= ([\"\\/bfnrt] | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))
'''
    return compiler.compile_grammar(xgr.Grammar.from_ebnf(grammar))


def compile_decision_grammar(
    tokenizer: Any, head_vocab_size: int, stop_token_ids: Sequence[int],
) -> CompiledDecisionGrammar:
    """Compile exactly the canonical ordered schema with installed xgrammar 0.2.8.

    The positive head size must cover every tokenizer ID. Stop IDs must be a
    nonempty unique sequence of known special/EOS vocab IDs, not padded IDs.
    Byte support is restricted to HF fast BPE with plain ByteLevel or exactly
    Replace(String("▁"), " ")/ByteFallback/Fuse, matching xgrammar vocabulary
    type and no prefix adjustment. All byte disagreements are blocked.
    """
    if type(head_vocab_size) is not int or head_vocab_size <= 0:
        raise ValueError('head_vocab_size must be a positive integer')
    vocab_ids = tuple(tokenizer.get_vocab().values())
    if not vocab_ids or any(type(i) is not int or not 0 <= i < head_vocab_size for i in vocab_ids):
        raise ValueError('head_vocab_size must cover all tokenizer IDs')
    if not isinstance(stop_token_ids, Sequence) or isinstance(stop_token_ids, (str, bytes)):
        raise ValueError('stop_token_ids must be a nonempty integer sequence')
    stops = tuple(stop_token_ids)
    if (not stops or any(type(i) is not int or not 0 <= i < head_vocab_size for i in stops)
            or len(set(stops)) != len(stops)):
        raise ValueError('stop_token_ids must be unique integers inside the head range')
    backend = _fast_backend(tokenizer)
    byte_policy = _decoder_byte_policy(backend)
    added, declared_specials = _declared_tokens(tokenizer, backend, head_vocab_size)
    if any(i not in vocab_ids or i not in declared_specials for i in stops):
        raise ValueError('stop IDs must be known special/EOS vocabulary tokens, never ordinary or padded')
    backend_version = version('xgrammar')
    if backend_version != '0.2.8':
        raise ValueError('this capability requires xgrammar==0.2.8')
    _, bytes_hash, schema_hash = _schema_record(CANONICAL_SCHEMA_PATH.resolve())
    try:
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=head_vocab_size, stop_token_ids=list(stops),
        )
    except Exception as exc:
        raise UnsupportedTokenizerError('backend tokenizer bytes could not be established') from exc
    decoded_vocab, mismatched = _lossless_vocab(tokenizer, info, added, byte_policy)
    if set(stops) & mismatched:
        raise UnsupportedTokenizerError('stop token bytes disagree with the backend tokenizer')
    compiler = xgr.GrammarCompiler(info)
    compiled = compiler.compile_json_schema(load_decision_schema(), **_COMPILER_POLICY)
    compiled = _correct_reason_string_rule(compiler, compiled)
    specials = set(info.special_token_ids) | declared_specials
    # Full-head coverage, including tokenizer gaps/padded IDs. Explicitly mask
    # declared special tokens as well as backend-detected control/reserved tokens.
    blocked = (specials | mismatched | (set(range(head_vocab_size)) - set(vocab_ids))) - set(stops)
    capability = CompiledDecisionGrammar(
        compiled, head_vocab_size, stops, tuple(sorted(blocked)), tuple(sorted(specials)),
        schema_hash, bytes_hash, _tokenizer_identity(tokenizer),
        sha256_json(json.loads(info.serialize_json())), backend_version,
        sha256_bytes(str(compiled.grammar).encode('utf-8')), byte_policy,
    )
    _FACTORY_CAPABILITIES[capability] = (_capability_metadata(capability), compiled, decoded_vocab)
    return capability


def _fast_backend(tokenizer: Any) -> dict[str, Any]:
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise UnsupportedTokenizerError('lossless bytes require a tested HF fast BPE tokenizer')
    try:
        backend = _strict_json(tokenizer.backend_tokenizer.to_str())
    except (ValueError, TypeError) as exc:
        raise UnsupportedTokenizerError('malformed tokenizer backend descriptor') from exc
    if not isinstance(backend, dict):
        raise UnsupportedTokenizerError('malformed tokenizer backend descriptor')
    return backend


def _decoder_byte_policy(backend: dict[str, Any]) -> str:
    """Admit native descriptors, not decoder substrings or generic pipelines."""
    decoder = backend.get('decoder')
    model = backend.get('model')
    bytelevel = (
        isinstance(decoder, dict)
        and set(decoder) == {'type', 'add_prefix_space', 'trim_offsets', 'use_regex'}
        and decoder['type'] == 'ByteLevel'
        and all(type(decoder[key]) is bool for key in ('add_prefix_space', 'trim_offsets', 'use_regex'))
    )
    bytefallback = decoder == {
        'type': 'Sequence', 'decoders': [
            {'type': 'Replace', 'pattern': {'String': '▁'}, 'content': ' '},
            {'type': 'ByteFallback'}, {'type': 'Fuse'},
        ],
    }
    if (not isinstance(model, dict) or model.get('type') != 'BPE'
            or not (bytelevel or bytefallback)
            or 'add_prefix_space":true' in json.dumps(backend.get('pre_tokenizer'), separators=(',', ':'))):
        raise UnsupportedTokenizerError(
            'unsupported lossless decoder: require tested BPE/ByteLevel or '
            'Replace/ByteFallback/Fuse without prefix adjustment')
    return ('hf-fast-bpe-bytelevel-strict-utf8-v1' if bytelevel
            else 'hf-fast-bpe-bytefallback-strict-utf8-v1')


def _declared_tokens(tokenizer: Any, backend: dict[str, Any],
                     head_vocab_size: int) -> tuple[dict[int, str], set[int]]:
    """Validate backend added records before using their literal bytes/special IDs."""
    vocab = tokenizer.get_vocab()
    vocab_ids = set(vocab.values())
    records = backend.get('added_tokens', [])
    if not isinstance(records, list):
        raise UnsupportedTokenizerError('malformed backend added-token declarations')
    added: dict[int, str] = {}
    contents: set[str] = set()
    specials = set()
    for record in records:
        if not isinstance(record, dict):
            raise UnsupportedTokenizerError('malformed backend added-token declaration')
        token_id, content, special = record.get('id'), record.get('content'), record.get('special')
        if (type(token_id) is not int or not 0 <= token_id < head_vocab_size
                or not isinstance(content, str) or type(special) is not bool
                or vocab.get(content) != token_id or token_id in added or content in contents):
            raise UnsupportedTokenizerError('malformed or conflicting backend added-token declaration')
        added[token_id] = content
        contents.add(content)
        if special:
            specials.add(token_id)
    for token_id in tokenizer.all_special_ids:
        if type(token_id) is not int or not 0 <= token_id < head_vocab_size or token_id not in vocab_ids:
            raise UnsupportedTokenizerError('malformed named HF special-token declaration')
        specials.add(token_id)
    return added, specials


def _lossless_vocab(tokenizer: Any, info: xgr.TokenizerInfo,
                    added: dict[int, str], byte_policy: str) -> tuple[tuple[bytes, ...], set[int]]:
    """Compare independent native bytes to xgrammar; never repair unsafe bytes.

    Native ByteFallback accepts two hex digits (either case) and a plus sign
    followed by one hex digit. Added tokens are scrutinized as literal content,
    even when the native decoder interprets their spelling as ordinary tokens.
    NUL truncation and all other backend disagreements remain blocked. Runtime
    strict UTF8 and full-sequence HF equality are additional mandatory gates.
    """
    bytelevel = byte_policy == 'hf-fast-bpe-bytelevel-strict-utf8-v1'
    expected_type = xgr.VocabType.BYTE_LEVEL if bytelevel else xgr.VocabType.BYTE_FALLBACK
    if info.vocab_type != expected_type or info.add_prefix_space:
        raise UnsupportedTokenizerError('unsupported lossless decoder: vocabulary type mismatch or prefix adjustment')
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = list(byte_values)
    missing_bytes = [byte for byte in range(256) if byte not in byte_values]
    for offset, byte in enumerate(missing_bytes):
        byte_values.append(byte)
        characters.append(256 + offset)
    byte_map = {chr(char): bytes([byte]) for char, byte in zip(characters, byte_values)}
    decoded = info.decoded_vocab
    mismatched = set()
    for token, token_id in tokenizer.get_vocab().items():
        try:
            if token_id in added:
                exact = added[token_id].encode('utf-8')
                native = tokenizer.decode([token_id], skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
                if native != added[token_id]:
                    mismatched.add(token_id)
            elif bytelevel:
                exact = b''.join(byte_map[char] if char in byte_map else char.encode('utf-8') for char in token)
            elif re.fullmatch(r'<0x(?:[0-9A-Fa-f]{2}|\+[0-9A-Fa-f])>', token):
                exact = bytes([int(token[3:-1], 16)])
            else:
                exact = token.replace('▁', ' ').encode('utf-8')
            if exact != decoded[token_id]:
                mismatched.add(token_id)
        except (UnicodeError, ValueError):
            mismatched.add(token_id)
    return tuple(decoded), mismatched


# Expected identities are factory-owned, never supplied by capability metadata.
# Weak keys do not keep unused compiled grammars (or any model tensors) alive.
_FACTORY_CAPABILITIES: WeakKeyDictionary = WeakKeyDictionary()


def _capability_metadata(capability: CompiledDecisionGrammar) -> tuple[Any, ...]:
    return tuple(getattr(capability, name) for name in capability.__dataclass_fields__
                 if name != 'compiled_grammar')


def _check_capability(capability: CompiledDecisionGrammar) -> tuple[bytes, ...]:
    if not isinstance(capability, CompiledDecisionGrammar):
        raise ValueError('capability must be factory-compiled')
    owned = _FACTORY_CAPABILITIES.get(capability)
    if (owned is None or owned[0] != _capability_metadata(capability)
            or owned[1] is not capability.compiled_grammar):
        raise ValueError('compiled capability is not bound to its factory-owned identity')
    compiled = capability.compiled_grammar
    if (version('xgrammar') != capability.backend_version
            or sha256_bytes(str(compiled.grammar).encode('utf-8')) != capability.grammar_sha256
            or sha256_json(json.loads(compiled.tokenizer_info.serialize_json()))
            != capability.tokenizer_info_sha256):
        raise ValueError('actual compiled grammar/tokenizer-info differs from factory capability')
    return owned[2]


@dataclass(frozen=True)
class VerifiedModelBinding:
    """Trusted caller attestation ONLY for unavailable model declarations.

    Supply the exact hf_model object, tokenizer identity from the capability,
    verified output-head size and stops, and a nonempty verification reference.
    This is not automatic checkpoint certification: the caller must actually
    verify these facts. It never overrides conflicting available declarations.
    """
    hf_model: Any = field(repr=False, compare=False)
    tokenizer_sha256: str
    head_vocab_size: int
    stop_token_ids: tuple[int, ...]
    verification_reference: str


def _bind_model(model: Any, capability: CompiledDecisionGrammar,
                verified: VerifiedModelBinding | None) -> dict[str, Any]:
    hf = model.hf_model
    sources: dict[str, list[str]] = {'tokenizer': [], 'head': [], 'stops': []}
    for label, holder in [('model', model), ('hf_model', hf)]:
        attached = getattr(holder, 'tokenizer', None)
        if attached is not None:
            if _tokenizer_identity(attached) != capability.tokenizer_sha256:
                raise ValueError(f'{label} attached tokenizer differs from compiled tokenizer')
            sources['tokenizer'].append(f'{label}.tokenizer')
        head = getattr(holder, 'lm_head', None)
        get_head = getattr(holder, 'get_output_embeddings', None)
        if callable(get_head):
            head = get_head()
        if head is not None:
            weight = getattr(head, 'weight', None)
            size = weight.shape[0] if weight is not None else getattr(head, 'out_features', None)
            if type(size) is not int or size != capability.head_vocab_size:
                raise ValueError(f'{label} output head differs from compiled head vocabulary')
            sources['head'].append(f'{label}.output_head')
        config = getattr(holder, 'config', None)
        configs = [('config', config), ('generation_config', getattr(holder, 'generation_config', None))]
        if config is not None and callable(getattr(config, 'get_text_config', None)):
            configs.append(('text_config', config.get_text_config()))
        for name, declared in configs:
            size = getattr(declared, 'vocab_size', None)
            if size is not None:
                if type(size) is not int or size != capability.head_vocab_size:
                    raise ValueError(f'{label}.{name} head vocabulary differs from compiled head')
                sources['head'].append(f'{label}.{name}.vocab_size')
            stops = getattr(declared, 'eos_token_id', None)
            if stops is not None:
                stops = [stops] if type(stops) is int else stops
                if (not isinstance(stops, (list, tuple)) or not stops
                        or any(type(i) is not int for i in stops)
                        or not set(capability.stop_token_ids).issubset(stops)):
                    raise ValueError(f'{label}.{name} stop declarations conflict with compiled stops')
                sources['stops'].append(f'{label}.{name}.eos_token_id')
    missing = [name for name, declarations in sources.items() if not declarations]
    if verified is not None:
        if (not isinstance(verified, VerifiedModelBinding) or verified.hf_model is not hf
                or verified.tokenizer_sha256 != capability.tokenizer_sha256
                or type(verified.head_vocab_size) is not int
                or verified.head_vocab_size != capability.head_vocab_size
                or verified.stop_token_ids != capability.stop_token_ids
                or not isinstance(verified.verification_reference, str)
                or not verified.verification_reference.strip()):
            raise ValueError('verified model binding does not match this model/tokenizer/head/stops')
    if missing and verified is None:
        raise UnsupportedModelBindingError(
            f'unsupported model binding: missing {", ".join(missing)}; supply explicit VerifiedModelBinding')
    return {'declarations': sources, 'attested_missing': missing,
            'verification_reference': verified.verification_reference if verified else None}


class NoLegalTokenError(RuntimeError):
    """Grammar-masked logits contain no finite legal next-token candidate."""


class _GenerationTimeout(TimeoutError):
    pass


class _UnsupportedChannelError(ValueError):
    pass


class DecisionGrammarProcessor(LogitsProcessor):
    """Single-use batch-one matcher; synchronizes every observed continuation ID."""
    def __init__(self, capability: CompiledDecisionGrammar, *, prompt_length: int,
                 deadline: float | None):
        _check_capability(capability)
        if type(prompt_length) is not int or prompt_length <= 0:
            raise ValueError('prompt_length must be a positive integer')
        self.capability = capability
        self.prompt_length = prompt_length
        self.deadline = deadline
        self.matcher = xgr.GrammarMatcher(capability.compiled_grammar)
        self.bitmask = xgr.allocate_token_bitmask(1, capability.head_vocab_size)
        self.generated_token_ids: list[int] = []
        self.accepted_count = 0

    def observe(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('structured generation requires batch one')
        tokens = input_ids[0, self.prompt_length:].tolist()
        if tokens[:len(self.generated_token_ids)] != self.generated_token_ids:
            raise ValueError('generation changed a previously observed continuation')
        self.generated_token_ids = tokens
        for token in tokens[self.accepted_count:]:
            if self.matcher.is_terminated():
                if token not in self.capability.stop_token_ids:
                    raise _UnsupportedChannelError('stop token is not confined to the terminal suffix')
                self.accepted_count += 1
                continue
            if token in self.capability.special_token_ids and token not in self.capability.stop_token_ids:
                raise _UnsupportedChannelError('non-stop special token in continuation')
            if token in self.capability.stop_token_ids and not self.matcher.is_completed():
                raise _UnsupportedChannelError('interior or premature stop token in continuation')
            if token in self.capability.blocked_token_ids:
                raise ValueError(f'generated blocked token {token} in continuation')
            if not self.matcher.accept_token(token):
                raise ValueError(f'generated token {token} rejected by decision grammar')
            self.accepted_count += 1

    def check_timeout(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _GenerationTimeout('structured generation exceeded monotonic deadline')

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.observe(input_ids)
        self.check_timeout()
        if scores.ndim != 2 or scores.shape != (1, self.capability.head_vocab_size):
            raise ValueError('logits head does not match the compiled head vocabulary')
        self.matcher.fill_next_token_bitmask(self.bitmask)
        xgr.apply_token_bitmask_inplace(
            scores, self.bitmask.to(scores.device), vocab_size=self.capability.head_vocab_size,
            backend='cpu' if scores.device.type == 'cpu' else 'auto',
        )
        if self.capability.blocked_token_ids:
            scores[:, list(self.capability.blocked_token_ids)] = -float('inf')
        return scores


class FiniteLegalTokenGuard(LogitsProcessor):
    """Independent guard after grammar masking; NaN/+inf cannot win argmax."""
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(scores)
        if not finite.any(dim=-1).all():
            raise NoLegalTokenError('no finite legal token after grammar masking')
        return scores.masked_fill(~finite, -float('inf'))


class _ObserveAndTimeout(StoppingCriteria):
    """Observe the last token too, including exact-budget closure and EOS."""
    def __init__(self, processor: DecisionGrammarProcessor):
        self.processor = processor

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.processor.observe(input_ids)
        self.processor.check_timeout()
        return torch.tensor([self.processor.matcher.is_terminated()], device=input_ids.device)


def _validate_policy(policy: StructuredGenerationPolicy, head_vocab_size: int) -> None:
    if not isinstance(policy, StructuredGenerationPolicy):
        raise ValueError('policy must be an explicit StructuredGenerationPolicy')
    if type(policy.max_new_tokens) is not int or policy.max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    if type(policy.use_cache) is not bool:
        raise ValueError('use_cache must be boolean')
    if type(policy.pad_token_id) is not int or not 0 <= policy.pad_token_id < head_vocab_size:
        raise ValueError('pad_token_id must be inside the head range')
    if (isinstance(policy.timeout_seconds, bool) or not isinstance(policy.timeout_seconds, (int, float))
            or not math.isfinite(policy.timeout_seconds) or policy.timeout_seconds <= 0):
        raise ValueError('timeout_seconds must be positive and finite')
    if policy.channel_policy not in ('plain_json', 'unsupported', 'harmony_no_tools'):
        raise ValueError('channel_policy must explicitly be plain_json, unsupported or harmony_no_tools')


def _hf_controls(policy: StructuredGenerationPolicy, stops: tuple[int, ...]) -> dict[str, Any]:
    # In Transformers 5, None-valued config fields inherit checkpoint defaults.
    # Explicit kwargs applied AFTER that merge must neutralize model-side forced
    # prefixes/stops/processors; a fresh config alone is insufficient.
    return {
        'max_new_tokens': policy.max_new_tokens, 'do_sample': False, 'num_beams': 1,
        'num_return_sequences': 1, 'use_cache': policy.use_cache,
        'eos_token_id': list(stops), 'pad_token_id': policy.pad_token_id,
        'forced_bos_token_id': None, 'forced_eos_token_id': None,
        'bad_words_ids': None, 'suppress_tokens': None, 'begin_suppress_tokens': None,
        'sequence_bias': None, 'exponential_decay_length_penalty': None,
        'watermarking_config': None, 'guidance_scale': None,
        'repetition_penalty': 1.0, 'encoder_repetition_penalty': 1.0,
        'no_repeat_ngram_size': 0, 'encoder_no_repeat_ngram_size': 0,
        'min_length': 0, 'min_new_tokens': 0, 'remove_invalid_values': False,
        'renormalize_logits': False, 'token_healing': False,
        'max_time': None, 'stop_strings': None, 'is_assistant': False,
        'prompt_lookup_num_tokens': None, 'assistant_early_exit': None,
        'assistant_ensemble_weight': None, 'use_mtp': False,
        'penalty_alpha': None, 'dola_layers': None,
        'constraints': None, 'force_words_ids': None, 'num_beam_groups': 1,
        'output_scores': False, 'output_logits': False, 'output_attentions': False,
        'output_hidden_states': False, 'return_dict_in_generate': False,
    }


def generate_structured(
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    capability: CompiledDecisionGrammar,
    *,
    policy: StructuredGenerationPolicy,
    transforms: Mapping[int, ResidualTransform] | None = None,
    verified_binding: VerifiedModelBinding | None = None,
) -> StructuredGenerationResult:
    """Separate greedy HF path, with grammar active after temporary forward hooks.

    Caller/input identity mistakes raise before execution. Execution errors become
    failure records and cannot certify compatibility. Timeout is cooperative at
    decoder callbacks (it cannot interrupt a hung forward). No arbitrary HF kwargs,
    prefill, sampling, beam search, answer margin or processor override is accepted.
    Available .tokenizer, output-head and config/generation_config declarations
    must agree before any forward. Missing tokenizer/head/stops declarations need
    verified_binding (a trusted, explicit caller attestation, never an override).
    """
    decoded_vocab = _check_capability(capability)
    _validate_policy(policy, capability.head_vocab_size)
    if policy.channel_policy == 'harmony_no_tools':
        raise ValueError('plain generate_structured does not accept harmony_no_tools policy')
    if (not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2
            or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0):
        raise ValueError('structured generation requires a nonempty batch-one prompt')
    if (prompt_ids.dtype not in (torch.int32, torch.int64)
            or (prompt_ids < 0).any() or (prompt_ids >= capability.head_vocab_size).any()):
        raise ValueError('prompt IDs must be integer tokens inside the head range')
    if _tokenizer_identity(tokenizer) != capability.tokenizer_sha256:
        raise ValueError('tokenizer identity differs from compiled grammar')
    binding = _bind_model(model, capability, verified_binding)
    # The legacy context registers hooks before its try/finally. Validate the
    # entire mapping first so a later invalid layer cannot leak an earlier hook.
    if transforms:
        layers = getattr(model, 'layers', None)
        if layers is None:
            raise ValueError('model must expose decoder layers for transforms')
        for layer, transform in transforms.items():
            if type(layer) is not int or not 0 <= layer < len(layers) or not callable(transform):
                raise ValueError('transforms must map valid integer layers to callable transforms')
    start = time.monotonic()
    controls = _hf_controls(policy, capability.stop_token_ids)
    provenance_bytes = canonical_json_bytes(
        _provenance_record(capability, binding, policy, controls, prompt_ids.device.type))
    failure: FailureType | None = None
    finish: FinishReason
    error = None
    tokens: tuple[int, ...] = ()
    if policy.channel_policy == 'unsupported':
        failure, finish = 'unsupported_channel', 'unsupported'
    else:
        processor = capability.new_processor(
            prompt_length=prompt_ids.shape[1], deadline=start + policy.timeout_seconds,
        )
        try:
            with residual_interventions(model, transforms or {}), torch.no_grad():
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
                    raise ValueError('generate must return the unchanged prompt plus one continuation')
                # Observe the returned suffix even if a custom generate skipped callbacks.
                tokens = tuple(sequences[0, prompt_ids.shape[1]:].tolist())
                if not torch.equal(sequences[:, :prompt_ids.shape[1]], prompt_ids):
                    raise ValueError('generate changed the prompt prefix')
                if len(tokens) > policy.max_new_tokens:
                    raise ValueError('generate returned more continuation tokens than max_new_tokens')
                if any(not 0 <= i < capability.head_vocab_size for i in tokens):
                    raise ValueError('generate returned token IDs outside the head range')
                processor.observe(sequences)
                processor.check_timeout()
                if len(tokens) < policy.max_new_tokens and not processor.matcher.is_completed():
                    raise ValueError('generate terminated early without a complete schema or explained stop')
            finish = 'eos' if tokens and tokens[-1] in capability.stop_token_ids else 'token_budget'
        except _GenerationTimeout as exc:
            failure, finish, error = 'timeout', 'timeout', _compact_error(exc)
        except NoLegalTokenError as exc:
            failure, finish, error = 'no_legal_token', 'no_legal_token', _compact_error(exc)
        except _UnsupportedChannelError as exc:
            failure, finish, error = 'unsupported_channel', 'unsupported', _compact_error(exc)
        except Exception as exc:
            failure, finish, error = 'exception', 'exception', _compact_error(exc)
        if not tokens:
            tokens = tuple(processor.generated_token_ids)
    return _finalize_result(tokenizer, capability, decoded_vocab, policy, tokens, failure,
                            finish, error, provenance_bytes, start)


def _provenance_record(capability: CompiledDecisionGrammar, binding: dict[str, Any],
                       policy: StructuredGenerationPolicy, controls: dict[str, Any],
                       device_type: str) -> dict[str, Any]:
    """Generation identity shared by batch-one and same-prompt row decoding."""
    return {
        'backend': 'xgrammar', 'backend_version': capability.backend_version,
        'model_binding': binding,
        'byte_policy': capability.byte_policy,
        'schema_sha256': capability.schema_sha256,
        'schema_bytes_sha256': capability.schema_bytes_sha256,
        'tokenizer_sha256': capability.tokenizer_sha256,
        'tokenizer_info_sha256': capability.tokenizer_info_sha256,
        'head_vocab_size': capability.head_vocab_size,
        'stop_token_ids': list(capability.stop_token_ids),
        'compiler_policy': dict(_COMPILER_POLICY),
        'grammar_correction': 'xgrammar-0.2.8-minLength-json-escapes-v1',
        'grammar_sha256': capability.grammar_sha256,
        'mask_backend': 'cpu' if device_type == 'cpu' else 'auto',
        'generation_policy': asdict(policy), 'hf_controls': controls,
        'generation_policy_sha256': sha256_json({'policy': asdict(policy), 'hf_controls': controls}),
    }


def _finalize_result(tokenizer: Any, capability: CompiledDecisionGrammar,
                     decoded_vocab: tuple[bytes, ...], policy: StructuredGenerationPolicy,
                     tokens: tuple[int, ...], failure: FailureType | None,
                     finish: FinishReason, error: str | None, provenance_bytes: bytes,
                     start: float, end: float | None = None) -> StructuredGenerationResult:
    """Decode, strictly validate and classify one observed continuation."""
    payload_ids = list(tokens)
    while payload_ids and payload_ids[-1] in capability.stop_token_ids:
        payload_ids.pop()
    text = payload = ''
    decode_error = None
    try:
        text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        payload = tokenizer.decode(payload_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # Never let replacement decoding turn invalid bytes into primary JSON.
        # Preserve legitimately encoded U+FFFD, multi-token emoji and Unicode.
        for ids, decoded in [(tokens, text), (payload_ids, payload)]:
            if any(type(i) is not int or not 0 <= i < len(decoded_vocab) for i in ids):
                raise ValueError('cannot decode token IDs outside the bound vocabulary')
            exact = b''.join(decoded_vocab[i] for i in ids).decode('utf-8', errors='strict')
            # Reject a decoder that exports actual Python surrogate characters
            # as well as one that replacement-decodes malformed token bytes.
            decoded.encode('utf-8', errors='strict')
            if decoded != exact:
                raise UnsupportedTokenizerError('HF decoded text differs from exact backend token bytes')
    except Exception as exc:
        decode_error = _compact_error(exc)
        # A decoder can itself return non-interoperable strings. Such text has
        # no lossless UTF8 export; keep IDs + decode_error rather than corrupt it.
        for name, decoded in [('text', text), ('payload', payload)]:
            try:
                decoded.encode('utf-8', errors='strict')
            except (UnicodeError, AttributeError):
                if name == 'text':
                    text = ''
                else:
                    payload = ''
        if failure is None:
            if isinstance(exc, UnsupportedTokenizerError):
                failure, finish = 'unsupported_tokenizer', 'unsupported'
            else:
                failure, finish = 'exception', 'exception'
            error = decode_error
    parsed = validate_decision_payload(payload if decode_error is None else '')
    if failure is None and any(i in capability.special_token_ids or i in capability.stop_token_ids
                               for i in payload_ids):
        failure, finish = 'unsupported_channel', 'unsupported'
    if failure is None:
        if parsed.failure_type is None:
            if finish != 'eos':
                finish = 'schema_complete'
        elif len(tokens) >= policy.max_new_tokens and not parsed.schema_complete:
            failure, finish = 'truncated', 'token_budget'
        else:
            failure = parsed.failure_type
            if parsed.schema_complete and finish != 'eos':
                finish = 'schema_complete'
    return StructuredGenerationResult(
        text, tokens, sha256_json(tokens), payload,
        parsed.decision if failure is None else None,
        parsed.reason if failure is None else None,
        parsed.decision_complete, parsed.schema_complete, parsed.reason_valid,
        finish, failure, (time.monotonic() if end is None else end) - start, provenance_bytes, error, decode_error,
    )


def _compact_error(exc: Exception) -> str:
    # Error text is diagnostic, UTF8-safe and bounded; never persist tensors.
    return f'{type(exc).__name__}: {exc}'.encode('utf-8', errors='backslashreplace').decode('utf-8')[:500]


__all__ = [
    'CANONICAL_SCHEMA_PATH', 'CompiledDecisionGrammar', 'DecisionGrammarProcessor',
    'DecisionPayloadValidation', 'FiniteLegalTokenGuard', 'NoLegalTokenError', 'StructuredGenerationPolicy',
    'StructuredGenerationResult', 'compile_decision_grammar', 'generate_structured',
    'load_decision_schema', 'validate_decision_payload', 'VerifiedModelBinding',
    'UnsupportedTokenizerError', 'UnsupportedModelBindingError',
]
