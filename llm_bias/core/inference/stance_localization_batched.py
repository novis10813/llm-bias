"""Same-prompt row-batched replacement for localization-discovery-v3.

Every row of one generate call is an exact copy of the target prompt, so rows
share length, attention mask and absolute positions; there is no padding. Each
row owns its grammar matcher, deadline, stop state and replacement coordinate.
Row 0 of every call is an unpatched control whose full output is compared with
the batch-one clean target. A control mismatch is recorded, never repaired.
Batched numerics are a declared execution property; equality with batch-one
execution is measured by the control row, not assumed. Plain JSON only.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import time

import torch
import xgrammar as xgr
from transformers import (
    GenerationConfig as HFGenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from ..artifact_paths import canonical_json_bytes
from ..stance_baseline_adapter import baseline_gate_input
from .interventions import residual_interventions
from .stance_interventions import (
    GenerationPositionTracker, PerturbationDiagnostics, UnsupportedPositionMetadata,
    _cache_length, _explicit_positions, _indices,
)
from .stance_localization_execution import (
    PromptReplacementExecution, _bind_alignment, _bind_expected, _bind_prompt_records,
    _execution, _input_embedding_weight, _prompt_tensor, _same_full_output,
)
from .stance_localization_grouped import ReplacementCell
from .stance_transient_capture import capture_prompt_residual
from .structured_output import (
    CompiledDecisionGrammar, DecisionGrammarProcessor, NoLegalTokenError, StructuredGenerationResult,
    _GenerationTimeout, _UnsupportedChannelError, _bind_model, _check_capability,
    _compact_error, _finalize_result, _hf_controls, _provenance_record,
    _tokenizer_identity, _validate_policy, generate_structured,
)

__all__ = ['BatchPositionTracker', 'BatchChunk', 'BatchedReplacementExecution',
           'generate_structured_rows', 'row_replacements',
           'execute_batched_prompt_replacement']


def _shared_row(value, batch):
    """Collapse batch-identical position coordinates; divergent rows are unsupported."""
    if not torch.is_tensor(value):
        return value
    if value.ndim == 2 and value.shape[0] == batch:
        first = value[:1]
    elif value.ndim == 3 and value.shape[1] == batch:
        first = value[:, :1]
    else:
        return value
    if not torch.equal(value, first.expand_as(value)):
        raise UnsupportedPositionMetadata('batched rows must share position coordinates')
    return first


class BatchPositionTracker(GenerationPositionTracker):
    """Absolute positions shared by identical unpadded prompt rows."""

    def __init__(self, prompt_length: int, use_cache: bool, batch_size: int):
        super().__init__(prompt_length, use_cache)
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch size must be a positive integer')
        self.batch_size = batch_size

    def _positions(self, args, kwargs):
        if args and kwargs.get('input_ids') is not None:
            raise UnsupportedPositionMetadata('input_ids supplied twice')
        input_ids = args[0] if args else kwargs.get('input_ids')
        embeds = kwargs.get('inputs_embeds')
        if input_ids is not None and embeds is not None:
            raise UnsupportedPositionMetadata('input_ids and inputs_embeds are mutually exclusive')
        values = input_ids if input_ids is not None else embeds
        ndim = 2 if input_ids is not None else 3
        if (not torch.is_tensor(values) or values.ndim != ndim
                or values.shape[0] != self.batch_size or values.shape[1] <= 0):
            raise UnsupportedPositionMetadata('require a nonempty root input with the declared batch')
        length = values.shape[1]
        explicit = []
        for name in ('cache_position', 'position_ids'):
            if kwargs.get(name) is not None:
                value = kwargs[name] if name == 'cache_position' else _shared_row(kwargs[name], self.batch_size)
                explicit.append(_explicit_positions(value, length=length, name=name))
        if explicit:
            if any(positions != explicit[0] for positions in explicit[1:]):
                raise UnsupportedPositionMetadata('cache_position and position_ids disagree')
            return explicit[0]
        if not self.use_cache:
            if length < self.prompt_length:
                raise UnsupportedPositionMetadata('implicit no-cache input must contain the full prompt')
            return tuple(range(length))
        offset = _cache_length(kwargs.get('past_key_values'))
        if offset == 0 and length != self.prompt_length:
            raise UnsupportedPositionMetadata('implicit empty-cache input must be the full initial prompt')
        return tuple(range(offset, offset + length))

    def select(self, values, *, scope, prompt_positions=None):
        """One position mask shared by every row of the active batch."""
        if not torch.is_tensor(values) or values.ndim != 3 or values.shape[0] != self.batch_size:
            raise ValueError('hidden batch does not match the declared row count')
        return super().select(values[:1], scope=scope, prompt_positions=prompt_positions)


@contextmanager
def row_replacements(model, *, tracker, rows):
    """Post-block prompt replacement per row: {row: (layer, target_positions, source)}.

    Each row is edited and measured exactly like the batch-one scoped
    replacement, on its own slice only. Unlisted rows pass through unchanged.
    """
    if not isinstance(tracker, BatchPositionTracker) or not tracker._tracking:
        raise ValueError('nest row replacements inside an active BatchPositionTracker')
    layers = getattr(model, 'layers', None)
    if layers is None:
        raise ValueError('model does not expose decoder layers')
    diagnostics, by_layer = {}, {}
    for row, (layer, positions, source) in rows.items():
        if type(row) is not int or not 0 <= row < tracker.batch_size:
            raise ValueError('invalid row index')
        if type(layer) is not int or not 0 <= layer < len(layers):
            raise ValueError('invalid layer coordinate')
        positions = _indices(positions, upper=tracker.prompt_length)
        if (not positions or not torch.is_tensor(source) or source.ndim != 3 or source.shape[0] != 1
                or source.shape[1] != len(positions) or not source.is_floating_point()
                or not torch.isfinite(source).all()):
            raise ValueError('replacement source must be a finite [1,n_positions,d_model] tensor')
        diagnostics[row] = PerturbationDiagnostics(layer, 'post', 'prompt_only', 'replacement')
        mapping = {position: index for index, position in enumerate(positions)}
        by_layer.setdefault(layer, []).append((row, positions, mapping, source))

    def make(entries):
        def transform(values):
            if not values.is_floating_point():
                raise ValueError('residual must be floating point')
            if min(tracker.active_positions) >= tracker.prompt_length:
                return values  # decode forward: prompt_only replacement selects nothing
            result = None
            for row, positions, mapping, source in entries:
                mask = tracker.select(values, scope='prompt_only', prompt_positions=positions)
                if (source.shape[-1] != values.shape[-1] or source.dtype != values.dtype
                        or source.device != values.device):
                    raise ValueError('replacement source must match hidden dimension, dtype and device')
                if not mask.any():
                    continue
                selected = values[row:row + 1, mask, :]
                indices = [mapping[p] for p, keep in zip(tracker.active_positions, mask.tolist()) if keep]
                edited = source[:, indices, :]
                if torch.equal(selected, edited):
                    diagnostics[row]._record_identity(selected)
                    continue
                diagnostics[row]._record(selected, edited)
                if result is None:
                    result = values.clone()
                result[row:row + 1, mask, :] = edited
            return values if result is None else result
        return transform

    with residual_interventions(model, {layer: make(entries) for layer, entries in by_layer.items()}):
        yield diagnostics


def _classify(exc):
    if isinstance(exc, _GenerationTimeout):
        return 'timeout', 'timeout', _compact_error(exc)
    if isinstance(exc, NoLegalTokenError):
        return 'no_legal_token', 'no_legal_token', _compact_error(exc)
    if isinstance(exc, _UnsupportedChannelError):
        return 'unsupported_channel', 'unsupported', _compact_error(exc)
    return 'exception', 'exception', _compact_error(exc)


class _RowState:
    """Per-row terminal state; a closed row only receives pad tokens afterwards."""

    def __init__(self):
        self.failures, self.lengths, self.ends = {}, {}, {}

    def closed(self, row):
        return row in self.failures or row in self.lengths

    def fail(self, row, exc):
        self.failures[row] = _classify(exc)
        self.ends[row] = time.monotonic()

    def finish(self, row, length):
        self.lengths[row] = length
        self.ends[row] = time.monotonic()


def _observe_open(processors, state, input_ids):
    """One device-to-host copy per step; each open row checks its own continuation."""
    host = input_ids.cpu()
    for row, processor in enumerate(processors):
        if not state.closed(row):
            try:
                processor.observe(host[row:row + 1])
                processor.check_timeout()
            except Exception as exc:
                state.fail(row, exc)


class _RowsProcessor(LogitsProcessor):
    """Batch-one grammar masking and finite guard for every open row in one pass.

    Matchers fill one shared CPU bitmask in parallel and a single kernel applies
    it to the open rows. Closed rows receive only the pad token.
    """

    def __init__(self, capability, processors, state, pad, device):
        self.capability, self.processors, self.state, self.pad = capability, processors, state, pad
        self.batch = xgr.BatchGrammarMatcher()
        self.bitmask = xgr.allocate_token_bitmask(len(processors), capability.head_vocab_size)
        self.blocked = torch.tensor(capability.blocked_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids, scores):
        rows = len(self.processors)
        if scores.shape != (rows, self.capability.head_vocab_size) or input_ids.shape[0] != rows:
            raise ValueError('logits rows or head do not match the declared rows and vocabulary')
        _observe_open(self.processors, self.state, input_ids)
        active = [row for row in range(rows) if not self.state.closed(row)]
        if active:
            self.batch.batch_fill_next_token_bitmask(
                [self.processors[row].matcher for row in active], self.bitmask, indices=active)
            xgr.apply_token_bitmask_inplace(
                scores, self.bitmask.to(scores.device), vocab_size=self.capability.head_vocab_size,
                indices=active, backend='cpu' if scores.device.type == 'cpu' else 'auto')
            if self.blocked.numel():
                scores[:, self.blocked] = -float('inf')
            finite = torch.isfinite(scores)
            for row in (~finite.any(dim=-1)).nonzero().flatten().tolist():
                if row in active:
                    self.state.fail(row, NoLegalTokenError('no finite legal token after grammar masking'))
            scores = scores.masked_fill(~finite, -float('inf'))
        closed = [row for row in range(rows) if self.state.closed(row)]
        if closed:
            scores[closed] = -float('inf')
            scores[closed, self.pad] = 0.0
        return scores


class _RowsStop(StoppingCriteria):
    """Observe each open row's last token; terminal or failed rows stop alone."""

    def __init__(self, processors, state, prompt_length):
        self.processors, self.state, self.prompt_length = processors, state, prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        _observe_open(self.processors, self.state, input_ids)
        for row, processor in enumerate(self.processors):
            if not self.state.closed(row) and processor.matcher.is_terminated():
                self.state.finish(row, input_ids.shape[1] - self.prompt_length)
        return torch.tensor([self.state.closed(row) for row in range(len(self.processors))],
                            dtype=torch.bool, device=input_ids.device)


def generate_structured_rows(model, tokenizer, prompt_ids, capability, *, policy, rows,
                             verified_binding=None):
    """Greedy-decode `rows` copies of one batch-one prompt; one result per row.

    Records use the batch-one provenance and classification. A row failure
    (grammar rejection, no legal token, timeout) closes that row only; an
    exception outside row callbacks fails every still-open row. Each row's
    deadline starts with the shared call and is checked when the row closes.
    """
    decoded_vocab = _check_capability(capability)
    _validate_policy(policy, capability.head_vocab_size)
    if policy.channel_policy != 'plain_json':
        raise ValueError('row-batched generation supports plain_json only')
    if type(rows) is not int or rows <= 0:
        raise ValueError('rows must be a positive integer')
    if (not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2
            or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0):
        raise ValueError('row-batched generation requires a nonempty batch-one prompt')
    if (prompt_ids.dtype not in (torch.int32, torch.int64)
            or (prompt_ids < 0).any() or (prompt_ids >= capability.head_vocab_size).any()):
        raise ValueError('prompt IDs must be integer tokens inside the head range')
    if _tokenizer_identity(tokenizer) != capability.tokenizer_sha256:
        raise ValueError('tokenizer identity differs from compiled grammar')
    binding = _bind_model(model, capability, verified_binding)
    start = time.monotonic()
    controls = _hf_controls(policy, capability.stop_token_ids)
    provenance_bytes = canonical_json_bytes(
        _provenance_record(capability, binding, policy, controls, prompt_ids.device.type))
    length = prompt_ids.shape[1]
    processors = [DecisionGrammarProcessor(capability, prompt_length=length,
                                           deadline=start + policy.timeout_seconds, _checked=True)
                  for _ in range(rows)]
    state = _RowState()
    batch = prompt_ids.expand(rows, -1).contiguous()
    sequences = shared = None
    try:
        with torch.no_grad():
            output = model.hf_model.generate(
                batch, attention_mask=torch.ones_like(batch),
                generation_config=HFGenerationConfig(**controls), **controls,
                logits_processor=LogitsProcessorList([_RowsProcessor(
                    capability, processors, state, policy.pad_token_id, prompt_ids.device)]),
                stopping_criteria=StoppingCriteriaList([_RowsStop(processors, state, length)]),
            )
        sequences = getattr(output, 'sequences', output)
        if (not isinstance(sequences, torch.Tensor) or sequences.ndim != 2
                or sequences.shape[0] != rows or sequences.shape[1] < length
                or sequences.dtype not in (torch.int32, torch.int64)):
            raise ValueError('generate must return the unchanged prompt rows plus continuations')
        if not torch.equal(sequences[:, :length], batch):
            raise ValueError('generate changed the prompt prefix')
        if sequences.shape[1] - length > policy.max_new_tokens:
            raise ValueError('generate returned more continuation tokens than max_new_tokens')
    except Exception as exc:
        shared = _classify(exc)
    finished = time.monotonic()
    results = []
    for row, processor in enumerate(processors):
        tokens: tuple[int, ...] = ()
        failure = error = None
        finish = None
        end = state.ends.get(row, finished)
        if row in state.failures:
            failure, finish, error = state.failures[row]
        elif shared is not None:
            failure, finish, error = shared
        else:
            count = state.lengths.get(row, sequences.shape[1] - length)
            try:
                tokens = tuple(sequences[row, length:length + count].tolist())
                if any(not 0 <= i < capability.head_vocab_size for i in tokens):
                    raise ValueError('generate returned token IDs outside the head range')
                processor.observe(sequences[row:row + 1, :length + count])
                if end >= processor.deadline:
                    raise _GenerationTimeout('structured generation exceeded monotonic deadline')
                if len(tokens) < policy.max_new_tokens and not processor.matcher.is_completed():
                    raise ValueError('generate terminated early without a complete schema or explained stop')
                finish = 'eos' if tokens and tokens[-1] in capability.stop_token_ids else 'token_budget'
            except Exception as exc:
                failure, finish, error = _classify(exc)
        if not tokens:
            tokens = tuple(processor.generated_token_ids)
        results.append(_finalize_result(tokenizer, capability, decoded_vocab, policy, tokens, failure,
                                        finish, error, provenance_bytes, start, end))
    return tuple(results)


@dataclass(frozen=True, slots=True)
class BatchChunk:
    """One generate call: row 0 is the unpatched control, rows 1.. are cells."""

    cell_indices: tuple[int, ...]
    control: StructuredGenerationResult
    control_match: bool


@dataclass(frozen=True, slots=True)
class BatchedReplacementExecution:
    """Clean outcomes, per-cell LC2 records and per-call controls; no tensors."""

    status: str
    cells: tuple[ReplacementCell, ...]
    donor: StructuredGenerationResult
    target_clean: StructuredGenerationResult | None
    executions: tuple[PromptReplacementExecution, ...]
    chunks: tuple[BatchChunk, ...]
    max_rows: int


def execute_batched_prompt_replacement(model, tokenizer, donor_prompt, target_prompt,
                                       capability, *, policy, cells, expected_donor,
                                       expected_target, max_rows, verified_binding=None):
    """Batch-one donor capture and clean target, then row-batched interventions.

    Cells are executed in order, at most max_rows - 1 per call after the
    control row. Clean-arm aborts reuse the grouped status names and contain no
    per-cell rows. Captures never escape the call or observe target forwards.
    """
    if not isinstance(cells, tuple) or not cells:
        raise ValueError('cells must be a nonempty immutable tuple')
    if type(max_rows) is not int or max_rows < 2:
        raise ValueError('max_rows must allow the control row and at least one cell')
    keys = set()
    for cell in cells:
        if not isinstance(cell, ReplacementCell):
            raise ValueError('cells must contain ReplacementCell records')
        if type(cell.layer) is not int or not 0 <= cell.layer < len(model.layers):
            raise ValueError('invalid layer coordinate')
        if cell.hook_site != 'post':
            raise ValueError('row batching supports the post-block site only')
        key = (cell.layer, cell.alignment.mapping_sha256)
        if key in keys:
            raise ValueError('duplicate cell coordinate')
        keys.add(key)
    if not isinstance(capability, CompiledDecisionGrammar):
        raise ValueError('row batching requires a plain-JSON factory grammar capability')
    _check_capability(capability)
    _validate_policy(policy, capability.head_vocab_size)
    if policy.channel_policy != 'plain_json':
        raise ValueError('policy does not match the plain-JSON route')
    if policy.use_cache is not True:
        # Without a cache every step recomputes the prompt, so a closed row would
        # keep receiving its replacement while other rows decode.
        raise ValueError('row-batched replacement requires use_cache=True')
    _bind_prompt_records(donor_prompt, target_prompt, capability)
    weight = _input_embedding_weight(model)
    upper = min(weight.shape[0], capability.head_vocab_size)
    donor_ids = _prompt_tensor(donor_prompt, weight.device, upper, 'donor')
    target_ids = _prompt_tensor(target_prompt, weight.device, upper, 'target')
    if _tokenizer_identity(tokenizer) != capability.tokenizer_sha256:
        raise ValueError('tokenizer identity differs from capability')
    _bind_model(model, capability, verified_binding)
    for cell in cells:
        _bind_alignment(donor_prompt, target_prompt, cell.alignment)
    _bind_expected(expected_donor, capability, policy, 'donor')
    _bind_expected(expected_target, capability, policy, 'target')

    def generate(prompt_ids):
        result = generate_structured(model, tokenizer, prompt_ids, capability, policy=policy,
                                     transforms=None, verified_binding=verified_binding)
        baseline_gate_input(result)
        return result

    unions = {}
    for cell in cells:
        unions.setdefault(cell.layer, set()).update(cell.alignment.donor_capture_positions)
    unions = {layer: tuple(sorted(positions)) for layer, positions in unions.items()}
    holders, sources = {}, {}
    donor = target_clean = None

    def abort(status):
        return BatchedReplacementExecution(status, cells, donor, target_clean, (), (), max_rows)

    try:
        with ExitStack() as lifetime:
            with ExitStack() as tracking:
                donor_tracker = GenerationPositionTracker(donor_ids.shape[1], policy.use_cache)
                tracking.enter_context(donor_tracker.track(model))
                for layer, positions in unions.items():
                    holders[layer] = lifetime.enter_context(capture_prompt_residual(
                        model, tracker=donor_tracker, layer=layer, hook_site='post',
                        prompt_positions=list(positions)))
                donor = generate(donor_ids)
            if donor.failure_type is not None:
                return abort('donor_failed')
            if not _same_full_output(donor, expected_donor):
                return abort('donor_mismatch')
            try:
                for layer, holder in holders.items():
                    sources[layer] = holder.require_source()
            except ValueError:
                return abort('source_unavailable')
            with GenerationPositionTracker(target_ids.shape[1], policy.use_cache).track(model):
                target_clean = generate(target_ids)
            if target_clean.failure_type is not None:
                return abort('target_failed')
            if not _same_full_output(target_clean, expected_target):
                return abort('target_mismatch')
            executions, chunks = [], []
            width = max_rows - 1
            for begin in range(0, len(cells), width):
                indices = tuple(range(begin, min(begin + width, len(cells))))
                rows = {}
                for row, index in enumerate(indices, start=1):
                    cell = cells[index]
                    order = {position: i for i, position in enumerate(unions[cell.layer])}
                    gather = [order[cell.alignment.donor_capture_positions[i]]
                              for i in cell.alignment.source_indices]
                    rows[row] = (cell.layer, list(cell.alignment.target_positions),
                                 sources[cell.layer][:, gather, :])
                tracker = BatchPositionTracker(target_ids.shape[1], policy.use_cache, len(indices) + 1)
                with tracker.track(model), row_replacements(model, tracker=tracker, rows=rows) as diagnostics:
                    results = generate_structured_rows(model, tokenizer, target_ids, capability,
                                                       policy=policy, rows=len(indices) + 1,
                                                       verified_binding=verified_binding)
                rows.clear()
                for result in results:
                    baseline_gate_input(result)
                chunks.append(BatchChunk(indices, results[0], _same_full_output(results[0], target_clean)))
                for row, index in enumerate(indices, start=1):
                    executions.append(_execution(
                        'executed', donor, target_clean, results[row], expected_donor,
                        expected_target, cells[index].alignment,
                        diagnostics=canonical_json_bytes(diagnostics[row].to_dict())))
            return BatchedReplacementExecution('executed', cells, donor, target_clean,
                                               tuple(executions), tuple(chunks), max_rows)
    finally:
        sources.clear()
        holders.clear()
