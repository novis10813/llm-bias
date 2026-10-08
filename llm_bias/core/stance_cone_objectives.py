"""Pure sequence surrogates and independent cone initialization, no model runner.

Logits stay transient. Teacher records contain tokens and provenance, never
activations or distributions. Callers must bind roles to their frozen inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

import torch
import torch.nn.functional as F

from .artifact_paths import sha256_json

CONE_DIMENSIONS = (2, 4)
INITIALIZATION_SEEDS = (20261003, 20261004, 20261005)
_OBJECTIVES = ('addition', 'ablation', 'retain')
_RESPONSE_SOURCES = ('construction_generated', 'construction_authored')


def _hash(value: str) -> None:
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected lowercase SHA-256 provenance')


@dataclass(frozen=True, slots=True)
class TeacherResponse:
    """Fit-only sequence target plus source, tokenizer, schema and review bindings.

    token_ids includes the original prompt and response (and optional padding).
    response_mask marks target tokens, not logit positions. Position zero cannot
    have a target because causal logits predict the next token.
    """

    ticker: str
    role: str
    purpose: str
    token_ids: tuple[int, ...]
    response_mask: tuple[bool, ...]
    source_sha256: str
    tokenizer_sha256: str
    schema_sha256: str
    token_ids_sha256: str
    response_source: str
    target_decision: str
    review_sha256: str

    def __post_init__(self) -> None:
        if type(self.ticker) is not str or not self.ticker.strip():
            raise ValueError('ticker must be nonblank')
        if self.role != 'fit' or self.purpose not in _OBJECTIVES:
            raise ValueError('teachers must be fit-only with a named objective')
        if self.response_source not in _RESPONSE_SOURCES or self.target_decision not in ('buy', 'sell'):
            raise ValueError('unknown construction response source or decision')
        if (type(self.token_ids) is not tuple or len(self.token_ids) < 2
                or any(type(t) is not int or t < 0 for t in self.token_ids)):
            raise ValueError('token_ids must be an immutable nonempty causal sequence')
        if (type(self.response_mask) is not tuple or len(self.response_mask) != len(self.token_ids)
                or any(type(m) is not bool for m in self.response_mask)
                or self.response_mask[0] or not any(self.response_mask[1:])):
            raise ValueError('response_mask must select at least one shifted target')
        for digest in (self.source_sha256, self.tokenizer_sha256, self.schema_sha256,
                       self.token_ids_sha256, self.review_sha256):
            _hash(digest)
        if self.token_ids_sha256 != sha256_json(list(self.token_ids)):
            raise ValueError('source token provenance hash mismatch')


def _finite_tensor(value: torch.Tensor, rank: int, label: str) -> None:
    if (not isinstance(value, torch.Tensor) or value.ndim != rank
            or not value.is_floating_point() or any(n == 0 for n in value.shape)
            or not torch.isfinite(value).all().item()):
        raise ValueError(f'{label} must be a nonempty finite floating tensor of rank {rank}')


def _targets(logits, records, roles, purpose):
    _finite_tensor(logits, 4, 'logits [direction, example, position, vocabulary]')
    if (type(records) is not tuple or not records or not isinstance(roles, Mapping)
            or purpose not in _OBJECTIVES):
        raise ValueError('expected immutable teachers, authoritative roles and a named objective')
    _, examples, length, vocab = logits.shape
    if len(records) != examples:
        raise ValueError('teacher/example shape mismatch')
    for record in records:
        if type(record) is not TeacherResponse:
            raise ValueError('expected TeacherResponse records')
        record.__post_init__()
        if record.role != 'fit' or roles.get(record.ticker) != 'fit':
            raise ValueError('teacher role does not match authoritative fit assignment')
        if record.purpose != purpose or len(record.token_ids) != length:
            raise ValueError('teacher purpose or sequence shape mismatch')
        if max(record.token_ids) >= vocab:
            raise ValueError('teacher token outside logit vocabulary')
    if (len({r.tokenizer_sha256 for r in records}) != 1
            or len({r.schema_sha256 for r in records}) != 1):
        raise ValueError('teachers must share tokenizer and schema bindings')
    tokens = torch.tensor([r.token_ids for r in records], device=logits.device, dtype=torch.long)
    mask = torch.tensor([r.response_mask for r in records], device=logits.device, dtype=torch.bool)
    return tokens[:, 1:], mask[:, 1:]


def _length_mean(token_losses, mask):
    # Mean per sequence first, then equal example mean. Directions stay separate.
    masked = token_losses.masked_fill(~mask[None], 0)
    result = (masked.sum(-1) / mask.sum(-1)[None]).mean(-1)
    _finite_tensor(result, 1, 'sequence losses')
    return result


def sequence_ce(
    logits: torch.Tensor, records: tuple[TeacherResponse, ...], *,
    roles: Mapping[str, str], purpose: str,
) -> torch.Tensor:
    """Addition/ablation CE per direction, on shifted response tokens.

    Input [D,B,T,V] contains teacher-forced edited sequence logits. The caller
    applies the named addition or projection-ablation intervention before this
    function. This function neither applies a hook nor assigns buy/sell targets.
    """
    if purpose not in ('addition', 'ablation'):
        raise ValueError('CE requires addition or ablation purpose')
    tokens, mask = _targets(logits, records, roles, purpose)
    log_probs = F.log_softmax(logits[:, :, :-1, :], dim=-1)
    losses = -log_probs.gather(-1, tokens[None, :, :, None].expand(logits.shape[0], -1, -1, 1)).squeeze(-1)
    return _length_mean(losses, mask)


def sequence_kl(
    logits: torch.Tensor, clean_logits: torch.Tensor,
    records: tuple[TeacherResponse, ...], *, roles: Mapping[str, str],
) -> torch.Tensor:
    """KL(clean || edited) per direction over the entire masked response.

    clean_logits [B,T,V] must come from the same fit teacher-forced sequence.
    Detachment freezes the clean teacher even if its input requires gradients.
    Neither clean nor edited logits belong in a persisted teacher record.
    """
    _, mask = _targets(logits, records, roles, 'retain')
    _finite_tensor(clean_logits, 3, 'clean_logits')
    if (clean_logits.shape != logits.shape[1:] or clean_logits.device != logits.device
            or clean_logits.dtype != logits.dtype):
        raise ValueError('clean/edited logit shape, device or dtype mismatch')
    clean_log = F.log_softmax(clean_logits.detach()[:, :-1, :], dim=-1)
    edited_log = F.log_softmax(logits[:, :, :-1, :], dim=-1)
    losses = (clean_log.exp()[None] * (clean_log[None] - edited_log)).sum(-1)
    return _length_mean(losses, mask)


@dataclass(frozen=True, slots=True)
class ConeObjective:
    """Transient differentiable scalars, not a serializable artifact."""

    addition: torch.Tensor
    ablation: torch.Tensor
    retain: torch.Tensor
    total: torch.Tensor
    weights: tuple[float, float, float] = (1., 1., 1.)


def cone_objective(
    basis: Mapping[str, torch.Tensor], samples: Mapping[str, torch.Tensor],
) -> ConeObjective:
    """Sum three objectives with weights 1/1/1; each is basis.mean + sample.mean.

    Inputs are the per-direction sequence losses from sequence_ce/sequence_kl.
    The two panels may have different direction counts. Counts must agree across
    objectives within each panel. Empty panels are errors, not zero-loss arms.
    """
    reference = None
    for panel in (basis, samples):
        if not isinstance(panel, Mapping) or set(panel) != set(_OBJECTIVES):
            raise ValueError('each panel needs addition, ablation and retain')
        count = None
        for name in _OBJECTIVES:
            value = panel[name]
            _finite_tensor(value, 1, f'{name} panel losses')
            if count is not None and value.shape[0] != count:
                raise ValueError('direction counts must match within each panel')
            count = value.shape[0]
            if reference is not None and (value.device != reference.device or value.dtype != reference.dtype):
                raise ValueError('panel dtype/device mismatch')
            reference = value
    terms = tuple(basis[name].mean() + samples[name].mean() for name in _OBJECTIVES)
    total = sum(terms)
    if not torch.isfinite(total).item():
        raise ValueError('nonfinite total objective')
    return ConeObjective(*terms, total)


def initialize_basis(
    d_model: int, *, dimension: int, seed: int, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Independent normalized Gaussian rows [dimension,d_model], initialized on CPU.

    A local torch generator leaves global RNG state untouched. No DIM, anchor,
    whitening scale or residual data enters initialization. The caller owns
    requires_grad, device placement and any subsequent optimization.
    """
    if (type(d_model) is not int or type(dimension) is not int
            or dimension not in CONE_DIMENSIONS or d_model < dimension or type(seed) is not int
            or seed not in INITIALIZATION_SEEDS or dtype not in (torch.float32, torch.float64)):
        raise ValueError('expected d_model >= dimension 2/4 and a declared initialization seed/dtype')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    basis = torch.randn(dimension, d_model, generator=generator, dtype=dtype)
    return _unit_rows(basis)


def _unit_rows(value):
    _finite_tensor(value, 2, 'directions')
    norm = value.norm(dim=-1, keepdim=True)
    if not torch.isfinite(norm).all().item() or (norm <= 0).any().item():
        raise ValueError('zero or nonfinite direction norm')
    result = value / norm
    _finite_tensor(result, 2, 'normalized directions')
    return result


def positive_rays(basis: torch.Tensor, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return L1-normalized positive coefficients and unit rays in positive cone C.

    Basis [K,H], coefficients [R,K]. Basis rows need not stay orthogonal during
    optimization. Strictly positive finite coefficients exclude boundary rays;
    individual basis axes are a separate evaluation panel. A cancelling ray fails.
    """
    _finite_tensor(basis, 2, 'basis')
    _finite_tensor(coefficients, 2, 'coefficients')
    if (coefficients.shape[1] != basis.shape[0] or coefficients.device != basis.device
            or coefficients.dtype != basis.dtype or (coefficients <= 0).any().item()):
        raise ValueError('expected matching positive coefficient/basis shapes, dtype and device')
    basis = _unit_rows(basis)
    normalized = coefficients / coefficients.sum(-1, keepdim=True)
    _finite_tensor(normalized, 2, 'normalized coefficients')
    if (normalized <= 0).any().item():
        raise ValueError('coefficient normalization underflow/overflow')
    return normalized, _unit_rows(normalized @ basis)


def negative_cone_adaptation(rays: torch.Tensor) -> torch.Tensor:
    """Return -C unit directions, a separately named bidirectional finance adaptation.

    Positive-cone ray coverage says nothing about these reversed directions.
    This helper changes direction only, not native dose magnitude or scope.
    """
    return -_unit_rows(rays)
