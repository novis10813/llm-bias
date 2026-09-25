"""Steering operators as per-token base shifts ``B[p]`` (the alpha = 1 dose) at the steer suffix.

All operators are anchored to the raw DIM difference ``d[p] = mean(Top10 h[p]) - mean(Bottom10 h[p])``
(post-block residual at the injection layer, instruction steer-suffix tokens). Equal-norm operators use
``B[p] = ||d[p]|| u[p]``; equal-projection operators rescale so that ``<B[p], d_hat[p]> = ||d[p]||``.
Only compact statistics and SHA-256 digests leave this module; raw states stay in memory.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping, Sequence

import torch

from llm_bias.core.inference.forward import record_residuals

from .prompts import FormattedPrompt
from .protocol import UNSPECIFIED_SECTOR, ModelSpec


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def collect_suffix_states(model: Any, fps: Sequence[FormattedPrompt], layers: Sequence[int],
                          ) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Post-block steer-suffix states ``{layer: [N, K, d]}`` (fp32, model device) and median ||h||."""
    device = getattr(model, "input_device", None)
    per_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for fp in fps:
        ids = torch.tensor([list(fp.ids)], dtype=torch.long, device=device)
        start, end = fp.spans["steer_suffix"]
        with torch.no_grad():
            states = record_residuals(model, ids, list(layers))
        for layer in layers:
            per_layer[layer].append(states[layer][0, start:end, :].float())
        del states
    stacked = {layer: torch.stack(rows) for layer, rows in per_layer.items()}
    for layer, value in stacked.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite suffix states at L{layer}")
    medians = {layer: float(value.norm(dim=-1).median()) for layer, value in stacked.items()}
    return stacked, medians


def fit_dim_difference(top: torch.Tensor, bottom: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Raw fp32 Top-minus-Bottom mean difference ``[K, d]`` (identical to V2 ``fit_dim_difference``)."""
    if top.shape != bottom.shape or top.ndim != 3 or top.shape[0] != 10 or top.shape[1] < 16:
        raise ValueError("DIM state shape must be [10, K>=16, d]")
    difference = top.float().mean(dim=0) - bottom.float().mean(dim=0)
    norms = difference.norm(dim=-1)
    if not torch.isfinite(norms).all():
        raise ValueError("DIM difference has nonfinite norm")
    return difference, {"min": float(norms.min()), "median": float(norms.median()), "max": float(norms.max()),
                        "difference_sha256": tensor_sha256(difference)}


def unit_rows(x: torch.Tensor, what: str = "direction") -> torch.Tensor:
    norms = x.norm(dim=-1, keepdim=True)
    if torch.any(norms <= 1e-12):
        raise ValueError(f"degenerate {what}: zero-norm row")
    return x / norms


def shared_random_direction(d_model: int, seed: int) -> tuple[torch.Tensor, str]:
    """One unit direction from a CPU fp32 generator, shared by every token (device independent)."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    u = torch.randn(d_model, generator=generator, dtype=torch.float32)
    u = u / u.norm()
    return u, tensor_sha256(u)


def equal_norm(d: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    """``B[p] = ||d[p]|| u[p]`` for a per-token ``[K, d]`` or shared ``[d]`` unit direction."""
    scale = d.norm(dim=-1, keepdim=True)
    u = unit.to(d.device, torch.float32)
    return scale * (u if u.ndim == 2 else u.unsqueeze(0).expand_as(d))


def equal_projection(d: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    """``B[p]`` along ``u[p]`` whose projection on ``d_hat[p]`` equals ``||d[p]||``."""
    u = unit.to(d.device, torch.float32)
    u = u if u.ndim == 2 else u.unsqueeze(0).expand_as(d)
    cos = (u * unit_rows(d)).sum(dim=-1, keepdim=True)
    if torch.any(cos <= 1e-3):
        raise ValueError("equal-projection dose undefined for directions nearly orthogonal to DIM")
    return d.norm(dim=-1, keepdim=True) * u / cos


def dose_table(d: torch.Tensor, base: torch.Tensor, median_h: float) -> dict[str, float]:
    """Per-operator dose facts at alpha = 1 (scale linearly with |alpha|)."""
    norms = base.norm(dim=-1)
    cos = (unit_rows(base, "operator") * unit_rows(d)).sum(dim=-1)
    return {"inject_norm_median": float(norms.median()), "inject_norm_min": float(norms.min()),
            "inject_norm_max": float(norms.max()), "cos_to_dim_median": float(cos.median()),
            "cos_to_dim_min": float(cos.min()), "projection_on_dim_median": float((norms * cos).median()),
            "relative_to_residual_median": float(norms.median()) / median_h, "base_sha256": tensor_sha256(base)}


def _orthonormalize_against(d_hat: torch.Tensor, vectors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Per-token Gram-Schmidt of shared vectors against ``d_hat[p]`` and each other -> ``[K, d]`` each."""
    basis = [d_hat]
    out = []
    for v in vectors:
        w = v.to(d_hat.device, torch.float32).unsqueeze(0).expand_as(d_hat).clone()
        for b in basis:
            w = w - (w * b).sum(dim=-1, keepdim=True) * b
        w = unit_rows(w, "orthogonalized axis")
        basis.append(w)
        out.append(w)
    return out


def cone_axes(top: torch.Tensor, bottom: torch.Tensor, d: torch.Tensor, top_margins: Sequence[float],
              bottom_margins: Sequence[float], n_extra: int = 3) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Pairing-free cone axes 2..(1+n_extra), shared across tokens, then orthogonalized per token.

    Rows are all 10x10 Top-minus-Bottom differences at every token with their ``d_hat[p]`` component
    removed and unit-normalized; the leading right singular vectors of the stacked rows are the shared
    axes. Orientation (pre-registered): each axis is signed so that the construction companies'
    token-mean projection on it correlates non-negatively (Pearson) with their clean fixed-prefix margin.
    """
    d_hat = unit_rows(d)
    diffs = top.float().unsqueeze(1) - bottom.float().unsqueeze(0)            # [10, 10, K, d]
    rows = diffs - (diffs * d_hat).sum(dim=-1, keepdim=True) * d_hat
    rows = unit_rows(rows.reshape(-1, rows.shape[-1]), "orthogonal contrast")  # [100K, d]
    gram = rows.double().T @ rows.double()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:n_extra]
    shared = [eigenvectors[:, i].float() for i in order.tolist()]
    states = torch.cat([top, bottom]).float().mean(dim=1)                     # [20, d] token-mean state
    margins = torch.tensor(list(top_margins) + list(bottom_margins), dtype=torch.float64)
    signed = []
    orientation = []
    for v in shared:
        proj = (states.double() @ v.double().to(states.device)).cpu()
        pc = proj - proj.mean()
        mc = margins - margins.mean()
        corr = float((pc * mc).sum() / (pc.norm() * mc.norm()).clamp_min(1e-30))
        if corr < 0 or (corr == 0 and float(v[v.abs().argmax()]) < 0):
            v = -v
        orientation.append(abs(corr))
        signed.append(v)
    axes = _orthonormalize_against(d_hat, signed)
    consistency = [float((a * v.to(a.device).unsqueeze(0)).sum(dim=-1).abs().mean()) for a, v in zip(axes, signed)]
    explained = eigenvalues[order].double() / eigenvalues.double().sum()
    return axes, {"n_rows": int(rows.shape[0]), "explained_fraction": [float(x) for x in explained],
                  "orientation_abs_pearson": orientation, "token_consistency_mean_abs_cos": consistency,
                  "shared_axes_sha256": [tensor_sha256(v) for v in signed]}


def cone_unit(d: torch.Tensor, axes: Sequence[torch.Tensor], k: int) -> torch.Tensor:
    """``(d_hat[p] + sum_{j=2..k} v_j[p]) / sqrt(k)`` -- unit because the axes are orthonormal."""
    if not 1 <= k <= len(axes) + 1:
        raise ValueError("cone dimension out of range")
    total = unit_rows(d)
    for axis in axes[:k - 1]:
        total = total + axis
    return total / math.sqrt(k)


def orth_random_axes(d: torch.Tensor, seeds: Sequence[int]) -> tuple[list[torch.Tensor], list[str]]:
    """DIM-orthogonal random control: shared random vectors orthogonalized per token like the cone axes."""
    vectors, digests = zip(*(shared_random_direction(d.shape[-1], seed) for seed in seeds))
    return _orthonormalize_against(unit_rows(d), vectors), list(digests)


# ---------------------------------------------------------------------------------------------------
# single-neuron write vector


def _read_tensor(model_dir: Any, key: str) -> torch.Tensor:
    from pathlib import Path

    from safetensors import safe_open

    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(path), "pt") as handle:
            if key in handle.keys():
                return handle.get_tensor(key)
    raise KeyError(f"{key} not found in {model_dir}")


def neuron_write_vectors(model_dir: Any, spec: ModelSpec, layer: int) -> tuple[torch.Tensor, list[str]]:
    """Residual write vectors of every MLP neuron at ``layer`` as rows ``[N, d]`` (CPU fp32) + row labels.

    Dense: column n of the down projection, times the post-MLP norm gain for sandwich-norm models
    (Gemma, GLM). MoE (GPT-OSS): row n of each expert's dequantized MXFP4 down projection
    ``[E, intermediate, hidden]``; the label records ``expert:neuron``.
    """
    if spec.mlp == "dense":
        weight = _read_tensor(model_dir, spec.down_proj_key.format(layer=layer)).float()  # [d, I]
        if spec.post_mlp_norm_key:
            gain = _read_tensor(model_dir, spec.post_mlp_norm_key.format(layer=layer)).float()
            weight = gain.unsqueeze(1) * weight
        return weight.T.contiguous(), [str(n) for n in range(weight.shape[1])]
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    base = spec.down_proj_key.format(layer=layer)
    blocks = _read_tensor(model_dir, base + "_blocks")
    scales = _read_tensor(model_dir, base + "_scales")
    weight = convert_moe_packed_tensors(blocks, scales, dtype=torch.float32)   # [E, I, H]
    experts, inter, hidden = weight.shape
    return weight.reshape(experts * inter, hidden).contiguous(), [f"{e}:{n}" for e in range(experts) for n in range(inter)]


def select_neuron(writes: torch.Tensor, labels: Sequence[str], target: torch.Tensor) -> dict[str, Any]:
    """Pre-registered rule: argmax_n cos(write_n, target) with target = mean_p d[p] (construction only)."""
    t = target.float().cpu()
    norms = writes.norm(dim=-1)
    valid = norms > 0
    cos = torch.full((writes.shape[0],), -2.0)
    cos[valid] = (writes[valid] @ t) / (norms[valid] * t.norm())
    best = int(cos.argmax())
    unit = writes[best] / writes[best].norm()
    ranked = torch.sort(cos, descending=True).values
    return {"neuron": labels[best], "cos": float(cos[best]), "second_cos": float(ranked[1]),
            "n_candidates": int(valid.sum()), "write_norm": float(norms[best]), "unit": unit,
            "unit_sha256": tensor_sha256(unit), "cos_by_label": cos}


# ---------------------------------------------------------------------------------------------------
# construction subsets


def top_bottom(ranking: Sequence[Mapping[str, Any]], exclude: set[str] = frozenset(), k: int = 10,
               ) -> tuple[list[str], list[str]]:
    """(top-k, bottom-k) of a ``(margin, ticker)``-ascending ranking after exclusion."""
    kept = [row["ticker"] for row in ranking if row["ticker"] not in exclude]
    if len(kept) < 2 * k:
        raise ValueError("not enough construction companies after exclusion")
    return kept[-k:], kept[:k]


def loso_folds(ranking: Sequence[Mapping[str, Any]], companies: Mapping[str, Mapping[str, str]], sector: str,
               seed: int) -> dict[str, dict[str, Any]]:
    """Leave-one-sector-out fold, its Unspecified-only comparator, and a size-matched placebo fold."""
    construction = [row["ticker"] for row in ranking]
    unspecified = {t for t in construction if companies[t]["sector"] == UNSPECIFIED_SECTOR}
    in_sector = {t for t in construction if companies[t]["sector"] == sector}
    others = sorted(t for t in construction if t not in unspecified and t not in in_sector)
    placebo = set(random.Random(seed).sample(others, len(in_sector)))
    folds = {}
    for name, excluded in (("loso", unspecified | in_sector), ("comparator", unspecified),
                           ("placebo", unspecified | placebo)):
        top, bottom = top_bottom(ranking, excluded)
        folds[name] = {"excluded_n": len(excluded), "top_10": top, "bottom_10": bottom}
    return folds


def shuffled_groups(ranking: Sequence[Mapping[str, Any]], seed: int, k: int = 10) -> tuple[list[str], list[str]]:
    """Label-shuffle null: two random disjoint groups of k from construction minus the real Top/Bottom k."""
    real_top, real_bottom = top_bottom(ranking)
    pool = sorted(row["ticker"] for row in ranking if row["ticker"] not in set(real_top) | set(real_bottom))
    drawn = random.Random(seed).sample(pool, 2 * k)
    return sorted(drawn[:k]), sorted(drawn[k:])
