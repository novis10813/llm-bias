#!/usr/bin/env python
"""Layer-band localisation for the Jacobian-lens workspace (Gurnee et al. Fig. 27-28).

Computes four per-layer lens statistics on pretraining-like text using an
existing validated canonical lens (no fitting):

  (a) top-k next-token prediction accuracy of the lens readout;
  (b) excess kurtosis of the per-position readout logit distribution;
  (c) cross-position agreement of the top-1 lens token at fixed lags,
      reported against its position-permutation null expectation;
  (d) effective dimensionality (participation ratio) of the J-lens
      vectors W_U J_l, plus components needed for 90% singular energy.

Also emits a layer x layer linear CKA matrix over J-lens vector column
spaces (Fig. 27 style), computed from d_model-side Gram matrices.

Only compact per-layer statistics and provenance are written; no raw
activations, residuals, or full-vocab logits are persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from llm_bias.core.artifacts.io import write_json, write_metadata
from llm_bias.core.lens_loader import load_validated_lens
from llm_bias.core.model import load_model

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_FILE = "wikitext-103-raw-v1/train-00000-of-00002.parquet"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus_documents(parquet_path: Path, tokenizer, num_docs: int, seq_len: int) -> list[str]:
    """Deterministically build ~seq_len-token documents from wikitext paragraphs."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["text"])
    texts = table.column("text").to_pylist()
    documents: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0
    for line in texts:
        stripped = line.strip()
        if not stripped or set(stripped) <= {"=", "'"}:
            continue  # skip blank lines and = = section = = headers
        buffer.append(stripped)
        buffer_tokens += len(tokenizer.encode(line, add_special_tokens=False))
        if buffer_tokens >= seq_len + 8:
            text = " ".join(buffer)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) >= seq_len + 1:
                documents.append(tokenizer.decode(ids[: seq_len + 1]))
                if len(documents) >= num_docs:
                    break
            buffer, buffer_tokens = [], 0
    if len(documents) < num_docs:
        raise RuntimeError(f"corpus exhausted after {len(documents)} documents (< {num_docs})")
    return documents


def excess_kurtosis(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise excess kurtosis of a [T, vocab] logits tensor."""
    centered = logits - logits.mean(dim=-1, keepdim=True)
    m2 = (centered**2).mean(dim=-1)
    m4 = (centered**4).mean(dim=-1)
    return m4 / (m2**2 + 1e-12) - 3.0


def lag_agreement(top1: np.ndarray, lag: int) -> tuple[float, float]:
    """Observed and permutation-null top-1 agreement rate at ``lag`` positions apart.

    Returns (observed, null); null is the expected match rate when the series is
    randomly permuted, i.e. sum_v n_v (n_v - 1) / (T (T - 1)).
    """
    pairs = len(top1) - lag
    if pairs <= 0:
        return float("nan"), float("nan")
    observed = float(np.mean(top1[:-lag] == top1[lag:]))
    _, counts = np.unique(top1, return_counts=True)
    total = len(top1)
    null = float(np.sum(counts * (counts - 1)) / (total * (total - 1)))
    return observed, null


def lens_vector_gram(lens_jacobian: torch.Tensor, unembed_gram: torch.Tensor) -> torch.Tensor:
    """d_model x d_model Gram matrix of the J-lens vectors W_U J_l.

    V_l^T V_m with V_l = W_U J_l reduces to J_l^T (W_U^T W_U) J_m, so all
    quantities stay at d_model scale without materialising [vocab, d_model].
    """
    return lens_jacobian.T @ unembed_gram @ lens_jacobian


def _load_unembedding_weight(model_path: Path) -> torch.Tensor:
    """Load lm_head.weight from a local HF checkpoint without loading the model."""
    import json

    from safetensors.torch import load_file

    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if "lm_head.weight" in weight_map:
            shard, key = weight_map["lm_head.weight"], "lm_head.weight"
        else:
            # tied embeddings: the unembedding is the input embedding table
            embed_key = next(
                k for k in weight_map if k.endswith("embed_tokens.weight")
            )
            shard, key = weight_map[embed_key], embed_key
    else:
        shard, key = "model.safetensors", "lm_head.weight"
    tensors = load_file(str(model_path / shard), device="cpu")
    return tensors[key].float()


def _run_geometry_only(args: argparse.Namespace, output_dir: Path) -> None:
    """CPU-only per-layer J-lens vector dimensionality + CKA ((d), Fig 27-28d)."""
    import jlens

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")
    import json

    config = json.loads((model_path / "config.json").read_text())
    text_config = config.get("text_config", config)

    if args.lens is not None:
        lens_path = Path(args.lens)
    else:
        # canonical layout: artifacts/<slug>/jacobian-lens/jacobian_lens.pt
        slug = model_path.name
        lens_path = Path(args.artifact_root) / slug / "jacobian-lens" / "jacobian_lens.pt"
    lens = jlens.JacobianLens.load(str(lens_path))
    print(f"lens: {lens_path} ({len(lens.source_layers)} source layers)")

    unembed_weight = _load_unembedding_weight(model_path)
    gram_wu = unembed_weight.T @ unembed_weight
    del unembed_weight

    layer_list = list(lens.source_layers)
    grams = {
        layer: lens_vector_gram(lens.jacobians[layer].float().cpu(), gram_wu)
        for layer in layer_list
    }
    per_layer: dict[int, dict] = {}
    for layer in layer_list:
        eigvals = torch.linalg.eigvalsh(grams[layer]).clamp_min(0.0).flip(0)
        ratio = float((eigvals.sum() ** 2) / ((eigvals**2).sum() + 1e-30))
        cumulative = torch.cumsum(eigvals, dim=0) / eigvals.sum()
        n_energy = int(torch.searchsorted(cumulative, torch.tensor(args.energy_threshold)).item()) + 1
        per_layer[layer] = {
            "jvector_effective_dim_pr": ratio,
            f"jvector_dims_for_{int(args.energy_threshold * 100)}pct_energy": n_energy,
        }

    cka = np.zeros((len(layer_list), len(layer_list)))
    self_norms = {
        layer: float(torch.linalg.matrix_norm(grams[layer], ord="fro"))
        for layer in layer_list
    }
    for i, li in enumerate(layer_list):
        for j, lj in enumerate(layer_list):
            if j < i:
                continue
            cross = jacobians_cpu_cross(grams, li, lj, gram_wu, lens)
            value = float(torch.linalg.matrix_norm(cross, ord="fro") ** 2) / (
                self_norms[li] * self_norms[lj]
            )
            cka[i, j] = cka[j, i] = min(value, 1.0)

    result = {
        "artifact_type": "jspace_layer_stats_geometry",
        "schema_version": 1,
        "model": str(model_path),
        "model_n_layers": text_config.get("num_hidden_layers"),
        "lens_path": str(lens_path),
        "readout_layers": layer_list,
        "stats": {str(layer): per_layer[layer] for layer in layer_list},
        "params": {
            "energy_threshold": args.energy_threshold,
            "mode": "geometry_only_cpu",
            "note": "(d) effective dimensionality and CKA of J-lens vectors W_U J_l "
            "on raw unembedding weights; no forward pass, no corpus.",
        },
    }
    write_json(output_dir / "layer_stats_geometry.json", result, overwrite=True)
    np.savez_compressed(output_dir / "cka.npz", layers=np.array(layer_list), cka=cka)
    write_metadata(
        output_dir / "layer_stats_geometry.json.metadata.json",
        {"lens_sha256": sha256_file(lens_path)},
        overwrite=True,
    )
    print(f"wrote {output_dir / 'layer_stats_geometry.json'}")


def jacobians_cpu_cross(grams, li, lj, gram_wu, lens):
    """Cross-Gram V_li^T V_lj = J_li^T G W_U J_lj."""
    return (
        lens.jacobians[li].float().cpu().T
        @ gram_wu
        @ lens.jacobians[lj].float().cpu()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--lens", default=None, help="defaults to the canonical lens")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first-tokens", type=int, default=16,
                        help="positions skipped at the start of every sequence")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--energy-threshold", type=float, default=0.90,
                        help="singular-energy coverage for the dimensionality stat")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="compute only the lens-vector geometry stats ((d) + CKA) on CPU, "
        "reading the unembedding weight directly from safetensors without "
        "loading the model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.geometry_only:
        _run_geometry_only(args, output_dir)
        return

    # ---------------------------------------------------------------- corpus
    from huggingface_hub import hf_hub_download

    parquet_path = Path(hf_hub_download(WIKITEXT_REPO, WIKITEXT_FILE, repo_type="dataset"))

    model, tokenizer, device = load_model(args.model)
    loaded = load_validated_lens(model=model, model_name=args.model, lens_path=args.lens)
    lens = loaded.lens
    print(f"lens: {loaded.path} ({loaded.source})")

    documents = load_corpus_documents(
        parquet_path, tokenizer, args.num_prompts, args.max_seq_len
    )
    print(f"loaded {len(documents)} documents from {parquet_path.name}")

    source_layers = list(lens.source_layers)
    final_layer = model.n_layers - 1
    readout_layers = source_layers + ([final_layer] if final_layer not in source_layers else [])

    # ------------------------------------------------------- forward + stats
    k_values = sorted({args.top_k})
    acc_hits = {layer: {f"top{args.top_k}": 0} for layer in readout_layers}
    acc_total = 0
    kurtosis_sum = {layer: 0.0 for layer in readout_layers}
    kurtosis_count = 0
    top1_series: dict[int, list[np.ndarray]] = {layer: [] for layer in readout_layers}

    with torch.no_grad():
        for doc_index, text in enumerate(documents):
            lens_logits_by_layer, model_logits, input_ids = lens.apply(
                model,
                text,
                layers=source_layers,
                positions=None,
                max_seq_len=args.max_seq_len,
            )
            lens_logits_by_layer[final_layer] = model_logits
            token_ids = input_ids[0].tolist()
            start = args.skip_first_tokens
            end = len(token_ids) - 1  # last scored position predicts token_ids[-1]
            if end - start < max(args.lags):
                raise RuntimeError(f"document {doc_index} too short after skipping")

            for layer in readout_layers:
                logits = lens_logits_by_layer[layer].to(device).float()
                scores = logits[start:end]
                targets = torch.tensor(
                    token_ids[start + 1 : end + 1], device=device
                )
                topk_indices = torch.topk(scores, k=max(k_values), dim=-1).indices
                hit = (topk_indices == targets[:, None]).any(dim=-1)
                acc_hits[layer][f"top{args.top_k}"] += int(hit.sum().item())
                kurtosis_sum[layer] += float(excess_kurtosis(scores).sum().item())
                top1_series[layer].append(
                    topk_indices[:, 0].cpu().numpy()
                )
                del logits, scores, topk_indices
            acc_total += end - start
            kurtosis_count += end - start
            if (doc_index + 1) % 8 == 0:
                print(f"  processed {doc_index + 1}/{len(documents)} documents")

    # ------------------------------------------------------ aggregate curves
    lags = sorted(args.lags)
    per_layer: dict[int, dict] = {}
    for layer in readout_layers:
        series = np.concatenate(top1_series[layer])
        lag_stats = {}
        for lag in lags:
            observed, null = lag_agreement(series, lag)
            lag_stats[str(lag)] = {
                "observed_agreement": observed,
                "null_agreement": round(null, 6),
                "excess_over_null": observed - null,
            }
        per_layer[layer] = {
            "next_token_accuracy_topk": acc_hits[layer][f"top{args.top_k}"] / acc_total,
            "mean_excess_kurtosis": kurtosis_sum[layer] / kurtosis_count,
            "top1_lag_agreement": lag_stats,
        }

    # ------------------------------------------- lens-vector geometry (CPU)
    print("computing lens-vector geometry (dimensionality + CKA)...")
    unembed_weight = model._lm_head.weight.detach().float().cpu()
    gram_wu = unembed_weight.T @ unembed_weight
    grams = {
        layer: lens_vector_gram(lens.jacobians[layer].float().cpu(), gram_wu)
        for layer in source_layers
    }
    for layer in source_layers:
        eigvals = torch.linalg.eigvalsh(grams[layer]).clamp_min(0.0)
        eigvals = eigvals.flip(0)
        total_energy = float(eigvals.sum())
        ratio = float((eigvals.sum() ** 2) / (eigvals**2).sum())
        cumulative = torch.cumsum(eigvals, dim=0) / total_energy
        n_for_energy = int(torch.searchsorted(cumulative, torch.tensor(args.energy_threshold)).item()) + 1
        per_layer[layer]["jvector_effective_dim_pr"] = ratio
        per_layer[layer][f"jvector_dims_for_{int(args.energy_threshold * 100)}pct_energy"] = n_for_energy

    layer_list = source_layers
    cka = np.zeros((len(layer_list), len(layer_list)))
    self_norms = {
        layer: float(torch.linalg.matrix_norm(grams[layer], ord="fro")) for layer in layer_list
    }
    jacobians_cpu = {layer: lens.jacobians[layer].float().cpu() for layer in layer_list}
    for i, li in enumerate(layer_list):
        for j, lj in enumerate(layer_list):
            if j < i:
                continue
            # cross-Gram of J-lens vector column spaces: V_li^T V_lj = J_li^T G W_U J_lj
            cross = jacobians_cpu[li].T @ gram_wu @ jacobians_cpu[lj]
            value = float(torch.linalg.matrix_norm(cross, ord="fro") ** 2) / (
                self_norms[li] * self_norms[lj]
            )
            cka[i, j] = cka[j, i] = min(value, 1.0)

    # ------------------------------------------------------------- artifacts
    result = {
        "artifact_type": "jspace_layer_stats",
        "schema_version": 1,
        "model": args.model,
        "model_n_layers": model.n_layers,
        "lens_path": str(loaded.path),
        "lens_source": loaded.source,
        "readout_layers": readout_layers,
        "final_layer_is_model_logits": final_layer in readout_layers,
        "stats": {str(layer): per_layer[layer] for layer in readout_layers},
        "params": {
            "num_prompts": len(documents),
            "max_seq_len": args.max_seq_len,
            "skip_first_tokens": args.skip_first_tokens,
            "top_k": args.top_k,
            "lags": lags,
            "energy_threshold": args.energy_threshold,
            "scored_positions_per_doc": acc_total // max(len(documents), 1),
            "corpus": f"{WIKITEXT_REPO}:{WIKITEXT_FILE}",
            "note": (
                "(a)-(c) use lens.readout via jlens.apply; the final-layer row uses "
                "actual model logits as the motor-segment reference. (d) and CKA use "
                "J-lens vectors W_U J_l on raw unembedding weights (no final-norm "
                "rescaling); CKA compares J-lens column spaces."
            ),
        },
    }
    write_json(output_dir / "layer_stats.json", result, overwrite=True)
    np.savez_compressed(
        output_dir / "cka.npz",
        layers=np.array(layer_list),
        cka=cka,
    )
    write_metadata(
        output_dir / "layer_stats.json.metadata.json",
        {
            "lens_sha256": sha256_file(loaded.path),
            "corpus_parquet_sha256": sha256_file(parquet_path),
        },
        overwrite=True,
    )
    print(f"wrote {output_dir / 'layer_stats.json'}")


if __name__ == "__main__":
    main()
