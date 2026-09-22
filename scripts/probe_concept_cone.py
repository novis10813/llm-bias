"""Probe: Multi-dimensional Concept Cone steering via Token-wise Sector-Demeaned Contrastive SVD.

Protocol / Theory: Wollschläger et al. (2025) Concept Cones applied to 200-company held-out cohort.
Implements:
1. Sector Demeaning across 100 instruction tokens (removes industry semantics).
2. Slice-wise Contrastive SVD on Top-20 vs Bottom-20 extreme stance pairs.
3. Sign alignment to construct an orthonormal basis for the Buy Polyhedral Cone.
4. Multi-company evaluation of individual basis rays vs Cone Centroid.

Outputs compact derived summaries and generation decisions. No raw activations are persisted.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.adapter import InjectedModelAdapter
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.generation import GenerationConfig, generate_tokens
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import format_prompt, token_span
from llm_bias.entity_to_dial.dial_probe import answer_token_ids, margin_from_log_probs
from llm_bias.entity_to_dial.heldout_transfer import HELDOUT_LAYER, _render_frozen_prompt
from llm_bias.entity_to_dial.spans import instruction_char_span


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Token-wise Sector-Demeaned Contrastive SVD Concept Cone")
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b", help="Model checkpoint path")
    parser.add_argument(
        "--heldout-run",
        default="artifacts/qwen3.5-4b/entity-to-dial-heldout-transfer/runs/entity-to-dial-heldout-transfer-v1-01",
        help="Path to heldout transfer run to obtain cohort clean rankings",
    )
    parser.add_argument("--k-contrast-pairs", type=int, default=20, help="Number of Top vs Bottom extreme pairs")
    parser.add_argument("--k-cone-dim", type=int, default=4, help="Dimensionality of the Concept Cone")
    parser.add_argument(
        "--target-tickers",
        nargs="+",
        default=["MO", "FOXA", "CNC"],
        help="Tickers to evaluate",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 2.0, 3.0, 4.0, 5.0],
        help="Alpha multiplier sweep",
    )
    parser.add_argument(
        "--eval-individual-rays",
        action="store_true",
        default=False,
        help="Also evaluate each basis ray individually on the first target ticker",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write compact derived results")
    return parser.parse_args()


def extract_concept_cone_basis(
    model: Any,
    tokenizer: Any,
    heldout_root: Path,
    k_pairs: int = 20,
    k_cone: int = 4,
) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    """Extract slice-wise sector-demeaned contrastive SVD basis [100, 2560, k_cone]."""
    cohort_manifest = json.loads((heldout_root / "prepare/cohort_manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (heldout_root / "forward/records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    clean_records = [r for r in records if r["arm"] == "clean"]

    ticker_margins: dict[str, float] = {}
    company_by_ticker: dict[str, dict[str, Any]] = {c["ticker"]: c for c in cohort_manifest["companies"]}
    for r in clean_records:
        ticker_margins[r["target_ticker"]] = r["target_margin"]
        ticker_margins[r["source_ticker"]] = r["source_margin"]
    ranked_tickers = sorted(ticker_margins.items(), key=lambda x: x[1])

    top_tickers = [t for t, _ in ranked_tickers[-k_pairs:]]
    bottom_tickers = [t for t, _ in ranked_tickers[:k_pairs]]

    def get_company_span_state(ticker: str) -> tuple[torch.Tensor, str]:
        c = company_by_ticker[ticker]
        prompt = _render_frozen_prompt(ticker, c["name"], order=0, reverse=False)
        fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
        c_start, c_end = instruction_char_span(prompt)
        b_start = fmt.find(prompt)
        span = token_span(tokenizer, fmt, b_start + c_start, b_start + c_end, add_special_tokens=True)
        ids = tokenizer(fmt, return_tensors="pt").input_ids.to(model.input_device)
        with torch.no_grad():
            res = record_residuals(model, ids, [HELDOUT_LAYER])[HELDOUT_LAYER]
        return res[0, span[0]:span[1], :].float(), c["sector"]

    print(f"Extracting L15 instruction-span states for Top {k_pairs} and Bottom {k_pairs} companies...")
    states: dict[str, torch.Tensor] = {}
    sector_map: dict[str, str] = {}
    for t in top_tickers + bottom_tickers:
        s, sec = get_company_span_state(t)
        states[t] = s
        sector_map[t] = sec

    # Sector Demeaning
    sector_groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for t, sec in sector_map.items():
        sector_groups[sec].append(states[t])
    sector_means = {sec: torch.stack(group, dim=0).mean(dim=0) for sec, group in sector_groups.items()}

    deltas: list[torch.Tensor] = []
    for i in range(k_pairs):
        t_top = top_tickers[k_pairs - 1 - i]
        t_bot = bottom_tickers[i]
        demeaned_top = states[t_top] - sector_means[sector_map[t_top]]
        demeaned_bot = states[t_bot] - sector_means[sector_map[t_bot]]
        deltas.append(demeaned_top - demeaned_bot)

    delta_tensor = torch.stack(deltas, dim=0)  # [k_pairs, 100, 2560]

    # Slice-wise SVD across 100 instruction tokens
    cone_bases = torch.zeros(100, 2560, k_cone, device=model.input_device)
    for p in range(100):
        D_p = delta_tensor[:, p, :]
        U_p, S_p, Vh_p = torch.linalg.svd(D_p, full_matrices=False)
        Vk_p = Vh_p[:k_cone, :].T
        mean_p = D_p.mean(dim=0)
        for j in range(k_cone):
            cos_j = torch.dot(Vk_p[:, j], mean_p)
            sign_j = 1.0 if cos_j >= 0 else -1.0
            cone_bases[p, :, j] = (sign_j * Vk_p[:, j]).to(model.input_device)

    return cone_bases, company_by_ticker


def run_cone_evaluation(
    model: Any,
    tokenizer: Any,
    cone_bases: torch.Tensor,
    company_by_ticker: dict[str, dict[str, Any]],
    target_tickers: Sequence[str],
    alphas: Sequence[float],
    eval_individual_rays: bool = False,
) -> dict[str, Any]:
    target_gen = InjectedModelAdapter(model, hf_model=getattr(model, "_hf_model", model))
    gen_config = GenerationConfig(max_new_tokens=48, temperature=0.0)
    final_layer = model.n_layers - 1
    k_cone = cone_bases.shape[-1]

    # Concept Cone Centroid across tokens
    w_cone = cone_bases.sum(dim=-1) / (float(k_cone) ** 0.5)

    results: dict[str, Any] = {
        "k_cone": k_cone,
        "targets": {},
    }

    def eval_ticker_with_ray(
        ticker: str,
        ray_tensor: torch.Tensor,
        ray_label: str,
    ) -> list[dict[str, Any]]:
        c = company_by_ticker[ticker]
        prompt = _render_frozen_prompt(ticker, c["name"], order=0, reverse=False)
        fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
        c_start, c_end = instruction_char_span(prompt)
        b_start = fmt.find(prompt)
        span = token_span(tokenizer, fmt, b_start + c_start, b_start + c_end, add_special_tokens=True)
        ids = tokenizer(fmt, return_tensors="pt").input_ids.to(model.input_device)
        buy_id, sell_id = answer_token_ids(tokenizer, fmt + '{"decision": "')
        start, end = span[0], span[1]

        rows = []
        print(f"\n========================================================")
        print(f"Subject: {ticker} ({c['name']}) | Ray: {ray_label}")
        print(f"========================================================")

        for a in alphas:
            shift = (a * ray_tensor).to(dtype=torch.bfloat16, device=model.input_device)
            def transform(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.ndim == 3 and tensor.shape[1] == 1: return tensor
                out = tensor.clone()
                out[:, start:end, :] = out[:, start:end, :] + shift.unsqueeze(0)
                return out

            interventions = {HELDOUT_LAYER: transform} if a != 0.0 else None
            if interventions:
                with residual_interventions(model, interventions):
                    res = record_residuals(model, ids, [final_layer])[final_layer]
                    seq = generate_tokens(target_gen, ids, gen_config)
            else:
                with torch.no_grad():
                    res = record_residuals(model, ids, [final_layer])[final_layer]
                    seq = generate_tokens(target_gen, ids, gen_config)

            margin = margin_from_log_probs(fp32_next_token_log_probs(model, res[:, -1, :]), buy_id, sell_id)
            gen_text = tokenizer.decode(seq[0, ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
            match = re.search(r'"decision"\s*:\s*"\s*(buy|sell)\s*"', gen_text, flags=re.IGNORECASE)
            dec = match.group(1).lower() if match else "unparsed"

            print(f"  alpha={a:4.1f} | margin={margin:+.3f} | dec={dec:4s} | {repr(gen_text[:60])}")
            rows.append({
                "alpha": a,
                "margin": margin,
                "decision": dec,
                "generated_text": gen_text,
            })
        return rows

    # Evaluate Cone Centroid across all target tickers
    for ticker in target_tickers:
        results["targets"][ticker] = {
            "cone_centroid": eval_ticker_with_ray(ticker, w_cone, f"{k_cone}D Concept Cone Centroid"),
        }

    # Optionally evaluate individual rays on first ticker
    if eval_individual_rays and target_tickers:
        first = target_tickers[0]
        results["targets"][first]["individual_rays"] = {}
        for j in range(k_cone):
            ray_j = cone_bases[:, :, j]
            results["targets"][first]["individual_rays"][f"b_{j+1}"] = eval_ticker_with_ray(
                first, ray_j, f"Basis Ray b_{j+1}"
            )

    return results


def main() -> None:
    args = parse_args()
    print(f"Loading model from {args.model}...")
    model, tokenizer, _ = load_model(args.model, dtype=None)

    cone_bases, company_by_ticker = extract_concept_cone_basis(
        model,
        tokenizer,
        Path(args.heldout_run),
        k_pairs=args.k_contrast_pairs,
        k_cone=args.k_cone_dim,
    )

    results = run_cone_evaluation(
        model,
        tokenizer,
        cone_bases,
        company_by_ticker,
        args.target_tickers,
        args.alphas,
        eval_individual_rays=args.eval_individual_rays,
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
