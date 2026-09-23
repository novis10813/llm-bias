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
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
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
from llm_bias.entity_to_dial.template import EVIDENCE_CLOSE, EVIDENCE_MARKER, NAME_LINE_PREFIX, TICKER_LINE_PREFIX

MACRO_SCENARIOS = {
    "macro_mixed": [
        "The Federal Reserve signaled a potential pause in interest rate hikes as labor markets show early cooling.",
        "Core inflation remains persistent at 4.8%, driven by housing and services costs, while goods prices contract.",
        "Corporate earnings revisions have diverged sharply: defensive dividend sectors gained 4% while cyclical growth fell 8%.",
    ],
    "macro_hawkish": [
        "The Federal Reserve raised interest rates by 50 bps, pushing benchmark borrowing costs to a 15-year high.",
        "Headline CPI inflation remains elevated at 6.2%, significantly eroding real household disposable income.",
        "Corporate bond yield spreads widened 80 bps as capital markets tightened liquidity and credit availability.",
    ],
    "macro_dovish": [
        "The Federal Reserve announced a 50 bps interest rate cut, signaling an accommodative monetary easing cycle.",
        "Headline CPI cooled to 2.1%, restoring real wage growth and supporting robust retail consumption.",
        "Benchmark equity indices reached record highs as corporate credit spreads compressed to historical lows.",
    ],
}


def render_custom_prompt(ticker: str, name: str, evidence_bullets: Sequence[str]) -> str:
    """Render a custom prompt with specific evidence bullets or zero evidence."""
    if evidence_bullets:
        evidence_text = "\n".join(f"- {b}" for b in evidence_bullets)
        evidence_block = f"{EVIDENCE_MARKER}\n\n{evidence_text}\n\n{EVIDENCE_CLOSE[2:]}"
    else:
        evidence_block = ""
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
        f"{TICKER_LINE_PREFIX}{ticker}]\n\n{NAME_LINE_PREFIX}{name}]\n\n"
        f"{evidence_block}"
        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
        "the following two keys:\n\n"
        '"decision": "buy" or "sell"\n\n'
        '"reason": A brief justification for your decision\n\n'
        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
        "should start with { and end with }. Do not include any other text."
    )


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
        "--evidence-mode",
        choices=["balanced", "zero_evidence", "macro_mixed", "macro_hawkish", "macro_dovish"],
        default="balanced",
        help="Evidence context for evaluation",
    )
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
    parser.add_argument(
        "--leave-out-sector",
        default=None,
        help="Exclude this sector from the contrast construction set (leave-one-sector-out)",
    )
    parser.add_argument(
        "--inject-layer",
        type=int,
        default=None,
        help="Layer for the residual intervention (default: heldout layer 15)",
    )
    parser.add_argument(
        "--measure-stance-cosine",
        action="store_true",
        default=False,
        help="Compute per-token cosine between cone axis b1 and the Top/Bottom difference-in-means direction",
    )
    parser.add_argument(
        "--centroid-dims",
        nargs="+",
        type=int,
        default=None,
        help="Evaluate centroids of the first j axes (dimension ablation; each must be <= k-cone-dim)",
    )
    parser.add_argument(
        "--persist-directions",
        default=None,
        help="Directory to persist the cone basis as a compact derived operator with provenance",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="Extract only (no evaluation); used for cosine measurement / persistence runs",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write compact derived results")
    return parser.parse_args()


def extract_concept_cone_basis(
    model: Any,
    tokenizer: Any,
    heldout_root: Path,
    k_pairs: int = 20,
    k_cone: int = 4,
    exclude_sector: str | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, dict[str, Any]],
    dict[str, torch.Tensor],
    list[str],
    list[str],
]:
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
    if exclude_sector:
        ranked_tickers = [
            (t, m) for t, m in ranked_tickers if company_by_ticker[t]["sector"] != exclude_sector
        ]
        print(f"Leave-one-sector-out: excluded sector '{exclude_sector}', {len(ranked_tickers)} companies remain")

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

    return cone_bases, company_by_ticker, states, top_tickers, bottom_tickers


def compute_stance_cosine(
    states: dict[str, torch.Tensor],
    top_tickers: list[str],
    bottom_tickers: list[str],
    cone_bases: torch.Tensor,
    k: int = 10,
) -> dict[str, Any]:
    """Per-token cosine between cone axis b1 and the non-demeaned Top/Bottom difference-in-means direction."""
    top_k = top_tickers[-k:]
    bot_k = bottom_tickers[:k]
    v_dim = torch.stack([states[t] for t in top_k], dim=0).mean(dim=0) - torch.stack(
        [states[t] for t in bot_k], dim=0
    ).mean(dim=0)
    b1 = cone_bases[..., 0]
    cos = (b1 * v_dim).sum(dim=-1) / (b1.norm(dim=-1) * v_dim.norm(dim=-1)).clamp_min(1e-8)
    cos_list = [float(x) for x in cos.cpu().tolist()]
    return {
        "k_reference": k,
        "top_reference": top_k,
        "bottom_reference": bot_k,
        "cos_mean": sum(cos_list) / len(cos_list),
        "cos_median": sorted(cos_list)[len(cos_list) // 2],
        "cos_min": min(cos_list),
        "cos_max": max(cos_list),
        "cos_per_token": cos_list,
    }


def persist_cone_basis(
    cone_bases: torch.Tensor,
    centroid: torch.Tensor,
    out_dir: str,
    meta: dict[str, Any],
) -> Path:
    """Persist cone basis + centroid (compact derived steering operator) with provenance. No raw activations."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensor_path = out / "cone.pt"
    payload = {
        "cone_bases": cone_bases.detach().to(torch.bfloat16).cpu(),
        "centroid": centroid.detach().to(torch.bfloat16).cpu(),
    }
    torch.save(payload, tensor_path)
    prov = {
        "kind": "concept_cone_basis",
        "shape": list(cone_bases.shape),
        "dtype_stored": "bfloat16",
        "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": meta.get("git_commit"),
        "script": "scripts/probe_concept_cone.py",
        "model": meta.get("model"),
        "heldout_run": meta.get("heldout_run"),
        "k_contrast_pairs": meta.get("k_contrast_pairs"),
        "k_cone": meta.get("k_cone"),
        "leave_out_sector": meta.get("leave_out_sector"),
        "torch_version": torch.__version__,
        "stance_cosine": meta.get("stance_cosine"),
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    return tensor_path


def run_cone_evaluation(
    model: Any,
    tokenizer: Any,
    cone_bases: torch.Tensor,
    company_by_ticker: dict[str, dict[str, Any]],
    target_tickers: Sequence[str],
    alphas: Sequence[float],
    evidence_mode: str = "balanced",
    eval_individual_rays: bool = False,
    inject_layer: int | None = None,
    centroid_dims: Sequence[int] | None = None,
) -> dict[str, Any]:
    target_gen = InjectedModelAdapter(model, hf_model=getattr(model, "_hf_model", model))
    gen_config = GenerationConfig(max_new_tokens=48, temperature=0.0)
    final_layer = model.n_layers - 1
    k_cone = cone_bases.shape[-1]
    inject_layer = inject_layer if inject_layer is not None else HELDOUT_LAYER

    # Cone centroid ray(s): full centroid by default, or first-j centroids for the dimension ablation
    rays: dict[str, torch.Tensor] = {}
    if centroid_dims:
        for j in centroid_dims:
            if not 1 <= j <= k_cone:
                raise ValueError(f"centroid dim {j} out of range [1, {k_cone}]")
            rays[f"centroid_dim{j}"] = cone_bases[..., :j].sum(dim=-1) / (float(j) ** 0.5)
    else:
        rays["cone_centroid"] = cone_bases.sum(dim=-1) / (float(k_cone) ** 0.5)

    def ray_label(name: str) -> str:
        if name == "cone_centroid":
            return f"{k_cone}D Concept Cone Centroid"
        return f"Cone Centroid first-{name.rsplit('dim', 1)[1]} axes"

    results: dict[str, Any] = {
        "k_cone": k_cone,
        "inject_layer": inject_layer,
        "evidence_mode": evidence_mode,
        "targets": {},
    }

    def eval_ticker_with_ray(
        ticker: str,
        ray_tensor: torch.Tensor,
        ray_label: str,
    ) -> list[dict[str, Any]]:
        c = company_by_ticker.get(ticker, {"name": ticker})
        if evidence_mode == "balanced":
            prompt = _render_frozen_prompt(ticker, c["name"], order=0, reverse=False)
        elif evidence_mode == "zero_evidence":
            prompt = render_custom_prompt(ticker, c["name"], [])
        else:
            bullets = MACRO_SCENARIOS[evidence_mode]
            prompt = render_custom_prompt(ticker, c["name"], bullets)

        fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
        inst_marker = "Your final response must be a single, valid JSON object."
        c_start = prompt.find(inst_marker)
        c_end = len(prompt)
        b_start = fmt.find(prompt)
        span = token_span(tokenizer, fmt, b_start + c_start, b_start + c_end, add_special_tokens=True)
        ids = tokenizer(fmt, return_tensors="pt").input_ids.to(model.input_device)
        buy_id, sell_id = answer_token_ids(tokenizer, fmt + '{"decision": "')
        start, end = span[0], span[1]
        assert (end - start) == 100, f"instruction span length {end - start} != 100"

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

            interventions = {inject_layer: transform} if a != 0.0 else None
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

    # Evaluate cone centroid ray(s) across all target tickers
    for ticker in target_tickers:
        results["targets"][ticker] = {
            ray_name: eval_ticker_with_ray(ticker, ray, ray_label(ray_name)) for ray_name, ray in rays.items()
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

    heldout_root = Path(args.heldout_run)
    cone_bases, company_by_ticker, states, top_tickers, bottom_tickers = extract_concept_cone_basis(
        model,
        tokenizer,
        heldout_root,
        k_pairs=args.k_contrast_pairs,
        k_cone=args.k_cone_dim,
        exclude_sector=args.leave_out_sector,
    )

    results: dict[str, Any] = {}

    if args.measure_stance_cosine:
        cosine = compute_stance_cosine(states, top_tickers, bottom_tickers, cone_bases)
        results["stance_cosine"] = cosine
        print(
            f"\nStance cosine b1 vs v_DIM (k={cosine['k_reference']}): "
            f"mean={cosine['cos_mean']:.4f} median={cosine['cos_median']:.4f} "
            f"min={cosine['cos_min']:.4f} max={cosine['cos_max']:.4f}"
        )

    if args.persist_directions:
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            git_commit = None
        w_full = cone_bases.sum(dim=-1) / (float(args.k_cone_dim) ** 0.5)
        persist_cone_basis(
            cone_bases,
            w_full,
            args.persist_directions,
            {
                "git_commit": git_commit,
                "model": args.model,
                "heldout_run": args.heldout_run,
                "k_contrast_pairs": args.k_contrast_pairs,
                "k_cone": args.k_cone_dim,
                "leave_out_sector": args.leave_out_sector,
                "stance_cosine": results.get("stance_cosine"),
            },
        )
        print(f"Persisted cone basis to {args.persist_directions}")

    target_tickers = args.target_tickers
    if target_tickers == ["all"]:
        manifest = json.loads((heldout_root / "prepare/cohort_manifest.json").read_text(encoding="utf-8"))
        target_tickers = [c["ticker"] for c in manifest["companies"]]
        print(f"Expanding --target-tickers all to {len(target_tickers)} cohort tickers")

    if args.skip_eval:
        results.update({"k_cone": args.k_cone_dim, "evidence_mode": args.evidence_mode, "targets": {}})
        print("Skipping evaluation (--skip-eval)")
    else:
        results.update(
            run_cone_evaluation(
                model,
                tokenizer,
                cone_bases,
                company_by_ticker,
                target_tickers,
                args.alphas,
                evidence_mode=args.evidence_mode,
                eval_individual_rays=args.eval_individual_rays,
                inject_layer=args.inject_layer,
                centroid_dims=args.centroid_dims,
            )
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
