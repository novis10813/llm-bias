"""Probe: Difference-in-Means (DIM) directional steering and evidence polarity on Qwen3.5-4B.

Tests token-wise residual steering along v_DIM[p] = mu_Top[p] - mu_Bottom[p] at Layer 15
instruction span across various evidence conditions (balanced, pure positive, pure negative)
and controls (random direction, anonymous prompt).

Outputs compact derived summaries and generation decisions. No raw activations are persisted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
from llm_bias.entity_to_dial.spans import anonymous_prompt, instruction_char_span
from llm_bias.entity_to_dial.template import EVIDENCE_CLOSE, EVIDENCE_MARKER, NAME_LINE_PREFIX, TICKER_LINE_PREFIX

POS_BULLETS = [
    "Q3 revenue grew 14% year over year, exceeding analyst consensus by 3%.",
    "Free cash flow reached a record quarterly high, up 22% year over year.",
]

NEG_BULLETS = [
    "Gross margin contracted 300 basis points due to rising input costs.",
    "Full-year revenue guidance was revised downward by 6%.",
]


def render_custom_prompt(ticker: str, name: str, evidence_bullets: Sequence[str]) -> str:
    """Render a frozen prompt with specific evidence bullets."""
    evidence_text = "\n".join(f"- {b}" for b in evidence_bullets)
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
        f"{TICKER_LINE_PREFIX}{ticker}]\n\n{NAME_LINE_PREFIX}{name}]\n\n"
        f"{EVIDENCE_MARKER}\n\n{evidence_text}\n\n"
        f"{EVIDENCE_CLOSE[2:]}"
        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
        "the following two keys:\n\n"
        '"decision": "buy" or "sell"\n\n'
        '"reason": A brief justification for your decision\n\n'
        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
        "should start with { and end with }. Do not include any other text."
    )


def _generation_safe_transform(transform: Any) -> Any:
    """Ensure prefill hook does not alter 1-token decode steps."""
    def apply(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return tensor
        return transform(tensor)
    return apply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe DIM directional steering across evidence conditions")
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b", help="Model checkpoint path")
    parser.add_argument(
        "--heldout-run",
        default="artifacts/qwen3.5-4b/entity-to-dial-heldout-transfer/runs/entity-to-dial-heldout-transfer-v1-01",
        help="Path to heldout transfer run to obtain cohort clean rankings",
    )
    parser.add_argument("--k-top-bottom", type=int, default=10, help="Number of Top/Bottom companies for v_DIM")
    parser.add_argument("--target-tickers", nargs="+", default=["MO"], help="Tickers to evaluate")
    parser.add_argument(
        "--evidence-mode",
        choices=["balanced", "pure_positive_2", "pure_positive_1", "pure_negative_2", "pure_negative_1", "all"],
        default="all",
        help="Evidence scenario to test",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        help="Alpha multiplier sweep",
    )
    parser.add_argument("--include-controls", action="store_true", default=True, help="Include random & anon controls")
    parser.add_argument("--output-json", default=None, help="Optional path to write compact derived results")
    return parser.parse_args()


def extract_v_dim(
    model: Any,
    tokenizer: Any,
    heldout_root: Path,
    k: int,
) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    """Extract token-wise v_DIM[p] = mu_Top[p] - mu_Bottom[p] on L15 instruction span."""
    cohort_manifest = json.loads((heldout_root / "prepare/cohort_manifest.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (heldout_root / "forward/records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    clean_records = [r for r in records if r["arm"] == "clean"]
    
    ticker_margins: dict[str, float] = {}
    company_by_ticker: dict[str, dict[str, Any]] = {c["ticker"]: c for c in cohort_manifest["companies"]}
    for r in clean_records:
        ticker_margins[r["target_ticker"]] = r["target_margin"]
        ticker_margins[r["source_ticker"]] = r["source_margin"]
    ranked_tickers = sorted(ticker_margins.items(), key=lambda x: x[1])

    bottom_k = [t for t, _ in ranked_tickers[:k]]
    top_k = [t for t, _ in ranked_tickers[-k:]]

    def get_span_state(ticker: str, name: str) -> torch.Tensor:
        prompt = _render_frozen_prompt(ticker, name, order=0, reverse=False)
        fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
        c_start, c_end = instruction_char_span(prompt)
        b_start = fmt.find(prompt)
        span = token_span(tokenizer, fmt, b_start + c_start, b_start + c_end, add_special_tokens=True)
        ids = tokenizer(fmt, return_tensors="pt").input_ids.to(model.input_device)
        with torch.no_grad():
            res = record_residuals(model, ids, [HELDOUT_LAYER])[HELDOUT_LAYER]
        return res[0, span[0]:span[1], :].float()

    top_states = [get_span_state(t, company_by_ticker[t]["name"]) for t in top_k]
    bottom_states = [get_span_state(t, company_by_ticker[t]["name"]) for t in bottom_k]

    v_dim = torch.stack(top_states, dim=0).mean(dim=0) - torch.stack(bottom_states, dim=0).mean(dim=0)
    return v_dim, company_by_ticker


def run_evaluation(
    model: Any,
    tokenizer: Any,
    v_dim: torch.Tensor,
    company_by_ticker: dict[str, dict[str, Any]],
    target_tickers: list[str],
    scenarios: list[tuple[str, list[str]]],
    alphas: list[float],
    include_controls: bool = True,
) -> dict[str, Any]:
    target_gen = InjectedModelAdapter(model, hf_model=getattr(model, "_hf_model", model))
    gen_config = GenerationConfig(max_new_tokens=48, temperature=0.0)
    final_layer = model.n_layers - 1

    # Matched-norm random vector control
    torch.manual_seed(42)
    v_rand = torch.randn_like(v_dim)
    for p in range(v_dim.shape[0]):
        v_rand[p] = (v_rand[p] / v_rand[p].norm()) * v_dim[p].norm()

    results: dict[str, Any] = {
        "v_dim_norm_per_token_mean": float(v_dim.norm(dim=-1).mean().item()),
        "scenarios": {},
    }

    def eval_prompt(
        label: str,
        prompt_text: str,
        vec: torch.Tensor,
        alpha_list: list[float],
    ) -> list[dict[str, Any]]:
        fmt = format_prompt(tokenizer, prompt_text, use_chat_template=True, enable_thinking=False)
        c_start, c_end = instruction_char_span(prompt_text)
        b_start = fmt.find(prompt_text)
        span = token_span(tokenizer, fmt, b_start + c_start, b_start + c_end, add_special_tokens=True)
        ids = tokenizer(fmt, return_tensors="pt").input_ids.to(model.input_device)
        buy_id, sell_id = answer_token_ids(tokenizer, fmt + '{"decision": "')
        start, end = span[0], span[1]

        rows = []
        print(f"\n--- {label} ---")
        for a in alpha_list:
            shift = (a * vec).to(dtype=torch.bfloat16, device=model.input_device)
            def transform(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.ndim == 3 and tensor.shape[1] == 1:
                    return tensor
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

            print(f"  alpha={a:+4.1f} | margin={margin:+.3f} | dec={dec:6s} | {repr(gen_text[:60])}")
            rows.append({
                "alpha": a,
                "margin": margin,
                "decision": dec,
                "generated_text": gen_text,
            })
        return rows

    for sc_name, bullets in scenarios:
        results["scenarios"][sc_name] = {}
        for ticker in target_tickers:
            name = company_by_ticker[ticker]["name"]
            p_text = render_custom_prompt(ticker, name, bullets)
            sc_key = f"{ticker}_{sc_name}"
            results["scenarios"][sc_name][ticker] = eval_prompt(
                f"{ticker} ({name}) | {sc_name}",
                p_text,
                v_dim,
                alphas,
            )

        if include_controls and target_tickers:
            # Test Random Control on first ticker
            first_ticker = target_tickers[0]
            name = company_by_ticker[first_ticker]["name"]
            p_text = render_custom_prompt(first_ticker, name, bullets)
            results["scenarios"][sc_name][f"{first_ticker}_random_control"] = eval_prompt(
                f"{first_ticker} (Random 1D Control) | {sc_name}",
                p_text,
                v_rand,
                [0.0, 2.0, 4.0, 6.0],
            )

            # Test Anonymous prompt
            anon_p = anonymous_prompt(p_text, first_ticker, name)
            results["scenarios"][sc_name]["anonymous_control"] = eval_prompt(
                f"ANONYMOUS (Identity-stripped) | {sc_name}",
                anon_p,
                v_dim,
                alphas,
            )

    return results


def main() -> None:
    args = parse_args()
    print(f"Loading model from {args.model}...")
    model, tokenizer, _ = load_model(args.model, dtype=None)

    print(f"Extracting v_DIM from {args.heldout_run} (Top/Bottom {args.k_top_bottom})...")
    v_dim, company_by_ticker = extract_v_dim(model, tokenizer, Path(args.heldout_run), args.k_top_bottom)

    all_scenarios = [
        ("balanced", POS_BULLETS + NEG_BULLETS),
        ("pure_positive_2", POS_BULLETS),
        ("pure_positive_1", [POS_BULLETS[0]]),
        ("pure_negative_2", NEG_BULLETS),
        ("pure_negative_1", [NEG_BULLETS[1]]),
    ]
    if args.evidence_mode != "all":
        scenarios = [s for s in all_scenarios if s[0] == args.evidence_mode]
    else:
        scenarios = all_scenarios

    results = run_evaluation(
        model,
        tokenizer,
        v_dim,
        company_by_ticker,
        args.target_tickers,
        scenarios,
        args.alphas,
        include_controls=args.include_controls,
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
