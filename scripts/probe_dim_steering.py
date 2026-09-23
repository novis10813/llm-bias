"""Probe: Difference-in-Means (DIM) directional steering and evidence polarity on Qwen3.5-4B.

Tests token-wise residual steering along v_DIM[p] = mu_Top[p] - mu_Bottom[p] at Layer 15
instruction span across various evidence conditions (balanced, pure positive, pure negative)
and controls (random direction, anonymous prompt).

Outputs compact derived summaries and generation decisions. No raw activations are persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    """Render a frozen prompt with specific evidence bullets or zero evidence."""
    if evidence_bullets:
        evidence_text = "\n".join(f"- {b}" for b in evidence_bullets)
        evidence_block = f"{EVIDENCE_MARKER}\n\n{evidence_text}\n\n{EVIDENCE_CLOSE[2:]}"
    else:
        evidence_block = ""
    return (
        "Make a final investment decision for the given stock.\n\n"
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
        choices=["balanced", "pure_positive_2", "pure_positive_1", "pure_negative_2", "pure_negative_1", "zero_evidence", "all"],
        default="all",
        help="Evidence scenario to test",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["custom", "frozen"],
        default="custom",
        help="Renderer for the balanced scenario (frozen = canonical entity-to-dial template)",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        help="Alpha multiplier sweep",
    )
    parser.add_argument("--include-controls", action="store_true", default=True, help="Include random & anon controls")
    parser.add_argument(
        "--rand-seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Seeds for matched-norm random direction controls (one vector per seed)",
    )
    parser.add_argument(
        "--rand-alphas",
        nargs="+",
        type=float,
        default=[0.0, 2.0, 4.0, 6.0],
        help="Alpha sweep used for random direction controls",
    )
    parser.add_argument(
        "--persist-directions",
        default=None,
        help="Directory to persist v_DIM as a compact derived operator with provenance",
    )
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


def persist_v_dim(
    v_dim: torch.Tensor,
    out_dir: str,
    meta: dict[str, Any],
) -> Path:
    """Persist v_DIM (compact derived steering operator) with provenance. No raw activations."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensor_path = out / "v_dim.pt"
    tensor = v_dim.detach().to(torch.bfloat16).cpu()
    torch.save(tensor, tensor_path)
    prov = {
        "kind": "v_dim",
        "shape": list(v_dim.shape),
        "dtype_stored": "bfloat16",
        "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": meta.get("git_commit"),
        "script": "scripts/probe_dim_steering.py",
        "model": meta.get("model"),
        "heldout_run": meta.get("heldout_run"),
        "k_top_bottom": meta.get("k_top_bottom"),
        "layer": meta.get("layer"),
        "torch_version": torch.__version__,
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    return tensor_path


def run_evaluation(
    model: Any,
    tokenizer: Any,
    v_dim: torch.Tensor,
    company_by_ticker: dict[str, dict[str, Any]],
    target_tickers: list[str],
    scenarios: list[tuple[str, list[str]]],
    alphas: list[float],
    include_controls: bool = True,
    rand_seeds: list[int] | None = None,
    rand_alphas: list[float] | None = None,
    prompt_style: str = "custom",
) -> dict[str, Any]:
    rand_seeds = rand_seeds if rand_seeds is not None else [42]
    rand_alphas = rand_alphas if rand_alphas is not None else [0.0, 2.0, 4.0, 6.0]
    target_gen = InjectedModelAdapter(model, hf_model=getattr(model, "_hf_model", model))
    gen_config = GenerationConfig(max_new_tokens=48, temperature=0.0)
    final_layer = model.n_layers - 1

    # Matched-norm random vector controls: one independent direction per seed
    v_rand_list: list[tuple[int, torch.Tensor]] = []
    for seed in rand_seeds:
        gen = torch.Generator(device=v_dim.device).manual_seed(seed)
        v_rand = torch.randn(v_dim.shape, generator=gen, device=v_dim.device, dtype=v_dim.dtype)
        for p in range(v_dim.shape[0]):
            v_rand[p] = (v_rand[p] / v_rand[p].norm()) * v_dim[p].norm()
        v_rand_list.append((seed, v_rand))

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
        inst_marker = "Your final response must be a single, valid JSON object."
        if inst_marker not in prompt_text:
            raise ValueError(f"instruction marker missing in prompt: {prompt_text[:100]}")
        c_start = prompt_text.find(inst_marker)
        c_end = len(prompt_text)
        b_start = fmt.find(prompt_text)
        span = token_span(tokenizer, fmt, b_start + c_start, b_start + c_end, add_special_tokens=True)
        ids = tokenizer(fmt, return_tensors="pt").input_ids.to(model.input_device)
        buy_id, sell_id = answer_token_ids(tokenizer, fmt + '{"decision": "')
        start, end = span[0], span[1]
        assert (end - start) == 100, f"instruction span length {end - start} != 100"

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

    def render_prompt(sc_name: str, ticker: str, name: str, bullets: Sequence[str]) -> str:
        if sc_name == "balanced" and prompt_style == "frozen":
            return _render_frozen_prompt(ticker, name, order=0, reverse=False)
        return render_custom_prompt(ticker, name, bullets)

    for sc_name, bullets in scenarios:
        results["scenarios"][sc_name] = {}
        for ticker in target_tickers:
            name = company_by_ticker.get(ticker, {}).get("name", ticker)
            p_text = render_prompt(sc_name, ticker, name, bullets)
            sc_key = f"{ticker}_{sc_name}"
            results["scenarios"][sc_name][ticker] = eval_prompt(
                f"{ticker} ({name}) | {sc_name}",
                p_text,
                v_dim,
                alphas,
            )

        if include_controls and target_tickers:
            # Test Random Controls on first ticker (one arm per seed)
            first_ticker = target_tickers[0]
            name = company_by_ticker.get(first_ticker, {}).get("name", first_ticker)
            p_text = render_prompt(sc_name, first_ticker, name, bullets)
            for seed, v_rand in v_rand_list:
                results["scenarios"][sc_name][f"{first_ticker}_random_control_seed{seed}"] = eval_prompt(
                    f"{first_ticker} (Random 1D Control seed={seed}) | {sc_name}",
                    p_text,
                    v_rand,
                    rand_alphas,
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
        ("zero_evidence", []),
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
        rand_seeds=args.rand_seeds,
        rand_alphas=args.rand_alphas,
        prompt_style=args.prompt_style,
    )

    if args.persist_directions:
        import subprocess

        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            git_commit = None
        persist_v_dim(
            v_dim,
            args.persist_directions,
            {
                "git_commit": git_commit,
                "model": args.model,
                "heldout_run": args.heldout_run,
                "k_top_bottom": args.k_top_bottom,
                "layer": HELDOUT_LAYER,
            },
        )
        print(f"Persisted v_DIM to {args.persist_directions}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
