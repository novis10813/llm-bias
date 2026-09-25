"""P2 operator comparison run on Qwen3.5-4B.

Protocol: docs/concept-cone-steering/details/proposal-p2-operator-comparison-v1.md
(frozen). Compares three residual steering operators — single-neuron dial
(L15/N8490), 1D token-wise DIM, and 4D concept cone centroid with 1/2/3-axis
ablations — under one matched per-token norm grid, plus multi-seed matched-norm
random controls and a paired anonymous-prompt control.

Every record is one FP32-tail fixed-token margin forward plus one greedy JSON
generation (max 48 tokens, temperature 0). Only compact derived JSON is
persisted; no raw activations, residuals, or KV caches.

Arms (protocol §6):
  A  {dial, DIM, cone centroid} x 10 Bottom-10 targets x 7 norm points
  B  cone axis ablations c1/c2/c3 x {MO, CNC, FOXA} x 7
  C  5 matched-norm random seeds x {MO, CNC, FOXA} x 7
  D  {DIM, cone centroid} x 1 anonymous prompt x 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from llm_bias.core.continuation_scoring import fp32_next_token_log_probs
from llm_bias.core.inference.adapter import InjectedModelAdapter
from llm_bias.core.inference.forward import record_residuals
from llm_bias.core.inference.generation import GenerationConfig, generate_tokens
from llm_bias.core.inference.interventions import residual_interventions
from llm_bias.core.inference.mlp import dense_down_projection
from llm_bias.core.inference.mlp_addition import mlp_addition
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import format_prompt, token_span
from llm_bias.entity_to_dial.dial_probe import answer_token_ids, margin_from_log_probs
from llm_bias.entity_to_dial.heldout_transfer import HELDOUT_LAYER, _render_frozen_prompt
from llm_bias.entity_to_dial.spans import anonymous_prompt

# Frozen protocol constants (proposal-p2-operator-comparison-v1.md §3–§6).
DEFAULT_MODEL = ".cache/models/qwen3.5-4b"
HELDOUT_RUN = "artifacts/qwen3.5-4b/entity-to-dial-heldout-transfer/runs/entity-to-dial-heldout-transfer-v1-01"
DIM_DIR = "artifacts/qwen3.5-4b/concept-cone-steering/directions/smoke-vdim"
CONE_DIR = "artifacts/qwen3.5-4b/concept-cone-steering/directions/smoke-cone4d"
EXPECTED_DIM_SHA = "991a70f597c7547fff247a0384eac3887923ac3c31b0b9adef6bdaf46eab2cca"
EXPECTED_CONE_SHA = "b872dc32f4812843af8acdb90fb08dcdfc75a1133d74c039fe5bff8e0c39b479"
N_DIM_FROZEN = 0.20902258157730103  # v_dim_norm_per_token_mean from the frozen3 pilot
DIAL_LAYER, DIAL_NEURON = 15, 8490
INSTRUCTION_MARKER = "Your final response must be a single, valid JSON object."

BOTTOM10 = ["MO", "CNC", "FOXA", "BR", "ERIE", "TSN", "BAX", "TPR", "FOX", "COO"]
ABLAB_TARGETS = ["MO", "CNC", "FOXA"]
RANDOM_SEEDS = [42, 1, 2, 3, 4]
ALPHA_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
SMOKE_ALPHA_GRID = [0.0, 2.0, 4.0]
SMOKE_SEEDS = [42]

DECISION_RE = re.compile(r'"decision"\s*:\s*"\s*(buy|sell)\s*"', flags=re.IGNORECASE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P2 steering operator comparison (frozen protocol v1)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default=None, help="Run directory (manifest.json + arms/)")
    p.add_argument("--smoke", action="store_true", help="Smoke subset: MO only, alpha {0,2,4}, 1 random seed")
    p.add_argument("--heldout-run", default=HELDOUT_RUN)
    p.add_argument("--dim-dir", default=DIM_DIR)
    p.add_argument("--cone-dir", default=CONE_DIR)
    return p.parse_args()


def load_directions(dim_dir: Path, cone_dir: Path, device: torch.device) -> dict[str, torch.Tensor]:
    dim_path = dim_dir / "v_dim.pt"
    cone_path = cone_dir / "cone.pt"
    if not dim_path.exists() or not cone_path.exists():
        raise FileNotFoundError(f"direction artifacts missing: {dim_path} / {cone_path}")
    dim_sha, cone_sha = sha256_file(dim_path), sha256_file(cone_path)
    if dim_sha != EXPECTED_DIM_SHA:
        raise ValueError(f"v_dim.pt sha256 mismatch: {dim_sha} != {EXPECTED_DIM_SHA}")
    if cone_sha != EXPECTED_CONE_SHA:
        raise ValueError(f"cone basis sha256 mismatch: {cone_sha} != {EXPECTED_CONE_SHA}")
    v_dim = torch.load(dim_path, map_location="cpu").float().to(device)
    cone_payload = torch.load(cone_path, map_location="cpu")
    if not isinstance(cone_payload, dict) or "cone_bases" not in cone_payload or "centroid" not in cone_payload:
        raise ValueError("cone.pt must be a dict with 'cone_bases' and 'centroid'")
    basis = cone_payload["cone_bases"].float().to(device)
    stored_centroid = cone_payload["centroid"].float().to(device)
    if v_dim.shape != (100, 2560):
        raise ValueError(f"v_dim shape {tuple(v_dim.shape)} != (100, 2560)")
    if basis.shape != (100, 2560, 4):
        raise ValueError(f"cone basis shape {tuple(basis.shape)} != (100, 2560, 4)")
    computed_centroid = basis.sum(dim=-1) / (4 ** 0.5)
    max_centroid_diff = float((computed_centroid - stored_centroid).abs().max())
    if max_centroid_diff > 1e-2:
        raise ValueError(f"stored centroid inconsistent with sum(bases)/sqrt(4): max diff {max_centroid_diff}")
    return {"v_dim": v_dim, "basis": basis}


def unit_norms(v_dim: torch.Tensor, basis: torch.Tensor, model: Any) -> dict[str, float]:
    n_dim = float(v_dim.norm(dim=-1).mean())
    if abs(n_dim - N_DIM_FROZEN) > 1e-4:
        raise ValueError(f"n_dim {n_dim} inconsistent with frozen value {N_DIM_FROZEN}")
    c4 = basis.sum(dim=-1) / math.sqrt(4)
    n_cone = float(c4.norm(dim=-1).mean())
    down = dense_down_projection(model.layers[DIAL_LAYER])
    n_dial = float(down.weight[:, DIAL_NEURON].float().norm())
    return {"n_dim": n_dim, "n_cone": n_cone, "n_dial": n_dial}


def cone_centroid(basis: torch.Tensor, k: int) -> torch.Tensor:
    return basis[..., :k].sum(dim=-1) / math.sqrt(k)


def make_random(v_dim: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=v_dim.device).manual_seed(seed)
    v_rand = torch.randn(v_dim.shape, generator=gen, device=v_dim.device, dtype=v_dim.dtype)
    for pos in range(v_rand.shape[0]):
        v_rand[pos] = (v_rand[pos] / v_rand[pos].norm()) * v_dim[pos].norm()
    return v_rand


class TargetContext:
    """Per-target precomputed prompt encoding (protocol §2)."""

    def __init__(self, model: Any, tokenizer: Any, ticker: str, name: str, anon: bool = False) -> None:
        self.ticker = ticker
        self.label = f"ANONYMOUS({ticker})" if anon else ticker
        prompt = _render_frozen_prompt(ticker, name, order=0, reverse=False)
        if anon:
            prompt = anonymous_prompt(prompt, ticker, name)
        self.prompt_text = prompt
        self.prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        inst_marker = INSTRUCTION_MARKER
        if inst_marker not in prompt:
            raise ValueError("instruction marker missing in rendered prompt")
        fmt = format_prompt(tokenizer, prompt, use_chat_template=True, enable_thinking=False)
        c_start = prompt.find(inst_marker)
        b_start = fmt.find(prompt)
        self.span = token_span(tokenizer, fmt, b_start + c_start, b_start + len(prompt), add_special_tokens=True)
        self.ids = tokenizer(fmt, return_tensors="pt").input_ids.to(model.input_device)
        buy_id, sell_id = answer_token_ids(tokenizer, fmt + '{"decision": "')
        self.buy_id, self.sell_id = buy_id, sell_id
        self.final_layer = model.n_layers - 1
        if self.span[1] - self.span[0] != 100:
            raise ValueError(f"instruction span length {self.span[1] - self.span[0]} != 100")


def _decode_decision(gen_text: str) -> str:
    match = DECISION_RE.search(gen_text)
    return match.group(1).lower() if match else "unparsed"


def eval_vector(
    model: Any,
    tokenizer: Any,
    target_gen: Any,
    gen_config: Any,
    ctx: TargetContext,
    vec: torch.Tensor,
    dose: float,
) -> dict[str, Any]:
    """One dose point of a span-injected vector operator (DIM / cone / random)."""
    start, end = ctx.span
    shift = (dose * vec).to(dtype=torch.bfloat16, device=model.input_device)

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            return tensor
        out = tensor.clone()
        out[:, start:end, :] = out[:, start:end, :] + shift.unsqueeze(0)
        return out

    interventions = {HELDOUT_LAYER: transform} if dose != 0.0 else None
    if interventions:
        with residual_interventions(model, interventions):
            res = record_residuals(model, ctx.ids, [ctx.final_layer])[ctx.final_layer]
            seq = generate_tokens(target_gen, ctx.ids, gen_config)
    else:
        with torch.no_grad():
            res = record_residuals(model, ctx.ids, [ctx.final_layer])[ctx.final_layer]
            seq = generate_tokens(target_gen, ctx.ids, gen_config)

    margin = margin_from_log_probs(fp32_next_token_log_probs(model, res[:, -1, :]), ctx.buy_id, ctx.sell_id)
    gen_text = tokenizer.decode(seq[0, ctx.ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
    return {"margin": margin, "decision": _decode_decision(gen_text), "generated_text": gen_text}


def eval_dial(
    model: Any,
    tokenizer: Any,
    target_gen: Any,
    gen_config: Any,
    ctx: TargetContext,
    delta: float,
) -> dict[str, Any]:
    """One dose point of the L15/N8490 dial (all-position, incl. decode)."""
    with mlp_addition(model, DIAL_LAYER, DIAL_NEURON, delta):
        res = record_residuals(model, ctx.ids, [ctx.final_layer])[ctx.final_layer]
        seq = generate_tokens(target_gen, ctx.ids, gen_config)
    margin = margin_from_log_probs(fp32_next_token_log_probs(model, res[:, -1, :]), ctx.buy_id, ctx.sell_id)
    gen_text = tokenizer.decode(seq[0, ctx.ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
    return {"margin": margin, "decision": _decode_decision(gen_text), "generated_text": gen_text}


def flip_point(rows: list[dict[str, Any]]) -> float | None:
    """First norm point where the greedy decision becomes 'buy' (protocol §9)."""
    for row in rows:
        if row["decision"] == "buy":
            return row["N"]
    return None


def main() -> None:
    args = parse_args()
    smoke = args.smoke
    started = datetime.now(timezone.utc)

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        args.output_dir = (
            f"artifacts/qwen3.5-4b/concept-cone-steering/runs/smoke-p2-{stamp}"
            if smoke
            else f"artifacts/qwen3.5-4b/concept-cone-steering/runs/operator-comparison-p2-v1-01"
        )
    out_root = Path(args.output_dir)
    (out_root / "arms").mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model} ...")
    model, tokenizer, _ = load_model(args.model, dtype=None)
    target_gen = InjectedModelAdapter(model, hf_model=getattr(model, "_hf_model", model))
    gen_config = GenerationConfig(max_new_tokens=48, temperature=0.0)

    dirs = load_directions(Path(args.dim_dir), Path(args.cone_dir), model.input_device)
    v_dim, basis = dirs["v_dim"], dirs["basis"]
    norms = unit_norms(v_dim, basis, model)
    n_dim, n_cone, n_dial = norms["n_dim"], norms["n_cone"], norms["n_dial"]

    alphas = SMOKE_ALPHA_GRID if smoke else ALPHA_GRID
    seeds = SMOKE_SEEDS if smoke else RANDOM_SEEDS
    targets = [ABLAB_TARGETS[0]] if smoke else BOTTOM10
    ablab_targets = [ABLAB_TARGETS[0]] if smoke else ABLAB_TARGETS
    grid = [a * n_dim for a in alphas]

    cohort_manifest = json.loads(
        (Path(args.heldout_run) / "prepare/cohort_manifest.json").read_text(encoding="utf-8")
    )
    names = {c["ticker"]: c["name"] for c in cohort_manifest["companies"]}
    for t in targets + ablab_targets:
        if t not in names:
            raise ValueError(f"target {t} missing from cohort manifest")

    print("Rendering target contexts ...")
    ctxs = {t: TargetContext(model, tokenizer, t, names[t]) for t in targets}
    anon_ctx = TargetContext(model, tokenizer, "MO", names["MO"], anon=True)

    def dose_for(operator: str, N: float) -> float:
        if operator == "dial":
            return N / n_dial
        if operator == "cone_centroid":
            return N / n_cone
        return N / n_dim  # dim / ablation / random are all span vectors

    c_centroids = {j: cone_centroid(basis, j) for j in (1, 2, 3, 4)}
    n_centroids = {j: float(c_centroids[j].norm(dim=-1).mean()) for j in (1, 2, 3, 4)}
    random_vecs = {s: make_random(v_dim, s) for s in seeds}

    arm_a: list[dict[str, Any]] = []
    arm_b: list[dict[str, Any]] = []
    arm_c: list[dict[str, Any]] = []
    arm_d: list[dict[str, Any]] = []
    n_records = 0

    def finish(rec: dict[str, Any]) -> None:
        nonlocal n_records
        n_records += 1
        print(
            f"[{n_records:3d}] {rec['arm']}/{rec['operator']:<13s} {rec['target']:<16s} "
            f"N={rec['N']:.3f} dose={rec['dose']:+.3f} margin={rec['margin']:+.3f} dec={rec['decision']:8s}"
        )

    # Arm A: dial / DIM / cone centroid x targets x norm grid
    for ticker in targets:
        ctx = ctxs[ticker]
        for operator in ("dial", "dim", "cone_centroid"):
            vec = v_dim if operator == "dim" else c_centroids[4]
            unit = n_dim if operator == "dim" else (n_dial if operator == "dial" else n_cone)
            for a, N in zip(alphas, grid):
                dose = dose_for(operator, N)
                out = eval_dial(model, tokenizer, target_gen, gen_config, ctx, dose) if operator == "dial" \
                    else eval_vector(model, tokenizer, target_gen, gen_config, ctx, vec, dose)
                rec = {
                    "arm": "A", "operator": operator, "target": ticker, "seed": None,
                    "alpha_dim_equiv": a, "N": N, "dose": dose,
                    "achieved_per_token_norm": abs(dose) * unit,
                    **out,
                }
                arm_a.append(rec)
                finish(rec)

    # Arm B: cone axis ablations (first j axes) x MO/CNC/FOXA x grid
    for ticker in ablab_targets:
        ctx = ctxs[ticker]
        for j in (1, 2, 3):
            vec = c_centroids[j]
            for a, N in zip(alphas, grid):
                dose = N / n_centroids[j]
                out = eval_vector(model, tokenizer, target_gen, gen_config, ctx, vec, dose)
                rec = {
                    "arm": "B", "operator": f"cone_c{j}", "target": ticker, "seed": None,
                    "alpha_dim_equiv": a, "N": N, "dose": dose,
                    "achieved_per_token_norm": abs(dose) * n_centroids[j],
                    **out,
                }
                arm_b.append(rec)
                finish(rec)

    # Arm C: matched-norm random controls x MO/CNC/FOXA x grid
    for seed in seeds:
        for ticker in ablab_targets:
            ctx = ctxs[ticker]
            for a, N in zip(alphas, grid):
                dose = N / n_dim  # random vectors are norm-matched to DIM by construction
                out = eval_vector(model, tokenizer, target_gen, gen_config, ctx, random_vecs[seed], dose)
                rec = {
                    "arm": "C", "operator": "random", "target": ticker, "seed": seed,
                    "alpha_dim_equiv": a, "N": N, "dose": dose,
                    "achieved_per_token_norm": abs(dose) * n_dim,
                    **out,
                }
                arm_c.append(rec)
                finish(rec)

    # Arm D: anonymous prompt x {DIM, cone centroid} x grid
    for operator in ("dim", "cone_centroid"):
        vec = v_dim if operator == "dim" else c_centroids[4]
        unit = n_dim if operator == "dim" else n_cone
        for a, N in zip(alphas, grid):
            dose = dose_for(operator, N)
            out = eval_vector(model, tokenizer, target_gen, gen_config, anon_ctx, vec, dose)
            rec = {
                "arm": "D", "operator": operator, "target": "anonymous(MO)", "seed": None,
                "alpha_dim_equiv": a, "N": N, "dose": dose,
                "achieved_per_token_norm": abs(dose) * unit,
                **out,
            }
            arm_d.append(rec)
            finish(rec)

    # Gates (protocol §8)
    notes: list[str] = []
    gates: dict[str, Any] = {}
    expected_counts = {
        "A": len(targets) * 3 * len(grid),
        "B": len(ablab_targets) * 3 * len(grid),
        "C": len(seeds) * len(ablab_targets) * len(grid),
        "D": 2 * len(grid),
    }
    all_rows = {"A": arm_a, "B": arm_b, "C": arm_c, "D": arm_d}
    gates["G3_count_and_schema"] = all(
        len(all_rows[k]) == expected_counts[k] for k in expected_counts
    ) and all(
        math.isfinite(r["margin"]) and all(k in r for k in
            ("arm", "operator", "target", "N", "dose", "achieved_per_token_norm", "margin", "decision", "generated_text"))
        for rows in all_rows.values() for r in rows
    )

    # G1: per-target dose=0 records agree across operators and with a clean forward
    g1_ok = True
    for ticker in targets:
        ctx = ctxs[ticker]
        with torch.no_grad():
            res = record_residuals(model, ctx.ids, [ctx.final_layer])[ctx.final_layer]
        clean = margin_from_log_probs(fp32_next_token_log_probs(model, res[:, -1, :]), ctx.buy_id, ctx.sell_id)
        zero_rows = [r for r in arm_a if r["target"] == ticker and r["alpha_dim_equiv"] == 0.0]
        if len(zero_rows) != 3:
            g1_ok = False
            continue
        deltas = [abs(r["margin"] - clean) for r in zero_rows]
        if any(d > 1e-3 for d in deltas):
            g1_ok = False
            notes.append(f"G1: {ticker} dose-0 margins deviate from clean by {deltas}")
    gates["G1_dose0_agrees_clean"] = g1_ok

    # G2: clean decision should be 'sell' for every named target
    g2_exceptions = []
    for ticker in targets:
        row = next(r for r in arm_a if r["target"] == ticker and r["operator"] == "dim" and r["alpha_dim_equiv"] == 0.0)
        if row["decision"] != "sell":
            g2_exceptions.append(f"{ticker}: clean decision {row['decision']} (margin {row['margin']:+.3f})")
    gates["G2_clean_decision_sell"] = not g2_exceptions
    if g2_exceptions:
        notes.extend(f"G2: {e}" for e in g2_exceptions)

    # G4: parse status (kept, never dropped)
    parse_stats = {
        k: {
            "n": len(rows),
            "parsed": sum(1 for r in rows if r["decision"] in ("buy", "sell")),
            "unparsed": sum(1 for r in rows if r["decision"] == "unparsed"),
        }
        for k, rows in all_rows.items()
    }
    gates["G4_parse_reported"] = True
    unparsed_total = sum(v["unparsed"] for v in parse_stats.values())

    # Flip summary (protocol §9)
    flips: dict[str, Any] = {}
    for operator in ("dial", "dim", "cone_centroid"):
        pts = {}
        for ticker in targets:
            rows = [r for r in arm_a if r["target"] == ticker and r["operator"] == operator]
            rows.sort(key=lambda r: r["N"])
            pts[ticker] = flip_point(rows)
        flips[operator] = {
            "per_target_flip_N": pts,
            "flip_rate": sum(1 for v in pts.values() if v is not None) / len(pts),
        }
    for seed in seeds:
        pts = {}
        for ticker in ablab_targets:
            rows = sorted((r for r in arm_c if r["seed"] == seed and r["target"] == ticker), key=lambda r: r["N"])
            pts[ticker] = flip_point(rows)
        flips[f"random_seed{seed}"] = {"per_target_flip_N": pts, "flip_rate": sum(1 for v in pts.values() if v is not None) / len(pts)}

    ended = datetime.now(timezone.utc)
    git_commit = None
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass

    status = "complete" if all(gates.values()) else "failed"
    manifest = {
        "run_id": "operator-comparison-p2-v1-01" if not smoke else "smoke-p2",
        "protocol": "docs/concept-cone-steering/details/proposal-p2-operator-comparison-v1.md",
        "protocol_version": "v1",
        "status": status,
        "smoke": smoke,
        "started_utc": started.isoformat(timespec="seconds"),
        "ended_utc": ended.isoformat(timespec="seconds"),
        "model_path": args.model,
        "git_commit": git_commit,
        "torch_version": torch.__version__,
        "intervention_layer": HELDOUT_LAYER,
        "intervention_span": "100-token instruction span from instruction marker to prompt end",
        "dial_coordinate": {"layer": DIAL_LAYER, "neuron": DIAL_NEURON, "semantics": "all-position mlp_addition incl. decode"},
        "span_operators": {"semantics": "prefill-only span shift; decode steps unshifted (KV cache carries effect)"},
        "alpha_dim_grid": alphas,
        "norm_grid": grid,
        "unit_norms": norms,
        "cone_centroid_unit_norms": n_centroids,
        "random_seeds": seeds,
        "direction_artifacts": {
            "v_dim": {"path": str(Path(args.dim_dir) / "v_dim.pt"), "sha256": EXPECTED_DIM_SHA},
            "cone_basis": {"path": str(Path(args.cone_dir) / "cone.pt"), "sha256": EXPECTED_CONE_SHA},
        },
        "construction_cohort": {
            "source_run": args.heldout_run,
            "top10": ["ED", "GM", "PG", "XEL", "MSFT", "ORLY", "AMZN", "NEE", "META", "OXY"],
            "bottom10": BOTTOM10,
            "note": "all named targets are in the DIM construction cohort (construction-cohort pilot; no generalization claim)",
        },
        "targets": {t: {"name": names[t], "prompt_sha256": ctxs[t].prompt_sha256} for t in targets},
        "anonymous_prompt": {"base_target": "MO", "prompt_sha256": anon_ctx.prompt_sha256},
        "expected_counts": expected_counts,
        "record_counts": {k: len(v) for k, v in all_rows.items()},
        "parse_stats": parse_stats,
        "unparsed_total": unparsed_total,
        "flip_summary": flips,
        "gates": gates,
        "notes": notes,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, rows in (("arm_a_full_grid", arm_a), ("arm_b_cone_ablation", arm_b),
                       ("arm_c_random_control", arm_c), ("arm_d_anonymous", arm_d)):
        (out_root / "arms" / f"{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ngates: {json.dumps(gates)}")
    if notes:
        print(f"notes: {notes}")
    print(f"status: {status}  records: {n_records}  unparsed: {unparsed_total}")
    print(f"wrote {out_root / 'manifest.json'} + 4 arm JSONs")
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
