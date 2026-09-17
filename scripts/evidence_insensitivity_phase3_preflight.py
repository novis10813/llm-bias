"""Phase 3 one-company GPU preflight: verify the real-model intervention
plumbing (record_block_states, prefill-only swap transform, FP32 margin
tail, patched greedy generation with decode steps) on one sample company.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/evidence_insensitivity_phase3_preflight.py \
      --model .cache/models/qwen3.5-4b --model-slug qwen3.5-4b --run-id phase3-gpu-bf16-01
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from llm_bias.core.artifacts.io import read_jsonl
from llm_bias.core.continuation_scoring import continuation_token_ids, fp32_next_token_log_probs
from llm_bias.core.inference.generation import GenerationConfig, generate_tokens
from llm_bias.core.inference.interventions import record_block_states, residual_interventions
from llm_bias.core.model import load_model
from llm_bias.core.prompt_input.encoding import input_ids

from llm_bias.evidence_insensitivity.phase3 import POSITIONS, _ids, _margin_at_end, _swap_transform
from llm_bias.evidence_insensitivity.screening import _generation_target, parse_decision
from llm_bias.evidence_insensitivity.template import DECISION_PREFIX


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/models/qwen3.5-4b")
    parser.add_argument("--model-slug", default="qwen3.5-4b")
    parser.add_argument("--run-id", default="phase3-gpu-bf16-01")
    args = parser.parse_args()
    model, tokenizer, device = load_model(args.model, dtype=None)
    sel_path = Path("artifacts") / args.model_slug / "evidence-insensitivity" / "runs" / args.run_id / "prepare" / "selection.json"
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    grid = selection["layer_grid"]
    run_root = Path("artifacts") / args.model_slug / "evidence-insensitivity" / "runs" / selection["phase1_run_id"]
    prompts = {(p["ticker"], p["condition"]): p["prompt_text"] for p in read_jsonl(run_root / "prepare" / "prompts.jsonl") if p["arm"] == "primary" and p["condition"] in ("N15", "P15")}

    t = selection["sample"][0]["ticker"]
    pos = selection["positions"][t]
    n15_text = prompts[(t, "N15")]
    p15_text = prompts[(t, "P15")]
    print(f"company={t} device={device}")

    # buy/sell ids
    _, buy_ids = continuation_token_ids(tokenizer, n15_text + DECISION_PREFIX, "buy")
    _, sell_ids = continuation_token_ids(tokenizer, n15_text + DECISION_PREFIX, "sell")
    assert len(buy_ids) == 1 and len(sell_ids) == 1, (buy_ids, sell_ids)
    buy_id, sell_id = buy_ids[0], sell_ids[0]
    print(f"buy={buy_id} sell={sell_id}")

    # source states (prefix sequences)
    t0 = time.time()
    ids_p15 = _ids(tokenizer, p15_text + DECISION_PREFIX, device)
    with torch.no_grad():
        states = record_block_states(model, ids_p15, layers=grid)
    final_pos = pos["N15"]["prefix"]["prompt_end"]
    anchor_L = selection["anchor_combo"][0]
    anchor_p = pos["N15"]["prefix"][selection["anchor_combo"][1]]
    print(f"record_block_states: {time.time() - t0:.2f}s ({len(grid)} layers, seq={ids_p15.shape[1]})")

    # margins
    ids_n15 = _ids(tokenizer, n15_text + DECISION_PREFIX, device)
    t0 = time.time()
    with torch.no_grad():
        base_m = _margin_at_end(model, ids_n15, buy_id, sell_id, None)
    t1 = _margin_at_end(model, ids_n15, buy_id, sell_id, {grid[-1]: _swap_transform(final_pos, states[grid[-1]]["post"][0, final_pos, :], ids_n15.shape[1])})
    t2 = _margin_at_end(model, ids_n15, buy_id, sell_id, {anchor_L: _swap_transform(anchor_p, states[anchor_L]["post"][0, anchor_p, :], ids_n15.shape[1])})
    print(f"margin forwards: {time.time() - t0:.2f}s  base={base_m:.4f} T1_final_prompt_end={t1:.4f} (d={t1 - base_m:+.4f}) anchor({anchor_L},{selection['anchor_combo'][1]})={t2:.4f} (d={t2 - base_m:+.4f})")
    del states

    # patched generation (raw prompt), including decode steps under the active transform
    src_raw_L = anchor_L
    t0 = time.time()
    raw_n15 = _ids(tokenizer, n15_text, device)
    raw_p15 = _ids(tokenizer, p15_text, device)
    with torch.no_grad():
        raw_states = record_block_states(model, raw_p15, layers=[src_raw_L])
    src_state = raw_states[src_raw_L]["post"][0, pos["P15"]["raw"][selection["anchor_combo"][1]], :].detach().clone()
    transform = _swap_transform(pos["N15"]["raw"][selection["anchor_combo"][1]], src_state, raw_n15.shape[1])
    with torch.no_grad(), residual_interventions(model, {src_raw_L: transform}):
        gen = generate_tokens(_generation_target(model), raw_n15, GenerationConfig(max_new_tokens=128, temperature=0.0, pad_token_id=getattr(tokenizer, "eos_token_id", None)))
    out_text = tokenizer.decode(gen[0][raw_n15.shape[1]:], skip_special_tokens=False)
    print(f"patched generation: {time.time() - t0:.2f}s  decision={parse_decision(out_text)!r}  text={out_text[:120]!r}")

    # baseline generation (no patch)
    t0 = time.time()
    with torch.no_grad():
        gen0 = generate_tokens(_generation_target(model), raw_n15, GenerationConfig(max_new_tokens=128, temperature=0.0, pad_token_id=getattr(tokenizer, "eos_token_id", None)))
    base_text = tokenizer.decode(gen0[0][raw_n15.shape[1]:], skip_special_tokens=False)
    print(f"baseline generation: {time.time() - t0:.2f}s  decision={parse_decision(base_text)!r}  text={base_text[:120]!r}")
    print("PREFLIGHT-OK")


if __name__ == "__main__":
    main()
