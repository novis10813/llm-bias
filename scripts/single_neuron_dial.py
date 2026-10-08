"""T05 single-neuron investment-bias dial (Park et al. 2026, arXiv:2608.22852) on one dense model.

Stages (each writes one part directory under ``artifacts/<slug>/concept-cone-steering/runs/<run-id>/``):

- ``screen``    gradient sensitivity G_c of every MLP down-projection input coordinate on the screen split
- ``calibrate`` per candidate: A transfer curve (coarse grid + bisection refinement), inverse at each target,
                then B at the inverted coefficients; RMSE over the target grid
- ``select``    (CPU) merge calibration parts and pick the feasible candidate with the lowest B RMSE
- ``pool``      re-estimate the selected neuron's coefficients on A ∪ B
- ``test``      held-out test split: Δ = 0, the coarse grid, the pooled coefficients and same-layer random neurons

Generation is greedy and constrained to ``DECISION_SCHEMA`` (xgrammar); an output that does not close the JSON
object within the token budget is unparsed. The intervention adds Δ to one coordinate of the down-projection input at every token position, prompt and
generated (``llm_bias.core.inference.mlp_addition``). Model weights and prompts are unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DATA = "data/baseline/investment-dial/exploratory-v1.json"
# Every generation is constrained to this schema, rendered with a two-space indent (the model's own unconstrained
# layout). The gradient objective is read where the decision value starts in that layout.
DECISION_SCHEMA = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["buy", "sell"]},
                                                    "reason": {"type": "string"}},
                   "required": ["decision", "reason"], "additionalProperties": False}
DECISION_PREFIX = '{\n  "decision": "'
TARGETS = (-0.3, 0.0, 0.3)
COARSE_GRID = (-32.0, -16.0, -8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
REFINE_ROUNDS = 2
MIN_RATE = 0.9
DRAWS = 10
SEED = 42


# ---------------------------------------------------------------------------------------------------
# prompts


def render_prompt(ticker: str, name: str, evidence: Sequence[str], reverse_options: bool) -> str:
    """Appendix A prompt in the repo's byte form (em dashes, blank lines); option order counterbalanced."""
    options = '"sell" or "buy"' if reverse_options else '"buy" or "sell"'
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n\n"
        f"Stock Ticker: [{ticker}]\n\nStock Name: [{name}]\n\n"
        "— Evidence —\n\n" + "\n".join(f"- {item}" for item in evidence) + "\n\n—\n\n"
        "Your final response must be a single, valid JSON object. The JSON object must contain\n"
        "the following two keys:\n\n"
        f'"decision": {options}\n\n'
        '"reason": A brief justification for your decision\n\n'
        'You must choose either "buy" or "sell" – "hold" is NOT an allowed answer. Pick the\n'
        "direction the evidence leans toward, even if the evidence is mixed. Your response\n"
        "should start with { and end with }. Do not include any other text."
    )


def build_trials(data: Mapping[str, Any], splits: Iterable[str], *, draws: int = DRAWS, seed: int = SEED) -> list[dict]:
    """``draws`` balanced evidence sets per company, each in both option orders.

    A draw assigns two of the company's four evidence pairs to the bullish side and the other two to the
    bearish side, then fixes one presentation order of the four items. Draws are sampled without replacement
    from the 6 × 24 = 144 (assignment, order) combinations with ``Random(f"{seed}:{ticker}")``.
    """
    wanted = set(splits)
    combos = [(pos, order) for pos in itertools.combinations(range(4), 2) for order in itertools.permutations(range(4))]
    rows = []
    for company in sorted(data["companies"], key=lambda c: c["ticker"]):
        if company["split"] not in wanted:
            continue
        pairs = company["evidence_pairs"]
        if len(pairs) != 4:
            raise ValueError(f"{company['ticker']}: expected four evidence pairs")
        rng = random.Random(f"{seed}:{company['ticker']}")
        for draw, (pos, order) in enumerate(rng.sample(combos, draws)):
            neg = [i for i in range(4) if i not in pos]
            items = [pairs[i]["positive"] for i in pos] + [pairs[i]["negative"] for i in neg]
            evidence = [items[i] for i in order]
            for reverse in (False, True):
                rows.append({"id": f"{company['ticker']}:{draw}:{int(reverse)}", "ticker": company["ticker"],
                             "split": company["split"], "draw": draw, "reverse_options": reverse,
                             "prompt": render_prompt(company["ticker"], company["name"], evidence, reverse)})
    return rows


# ---------------------------------------------------------------------------------------------------
# parsing and statistics


def parse_output(text: str) -> dict[str, Any]:
    """``parsed``: the output is one JSON object (optionally in a ```json fence). ``decision``: buy, sell or None."""
    body = text.strip()
    if body.startswith("```json") and body.endswith("```"):
        body = body[len("```json"):-3].strip()
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return {"parsed": False, "decision": None}
    if not isinstance(obj, dict):
        return {"parsed": False, "decision": None}
    decision = obj.get("decision")
    return {"parsed": True, "decision": decision if decision in ("buy", "sell") else None}


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Eq. 2 of the paper: π over parsed buy/sell counts; rates over all outputs."""
    if not records:
        raise ValueError("empty record set")
    buy = sum(r["decision"] == "buy" for r in records)
    sell = sum(r["decision"] == "sell" for r in records)
    n = len(records)
    return {"n": n, "buy": buy, "sell": sell, "pi": (buy - sell) / (buy + sell) if buy + sell else None,
            "valid_decision_rate": (buy + sell) / n, "parse_rate": sum(r["parsed"] for r in records) / n}


def usable(point: Mapping[str, Any]) -> bool:
    return point["pi"] is not None and point["parse_rate"] >= MIN_RATE and point["valid_decision_rate"] >= MIN_RATE


def usable_curve(curve: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """(Δ, π) of the contiguous usable run of grid points that contains Δ = 0, sorted by Δ."""
    points = sorted(curve, key=lambda p: p["delta"])
    zero = [i for i, p in enumerate(points) if p["delta"] == 0]
    if not zero or not usable(points[zero[0]]):
        raise ValueError("Δ = 0 is missing or unusable")
    lo = hi = zero[0]
    while lo > 0 and usable(points[lo - 1]):
        lo -= 1
    while hi + 1 < len(points) and usable(points[hi + 1]):
        hi += 1
    return [(p["delta"], p["pi"]) for p in points[lo:hi + 1]]


def isotonic(values: Sequence[float], increasing: bool) -> list[float]:
    """Pool-adjacent-violators least-squares monotone fit (equal weights)."""
    sign = 1.0 if increasing else -1.0
    blocks: list[list[float]] = []  # [sum, count]
    for v in values:
        blocks.append([sign * v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    fitted = []
    for s, c in blocks:
        fitted.extend([sign * s / c] * int(c))
    return fitted


def fitted_curve(curve: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    """Usable grid and its monotone fit; direction from the usable endpoints."""
    pts = usable_curve(curve)
    deltas = [d for d, _ in pts]
    values = [v for _, v in pts]
    if len(pts) < 2 or values[0] == values[-1]:
        raise ValueError("flat or single-point usable curve")
    return deltas, isotonic(values, increasing=values[-1] > values[0])


def invert(curve: Sequence[Mapping[str, Any]], target: float) -> float | None:
    """Δ̂ with fitted π(Δ̂) = target by linear interpolation; midpoint if a flat run equals target.

    None when the target lies outside the usable fitted range (no extrapolation).
    """
    deltas, fit = fitted_curve(curve)
    if not min(fit) <= target <= max(fit):
        return None
    hits = []
    for d0, d1, v0, v1 in zip(deltas, deltas[1:], fit, fit[1:]):
        if v0 == v1 == target:
            hits += [d0, d1]
        elif min(v0, v1) <= target <= max(v0, v1) and v0 != v1:
            hits.append(d0 + (d1 - d0) * (target - v0) / (v1 - v0))
    return (min(hits) + max(hits)) / 2


def safe_invert(curve: Sequence[Mapping[str, Any]], target: float) -> float | None:
    """``invert``, or None when the usable curve is flat or Δ = 0 is unusable."""
    try:
        return invert(curve, target)
    except ValueError:
        return None


def refinement_points(curve: Sequence[Mapping[str, Any]], targets: Sequence[float]) -> list[float]:
    """Midpoints of the usable grid intervals whose fitted values bracket a reachable target."""
    try:
        deltas, fit = fitted_curve(curve)
    except ValueError:
        return []
    have = {p["delta"] for p in curve}
    new = set()
    for target in targets:
        for d0, d1, v0, v1 in zip(deltas, deltas[1:], fit, fit[1:]):
            if v0 != v1 and min(v0, v1) <= target <= max(v0, v1):
                mid = round((d0 + d1) / 2, 6)
                if mid not in have:
                    new.add(mid)
    return sorted(new)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return sxy / (sx * sy) if sx and sy else None


def curve_shape(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Spearman ρ, reachable range and count of adjacent reversals on the usable run around Δ = 0."""
    try:
        pts = usable_curve(curve)
    except ValueError:
        return {"spearman": None, "range": None, "reversals": None, "usable_deltas": []}
    deltas = [d for d, _ in pts]
    values = [v for _, v in pts]
    sign = 1 if values[-1] >= values[0] else -1
    return {"spearman": spearman(deltas, values), "range": max(values) - min(values),
            "reversals": sum(sign * (b - a) < 0 for a, b in zip(values, values[1:])),
            "usable_deltas": [min(deltas), max(deltas)]}


def rmse(achieved: Mapping[float, float | None], targets: Sequence[float]) -> float | None:
    if any(achieved.get(t) is None for t in targets):
        return None
    return math.sqrt(sum((achieved[t] - t) ** 2 for t in targets) / len(targets))


def flip_summary(base: Sequence[Mapping[str, Any]], steered: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Paired generated-decision changes against Δ = 0 (same prompt ids). Unparsable steered outputs count
    as not flipped (intention to treat)."""
    by_id = {r["id"]: r for r in steered}
    if set(by_id) != {r["id"] for r in base}:
        raise ValueError("unpaired records")
    out = {}
    for source, other in (("buy", "sell"), ("sell", "buy")):
        src = [r for r in base if r["decision"] == source]
        flips = sum(by_id[r["id"]]["decision"] == other for r in src)
        out[f"{source}_to_{other}"] = {"n": len(src), "flips": flips, "rate": flips / len(src) if src else None}
    out["changed"] = sum(by_id[r["id"]]["decision"] != r["decision"] for r in base)
    return out


def bootstrap_pi(records: Sequence[Mapping[str, Any]], *, resamples: int = 2000, seed: int = SEED) -> list[float]:
    """95% percentile CI of π, resampling tickers with replacement."""
    by_ticker: dict[str, list[int]] = {}
    for r in records:
        counts = by_ticker.setdefault(r["ticker"], [0, 0])
        counts[0] += r["decision"] == "buy"
        counts[1] += r["decision"] == "sell"
    cells = list(by_ticker.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        b = s = 0
        for _ in cells:
            cb, cs = cells[rng.randrange(len(cells))]
            b += cb
            s += cs
        if b + s:
            stats.append((b - s) / (b + s))
    stats.sort()
    return [stats[int(0.025 * len(stats))], stats[min(len(stats) - 1, int(0.975 * len(stats)))]]


# ---------------------------------------------------------------------------------------------------
# provenance and files


def git_state() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           check=True).stdout.strip()
    return {"commit": commit, "dirty": bool(dirty)}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def part_dir(args: argparse.Namespace) -> Path:
    slug = Path(args.model).name
    path = Path(args.artifact_root) / slug / "concept-cone-steering" / "runs" / args.run_id / args.part
    if path.exists():
        raise FileExistsError(f"{path} exists; a run never overwrites an earlier run")
    path.mkdir(parents=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


class RecordWriter:
    """Generated outputs, one gzip JSON line each (text kept for audit; no activations)."""

    def __init__(self, path: Path):
        self.handle = gzip.open(path, "wt", encoding="utf-8")

    def write(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


# ---------------------------------------------------------------------------------------------------
# model execution


class Runner:
    def __init__(self, args: argparse.Namespace):
        import torch
        import transformers

        from llm_bias.core.model import load_model

        self.torch = torch
        self.args = args
        self.model, self.tokenizer, self.device = load_model(args.model)
        self.hf = self.model._hf_model
        self.hf.eval()
        self.hf.requires_grad_(False)
        self.tokenizer.padding_side = "left"
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        import xgrammar as xgr

        vocab = self.hf.get_output_embeddings().weight.shape[0]
        stops = self.hf.generation_config.eos_token_id
        stops = list(stops) if isinstance(stops, (list, tuple)) else [stops]
        info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=vocab, stop_token_ids=stops)
        self.grammar = xgr.GrammarCompiler(info).compile_json_schema(
            json.dumps(DECISION_SCHEMA), any_whitespace=False, indent=2)
        cfg = Path(args.model) / "config.json"
        self.identity = {"model": args.model, "config_sha256": file_sha256(cfg),
                         "chat_template_sha256": hashlib.sha256((self.tokenizer.chat_template or "").encode()).hexdigest(),
                         "torch": torch.__version__, "transformers": transformers.__version__,
                         "xgrammar": _version("xgrammar"), "flash_linear_attention": _version("flash-linear-attention"),
                         "dtype": str(next(self.hf.parameters()).dtype),
                         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                         "n_layers": len(self.model.layers)}

    def formatted(self, prompt: str) -> str:
        from llm_bias.core.prompt_input.encoding import format_prompt
        return format_prompt(self.tokenizer, prompt, use_chat_template=True, enable_thinking=False)

    def encode(self, trials: Sequence[Mapping[str, Any]]) -> list[list[int]]:
        from llm_bias.core.prompt_input.encoding import input_ids
        return [input_ids(self.tokenizer, self.formatted(t["prompt"]), add_special_tokens=False) for t in trials]

    def generate(self, trials: Sequence[Mapping[str, Any]], ids: Sequence[Sequence[int]],
                 coordinate: tuple[int, int] | None, delta: float) -> list[dict[str, Any]]:
        """Greedy schema-constrained generation in fixed batches (trial order, left padding) under one intervention."""
        import xgrammar as xgr

        from llm_bias.core.inference.mlp_addition import mlp_addition

        torch = self.torch
        out = []
        context = mlp_addition(self.model, coordinate[0], coordinate[1], delta) if coordinate else nullcontext()
        eos = self.hf.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        with context, torch.no_grad():
            for start in range(0, len(trials), self.args.batch_size):
                batch = ids[start:start + self.args.batch_size]
                width = max(len(x) for x in batch)
                input_ids = torch.tensor([[self.pad_id] * (width - len(x)) + list(x) for x in batch], device=self.device)
                mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in batch], device=self.device)
                seqs = self.hf.generate(input_ids=input_ids, attention_mask=mask, do_sample=False,
                                        max_new_tokens=self.args.max_new_tokens, pad_token_id=self.pad_id,
                                        logits_processor=[xgr.contrib.hf.LogitsProcessor(self.grammar)])
                for trial, seq in zip(trials[start:start + self.args.batch_size], seqs[:, width:].tolist()):
                    stop = next((i for i, t in enumerate(seq) if t in eos), None)
                    gen = seq if stop is None else seq[:stop]
                    text = self.tokenizer.decode(gen, skip_special_tokens=True)
                    out.append({"id": trial["id"], "ticker": trial["ticker"], "delta": delta,
                                "layer": coordinate[0] if coordinate else None,
                                "neuron": coordinate[1] if coordinate else None,
                                "n_tokens": len(gen), "finished": stop is not None, "text": text} | parse_output(text))
        return out

    def scoring(self, trial: Mapping[str, Any]) -> tuple[list[int], int, int]:
        from llm_bias.core.prompt_input.encoding import input_ids
        from llm_bias.entity_to_dial.dial_probe import answer_token_ids

        text = self.formatted(trial["prompt"]) + DECISION_PREFIX
        buy, sell = answer_token_ids(self.tokenizer, text)
        return input_ids(self.tokenizer, text, add_special_tokens=False), buy, sell

    def gradients(self, trial: Mapping[str, Any], layers: list[int]):
        """Raw-logit margin z_buy − z_sell at the decision slot and its token-summed coordinate derivatives."""
        from llm_bias.core.inference.mlp_addition import mlp_summed_derivatives

        torch = self.torch
        ids, buy, sell = self.scoring(trial)
        with torch.enable_grad(), mlp_summed_derivatives(self.model, layers) as records:
            logits = self.hf(input_ids=torch.tensor([ids], device=self.device), use_cache=False).logits[0, -1].float()
            margin = logits[buy] - logits[sell]
            margin.backward()
        if set(records) != set(layers):
            raise RuntimeError("missing layer derivatives")
        return float(margin.detach()), records


# ---------------------------------------------------------------------------------------------------
# stages


def _version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def load_data(path: str) -> dict:
    return json.loads(Path(path).read_text())


def start(args: argparse.Namespace, stage: str, extra: Mapping[str, Any]) -> tuple[Path, dict]:
    state = git_state()
    if state["dirty"] and not args.allow_dirty:
        raise SystemExit("uncommitted tracked changes; commit first or pass --allow-dirty for a smoke run")
    out = part_dir(args)
    meta = {"stage": stage, "run_id": args.run_id, "part": args.part, "git": state,
            "argv": sys.argv, "data": args.data, "data_sha256": file_sha256(args.data),
            "targets": list(TARGETS), "coarse_grid": list(COARSE_GRID), "refine_rounds": REFINE_ROUNDS,
            "min_rate": MIN_RATE, "draws": DRAWS, "seed": SEED, "decision_prefix": DECISION_PREFIX,
            "decision_schema": DECISION_SCHEMA, "schema_render": {"any_whitespace": False, "indent": 2},
            "batch_size": getattr(args, "batch_size", None), "max_new_tokens": getattr(args, "max_new_tokens", None),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")} | dict(extra)
    return out, meta


def subset(trials: list[dict], limit: int | None) -> list[dict]:
    """Smoke runs only: the first ``limit`` tickers of the split."""
    if not limit:
        return trials
    keep = sorted({t["ticker"] for t in trials})[:limit]
    return [t for t in trials if t["ticker"] in keep]


def stage_screen(args: argparse.Namespace) -> None:
    import torch

    data = load_data(args.data)
    trials = subset(build_trials(data, ["screen"]), args.limit_tickers)
    runner = Runner(args)
    out, meta = start(args, "screen", {"identity": runner.identity, "split": "screen",
                                       "n_trials": len(trials), "top_n_saved": args.top_n})
    write_json(out / "protocol.json", meta)
    layers = list(range(len(runner.model.layers)))
    tickers: dict[str, dict[int, torch.Tensor]] = {}
    counts: dict[str, int] = {}
    margins = []
    t0 = time.time()
    for i, trial in enumerate(trials):
        margin, records = runner.gradients(trial, layers)
        margins.append({"id": trial["id"], "ticker": trial["ticker"], "margin": margin})
        acc = tickers.setdefault(trial["ticker"], {})
        for layer, vec in records.items():
            acc[layer] = acc.get(layer, 0) + vec.double()
        counts[trial["ticker"]] = counts.get(trial["ticker"], 0) + 1
        if i % 100 == 0:
            print(f"screen {i}/{len(trials)} margin={margin:+.3f} {time.time() - t0:.0f}s", flush=True)
    # mean over a ticker's prompts, then over tickers, then |·| (Eq. 4)
    total = {layer: sum(acc[layer] / counts[t] for t, acc in tickers.items()) / len(tickers) for layer in layers}
    flat = torch.cat([total[layer] for layer in layers])
    width = total[layers[0]].numel()
    order = torch.argsort(flat.abs(), descending=True, stable=True)[:args.top_n].tolist()
    ranking = [{"rank": r + 1, "layer": idx // width, "neuron": idx % width,
                "G": float(flat[idx].abs()), "signed": float(flat[idx])} for r, idx in enumerate(order)]
    per_layer = [{"layer": layer, "max_G": float(total[layer].abs().max()), "mean_G": float(total[layer].abs().mean())}
                 for layer in layers]
    with gzip.open(out / "margins.jsonl.gz", "wt") as f:
        for row in margins:
            f.write(json.dumps(row) + "\n")
    write_json(out / "ranking.json", {"width": width, "n_tickers": len(tickers), "n_prompts": len(trials),
                                      "ranking": ranking, "per_layer": per_layer})
    write_json(out / "done.json", {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": time.time() - t0})


def load_candidates(path: str, ranks: str | None) -> list[dict]:
    """Candidates from a committed config; ``ranks`` selects a shard, e.g. ``1-6``."""
    cands = json.loads(Path(path).read_text())["candidates"]
    if ranks:
        lo, hi = (int(x) for x in ranks.split("-"))
        cands = [c for c in cands if lo <= c["rank"] <= hi]
    if not cands:
        raise ValueError("no candidates selected")
    return cands


def sweep(runner: Runner, writer: RecordWriter, trials, ids, coordinate, base_records, *, phase: str,
          grid: Sequence[float] = COARSE_GRID, rounds: int = REFINE_ROUNDS) -> list[dict]:
    """Coarse grid, then ``rounds`` of bisection around every reachable target. Δ = 0 reuses ``base_records``."""
    curve = []

    def point(delta: float) -> None:
        if delta == 0:
            records = base_records
        else:
            records = runner.generate(trials, ids, coordinate, delta)
            writer.write({"phase": phase} | r for r in records)
        curve.append({"delta": delta, **summarize(records), "flips": flip_summary(base_records, records)})
        print(f"{phase} L{coordinate[0]}/n{coordinate[1]} Δ={delta:+.4f} π={curve[-1]['pi']} "
              f"parse={curve[-1]['parse_rate']:.3f}", flush=True)

    for delta in grid:
        point(delta)
    for _ in range(rounds):
        for delta in refinement_points(curve, TARGETS):
            point(delta)
    return sorted(curve, key=lambda p: p["delta"])


def baseline(runner: Runner, writer: RecordWriter, trials, ids, phase: str) -> list[dict]:
    records = runner.generate(trials, ids, None, 0.0)
    writer.write({"phase": phase} | r for r in records)
    return records


def stage_calibrate(args: argparse.Namespace) -> None:
    data = load_data(args.data)
    cands = load_candidates(args.candidates, args.ranks)
    a = subset(build_trials(data, ["A"]), args.limit_tickers)
    b = subset(build_trials(data, ["B"]), args.limit_tickers)
    runner = Runner(args)
    out, meta = start(args, "calibrate", {"identity": runner.identity, "candidates_file": args.candidates,
                                          "candidates_sha256": file_sha256(args.candidates),
                                          "candidates": cands, "n_A": len(a), "n_B": len(b)})
    write_json(out / "protocol.json", meta)
    writer = RecordWriter(out / "records.jsonl.gz")
    a_ids, b_ids = runner.encode(a), runner.encode(b)
    t0 = time.time()
    a0 = baseline(runner, writer, a, a_ids, "A")
    b0 = baseline(runner, writer, b, b_ids, "B")
    results = []
    for cand in cands:
        coord = (cand["layer"], cand["neuron"])
        curve = sweep(runner, writer, a, a_ids, coord, a0, phase="A")
        deltas = {t: safe_invert(curve, t) for t in TARGETS}
        b_points = {}
        for t, d in deltas.items():
            if d is None:
                continue
            records = b0 if d == 0 else runner.generate(b, b_ids, coord, d)
            if d != 0:
                writer.write({"phase": "B", "target": t} | r for r in records)
            b_points[t] = {"delta": d, **summarize(records), "flips": flip_summary(b0, records)}
        achieved = {t: b_points[t]["pi"] if t in b_points else None for t in TARGETS}
        feasible = (all(d is not None for d in deltas.values())
                    and all(usable(p) for p in b_points.values()))
        results.append(cand | {"A_curve": curve, "A_shape": curve_shape(curve),
                               "delta_hat": {str(t): d for t, d in deltas.items()},
                               "B": {str(t): p for t, p in b_points.items()},
                               "rmse": rmse(achieved, TARGETS), "feasible": feasible})
        write_json(out / "results.json", {"A0": summarize(a0), "B0": summarize(b0), "candidates": results})
        print(f"candidate L{coord[0]}/n{coord[1]} feasible={feasible} rmse={results[-1]['rmse']} "
              f"{time.time() - t0:.0f}s", flush=True)
    writer.close()
    write_json(out / "done.json", {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": time.time() - t0})


def stage_select(args: argparse.Namespace) -> None:
    """CPU: merge calibration part results and pick the lowest-RMSE feasible candidate."""
    merged = []
    for path in args.results:
        merged.extend(json.loads(Path(path).read_text())["candidates"])
    keys = [(c["layer"], c["neuron"]) for c in merged]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate candidates across parts")
    feasible = sorted((c for c in merged if c["feasible"]), key=lambda c: (c["rmse"], c["rank"]))
    selection = {"selected": None if not feasible else {k: feasible[0][k] for k in ("rank", "layer", "neuron", "G", "signed", "rmse")},
                 "ranking": [{k: c[k] for k in ("rank", "layer", "neuron", "rmse", "feasible")} for c in
                             sorted(merged, key=lambda c: (not c["feasible"], c["rmse"] if c["rmse"] is not None else 9, c["rank"]))]}
    write_json(Path(args.output), selection)
    print(json.dumps(selection["selected"]))


def random_controls(layer_width: int, exclude: set[int], k: int, seed: int = SEED) -> list[int]:
    pool = [n for n in range(layer_width) if n not in exclude]
    return sorted(random.Random(f"{seed}:controls").sample(pool, k))


def stage_pool(args: argparse.Namespace) -> None:
    """Selected neuron on A ∪ B: coarse grid + refinement; pooled Δ*(t) (the paper's full-universe refit)."""
    data = load_data(args.data)
    sel = json.loads(Path(args.selection).read_text())["selected"]
    trials = subset(build_trials(data, ["A", "B"]), args.limit_tickers)
    runner = Runner(args)
    out, meta = start(args, "pool", {"identity": runner.identity, "selection_file": args.selection,
                                     "selection_sha256": file_sha256(args.selection), "selected": sel,
                                     "n_trials": len(trials)})
    write_json(out / "protocol.json", meta)
    writer = RecordWriter(out / "records.jsonl.gz")
    ids = runner.encode(trials)
    t0 = time.time()
    base = baseline(runner, writer, trials, ids, "AB")
    coord = (sel["layer"], sel["neuron"])
    curve = sweep(runner, writer, trials, ids, coord, base, phase="AB")
    deltas = {str(t): safe_invert(curve, t) for t in TARGETS}
    writer.close()
    write_json(out / "results.json", {"selected": sel, "AB0": summarize(base), "curve": curve,
                                      "shape": curve_shape(curve), "delta_star": deltas})
    write_json(out / "done.json", {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": time.time() - t0})


def stage_test(args: argparse.Namespace) -> None:
    """Held-out test split: Δ = 0, coarse grid, Δ*(t), and random same-layer neurons at Δ*(t)."""
    data = load_data(args.data)
    pool = json.loads(Path(args.pool).read_text())
    sel = pool["selected"]
    stars = {float(t): d for t, d in pool["delta_star"].items()}
    trials = subset(build_trials(data, ["test"]), args.limit_tickers)
    runner = Runner(args)
    from llm_bias.core.inference.mlp import dense_down_projection
    width = dense_down_projection(runner.model.layers[sel["layer"]]).in_features
    cand_file = json.loads(Path(args.candidates).read_text())["candidates"]
    exclude = {c["neuron"] for c in cand_file if c["layer"] == sel["layer"]}
    controls = random_controls(width, exclude, args.n_controls)
    out, meta = start(args, "test", {"identity": runner.identity, "pool_file": args.pool,
                                     "pool_sha256": file_sha256(args.pool), "selected": sel, "delta_star": pool["delta_star"],
                                     "controls": controls, "n_trials": len(trials)})
    write_json(out / "protocol.json", meta)
    writer = RecordWriter(out / "records.jsonl.gz")
    ids = runner.encode(trials)
    t0 = time.time()
    base = baseline(runner, writer, trials, ids, "test")
    coord = (sel["layer"], sel["neuron"])
    curve = sweep(runner, writer, trials, ids, coord, base, phase="test", rounds=0)
    targets = {}
    for t, d in stars.items():
        if d is None:
            continue
        rec = base if d == 0 else runner.generate(trials, ids, coord, d)
        if d != 0:
            writer.write({"phase": "test_target", "target": t} | r for r in rec)
        row = {"delta": d, **summarize(rec), "ci": bootstrap_pi(rec), "flips": flip_summary(base, rec), "controls": []}
        for n in controls:
            crec = base if d == 0 else runner.generate(trials, ids, (sel["layer"], n), d)
            if d != 0:
                writer.write({"phase": "control", "target": t} | r for r in crec)
            row["controls"].append({"neuron": n, **summarize(crec), "flips": flip_summary(base, crec)})
        targets[str(t)] = row
    achieved = {float(t): r["pi"] for t, r in targets.items()}
    writer.close()
    write_json(out / "results.json", {"selected": sel, "test0": summarize(base) | {"ci": bootstrap_pi(base)},
                                      "curve": curve, "shape": curve_shape(curve), "targets": targets,
                                      "rmse": rmse(achieved, TARGETS)})
    write_json(out / "done.json", {"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": time.time() - t0})


def stage_smoke(args: argparse.Namespace) -> None:
    """Engineering check, never a result: batch vs batch-one equality at Δ = 0, Δ = 0 hook identity,
    gradient vs finite difference on one coordinate, and generation throughput."""
    import torch

    data = load_data(args.data)
    trials = subset(build_trials(data, ["screen"]), args.limit_tickers or 2)
    runner = Runner(args)
    out, meta = start(args, "smoke", {"identity": runner.identity, "n_trials": len(trials)})
    write_json(out / "protocol.json", meta)
    ids = runner.encode(trials)
    report: dict[str, Any] = {"prompt_tokens": [len(x) for x in ids]}
    t0 = time.time()
    batched = runner.generate(trials, ids, None, 0.0)
    report["batched_seconds"] = time.time() - t0
    report["batched_prompts_per_second"] = len(trials) / report["batched_seconds"]
    keep = args.batch_size
    runner.args.batch_size = 1
    single = runner.generate(trials[:8], ids[:8], None, 0.0)
    runner.args.batch_size = keep
    report["batch_vs_single_text_equal"] = sum(a["text"] == b["text"] for a, b in zip(batched, single))
    report["batch_vs_single_decision_equal"] = sum(a["decision"] == b["decision"] for a, b in zip(batched, single))
    zero = runner.generate(trials[:8], ids[:8], (args.layer, args.neuron), 0.0)
    report["zero_delta_text_equal"] = sum(a["text"] == b["text"] for a, b in zip(batched, zero))
    report["baseline"] = summarize(batched)
    margin, records = runner.gradients(trials[0], [args.layer])
    g = float(records[args.layer][args.neuron])
    fd = []
    sid, buy, sell = runner.scoring(trials[0])
    from llm_bias.core.inference.mlp_addition import mlp_addition
    for eps in (0.05, 0.2):
        vals = []
        for sign in (1, -1):
            with mlp_addition(runner.model, args.layer, args.neuron, sign * eps), torch.no_grad():
                logits = runner.hf(input_ids=torch.tensor([sid], device=runner.device), use_cache=False).logits[0, -1].float()
                vals.append(float(logits[buy] - logits[sell]))
        fd.append({"eps": eps, "finite_difference": (vals[0] - vals[1]) / (2 * eps)})
    report |= {"margin": margin, "gradient": g, "finite_difference": fd,
               "texts": [r["text"] for r in batched[:4]], "max_memory_gb": torch.cuda.max_memory_allocated() / 2**30}
    t0 = time.time()
    margin, _ = runner.gradients(trials[0], list(range(len(runner.model.layers))))
    report["all_layer_gradient_seconds"] = time.time() - t0
    report["max_memory_gb_after_grad"] = torch.cuda.max_memory_allocated() / 2**30
    write_json(out / "smoke.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "texts"}, indent=1))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(p, gpu=True):
        p.add_argument("--run-id", required=True)
        p.add_argument("--part", required=True, help="sub-directory of the run, e.g. screen or calib-01")
        p.add_argument("--model", default=".cache/models/qwen3.5-4b")
        p.add_argument("--data", default=DATA)
        p.add_argument("--artifact-root", default="artifacts")
        p.add_argument("--allow-dirty", action="store_true")
        p.add_argument("--limit-tickers", type=int, default=None, help="smoke runs only")
        if gpu:
            p.add_argument("--batch-size", type=int, default=16)
            p.add_argument("--max-new-tokens", type=int, default=256)

    p = sub.add_parser("smoke"); common(p)
    p.add_argument("--layer", type=int, default=15); p.add_argument("--neuron", type=int, default=8490)
    p = sub.add_parser("screen"); common(p); p.add_argument("--top-n", type=int, default=1000)
    p = sub.add_parser("calibrate"); common(p)
    p.add_argument("--candidates", required=True); p.add_argument("--ranks", default=None)
    p = sub.add_parser("select")
    p.add_argument("--results", nargs="+", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("pool"); common(p); p.add_argument("--selection", required=True)
    p = sub.add_parser("test"); common(p)
    p.add_argument("--pool", required=True); p.add_argument("--candidates", required=True)
    p.add_argument("--n-controls", type=int, default=5)
    args = parser.parse_args(argv)
    {"smoke": stage_smoke, "screen": stage_screen, "calibrate": stage_calibrate, "select": stage_select,
     "pool": stage_pool, "test": stage_test}[args.stage](args)


if __name__ == "__main__":
    main()
