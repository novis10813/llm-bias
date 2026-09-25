"""A1: admissibility audit and fixed-prefix-margin vs generated-decision boundary analysis of existing runs.

Reads completed concept-cone runs (Qwen V1 DIM, 427-selected cones, V2 DIM) without inference, re-derives
every row's complete-object decision, strict decision and output path class from its stored text, and
reports where the sign of the fixed-prefix margin disagrees with the generated decision (C4), by path
class and |margin| bin. Expected counts are produced here, never hard-coded.

Writes artifacts/<slug>/concept-cone-steering/runs/audit-v1-<date>/decision_boundary.{json,md}; sources are
referenced by path and SHA-256 only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from llm_bias.core.decision_parsing import parse_complete_decision, parse_strict_decision
from llm_bias.core.decision_readout import path_class
from llm_bias.core.steering.protocol import sha256_bytes

RUNS = {
    "qwen3.5-4b": ["dim-layer-sweep-v1-20260925-paper-01/tokenwise", "dim-layer-sweep-v1-20260925-paper-01/single_all",
                   "c2-guided-paper-20260924"],
    "gemma4-12b-it": ["crossmodel-v1-eval-20260924", "dim-crossmodel-layer-sweep-v2-20260925-full-01/tokenwise",
                      "dim-crossmodel-layer-sweep-v2-20260925-full-01/single_all"],
    "glm4-9b-0414": ["crossmodel-v1-eval-20260924", "dim-crossmodel-layer-sweep-v2-20260925-full-01/tokenwise",
                     "dim-crossmodel-layer-sweep-v2-20260925-full-01/single_all"],
    "gpt-oss-20b": ["crossmodel-v1-eval-20260924", "dim-crossmodel-layer-sweep-v2-20260925-full-01/tokenwise",
                    "dim-crossmodel-layer-sweep-v2-20260925-full-01/single_all"],
}
BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, float("inf")))
ADMISSIBILITY_FIELDS = ("split_seed", "split_sha256", "prompt_family_sha256", "git_commit", "chat_template_sha256",
                        "model_config_sha256", "tokenizer_config_sha256", "checkpoint_files")


def iter_rows(result: dict[str, Any]) -> Iterator[tuple[str, str, float, float, str]]:
    """(ticker, cell, alpha, margin, text) for every stored generation of any known schema."""
    for ticker, entry in result["targets"].items():
        if "cone_centroid" in entry:
            for row in entry["cone_centroid"]:
                yield ticker, "cone_centroid", row["alpha"], row["margin"], row["generated_text"]
            continue
        if "baseline" in entry:
            row = entry["baseline"]
            yield ticker, "baseline", row["alpha"], row["margin"], row["generated_text"]
        for cell, value in entry.items():
            if cell == "baseline" or not isinstance(value, dict):
                continue
            for row in value["rows"]:
                yield ticker, cell, row["alpha"], row["margin"], row["generated_text"]


def audit_run(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    result = json.loads(raw)
    meta = result.get("metadata", {})
    rows = list(iter_rows(result))
    counts: Counter = Counter()
    by_path: dict[str, Counter] = {}
    by_bin: dict[str, Counter] = {f"[{lo},{hi})": Counter() for lo, hi in BINS}
    truncated_text = 0
    for _ticker, _cell, _alpha, margin, text in rows:
        decision, _kind = parse_complete_decision(text)
        strict = parse_strict_decision(text)
        cls = path_class(text)
        truncated_text += len(text) == 1024
        counts["rows"] += 1
        counts["complete_parsed"] += decision is not None
        counts["strict_parsed"] += strict is not None
        bucket = by_path.setdefault(cls, Counter())
        bucket["rows"] += 1
        if decision is None:
            continue
        diverge = (margin > 0 and decision == "sell") or (margin < 0 and decision == "buy")
        counts["divergent"] += diverge
        bucket["parsed"] += 1
        bucket["divergent"] += diverge
        for lo, hi in BINS:
            if lo <= abs(margin) < hi:
                by_bin[f"[{lo},{hi})"]["parsed"] += 1
                by_bin[f"[{lo},{hi})"]["divergent"] += diverge
    return {
        "path": str(path), "sha256": sha256_bytes(raw), "schema": meta.get("schema"), "mode": meta.get("mode"),
        "complete": result.get("complete"),
        "admissibility": {field: field in meta for field in ADMISSIBILITY_FIELDS},
        "text_possibly_truncated_at_1024": truncated_text,
        "counts": dict(counts), "by_path_class": {k: dict(v) for k, v in sorted(by_path.items())},
        "by_abs_margin": {k: dict(v) for k, v in by_bin.items()},
    }


def render(slug: str, audits: list[dict[str, Any]]) -> str:
    lines = [f"# {slug}: fixed-prefix margin vs generated decision (A1)", "",
             "Divergent = complete-object decision parsed and sign(margin) disagrees with it.", "",
             "| run | complete | rows | complete-parsed | strict-parsed | divergent |", "|---|---|---|---|---|---|"]
    for a in audits:
        c = a["counts"]
        lines.append(f"| {a['path']} | {a['complete']} | {c.get('rows', 0)} | {c.get('complete_parsed', 0)} | "
                     f"{c.get('strict_parsed', 0)} | {c.get('divergent', 0)} |")
    for a in audits:
        lines += ["", f"## {a['path']}", "", "| path class | rows | parsed | divergent |", "|---|---|---|---|"]
        lines += [f"| {k} | {v.get('rows', 0)} | {v.get('parsed', 0)} | {v.get('divergent', 0)} |"
                  for k, v in a["by_path_class"].items()]
        lines += ["", "| abs margin | parsed | divergent |", "|---|---|---|"]
        lines += [f"| {k} | {v.get('parsed', 0)} | {v.get('divergent', 0)} |" for k, v in a["by_abs_margin"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True, help="audit-v1-<date>")
    parser.add_argument("--models", nargs="+", default=list(RUNS))
    args = parser.parse_args()
    for slug in args.models:
        base = Path("artifacts") / slug / "concept-cone-steering" / "runs"
        audits = []
        for run in RUNS[slug]:
            path = base / run / "result.json"
            if not path.exists():
                audits.append({"path": str(path), "missing": True, "complete": None, "counts": {},
                               "by_path_class": {}, "by_abs_margin": {}})
                continue
            audits.append(audit_run(path))
        out = base / args.run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "decision_boundary.json").write_text(json.dumps({"schema": "audit-v1-decision-boundary",
                                                                "model_slug": slug, "runs": audits}, indent=1) + "\n")
        (out / "decision_boundary.md").write_text(render(slug, audits))
        for a in audits:
            print(slug, a["path"].split("runs/")[-1], a.get("complete"), a["counts"], flush=True)


if __name__ == "__main__":
    main()
