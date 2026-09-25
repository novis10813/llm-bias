"""CPU freeze manifest for confirmation-v1: prompt families, K suffix identity, anon identities, pinned date.

For each registry model this renders 503 companies x 5 evidence conditions plus 10 anonymous identities x 5
conditions with the inference-time tokenizer and asserts (fail-closed):
  * balanced family hash == V2 / Qwen V1 ``prompt_family_sha256`` (read from the stored runs);
  * the common instruction suffix K equals the registry and its token ids are identical in every family;
  * no anonymous placeholder collides with any constituent ticker or name in data/*constituents*.csv;
  * a template that reads the date renders identically under a different wall-clock date (pin works).
Double-BOS (a chat-template BOS plus the loader's forced BOS) is recorded, not changed.

Writes artifacts/<slug>/concept-cone-steering/runs/confirmation-v1-freeze-<date>/manifest.json per model.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from llm_bias.core.model import load_tokenizer_for_inference
from llm_bias.core.steering import prompts as P
from llm_bias.core.steering import protocol as R

V2_REFERENCE = "artifacts/{slug}/concept-cone-steering/runs/dim-crossmodel-layer-sweep-v2-20260925-smoke-01/tokenwise/result.json"
QWEN_REFERENCE = "artifacts/qwen3.5-4b/concept-cone-steering/runs/c2-guided-paper-20260924/result.json"


def constituent_identities() -> tuple[list[str], list[str]]:
    tickers, names = [], []
    for path in sorted(Path("data").glob("*constituents*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                tickers.append(row.get("ticker", ""))
                names.append(row.get("company_name", ""))
    return tickers, names


def date_pin_holds(tokenizer) -> bool | None:
    """Render under a patched 'tomorrow' clock; None when the template never reads the date."""
    if not R.template_uses_date(tokenizer):
        return None
    import transformers.utils.chat_template_utils as ctu

    prompt = P.render_decision_prompt("ABNB", "Airbnb", "balanced")
    fp_today = P.format_decision_prompt(tokenizer, prompt, suffix_tokens=16)
    original = ctu.datetime

    class Tomorrow(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2027, 1, 2, 3, 4)

    ctu.datetime = Tomorrow
    try:
        fp_other = P.format_decision_prompt(tokenizer, prompt, suffix_tokens=16)
    finally:
        ctu.datetime = original
    return fp_today.formatted == fp_other.formatted and R.TEMPLATE_DATE.strftime("%Y-%m-%d") in fp_today.formatted


def freeze_model(slug: str, companies: dict, run_id: str) -> dict:
    spec = R.model_spec(slug)
    tokenizer = load_tokenizer_for_inference(f".cache/models/{slug}")
    families = {}
    suffix_ids = set()
    for condition in P.CONDITIONS:
        named = [P.render_decision_prompt(t, c["name"], condition) for t, c in companies.items()]
        anon = [P.render_decision_prompt(t, n, condition) for t, n in P.ANON_IDENTITIES]
        k, ids = P.common_instruction_suffix(tokenizer, named + anon)
        if k != spec.suffix_tokens:
            raise ValueError(f"{slug}/{condition}: K={k} differs from registry {spec.suffix_tokens}")
        suffix_ids.add(ids[-spec.suffix_tokens:])
        families[condition] = {"prompt_family_sha256": P.prompt_family_sha256(companies, condition), "K": k}
    if len(suffix_ids) != 1:
        raise ValueError(f"{slug}: steer-suffix token ids differ across evidence families")
    reference = Path(QWEN_REFERENCE if slug == "qwen3.5-4b" else V2_REFERENCE.format(slug=slug))
    ref_sha = json.loads(reference.read_text(encoding="utf-8"))["metadata"]["prompt_family_sha256"]
    if families["balanced"]["prompt_family_sha256"] != ref_sha:
        raise ValueError(f"{slug}: balanced family hash differs from {reference}")
    sample = P.format_decision_prompt(tokenizer, P.render_decision_prompt("ABNB", "Airbnb", "balanced"),
                                      suffix_tokens=spec.suffix_tokens, key="ABNB")
    bos = getattr(tokenizer, "bos_token_id", None)
    pin = date_pin_holds(tokenizer)
    if pin is False:
        raise ValueError(f"{slug}: template date is not pinned")
    manifest = {
        "schema": "confirmation-v1-freeze", "model_slug": slug, "run_id": run_id,
        "tokenizer_config_sha256": R.sha256_bytes(Path(f".cache/models/{slug}/tokenizer_config.json").read_bytes()),
        **R.template_provenance(tokenizer), "date_pin_holds": pin, "families": families,
        "balanced_reference": str(reference), "balanced_matches_reference": True,
        "steer_suffix_ids_sha256": R.sha256_json(list(next(iter(suffix_ids)))),
        "double_bos": bool(bos is not None and sample.ids[:2] == (bos, bos)),
        "abnb_balanced": sample.provenance(),
        "anon_formatted_sha256": {f"anon:{i}": P.format_decision_prompt(
            tokenizer, P.render_decision_prompt(t, n, "balanced"), suffix_tokens=spec.suffix_tokens).ids_sha256
            for i, (t, n) in enumerate(P.ANON_IDENTITIES)},
    }
    out = Path("artifacts") / slug / "concept-cone-steering" / "runs" / run_id / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True, help="confirmation-v1-freeze-<date>")
    parser.add_argument("--models", nargs="+", default=list(R.MODEL_REGISTRY))
    args = parser.parse_args()
    companies = R.load_population(Path(R.POPULATION_CSV))
    tickers, names = constituent_identities()
    collisions = P.anon_collisions(companies, tickers, names)
    if collisions:
        raise ValueError(f"anonymous placeholders collide with real identities: {collisions}")
    for slug in args.models:
        manifest = freeze_model(slug, companies, args.run_id)
        print(slug, {c: f["K"] for c, f in manifest["families"].items()}, "double_bos", manifest["double_bos"],
              "date_pin", manifest["date_pin_holds"], flush=True)


if __name__ == "__main__":
    main()
