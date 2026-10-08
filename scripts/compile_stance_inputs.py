"""Compile frozen stance inputs without models, ranking, or research eligibility.

The pure compiler permits small synthetic fixtures. Only main authorizes the
frozen 503-ticker research population and reviewed FactSet policy. Issuer IDs
are a disclosed snapshot-name/share-class mapping, not legal-entity verification.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from llm_bias.core.artifact_paths import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json
from llm_bias.core.population import (
    PopulationSnapshot, RoleAssignments, assign_roles, load_population,
    validate_members, validate_population_rows,
)
from llm_bias.core.stance_evidence import (
    EvidenceItem, EvidenceOrderContrast, EvidencePair, validate_evidence_item,
    validate_evidence_pairs,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/concept-cone-steering/rebuild-v1/inputs.json"
POPULATION_SHA256 = "27c454d250ce513fda2016b43e7009ccbe0f5381e6c7648d3efe0d418c501a38"
SOURCE_SHA256 = "677776cd01ddf02efcfb5fb4b437c336baa825f0dd41f3cdf9f3b1b829078486"
SOURCE_BYTE_SIZE = 1080178
# Semantic digest of the exact approved config, independent of file formatting.
# This guards reviewed prose/reviews/provenance as well as the explicit gates below.
APPROVED_POLICY_SHA256 = "4eb609d4449f77cac80c6736e16d7de1064e3f3dd75962428e440481fe1af9b4"
ISSUER_POLICY = {
    "prefix": "issuer:",
    "groups": [
        {"group_id": "issuer:fox-corporation", "tickers": ["FOX", "FOXA"],
         "expected_company_prefix": "Fox Corporation"},
        {"group_id": "issuer:alphabet-inc", "tickers": ["GOOG", "GOOGL"],
         "expected_company_prefix": "Alphabet Inc."},
        {"group_id": "issuer:news-corp", "tickers": ["NWS", "NWSA"],
         "expected_company_prefix": "News Corp"},
    ],
}
ROLE_COUNTS = {"fit": 300, "validation": 75, "calibration": 25, "evaluation": 100}
PAIRING = {"++": ["F24-P1", "F24-P2"], "+-": ["F24-P1", "F24-N1"],
           "-+": ["F24-N1", "F24-P1"], "--": ["F24-N1", "F24-N2"]}
TRIALS = ["factset-20241115-a"]
DATA_FILES = ("population.json", "roles.json", "evidence_pairs.jsonl")
MANIFEST = "inputs_manifest.json"
CONFIG_KEYS = {"schema_version", "population", "issuer_policy", "roles", "trials",
               "items", "pairing", "order_contrasts", "provenance"}
CODE_FILES = ("scripts/compile_stance_inputs.py", "llm_bias/core/population.py",
              "llm_bias/core/stance_evidence.py", "llm_bias/core/artifact_paths.py")


@dataclass(frozen=True)
class CompiledInputs:
    snapshot: PopulationSnapshot
    roles: RoleAssignments
    pairs: tuple[EvidencePair, ...]
    order_contrasts: tuple[EvidenceOrderContrast, ...]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite config number: {value}")


def parse_config(payload: bytes) -> dict[str, Any]:
    config = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("config must contain exactly the registered root fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("config schema_version must be 1")
    return config


def _positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def issuer_mapping(names: Mapping[str, str], policy: dict) -> dict[str, str]:
    """Apply explicit share-class overrides, never fuzzy name matching."""
    if set(policy) != {"prefix", "groups"}:
        raise ValueError("issuer policy fields must be prefix and groups")
    prefix = policy["prefix"]
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("issuer policy prefix must be nonblank")
    mapping = {ticker: prefix + ticker for ticker in names}
    seen_tickers, seen_ids = set(), set()
    for group in policy["groups"]:
        if set(group) != {"group_id", "tickers", "expected_company_prefix"}:
            raise ValueError("issuer group fields differ from registered shape")
        group_id, tickers, name = group["group_id"], group["tickers"], group["expected_company_prefix"]
        if (not isinstance(group_id, str) or not group_id.startswith(prefix)
                or group_id == prefix or group_id in seen_ids):
            raise ValueError("issuer group ID must be unique and use the policy prefix")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("issuer group company prefix must be nonblank")
        if not isinstance(tickers, list) or len(tickers) < 2:
            raise ValueError("issuer group must declare at least two share-class tickers")
        for ticker in tickers:
            if ticker not in names or ticker in seen_tickers:
                raise ValueError("issuer grouping tickers must be present and nonoverlapping")
            if not names[ticker].startswith(name):
                raise ValueError(f"issuer grouping snapshot name mismatch for {ticker}")
            seen_tickers.add(ticker)
            mapping[ticker] = group_id
        seen_ids.add(group_id)
    if seen_ids & {mapping[t] for t in names if t not in seen_tickers}:
        raise ValueError("issuer group ID collides with a unique snapshot ticker issuer")
    return mapping


def compile_input_records(
    snapshot: PopulationSnapshot, source_bytes: bytes, config: dict[str, Any],
) -> CompiledInputs:
    """Pure, fully covered compilation; fixture policy never authorizes CLI inputs."""
    parse_config(canonical_json_bytes(config))
    members = validate_members(snapshot.members)
    population = config["population"]
    if set(population) != {"source_sha256", "expected_tickers", "expected_issuers"}:
        raise ValueError("population config fields differ from registered shape")
    ticker_count = _positive_int(population["expected_tickers"], "expected ticker count")
    issuer_count = _positive_int(population["expected_issuers"], "expected issuer count")
    if snapshot.source_sha256 != population["source_sha256"]:
        raise ValueError("population source SHA-256 mismatch")
    membership_hash = sha256_json([asdict(member) for member in members])
    if membership_hash != snapshot.membership_sha256:
        raise ValueError("population membership hash mismatch")
    if len(members) != ticker_count or len({m.issuer_id for m in members}) != issuer_count:
        raise ValueError("population ticker/issuer count mismatch")
    mapping = issuer_mapping({m.ticker: m.name for m in members}, config["issuer_policy"])
    if mapping != {m.ticker: m.issuer_id for m in members}:
        raise ValueError("snapshot issuer mapping differs from explicit policy")
    snapshot = PopulationSnapshot(members, snapshot.source_sha256, membership_hash)

    provenance = config["provenance"]
    source_hash = sha256_bytes(source_bytes)
    if source_hash != provenance["source_pdf_sha256"]:
        raise ValueError("evidence source PDF SHA-256 mismatch")
    if len(source_bytes) != _positive_int(provenance["source_byte_size"], "source byte size"):
        raise ValueError("evidence source PDF byte size mismatch")
    if set(config["roles"]) != {"seed", "issuer_counts"} or type(config["roles"]["seed"]) is not int:
        raise ValueError("roles must specify integer seed and issuer_counts")
    roles = assign_roles(members, config["roles"]["seed"], config["roles"]["issuer_counts"])

    items = {}
    for record in config["items"]:
        item = validate_evidence_item(EvidenceItem(**record))
        if item.item_id in items:
            raise ValueError(f"duplicate evidence item ID: {item.item_id}")
        if (item.source_sha256 != source_hash or item.source != provenance["source_url"]
                or item.review_id != provenance["review_id"]):
            raise ValueError("item source/review does not match acquired source provenance")
        items[item.item_id] = item
    pairing = config["pairing"]
    if set(pairing) != {"++", "+-", "-+", "--"}:
        raise ValueError("pairing must contain exactly all four conditions")
    used = set()
    for slots in pairing.values():
        if not isinstance(slots, list) or len(slots) != 2:
            raise ValueError("pairing must declare exactly two item IDs per condition")
        if any(item_id not in items for item_id in slots):
            raise ValueError("pairing references foreign item ID")
        used.update(slots)
    if used != set(items):
        raise ValueError("all declared items must be used in pairing")
    contrasts = tuple(EvidenceOrderContrast(**record) for record in config["order_contrasts"])
    pairs = validate_evidence_pairs(
        members, config["trials"],
        (EvidencePair(member.ticker, trial, condition, items[a], items[b])
         for member in members for trial in config["trials"]
         for condition, (a, b) in pairing.items()),
        order_contrasts=contrasts,
    )
    expected_contrasts = {(trial, frozenset(("+-", "-+"))) for trial in config["trials"]}
    if {(c.trial_id, frozenset((c.condition1, c.condition2))) for c in contrasts} != expected_contrasts:
        raise ValueError("each trial requires the exact mixed order contrast")
    return CompiledInputs(snapshot, roles, pairs,
                          tuple(sorted(contrasts, key=lambda c: (c.trial_id, c.condition1, c.condition2))))


def serialize_inputs(compiled: CompiledInputs, *, config_bytes: bytes) -> dict[str, bytes]:
    """Canonical portable payloads; local acquisition paths are not identity."""
    config = parse_config(config_bytes)
    snapshot, roles = compiled.snapshot, compiled.roles
    payloads = {
        "population.json": canonical_json_bytes(asdict(snapshot)) + b"\n",
        "roles.json": canonical_json_bytes(asdict(roles)) + b"\n",
        "evidence_pairs.jsonl": b"".join(canonical_json_bytes(asdict(pair)) + b"\n"
                                        for pair in compiled.pairs),
    }
    manifest = {
        "schema_version": 1,
        "config_sha256": sha256_bytes(config_bytes),
        "population_source_sha256": snapshot.source_sha256,
        "evidence_source_sha256": config["provenance"]["source_pdf_sha256"],
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "output_hashes": {name: sha256_bytes(payload) for name, payload in payloads.items()},
        "counts": {
            "population_count": len(snapshot.members),
            "issuer_count": len({m.issuer_id for m in snapshot.members}),
            "unspecified_sector_count": sum(m.sector == "Unspecified" for m in snapshot.members),
            "evidence_pair_count": len(compiled.pairs),
            "ticker_counts": roles.ticker_counts,
            "issuer_counts": roles.issuer_counts,
        },
        "semantic_hashes": {
            "membership_hash": snapshot.membership_sha256,
            "roles_hash": roles.assignment_hash,
            "issuer_mapping_hash": sha256_json({m.ticker: m.issuer_id for m in snapshot.members}),
        },
        "eligibility": {
            "research_eligible": False,
            "note": "Compiled inputs only; no model execution, generation outcomes or research gates.",
        },
        "provenance": config["provenance"],
    }
    payloads[MANIFEST] = canonical_json_bytes(manifest) + b"\n"
    return payloads


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_outputs(out_dir: str | Path, payloads: dict[str, bytes]) -> None:
    """Stage durable files, then publish one directory; never overwrite outputs."""
    out = Path(out_dir)
    expected = set(DATA_FILES) | {MANIFEST}
    if set(payloads) != expected or any(not isinstance(p, bytes) for p in payloads.values()):
        raise ValueError("outputs must be exactly four byte payloads")
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise ValueError("output must be a regular directory, not a symlink")
    if out.exists() and any(out.iterdir()):
        if set(p.name for p in out.iterdir()) != expected:
            raise ValueError("no overwrite: existing nonempty directory needs exactly four manifest outputs")
        if any((out / name).is_symlink() or not (out / name).is_file()
               or (out / name).read_bytes() != payloads[name] for name in expected):
            raise ValueError("no overwrite: existing outputs must be byte-identical")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        for name in (*DATA_FILES, MANIFEST):
            with (stage / name).open("xb") as stream:
                stream.write(payloads[name])
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(stage)
        # POSIX rename may replace an empty directory, never a nonempty one.
        os.rename(stage, out)
        _fsync_directory(out.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def validate_research_policy(config: dict[str, Any]) -> None:
    """Immutable research gates cannot be weakened by editing input config."""
    if (config["population"] != {"source_sha256": POPULATION_SHA256,
                                "expected_tickers": 503, "expected_issuers": 500}
            or config["issuer_policy"] != ISSUER_POLICY
            or config["roles"] != {"seed": 20260930, "issuer_counts": ROLE_COUNTS}
            or config["pairing"] != PAIRING or config["trials"] != TRIALS
            or config["order_contrasts"] != [{"trial_id": TRIALS[0], "condition1": "+-", "condition2": "-+"}]
            or config["provenance"]["source_pdf_sha256"] != SOURCE_SHA256
            or config["provenance"]["source_byte_size"] != SOURCE_BYTE_SIZE
            or sha256_json(config) != APPROVED_POLICY_SHA256):
        raise ValueError("config differs from the immutable approved research input policy")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--population-csv", type=Path, default=ROOT / "data/sp500_constituents_2020_2025.csv")
    parser.add_argument("--source-pdf", type=Path,
                        default=ROOT / "data/concept-cone-steering/rebuild-v1/sources/factset_20241115.pdf")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config_bytes = args.config.read_bytes()
        config = parse_config(config_bytes)
        validate_research_policy(config)
        population_bytes = args.population_csv.read_bytes()
        if sha256_bytes(population_bytes) != POPULATION_SHA256:
            raise ValueError("population source SHA-256 differs from the frozen digest")
        rows = validate_population_rows(csv.DictReader(io.StringIO(population_bytes.decode("utf-8"), newline="")))
        mapping = issuer_mapping({r["ticker"]: r["company_name"] for r in rows}, config["issuer_policy"])
        snapshot = load_population(args.population_csv, mapping)
        if len(snapshot.members) != 503 or len(set(mapping.values())) != 500:
            raise ValueError("research population must contain exactly 503 tickers / 500 issuers")
        try:
            source_bytes = args.source_pdf.read_bytes()
        except OSError as exc:
            raise ValueError(f"evidence source PDF unavailable: {args.source_pdf}: {exc}") from exc
        if sha256_bytes(source_bytes) != SOURCE_SHA256 or len(source_bytes) != SOURCE_BYTE_SIZE:
            raise ValueError("evidence source PDF differs from the frozen SHA-256 / byte size")
        compiled = compile_input_records(snapshot, source_bytes, config)
        if compiled.roles.issuer_counts != ROLE_COUNTS or len(compiled.pairs) != 2012:
            raise ValueError("research roles/pairs differ from the fixed counts")
        payloads = serialize_inputs(compiled, config_bytes=config_bytes)
        write_outputs(args.out_dir, payloads)
        print(json.dumps({"out_dir": str(args.out_dir),
                          "source_paths": {"population_csv": str(args.population_csv),
                                           "source_pdf": str(args.source_pdf)},
                          "manifest_sha256": sha256_bytes(payloads[MANIFEST]),
                          "counts": json.loads(payloads[MANIFEST])["counts"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"compile_stance_inputs: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
