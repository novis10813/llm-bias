"""Build an auditable, prompt-agnostic 10-K metadata-change window CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "ten-k-change-window-items-v1"
DEFAULT_INPUT = "../10-k/edgar-crawler/datasets/EXTRACTED_FILINGS/10-K"
DEFAULT_OUTPUT = "artifacts/ten_k_change_windows/v1"
TRACKED_FIELDS = ("company", "state_location", "state_of_inc", "sic")
CSV_FIELDS = ("year", "cik", "item")
OUTPUT_NAMES = (
    "change_window_items.csv",
    "change_events.jsonl",
    "canonical_exclusions.jsonl",
    "input_issues.jsonl",
)


class TenKChangeDataError(RuntimeError):
    """Raised when source filings or published metadata artifacts are invalid."""


@dataclass(frozen=True)
class FilingObservation:
    cik: str
    company: str
    filing_date: str
    period_of_report: str
    sic: str
    state_location: str
    state_of_inc: str
    source_file: str
    source_sha256: str

    @property
    def fiscal_year(self) -> int:
        return int(self.period_of_report[:4])

    def audit_row(self) -> dict[str, str | int]:
        return {
            "cik": self.cik,
            "fiscal_year": self.fiscal_year,
            "period_of_report": self.period_of_report,
            "filing_date": self.filing_date,
            "company": self.company,
            "state_location": self.state_location,
            "state_of_inc": self.state_of_inc,
            "sic": self.sic,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class ChangeEvent:
    event_id: str
    before: FilingObservation
    after: FilingObservation
    changed_fields: tuple[str, ...]

    @property
    def fiscal_year(self) -> int:
        return self.after.fiscal_year

    def audit_row(self, window: Iterable[FilingObservation]) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "cik": self.after.cik,
            "event_fiscal_year": self.fiscal_year,
            "event_period_of_report": self.after.period_of_report,
            "before": self.before.audit_row(),
            "after": self.after.audit_row(),
            "changed_fields": {
                field: {"before": getattr(self.before, field), "after": getattr(self.after, field)}
                for field in self.changed_fields
            },
            "missing_window_years": [
                year
                for year in range(self.fiscal_year - 2, self.fiscal_year + 3)
                if year not in {row.fiscal_year for row in window}
            ],
            "window_filings": [row.audit_row() for row in window],
        }


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    with path.open("w", encoding="utf-8") as handle:
        return sum(handle.write(_json_line(row)) is not None for row in rows)


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _required_string(data: dict[str, Any], field: str, source: Path) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TenKChangeDataError(f"{source.name}: missing non-empty {field!r}")
    return value.strip()


def _optional_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    return value.strip() if isinstance(value, str) else ""


def _iso_date(value: str, field: str, source: Path) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise TenKChangeDataError(f"{source.name}: invalid {field} {value!r}") from error
    return value


def _load_observation(source: Path) -> FilingObservation:
    try:
        content = source.read_bytes()
        data = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TenKChangeDataError(f"{source.name}: invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise TenKChangeDataError(f"{source.name}: JSON root must be an object")
    if _required_string(data, "filing_type", source) != "10-K":
        raise TenKChangeDataError(f"{source.name}: filing_type must be '10-K'")
    cik = _required_string(data, "cik", source).lstrip("0") or "0"
    return FilingObservation(
        cik=cik,
        company=_required_string(data, "company", source),
        filing_date=_iso_date(_required_string(data, "filing_date", source), "filing_date", source),
        period_of_report=_iso_date(_required_string(data, "period_of_report", source), "period_of_report", source),
        sic=_optional_string(data, "sic"),
        state_location=_optional_string(data, "state_location"),
        state_of_inc=_optional_string(data, "state_of_inc"),
        source_file=source.name,
        source_sha256=_sha256_bytes(content),
    )


def _input_issue(source: Path, error: TenKChangeDataError) -> dict[str, Any]:
    try:
        content = source.read_bytes()
    except OSError:
        content = b""
    return {
        "source_file": source.name,
        "reason": "zero_byte_json" if not content else "invalid_source",
        "bytes": len(content),
        "source_sha256": _sha256_bytes(content),
        "message": str(error),
        "action": "excluded_from_observations",
    }


def _canonicalize(observations: Iterable[FilingObservation]) -> tuple[list[FilingObservation], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[FilingObservation]] = defaultdict(list)
    for row in observations:
        groups[(row.cik, row.period_of_report)].append(row)
    canonical, exclusions = [], []
    for key in sorted(groups, key=lambda value: (int(value[0]), value[1])):
        rows = sorted(groups[key], key=lambda row: (row.filing_date, row.source_file), reverse=True)
        canonical.append(rows[0])
        exclusions.extend(
            {
                "cik": row.cik,
                "period_of_report": row.period_of_report,
                "source_file": row.source_file,
                "selected_source_file": rows[0].source_file,
                "reason": "same_cik_and_period_select_latest_filing_date_then_filename",
            }
            for row in rows[1:]
        )
    return sorted(canonical, key=lambda row: (int(row.cik), row.period_of_report, row.source_file)), exclusions


def _changed_fields(before: FilingObservation, after: FilingObservation) -> tuple[str, ...]:
    return tuple(
        field
        for field in TRACKED_FIELDS
        if (before_value := getattr(before, field)) and (after_value := getattr(after, field)) and before_value != after_value
    )


def _change_events(canonical: Iterable[FilingObservation]) -> list[ChangeEvent]:
    by_cik: dict[str, list[FilingObservation]] = defaultdict(list)
    for row in canonical:
        by_cik[row.cik].append(row)
    events = []
    for cik, rows in sorted(by_cik.items(), key=lambda pair: int(pair[0])):
        ordered = sorted(rows, key=lambda row: (row.period_of_report, row.source_file))
        for before, after in zip(ordered, ordered[1:]):
            changed = _changed_fields(before, after)
            if changed:
                events.append(ChangeEvent(f"{cik}:{after.period_of_report}", before, after, changed))
    return events


def _window_rows(canonical: Iterable[FilingObservation], event: ChangeEvent) -> list[FilingObservation]:
    return sorted(
        (
            row
            for row in canonical
            if row.cik == event.after.cik and abs(row.fiscal_year - event.fiscal_year) <= 2
        ),
        key=lambda row: (row.fiscal_year, row.period_of_report, row.source_file),
    )


def _csv_rows(events: Iterable[ChangeEvent], canonical: Iterable[FilingObservation]) -> list[dict[str, str | int]]:
    rows = []
    for event in events:
        for filing in _window_rows(canonical, event):
            for field in event.changed_fields:
                rows.append({"year": filing.fiscal_year, "cik": filing.cik, "item": f"{field}={getattr(filing, field)}"})
    return sorted(rows, key=lambda row: (int(row["cik"]), int(row["year"]), str(row["item"])))


def build_change_dataset(
    input_dir: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    max_files: int | None = None,
    fail_on_input_issues: bool = False,
) -> dict[str, Any]:
    """Publish one ``year,cik,item`` CSV scoped to metadata-change windows."""
    source_root, destination = Path(input_dir).resolve(), Path(output_dir)
    if not source_root.is_dir():
        raise TenKChangeDataError(f"input directory does not exist: {source_root}")
    if destination.exists():
        raise TenKChangeDataError(f"output directory already exists: {destination}")
    if max_files is not None and max_files < 1:
        raise TenKChangeDataError("max_files must be positive")
    sources = sorted(source_root.glob("*.json"), key=lambda path: path.name)
    if max_files is not None:
        sources = sources[:max_files]
    if not sources:
        raise TenKChangeDataError(f"no JSON files found under {source_root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    started = datetime.now(UTC)
    try:
        observations, issues = [], []
        for source in sources:
            try:
                observations.append(_load_observation(source))
            except TenKChangeDataError as error:
                issues.append(_input_issue(source, error))
        if not observations:
            raise TenKChangeDataError("no valid 10-K observations")
        canonical, exclusions = _canonicalize(observations)
        events = _change_events(canonical)
        csv_rows = _csv_rows(events, canonical)
        audit_events = [event.audit_row(_window_rows(canonical, event)) for event in events]
        with (temporary / "change_window_items.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        _write_jsonl(audit_events, temporary / "change_events.jsonl")
        _write_jsonl(exclusions, temporary / "canonical_exclusions.jsonl")
        _write_jsonl(issues, temporary / "input_issues.jsonl")
        outputs = {name: {"sha256": _sha256_file(temporary / name), "bytes": (temporary / name).stat().st_size} for name in OUTPUT_NAMES}
        missing_metadata = Counter(field for row in canonical for field in TRACKED_FIELDS if not getattr(row, field))
        changed_field_counts = Counter(field for event in events for field in event.changed_fields)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "stage": "build",
            "status": "complete_with_input_issues" if issues else "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "duration_seconds": (datetime.now(UTC) - started).total_seconds(),
            "input_root": str(source_root),
            "input_glob": "*.json",
            "parameters": {"max_files": max_files, "fail_on_input_issues": fail_on_input_issues, "tracked_fields": list(TRACKED_FIELDS), "window_years": 2},
            "counts": {
                "source_files": len(sources),
                "valid_source_files": len(observations),
                "excluded_source_files": len(issues),
                "input_issues": len(issues),
                "canonical_filings": len(canonical),
                "ciks": len({row.cik for row in canonical}),
                "canonical_exclusions": len(exclusions),
                "change_events": len(events),
                "change_window_item_rows": len(csv_rows),
                "changed_field_counts": dict(sorted(changed_field_counts.items())),
                "missing_metadata_counts": dict(sorted(missing_metadata.items())),
            },
            "csv_fields": list(CSV_FIELDS),
            "outputs": outputs,
            "source_limitations": [
                "Only successfully extracted JSON files are observed; absent or excluded files do not prove absent filings.",
                "year is derived from period_of_report; filing_date remains audit-only submission provenance.",
                "The only CSV is prompt-agnostic and contains no inferred answer or model output.",
                "Rows are restricted to fiscal-year windows around detected metadata changes; overlapping events may duplicate rows.",
            ],
        }
        _write_json(manifest, temporary / "manifest.json")
        os.replace(temporary, destination)
        if fail_on_input_issues and issues:
            raise TenKChangeDataError(f"published dataset has {len(issues)} input issue(s)")
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise TenKChangeDataError(f"{path.name}:{number}: blank JSONL line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise TenKChangeDataError(f"{path.name}:{number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise TenKChangeDataError(f"{path.name}:{number}: row must be an object")
        rows.append(row)
    return rows


def _validate_event(row: dict[str, Any]) -> list[dict[str, str | int]]:
    try:
        cik = row["cik"]
        event_year = int(row["event_fiscal_year"])
        event_period = row["event_period_of_report"]
        before, after = row["before"], row["after"]
        changes, filings = row["changed_fields"], row["window_filings"]
    except (KeyError, TypeError, ValueError) as error:
        raise TenKChangeDataError("change_events.jsonl: invalid event record") from error
    if not isinstance(cik, str) or row.get("event_id") != f"{cik}:{event_period}" or not isinstance(changes, dict) or not changes:
        raise TenKChangeDataError("change_events.jsonl: invalid event identity")
    if not isinstance(before, dict) or not isinstance(after, dict) or after.get("cik") != cik or int(after.get("fiscal_year", -1)) != event_year:
        raise TenKChangeDataError("change_events.jsonl: invalid event boundary")
    expected_items = []
    years = set()
    for field, change in changes.items():
        if field not in TRACKED_FIELDS or not isinstance(change, dict) or not change.get("before") or not change.get("after") or change["before"] == change["after"]:
            raise TenKChangeDataError("change_events.jsonl: invalid changed field")
        if before.get(field) != change["before"] or after.get(field) != change["after"]:
            raise TenKChangeDataError("change_events.jsonl: changed field disagrees with boundary")
    if not isinstance(filings, list) or not filings:
        raise TenKChangeDataError("change_events.jsonl: missing event window")
    for filing in filings:
        try:
            filing_year = int(filing["fiscal_year"])
            date.fromisoformat(filing["period_of_report"])
            date.fromisoformat(filing["filing_date"])
        except (KeyError, TypeError, ValueError) as error:
            raise TenKChangeDataError("change_events.jsonl: invalid window filing") from error
        if filing.get("cik") != cik or abs(filing_year - event_year) > 2:
            raise TenKChangeDataError("change_events.jsonl: filing outside event window")
        years.add(filing_year)
        for field in TRACKED_FIELDS:
            if field in changes:
                expected_items.append({"year": filing_year, "cik": cik, "item": f"{field}={filing.get(field, '')}"})
    if not any(filing.get("source_sha256") == after.get("source_sha256") for filing in filings):
        raise TenKChangeDataError("change_events.jsonl: event filing absent from window")
    missing = row.get("missing_window_years")
    expected_missing = [year for year in range(event_year - 2, event_year + 3) if year not in years]
    if missing != expected_missing:
        raise TenKChangeDataError("change_events.jsonl: invalid missing window years")
    return expected_items


def validate_change_dataset(input_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Verify the one CSV, its event audit, and manifest integrity."""
    root = Path(input_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise TenKChangeDataError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise TenKChangeDataError("unsupported manifest schema_version")
    if tuple(manifest.get("csv_fields", ())) != CSV_FIELDS:
        raise TenKChangeDataError("manifest: invalid CSV fields")
    if set(manifest.get("outputs", {})) != set(OUTPUT_NAMES):
        raise TenKChangeDataError("manifest: invalid outputs")
    for name, metadata in manifest["outputs"].items():
        if _sha256_file(root / name) != metadata.get("sha256"):
            raise TenKChangeDataError(f"hash mismatch: {root / name}")
    with (root / "change_window_items.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise TenKChangeDataError("change_window_items.csv: invalid header")
        csv_rows = list(reader)
    events = _read_jsonl(root / "change_events.jsonl")
    expected_rows = []
    event_ids = set()
    for event in events:
        if event.get("event_id") in event_ids:
            raise TenKChangeDataError("change_events.jsonl: duplicate event")
        event_ids.add(event.get("event_id"))
        expected_rows.extend(_validate_event(event))
    expected_rows.sort(key=lambda row: (int(row["cik"]), int(row["year"]), str(row["item"])))
    normalized_rows = []
    for row in csv_rows:
        try:
            year = int(row["year"])
        except (KeyError, ValueError) as error:
            raise TenKChangeDataError("change_window_items.csv: invalid year") from error
        item = row.get("item", "")
        if not any(item.startswith(f"{field}=") for field in TRACKED_FIELDS):
            raise TenKChangeDataError("change_window_items.csv: invalid item")
        normalized_rows.append({"year": year, "cik": row.get("cik", ""), "item": item})
    if normalized_rows != expected_rows:
        raise TenKChangeDataError("change_window_items.csv: rows disagree with event audit")
    issues = _read_jsonl(root / "input_issues.jsonl")
    if any(row.get("reason") not in {"zero_byte_json", "invalid_source"} or row.get("action") != "excluded_from_observations" for row in issues):
        raise TenKChangeDataError("input_issues.jsonl: invalid issue record")
    if bool(issues) != (manifest.get("status") == "complete_with_input_issues"):
        raise TenKChangeDataError("manifest status does not match input issues")
    actual = {"change_events": len(events), "change_window_item_rows": len(csv_rows), "input_issues": len(issues), "excluded_source_files": len(issues)}
    if any(manifest.get("counts", {}).get(key) != value for key, value in actual.items()):
        raise TenKChangeDataError(f"manifest count mismatch: expected {manifest.get('counts')}, got {actual}")
    report = {"schema_version": SCHEMA_VERSION, "stage": "validate", "status": "passed", "counts": actual}
    _write_json(report, root / "validation_report.json")
    return report
