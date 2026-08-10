# 10-K Metadata-Change Dataset (year,cik,item) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update `prepare-10k-change-data` pipeline, CLI, tests, and documentation to produce a `year,cik,item` CSV format instead of `year,sic,item`.

**Architecture:** Modify `CSV_FIELDS` in `pipeline.py` to `("year", "cik", "item")`, update row construction in `_csv_rows`, update validation functions `_validate_event` and `validate_change_dataset`, update tests in `test_ten_k_change_data.py`, and update CLI description, `docs/ten-k-change-dataset.md`, and `README.md`.

**Tech Stack:** Python 3.10+, pytest, uv.

## Global Constraints
- Python dependencies managed by `uv`.
- Format of CSV header must be `year,cik,item`.
- Tracked fields: `("company", "state_location", "state_of_inc", "sic")`.
- Window range: 5 fiscal years (`[event - 2, event + 2]`).

---

### Task 1: Update Pipeline and CLI for `year,cik,item` CSV format

**Files:**
- Modify: `llm_bias/ten_k_change_data/pipeline.py:21,235-243,375-385,420-430`
- Modify: `llm_bias/ten_k_change_data/cli.py:1,19`
- Test: `tests/test_ten_k_change_data.py`

**Interfaces:**
- Consumes: Existing `FilingObservation` and `ChangeEvent` data classes.
- Produces: Updated `build_change_dataset` and `validate_change_dataset` functions producing `year,cik,item` CSV format.

- [ ] **Step 1: Write failing test in `tests/test_ten_k_change_data.py`**

Update `tests/test_ten_k_change_data.py` to expect `["year", "cik", "item"]` header and `"cik"` in rows.

```python
def test_build_writes_only_changed_items_for_event_window(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    _years(source, company="NEW ACME INC", state_location="NY")
    manifest = build_change_dataset(source, tmp_path / "output")
    rows = _csv(tmp_path / "output" / "change_window_items.csv")
    assert list(rows[0]) == ["year", "cik", "item"]
    assert manifest["counts"]["change_events"] == 1
    assert manifest["counts"]["change_window_item_rows"] == 10
    expected = []
    for year in range(2018, 2023):
        expected.extend(
            [
                {"year": str(year), "cik": "1", "item": f"company={'ACME INC' if year < 2020 else 'NEW ACME INC'}"},
                {"year": str(year), "cik": "1", "item": f"state_location={'CA' if year < 2020 else 'NY'}"},
            ]
        )
    assert rows == expected
    assert validate_change_dataset(tmp_path / "output")["status"] == "passed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ten_k_change_data.py -v`
Expected: FAIL due to header mismatch (`["year", "sic", "item"]` vs `["year", "cik", "item"]`).

- [ ] **Step 3: Update `pipeline.py` and `cli.py`**

In `llm_bias/ten_k_change_data/pipeline.py`:
1. Change `CSV_FIELDS = ("year", "sic", "item")` to `CSV_FIELDS = ("year", "cik", "item")`.
2. In `_csv_rows()`, change `"sic": filing.sic` to `"cik": filing.cik`.
3. In `_validate_event()`, change `"sic": filing.get("sic", "")` to `"cik": cik`.
4. In `validate_change_dataset()`, change `"sic": row.get("sic", "")` to `"cik": row.get("cik", "")`.

In `llm_bias/ten_k_change_data/cli.py`:
1. Update docstring and subcommand description to reference `year,cik,item`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ten_k_change_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add llm_bias/ten_k_change_data/pipeline.py llm_bias/ten_k_change_data/cli.py tests/test_ten_k_change_data.py
git commit -m "feat(ten_k_change_data): update CSV output header and rows to year,cik,item"
```

---

### Task 2: Update Tests for Complete Coverage of `year,cik,item` CSV

**Files:**
- Modify: `tests/test_ten_k_change_data.py`

**Interfaces:**
- Consumes: `build_change_dataset` and `validate_change_dataset`.
- Produces: Updated test assertions for `year,cik,item` CSV.

- [ ] **Step 1: Update all test helper assertions in `tests/test_ten_k_change_data.py`**

Update `test_window_excludes_missing_years_and_empty_metadata_is_not_change` and `test_validator_detects_tampered_csv_and_cli_defaults`:
- Replace `{"year": "2018", "sic": "3571", ...}` with `{"year": "2018", "cik": "1", ...}`.
- Replace tampered CSV write string `2020,9999,company=bad\n` with `2020,1,company=bad\n`.

- [ ] **Step 2: Run pytest to verify all tests pass**

Run: `uv run pytest tests/test_ten_k_change_data.py -v`
Expected: PASS.

- [ ] **Step 3: Commit changes**

```bash
git add tests/test_ten_k_change_data.py
git commit -m "test(ten_k_change_data): update all test fixtures and assertions to year,cik,item"
```

---

### Task 3: Update Documentation and Repository Guidance

**Files:**
- Modify: `docs/ten-k-change-dataset.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Updated `year,cik,item` CSV schema.
- Produces: Updated documentation files.

- [ ] **Step 1: Update `docs/ten-k-change-dataset.md`**

Update description and code snippets:
- Header description: `year,cik,item`
- Item field description: `cik`: Filing entity CIK.
- CSV snippet example:
```csv
year,cik,item
2019,320193,company=ACME INC
2019,320193,state_location=CA
2020,320193,company=NEW ACME INC
2020,320193,state_location=NY
```
- Update text references from `year,sic,item` to `year,cik,item`.

- [ ] **Step 2: Update `README.md`**

Update workflow and map descriptions from `year,sic,item` to `year,cik,item`.

- [ ] **Step 3: Run standard verification commands**

Run:
```bash
uv lock --check
uv run pytest -q
uv run python -m compileall -q llm_bias
```
Expected: All pass without errors.

- [ ] **Step 4: Commit changes**

```bash
git add docs/ten-k-change-dataset.md README.md
git commit -m "docs(ten_k_change_data): update documentation to reflect year,cik,item CSV schema"
```
