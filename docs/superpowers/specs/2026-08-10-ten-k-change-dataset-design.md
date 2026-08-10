# 10-K Metadata-Change Dataset (year,cik,item) Design Spec

## Overview
Update the 10-K metadata change dataset pipeline (`llm_bias/ten_k_change_data/pipeline.py`, CLI, tests, and documentation) to produce a `year,cik,item` CSV format instead of `year,sic,item`.

## CSV Schema & Output
- Output CSV filename: `change_window_items.csv`
- Header: `year,cik,item`
- Columns:
  - `year`: Fiscal year of the filing (integer, derived from `period_of_report`)
  - `cik`: CIK of the entity (string)
  - `item`: Field name and value formatted as `field=value` (string)

## Tracked Fields
- `company`: Company name
- `state_location`: Business location (state)
- `state_of_inc`: State of incorporation
- `sic`: Standard Industrial Classification code

## Window Logic
- For a metadata change event occurring at year $T$, filings within fiscal year window $[T-2, T+2]$ (5 fiscal years) are included.
- For each canonical filing in the window, a CSV row `{"year": filing.fiscal_year, "cik": filing.cik, "item": f"{field}={getattr(filing, field)}"}` is produced for every field changed in the event.

## Affected Files
1. `llm_bias/ten_k_change_data/pipeline.py`
2. `llm_bias/ten_k_change_data/cli.py`
3. `tests/test_ten_k_change_data.py`
4. `docs/ten-k-change-dataset.md`
5. `README.md`
