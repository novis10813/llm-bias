# Sector and Context Follow-up Experiments: Discovery Report

A V1、B V1 與 C V1 的問題、interventions、controls、outcomes 與 artifact contract 見
[實驗提案](proposal.md)。本報告只保存 discovery run record 與 version status。

## Discovered runs

以下為 2026-08-31 discovery batch 的 run 狀態；三者都還沒有 frozen calibration/test
gate，discovery outputs 不支撐 formal claim。

### A

- `cross-sector-header-aapl-jpm-smoke-20260831`：AAPL↔JPM smoke pair；complete。
- `cross-sector-header-discovery-20260831`：forward stage failed；未產出 analysis，由下一
  run 取代。
- `cross-sector-header-discovery-rot13-bounded-20260831`：complete。602 prepared records
  （slug-level `prepared/discovery_pairs_v1-rot13.jsonl`；control types：`cross_sector`
  116、`name_form` 116、`same_sector_peer` 138、`self_source` 232）、L0–L30、兩個 sector
  directions；30,132 compact records。

### B

- `cross-sector-context-aapl-jpm-smoke-20260831`：smoke pair；complete。
- `cross-sector-context-discovery-20260831`：complete。301 negative-evidence pairs、
  L14–L21、primary `instruction_context` 加 `header` 與 `final_position` control spans、
  4 個 control types；11,664 records；`analyze/discovery_localization.json` 提供
  per-layer toward-source localization（descriptive only）。
- `cross-sector-context-name-form-rot13-followup-20260831`：run failed（只有 prepare
  完成）；由 v2 取代。
- `cross-sector-context-name-form-rot13-followup-v2-20260831`：complete。58 pairs、
  2,784 records 的 name-form control follow-up。

### C

- `l16-context-readout-smoke-20260831`：smoke；complete。
- `l16-context-readout-discovery-20260831`：complete；Technology discovery split，綁定
  frozen config `configs/l16-context-readout-v1.json` 與 canonical lens SHA-256。
- `l16-context-readout-financial-services-discovery-v2-20260831`：complete；Financial
  Services discovery split。

## Version Record

| Experiment | Version | Status |
|---|---|---|
| Cross-sector header-state patching | A V1 | Discovery complete（`cross-sector-header-discovery-rot13-bounded-20260831`）；calibration/test gate 未凍結 |
| Cross-sector context overriding | B V1 | Discovery complete（`cross-sector-context-discovery-20260831`）與 name-form control follow-up v2；calibration/test gate 未凍結 |
| L16 instruction-context readout | C V1 | Implemented；Technology 與 Financial Services discovery runs complete；calibration/test gate 未凍結 |
