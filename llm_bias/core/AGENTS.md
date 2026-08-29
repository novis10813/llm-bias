# `llm_bias/core/` scope

本目錄保存 active experiments 共用且不帶單一研究線語意的 infrastructure。上層
`llm_bias/AGENTS.md` 定義 package ownership；模組責任、相容入口與 lens boundary 見
[`../../docs/shared-experiment-core.md`](../../docs/shared-experiment-core.md)。

## Local architecture

- `prompt_input/`：prepare 階段的 formatting、tokenization 與 span alignment。
- `inference/`：forward、generation、logit extraction 與暫態 intervention hooks。
- `analysis/`：transported readout、distribution/direction statistics 與 compact records。
- `artifacts/`：atomic serialization、parent hash verification、stage lifecycle 與 manifest
  finalization。
- 頂層 `model.py`、`continuation_scoring.py`、`artifact_*` 與 `lens_*` modules 保存跨
  experiment 的 model、scoring、artifact identity 與 validated lens loading mechanics。

Research-specific estimand、success gate、vocabulary、artifact schema 與 presentation 留在
owning experiment package。Core 不 import experiment package，也不執行 lens fitting、
installation 或 promotion。

## Compatibility rules

`prompting.py`、`readout.py`、`directions.py` 及 `artifacts/{manifest,paths}.py` 支援既有
callers。修改前搜尋 active code 與 `archive/`，不要把 facade 當成未使用 duplicate。
部分 inference/analysis symbols 只從 submodule 公開；沿用現有 import path。

`prompt_input.continuation_token_ids` 與 `continuation_scoring.continuation_token_ids` 的
return contract 不同。新增 caller 時使用完整 module path，並以測試固定預期型別。

## Artifact and lens boundaries

- Serializer 與 manifest 只接受 compact derived outputs；禁止持久化 raw activations、
  residuals、hidden states、gradients、Jacobians 或 KV caches。
- `ArtifactRun.finalize()` 是 required-stage postcheck，不是獨立 stage。
- Runtime lens loading 經 `lens_loader.load_validated_lens` 驗證 identity、shape、layer
  coverage 與 metadata。Canonical lens 規則見
  [`../../docs/qwen-jacobian-lens-selection.md`](../../docs/qwen-jacobian-lens-selection.md)。
- Artifact path、hash 與 lifecycle 契約見
  [`../../docs/artifact-contract.md`](../../docs/artifact-contract.md)。

## Instruction Index

目前 `core/` 的直接子目錄沒有 `AGENTS.md`。某個子套件出現無法由
[`../../docs/shared-experiment-core.md`](../../docs/shared-experiment-core.md) 與本檔一兩句
覆蓋的獨立 compatibility 或 lifecycle 規則時，才在該子套件新增 `AGENTS.md`，並只更新
本節。

## Local verification

修改 shared mechanics 時執行對應的 `tests/test_core_*.py`，並執行：

```bash
uv run pytest -q \
  tests/test_prompt_input.py \
  tests/test_artifact_manifest.py \
  tests/test_artifact_paths.py \
  tests/test_continuation_scoring.py \
  tests/test_workflow_boundaries.py
```

修改 lens identity、path 或 loader 時，依
[`../../docs/shared-experiment-core.md#verification`](../../docs/shared-experiment-core.md#verification)
補跑 lens tests。GPU/model inference 只作 smoke 或 integration，不加入一般 unit-test gate。
