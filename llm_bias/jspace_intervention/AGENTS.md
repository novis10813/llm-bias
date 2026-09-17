# `llm_bias/jspace_intervention/` guidance

本目錄擁有 active J-space、residual intervention 與 transported readout 實驗。上層
[`../AGENTS.md`](../AGENTS.md) 定義 package ownership 與 shared-core 邊界；本檔只補充
本 package 內的 protocol 分工與局部驗證。實驗定義、gates 與 run records 以對應的
`docs/<experiment-name>/` proposal/report 為準。

## Local architecture

- `pipeline.py`、`intervention.py`、`analysis.py`、`concepts.py` 與 config modules：sector
  coordinate swap/gain 與 dose-matched controls；見
  [`../../docs/jspace-sector-intervention/proposal.md`](../../docs/jspace-sector-intervention/proposal.md)。
- `valence_readout.py`：positive/negative evidence 的 transported vocabulary readout；見
  [`../../docs/jspace-valence-readout/proposal.md`](../../docs/jspace-valence-readout/proposal.md)。
- `token_screen.py` 與相關 pipeline/config：J-space token experiment V1；V1/V2 邊界由
  [J-space token V1/V2 報告](../../docs/jspace-token-experiments/report.md)
  定義。
- `outcome_flip.py`、`prior_probe.py`、`outcome_decode.py`、`outcome_geometry.py`：V2
  outcome-conditioned direction 的 formal intervention 與輔助 diagnostics；不得把
  diagnostic 結果回填為 V2 primary gate。
- `activation_patching.py`：positive/negative valence pair 的 residual resample patching
  與 frozen confirmation analysis；見
  [`../../docs/activation-patching-causal-tracing/proposal.md`](../../docs/activation-patching-causal-tracing/proposal.md)。
- `cross_sector_patching.py`、`context_overriding.py`、`context_readout.py`：sector/context
  follow-up A V1、B V1、C V1；見
  [`../../docs/sector-context-followup/proposal.md`](../../docs/sector-context-followup/proposal.md)。
- `cli.py` 是單一 public command router。新增 subcommand 時，先在 owning proposal 定義
  protocol 與 artifact identity，再更新 parser、dispatch、tests 與
  [`../../docs/research-scripts.md`](../../docs/research-scripts.md) 或 README 的入口摘要。

## Local rules

- V1、V2、activation patching 與 sector/context A/B/C 是不同 experiments。不得共用
  result schema、run verdict 或 success gate，除非 canonical proposal 明確定義 shared
  input contract。
- V2 auxiliary workflows 應重用 `outcome_flip.py` 的 deterministic direction recompute、
  identity verification 與 fail-closed input binding；不要複製另一套驗證。
- B V1 重用 A V1 的 prepared pair records；C V1 另綁 frozen vocabulary config 與
  canonical lens SHA。改動這些 bindings 時同步更新 proposal、artifact tests 與 CLI tests。
- Activation patching 和 cross-sector patching 只在記憶體保留 residual states。Disk 輸出
  限 compact margin、transfer、flip、top-k/rank、統計與 provenance。
- Research-specific span definitions、estimands、controls、thresholds 與 report rendering
  留在本 package；只有跨 experiment 的 mechanics 才能移入 `llm_bias/core/`。

## Instruction Index

目前 `jspace_intervention/` 的直接子目錄沒有 `AGENTS.md`。若未來某個 workflow 移入獨立
子package，且形成自己的 schema/versioning 或 runtime 規則，再新增局部 `AGENTS.md`，並
只更新本節。

## Local verification

依修改範圍執行 targeted tests：

```bash
uv run pytest -q \
  tests/test_jspace_intervention.py \
  tests/test_jspace_valence_readout.py \
  tests/test_jspace_token_screen.py \
  tests/test_jspace_outcome_flip.py \
  tests/test_jspace_prior_probe.py \
  tests/test_jspace_outcome_decode.py \
  tests/test_jspace_outcome_geometry.py \
  tests/test_jspace_activation_patching.py \
  tests/test_cross_sector_patching.py \
  tests/test_context_overriding.py \
  tests/test_context_readout.py
```

修改 `cli.py`、artifact lifecycle 或 import boundary 時，另執行：

```bash
uv run pytest -q tests/test_workflow_boundaries.py
uv run python -m compileall -q llm_bias/jspace_intervention
```
