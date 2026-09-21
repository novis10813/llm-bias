# Active Research Boundary Cleanup Implementation Spec

**Goal:** move the completed, independent financial-soundness exploration out of the active execution tree, while making active documentation distinguish core evidence, boundary evidence, ongoing exploration, and shared infrastructure.

**Architecture & Tech Stack:** this is a Git move-and-navigation change in the existing Python/Markdown repository. The frozen financial-soundness package continues to rely on active `llm_bias.core` only when restored. Its original package path and artifact schemas are preserved inside the frozen source; the active `financial-soundness` console entry point is removed. Archive contents are not included in root pytest collection.

**Global Constraints:** make every change only in `.worktrees/research-axis-cleanup`; do not modify `main`, artifacts, model caches, numerical results, frozen protocol wording, or raw activation policy. Use `git mv` for tracked moves. Do not add dependencies or refactor active package code. Relative Markdown links must resolve from their new source locations. The existing three modified files (`README.md`, `docs/proposal/entity-bias-research-proposal.md`, and `docs/proposal/entity-bias-roadmap.md`) are in-scope only where this spec explicitly directs navigation updates.

## 1. Scope and Non-Goals

### In scope

1. Freeze the financial-soundness localization and causal-validation workflow: Python package, its deterministic test, tracked prompt inputs, and its canonical documentation.
2. Remove the active CLI and active-package declarations that would otherwise advertise that frozen workflow as runnable.
3. Add an archive inventory/restore contract for the newly frozen line.
4. Reclassify the active research navigation without suppressing core null results or boundary evidence.
5. Update the root research map from 18 to 17 active research items after the financial-soundness pair moves to the archive, adding Evidence-insensitivity so the map matches the retained core-evidence category.

### Strict non-goals

- Do not archive, rename, or relocate `balanced_evidence_gap`, `entity_to_dial`, `entity_cell`, `investment_dial`, `entity_concept_decision`, `evidence_insensitivity`, `selective_intervention`, `jspace_intervention`, or `span_sensitivity`.
- Do not move `core`, `baseline_trial`, `prompt_analysis`, `lens_fitting`, `lens_install`, or `lens_cli.py`; do not perform the known `baseline_trial`/`prompt_analysis` shared-core refactor.
- Do not change research findings, gates, run IDs, values, artifact contents, or historical protocol text. Changes inside moved Markdown documents are limited to repairing relative links caused by their move.
- Do not invent a new active source-package hierarchy or change Python import paths for retained packages.
- Do not move any file from `scripts/`: the dependency audit found no financial-soundness script there.
- Do not add archive packages to `pyproject.toml`, root test collection, or the active workflow list.

## 2. Requirements and Contracts

### REQ-1: Freeze the whole financial-soundness execution unit

Move exactly these tracked resources with Git rename semantics:

- `llm_bias/financial_soundness/` → `archive/llm_bias/financial_soundness/`
- `tests/test_financial_soundness.py` → `archive/tests/test_financial_soundness.py`
- `data/baseline/financial-soundness-localization/` → `archive/data/baseline/financial-soundness-localization/`
- `docs/financial-soundness-localization/` → `docs/archive/financial-soundness-localization/`
- `docs/financial-soundness-causal-validation/` → `docs/archive/financial-soundness-causal-validation/`

The archived package must retain its original module name (`llm_bias.financial_soundness`) and source provenance strings. It is a restoration contract, not an importable active module. The two prompt JSON files move with the workflow because they are tracked, workflow-specific inputs; no ignored artifact is moved.

### REQ-2: Remove all active execution entry points for the frozen unit

- Remove the `financial-soundness` project script from `pyproject.toml`.
- Remove its active-package ownership/CLI statements from `llm_bias/AGENTS.md`.
- Remove its root-test listing from `tests/AGENTS.md`.
- Do not add an archive console entry point. Archive documentation must state that restoration requires moving the package, test, data, and its two canonical document directories back before re-adding the same project-script entry.

The active source tree must contain no `llm_bias/financial_soundness` directory, and root pytest must not collect `test_financial_soundness.py`.

### REQ-3: Preserve archive documentation and repair only move-induced links

- Extend `archive/README.md` with one inventory row for the package, test/data/document locations, its exploratory/no-certification status, active-core dependency, and a restore step consistent with its existing restoration convention. Correct its scope statement so archive is not falsely described as containing only workflows unrelated to `data/baseline/`.
- Update `archive/AGENTS.md` only to extend its frozen-workflow scope statement to include financial-soundness; do not change archive rules.
- Keep the two canonical directories under `docs/archive/` and preserve their report/proposal/detail file names and historical content.
- Repair only their relative links that break because `docs/archive/` is one level deeper. In particular, links to active documentation such as `documentation-system.md` must resolve from their new locations.
- Replace active inbound navigation links with archive links where they formerly represented the frozen workflow. The Phase 3 reference in `docs/balanced-evidence-gap/details/proposal-phase3.md` is historical method provenance: retain the sentence and point it to the archived causal-validation proposal.

### REQ-4: Active navigation must be organized by evidence role, not directory flatness

`docs/README.md` and root `README.md` must make these four categories explicit:

1. **Core evidence for the current paper narrative:** Entity Cell, Balanced Evidence Gap, Entity-to-Dial, and Evidence-insensitivity. Their positive and null findings remain directly linked.
2. **Boundary/control evidence:** Investment-dial, Entity concept decision, and Selective-intervention. They are retained because they constrain the interpretation of single-neuron, entity-concept, and intervention claims.
3. **Active discovery and localisation controls:** span sensitivity plus the J-space sector/valence/token/activation-patching/context lines, and any still-proposed active evaluation. Their exploratory/failed status remains stated.
4. **Shared inputs and instruments:** baseline trial and Jacobian-lens selection; these are not presented as independent entity-bias evidence.

The frozen financial-soundness line is absent from all active category tables, active relationship edges, and active map nodes. Each active overview has a concise archive link that names it as a frozen exploratory financial-judgment workflow, without relitigating its result.

### REQ-5: Keep active research-map semantics consistent

Update the existing root README Mermaid map and its immediately preceding caption so that it describes active work after REQ-1:

- exactly 17 experiment nodes; remove `fsloc` and `fscv`, and add `ei["Evidence-insensitivity<br/>(completed: no generation flips)"]:::neutral` to the entity-decision subgraph;
- exactly 16 major edges; remove `ecell -.-> fsloc` and `fsloc --> fscv`, then add `beg -.-> ei` as the existing Balanced Evidence Gap → Evidence-insensitivity research-succession relation recorded in `docs/README.md`;
- retain the 3 subgraphs, classes, no `click`, no `linkStyle`, no inner `direction LR`, and all other retained edges unchanged;
- replace its assertion that archive lines are merely excluded with wording that includes the frozen financial-soundness exploration among archive material.

Do not alter labels or verdicts for remaining nodes. `docs/dev/research-map_spec.md` records the original completed 18-node implementation and is historical development evidence; do not rewrite it in this cleanup.

### REQ-6: Preserve active package isolation and no stale executable reference

After the move:

- `rg` over active `llm_bias/`, `tests/`, `scripts/`, and `pyproject.toml` must find no executable import or CLI reference to `financial_soundness`/`financial-soundness`.
- The existing tests that name `llm_bias.financial_soundness` in forbidden-import lists remain valid: they assert isolation rather than require the package to exist.
- Active documentation may mention the frozen workflow only as a direct link into `docs/archive/`; the Balanced Evidence Gap Phase 3 proposal may additionally retain its historical method-provenance sentence with that archive link. No active document may advertise `financial-soundness` as an available command or as an active experiment.

## 3. Implementation Slices

### Slice 1: Freeze the financial-soundness workflow

- **Requirements covered:** REQ-1, REQ-2, REQ-3, REQ-6.
- **Files:**
  - Move: the five exact source/test/data/document path groups listed in REQ-1.
  - Modify: `pyproject.toml`, `llm_bias/AGENTS.md`, `tests/AGENTS.md`, `archive/README.md`, `archive/AGENTS.md`, `docs/balanced-evidence-gap/details/proposal-phase3.md`.
  - Do not modify: any retained active experiment package, `README.md`, `docs/README.md`, or `docs/proposal/` in this slice.
- **Detailed steps:**
  1. `git mv` the package, test, tracked input directory, and two documentation directories to their REQ-1 destinations.
  2. Repair only links broken by the two documentation moves; update the one active Phase 3 historical method link to the archived proposal.
  3. Remove the active console script and active ownership/test-index references.
  4. Add the archive inventory row and restore instructions. State the package uses active `llm_bias.core`, archived tests do not run in the root suite, and restoration includes re-adding the project script.
  5. Verify no active source/test/script/pyproject execution reference remains.
- **Acceptance commands:**
  - `git diff --check`
  - `test ! -e llm_bias/financial_soundness && test -d archive/llm_bias/financial_soundness && test -f archive/tests/test_financial_soundness.py && test -d archive/data/baseline/financial-soundness-localization`
  - `! rg -n "financial_soundness|financial-soundness" pyproject.toml llm_bias scripts`
  - `! rg -n "^(from|import) llm_bias\.financial_soundness" tests`
  - `PATH=/mnt/train-data-1-hdd/sam/llm-bias/.venv/bin:$PATH /mnt/train-data-1-hdd/sam/llm-bias/.venv/bin/pytest -q tests/test_workflow_boundaries.py`
  - `PATH=/mnt/train-data-1-hdd/sam/llm-bias/.venv/bin:$PATH /mnt/train-data-1-hdd/sam/llm-bias/.venv/bin/pytest -q`
  - `uv lock --check` and `uv build` when the worktree dependencies are restored; if the worktree cannot resolve ignored workspace members, report that environmental blocker separately rather than editing `pyproject.toml`.

### Slice 2: Reorganize active research navigation

- **Requirements covered:** REQ-4, REQ-5, REQ-6.
- **Files:**
  - Modify: `README.md`, `docs/README.md`, `docs/proposal/README.md`, `docs/proposal/entity-bias-research-proposal.md`, `docs/proposal/entity-bias-roadmap.md`.
  - Do not modify: `docs/dev/research-map_spec.md`; experiment reports/proposals other than the Phase 3 link already handled in Slice 1; Python, tests, scripts, data, artifacts, or archive contents.
- **Detailed steps:**
  1. Replace the flat active experiment presentation in `docs/README.md` with the four REQ-4 headings. Preserve each retained experiment’s current status, one-sentence finding, and canonical report/proposal links; do not promote findings or erase null results.
  2. Add one compact frozen-workflow pointer in `docs/README.md`, `docs/proposal/README.md`, the proposal, and the roadmap where financial-soundness was formerly listed as active evidence. It must link into `docs/archive/` and distinguish it from the retained core/boundary lines.
  3. Remove the two financial nodes/edges from the Mermaid map, adjust its caption as REQ-5 requires, and leave every remaining node/edge/style intact.
  4. Confirm all active documents comply with REQ-6 and all relative links resolve.
- **Acceptance commands:**
  - Mermaid parser check against `README.md`, expecting `MERMAID OK flowchart-v2`.
  - Node count: `awk '/^```mermaid/{f=1;next} /^```/{if(f)exit} f' README.md | grep -cE '^\s+[a-z0-9]+\["'` prints `17`.
  - Edge count: `awk '/^```mermaid/{f=1;next} /^```/{if(f)exit} f' README.md | grep -cE '^\s+[a-z0-9]+ (==>|-->|-\.->)'` prints `16`.
  - `! rg -n "financial-soundness =|financial-soundness (run-|prepare-|--help)" pyproject.toml llm_bias tests scripts README.md docs/README.md docs/proposal`
  - `! rg -n "llm_bias\.financial_soundness" pyproject.toml llm_bias scripts README.md docs/README.md docs/proposal`
  - `rg -n "docs/archive/financial-soundness|archive/financial-soundness" README.md docs/README.md docs/proposal/README.md docs/proposal/entity-bias-research-proposal.md docs/proposal/entity-bias-roadmap.md`
  - `git diff --check`
  - `PATH=/mnt/train-data-1-hdd/sam/llm-bias/.venv/bin:$PATH /mnt/train-data-1-hdd/sam/llm-bias/.venv/bin/pytest -q tests/test_workflow_boundaries.py`

## 4. Full Verification and Acceptance Criteria

1. The original repository checkout remains clean; only `.worktrees/research-axis-cleanup` changes.
2. Git detects moves for the financial package, its test, tracked inputs, and canonical documentation directories; no ignored artifact/model/cache is staged.
3. Root `pytest` passes without archived tests; active `financial-soundness` command and package path are absent.
4. Archive documentation gives an executable restore sequence and correctly states active-core dependency.
5. Root research navigation has 17 map nodes and the four evidence-role categories. It continues to link all retained core and boundary evidence, including negative results.
6. Markdown links changed by the move resolve, `git diff --check` is clean, and Mermaid parses as `flowchart-v2`.
7. Run `uv lock --check`, `uv run pytest -q`, `uv run python -m compileall -q llm_bias`, and `uv build` in a worktree with the ignored workspace members available. If this worktree cannot install because `third_party/` is absent, state that as environment setup rather than changing dependency declarations.
