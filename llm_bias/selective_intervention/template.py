"""Frozen selective-intervention V1 constants.

All frozen values (arm grid, centers, gates, smoke grid) are defined in
docs/selective-intervention/proposal-v1.md (Rev 1, frozen 2026-09-13);
changes require a new protocol version, not an edit here.
"""
from __future__ import annotations

SCHEMA_VERSION = "selective-intervention-v1"
DATASET = "selective-intervention"
PROTOCOL = "docs/selective-intervention/proposal-v1.md"
PROTOCOL_REV = 1

DECISION_PREFIX = '{"decision": "'
POSITIVE_CANDIDATE = "buy"
NEGATIVE_CANDIDATE = "sell"

# Frozen population (balanced-evidence-gap Phase 2A test split).
N_COMPANIES = 16
N_2A_VARIANTS = 4
N_PROMPTS = N_COMPANIES * N_2A_VARIANTS  # 64
TOP_GROUP: tuple[str, ...] = ("NSC", "BLK")
BOTTOM_GROUP: tuple[str, ...] = ("IT", "BDX")

# Entity-difference subspace (entity-to-dial e-01 artifact).
E01_PCA_DIM = 16          # persisted basis width (top-16 right singular vectors)
K_PRIMARY = 8             # subspace removal dimension (frozen k=8)

# Intervention grid (proposal §3).
INTERVENTION_LAYER = 15            # L15 post-block (primary)
CONTROL_LAYERS: tuple[int, ...] = (13, 14, 16, 17)
ALPHA_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
POSITION_SCOPES: tuple[str, ...] = ("instruction", "full")
CENTERINGS: tuple[str, ...] = ("cloud", "zero", "anon")
RANDOM_SEED = 20260913

# Gates (proposal §7, pre-registered, frozen).
GATE_GAP_FRACTION = 0.5             # G1a/G1b: |G_int| / |G_clean|, S_int / S_clean
GATE_SPECIFICITY_FRACTION = 0.25    # G2: random-arm reduction / main-arm reduction
GATE_MEAN_SHIFT = 0.15              # G3: |mean margin shift|, nats
GATE_ANON_SHIFT = 0.10              # G4: |anonymous margin shift|, nats
# Fail-closed guard: clean group gap must be meaningfully nonzero.
GATE_MIN_CLEAN_GAP = 0.05

# Dial probe (investment-dial coordinate, native intermediate units).
DIAL_LAYER = 15
DIAL_NEURON = 8490
DIAL_PROBE_DELTAS: tuple[float, ...] = (4.0, -4.0)
# Smoke sanity bound for probe margins (nats).
MARGIN_BOUND = 10.0

# No-op / bit-exact conventions (repo-wide).
NOOP_TOLERANCE = 1e-12
BITEXACT_TOLERANCE = 0.0

# Entity header line layout + evidence markers (balanced-evidence-gap
# Phase 2 template; the stored 2A prompt text is never re-derived).
TICKER_LINE_PREFIX = "Stock Ticker: ["
NAME_LINE_PREFIX = "Stock Name: ["
ANON_TICKER = "TICKER"
ANON_NAME = "Company X"
EVIDENCE_MARKER = "— Evidence —"
EVIDENCE_CLOSE = "\n\n—\n\n"

# Smoke grid (proposal §10; mechanism validation, no gate). The four frozen
# group companies so group-gap descriptive statistics remain evaluable.
SMOKE_COMPANIES: tuple[str, ...] = ("NSC", "BLK", "IT", "BDX")
SMOKE_ALPHA_GRID: tuple[float, ...] = (0.5, 1.0)
SMOKE_CONTROL_LAYERS: tuple[int, ...] = (16,)

# Upstream run locations (defaults; overridable via operator flags).
DEFAULT_PHASE2A_RUN = (
    "artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01"
)
DEFAULT_E01_RUN = "artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01"
