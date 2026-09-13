"""Frozen entity-to-dial constants.

All frozen values (dial coordinate, margin groups, thresholds, smoke
grid) are defined in docs/entity-to-dial/proposal.md (Rev 1); changes
require a new protocol version, not an edit here.
"""
from __future__ import annotations

SCHEMA_VERSION = "entity-to-dial-v1"
DATASET = "entity-to-dial"
PROTOCOL = "docs/entity-to-dial/proposal.md"

# Investment-dial coordinate (L15/n8490, native intermediate units).
DIAL_LAYER = 15
DIAL_NEURON = 8490

DECISION_PREFIX = '{"decision": "'
POSITIVE_CANDIDATE = "buy"
NEGATIVE_CANDIDATE = "sell"

# Frozen pure-entity-margin groups (phase2a-rev2-gate-01, protocol §4.1).
TOP_GROUP: tuple[str, ...] = ("NSC", "BLK")
BOTTOM_GROUP: tuple[str, ...] = ("IT", "BDX")

# Pre-check 1 (§4.1): group gap threshold in nats.
GROUP_GAP_MIN = 0.5
# Pre-check 2 (§4.1): self-source no-op tolerance (bit-exact expected).
NOOP_TOLERANCE = 1e-12
# Phase C 2A cross-check warning threshold (~2x Phase 3 bf16 jitter band).
CROSS_CHECK_WARNING_NATS = 0.1
# Gate C thresholds (§4.4).
C2_RATIO_MIN = 0.25
C1_RHO_WARNING = 0.3
# Bootstrap CI convention (same as balanced-evidence-gap Phase 2).
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42

# Entity header line layout (balanced-evidence-gap Phase 2 template).
TICKER_LINE_PREFIX = "Stock Ticker: ["
NAME_LINE_PREFIX = "Stock Name: ["
ANON_TICKER = "TICKER"
ANON_NAME = "Company X"
# Evidence section markers (balanced-evidence-gap Phase 2 template);
# the instruction region runs after the closing "—" line to end of prompt.
EVIDENCE_MARKER = "— Evidence —"
EVIDENCE_CLOSE = "\n\n—\n\n"

# Phase grids (§4).
PHASE_A_LAYERS: tuple[int, ...] = tuple(range(0, 12))
PHASE_B_LAYERS: tuple[int, ...] = (12, 13, 14, 15)
PHASE_A_EARLY_LAYERS: tuple[int, ...] = tuple(range(0, 6))

# Smoke grid (§14).
SMOKE_DIRECTIONS: tuple[tuple[str, str], ...] = (("NSC", "IT"), ("IT", "NSC"))
SMOKE_A_LAYERS: tuple[int, ...] = (0, 3, 5, 9)
SMOKE_B_LAYERS: tuple[int, ...] = (12, 15)
SMOKE_C_TICKERS: tuple[str, ...] = ("NSC", "IT")

# ── Phase D (proposal-phase-d.md Rev 1) ─────────────────────────────────────────
PROTOCOL_D = "docs/entity-to-dial/proposal-phase-d.md"
PROTOCOL_D_REV = 1

# Phase D grid (§4.2): L12–31 instruction-span block sweep.
PHASE_D_LAYERS: tuple[int, ...] = tuple(range(12, 32))
# Smoke grid (Phase D): linear-attention / full-attention / final layer.
SMOKE_D_LAYERS: tuple[int, ...] = (12, 15, 31)
# D2 smoke company (§4.3 gradient check: 1 company).
SMOKE_D2_TICKER: str = "NSC"
# D2 smoke acceptance (protocol Rev 1.1 §4.3): norm-relative partition
# identity tolerance between the fp32 all-position sum and the core
# all-position-summed derivative (bf16 accumulation; measured 2.1e-3).
D2_PARTITION_TOLERANCE = 0.05
# D2 matched controls (protocol §2: per-layer seed 42 + layer, 10 channels).
D2_CONTROLS_N = 10
D2_CONTROLS_SEED_BASE = 42

# ── Phase E (proposal-phase-e.md Rev 1) ────────────────────────────────────
PROTOCOL_E = "docs/entity-to-dial/proposal-phase-e.md"
PROTOCOL_E_REV = 1

# E1 grid (§4.2): L12–18 joint + full-swap arms.
PHASE_E_LAYERS: tuple[int, ...] = tuple(range(12, 19))
# Smoke grid (protocol §4.1): layers, directions, E2 k-values.
SMOKE_E_LAYERS: tuple[int, ...] = (12, 15)
SMOKE_E_DIRECTIONS: tuple[str, ...] = ("NSC->IT", "IT->NSC")
# Smoke E2 k-values (protocol §4.1 grid: k ∈ {1, 8, full} + dial arm).
SMOKE_E2_K_SWEEP: tuple[int, ...] = (1, 8)
# E2 layer and PCA sweep (protocol §4.3).
E2_LAYER: int = 15
E2_K_SWEEP: tuple[int, ...] = (1, 3, 8, 16)
E2_PCA_DIM: int = 16
# Smoke acceptance (§4.1 #2): in-run full-arm vs 2B archive per-direction
# toward ΔM band (nats); expected 0.0 (bit-exact transform, deterministic
# forward).
E1_SMOKE_FULL_BAND = 0.05
# R1 denominator rule (protocol §2, pre-registered, frozen): ratio medians
# only over directions with |ΔM_full| ≥ this (nats).
RATIO_MIN_FULL_DM = 0.2
# Gates (protocol §3, pre-registered, frozen).
GATE_E1_RATIO_MIN = 0.5
GATE_E2B_RATIO_MIN = 0.5
GATE_E2B_FALSIFIER_MAX = 0.05
# H_E2a descriptive target (protocol §3; not a gate).
E2A_RATIO_TARGET = 0.8

# ── Phase F: dual-path additivity + directional push (proposal-phase-f.md) ──

PROTOCOL_F = "docs/entity-to-dial/proposal-phase-f.md"
PROTOCOL_F_REV = 1

# Intervention layer (Phase E's strongest layer; frozen).
F_LAYER: int = 15
# PCA subspaces for F1 arms (k=1 → v₁, k=8 → saturation reference).
F1_K_VALUES: tuple[int, ...] = (1, 8)
# F2 push doses (protocol §4.3; α=0.5 descriptive dose-response only).
F2_ALPHAS: tuple[float, ...] = (0.5, 1.0, 2.0)
# F2 verdict points: α ∈ {1.0, 2.0} (two per sign) drive the four-way
# classification; α = 0.5 is dose-response descriptive only.
F2_VERDICT_ALPHAS: tuple[float, ...] = (1.0, 2.0)
# F2 jitter band (nats; |ΔM| ≤ band → jitter_band, not decisive).
F2_JITTER_BAND = 0.05
# F2 anonymous clean-margin reference (Phase A/C archive: all 16 companies
# identical at −3.2280; pre-registered band −3.23 ± 0.1, protocol §5 #5).
F2_M_ANON_REF = -3.23
F2_M_ANON_BAND = 0.1
# F2 push sanity bound (|ΔM| ≤ bound, nats; smoke acceptance #6).
F2_PUSH_BOUND = 5.0
# Cross-run consistency band vs e-01 / 2B archive (nats; smoke acceptance
# #3 fail-closed, formal descriptive).
F1_CONSISTENCY_TOLERANCE = 0.01
# Gate F1 (protocol §3, pre-registered, frozen): median additivity ratio
# (combined / in-run full, R1-effective directions) ≥ this.
GATE_F1_RATIO_MIN = 0.85

# Smoke grid (protocol §5; mechanism validation, no gate).
SMOKE_F_DIRECTION: tuple[str, str] = ("BDX", "BLK")  # strong bottom→top
SMOKE_F2_ALPHAS: tuple[float, ...] = (1.0,)

# Upstream run locations (defaults; overridable via operator flags).
DEFAULT_PHASE2A_RUN = (
    "artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-gpu-bf16-01"
)
DEFAULT_PHASE2B_RUN = (
    "artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2b-gpu-bf16-01"
)
DEFAULT_PHASE2A_REV2_RUN = (
    "artifacts/qwen3.5-4b/balanced-evidence-gap-phase2/runs/phase2a-rev2-gate-01"
)
DEFAULT_PHASEE_RUN = (
    "artifacts/qwen3.5-4b/entity-to-dial/runs/entity-to-dial-e-01"
)
