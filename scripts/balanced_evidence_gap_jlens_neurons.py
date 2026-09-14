# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""J-lens structural readout of the Phase 3 neuron coordinates.

Reads docs/balanced-evidence-gap/details/diagnostic-jlens-neurons.md (frozen
2026-09-10). For each (layer, neuron) coordinate, transports the
down-projection row to the final-layer basis with the canonical Jacobian
lens and decodes it with the model's unembed path. No forwards, no
prompts. Descriptive, non-causal.

Usage:
    uv run --no-sync python scripts/balanced_evidence_gap_jlens_neurons.py \
        --model .cache/models/qwen3.5-4b --run-id phase3-jlens-neurons-01

    --smoke: 4 primary coordinates only (no controls), run id must end in -smoke.
"""
from __future__ import annotations

import argparse

from llm_bias.balanced_evidence_gap.jlens_neurons import run_jlens_neurons_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="primary coordinates only (no controls)")
    args = parser.parse_args()

    if args.smoke:
        if not args.run_id.endswith("-smoke"):
            raise SystemExit("--smoke requires a run id ending in '-smoke'")
        import llm_bias.balanced_evidence_gap.jlens_neurons as m
        saved = m.CONTROL_LAYERS
        m.CONTROL_LAYERS = ()  # primary coordinates only
        try:
            run_directory = m.run_jlens_neurons_diagnostic(
                model_name=args.model,
                run_id=args.run_id,
            )
        finally:
            m.CONTROL_LAYERS = saved
    else:
        run_directory = run_jlens_neurons_diagnostic(
            model_name=args.model,
            run_id=args.run_id,
        )
    print(f"done: {run_directory}")


if __name__ == "__main__":
    main()
