"""Describe the repository's independent command-line workflows."""

from __future__ import annotations


def main() -> None:
    print(
        "Choose one independent workflow:\n"
        "  jacobian-lens           Fit or install a reusable Jacobian lens\n"
        "  counterfactual-patching Run residual activation patch experiments\n"
        "  prompt-analysis         Run prompt readout and attribution experiments\n"
        "  prepare-edgar-8k        Clean extracted 8-K filings into staging data\n"
        "  prepare-counterfactual-data Build reviewed entity-only bias pairs\n"
        "  prepare-10k-change-data Build and run 10-K metadata-change experiments"
    )


if __name__ == "__main__":
    main()
