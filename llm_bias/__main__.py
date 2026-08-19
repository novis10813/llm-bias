"""Describe the repository's independent command-line workflows."""

from __future__ import annotations


def main() -> None:
    print(
        "Choose one independent workflow:\n"
        "  jacobian-lens    Fit or install a reusable Jacobian lens\n"
        "  prompt-analysis  Run prompt readout and attribution experiments\n"
        "  baseline-trial   Run baseline trial-plan prompts through prompt-analysis stages"
    )


if __name__ == "__main__":
    main()
