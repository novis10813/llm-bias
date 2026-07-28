"""Describe the repository's independent command-line workflows."""

from __future__ import annotations


def main() -> None:
    print(
        "Choose one independent workflow:\n"
        "  fit-jacobian-lens       Fit a reusable Jacobian-lens artifact\n"
        "  counterfactual-patching Run residual activation patch experiments\n"
        "  prompt-analysis         Run prompt readout and attribution experiments"
    )


if __name__ == "__main__":
    main()
