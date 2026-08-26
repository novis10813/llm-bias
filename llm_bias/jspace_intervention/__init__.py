"""Causal J-space token steering and sector-coordinate replacement.

Method provenance
-----------------
The token-direction construction, additive token steering operator, and
least-squares J-lens coordinate swap follow Gurnee et al. (2026),
https://transformer-circuits.pub/2026/workspace/. Sector prototypes,
discovery-only contrastive concept selection, ticker-grouped splits,
clean-pass loaded-evidence position selection, financial continuation-margin
outcomes, and band-wide dosing are project-specific adaptations. These
interventions provide causal evidence only for the preregistered task-local
outcomes; they do not establish a global workspace claim.
"""

from .concepts import concept_coordinate, sector_prototype, token_direction
from .positions import LoadedPositions, select_loaded_positions
from .transforms import coordinate_intervention, coordinate_swap, steer_positions

__all__ = [
    "LoadedPositions",
    "concept_coordinate",
    "coordinate_intervention",
    "coordinate_swap",
    "sector_prototype",
    "select_loaded_positions",
    "steer_positions",
    "token_direction",
]
