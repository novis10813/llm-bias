"""Selective-intervention V1: L15 k=8 subspace removal (proposal-v1.md Rev 1).

Inference-time removal of the entity-difference subspace component from
the L15 post-block residual at instruction-span positions, with strength
(dose) sweep, centering/layer/position/random-subspace controls, and
pre-registered efficacy / specificity / task-preservation gates.
"""

__all__ = [
    "analysis",
    "m6_analysis",
    "m6_manifest",
    "m6_pipeline",
    "pipeline",
    "scoring",
    "spans",
    "subspace",
    "template",
]
