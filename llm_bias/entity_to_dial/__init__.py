"""Entity-to-dial path dissection experiment package.

Protocol: docs/entity-to-dial/details/proposal-phase-abc.md (Rev 1). Three phases:
A = token-group x layer sufficiency map, B = handoff-interval
block-level (MLP vs attention) patch, C = dial-path probe.
This package does not import other experiment packages; shared
mechanics come from llm_bias.core.
"""
