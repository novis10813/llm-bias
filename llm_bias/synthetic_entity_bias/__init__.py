"""Synthetic entity-bias pilot."""
from .spec import LABELS, SCORES, TEMPLATES, SCORING_INSTRUCTION
from .entities import EntityRecord, load_entity_pool

__all__ = ["LABELS", "SCORES", "TEMPLATES", "SCORING_INSTRUCTION", "EntityRecord", "load_entity_pool"]
