"""
Semantic ID generative recommender models.

This module provides:
- SemanticIDBaseRecommender: Base class for semantic ID recommenders
- SemanticIDEncoderDecoder: TIGER-style encoder-decoder model
- SemanticIDMTPRecommender: MTP-based generative recommender
"""

from src.models.modules.semantic_id.base import SemanticIDBaseRecommender
from src.models.modules.semantic_id.components import DualHashEmbedding, ModelMode
from src.models.modules.semantic_id.mtp_generation_model import SemanticIDMTPRecommender
from src.models.modules.semantic_id.tiger_generation_model import (
    SemanticIDDecoderModule,
    SemanticIDEncoderDecoder,
)

__all__ = [
    # Base class
    "SemanticIDBaseRecommender",
    # Concrete implementations
    "SemanticIDEncoderDecoder",
    "SemanticIDMTPRecommender",
    # Decoder module
    "SemanticIDDecoderModule",
    # Shared components
    "ModelMode",
    "DualHashEmbedding",
]