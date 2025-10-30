"""Subpaquete que contiene scripts de entrenamiento de AuraLLM."""

from .train_aura import (
    Config,
    CUSTOM_MODEL_PRESETS,
    GPT2,
    SpanishCorpus,
)

__all__ = [
    "Config",
    "CUSTOM_MODEL_PRESETS",
    "GPT2",
    "SpanishCorpus",
]