"""
Domain Effects System

Provides domain-specific visual effects for the synthetic dataset generator.
Effects are organized by domain (underwater, fire_smoke, aerial, etc.)
and can be dynamically loaded based on the active domain.
"""

from .effect_registry import EffectRegistry, get_effect_registry
from .base_effect import BaseEffect, EffectResult

__all__ = [
    'EffectRegistry',
    'get_effect_registry',
    'BaseEffect',
    'EffectResult',
]
