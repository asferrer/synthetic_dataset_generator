"""
Fire & Smoke Domain Effects

Visual effects optimized for fire and smoke detection datasets.
"""

from .heat_distortion import HeatDistortionEffect
from .smoke_haze import SmokeHazeEffect
from .fire_glow import FireGlowEffect
from .ember_glow import EmberGlowEffect

__all__ = [
    'HeatDistortionEffect',
    'SmokeHazeEffect',
    'FireGlowEffect',
    'EmberGlowEffect',
]
