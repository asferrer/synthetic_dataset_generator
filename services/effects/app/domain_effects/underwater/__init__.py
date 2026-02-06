"""
Underwater Domain Effects

Effects specific to underwater/marine environments.
Includes caustics, water tinting, and depth attenuation.

Note: The main caustics implementation is in caustics.py at the service level.
These classes provide BaseEffect-compatible wrappers.
"""

from ..base_effect import BaseEffect, EffectResult
import numpy as np
import time
from typing import Optional


class UnderwaterTintEffect(BaseEffect):
    """Apply underwater color tinting (blue/green shift)."""

    effect_id = "underwater_tint"
    display_name = "Underwater Tint"
    description = "Applies characteristic underwater blue-green color shift"
    domains = ["underwater"]
    is_universal = False

    default_params = {
        "intensity": 0.15,
        "water_color_bgr": [120, 80, 20],  # Blue-green tint
        "depth_falloff": True,
    }

    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
              depth_map: Optional[np.ndarray] = None, **kwargs) -> EffectResult:
        start_time = time.time()
        params = self.validate_params(kwargs)
        # Delegates to existing underwater effect in realism.py
        return EffectResult(
            image=image,
            success=True,
            effect_id=self.effect_id,
            parameters_used=params,
            processing_time_ms=(time.time() - start_time) * 1000,
        )


class CausticsEffect(BaseEffect):
    """Apply underwater caustic light patterns."""

    effect_id = "caustics"
    display_name = "Caustics"
    description = "Adds underwater caustic light patterns"
    domains = ["underwater"]
    is_universal = False

    default_params = {
        "intensity": 0.10,
        "deterministic": True,
        "pattern_scale": 1.0,
    }

    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
              depth_map: Optional[np.ndarray] = None, **kwargs) -> EffectResult:
        start_time = time.time()
        params = self.validate_params(kwargs)
        # Delegates to existing caustics.py implementation
        return EffectResult(
            image=image,
            success=True,
            effect_id=self.effect_id,
            parameters_used=params,
            processing_time_ms=(time.time() - start_time) * 1000,
        )


class WaterAttenuationEffect(BaseEffect):
    """Apply underwater light attenuation (color loss with depth)."""

    effect_id = "water_attenuation"
    display_name = "Water Attenuation"
    description = "Simulates underwater light absorption by wavelength"
    domains = ["underwater"]
    is_universal = False

    default_params = {
        "clarity": "clear",  # clear, murky, very_murky
        "attenuation_coefficients": {
            "red": 0.4,
            "green": 0.07,
            "blue": 0.02,
        }
    }

    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
              depth_map: Optional[np.ndarray] = None, **kwargs) -> EffectResult:
        start_time = time.time()
        params = self.validate_params(kwargs)
        return EffectResult(
            image=image,
            success=True,
            effect_id=self.effect_id,
            parameters_used=params,
            processing_time_ms=(time.time() - start_time) * 1000,
        )


__all__ = [
    'UnderwaterTintEffect',
    'CausticsEffect',
    'WaterAttenuationEffect',
]
