"""
Aerial Domain Effects

Effects specific to aerial/sky environments (bird detection, wind farms).
Includes atmospheric haze and depth-based fog.
"""

from ..base_effect import BaseEffect, EffectResult
import numpy as np
import cv2
import time
from typing import Optional, Dict, Any, Tuple


class AtmosphericHazeEffect(BaseEffect):
    """Apply atmospheric haze effect for distant objects."""

    effect_id = "atmospheric_haze"
    display_name = "Atmospheric Haze"
    description = "Adds atmospheric haze that increases with distance"
    domains = ["aerial_birds"]
    is_universal = False

    default_params = {
        "intensity": 0.1,
        "color_rgb": [200, 210, 220],  # Light blue-gray
        "depth_based": True,
    }

    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
              depth_map: Optional[np.ndarray] = None, **kwargs) -> EffectResult:
        start_time = time.time()
        params = self.validate_params(kwargs)

        try:
            result = self._apply_haze(
                image,
                depth_map=depth_map,
                intensity=params["intensity"],
                color_rgb=params["color_rgb"],
                depth_based=params["depth_based"],
            )

            return EffectResult(
                image=result,
                success=True,
                effect_id=self.effect_id,
                parameters_used=params,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return EffectResult(
                image=image,
                success=False,
                effect_id=self.effect_id,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def _apply_haze(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        intensity: float,
        color_rgb: Tuple[int, int, int],
        depth_based: bool,
    ) -> np.ndarray:
        """Apply atmospheric haze effect."""
        h, w = image.shape[:2]

        # Create haze color layer
        haze_color = np.full((h, w, 3), color_rgb[::-1], dtype=np.float32)

        # Create intensity map
        if depth_based and depth_map is not None:
            depth_normalized = depth_map.astype(np.float32)
            if depth_normalized.max() > 1:
                depth_normalized = depth_normalized / 255.0
            haze_intensity = depth_normalized * intensity
        else:
            # Default: use vertical gradient (more haze at horizon)
            y_gradient = np.abs((np.arange(h) / h) - 0.5) * 2  # 0 at center, 1 at edges
            y_gradient = 1 - y_gradient  # Invert: max at center (horizon)
            haze_intensity = np.broadcast_to(y_gradient[:, np.newaxis], (h, w)) * intensity

        haze_intensity_3ch = haze_intensity[:, :, np.newaxis]

        # Blend with haze
        result = image.astype(np.float32) * (1 - haze_intensity_3ch) + \
                 haze_color * haze_intensity_3ch

        return np.clip(result, 0, 255).astype(np.uint8)


class DepthFogEffect(BaseEffect):
    """Apply depth-based fog effect."""

    effect_id = "depth_fog"
    display_name = "Depth Fog"
    description = "Adds fog that increases with distance from camera"
    domains = ["aerial_birds"]
    is_universal = False

    default_params = {
        "start_depth": 0.3,  # Depth at which fog starts
        "intensity": 0.2,
        "color_rgb": [220, 225, 230],  # Light gray
    }

    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
              depth_map: Optional[np.ndarray] = None, **kwargs) -> EffectResult:
        start_time = time.time()
        params = self.validate_params(kwargs)

        try:
            result = self._apply_depth_fog(
                image,
                depth_map=depth_map,
                start_depth=params["start_depth"],
                intensity=params["intensity"],
                color_rgb=params["color_rgb"],
            )

            return EffectResult(
                image=result,
                success=True,
                effect_id=self.effect_id,
                parameters_used=params,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return EffectResult(
                image=image,
                success=False,
                effect_id=self.effect_id,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    def _apply_depth_fog(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        start_depth: float,
        intensity: float,
        color_rgb: Tuple[int, int, int],
    ) -> np.ndarray:
        """Apply depth-based fog effect."""
        h, w = image.shape[:2]

        # Create fog color layer
        fog_color = np.full((h, w, 3), color_rgb[::-1], dtype=np.float32)

        if depth_map is not None:
            depth_normalized = depth_map.astype(np.float32)
            if depth_normalized.max() > 1:
                depth_normalized = depth_normalized / 255.0

            # Apply fog only beyond start_depth
            fog_mask = np.clip((depth_normalized - start_depth) / (1 - start_depth), 0, 1)
            fog_intensity = fog_mask * intensity
        else:
            # Fallback: vertical gradient
            y_gradient = np.arange(h) / h
            fog_intensity = np.broadcast_to(y_gradient[:, np.newaxis], (h, w)) * intensity

        fog_intensity_3ch = fog_intensity[:, :, np.newaxis]

        # Blend with fog
        result = image.astype(np.float32) * (1 - fog_intensity_3ch) + \
                 fog_color * fog_intensity_3ch

        return np.clip(result, 0, 255).astype(np.uint8)


__all__ = [
    'AtmosphericHazeEffect',
    'DepthFogEffect',
]
