"""
Heat Distortion Effect

Simulates the visual distortion caused by rising heat from fires.
Creates a wavy/rippling effect that is stronger near heat sources.
"""

import numpy as np
import cv2
import time
from typing import Optional, Dict, Any

from ..base_effect import BaseEffect, EffectResult


class HeatDistortionEffect(BaseEffect):
    """
    Simulates heat distortion (heat shimmer) near fire sources.

    Creates a rippling/wavy effect that simulates the refraction
    of light through rising hot air.
    """

    effect_id = "heat_distortion"
    display_name = "Heat Distortion"
    description = "Simulates visual distortion from rising heat"
    domains = ["fire_smoke"]
    is_universal = False

    default_params = {
        "intensity": 0.15,
        "wave_frequency": 0.02,
        "wave_speed": 1.0,
        "apply_near_fire": True,
        "falloff_distance": 100,  # pixels
    }

    def apply(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        fire_mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> EffectResult:
        """
        Apply heat distortion effect.

        Args:
            image: Input image (BGR)
            mask: Object mask (optional)
            depth_map: Depth map (optional)
            fire_mask: Mask of fire/heat source areas (optional)
            **kwargs: Additional parameters

        Returns:
            EffectResult with distorted image
        """
        start_time = time.time()
        params = self.validate_params(kwargs)

        try:
            result = self._apply_heat_distortion(
                image,
                fire_mask=fire_mask,
                intensity=params["intensity"],
                wave_frequency=params["wave_frequency"],
                falloff_distance=params["falloff_distance"],
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

    def _apply_heat_distortion(
        self,
        image: np.ndarray,
        fire_mask: Optional[np.ndarray],
        intensity: float,
        wave_frequency: float,
        falloff_distance: int,
    ) -> np.ndarray:
        """Apply the heat distortion effect."""
        h, w = image.shape[:2]

        # Create displacement maps
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

        # Generate wave patterns
        # Horizontal waves (heat rises, so mainly vertical displacement)
        time_offset = np.random.random() * 2 * np.pi  # Random phase
        wave_y = np.sin(y_coords * wave_frequency * np.pi * 2 + time_offset)
        wave_x = np.sin(x_coords * wave_frequency * np.pi * 3 + time_offset * 0.7)

        # Create intensity mask based on fire location
        if fire_mask is not None:
            # Dilate fire mask to create heat zone
            kernel = np.ones((falloff_distance, falloff_distance), np.uint8)
            heat_zone = cv2.dilate(fire_mask.astype(np.uint8), kernel, iterations=1)

            # Create distance-based falloff
            dist_transform = cv2.distanceTransform(
                (1 - fire_mask.astype(np.uint8)) * 255,
                cv2.DIST_L2, 5
            )
            # Normalize and invert (closer to fire = stronger effect)
            intensity_map = 1.0 - np.clip(dist_transform / falloff_distance, 0, 1)

            # Heat rises - apply stronger effect above fire
            y_gradient = 1.0 - (y_coords / h)  # Stronger at top
            intensity_map = intensity_map * (0.5 + 0.5 * y_gradient)
        else:
            # Default: apply to upper portion of image (heat rises)
            intensity_map = 1.0 - (y_coords / h)
            intensity_map = np.clip(intensity_map, 0, 0.7)

        # Calculate displacement
        displacement_x = wave_x * intensity * w * 0.01 * intensity_map
        displacement_y = wave_y * intensity * h * 0.02 * intensity_map

        # Create remapping coordinates
        map_x = (x_coords + displacement_x).astype(np.float32)
        map_y = (y_coords + displacement_y).astype(np.float32)

        # Apply remapping
        result = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return result
