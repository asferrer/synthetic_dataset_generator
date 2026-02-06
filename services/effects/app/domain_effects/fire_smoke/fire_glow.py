"""
Fire Glow Effect

Adds realistic glow/illumination around fire sources.
Creates the characteristic orange/red glow that fires cast on surroundings.
"""

import numpy as np
import cv2
import time
from typing import Optional, Dict, Any, Tuple

from ..base_effect import BaseEffect, EffectResult


class FireGlowEffect(BaseEffect):
    """
    Adds fire glow/illumination to images.

    Creates a warm glow effect around fire sources that
    illuminates nearby objects with fire-like lighting.
    """

    effect_id = "fire_glow"
    display_name = "Fire Glow"
    description = "Adds warm glow illumination from fire sources"
    domains = ["fire_smoke"]
    is_universal = False

    default_params = {
        "intensity": 0.6,
        "radius": 1.5,  # Multiplier for glow spread
        "color_rgb": [255, 100, 0],  # Orange-red fire glow
        "falloff": "exponential",  # linear, exponential, or inverse_square
        "flicker": False,  # Add randomized flicker
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
        Apply fire glow effect.

        Args:
            image: Input image (BGR)
            mask: Object mask (optional)
            depth_map: Depth map (optional)
            fire_mask: Mask of fire/flame areas
            **kwargs: Additional parameters

        Returns:
            EffectResult with fire glow applied
        """
        start_time = time.time()
        params = self.validate_params(kwargs)

        try:
            result = self._apply_fire_glow(
                image,
                fire_mask=fire_mask,
                intensity=params["intensity"],
                radius=params["radius"],
                color_rgb=params["color_rgb"],
                falloff=params["falloff"],
                flicker=params["flicker"],
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

    def _apply_fire_glow(
        self,
        image: np.ndarray,
        fire_mask: Optional[np.ndarray],
        intensity: float,
        radius: float,
        color_rgb: Tuple[int, int, int],
        falloff: str,
        flicker: bool,
    ) -> np.ndarray:
        """Apply the fire glow effect."""
        h, w = image.shape[:2]

        if fire_mask is None:
            # If no fire mask, create one from bright orange/red areas
            fire_mask = self._detect_fire_areas(image)

        if fire_mask.sum() == 0:
            return image  # No fire detected

        # Ensure mask is float
        fire_mask = fire_mask.astype(np.float32)
        if fire_mask.max() > 1:
            fire_mask = fire_mask / 255.0

        # Calculate glow spread radius
        kernel_size = int(min(h, w) * 0.1 * radius)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(5, kernel_size)

        # Create glow map using distance transform and blur
        glow_map = cv2.GaussianBlur(fire_mask, (kernel_size, kernel_size), 0)

        # Apply falloff function
        if falloff == "exponential":
            glow_map = np.power(glow_map, 0.5)
        elif falloff == "inverse_square":
            glow_map = glow_map / (1 + glow_map * 2)
        # linear: no transformation

        # Add flicker variation
        if flicker:
            flicker_factor = 0.8 + 0.4 * np.random.random()
            glow_map = glow_map * flicker_factor

        # Apply intensity
        glow_map = glow_map * intensity

        # Create colored glow
        glow_color = np.array(color_rgb[::-1], dtype=np.float32)  # RGB to BGR

        # Apply glow as additive light
        glow_map_3ch = glow_map[:, :, np.newaxis]

        # Add glow (additive blending for light effect)
        result = image.astype(np.float32) + glow_color * glow_map_3ch * 0.5

        # Also warm up the existing colors in the glow zone
        warmth_blend = image.astype(np.float32) * (1 - glow_map_3ch * 0.3) + \
                       self._apply_color_temperature_shift(image, 2500) * glow_map_3ch * 0.3

        result = result * 0.7 + warmth_blend * 0.3

        return np.clip(result, 0, 255).astype(np.uint8)

    def _detect_fire_areas(self, image: np.ndarray) -> np.ndarray:
        """Detect fire/flame areas in image based on color."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Fire is typically in the red-orange-yellow range
        # Red-orange (0-30 hue, high saturation, high value)
        lower_fire1 = np.array([0, 100, 150])
        upper_fire1 = np.array([30, 255, 255])

        # Yellow (15-35 hue)
        lower_fire2 = np.array([15, 100, 200])
        upper_fire2 = np.array([35, 255, 255])

        # Wrap-around red (170-180 hue)
        lower_fire3 = np.array([160, 100, 150])
        upper_fire3 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
        mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        mask3 = cv2.inRange(hsv, lower_fire3, upper_fire3)

        fire_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

        return fire_mask.astype(np.float32) / 255.0

    def _apply_color_temperature_shift(
        self,
        image: np.ndarray,
        temperature_kelvin: int
    ) -> np.ndarray:
        """Shift image color temperature (warm = lower K, cool = higher K)."""
        # Simplified color temperature adjustment
        # Lower temperature = warmer (more red/orange)
        if temperature_kelvin < 5000:
            # Warm shift
            shift = (5000 - temperature_kelvin) / 5000
            r_mult = 1.0 + shift * 0.3
            g_mult = 1.0 - shift * 0.05
            b_mult = 1.0 - shift * 0.2
        else:
            # Cool shift (not used for fire glow)
            shift = (temperature_kelvin - 5000) / 5000
            r_mult = 1.0 - shift * 0.1
            g_mult = 1.0
            b_mult = 1.0 + shift * 0.2

        result = image.astype(np.float32)
        result[:, :, 2] = result[:, :, 2] * r_mult  # R
        result[:, :, 1] = result[:, :, 1] * g_mult  # G
        result[:, :, 0] = result[:, :, 0] * b_mult  # B

        return np.clip(result, 0, 255)
