"""
Smoke Haze Effect

Adds atmospheric haze/smoke effect to images.
Simulates reduced visibility due to smoke particles in the air.
"""

import numpy as np
import cv2
import time
from typing import Optional, Dict, Any, Tuple

from ..base_effect import BaseEffect, EffectResult


class SmokeHazeEffect(BaseEffect):
    """
    Adds smoke/haze to images.

    Creates a fog-like effect that reduces visibility and
    adds a smoky tint to the image.
    """

    effect_id = "smoke_haze"
    display_name = "Smoke Haze"
    description = "Adds smoke/haze atmospheric effect"
    domains = ["fire_smoke"]
    is_universal = False

    default_params = {
        "density": 0.15,
        "color_rgb": [180, 170, 160],  # Light gray smoke
        "depth_based": True,
        "turbulence": 0.3,
        "vertical_gradient": True,  # More smoke at top
    }

    def apply(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        smoke_source_mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> EffectResult:
        """
        Apply smoke haze effect.

        Args:
            image: Input image (BGR)
            mask: Object mask (optional)
            depth_map: Depth map (optional, for depth-based haze)
            smoke_source_mask: Mask indicating smoke source areas
            **kwargs: Additional parameters

        Returns:
            EffectResult with hazy image
        """
        start_time = time.time()
        params = self.validate_params(kwargs)

        try:
            result = self._apply_smoke_haze(
                image,
                depth_map=depth_map,
                smoke_source_mask=smoke_source_mask,
                density=params["density"],
                color_rgb=params["color_rgb"],
                depth_based=params["depth_based"],
                turbulence=params["turbulence"],
                vertical_gradient=params["vertical_gradient"],
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

    def _apply_smoke_haze(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray],
        smoke_source_mask: Optional[np.ndarray],
        density: float,
        color_rgb: Tuple[int, int, int],
        depth_based: bool,
        turbulence: float,
        vertical_gradient: bool,
    ) -> np.ndarray:
        """Apply the smoke haze effect."""
        h, w = image.shape[:2]

        # Create smoke color layer (RGB to BGR)
        smoke_color = np.full((h, w, 3), color_rgb[::-1], dtype=np.float32)

        # Create base haze intensity map
        haze_map = np.ones((h, w), dtype=np.float32) * density

        # Apply vertical gradient (smoke rises)
        if vertical_gradient:
            y_gradient = 1.0 - (np.arange(h)[:, np.newaxis] / h)  # More at top
            y_gradient = np.broadcast_to(y_gradient, (h, w))
            haze_map = haze_map * (0.3 + 0.7 * y_gradient)

        # Apply depth-based variation
        if depth_based and depth_map is not None:
            depth_normalized = depth_map.astype(np.float32)
            if depth_normalized.max() > 1:
                depth_normalized = depth_normalized / 255.0
            # More haze at distance
            haze_map = haze_map * (0.5 + 0.5 * depth_normalized)

        # Add turbulence (Perlin-like noise)
        if turbulence > 0:
            noise = self._generate_turbulence(h, w, turbulence)
            haze_map = haze_map * (0.7 + 0.3 * noise)

        # Apply smoke source influence
        if smoke_source_mask is not None:
            # Dilate smoke source to create dispersal area
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
            dispersed = cv2.dilate(smoke_source_mask.astype(np.float32), kernel)
            dispersed = cv2.GaussianBlur(dispersed, (101, 101), 0)
            haze_map = haze_map + dispersed * 0.3

        # Clip haze intensity
        haze_map = np.clip(haze_map, 0, 0.8)

        # Expand for broadcasting
        haze_map_3ch = haze_map[:, :, np.newaxis]

        # Blend image with smoke
        result = image.astype(np.float32) * (1 - haze_map_3ch) + \
                 smoke_color * haze_map_3ch

        return np.clip(result, 0, 255).astype(np.uint8)

    def _generate_turbulence(self, h: int, w: int, intensity: float) -> np.ndarray:
        """Generate turbulence noise pattern."""
        # Multi-scale noise for realistic smoke turbulence
        noise = np.zeros((h, w), dtype=np.float32)

        scales = [4, 8, 16, 32]
        weights = [0.4, 0.3, 0.2, 0.1]

        for scale, weight in zip(scales, weights):
            small_h, small_w = max(1, h // scale), max(1, w // scale)
            small_noise = np.random.random((small_h, small_w)).astype(np.float32)
            scaled_noise = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_LINEAR)
            noise += scaled_noise * weight

        # Normalize and apply intensity
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        noise = noise * intensity + (1 - intensity) * 0.5

        return noise
