"""
Ember Glow Effect

Adds glowing ember/spark effect to composite images.
Creates the characteristic bright orange glow of embers and sparks.
"""

import numpy as np
import cv2
import time
from typing import Optional, Dict, Any, Tuple

from ..base_effect import BaseEffect, EffectResult


class EmberGlowEffect(BaseEffect):
    """
    Adds ember/spark glow to images.

    Creates a bright, localized glow effect for ember and
    spark objects in fire scenes.
    """

    effect_id = "ember_glow"
    display_name = "Ember Glow"
    description = "Adds bright glow effect for embers and sparks"
    domains = ["fire_smoke"]
    is_universal = False

    default_params = {
        "intensity": 0.8,
        "color_rgb": [255, 140, 0],  # Bright orange
        "glow_radius": 15,  # pixels
        "bloom": True,  # Add bloom/halo effect
        "bloom_threshold": 200,  # Brightness threshold for bloom
    }

    def apply(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        ember_mask: Optional[np.ndarray] = None,
        **kwargs
    ) -> EffectResult:
        """
        Apply ember glow effect.

        Args:
            image: Input image (BGR)
            mask: Object mask (for ember objects)
            depth_map: Depth map (optional)
            ember_mask: Explicit mask for ember areas
            **kwargs: Additional parameters

        Returns:
            EffectResult with ember glow applied
        """
        start_time = time.time()
        params = self.validate_params(kwargs)

        try:
            # Use ember_mask if provided, otherwise use object mask
            glow_mask = ember_mask if ember_mask is not None else mask

            result = self._apply_ember_glow(
                image,
                glow_mask=glow_mask,
                intensity=params["intensity"],
                color_rgb=params["color_rgb"],
                glow_radius=params["glow_radius"],
                bloom=params["bloom"],
                bloom_threshold=params["bloom_threshold"],
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

    def _apply_ember_glow(
        self,
        image: np.ndarray,
        glow_mask: Optional[np.ndarray],
        intensity: float,
        color_rgb: Tuple[int, int, int],
        glow_radius: int,
        bloom: bool,
        bloom_threshold: int,
    ) -> np.ndarray:
        """Apply the ember glow effect."""
        h, w = image.shape[:2]

        if glow_mask is None:
            # Auto-detect bright small regions (potential embers)
            glow_mask = self._detect_bright_spots(image, bloom_threshold)

        if glow_mask is None or glow_mask.sum() == 0:
            # If still no mask and bloom is enabled, apply bloom effect
            if bloom:
                return self._apply_bloom(image, bloom_threshold, intensity)
            return image

        # Ensure mask is float
        glow_mask = glow_mask.astype(np.float32)
        if glow_mask.max() > 1:
            glow_mask = glow_mask / 255.0

        # Create glow around embers
        kernel_size = max(3, glow_radius * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create soft glow
        glow = cv2.GaussianBlur(glow_mask, (kernel_size, kernel_size), glow_radius / 3)

        # Create colored glow
        glow_color = np.array(color_rgb[::-1], dtype=np.float32)  # RGB to BGR

        # Apply glow as additive lighting
        glow_3ch = glow[:, :, np.newaxis]
        result = image.astype(np.float32) + glow_color * glow_3ch * intensity

        # Apply bloom if enabled
        if bloom:
            result = self._apply_bloom(
                np.clip(result, 0, 255).astype(np.uint8),
                bloom_threshold,
                intensity * 0.3
            ).astype(np.float32)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _detect_bright_spots(
        self,
        image: np.ndarray,
        threshold: int
    ) -> Optional[np.ndarray]:
        """Detect small bright spots that could be embers."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find bright pixels
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Filter for small regions (embers are small)
        contours, _ = cv2.findContours(
            bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ember_mask = np.zeros_like(gray)
        for contour in contours:
            area = cv2.contourArea(contour)
            # Embers are typically small (< 500 pixels)
            if 5 < area < 500:
                cv2.drawContours(ember_mask, [contour], -1, 255, -1)

        if ember_mask.sum() == 0:
            return None

        return ember_mask.astype(np.float32) / 255.0

    def _apply_bloom(
        self,
        image: np.ndarray,
        threshold: int,
        intensity: float
    ) -> np.ndarray:
        """Apply bloom/glow effect to bright areas."""
        # Extract bright areas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Create bloom from bright areas
        bloom = cv2.GaussianBlur(bright_mask.astype(np.float32), (0, 0), 30)
        bloom = bloom / 255.0

        # Color the bloom (warm tones)
        bloom_colored = np.zeros_like(image, dtype=np.float32)
        bloom_colored[:, :, 2] = bloom * 255 * 1.0    # Red
        bloom_colored[:, :, 1] = bloom * 255 * 0.7    # Green
        bloom_colored[:, :, 0] = bloom * 255 * 0.3    # Blue

        # Add bloom to image
        result = image.astype(np.float32) + bloom_colored * intensity

        return np.clip(result, 0, 255).astype(np.uint8)
