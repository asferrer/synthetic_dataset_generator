# src/augmentation/realism.py
"""
Realism Effects Module

Provides photorealistic effects for synthetic image generation including:
- Color correction
- Blur matching
- Lighting effects
- Underwater effects
- Shadows
- Caustics
- Blending methods
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


def apply_poisson_blending(src_obj: np.ndarray, background: np.ndarray,
                           mask: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
    """Apply Poisson Blending for seamless object cloning.

    Args:
        src_obj: Source object image (BGR)
        background: Background image (BGR)
        mask: Binary mask of the object
        center: (x, y) center position for placement

    Returns:
        Blended image
    """
    try:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return cv2.seamlessClone(src_obj, background, mask, center, cv2.NORMAL_CLONE)
    except cv2.error as e:
        logger.error(f"Poisson Blending error: {e}. Falling back to simple paste.")
        inv_mask = cv2.bitwise_not(mask)
        fg = cv2.bitwise_and(src_obj, src_obj, mask=mask)
        bg = cv2.bitwise_and(background, background, mask=inv_mask)
        return cv2.add(bg, fg)


def transfer_color_correction(obj_img: np.ndarray, bg_roi: np.ndarray,
                              mask: np.ndarray, intensity: float = 0.4,
                              preserve_object_colors: bool = True) -> np.ndarray:
    """Adjust object colors to match underwater background lighting conditions.

    This is an improved algorithm specifically designed for synthetic underwater
    data generation. Unlike traditional histogram matching, this preserves object
    distinctiveness while adapting to the scene's lighting conditions.

    The algorithm:
    1. Matches color TEMPERATURE (tint) - shifts colors toward background's tone
    2. Preserves color VARIANCE - objects keep their detail and recognizability
    3. Adjusts LUMINANCE subtly - adapts to scene brightness without destroying contrast

    Args:
        obj_img: Object image (BGR)
        bg_roi: Background region of interest (BGR)
        mask: Binary mask
        intensity: Blending intensity (0.0-1.0), default 0.4 for subtle effect
        preserve_object_colors: If True, limits variance compression to preserve
                               object recognizability (recommended for underwater)

    Returns:
        Color-corrected object image that fits the scene while remaining recognizable
    """
    try:
        if cv2.countNonZero(mask) == 0:
            return obj_img

        obj_lab = cv2.cvtColor(obj_img, cv2.COLOR_BGR2LAB)
        bg_lab = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2LAB)

        mean_bg, std_bg = cv2.meanStdDev(bg_lab, mask=mask)
        mean_obj, std_obj = cv2.meanStdDev(obj_lab, mask=mask)

        # Prevent division by zero
        std_obj[std_obj == 0] = 1
        std_bg[std_bg == 0] = 1

        l, a, b = cv2.split(obj_lab)

        if preserve_object_colors:
            # IMPROVED ALGORITHM FOR UNDERWATER SYNTHETIC DATA
            # Key insight: We want to match COLOR TEMPERATURE (mean shift)
            # but PRESERVE object's color variance (its distinctive features)

            # Luminance (L channel): Very subtle adjustment
            # Only shift brightness slightly toward background, preserve contrast
            l_ratio = np.clip(std_bg[0] / std_obj[0], 0.7, 1.3)  # Limit to ±30%
            l_shift = (mean_bg[0] - mean_obj[0]) * 0.3  # Only 30% of mean shift
            l = np.clip(l.astype(np.float32) + l_shift, 0, 255).astype(np.uint8)

            # Color channels (a, b): Match color temperature/tint
            # Shift the mean (color cast) but preserve variance (color detail)
            # This makes the object look like it's lit by the same light source
            # without destroying its distinctive colors

            # Limit variance ratio to prevent color destruction
            # Min 0.6 means we never compress variance by more than 40%
            a_ratio = np.clip(std_bg[1] / std_obj[1], 0.6, 1.5)
            b_ratio = np.clip(std_bg[2] / std_obj[2], 0.6, 1.5)

            # Apply gentler transformation
            # Weight toward mean shift (color temperature) vs variance change
            temperature_weight = 0.7  # 70% color temperature, 30% variance matching

            a_float = a.astype(np.float32)
            b_float = b.astype(np.float32)

            # Color temperature shift (move toward background's color cast)
            a_temp_shift = (mean_bg[1] - mean_obj[1]) * temperature_weight
            b_temp_shift = (mean_bg[2] - mean_obj[2]) * temperature_weight

            # Gentle variance adjustment (preserve object's color variety)
            a_centered = a_float - mean_obj[1]
            b_centered = b_float - mean_obj[2]

            # Blend between original variance and matched variance
            variance_blend = 0.3  # Only 30% variance matching
            a_scaled = a_centered * (1 - variance_blend + variance_blend * a_ratio)
            b_scaled = b_centered * (1 - variance_blend + variance_blend * b_ratio)

            a = np.clip(a_scaled + mean_obj[1] + a_temp_shift, 0, 255).astype(np.uint8)
            b = np.clip(b_scaled + mean_obj[2] + b_temp_shift, 0, 255).astype(np.uint8)

        else:
            # Original aggressive algorithm (not recommended for underwater)
            l = np.clip((l - mean_obj[0]) * (std_bg[0] / std_obj[0]) + mean_bg[0], 0, 255).astype(np.uint8)
            a = np.clip((a - mean_obj[1]) * (std_bg[1] / std_obj[1]) + mean_bg[1], 0, 255).astype(np.uint8)
            b = np.clip((b - mean_obj[2]) * (std_bg[2] / std_obj[2]) + mean_bg[2], 0, 255).astype(np.uint8)

        corrected_lab = cv2.merge([l, a, b])
        corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

        # Blend with original - use the specified intensity
        blended_bgr = cv2.addWeighted(obj_img, 1 - intensity, corrected_bgr, intensity, 0)

        return cv2.bitwise_and(blended_bgr, blended_bgr, mask=mask)
    except Exception as e:
        logger.error(f"Color correction error: {e}")
        return obj_img


def estimate_blur(image: np.ndarray) -> float:
    """Estimate blur level using Laplacian variance.

    Args:
        image: Input image

    Returns:
        Blur estimate (higher = sharper)
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def match_blur(obj_img: np.ndarray, bg_roi: np.ndarray, mask: np.ndarray,
               strength: float = 1.0) -> np.ndarray:
    """Match object blur to background blur level.

    Args:
        obj_img: Object image
        bg_roi: Background region of interest
        mask: Binary mask
        strength: Blur strength multiplier

    Returns:
        Blur-adjusted object image
    """
    try:
        if cv2.countNonZero(mask) == 0:
            return obj_img

        bg_blur = estimate_blur(bg_roi)
        obj_blur = estimate_blur(obj_img)

        EPSILON = 1e-6
        bg_blur = max(bg_blur, EPSILON)
        obj_blur = max(obj_blur, EPSILON)

        # Apply blur only if object is significantly sharper
        if obj_blur > bg_blur * 1.5:
            kernel_ratio = obj_blur / bg_blur
            kernel_size = int(min(kernel_ratio * 0.5 * strength, 10)) * 2 + 1
            kernel_size = max(3, min(kernel_size, 21))

            blurred_obj = cv2.GaussianBlur(obj_img, (kernel_size, kernel_size), 0)
            return cv2.bitwise_and(blurred_obj, blurred_obj, mask=mask)

        return obj_img

    except Exception as e:
        logger.error(f"Blur matching error: {e}")
        return obj_img


def generate_shadow(mask: np.ndarray, strength: float = 0.7,
                    blur_kernel_size: int = 21,
                    offset_x: int = 10, offset_y: int = 10) -> np.ndarray:
    """Generate a soft projected shadow from object mask.

    Args:
        mask: Binary mask
        strength: Shadow opacity (0.0-1.0)
        blur_kernel_size: Gaussian blur kernel size
        offset_x: Horizontal shadow offset
        offset_y: Vertical shadow offset

    Returns:
        Shadow layer (BGRA)
    """
    shadow_layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    shadow_layer[mask == 255] = (0, 0, 0, int(255 * strength))
    shadow_layer = cv2.GaussianBlur(shadow_layer, (blur_kernel_size, blur_kernel_size), 0)
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return cv2.warpAffine(shadow_layer, M, (shadow_layer.shape[1], shadow_layer.shape[0]))


def extract_light_direction(background_roi: np.ndarray) -> float:
    """Estimate light direction from background gradients.

    Args:
        background_roi: Background region

    Returns:
        Light angle in radians
    """
    try:
        gray = cv2.cvtColor(background_roi, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        angle = np.arctan2(np.mean(grad_y), np.mean(grad_x))
        return angle
    except Exception as e:
        logger.warning(f"Light direction extraction error: {e}. Using default.")
        return np.pi / 4


def generate_dynamic_shadow(mask: np.ndarray, background_roi: np.ndarray,
                            strength_range: Tuple[float, float] = (0.3, 0.7),
                            distance_range: Tuple[int, int] = (5, 25),
                            blur_range: Tuple[int, int] = (11, 31)) -> np.ndarray:
    """Generate dynamic shadow based on background lighting.

    Args:
        mask: Object mask
        background_roi: Background region for light analysis
        strength_range: Shadow intensity range
        distance_range: Shadow distance range in pixels
        blur_range: Blur kernel size range

    Returns:
        Shadow layer (BGRA)
    """
    try:
        light_angle = extract_light_direction(background_roi)

        strength = np.random.uniform(*strength_range)
        distance = np.random.randint(*distance_range)
        blur_size = np.random.randint(*blur_range) | 1

        offset_x = int(distance * np.cos(light_angle))
        offset_y = int(distance * np.sin(light_angle))

        shadow_base = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        shadow_base[mask == 255] = (0, 0, 0, int(255 * strength))

        # Penumbra
        penumbra = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
        penumbra = cv2.bitwise_xor(penumbra, mask)
        shadow_base[penumbra == 255] = (0, 0, 0, int(255 * strength * 0.3))

        shadow_blurred = cv2.GaussianBlur(shadow_base, (blur_size, blur_size), 0)

        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        shadow_final = cv2.warpAffine(shadow_blurred, M,
                                       (shadow_blurred.shape[1], shadow_blurred.shape[0]))

        return shadow_final
    except Exception as e:
        logger.error(f"Dynamic shadow error: {e}. Using simple shadow.")
        return generate_shadow(mask, strength=0.5, blur_kernel_size=21, offset_x=10, offset_y=10)


def add_lighting_effect(obj_img: np.ndarray, light_type: str = 'spotlight',
                        strength: float = 1.5) -> np.ndarray:
    """Add lighting effect to object.

    Args:
        obj_img: Object image (BGR)
        light_type: Type of lighting ('spotlight', 'gradient', 'ambient')
        strength: Lighting intensity

    Returns:
        Lit object image
    """
    h, w = obj_img.shape[:2]
    light_map = np.zeros((h, w), dtype=np.float32)

    if light_type == 'spotlight':
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        radius = max(h, w) / 2.0
        light_map = np.exp(-(dist_from_center / radius)**2)
        light_map = (light_map - np.min(light_map)) / (np.max(light_map) - np.min(light_map))
        light_map *= strength
    elif light_type == 'gradient':
        # Top-down gradient
        light_map = np.tile(np.linspace(strength, 0.5, h), (w, 1)).T.astype(np.float32)
    else:  # ambient
        light_map = np.ones((h, w), dtype=np.float32) * strength

    hls = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HLS).astype(np.float32)
    hls[:, :, 1] *= light_map
    hls[:, :, 1] = np.clip(hls[:, :, 1], 0, 255)
    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)


def apply_underwater_effect(obj_img: np.ndarray,
                            color_cast: Tuple[int, int, int] = (120, 80, 20),
                            intensity: float = 0.25) -> np.ndarray:
    """Apply underwater color effect.

    Args:
        obj_img: Object image (BGR)
        color_cast: BGR water tint color
        intensity: Effect intensity (0.0-1.0)

    Returns:
        Tinted image
    """
    try:
        haze_color = np.full(obj_img.shape, color_cast, dtype=obj_img.dtype)
        hazed_obj = cv2.addWeighted(obj_img, 1 - intensity, haze_color, intensity, 0)
        return hazed_obj
    except Exception as e:
        logger.error(f"Underwater effect error: {e}")
        return obj_img


def add_upscaling_noise(image: np.ndarray, intensity: float = 10) -> np.ndarray:
    """Add Gaussian noise to simulate upscaling artifacts.

    Args:
        image: Input image
        intensity: Noise intensity

    Returns:
        Noisy image
    """
    h, w, c = image.shape
    noise = np.random.randn(h, w, c) * intensity
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def laplacian_pyramid_blending(obj_img: np.ndarray, background: np.ndarray,
                                mask: np.ndarray, num_levels: int = 4) -> np.ndarray:
    """Blend using Laplacian pyramids for smooth edges.

    Args:
        obj_img: Object image (BGR)
        background: Background image (BGR)
        mask: Binary mask (single channel)
        num_levels: Pyramid levels

    Returns:
        Blended image
    """
    gp_obj = [obj_img.astype(np.float32)]
    gp_bg = [background.astype(np.float32)]
    gp_mask = [mask.astype(np.float32) / 255.0]

    for i in range(num_levels):
        gp_obj.append(cv2.pyrDown(gp_obj[i]))
        gp_bg.append(cv2.pyrDown(gp_bg[i]))
        gp_mask.append(cv2.pyrDown(gp_mask[i]))

    lp_obj = [gp_obj[num_levels-1]]
    lp_bg = [gp_bg[num_levels-1]]

    for i in range(num_levels - 1, 0, -1):
        size = (gp_obj[i-1].shape[1], gp_obj[i-1].shape[0])
        expanded_obj = cv2.pyrUp(gp_obj[i], dstsize=size)
        lp_obj.append(cv2.subtract(gp_obj[i-1], expanded_obj))
        size = (gp_bg[i-1].shape[1], gp_bg[i-1].shape[0])
        expanded_bg = cv2.pyrUp(gp_bg[i], dstsize=size)
        lp_bg.append(cv2.subtract(gp_bg[i-1], expanded_bg))

    lp_fused = []
    for l_obj, l_bg, g_mask in zip(lp_obj, lp_bg, gp_mask[::-1]):
        g_mask_3c = cv2.merge([g_mask, g_mask, g_mask])
        fused_level = l_obj * g_mask_3c + l_bg * (1 - g_mask_3c)
        lp_fused.append(fused_level)

    fused_reconstruction = lp_fused[0]
    for i in range(1, len(lp_fused)):
        size = (lp_fused[i].shape[1], lp_fused[i].shape[0])
        fused_reconstruction = cv2.pyrUp(fused_reconstruction, dstsize=size)
        fused_reconstruction = cv2.add(fused_reconstruction, lp_fused[i])

    return np.clip(fused_reconstruction, 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, kernel_size: int = 15,
                      angle: Optional[float] = None) -> np.ndarray:
    """Apply motion blur effect.

    Args:
        image: Input image
        kernel_size: Motion blur kernel size
        angle: Blur direction in degrees (random if None)

    Returns:
        Motion-blurred image
    """
    if angle is None:
        angle = np.random.uniform(0, 180)

    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    # Rotate kernel
    center = (kernel_size // 2, kernel_size // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    return cv2.filter2D(image, -1, kernel)


def smooth_edges(image: np.ndarray, mask: np.ndarray, feather_radius: int = 5) -> np.ndarray:
    """Smooth edges of an object for better blending.

    Args:
        image: Input image (BGRA or BGR)
        mask: Binary mask
        feather_radius: Edge feathering radius

    Returns:
        Edge-smoothed image
    """
    # Create feathered mask
    feathered_mask = cv2.GaussianBlur(mask.astype(np.float32),
                                       (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
    feathered_mask = (feathered_mask / 255.0).astype(np.float32)

    if len(image.shape) == 3 and image.shape[2] == 4:
        # BGRA - modify alpha channel
        image = image.copy()
        image[:, :, 3] = (feathered_mask * 255).astype(np.uint8)
        return image
    else:
        # BGR - return with feathered mask
        return image, feathered_mask


class CausticsCache:
    """Sistema de caché para mapas de cáusticas pre-generados con transformaciones aleatorias."""

    def __init__(self, cache_dir=".caustics_cache", num_templates=15):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.num_templates = num_templates
        self.templates = {}  # template_key -> numpy array
        self._initialized = False

    def generate_templates(self, sizes=[(1024, 1024), (2048, 2048)]):
        """Pre-genera templates de cáusticas en diferentes tamaños."""
        try:
            import noise
        except ImportError:
            logger.warning("noise library not available. Using fallback pattern.")
            self._generate_fallback_templates(sizes)
            return

        logger.info(f"Generando {self.num_templates} templates de cáusticas...")

        for size in sizes:
            width, height = size

            for template_id in range(self.num_templates):
                complexity = np.random.uniform(0.008, 0.015)
                octaves = np.random.randint(3, 6)

                caustics_map = np.zeros((height, width), dtype=np.float32)

                for y in range(height):
                    for x in range(width):
                        value = noise.pnoise2(
                            x * complexity,
                            y * complexity,
                            octaves=octaves
                        )
                        caustics_map[y, x] = value

                caustics_map = (caustics_map - np.min(caustics_map)) / \
                              (np.max(caustics_map) - np.min(caustics_map))

                template_key = f"{width}x{height}_{template_id}"
                cache_file = self.cache_dir / f"{template_key}.npy"
                np.save(cache_file, caustics_map)
                self.templates[template_key] = caustics_map

        self._initialized = True
        logger.info(f"Caché de cáusticas generado: {len(self.templates)} templates")

    def _generate_fallback_templates(self, sizes: List[Tuple[int, int]]) -> None:
        """Generate templates using fast sinusoidal patterns."""
        logger.info("Generating fallback sinusoidal templates...")

        for size in sizes:
            width, height = size

            for template_id in range(self.num_templates):
                freq_x = np.random.uniform(3, 6)
                freq_y = np.random.uniform(3, 6)
                phase = np.random.uniform(0, 2 * np.pi)

                x = np.linspace(0, freq_x * np.pi, width)
                y = np.linspace(0, freq_y * np.pi, height)
                X, Y = np.meshgrid(x, y)

                caustics_map = (np.sin(X + phase) * np.cos(Y) +
                               np.sin(2 * X + Y + phase) * 0.5 +
                               np.cos(X - 2 * Y) * 0.3)

                caustics_map = (caustics_map - caustics_map.min()) / \
                              (caustics_map.max() - caustics_map.min())

                template_key = f"{width}x{height}_{template_id}"
                cache_file = self.cache_dir / f"{template_key}.npy"
                np.save(cache_file, caustics_map.astype(np.float32))
                self.templates[template_key] = caustics_map.astype(np.float32)

        self._initialized = True
        logger.info(f"Fallback caustics templates generated: {len(self.templates)}")

    def load_from_disk(self) -> bool:
        """Load templates from disk if they exist."""
        cache_files = list(self.cache_dir.glob("*.npy"))

        if not cache_files:
            return False

        for cache_file in cache_files:
            template_key = cache_file.stem
            self.templates[template_key] = np.load(cache_file)

        self._initialized = True
        logger.info(f"Loaded {len(cache_files)} caustics templates from disk")
        return True

    def ensure_initialized(self) -> None:
        """Ensure cache is initialized, loading or generating as needed."""
        if self._initialized:
            return

        if not self.load_from_disk():
            self.generate_templates()

    def get_caustics(self, width: int, height: int, brightness: float = 0.4) -> np.ndarray:
        """Obtiene mapa de cáusticas adaptado al tamaño con transformaciones aleatorias."""
        self.ensure_initialized()

        target_size = width * height
        best_template_key = None
        best_diff = float('inf')

        for template_key in self.templates.keys():
            size_str = template_key.rsplit('_', 1)[0]
            w, h = map(int, size_str.split('x'))
            diff = abs(w * h - target_size)

            if diff < best_diff:
                best_diff = diff
                best_template_key = template_key

        if best_template_key is None:
            return self._generate_instant_fallback(width, height, brightness)

        caustics = self.templates[best_template_key].copy()
        caustics = self._random_transform(caustics)

        if caustics.shape[1] != width or caustics.shape[0] != height:
            caustics = cv2.resize(caustics, (width, height), interpolation=cv2.INTER_LINEAR)

        caustics = np.clip(caustics * brightness + (1 - brightness), 0, 2)
        return cv2.cvtColor(caustics.astype(np.float32), cv2.COLOR_GRAY2BGR)

    def _random_transform(self, caustics: np.ndarray) -> np.ndarray:
        """Aplica transformaciones aleatorias para variación visual."""
        k = np.random.randint(0, 4)
        if k > 0:
            caustics = np.rot90(caustics, k)

        if np.random.random() > 0.5:
            caustics = np.fliplr(caustics)
        if np.random.random() > 0.5:
            caustics = np.flipud(caustics)

        contrast = np.random.uniform(0.8, 1.2)
        mean = np.mean(caustics)
        caustics = (caustics - mean) * contrast + mean
        caustics = np.clip(caustics, 0, 1)

        return caustics

    def _generate_instant_fallback(self, width: int, height: int,
                                    brightness: float = 0.4) -> np.ndarray:
        """Generate instant fallback pattern (faster than Perlin)."""
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        X, Y = np.meshgrid(x, y)

        caustics = (np.sin(X) * np.cos(Y) +
                   np.sin(2 * X + Y) * 0.5 +
                   np.cos(X - 2 * Y) * 0.3)

        caustics = (caustics - caustics.min()) / (caustics.max() - caustics.min())
        caustics = np.clip(caustics * brightness + (1 - brightness), 0, 2)

        return cv2.cvtColor(caustics.astype(np.float32), cv2.COLOR_GRAY2BGR)


# Global singleton instance
_caustics_cache: Optional[CausticsCache] = None


def get_caustics_cache() -> CausticsCache:
    """Get singleton caustics cache instance."""
    global _caustics_cache
    if _caustics_cache is None:
        _caustics_cache = CausticsCache()
        _caustics_cache.ensure_initialized()
    return _caustics_cache


def generate_caustics_map(width: int, height: int,
                          brightness: float = 0.4) -> np.ndarray:
    """Generate caustics map using cache system."""
    cache = get_caustics_cache()
    return cache.get_caustics(width, height, brightness)


def apply_caustics(image: np.ndarray, caustics_map: np.ndarray) -> np.ndarray:
    """Apply caustics map to an image."""
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)
    hls[:, :, 1] *= caustics_map[:, :, 1]
    hls[:, :, 1] = np.clip(hls[:, :, 1], 0, 255)
    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)