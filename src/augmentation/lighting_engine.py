"""
Advanced Lighting Estimation Engine

Provides HDR-based multi-light source detection and realistic shadow generation:
- HDR environment map estimation from single LDR images
- Multi-light source extraction (directional, point, area lights)
- Physically-based shadow generation with proper penumbra
- Color temperature and intensity estimation
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LightSource:
    """Represents a detected light source"""
    position: Tuple[float, float]  # (azimuth, elevation) in radians
    intensity: float  # 0-1 scale
    color: List[float]  # [R, G, B] in 0-255 range
    light_type: str  # 'directional', 'point', 'area'
    angular_size: float  # Size in image space (for area lights)

    def __repr__(self):
        return (f"LightSource(type={self.light_type}, "
                f"pos=({self.position[0]:.2f}, {self.position[1]:.2f}), "
                f"intensity={self.intensity:.2f})")


@dataclass
class HDREnvironmentMap:
    """HDR environment representation"""
    panorama: Optional[np.ndarray]  # HDR panorama (H, W, 3) if available
    light_sources: List[LightSource]  # Extracted light sources
    dominant_direction: Tuple[float, float]  # Main light direction (azimuth, elevation)
    color_temperature: float  # In Kelvin (e.g., 5500 for daylight)
    ambient_intensity: float  # Overall ambient light level 0-1


class AdvancedLightingEstimator:
    """
    Advanced lighting estimation for photorealistic shadow generation

    Uses multiple techniques to estimate lighting conditions:
    1. Gradient-based directional light detection (fast)
    2. Histogram-based light intensity and color extraction
    3. Multi-peak detection for multiple light sources
    4. Underwater-specific adjustments (depth-dependent attenuation)
    """

    def __init__(self,
                 max_light_sources: int = 3,
                 intensity_threshold: float = 0.6,
                 use_hdr_estimation: bool = False):
        """
        Args:
            max_light_sources: Maximum number of light sources to extract
            intensity_threshold: Minimum intensity for light source detection (0-1)
            use_hdr_estimation: Enable full HDR panorama estimation (experimental)
        """
        self.max_light_sources = max_light_sources
        self.intensity_threshold = intensity_threshold
        self.use_hdr_estimation = use_hdr_estimation

        logger.info(f"Advanced Lighting Estimator initialized "
                   f"(max_lights={max_light_sources}, threshold={intensity_threshold})")

    def estimate_lighting(self, background_img: np.ndarray) -> HDREnvironmentMap:
        """
        Estimate lighting conditions from background image

        Args:
            background_img: BGR background image

        Returns:
            HDREnvironmentMap with detected light sources
        """
        # Convert to float for processing
        img_float = background_img.astype(np.float32) / 255.0

        # Extract light sources using multiple methods
        light_sources = self._detect_light_sources(img_float)

        # Compute dominant direction
        dominant_dir = self._compute_dominant_direction(img_float, light_sources)

        # Estimate color temperature
        color_temp = self._estimate_color_temperature(img_float)

        # Compute ambient intensity
        ambient_intensity = self._compute_ambient_intensity(img_float)

        # Optional: Generate HDR panorama (expensive, experimental)
        panorama = None
        if self.use_hdr_estimation:
            panorama = self._estimate_hdr_panorama(img_float)

        return HDREnvironmentMap(
            panorama=panorama,
            light_sources=light_sources,
            dominant_direction=dominant_dir,
            color_temperature=color_temp,
            ambient_intensity=ambient_intensity
        )

    def _detect_light_sources(self, img: np.ndarray) -> List[LightSource]:
        """
        Detect multiple light sources from image

        Uses:
        1. Gradient analysis for directional lights
        2. Bright region detection for area lights
        3. Specular highlight detection for point lights
        """
        light_sources = []
        h, w = img.shape[:2]

        # Convert to LAB for better luminance analysis
        img_bgr = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        luminance = lab[:, :, 0].astype(np.float32) / 255.0

        # Method 1: Gradient-based directional light detection
        directional_lights = self._detect_directional_lights(img, luminance)
        light_sources.extend(directional_lights)

        # Method 2: Bright region detection for area/ambient lights
        area_lights = self._detect_area_lights(img, luminance)
        light_sources.extend(area_lights)

        # Method 3: High-frequency specular detection
        point_lights = self._detect_point_lights(img, luminance)
        light_sources.extend(point_lights)

        # Sort by intensity and limit to max_light_sources
        light_sources.sort(key=lambda x: x.intensity, reverse=True)
        light_sources = light_sources[:self.max_light_sources]

        # Ensure at least one light source (fallback)
        if len(light_sources) == 0:
            logger.warning("No light sources detected, using default overhead light")
            light_sources.append(LightSource(
                position=(0.0, np.pi/3),  # 60 degrees elevation, front
                intensity=0.7,
                color=[255, 255, 255],
                light_type='directional',
                angular_size=0.0
            ))

        return light_sources

    def _detect_directional_lights(self, img: np.ndarray,
                                   luminance: np.ndarray) -> List[LightSource]:
        """Detect directional lights using gradient analysis"""
        lights = []

        # Compute gradients in X and Y
        grad_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=5)

        # Compute gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Weight gradients by magnitude
        total_magnitude = np.sum(magnitude)
        if total_magnitude == 0:
            return lights

        # Compute average gradient direction (light comes from opposite)
        avg_grad_x = np.sum(grad_x * magnitude) / total_magnitude
        avg_grad_y = np.sum(grad_y * magnitude) / total_magnitude

        # Light direction (opposite of gradient for illumination)
        light_azimuth = np.arctan2(-avg_grad_y, -avg_grad_x)

        # Estimate elevation from gradient strength
        # Strong gradients -> low angle light, weak gradients -> overhead
        normalized_strength = np.mean(magnitude)
        elevation = np.pi/6 + (1 - normalized_strength) * np.pi/3  # 30-90 degrees

        # Estimate intensity from overall brightness
        intensity = float(np.mean(luminance))

        if intensity >= self.intensity_threshold:
            # Extract color from bright regions
            bright_mask = luminance > 0.7
            if np.sum(bright_mask) > 100:
                bright_regions = img[bright_mask]
                avg_color = np.mean(bright_regions, axis=0) * 255
                avg_color = avg_color[::-1]  # BGR to RGB
            else:
                avg_color = [255, 255, 255]

            lights.append(LightSource(
                position=(float(light_azimuth), float(elevation)),
                intensity=min(intensity, 1.0),
                color=avg_color.tolist(),
                light_type='directional',
                angular_size=0.0
            ))

        return lights

    def _detect_area_lights(self, img: np.ndarray,
                           luminance: np.ndarray) -> List[LightSource]:
        """Detect area lights from bright regions"""
        lights = []

        # Threshold for bright regions (top 10% brightness)
        threshold = np.percentile(luminance, 90)
        bright_mask = (luminance > max(threshold, 0.6)).astype(np.uint8)

        # Find connected components (bright regions)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bright_mask, connectivity=8
        )

        h, w = luminance.shape
        min_area = h * w * 0.01  # At least 1% of image

        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area:
                continue

            # Get centroid position
            cx, cy = centroids[i]

            # Convert to spherical coordinates
            # Assume image represents hemispherical view
            azimuth = (cx / w - 0.5) * np.pi  # -90 to +90 degrees
            elevation = (0.5 - cy / h) * np.pi/2  # 0 to 90 degrees

            # Get average intensity in this region
            region_mask = (labels == i)
            intensity = float(np.mean(luminance[region_mask]))

            if intensity >= self.intensity_threshold:
                # Get average color
                region_color = img[region_mask]
                avg_color = np.mean(region_color, axis=0) * 255
                avg_color = avg_color[::-1]  # BGR to RGB

                # Angular size (normalized by image size)
                angular_size = area / (h * w)

                lights.append(LightSource(
                    position=(float(azimuth), float(elevation)),
                    intensity=min(intensity, 1.0),
                    color=avg_color.tolist(),
                    light_type='area',
                    angular_size=float(angular_size)
                ))

        return lights

    def _detect_point_lights(self, img: np.ndarray,
                            luminance: np.ndarray) -> List[LightSource]:
        """Detect point lights from specular highlights"""
        lights = []

        # Detect very bright, small regions (specular highlights)
        # Use Laplacian to find sharp intensity changes
        laplacian = cv2.Laplacian(luminance, cv2.CV_32F, ksize=3)
        laplacian_abs = np.abs(laplacian)

        # Threshold for sharp peaks
        threshold = np.percentile(laplacian_abs, 98)
        peaks = (laplacian_abs > max(threshold, 0.1)).astype(np.uint8)

        # Find small peak regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            peaks, connectivity=8
        )

        h, w = luminance.shape
        max_area = h * w * 0.005  # Max 0.5% of image for point lights

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area > max_area or area < 3:
                continue

            # Get centroid
            cx, cy = centroids[i]

            # Convert to spherical
            azimuth = (cx / w - 0.5) * np.pi
            elevation = (0.5 - cy / h) * np.pi/2

            # Get intensity
            region_mask = (labels == i)
            intensity = float(np.mean(luminance[region_mask]))

            if intensity >= self.intensity_threshold:
                # Get color
                region_color = img[region_mask]
                avg_color = np.mean(region_color, axis=0) * 255
                avg_color = avg_color[::-1]

                lights.append(LightSource(
                    position=(float(azimuth), float(elevation)),
                    intensity=min(intensity, 1.0),
                    color=avg_color.tolist(),
                    light_type='point',
                    angular_size=0.0
                ))

        return lights

    def _compute_dominant_direction(self, img: np.ndarray,
                                   light_sources: List[LightSource]) -> Tuple[float, float]:
        """Compute weighted dominant light direction"""
        if not light_sources:
            return (0.0, np.pi/3)  # Default: front, 60 degrees

        # Weight by intensity
        total_intensity = sum(light.intensity for light in light_sources)

        if total_intensity == 0:
            return light_sources[0].position

        # Weighted average of directions (convert to Cartesian, average, convert back)
        avg_x = sum(light.intensity * np.cos(light.position[0]) * np.cos(light.position[1])
                   for light in light_sources) / total_intensity
        avg_y = sum(light.intensity * np.sin(light.position[0]) * np.cos(light.position[1])
                   for light in light_sources) / total_intensity
        avg_z = sum(light.intensity * np.sin(light.position[1])
                   for light in light_sources) / total_intensity

        # Convert back to spherical
        azimuth = np.arctan2(avg_y, avg_x)
        elevation = np.arctan2(avg_z, np.sqrt(avg_x**2 + avg_y**2))

        return (float(azimuth), float(elevation))

    def _estimate_color_temperature(self, img: np.ndarray) -> float:
        """
        Estimate color temperature in Kelvin

        Based on blue/red ratio in the image
        Warm light (sunset): 2000-3000K
        Neutral (daylight): 5000-6500K
        Cool (cloudy/underwater): 7000-10000K
        """
        # Get average color
        avg_color_bgr = np.mean(img, axis=(0, 1))
        b, g, r = avg_color_bgr

        # Blue/red ratio
        if r > 0:
            blue_red_ratio = b / r
        else:
            blue_red_ratio = 1.0

        # Map to Kelvin (empirical mapping)
        if blue_red_ratio < 0.8:
            # Warm light (more red)
            temp = 2000 + (0.8 - blue_red_ratio) * 2000
        elif blue_red_ratio > 1.2:
            # Cool light (more blue) - common underwater
            temp = 6500 + (blue_red_ratio - 1.2) * 5000
        else:
            # Neutral
            temp = 5000 + (blue_red_ratio - 1.0) * 3000

        return float(np.clip(temp, 2000, 12000))

    def _compute_ambient_intensity(self, img: np.ndarray) -> float:
        """Compute overall ambient light intensity"""
        # Use LAB luminance for perceptual uniformity
        img_bgr = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        luminance = lab[:, :, 0].astype(np.float32) / 255.0

        # Use median (robust to specular highlights)
        ambient = float(np.median(luminance))

        return np.clip(ambient, 0.0, 1.0)

    def _estimate_hdr_panorama(self, img: np.ndarray) -> np.ndarray:
        """
        Estimate HDR panorama from LDR image (experimental)

        This is a placeholder for future integration with models like:
        - 360U-Former (ViT-based HDR estimation)
        - PanoDiT (Diffusion-based panorama generation)

        For now, returns a simple equirectangular projection
        """
        # TODO: Integrate with 360U-Former or PanoDiT when available
        # For now, create a simple panorama representation

        h, w = img.shape[:2]
        panorama_h = 256
        panorama_w = 512

        # Resize to panorama dimensions
        panorama = cv2.resize(img, (panorama_w, panorama_h))

        # Simple HDR approximation (expand dynamic range)
        # Boost bright regions
        luminance = cv2.cvtColor((panorama * 255).astype(np.uint8),
                                 cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        boost_map = np.power(luminance, 0.5)  # Gamma correction
        boost_map = np.stack([boost_map] * 3, axis=-1)

        panorama_hdr = panorama * boost_map * 1.5  # HDR boost
        panorama_hdr = np.clip(panorama_hdr, 0.0, 2.0)  # Allow values > 1.0

        return panorama_hdr


def generate_multi_source_shadows(
    object_mask: np.ndarray,
    object_bbox: Tuple[int, int, int, int],  # x, y, w, h
    light_sources: List[LightSource],
    image_size: Tuple[int, int],  # h, w
    object_height_estimate: float = 0.2,
    max_shadow_intensity: float = 0.8
) -> np.ndarray:
    """
    Generate realistic shadows from multiple light sources

    Args:
        object_mask: Binary mask of the object (same size as image)
        object_bbox: Bounding box [x, y, w, h]
        light_sources: List of detected light sources
        image_size: Output image size (h, w)
        object_height_estimate: Estimated object height in relative units
        max_shadow_intensity: Maximum shadow darkness (0-1)

    Returns:
        Shadow composite mask (h, w) in 0-1 range
    """
    h, w = image_size
    shadow_composite = np.zeros((h, w), dtype=np.float32)

    if len(light_sources) == 0:
        return shadow_composite

    # Object center (for shadow direction calculation)
    obj_x, obj_y, obj_w, obj_h = object_bbox
    obj_center_x = obj_x + obj_w // 2
    obj_center_y = obj_y + obj_h // 2

    for light in light_sources:
        # Compute shadow offset from light direction
        azimuth, elevation = light.position

        # Shadow offset (larger offset for lower elevation angles)
        # Account for object height
        elevation_factor = np.cos(elevation)  # 1.0 at horizon, 0.0 overhead
        height_factor = object_height_estimate * 200  # Scale to pixels

        offset_x = int(np.cos(azimuth) * elevation_factor * height_factor)
        offset_y = int(np.sin(elevation) * height_factor)

        # Shadow direction is opposite to light direction
        offset_x = -offset_x

        # Create shadow mask by shifting object mask
        shadow_mask = np.zeros((h, w), dtype=np.float32)

        # Shift object mask to create shadow
        if offset_x != 0 or offset_y != 0:
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            shadow_mask = cv2.warpAffine(
                object_mask.astype(np.float32),
                M,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            shadow_mask = object_mask.astype(np.float32)

        # Shadow intensity based on light intensity and type
        base_intensity = light.intensity * max_shadow_intensity

        # Area lights create softer, less intense shadows
        if light.light_type == 'area':
            base_intensity *= 0.6

        # Penumbra (soft edges)
        # Larger blur for:
        # - Area lights
        # - Low elevation (long shadows)
        # - High intensity lights
        blur_size = 5

        if light.light_type == 'area':
            blur_size = int(15 * light.angular_size * 100)
        elif elevation < np.pi/4:  # < 45 degrees
            blur_size = int(15 * (1 - elevation / (np.pi/2)))

        blur_size = max(5, min(blur_size, 51))  # Clamp to reasonable range
        if blur_size % 2 == 0:
            blur_size += 1  # Must be odd

        shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_size, blur_size), 0)

        # Apply intensity
        shadow_mask *= base_intensity

        # Accumulate shadows (multiply for proper darkening)
        # Multiple shadows compound: (1-s1) * (1-s2) = 1 - (s1 + s2 - s1*s2)
        shadow_composite = 1.0 - (1.0 - shadow_composite) * (1.0 - shadow_mask)

    # Ensure shadow doesn't extend under the object itself
    # (shadows behind object relative to light source)
    shadow_composite = np.where(object_mask > 0, 0, shadow_composite)

    return np.clip(shadow_composite, 0.0, 1.0)


def apply_underwater_light_attenuation(
    light_sources: List[LightSource],
    depth_category: str,  # 'shallow', 'mid', 'deep'
    water_clarity: str = 'clear'  # 'clear', 'murky', 'very_murky'
) -> List[LightSource]:
    """
    Adjust light sources for underwater attenuation

    Water absorbs light, especially red wavelengths
    Deeper water -> more blue shift, less intensity
    """
    attenuation_factors = {
        'shallow': {'clear': 0.9, 'murky': 0.7, 'very_murky': 0.5},
        'mid': {'clear': 0.7, 'murky': 0.5, 'very_murky': 0.3},
        'deep': {'clear': 0.4, 'murky': 0.2, 'very_murky': 0.1}
    }

    blue_shift_factors = {
        'shallow': 1.1,
        'mid': 1.3,
        'deep': 1.6
    }

    attenuation = attenuation_factors.get(depth_category, {}).get(water_clarity, 0.7)
    blue_shift = blue_shift_factors.get(depth_category, 1.0)

    adjusted_lights = []

    for light in light_sources:
        # Reduce intensity
        new_intensity = light.intensity * attenuation

        # Shift color toward blue (reduce red, increase blue)
        r, g, b = light.color
        new_r = r * (1.0 / blue_shift)
        new_g = g
        new_b = b * blue_shift

        # Normalize to maintain rough brightness
        color_array = np.array([new_r, new_g, new_b])
        color_array = np.clip(color_array, 0, 255)

        adjusted_lights.append(LightSource(
            position=light.position,
            intensity=new_intensity,
            color=color_array.tolist(),
            light_type=light.light_type,
            angular_size=light.angular_size
        ))

    return adjusted_lights
