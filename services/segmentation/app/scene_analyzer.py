"""
Semantic Scene Analyzer
=======================

Provides semantic understanding of underwater scenes for intelligent object placement:
- Scene region classification (water column, seafloor, surface, vegetation)
- Object-scene compatibility validation
- Text-driven segmentation support (SAM3-ready)
- Heuristic fallbacks for robust operation

This is the canonical implementation - used by the Segmentation microservice.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SceneRegion(Enum):
    """Types of scene regions in underwater environments"""
    OPEN_WATER = "open_water"
    SEAFLOOR = "seafloor"
    SURFACE = "surface"
    VEGETATION = "vegetation"
    ROCKY = "rocky"
    SANDY = "sandy"
    MURKY = "murky"
    UNKNOWN = "unknown"


@dataclass
class SceneAnalysis:
    """Results of scene analysis"""
    dominant_region: SceneRegion
    region_map: np.ndarray
    region_scores: Dict[str, float]
    depth_zones: Dict[str, Tuple[float, float]]
    placement_zones: List[Tuple[int, int, int, int]]
    scene_brightness: float
    water_clarity: str
    color_temperature: str


@dataclass
class DebugInfo:
    """Debug information for scene analysis explainability"""
    analysis_method: str
    processing_time_ms: float
    region_masks: Dict[str, np.ndarray]
    region_confidences: Dict[str, float]
    sam3_prompts_used: List[str]
    decision_log: List[str]
    visualization_path: Optional[str] = None


@dataclass
class PlacementDecision:
    """Debug info for a single placement decision"""
    object_class: str
    requested_position: Tuple[int, int]
    region_at_position: str
    compatibility_score: float
    reason: str
    alternative_positions: List[Tuple[int, int, float]]
    decision: str


# Object-Scene compatibility rules
SCENE_COMPATIBILITY = {
    'fish': {
        SceneRegion.OPEN_WATER: 1.0,
        SceneRegion.SURFACE: 0.8,
        SceneRegion.VEGETATION: 0.7,
        SceneRegion.SEAFLOOR: 0.5,
        SceneRegion.ROCKY: 0.6,
        SceneRegion.MURKY: 0.4,
    },
    'shark': {
        SceneRegion.OPEN_WATER: 1.0,
        SceneRegion.SURFACE: 0.9,
        SceneRegion.SEAFLOOR: 0.3,
        SceneRegion.VEGETATION: 0.4,
    },
    'jellyfish': {
        SceneRegion.OPEN_WATER: 1.0,
        SceneRegion.SURFACE: 0.9,
        SceneRegion.SEAFLOOR: 0.2,
    },
    'octopus': {
        SceneRegion.SEAFLOOR: 1.0,
        SceneRegion.ROCKY: 1.0,
        SceneRegion.OPEN_WATER: 0.4,
        SceneRegion.VEGETATION: 0.7,
    },
    'crab': {
        SceneRegion.SEAFLOOR: 1.0,
        SceneRegion.SANDY: 1.0,
        SceneRegion.ROCKY: 0.9,
        SceneRegion.OPEN_WATER: 0.1,
    },
    'starfish': {
        SceneRegion.SEAFLOOR: 1.0,
        SceneRegion.SANDY: 1.0,
        SceneRegion.ROCKY: 0.8,
        SceneRegion.OPEN_WATER: 0.1,
    },
    'plastic': {
        SceneRegion.OPEN_WATER: 0.9,
        SceneRegion.SURFACE: 1.0,
        SceneRegion.SEAFLOOR: 0.7,
        SceneRegion.VEGETATION: 0.6,
    },
    'plastic_bag': {
        SceneRegion.OPEN_WATER: 1.0,
        SceneRegion.SURFACE: 1.0,
        SceneRegion.SEAFLOOR: 0.5,
    },
    'plastic_bottle': {
        SceneRegion.SURFACE: 1.0,
        SceneRegion.OPEN_WATER: 0.8,
        SceneRegion.SEAFLOOR: 0.6,
    },
    'bottle': {
        SceneRegion.SURFACE: 0.9,
        SceneRegion.OPEN_WATER: 0.7,
        SceneRegion.SEAFLOOR: 0.8,
    },
    'can': {
        SceneRegion.SEAFLOOR: 1.0,
        SceneRegion.SANDY: 1.0,
        SceneRegion.ROCKY: 0.8,
        SceneRegion.OPEN_WATER: 0.3,
        SceneRegion.SURFACE: 0.1,
    },
    'metal': {
        SceneRegion.SEAFLOOR: 1.0,
        SceneRegion.ROCKY: 0.9,
        SceneRegion.OPEN_WATER: 0.2,
        SceneRegion.SURFACE: 0.1,
    },
    'glass': {
        SceneRegion.SEAFLOOR: 1.0,
        SceneRegion.SANDY: 1.0,
        SceneRegion.OPEN_WATER: 0.3,
    },
    'glass_bottle': {
        SceneRegion.SEAFLOOR: 1.0,
        SceneRegion.SANDY: 0.9,
        SceneRegion.ROCKY: 0.8,
        SceneRegion.OPEN_WATER: 0.2,
    },
    'debris': {
        SceneRegion.SEAFLOOR: 0.9,
        SceneRegion.OPEN_WATER: 0.7,
        SceneRegion.SURFACE: 0.6,
        SceneRegion.VEGETATION: 0.5,
    },
    'trash': {
        SceneRegion.SEAFLOOR: 0.8,
        SceneRegion.OPEN_WATER: 0.7,
        SceneRegion.SURFACE: 0.7,
    },
    'rope': {
        SceneRegion.SEAFLOOR: 0.9,
        SceneRegion.OPEN_WATER: 0.8,
        SceneRegion.VEGETATION: 0.7,
    },
    'net': {
        SceneRegion.OPEN_WATER: 0.9,
        SceneRegion.SEAFLOOR: 0.8,
        SceneRegion.VEGETATION: 0.7,
    },
}

DEFAULT_COMPATIBILITY = {
    SceneRegion.OPEN_WATER: 0.7,
    SceneRegion.SEAFLOOR: 0.7,
    SceneRegion.SURFACE: 0.5,
    SceneRegion.VEGETATION: 0.5,
    SceneRegion.ROCKY: 0.6,
    SceneRegion.SANDY: 0.6,
    SceneRegion.MURKY: 0.4,
    SceneRegion.UNKNOWN: 0.5,
}


class SemanticSceneAnalyzer:
    """
    Analyzes underwater scenes for semantic understanding.

    This is the canonical implementation used by the Segmentation microservice.
    """

    def __init__(
        self,
        use_sam3: bool = False,
        sam3_model=None,
        sam3_processor=None,
        min_compatibility_score: float = 0.4,
        device: str = 'cuda',
        debug: bool = False,
        debug_output_dir: Optional[str] = None,
    ):
        self.use_sam3 = use_sam3
        self.min_compatibility_score = min_compatibility_score
        self.device = device
        self.debug = debug
        self.debug_output_dir = debug_output_dir or "/shared/segmentation/debug"

        # SAM3 model (passed from service state)
        self._sam3_model = sam3_model
        self._sam3_processor = sam3_processor

        # Debug state
        self._last_debug_info: Optional[DebugInfo] = None
        self._placement_decisions: List[PlacementDecision] = []

        if debug:
            import os
            os.makedirs(self.debug_output_dir, exist_ok=True)

        logger.info(f"SemanticSceneAnalyzer initialized (SAM3: {self.use_sam3}, Debug: {self.debug})")

    def analyze_scene(self, image: np.ndarray) -> SceneAnalysis:
        """Analyze scene to identify regions and characteristics."""
        h, w = image.shape[:2]

        if self.use_sam3 and self._sam3_model is not None:
            region_map, region_scores = self._analyze_with_sam3(image)
        else:
            region_map, region_scores = self._analyze_with_heuristics(image)

        dominant_region = max(region_scores, key=region_scores.get)
        dominant_region = SceneRegion(dominant_region) if dominant_region in [r.value for r in SceneRegion] else SceneRegion.UNKNOWN

        depth_zones = self._compute_depth_zones(image, region_map)
        placement_zones = self._find_placement_zones(region_map, h, w)
        scene_brightness = self._compute_brightness(image)
        water_clarity = self._estimate_water_clarity(image)
        color_temperature = self._estimate_color_temperature(image)

        return SceneAnalysis(
            dominant_region=dominant_region,
            region_map=region_map,
            region_scores=region_scores,
            depth_zones=depth_zones,
            placement_zones=placement_zones,
            scene_brightness=scene_brightness,
            water_clarity=water_clarity,
            color_temperature=color_temperature,
        )

    def _analyze_with_sam3(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Analyze scene using SAM3 text-driven segmentation."""
        import torch
        from PIL import Image

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        region_prompts = [
            ("water", SceneRegion.OPEN_WATER),
            ("seafloor", SceneRegion.SEAFLOOR),
            ("water surface", SceneRegion.SURFACE),
            ("seaweed", SceneRegion.VEGETATION),
            ("rock", SceneRegion.ROCKY),
            ("sand", SceneRegion.SANDY),
        ]

        region_map = np.zeros((h, w), dtype=np.uint8)
        region_scores = {region.value: 0.0 for region in SceneRegion}
        confidence_map = np.zeros((h, w), dtype=np.float32)

        region_value_map = {
            SceneRegion.OPEN_WATER: 1, SceneRegion.SEAFLOOR: 2,
            SceneRegion.SURFACE: 3, SceneRegion.VEGETATION: 4,
            SceneRegion.ROCKY: 5, SceneRegion.SANDY: 6, SceneRegion.MURKY: 7,
        }

        try:
            for prompt, region_type in region_prompts:
                inputs = self._sam3_processor(
                    images=pil_image, text=prompt, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._sam3_model(**inputs)

                results = self._sam3_processor.post_process_instance_segmentation(
                    outputs, threshold=0.5, mask_threshold=0.5, target_sizes=[(h, w)]
                )[0]

                combined_mask = np.zeros((h, w), dtype=np.float32)
                if 'masks' in results and len(results['masks']) > 0:
                    for mask, score in zip(results['masks'], results['scores']):
                        mask_np = mask.cpu().numpy().astype(np.float32)
                        score_val = score.cpu().item()
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h))
                        combined_mask = np.maximum(combined_mask, mask_np * score_val)

                region_value = region_value_map.get(region_type, 0)
                update_mask = (combined_mask > 0.5) & (combined_mask > confidence_map)
                region_map = np.where(update_mask, region_value, region_map)
                confidence_map = np.maximum(confidence_map, combined_mask)
                region_scores[region_type.value] = float((combined_mask > 0.5).sum()) / (h * w)

            # Normalize
            total = sum(region_scores.values())
            if total > 0:
                region_scores = {k: v / total for k, v in region_scores.items()}

            # Fill unclassified
            unclassified = (region_map == 0)
            if unclassified.sum() > 0:
                y_pos = np.arange(h)[:, np.newaxis] / h
                y_pos = np.broadcast_to(y_pos, (h, w))
                region_map = np.where(unclassified & (y_pos < 0.5), 1, region_map)
                region_map = np.where(unclassified & (y_pos >= 0.5), 2, region_map)

        except Exception as e:
            logger.warning(f"SAM3 analysis failed: {e}. Using heuristic fallback.")
            return self._analyze_with_heuristics(image)

        return region_map, region_scores

    def _analyze_with_heuristics(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Analyze scene using color and texture heuristics."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        b, g, r = cv2.split(image)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        y_pos = np.arange(h)[:, np.newaxis] / h
        y_pos = np.broadcast_to(y_pos, (h, w))

        region_map = np.zeros((h, w), dtype=np.uint8)
        region_scores = {region.value: 0.0 for region in SceneRegion}

        # Surface detection
        brightness_norm = value.astype(float) / 255
        surface_mask = (brightness_norm > 0.75) & (y_pos < 0.3)
        top_brightness = np.mean(value[:h//5, :])
        mid_brightness = np.mean(value[h//3:2*h//3, :])
        if top_brightness > mid_brightness + 30:
            surface_mask = surface_mask | ((brightness_norm > 0.6) & (y_pos < 0.2))
        region_map = np.where(surface_mask, 3, region_map)
        region_scores[SceneRegion.SURFACE.value] = float(surface_mask.sum()) / (h * w)

        # Seafloor detection
        bottom_zone = y_pos > 0.55
        warm_colors = (r.astype(float) > b.astype(float) * 0.9)
        desaturated_bottom = (saturation < 100) & bottom_zone
        seafloor_mask = (bottom_zone & warm_colors) | desaturated_bottom
        seafloor_mask = seafloor_mask & (region_map == 0)
        region_map = np.where(seafloor_mask, 2, region_map)
        region_scores[SceneRegion.SEAFLOOR.value] = float(seafloor_mask.sum()) / (h * w)

        # Vegetation detection
        green_mask = (hue > 30) & (hue < 90) & (saturation > 30) & (g > b)
        green_mask = green_mask & (region_map == 0)
        region_map = np.where(green_mask, 4, region_map)
        region_scores[SceneRegion.VEGETATION.value] = float(green_mask.sum()) / (h * w)

        # Rocky detection
        gray_colors = (saturation < 60) & (value > 20) & (value < 200)
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        texture_mask = np.abs(laplacian) > 10
        rocky_mask = gray_colors & texture_mask & (region_map == 0) & (y_pos > 0.4)
        region_map = np.where(rocky_mask, 5, region_map)
        region_scores[SceneRegion.ROCKY.value] = float(rocky_mask.sum()) / (h * w)

        # Murky detection
        contrast = value.std()
        if contrast < 35:
            region_scores[SceneRegion.MURKY.value] = min(0.2 + (35 - contrast) / 100, 0.4)

        # Open water
        blue_dominant = (b.astype(float) > r.astype(float) * 1.1)
        cyan_hue = (hue > 80) & (hue < 135)
        water_mask = (blue_dominant | cyan_hue) & (region_map == 0)
        region_map = np.where(water_mask, 1, region_map)
        region_scores[SceneRegion.OPEN_WATER.value] = float(water_mask.sum()) / (h * w)

        # Remaining
        remaining = (region_map == 0)
        if remaining.sum() > 0:
            top_remaining = remaining & (y_pos < 0.4)
            bottom_remaining = remaining & (y_pos >= 0.4)
            region_map = np.where(top_remaining, 1, region_map)
            region_map = np.where(bottom_remaining, 2, region_map)
            region_scores[SceneRegion.OPEN_WATER.value] += float(top_remaining.sum()) / (h * w)
            region_scores[SceneRegion.SEAFLOOR.value] += float(bottom_remaining.sum()) / (h * w)

        # Normalize
        murky_score = region_scores.get(SceneRegion.MURKY.value, 0.0)
        other_scores = {k: v for k, v in region_scores.items() if k != SceneRegion.MURKY.value}
        total = sum(other_scores.values())
        if total > 0:
            for k in other_scores:
                region_scores[k] = other_scores[k] / total * (1 - murky_score)
        region_scores[SceneRegion.MURKY.value] = murky_score

        return region_map, region_scores

    def _compute_depth_zones(self, image: np.ndarray, region_map: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Compute vertical depth zones."""
        h, w = image.shape[:2]
        zones = {
            'surface': (0.0, 0.15),
            'upper_water': (0.15, 0.35),
            'mid_water': (0.35, 0.60),
            'lower_water': (0.60, 0.80),
            'seafloor': (0.80, 1.0),
        }

        seafloor_rows = np.where(region_map == 2)[0]
        if len(seafloor_rows) > 0:
            seafloor_start = seafloor_rows.min() / h
            if seafloor_start < 0.9:
                zones['seafloor'] = (max(0.5, seafloor_start), 1.0)
                zones['lower_water'] = (zones['mid_water'][1], seafloor_start)

        surface_rows = np.where(region_map == 3)[0]
        if len(surface_rows) > 0:
            surface_end = surface_rows.max() / h
            if surface_end > 0.05:
                zones['surface'] = (0.0, min(0.3, surface_end))
                zones['upper_water'] = (surface_end, zones['mid_water'][0])

        return zones

    def _find_placement_zones(self, region_map: np.ndarray, h: int, w: int, min_zone_size: int = 50) -> List[Tuple[int, int, int, int]]:
        """Find valid areas for object placement."""
        margin = int(min(h, w) * 0.05)
        valid_mask = np.ones((h, w), dtype=np.uint8)
        valid_mask[:margin, :] = 0
        valid_mask[-margin:, :] = 0
        valid_mask[:, :margin] = 0
        valid_mask[:, -margin:] = 0

        contours, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        zones = []
        for contour in contours:
            if cv2.contourArea(contour) > min_zone_size * min_zone_size:
                x, y, bw, bh = cv2.boundingRect(contour)
                zones.append((x, y, bw, bh))

        if not zones:
            cx, cy = w // 2, h // 2
            zones.append((cx - w // 4, cy - h // 4, w // 2, h // 2))
        return zones

    def _compute_brightness(self, image: np.ndarray) -> float:
        """Compute overall scene brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) / 255.0

    def _estimate_water_clarity(self, image: np.ndarray) -> str:
        """Estimate water clarity."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        if laplacian_var > 500 and contrast > 50:
            return 'clear'
        elif laplacian_var > 200 and contrast > 30:
            return 'moderate'
        return 'murky'

    def _estimate_color_temperature(self, image: np.ndarray) -> str:
        """Estimate color temperature."""
        b, _, r = cv2.split(image)
        ratio = b.mean() / (r.mean() + 1e-6)
        if ratio > 1.3:
            return 'cool'
        elif ratio < 0.8:
            return 'warm'
        return 'neutral'

    def check_object_scene_compatibility(
        self,
        object_class: str,
        position: Tuple[int, int],
        scene_analysis: SceneAnalysis,
        image_size: Tuple[int, int],
    ) -> Tuple[float, str]:
        """Check if object placement is compatible with scene."""
        h, w = image_size
        x, y = max(0, min(position[0], w - 1)), max(0, min(position[1], h - 1))

        region_value = scene_analysis.region_map[y, x]
        region_mapping = {
            0: SceneRegion.UNKNOWN, 1: SceneRegion.OPEN_WATER, 2: SceneRegion.SEAFLOOR,
            3: SceneRegion.SURFACE, 4: SceneRegion.VEGETATION, 5: SceneRegion.ROCKY,
            6: SceneRegion.SANDY, 7: SceneRegion.MURKY,
        }
        region = region_mapping.get(region_value, SceneRegion.UNKNOWN)

        object_key = self._normalize_object_class(object_class)
        compatibility_rules = SCENE_COMPATIBILITY.get(object_key, DEFAULT_COMPATIBILITY)
        score = compatibility_rules.get(region, DEFAULT_COMPATIBILITY.get(region, 0.5))

        if score >= 0.8:
            reason = f"{object_class} is well-suited for {region.value}"
        elif score >= 0.5:
            reason = f"{object_class} is acceptable in {region.value}"
        elif score >= self.min_compatibility_score:
            reason = f"{object_class} is marginally compatible with {region.value}"
        else:
            reason = f"{object_class} is incompatible with {region.value}"

        return score, reason

    def _normalize_object_class(self, object_class: str) -> str:
        """Normalize object class name."""
        class_lower = object_class.lower().strip()
        if class_lower in SCENE_COMPATIBILITY:
            return class_lower

        keyword_map = {
            'fish': ['fish', 'tuna', 'salmon', 'bass'],
            'shark': ['shark'],
            'jellyfish': ['jellyfish', 'jelly'],
            'plastic': ['plastic', 'wrapper'],
            'plastic_bottle': ['bottle'],
            'can': ['can', 'aluminum'],
            'glass': ['glass', 'jar'],
            'debris': ['debris', 'trash', 'waste'],
            'rope': ['rope', 'line'],
            'net': ['net', 'fishing'],
        }
        for key, keywords in keyword_map.items():
            if any(kw in class_lower for kw in keywords):
                return key
        return 'debris'

    def get_best_placement_region(self, object_class: str, scene_analysis: SceneAnalysis) -> Optional[SceneRegion]:
        """Get the best region for placing an object."""
        object_key = self._normalize_object_class(object_class)
        compatibility_rules = SCENE_COMPATIBILITY.get(object_key, DEFAULT_COMPATIBILITY)

        best_region, best_score = None, 0.0
        for region, compat_score in compatibility_rules.items():
            region_presence = scene_analysis.region_scores.get(region.value, 0.0)
            combined = compat_score * (0.5 + 0.5 * region_presence)
            if combined > best_score:
                best_score = combined
                best_region = region

        return best_region if best_score >= self.min_compatibility_score else None

    def suggest_placement_position(
        self,
        object_class: str,
        object_size: Tuple[int, int],
        scene_analysis: SceneAnalysis,
        image_size: Tuple[int, int],
        existing_positions: List[Tuple[int, int]] = None,
        min_distance: int = 50,
    ) -> Optional[Tuple[int, int]]:
        """Suggest a good position for placing an object."""
        import random

        h, w = image_size
        obj_w, obj_h = object_size
        existing = existing_positions or []

        best_region = self.get_best_placement_region(object_class, scene_analysis)
        if best_region is None:
            return None

        region_value_map = {
            SceneRegion.OPEN_WATER: 1, SceneRegion.SEAFLOOR: 2, SceneRegion.SURFACE: 3,
            SceneRegion.VEGETATION: 4, SceneRegion.ROCKY: 5, SceneRegion.SANDY: 6,
        }
        target_value = region_value_map.get(best_region, 1)
        region_mask = (scene_analysis.region_map == target_value)

        kernel = np.ones((max(1, obj_h // 2), max(1, obj_w // 2)), np.uint8)
        region_mask = cv2.erode(region_mask.astype(np.uint8), kernel, iterations=1)

        candidates = np.where(region_mask > 0)
        if len(candidates[0]) == 0:
            return self._random_valid_position(h, w, obj_w, obj_h, existing, min_distance)

        indices = list(range(len(candidates[0])))
        random.shuffle(indices)

        for idx in indices[:100]:
            y, x = candidates[0][idx], candidates[1][idx]
            too_close = any(np.sqrt((x - ex)**2 + (y - ey)**2) < min_distance for ex, ey in existing)
            if not too_close and x + obj_w <= w and y + obj_h <= h:
                return (x, y)

        return self._random_valid_position(h, w, obj_w, obj_h, existing, min_distance)

    def _random_valid_position(self, h: int, w: int, obj_w: int, obj_h: int, existing: List[Tuple[int, int]], min_distance: int) -> Optional[Tuple[int, int]]:
        """Generate a random valid position."""
        import random
        margin = 20
        for _ in range(50):
            x = random.randint(margin, max(margin + 1, w - obj_w - margin))
            y = random.randint(margin, max(margin + 1, h - obj_h - margin))
            if all(np.sqrt((x - ex)**2 + (y - ey)**2) >= min_distance for ex, ey in existing):
                return (x, y)
        return None

    # =========================================================================
    # DEBUG AND EXPLAINABILITY METHODS
    # =========================================================================

    def analyze_scene_with_debug(
        self,
        image: np.ndarray,
        save_visualization: bool = True,
        image_id: str = None,
    ) -> Tuple[SceneAnalysis, DebugInfo]:
        """Analyze scene with full debug information."""
        import time
        import os

        start_time = time.time()
        h, w = image.shape[:2]
        decision_log = []
        region_masks = {}
        region_confidences = {}
        sam3_prompts = []

        decision_log.append(f"[START] Analyzing image {h}x{w}")
        decision_log.append(f"[CONFIG] SAM3 enabled: {self.use_sam3}, Device: {self.device}")

        if self.use_sam3 and self._sam3_model is not None:
            analysis_method = "sam3"
            decision_log.append("[METHOD] Using SAM3 text-prompted segmentation")
            region_map, region_scores, masks, confidences, prompts = self._analyze_with_sam3_debug(image)
            region_masks, region_confidences, sam3_prompts = masks, confidences, prompts
        else:
            analysis_method = "heuristic"
            decision_log.append("[METHOD] Using heuristic-based analysis")
            region_map, region_scores, masks = self._analyze_with_heuristics_debug(image)
            region_masks = masks

        decision_log.append("[REGIONS] Detected region scores:")
        for region, score in sorted(region_scores.items(), key=lambda x: -x[1]):
            if score > 0.01:
                decision_log.append(f"  - {region}: {score:.2%}")

        dominant_region = max(region_scores, key=region_scores.get)
        dominant_region = SceneRegion(dominant_region) if dominant_region in [r.value for r in SceneRegion] else SceneRegion.UNKNOWN
        decision_log.append(f"[DOMINANT] {dominant_region.value}")

        depth_zones = self._compute_depth_zones(image, region_map)
        placement_zones = self._find_placement_zones(region_map, h, w)
        scene_brightness = self._compute_brightness(image)
        water_clarity = self._estimate_water_clarity(image)
        color_temperature = self._estimate_color_temperature(image)

        decision_log.append(f"[SCENE] Brightness: {scene_brightness:.2f}, Clarity: {water_clarity}, Temp: {color_temperature}")
        decision_log.append(f"[ZONES] Found {len(placement_zones)} placement zones")

        processing_time = (time.time() - start_time) * 1000
        decision_log.append(f"[END] Analysis complete in {processing_time:.1f}ms")

        analysis = SceneAnalysis(
            dominant_region=dominant_region,
            region_map=region_map,
            region_scores=region_scores,
            depth_zones=depth_zones,
            placement_zones=placement_zones,
            scene_brightness=scene_brightness,
            water_clarity=water_clarity,
            color_temperature=color_temperature,
        )

        viz_path = None
        if save_visualization and self.debug:
            image_id = image_id or f"scene_{int(time.time())}"
            viz_path = self._save_debug_visualization(image, analysis, region_masks, image_id)
            decision_log.append(f"[VIZ] Saved to {viz_path}")

        debug_info = DebugInfo(
            analysis_method=analysis_method,
            processing_time_ms=processing_time,
            region_masks=region_masks,
            region_confidences=region_confidences,
            sam3_prompts_used=sam3_prompts,
            decision_log=decision_log,
            visualization_path=viz_path,
        )

        self._last_debug_info = debug_info
        return analysis, debug_info

    def _analyze_with_sam3_debug(self, image: np.ndarray):
        """SAM3 analysis with debug output."""
        import torch
        from PIL import Image

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        region_prompts = [
            ("water", SceneRegion.OPEN_WATER), ("seafloor", SceneRegion.SEAFLOOR),
            ("water surface", SceneRegion.SURFACE), ("seaweed", SceneRegion.VEGETATION),
            ("rock", SceneRegion.ROCKY), ("sand", SceneRegion.SANDY),
        ]

        region_map = np.zeros((h, w), dtype=np.uint8)
        region_scores = {r.value: 0.0 for r in SceneRegion}
        region_masks, region_confidences = {}, {}
        confidence_map = np.zeros((h, w), dtype=np.float32)
        prompts_used = [p[0] for p in region_prompts]

        region_value_map = {
            SceneRegion.OPEN_WATER: 1, SceneRegion.SEAFLOOR: 2, SceneRegion.SURFACE: 3,
            SceneRegion.VEGETATION: 4, SceneRegion.ROCKY: 5, SceneRegion.SANDY: 6, SceneRegion.MURKY: 7,
        }

        try:
            for prompt, region_type in region_prompts:
                inputs = self._sam3_processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self._sam3_model(**inputs)
                results = self._sam3_processor.post_process_instance_segmentation(
                    outputs, threshold=0.5, mask_threshold=0.5, target_sizes=[(h, w)]
                )[0]

                combined_mask = np.zeros((h, w), dtype=np.float32)
                max_confidence = 0.0
                if 'masks' in results and len(results['masks']) > 0:
                    for mask, score in zip(results['masks'], results['scores']):
                        mask_np = mask.cpu().numpy().astype(np.float32)
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h))
                        combined_mask = np.maximum(combined_mask, mask_np * score.cpu().item())
                        max_confidence = max(max_confidence, score.cpu().item())

                region_masks[region_type.value] = (combined_mask > 0.5).astype(np.uint8) * 255
                region_confidences[region_type.value] = max_confidence

                region_value = region_value_map.get(region_type, 0)
                update_mask = (combined_mask > 0.5) & (combined_mask > confidence_map)
                region_map = np.where(update_mask, region_value, region_map)
                confidence_map = np.maximum(confidence_map, combined_mask)
                region_scores[region_type.value] = float((combined_mask > 0.5).sum()) / (h * w)

            total = sum(region_scores.values())
            if total > 0:
                region_scores = {k: v / total for k, v in region_scores.items()}

            unclassified = (region_map == 0)
            if unclassified.sum() > 0:
                y_pos = np.broadcast_to(np.arange(h)[:, np.newaxis] / h, (h, w))
                region_map = np.where(unclassified & (y_pos < 0.5), 1, region_map)
                region_map = np.where(unclassified & (y_pos >= 0.5), 2, region_map)

        except Exception as e:
            logger.warning(f"SAM3 debug analysis failed: {e}")
            region_map, region_scores, region_masks = self._analyze_with_heuristics_debug(image)

        return region_map, region_scores, region_masks, region_confidences, prompts_used

    def _analyze_with_heuristics_debug(self, image: np.ndarray):
        """Heuristic analysis with debug mask output."""
        region_map, region_scores = self._analyze_with_heuristics(image)
        region_value_map = {
            1: SceneRegion.OPEN_WATER.value, 2: SceneRegion.SEAFLOOR.value,
            3: SceneRegion.SURFACE.value, 4: SceneRegion.VEGETATION.value,
            5: SceneRegion.ROCKY.value, 6: SceneRegion.SANDY.value, 7: SceneRegion.MURKY.value,
        }
        region_masks = {name: (region_map == val).astype(np.uint8) * 255 for val, name in region_value_map.items()}
        return region_map, region_scores, region_masks

    def _save_debug_visualization(self, image: np.ndarray, analysis: SceneAnalysis, region_masks: Dict[str, np.ndarray], image_id: str) -> str:
        """Save debug visualization."""
        import os

        h, w = image.shape[:2]
        region_colors = {
            'open_water': (255, 150, 50), 'seafloor': (50, 150, 200),
            'surface': (255, 255, 200), 'vegetation': (50, 200, 50),
            'rocky': (100, 100, 100), 'sandy': (100, 180, 220),
            'murky': (80, 80, 60), 'unknown': (128, 128, 128),
        }

        region_viz = np.zeros((h, w, 3), dtype=np.uint8)
        for val, name in [(1, 'open_water'), (2, 'seafloor'), (3, 'surface'), (4, 'vegetation'), (5, 'rocky'), (6, 'sandy'), (7, 'murky')]:
            region_viz[analysis.region_map == val] = region_colors.get(name, (128, 128, 128))

        overlay = cv2.addWeighted(image, 0.6, region_viz, 0.4, 0)
        combined = np.hstack([image, overlay, region_viz])

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Overlay", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Region Map", (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)

        output_path = os.path.join(self.debug_output_dir, f"{image_id}_debug.png")
        os.makedirs(self.debug_output_dir, exist_ok=True)
        cv2.imwrite(output_path, combined)
        return output_path

    def check_object_scene_compatibility_with_debug(
        self,
        object_class: str,
        position: Tuple[int, int],
        scene_analysis: SceneAnalysis,
        image_size: Tuple[int, int],
    ) -> Tuple[float, str, PlacementDecision]:
        """Check compatibility with full debug information."""
        score, reason = self.check_object_scene_compatibility(object_class, position, scene_analysis, image_size)

        h, w = image_size
        x, y = max(0, min(position[0], w - 1)), max(0, min(position[1], h - 1))

        region_value = scene_analysis.region_map[y, x]
        region_mapping = {0: "unknown", 1: "open_water", 2: "seafloor", 3: "surface", 4: "vegetation", 5: "rocky", 6: "sandy", 7: "murky"}
        region_at_pos = region_mapping.get(region_value, "unknown")

        alternatives = []
        best_region = self.get_best_placement_region(object_class, scene_analysis)
        if best_region:
            import random
            target_value = {SceneRegion.OPEN_WATER: 1, SceneRegion.SEAFLOOR: 2, SceneRegion.SURFACE: 3, SceneRegion.VEGETATION: 4, SceneRegion.ROCKY: 5, SceneRegion.SANDY: 6}.get(best_region, 1)
            candidates = np.where(scene_analysis.region_map == target_value)
            if len(candidates[0]) > 0:
                indices = random.sample(range(len(candidates[0])), min(5, len(candidates[0])))
                for idx in indices:
                    alt_y, alt_x = candidates[0][idx], candidates[1][idx]
                    alt_score, _ = self.check_object_scene_compatibility(object_class, (alt_x, alt_y), scene_analysis, image_size)
                    alternatives.append((alt_x, alt_y, alt_score))
                alternatives.sort(key=lambda x: -x[2])

        decision = "accepted" if score >= self.min_compatibility_score else ("relocated" if alternatives and alternatives[0][2] > score else "rejected")

        placement_decision = PlacementDecision(
            object_class=object_class,
            requested_position=position,
            region_at_position=region_at_pos,
            compatibility_score=score,
            reason=reason,
            alternative_positions=alternatives,
            decision=decision,
        )
        self._placement_decisions.append(placement_decision)
        return score, reason, placement_decision

    def get_last_debug_info(self) -> Optional[DebugInfo]:
        return self._last_debug_info

    def clear_debug_state(self):
        self._last_debug_info = None
        self._placement_decisions = []
