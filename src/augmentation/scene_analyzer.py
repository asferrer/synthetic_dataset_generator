"""
Semantic Scene Analyzer
=======================

Provides semantic understanding of underwater scenes for intelligent object placement:
- Scene region classification (water column, seafloor, surface, vegetation)
- Object-scene compatibility validation
- Text-driven segmentation support (SAM3-ready)
- Heuristic fallbacks for robust operation
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SceneRegion(Enum):
    """Types of scene regions in underwater environments"""
    OPEN_WATER = "open_water"        # Mid-water column, blue/green areas
    SEAFLOOR = "seafloor"            # Bottom, sandy/rocky areas
    SURFACE = "surface"              # Near water surface, bright areas
    VEGETATION = "vegetation"        # Seaweed, coral, plants
    ROCKY = "rocky"                  # Rocky formations
    SANDY = "sandy"                  # Sandy bottom areas
    MURKY = "murky"                  # Low visibility areas
    UNKNOWN = "unknown"


@dataclass
class SceneAnalysis:
    """Results of scene analysis"""
    dominant_region: SceneRegion
    region_map: np.ndarray           # H x W map of region classifications
    region_scores: Dict[str, float]  # Confidence scores per region type
    depth_zones: Dict[str, Tuple[float, float]]  # Zone name -> (y_start, y_end) normalized
    placement_zones: List[Tuple[int, int, int, int]]  # Valid placement areas (x,y,w,h)
    scene_brightness: float          # Overall brightness 0-1
    water_clarity: str               # 'clear', 'moderate', 'murky'
    color_temperature: str           # 'warm', 'neutral', 'cool'


@dataclass
class DebugInfo:
    """Debug information for scene analysis explainability"""
    analysis_method: str             # 'sam3' or 'heuristic'
    processing_time_ms: float        # Time taken for analysis
    region_masks: Dict[str, np.ndarray]  # Individual region masks
    region_confidences: Dict[str, float]  # Per-region confidence from SAM3
    sam3_prompts_used: List[str]     # Text prompts sent to SAM3
    decision_log: List[str]          # Step-by-step decision explanations
    visualization_path: Optional[str] = None  # Path to saved visualization


@dataclass
class PlacementDecision:
    """Debug info for a single placement decision"""
    object_class: str
    requested_position: Tuple[int, int]
    region_at_position: str
    compatibility_score: float
    reason: str
    alternative_positions: List[Tuple[int, int, float]]  # (x, y, score)
    decision: str  # 'accepted', 'rejected', 'relocated'


# Object-Scene compatibility rules
# Higher score = better compatibility (0.0 to 1.0)
SCENE_COMPATIBILITY = {
    # Marine life - prefer open water
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

    # Debris - can be anywhere but some preferences
    'plastic': {
        SceneRegion.OPEN_WATER: 0.9,
        SceneRegion.SURFACE: 1.0,  # Floats
        SceneRegion.SEAFLOOR: 0.7,
        SceneRegion.VEGETATION: 0.6,
    },
    'plastic_bag': {
        SceneRegion.OPEN_WATER: 1.0,
        SceneRegion.SURFACE: 1.0,
        SceneRegion.SEAFLOOR: 0.5,
    },
    'plastic_bottle': {
        SceneRegion.SURFACE: 1.0,  # Floats
        SceneRegion.OPEN_WATER: 0.8,
        SceneRegion.SEAFLOOR: 0.6,
    },
    'bottle': {
        SceneRegion.SURFACE: 0.9,
        SceneRegion.OPEN_WATER: 0.7,
        SceneRegion.SEAFLOOR: 0.8,
    },

    # Metal objects - sink to bottom
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

    # Glass - sinks
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

    # Generic debris
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

# Default compatibility for unknown objects
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

    Provides:
    - Scene region classification (water column, seafloor, etc.)
    - Object-scene compatibility scoring
    - Intelligent placement zone suggestions
    - SAM3 integration when available (with heuristic fallback)
    """

    def __init__(
        self,
        use_sam3: bool = False,
        sam3_model_path: Optional[str] = None,
        min_compatibility_score: float = 0.4,
        use_gpu: bool = True,
        debug: bool = False,
        debug_output_dir: Optional[str] = None,
    ):
        """
        Initialize scene analyzer.

        Args:
            use_sam3: Enable SAM3-based segmentation (experimental)
            sam3_model_path: Path to SAM3 model checkpoint
            min_compatibility_score: Minimum score to allow placement
            use_gpu: Use GPU for SAM3 inference
            debug: Enable debug mode with detailed logging and visualizations
            debug_output_dir: Directory to save debug visualizations
        """
        self.use_sam3 = use_sam3
        self.sam3_model_path = sam3_model_path
        self.min_compatibility_score = min_compatibility_score
        self.device = 'cuda' if use_gpu else 'cpu'
        self.debug = debug
        self.debug_output_dir = debug_output_dir or "./debug_output"

        # SAM3 model (lazy loaded)
        self._sam3_model = None
        self._sam3_processor = None

        # Debug state
        self._last_debug_info: Optional[DebugInfo] = None
        self._placement_decisions: List[PlacementDecision] = []

        if use_sam3:
            self._init_sam3()

        if debug:
            import os
            os.makedirs(self.debug_output_dir, exist_ok=True)
            logger.setLevel(logging.DEBUG)

        logger.info(f"SemanticSceneAnalyzer initialized (SAM3: {self.use_sam3}, Debug: {self.debug})")

    @staticmethod
    def is_sam3_available() -> bool:
        """Check if SAM3 model and dependencies are available."""
        try:
            from transformers import Sam3Processor, Sam3Model
            return True
        except ImportError:
            return False

    def _init_sam3(self):
        """Initialize SAM3 model if available"""
        try:
            # SAM3 (Segment Anything Model 3) from Meta - released 2025-11-19
            # Available via transformers library with text-prompted segmentation
            from transformers import Sam3Processor, Sam3Model
            import torch

            if self.device == 'cuda' and not torch.cuda.is_available():
                self.device = 'cpu'
                logger.warning("CUDA not available, using CPU for SAM3")

            # Official model: facebook/sam3 (848M-0.9B parameters)
            model_id = self.sam3_model_path or "facebook/sam3"

            logger.info(f"Loading SAM3 model: {model_id}...")
            self._sam3_processor = Sam3Processor.from_pretrained(model_id)
            self._sam3_model = Sam3Model.from_pretrained(model_id).to(self.device)
            self._sam3_model.eval()

            logger.info(f"SAM3 model loaded successfully: {model_id}")

        except ImportError as e:
            logger.warning(f"SAM3 not available (transformers version may not support it): {e}")
            logger.info("Install with: pip install transformers>=4.45.0")
            self.use_sam3 = False
        except Exception as e:
            logger.warning(f"Failed to load SAM3: {e}. Using heuristic fallback.")
            self.use_sam3 = False

    def analyze_scene(self, image: np.ndarray) -> SceneAnalysis:
        """
        Analyze scene to identify regions and characteristics.

        Args:
            image: BGR input image

        Returns:
            SceneAnalysis with region map and characteristics
        """
        h, w = image.shape[:2]

        # Use SAM3 if available, otherwise fall back to heuristics
        if self.use_sam3 and self._sam3_model is not None:
            region_map, region_scores = self._analyze_with_sam3(image)
        else:
            region_map, region_scores = self._analyze_with_heuristics(image)

        # Determine dominant region
        dominant_region = max(region_scores, key=region_scores.get)
        dominant_region = SceneRegion(dominant_region) if dominant_region in [r.value for r in SceneRegion] else SceneRegion.UNKNOWN

        # Compute depth zones (vertical divisions)
        depth_zones = self._compute_depth_zones(image, region_map)

        # Find valid placement zones
        placement_zones = self._find_placement_zones(region_map, h, w)

        # Analyze scene characteristics
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

    def _analyze_with_sam3(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Analyze scene using SAM3 text-driven segmentation.

        SAM3 uses Promptable Concept Segmentation (PCS) to segment
        all instances in an image that match a given text concept.
        """
        import torch
        from PIL import Image

        h, w = image.shape[:2]

        # Convert BGR to RGB PIL Image (required by SAM3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Text prompts for underwater scene regions
        # SAM3 works best with simple, clear concept descriptions
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

        try:
            for prompt, region_type in region_prompts:
                # Process with SAM3 text-prompted segmentation
                inputs = self._sam3_processor(
                    images=pil_image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._sam3_model(**inputs)

                # Post-process to get instance segmentation results
                results = self._sam3_processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=[(h, w)]
                )[0]

                # Combine all masks for this region type
                combined_mask = np.zeros((h, w), dtype=np.float32)
                total_score = 0.0

                if 'masks' in results and len(results['masks']) > 0:
                    for mask, score in zip(results['masks'], results['scores']):
                        mask_np = mask.cpu().numpy().astype(np.float32)
                        score_val = score.cpu().item()

                        # Resize if needed
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h))

                        # Weighted combination
                        combined_mask = np.maximum(combined_mask, mask_np * score_val)
                        total_score = max(total_score, score_val)

                # Get region value for region map
                region_value_map = {
                    SceneRegion.OPEN_WATER: 1,
                    SceneRegion.SEAFLOOR: 2,
                    SceneRegion.SURFACE: 3,
                    SceneRegion.VEGETATION: 4,
                    SceneRegion.ROCKY: 5,
                    SceneRegion.SANDY: 6,
                    SceneRegion.MURKY: 7,
                }
                region_value = region_value_map.get(region_type, 0)

                # Update region map where this mask has higher confidence
                update_mask = (combined_mask > 0.5) & (combined_mask > confidence_map)
                region_map = np.where(update_mask, region_value, region_map)
                confidence_map = np.maximum(confidence_map, combined_mask)

                # Store region coverage score
                coverage = float((combined_mask > 0.5).sum()) / (h * w)
                region_scores[region_type.value] = coverage

            # Normalize scores
            total = sum(region_scores.values())
            if total > 0:
                for key in region_scores:
                    region_scores[key] = region_scores[key] / total

            # Fill unclassified areas with heuristic-based assignment
            unclassified = (region_map == 0)
            if unclassified.sum() > 0:
                # Use vertical position to assign unclassified pixels
                y_pos = np.arange(h)[:, np.newaxis] / h
                y_pos = np.broadcast_to(y_pos, (h, w))

                # Top -> water, bottom -> seafloor
                top_unclassified = unclassified & (y_pos < 0.5)
                bottom_unclassified = unclassified & (y_pos >= 0.5)
                region_map = np.where(top_unclassified, 1, region_map)  # OPEN_WATER
                region_map = np.where(bottom_unclassified, 2, region_map)  # SEAFLOOR

            logger.debug(f"SAM3 analysis complete. Region scores: {region_scores}")

        except Exception as e:
            logger.warning(f"SAM3 analysis failed: {e}. Using heuristic fallback.")
            return self._analyze_with_heuristics(image)

        return region_map, region_scores

    def _analyze_with_heuristics(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Analyze scene using color and texture heuristics for underwater scenes"""
        h, w = image.shape[:2]

        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract channels
        b, g, r = cv2.split(image)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Create vertical position matrix (0 at top, 1 at bottom)
        y_pos = np.arange(h)[:, np.newaxis] / h
        y_pos = np.broadcast_to(y_pos, (h, w))

        # Initialize region map
        region_map = np.zeros((h, w), dtype=np.uint8)
        region_scores = {region.value: 0.0 for region in SceneRegion}

        # 1. Surface detection: bright areas at top of image
        # High brightness, top 30% of image
        brightness_norm = value.astype(float) / 255
        surface_mask = (brightness_norm > 0.75) & (y_pos < 0.3)
        # Also detect bright gradient at top
        top_brightness = np.mean(value[:h//5, :])
        mid_brightness = np.mean(value[h//3:2*h//3, :])
        if top_brightness > mid_brightness + 30:
            # Extend surface detection when there's clear brightness gradient
            surface_mask = surface_mask | ((brightness_norm > 0.6) & (y_pos < 0.2))
        region_map = np.where(surface_mask, 3, region_map)  # SURFACE = 3
        region_scores[SceneRegion.SURFACE.value] = float(surface_mask.sum()) / (h * w)

        # 2. Seafloor detection: bottom area with warmer/sandy colors
        # Seafloor typically has: more red than blue, lower in image
        bottom_zone = y_pos > 0.55
        # Sandy/warm colors: red > blue, not too saturated
        warm_colors = (r.astype(float) > b.astype(float) * 0.9)
        # Or low saturation in bottom area (gray sandy bottom)
        desaturated_bottom = (saturation < 100) & bottom_zone
        seafloor_mask = (bottom_zone & warm_colors) | desaturated_bottom
        # Exclude areas already marked as surface
        seafloor_mask = seafloor_mask & (region_map == 0)
        region_map = np.where(seafloor_mask, 2, region_map)  # SEAFLOOR = 2
        region_scores[SceneRegion.SEAFLOOR.value] = float(seafloor_mask.sum()) / (h * w)

        # 3. Vegetation detection: green hues anywhere in image
        # Green in HSV: hue 35-85 (cyan-ish green to yellow-green)
        green_mask = (hue > 30) & (hue < 90) & (saturation > 30) & (g > b)
        green_mask = green_mask & (region_map == 0)
        region_map = np.where(green_mask, 4, region_map)  # VEGETATION = 4
        region_scores[SceneRegion.VEGETATION.value] = float(green_mask.sum()) / (h * w)

        # 4. Rocky detection: textured areas with low saturation
        gray_colors = (saturation < 60) & (value > 20) & (value < 200)
        # Texture analysis using Laplacian variance
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
        texture_mask = np.abs(laplacian) > 10
        rocky_mask = gray_colors & texture_mask & (region_map == 0)
        # Weight by position (rocks more likely at bottom)
        rocky_weight = y_pos > 0.4
        rocky_mask = rocky_mask & rocky_weight
        region_map = np.where(rocky_mask, 5, region_map)  # ROCKY = 5
        region_scores[SceneRegion.ROCKY.value] = float(rocky_mask.sum()) / (h * w)

        # 5. Murky detection: low contrast overall
        contrast = value.std()
        if contrast < 35:
            murky_score = 0.2 + (35 - contrast) / 100
            region_scores[SceneRegion.MURKY.value] = min(murky_score, 0.4)

        # 6. Open water: blue/cyan areas not yet classified
        # Blue dominant (B > R) or cyan (high hue 80-130)
        blue_dominant = (b.astype(float) > r.astype(float) * 1.1)
        cyan_hue = (hue > 80) & (hue < 135)
        water_mask = (blue_dominant | cyan_hue) & (region_map == 0)
        region_map = np.where(water_mask, 1, region_map)  # OPEN_WATER = 1
        region_scores[SceneRegion.OPEN_WATER.value] = float(water_mask.sum()) / (h * w)

        # 7. Remaining unclassified areas
        remaining = (region_map == 0)
        remaining_count = float(remaining.sum()) / (h * w)
        if remaining_count > 0:
            # Assign based on vertical position
            top_remaining = remaining & (y_pos < 0.4)
            bottom_remaining = remaining & (y_pos >= 0.4)
            region_map = np.where(top_remaining, 1, region_map)  # Top -> open water
            region_map = np.where(bottom_remaining, 2, region_map)  # Bottom -> seafloor
            region_scores[SceneRegion.OPEN_WATER.value] += float(top_remaining.sum()) / (h * w)
            region_scores[SceneRegion.SEAFLOOR.value] += float(bottom_remaining.sum()) / (h * w)

        # Normalize scores (excluding murky which is scene-wide)
        murky_score = region_scores.get(SceneRegion.MURKY.value, 0.0)
        other_scores = {k: v for k, v in region_scores.items() if k != SceneRegion.MURKY.value}
        total = sum(other_scores.values())
        if total > 0:
            for k in other_scores:
                region_scores[k] = other_scores[k] / total * (1 - murky_score)
        region_scores[SceneRegion.MURKY.value] = murky_score

        return region_map, region_scores

    def _compute_depth_zones(
        self,
        image: np.ndarray,
        region_map: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """Compute vertical depth zones"""
        h, w = image.shape[:2]

        # Default zones based on typical underwater scene structure
        zones = {
            'surface': (0.0, 0.15),      # Top 15%
            'upper_water': (0.15, 0.35),  # 15-35%
            'mid_water': (0.35, 0.60),    # 35-60%
            'lower_water': (0.60, 0.80),  # 60-80%
            'seafloor': (0.80, 1.0),      # Bottom 20%
        }

        # Adjust based on region map analysis
        # Find where seafloor actually starts
        seafloor_rows = np.where(region_map == 2)[0]  # SEAFLOOR = 2
        if len(seafloor_rows) > 0:
            seafloor_start = seafloor_rows.min() / h
            if seafloor_start < 0.9:  # Seafloor visible in image
                zones['seafloor'] = (max(0.5, seafloor_start), 1.0)
                zones['lower_water'] = (zones['mid_water'][1], seafloor_start)

        # Find surface region
        surface_rows = np.where(region_map == 3)[0]  # SURFACE = 3
        if len(surface_rows) > 0:
            surface_end = surface_rows.max() / h
            if surface_end > 0.05:
                zones['surface'] = (0.0, min(0.3, surface_end))
                zones['upper_water'] = (surface_end, zones['mid_water'][0])

        return zones

    def _find_placement_zones(
        self,
        region_map: np.ndarray,
        h: int,
        w: int,
        min_zone_size: int = 50
    ) -> List[Tuple[int, int, int, int]]:
        """Find valid areas for object placement"""
        zones = []

        # Find connected regions that are suitable for placement
        # Exclude very edge areas
        margin = int(min(h, w) * 0.05)

        valid_mask = np.ones((h, w), dtype=np.uint8)
        valid_mask[:margin, :] = 0
        valid_mask[-margin:, :] = 0
        valid_mask[:, :margin] = 0
        valid_mask[:, -margin:] = 0

        # Find contours of valid regions
        contours, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_zone_size * min_zone_size:
                x, y, bw, bh = cv2.boundingRect(contour)
                zones.append((x, y, bw, bh))

        # If no zones found, use center of image
        if not zones:
            cx, cy = w // 2, h // 2
            zone_w, zone_h = w // 2, h // 2
            zones.append((cx - zone_w // 2, cy - zone_h // 2, zone_w, zone_h))

        return zones

    def _compute_brightness(self, image: np.ndarray) -> float:
        """Compute overall scene brightness (0-1)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) / 255.0

    def _estimate_water_clarity(self, image: np.ndarray) -> str:
        """Estimate water clarity based on contrast and color"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use Laplacian variance as sharpness indicator
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Use contrast
        contrast = gray.std()

        if laplacian_var > 500 and contrast > 50:
            return 'clear'
        elif laplacian_var > 200 and contrast > 30:
            return 'moderate'
        else:
            return 'murky'

    def _estimate_color_temperature(self, image: np.ndarray) -> str:
        """Estimate color temperature (warm/neutral/cool)"""
        # Average BGR values
        b, g, r = cv2.split(image)
        avg_b, avg_r = b.mean(), r.mean()

        ratio = avg_b / (avg_r + 1e-6)

        if ratio > 1.3:
            return 'cool'  # More blue - typical underwater
        elif ratio < 0.8:
            return 'warm'  # More red - shallow/sunset
        else:
            return 'neutral'

    def check_object_scene_compatibility(
        self,
        object_class: str,
        position: Tuple[int, int],
        scene_analysis: SceneAnalysis,
        image_size: Tuple[int, int],
    ) -> Tuple[float, str]:
        """
        Check if object placement is compatible with scene.

        Args:
            object_class: Object category name
            position: (x, y) position in image
            scene_analysis: Result from analyze_scene()
            image_size: (height, width) of image

        Returns:
            Tuple of (compatibility_score, reason)
        """
        h, w = image_size
        x, y = position

        # Clamp position to valid range
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        # Get region at position
        region_value = scene_analysis.region_map[y, x]

        # Map region value to SceneRegion enum
        region_mapping = {
            0: SceneRegion.UNKNOWN,
            1: SceneRegion.OPEN_WATER,
            2: SceneRegion.SEAFLOOR,
            3: SceneRegion.SURFACE,
            4: SceneRegion.VEGETATION,
            5: SceneRegion.ROCKY,
            6: SceneRegion.SANDY,
            7: SceneRegion.MURKY,
        }
        region = region_mapping.get(region_value, SceneRegion.UNKNOWN)

        # Get compatibility rules for this object class
        object_key = self._normalize_object_class(object_class)
        compatibility_rules = SCENE_COMPATIBILITY.get(object_key, DEFAULT_COMPATIBILITY)

        # Get score for this region
        score = compatibility_rules.get(region, DEFAULT_COMPATIBILITY.get(region, 0.5))

        # Generate reason
        if score >= 0.8:
            reason = f"{object_class} is well-suited for {region.value}"
        elif score >= 0.5:
            reason = f"{object_class} is acceptable in {region.value}"
        elif score >= self.min_compatibility_score:
            reason = f"{object_class} is marginally compatible with {region.value}"
        else:
            reason = f"{object_class} is incompatible with {region.value} (e.g., heavy object floating)"

        return score, reason

    def _normalize_object_class(self, object_class: str) -> str:
        """Normalize object class name for compatibility lookup"""
        class_lower = object_class.lower().strip()

        # Direct match
        if class_lower in SCENE_COMPATIBILITY:
            return class_lower

        # Keyword matching
        keyword_map = {
            'fish': ['fish', 'tuna', 'salmon', 'bass', 'trout', 'anchovy', 'sardine'],
            'shark': ['shark', 'tiburon'],
            'jellyfish': ['jellyfish', 'jelly', 'medusa'],
            'plastic': ['plastic', 'wrapper', 'film', 'bag'],
            'plastic_bottle': ['bottle', 'botella'],
            'can': ['can', 'lata', 'aluminum', 'aluminio'],
            'glass': ['glass', 'vidrio', 'jar'],
            'debris': ['debris', 'basura', 'trash', 'waste', 'garbage'],
            'rope': ['rope', 'cuerda', 'line', 'string'],
            'net': ['net', 'red', 'fishing'],
        }

        for key, keywords in keyword_map.items():
            if any(kw in class_lower for kw in keywords):
                return key

        return 'debris'  # Default fallback

    def get_best_placement_region(
        self,
        object_class: str,
        scene_analysis: SceneAnalysis,
    ) -> Optional[SceneRegion]:
        """
        Get the best region for placing an object.

        Args:
            object_class: Object category name
            scene_analysis: Scene analysis result

        Returns:
            Best SceneRegion for this object, or None if all incompatible
        """
        object_key = self._normalize_object_class(object_class)
        compatibility_rules = SCENE_COMPATIBILITY.get(object_key, DEFAULT_COMPATIBILITY)

        # Weight by both compatibility and region presence in scene
        best_region = None
        best_score = 0.0

        for region, compat_score in compatibility_rules.items():
            # Get how much of this region exists in the scene
            region_presence = scene_analysis.region_scores.get(region.value, 0.0)

            # Combined score
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
        """
        Suggest a good position for placing an object.

        Args:
            object_class: Object category name
            object_size: (width, height) of object
            scene_analysis: Scene analysis result
            image_size: (height, width) of image
            existing_positions: List of already used positions
            min_distance: Minimum distance from existing objects

        Returns:
            Suggested (x, y) position, or None if no good position found
        """
        h, w = image_size
        obj_w, obj_h = object_size
        existing = existing_positions or []

        # Get best region for this object
        best_region = self.get_best_placement_region(object_class, scene_analysis)
        if best_region is None:
            return None

        # Map region to value
        region_value_map = {
            SceneRegion.OPEN_WATER: 1,
            SceneRegion.SEAFLOOR: 2,
            SceneRegion.SURFACE: 3,
            SceneRegion.VEGETATION: 4,
            SceneRegion.ROCKY: 5,
            SceneRegion.SANDY: 6,
            SceneRegion.MURKY: 7,
        }
        target_value = region_value_map.get(best_region, 1)

        # Find pixels in target region
        region_mask = (scene_analysis.region_map == target_value)

        # Erode to avoid edges
        kernel = np.ones((obj_h // 2, obj_w // 2), np.uint8)
        region_mask = cv2.erode(region_mask.astype(np.uint8), kernel, iterations=1)

        # Get candidate positions
        candidates = np.where(region_mask > 0)
        if len(candidates[0]) == 0:
            # Fall back to any valid position
            return self._random_valid_position(h, w, obj_w, obj_h, existing, min_distance)

        # Sample random positions from candidates
        import random
        indices = list(range(len(candidates[0])))
        random.shuffle(indices)

        for idx in indices[:100]:  # Check up to 100 candidates
            y, x = candidates[0][idx], candidates[1][idx]

            # Check distance from existing positions
            too_close = False
            for ex, ey in existing:
                dist = np.sqrt((x - ex)**2 + (y - ey)**2)
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                # Ensure object fits in image
                if x + obj_w <= w and y + obj_h <= h:
                    return (x, y)

        # Fallback
        return self._random_valid_position(h, w, obj_w, obj_h, existing, min_distance)

    def _random_valid_position(
        self,
        h: int,
        w: int,
        obj_w: int,
        obj_h: int,
        existing: List[Tuple[int, int]],
        min_distance: int,
    ) -> Optional[Tuple[int, int]]:
        """Generate a random valid position"""
        import random

        margin = 20
        for _ in range(50):  # Try up to 50 times
            x = random.randint(margin, w - obj_w - margin)
            y = random.randint(margin, h - obj_h - margin)

            # Check distance
            valid = True
            for ex, ey in existing:
                if np.sqrt((x - ex)**2 + (y - ey)**2) < min_distance:
                    valid = False
                    break

            if valid:
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
        """
        Analyze scene with full debug information for explainability.

        Args:
            image: BGR input image
            save_visualization: Whether to save visualization to disk
            image_id: Unique identifier for this image (for filenames)

        Returns:
            Tuple of (SceneAnalysis, DebugInfo)
        """
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

        # Determine analysis method
        if self.use_sam3 and self._sam3_model is not None:
            analysis_method = "sam3"
            decision_log.append("[METHOD] Using SAM3 text-prompted segmentation")

            # Get detailed SAM3 analysis
            region_map, region_scores, masks, confidences, prompts = self._analyze_with_sam3_debug(image)
            region_masks = masks
            region_confidences = confidences
            sam3_prompts = prompts
        else:
            analysis_method = "heuristic"
            decision_log.append("[METHOD] Using heuristic-based analysis (SAM3 not available)")
            region_map, region_scores, masks = self._analyze_with_heuristics_debug(image)
            region_masks = masks

        # Log region detection results
        decision_log.append(f"[REGIONS] Detected region scores:")
        for region, score in sorted(region_scores.items(), key=lambda x: -x[1]):
            if score > 0.01:
                decision_log.append(f"  - {region}: {score:.2%}")

        # Determine dominant region
        dominant_region = max(region_scores, key=region_scores.get)
        dominant_region = SceneRegion(dominant_region) if dominant_region in [r.value for r in SceneRegion] else SceneRegion.UNKNOWN
        decision_log.append(f"[DOMINANT] {dominant_region.value}")

        # Compute additional analysis
        depth_zones = self._compute_depth_zones(image, region_map)
        placement_zones = self._find_placement_zones(region_map, h, w)
        scene_brightness = self._compute_brightness(image)
        water_clarity = self._estimate_water_clarity(image)
        color_temperature = self._estimate_color_temperature(image)

        decision_log.append(f"[SCENE] Brightness: {scene_brightness:.2f}, Clarity: {water_clarity}, Temp: {color_temperature}")
        decision_log.append(f"[ZONES] Found {len(placement_zones)} placement zones")

        processing_time = (time.time() - start_time) * 1000
        decision_log.append(f"[END] Analysis complete in {processing_time:.1f}ms")

        # Create analysis result
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

        # Save visualization if requested
        viz_path = None
        if save_visualization and self.debug:
            image_id = image_id or f"scene_{int(time.time())}"
            viz_path = self.generate_debug_visualization(
                image, analysis, region_masks, image_id
            )
            decision_log.append(f"[VIZ] Saved to {viz_path}")

        # Create debug info
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

    def _analyze_with_sam3_debug(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray], Dict[str, float], List[str]]:
        """SAM3 analysis with detailed debug output"""
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
        region_masks = {}
        region_confidences = {}
        confidence_map = np.zeros((h, w), dtype=np.float32)
        prompts_used = [p[0] for p in region_prompts]

        try:
            for prompt, region_type in region_prompts:
                inputs = self._sam3_processor(
                    images=pil_image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

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
                        score_val = score.cpu().item()
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(mask_np, (w, h))
                        combined_mask = np.maximum(combined_mask, mask_np * score_val)
                        max_confidence = max(max_confidence, score_val)

                # Store individual mask for debug
                region_masks[region_type.value] = (combined_mask > 0.5).astype(np.uint8) * 255
                region_confidences[region_type.value] = max_confidence

                # Update region map
                region_value_map = {
                    SceneRegion.OPEN_WATER: 1, SceneRegion.SEAFLOOR: 2, SceneRegion.SURFACE: 3,
                    SceneRegion.VEGETATION: 4, SceneRegion.ROCKY: 5, SceneRegion.SANDY: 6, SceneRegion.MURKY: 7,
                }
                region_value = region_value_map.get(region_type, 0)
                update_mask = (combined_mask > 0.5) & (combined_mask > confidence_map)
                region_map = np.where(update_mask, region_value, region_map)
                confidence_map = np.maximum(confidence_map, combined_mask)

                coverage = float((combined_mask > 0.5).sum()) / (h * w)
                region_scores[region_type.value] = coverage

            # Normalize scores
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
            logger.warning(f"SAM3 debug analysis failed: {e}")
            region_map, region_scores, region_masks = self._analyze_with_heuristics_debug(image)

        return region_map, region_scores, region_masks, region_confidences, prompts_used

    def _analyze_with_heuristics_debug(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
        """Heuristic analysis with debug mask output"""
        h, w = image.shape[:2]
        region_masks = {}

        # Run standard heuristic analysis
        region_map, region_scores = self._analyze_with_heuristics(image)

        # Extract individual masks for debug visualization
        region_value_map = {
            1: SceneRegion.OPEN_WATER.value,
            2: SceneRegion.SEAFLOOR.value,
            3: SceneRegion.SURFACE.value,
            4: SceneRegion.VEGETATION.value,
            5: SceneRegion.ROCKY.value,
            6: SceneRegion.SANDY.value,
            7: SceneRegion.MURKY.value,
        }

        for value, region_name in region_value_map.items():
            mask = (region_map == value).astype(np.uint8) * 255
            region_masks[region_name] = mask

        return region_map, region_scores, region_masks

    def generate_debug_visualization(
        self,
        image: np.ndarray,
        analysis: SceneAnalysis,
        region_masks: Dict[str, np.ndarray],
        image_id: str,
    ) -> str:
        """
        Generate debug visualization with colored region overlay.

        Args:
            image: Original BGR image
            analysis: Scene analysis result
            region_masks: Individual region masks
            image_id: Unique identifier for filename

        Returns:
            Path to saved visualization
        """
        import os

        h, w = image.shape[:2]

        # Color map for regions (BGR format)
        region_colors = {
            SceneRegion.OPEN_WATER.value: (255, 150, 50),    # Blue
            SceneRegion.SEAFLOOR.value: (50, 150, 200),      # Sandy brown
            SceneRegion.SURFACE.value: (255, 255, 200),      # Light cyan
            SceneRegion.VEGETATION.value: (50, 200, 50),     # Green
            SceneRegion.ROCKY.value: (100, 100, 100),        # Gray
            SceneRegion.SANDY.value: (100, 180, 220),        # Sand
            SceneRegion.MURKY.value: (80, 80, 60),           # Dark murky
            SceneRegion.UNKNOWN.value: (128, 128, 128),      # Neutral gray
        }

        # Create colored region overlay
        overlay = image.copy()
        for region_name, color in region_colors.items():
            if region_name in region_masks:
                mask = region_masks[region_name]
                if mask.max() > 0:
                    colored = np.zeros_like(image)
                    colored[:] = color
                    mask_3ch = np.stack([mask, mask, mask], axis=-1) / 255.0
                    overlay = (overlay * (1 - mask_3ch * 0.4) + colored * mask_3ch * 0.4).astype(np.uint8)

        # Create comparison image: original | overlay | region_map
        region_viz = np.zeros((h, w, 3), dtype=np.uint8)
        for value, region_name in [(1, 'open_water'), (2, 'seafloor'), (3, 'surface'),
                                    (4, 'vegetation'), (5, 'rocky'), (6, 'sandy'), (7, 'murky')]:
            mask = (analysis.region_map == value)
            color = region_colors.get(region_name, (128, 128, 128))
            region_viz[mask] = color

        # Stack images horizontally
        combined = np.hstack([image, overlay, region_viz])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Overlay", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Region Map", (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)

        # Add region scores legend
        y_offset = 60
        cv2.putText(combined, f"Dominant: {analysis.dominant_region.value}", (2*w + 10, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 25
        for region, score in sorted(analysis.region_scores.items(), key=lambda x: -x[1])[:5]:
            if score > 0.01:
                color = region_colors.get(region, (255, 255, 255))
                cv2.putText(combined, f"{region}: {score:.1%}", (2*w + 10, y_offset), font, 0.4, color, 1)
                y_offset += 20

        # Save
        output_path = os.path.join(self.debug_output_dir, f"{image_id}_debug.png")
        cv2.imwrite(output_path, combined)

        # Also save individual region masks
        masks_dir = os.path.join(self.debug_output_dir, f"{image_id}_masks")
        os.makedirs(masks_dir, exist_ok=True)
        for region_name, mask in region_masks.items():
            mask_path = os.path.join(masks_dir, f"{region_name}.png")
            cv2.imwrite(mask_path, mask)

        return output_path

    def export_debug_report(self, image_id: str = None) -> Dict:
        """
        Export full debug report as dictionary (JSON-serializable).

        Returns:
            Dictionary with complete debug information
        """
        if self._last_debug_info is None:
            return {"error": "No debug info available. Run analyze_scene_with_debug first."}

        def to_native(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [to_native(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            return obj

        debug = self._last_debug_info
        image_id = image_id or "unknown"

        report = {
            "image_id": image_id,
            "analysis_method": debug.analysis_method,
            "processing_time_ms": float(debug.processing_time_ms),
            "sam3_prompts_used": debug.sam3_prompts_used,
            "region_confidences": to_native(debug.region_confidences),
            "decision_log": debug.decision_log,
            "visualization_path": debug.visualization_path,
            "placement_decisions": [
                {
                    "object_class": d.object_class,
                    "requested_position": to_native(d.requested_position),
                    "region_at_position": d.region_at_position,
                    "compatibility_score": float(d.compatibility_score),
                    "reason": d.reason,
                    "decision": d.decision,
                    "alternatives": to_native(d.alternative_positions[:3]),  # Top 3 alternatives
                }
                for d in self._placement_decisions
            ],
        }

        # Save to JSON file if debug is enabled
        if self.debug:
            import json
            import os
            report_path = os.path.join(self.debug_output_dir, f"{image_id}_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            report["report_path"] = report_path

        return report

    def check_object_scene_compatibility_with_debug(
        self,
        object_class: str,
        position: Tuple[int, int],
        scene_analysis: SceneAnalysis,
        image_size: Tuple[int, int],
    ) -> Tuple[float, str, PlacementDecision]:
        """
        Check compatibility with full debug information.

        Returns:
            Tuple of (score, reason, PlacementDecision)
        """
        score, reason = self.check_object_scene_compatibility(
            object_class, position, scene_analysis, image_size
        )

        h, w = image_size
        x, y = position
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        # Get region at position
        region_value = scene_analysis.region_map[y, x]
        region_mapping = {
            0: "unknown", 1: "open_water", 2: "seafloor", 3: "surface",
            4: "vegetation", 5: "rocky", 6: "sandy", 7: "murky",
        }
        region_at_pos = region_mapping.get(region_value, "unknown")

        # Find alternative positions
        alternatives = []
        best_region = self.get_best_placement_region(object_class, scene_analysis)
        if best_region:
            # Sample some positions in the best region
            target_value = {
                SceneRegion.OPEN_WATER: 1, SceneRegion.SEAFLOOR: 2, SceneRegion.SURFACE: 3,
                SceneRegion.VEGETATION: 4, SceneRegion.ROCKY: 5, SceneRegion.SANDY: 6,
            }.get(best_region, 1)

            region_mask = (scene_analysis.region_map == target_value)
            candidates = np.where(region_mask)
            if len(candidates[0]) > 0:
                import random
                indices = random.sample(range(len(candidates[0])), min(5, len(candidates[0])))
                for idx in indices:
                    alt_y, alt_x = candidates[0][idx], candidates[1][idx]
                    alt_score, _ = self.check_object_scene_compatibility(
                        object_class, (alt_x, alt_y), scene_analysis, image_size
                    )
                    alternatives.append((alt_x, alt_y, alt_score))
                alternatives.sort(key=lambda x: -x[2])

        # Determine decision
        if score >= self.min_compatibility_score:
            decision = "accepted"
        elif alternatives and alternatives[0][2] > score:
            decision = "relocated"
        else:
            decision = "rejected"

        placement_decision = PlacementDecision(
            object_class=object_class,
            requested_position=position,
            region_at_position=region_at_pos,
            compatibility_score=score,
            reason=reason,
            alternative_positions=alternatives,
            decision=decision,
        )

        # Store for report
        self._placement_decisions.append(placement_decision)

        return score, reason, placement_decision

    def get_last_debug_info(self) -> Optional[DebugInfo]:
        """Get the last debug info from analyze_scene_with_debug"""
        return self._last_debug_info

    def clear_debug_state(self):
        """Clear accumulated debug state"""
        self._last_debug_info = None
        self._placement_decisions = []


# Convenience function for quick compatibility check
def is_placement_valid(
    object_class: str,
    position: Tuple[int, int],
    image: np.ndarray,
    min_score: float = 0.4,
) -> Tuple[bool, float, str]:
    """
    Quick check if object placement is valid.

    Args:
        object_class: Object category name
        position: (x, y) position
        image: BGR image
        min_score: Minimum compatibility score

    Returns:
        Tuple of (is_valid, score, reason)
    """
    analyzer = SemanticSceneAnalyzer(use_sam3=False)
    analysis = analyzer.analyze_scene(image)
    score, reason = analyzer.check_object_scene_compatibility(
        object_class,
        position,
        analysis,
        image.shape[:2]
    )
    return score >= min_score, score, reason
