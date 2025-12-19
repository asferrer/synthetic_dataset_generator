"""
Image Composer Module
=====================
Handles synthetic image composition with depth-aware placement,
multi-light shadows, and realistic effects.

Migrated from SyntheticDataAugmentor for microservice architecture.
"""

import os
import cv2
import json
import glob
import random
import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import httpx

from app.models.schemas import (
    ObjectPlacement,
    AnnotationBox,
    EffectsConfig,
    EffectType,
    LightingInfo,
    LightSourceInfo,
    LightType,
    DepthZone,
    WaterClarity,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComposeResult:
    """Result from single image composition"""
    annotations: List[AnnotationBox] = field(default_factory=list)
    objects_placed: int = 0
    depth_used: bool = False
    effects_applied: List[str] = field(default_factory=list)


@dataclass
class BatchResult:
    """Result from batch composition"""
    images_generated: int = 0
    images_rejected: int = 0
    synthetic_counts: Dict[str, int] = field(default_factory=dict)
    output_coco_path: Optional[str] = None


# =============================================================================
# Real-World Object Sizes (in meters)
# =============================================================================
# These are approximate real-world sizes for common underwater objects.
# Used to calculate realistic scaling based on depth/distance.

REAL_WORLD_SIZES = {
    # Marine life
    'fish': 0.25,              # Average small-medium fish
    'small_fish': 0.10,
    'large_fish': 0.60,
    'shark': 2.5,              # Average shark
    'whale': 12.0,
    'dolphin': 2.5,
    'turtle': 0.8,
    'jellyfish': 0.30,
    'octopus': 0.50,
    'squid': 0.40,
    'crab': 0.15,
    'lobster': 0.35,
    'starfish': 0.20,
    'seahorse': 0.15,
    'ray': 1.5,
    'eel': 0.80,
    'coral': 0.40,

    # Debris and trash (common sizes)
    'plastic': 0.15,           # Generic plastic piece
    'plastic_bag': 0.30,
    'plastic_bottle': 0.25,
    'bottle': 0.25,
    'can': 0.12,               # Soda/beer can
    'glass': 0.20,             # Glass bottle
    'glass_bottle': 0.25,
    'tire': 0.65,              # Car tire diameter
    'rope': 0.03,              # Rope diameter (length varies)
    'net': 0.50,               # Fishing net section
    'fishing_net': 1.0,
    'debris': 0.25,            # Generic debris
    'metal_debris': 0.30,
    'wood': 0.40,
    'paper': 0.20,
    'cloth': 0.30,
    'shoe': 0.30,
    'bucket': 0.35,
    'container': 0.50,
    'barrel': 0.90,

    # Equipment
    'buoy': 0.50,
    'anchor': 0.80,
    'chain': 0.05,             # Link diameter

    # Default for unknown objects
    'default': 0.25,
}

# Reference distance for "natural" object appearance (in meters)
# Objects photographed at this distance look "normal sized" in their source image
REFERENCE_CAPTURE_DISTANCE = 2.0

# Underwater visibility parameters
UNDERWATER_VISIBILITY = {
    'clear': 30.0,      # meters - very clear tropical water
    'moderate': 15.0,   # meters - typical ocean
    'murky': 5.0,       # meters - low visibility
}


def get_real_world_size(object_class: str) -> float:
    """Get the real-world size of an object in meters."""
    class_lower = object_class.lower().strip()

    # Direct match
    if class_lower in REAL_WORLD_SIZES:
        return REAL_WORLD_SIZES[class_lower]

    # Keyword matching
    keyword_map = {
        'fish': ['fish', 'tuna', 'salmon', 'bass', 'trout', 'cod'],
        'shark': ['shark', 'tiburon'],
        'plastic_bottle': ['bottle', 'botella'],
        'can': ['can', 'lata', 'aluminum'],
        'plastic': ['plastic', 'plastico', 'wrapper', 'bag'],
        'tire': ['tire', 'tyre', 'llanta', 'neumatico'],
        'rope': ['rope', 'cuerda', 'line', 'string'],
        'net': ['net', 'red', 'fishing'],
        'debris': ['debris', 'trash', 'waste', 'basura', 'residuo'],
    }

    for key, keywords in keyword_map.items():
        if any(kw in class_lower for kw in keywords):
            return REAL_WORLD_SIZES.get(key, REAL_WORLD_SIZES['default'])

    return REAL_WORLD_SIZES['default']


# =============================================================================
# Helper Functions
# =============================================================================

def compute_iou(boxA: Tuple, boxB: Tuple) -> float:
    """Calculate Intersection over Union of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea != 0 else 0


def check_mask_overlap(
    placement_mask: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_pos: Tuple[int, int],
    threshold: float = 0.05
) -> bool:
    """Check overlap between masks with robust validation."""
    try:
        if placement_mask is None or candidate_mask is None:
            return False

        if len(placement_mask.shape) != 2 or len(candidate_mask.shape) != 2:
            return False

        x, y = candidate_pos
        cand_h, cand_w = candidate_mask.shape
        bg_h, bg_w = placement_mask.shape

        if cand_h == 0 or cand_w == 0 or bg_h == 0 or bg_w == 0:
            return False

        roi_x_start = max(0, x)
        roi_y_start = max(0, y)
        roi_x_end = min(bg_w, x + cand_w)
        roi_y_end = min(bg_h, y + cand_h)

        if roi_x_start >= roi_x_end or roi_y_start >= roi_y_end:
            return False

        cand_roi_x_start = max(0, roi_x_start - x)
        cand_roi_y_start = max(0, roi_y_start - y)
        cand_roi_x_end = min(cand_roi_x_start + (roi_x_end - roi_x_start), cand_w)
        cand_roi_y_end = min(cand_roi_y_start + (roi_y_end - roi_y_start), cand_h)

        roi_placement = placement_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        roi_candidate = candidate_mask[cand_roi_y_start:cand_roi_y_end, cand_roi_x_start:cand_roi_x_end]

        if roi_placement.shape != roi_candidate.shape:
            min_h = min(roi_placement.shape[0], roi_candidate.shape[0])
            min_w = min(roi_placement.shape[1], roi_candidate.shape[1])
            roi_placement = roi_placement[:min_h, :min_w]
            roi_candidate = roi_candidate[:min_h, :min_w]

        intersection = cv2.bitwise_and(roi_placement, roi_candidate)
        candidate_area = cv2.countNonZero(roi_candidate)

        if candidate_area == 0:
            return False

        overlap_area = cv2.countNonZero(intersection)
        overlap_ratio = overlap_area / candidate_area

        return overlap_ratio > threshold

    except Exception as e:
        logger.error(f"Error in check_mask_overlap: {e}")
        return True


def refine_object_mask(
    alpha_channel: np.ndarray,
    mask_bin: np.ndarray,
    feather_radius: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine binary mask for smoother edges with proper feathering.

    Creates a gradual transition at object edges to prevent hard cutoffs.
    Uses distance transform for natural edge falloff.

    Args:
        alpha_channel: Original alpha channel (0-255)
        mask_bin: Binary mask (0 or 255)
        feather_radius: Pixels to feather at edges (default: 3)

    Returns:
        Tuple of (refined_alpha as float 0-1, binary_mask)
    """
    unique_vals = np.unique(alpha_channel)
    has_transparency = len(unique_vals) > 5

    if has_transparency:
        # Image already has gradual alpha - enhance edges slightly
        alpha_refined = alpha_channel.astype(np.float32) / 255.0
        alpha_refined[alpha_refined < 0.05] = 0.0

        # Apply slight blur to smooth any jagged edges
        alpha_refined = cv2.GaussianBlur(alpha_refined, (5, 5), 1.0)

        _, mask_refined_bin = cv2.threshold(
            (alpha_refined * 255).astype(np.uint8), 10, 255, cv2.THRESH_BINARY
        )
        return alpha_refined, mask_refined_bin

    # For binary masks, create smooth feathered edges using distance transform
    mask_float = mask_bin.astype(np.float32) / 255.0

    # Calculate distance from edge (inward)
    dist_inside = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)

    # Calculate distance from edge (outward) - invert mask
    mask_inv = 255 - mask_bin
    dist_outside = cv2.distanceTransform(mask_inv, cv2.DIST_L2, 5)

    # Create signed distance field (positive inside, negative outside)
    # Then convert to smooth alpha using feather radius
    feather = max(feather_radius, 2)

    # Normalize distances to feather zone
    alpha_inside = np.clip(dist_inside / feather, 0, 1)
    alpha_outside = np.clip(dist_outside / feather, 0, 1)

    # Combine: fully opaque in center, gradient at edges
    # Inside the mask: 1 where far from edge, gradient near edge
    # Outside the mask: 0
    alpha_refined = alpha_inside * mask_float

    # Apply smoothing for natural falloff
    alpha_refined = cv2.GaussianBlur(alpha_refined, (5, 5), 1.5)

    # Ensure core of object remains fully opaque
    core_mask = dist_inside > feather
    alpha_refined[core_mask] = 1.0

    return alpha_refined, mask_bin


# =============================================================================
# Image Composer Class
# =============================================================================

class ImageComposer:
    """
    Handles synthetic image composition with realistic effects.

    Features:
    - Depth-aware object scaling and placement
    - Multi-light source shadow generation
    - 11+ realism effects (color correction, blur, caustics, etc.)
    - Physics-aware composition
    """

    def __init__(
        self,
        use_gpu: bool = True,
        depth_service_url: Optional[str] = None,
        max_upscale_ratio: float = 4.0,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.4,
        overlap_threshold: float = 0.1,
        try_count: int = 5,
    ):
        self.use_gpu = use_gpu
        self.depth_service_url = depth_service_url or os.environ.get("DEPTH_SERVICE_URL", "http://depth:8001")
        self.max_upscale_ratio = max_upscale_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.overlap_threshold = overlap_threshold
        self.try_count = try_count

        # Lazy-loaded components
        self._lighting_estimator = None
        self._segmentation_client = None
        self._caustics_cache = {}

        # Segmentation service URL
        self.segmentation_service_url = os.environ.get("SEGMENTATION_SERVICE_URL", "http://segmentation:8002")

        logger.info(f"ImageComposer initialized (GPU: {use_gpu})")

    @property
    def lighting_estimator(self):
        """Lazy load lighting estimator"""
        if self._lighting_estimator is None:
            try:
                from app.lighting_engine import AdvancedLightingEstimator
                self._lighting_estimator = AdvancedLightingEstimator(
                    max_light_sources=3,
                    intensity_threshold=0.6,
                    use_hdr_estimation=False
                )
                logger.info("AdvancedLightingEstimator loaded")
            except ImportError as e:
                logger.warning(f"LightingEstimator not available: {e}, using fallback")
        return self._lighting_estimator

    @property
    def segmentation_client(self):
        """Lazy load segmentation client for scene analysis via HTTP"""
        if self._segmentation_client is None:
            try:
                from app.segmentation_client import SegmentationClient
                self._segmentation_client = SegmentationClient(
                    service_url=self.segmentation_service_url,
                    timeout=30,
                    debug=True,
                )
                logger.info(f"SegmentationClient initialized: {self.segmentation_service_url}")
            except ImportError as e:
                logger.warning(f"SegmentationClient not available: {e}")
        return self._segmentation_client

    # =========================================================================
    # Main Composition Methods
    # =========================================================================

    async def compose(
        self,
        background_path: str,
        objects: List[ObjectPlacement],
        depth_map_path: Optional[str] = None,
        lighting_info: Optional[LightingInfo] = None,
        effects: List[EffectType] = None,
        effects_config: EffectsConfig = None,
        output_path: str = None,
        save_annotations: bool = True,
        debug_output_dir: Optional[str] = None,
    ) -> ComposeResult:
        """
        Compose a single synthetic image.

        Args:
            background_path: Path to background image
            objects: List of objects to place
            depth_map_path: Optional pre-computed depth map
            lighting_info: Optional pre-computed lighting
            effects: Effects to apply
            effects_config: Effect configuration
            output_path: Where to save result
            save_annotations: Whether to save annotations JSON
            debug_output_dir: Optional directory to save intermediate pipeline images

        Returns:
            ComposeResult with annotations and metadata
        """
        effects = effects or [EffectType.COLOR_CORRECTION, EffectType.BLUR_MATCHING]
        effects_config = effects_config or EffectsConfig()

        # Load background
        bg_image = cv2.imread(background_path)
        if bg_image is None:
            raise FileNotFoundError(f"Background not found: {background_path}")

        bg_h, bg_w = bg_image.shape[:2]
        result = ComposeResult()
        result.effects_applied = [e.value for e in effects]

        # Debug: Save background
        if debug_output_dir:
            cv2.imwrite(os.path.join(debug_output_dir, "00_background.jpg"), bg_image)
            logger.info(f"Pipeline debug: saved 00_background.jpg")

        # Load or compute depth map
        depth_map = None
        if depth_map_path:
            depth_map = np.load(depth_map_path)
            result.depth_used = True
        elif EffectType.BLUR_MATCHING in effects or EffectType.SHADOWS in effects:
            depth_map = await self._get_depth_map(bg_image)
            if depth_map is not None:
                result.depth_used = True

        # Debug: Save depth map as colored heatmap
        if debug_output_dir and depth_map is not None:
            depth_normalized = (depth_map * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            cv2.imwrite(os.path.join(debug_output_dir, "01_depth_map.png"), depth_colored)
            logger.info(f"Pipeline debug: saved 01_depth_map.png")

        # Estimate or use provided lighting
        if lighting_info is None and EffectType.SHADOWS in effects:
            lighting_info = await self._estimate_lighting(bg_image)

        # Debug: Save lighting analysis visualization
        if debug_output_dir and lighting_info is not None:
            lighting_vis = self._visualize_lighting(bg_image.copy(), lighting_info)
            cv2.imwrite(os.path.join(debug_output_dir, "02_lighting_analysis.jpg"), lighting_vis)
            logger.info(f"Pipeline debug: saved 02_lighting_analysis.jpg")

        # Scene analysis for semantic understanding (ALWAYS enabled when segmentation service available)
        scene_analysis = None
        debug_info = None
        placement_decisions = []
        temp_bg_path = None  # Will be set if we save background for segmentation

        # Perform scene analysis for intelligent placement (not just debug mode)
        if self.segmentation_client is not None:
            try:
                # Save background to shared volume for segmentation service
                import time
                temp_bg_path = f"/shared/temp/pipeline_bg_{int(time.time() * 1000)}.jpg"
                os.makedirs(os.path.dirname(temp_bg_path), exist_ok=True)
                cv2.imwrite(temp_bg_path, bg_image)

                if debug_output_dir:
                    # Full debug analysis with visualization
                    scene_analysis, debug_info = await self.segmentation_client.analyze_scene_with_debug(
                        bg_image,
                        image_path=temp_bg_path,
                        save_visualization=False,  # We'll create our own
                        image_id="pipeline_debug",
                    )
                    # Create scene analysis visualization
                    scene_vis = self._visualize_scene_analysis(bg_image.copy(), scene_analysis, debug_info)
                    cv2.imwrite(os.path.join(debug_output_dir, "02b_scene_analysis.jpg"), scene_vis)
                    logger.info(f"Pipeline debug: saved 02b_scene_analysis.jpg")
                else:
                    # Normal mode: just analyze scene (no debug overhead)
                    scene_analysis = await self.segmentation_client.analyze_scene(
                        bg_image,
                        image_path=temp_bg_path,
                    )
                    logger.debug(f"Scene analysis: dominant={scene_analysis.dominant_region}")
            except Exception as e:
                logger.warning(f"Scene analysis failed: {e}, creating heuristic analysis")
                scene_analysis = self._create_heuristic_scene_analysis(bg_image)

        # Initialize placement tracking
        placement_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
        existing_bboxes = []
        composite = bg_image.copy()

        # Place each object - track all objects in debug mode
        object_debug_results = []  # Collect debug info for all objects
        for obj_idx, obj_placement in enumerate(objects):
            # Create per-object debug directory if in debug mode
            obj_debug_dir = None
            if debug_output_dir:
                obj_debug_dir = os.path.join(debug_output_dir, f"object_{obj_idx:02d}_{obj_placement.class_name}")
                os.makedirs(obj_debug_dir, exist_ok=True)

            # Track placement decision if scene analysis is available
            placement_decision = None
            if scene_analysis is not None and self.segmentation_client is not None:
                try:
                    # Determine position for compatibility check
                    check_pos = obj_placement.position or (bg_w // 2, bg_h // 2)
                    _, reason, placement_decision = await self.segmentation_client.check_compatibility_with_debug(
                        obj_placement.class_name,
                        check_pos,
                        bg_image,
                        image_path=temp_bg_path,
                    )
                except Exception as e:
                    logger.debug(f"Placement decision tracking failed: {e}")

            obj_result = await self._place_object(
                composite=composite,
                obj_placement=obj_placement,
                depth_map=depth_map,
                lighting_info=lighting_info,
                effects=effects,
                effects_config=effects_config,
                placement_mask=placement_mask,
                existing_bboxes=existing_bboxes,
                debug_output_dir=obj_debug_dir,
                object_index=obj_idx,  # Pass object index for debug naming
            )

            if obj_result is not None:
                composite, bbox, obj_mask = obj_result

                # Collect debug info
                if debug_output_dir:
                    object_debug_results.append({
                        'index': obj_idx,
                        'class_name': obj_placement.class_name,
                        'bbox': bbox,
                        'placement_decision': placement_decision,
                        'debug_dir': obj_debug_dir,
                    })
                existing_bboxes.append(bbox)

                # Update placement decision with actual position
                if placement_decision is not None:
                    placement_decision.requested_position = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    placement_decisions.append(placement_decision)

                # Update placement mask
                if obj_mask is not None:
                    kernel = np.ones((5, 5), np.uint8)
                    dilated = cv2.dilate(obj_mask, kernel, iterations=1)
                    placement_mask = cv2.bitwise_or(placement_mask, dilated)

                # Create annotation
                x1, y1, x2, y2 = bbox
                annotation = AnnotationBox(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    class_name=obj_placement.class_name,
                    confidence=1.0,
                    area=(x2 - x1) * (y2 - y1),
                )
                result.annotations.append(annotation)
                result.objects_placed += 1

        # Debug: Save composite after object placement (before global effects)
        if debug_output_dir:
            cv2.imwrite(os.path.join(debug_output_dir, "08_objects_placed.jpg"), composite)
            logger.info(f"Pipeline debug: saved 08_objects_placed.jpg")

            # Save placement decisions visualization
            if placement_decisions and scene_analysis is not None:
                placement_vis = self._visualize_placement_decisions(
                    composite.copy(),
                    scene_analysis,
                    placement_decisions,
                    existing_bboxes,
                )
                cv2.imwrite(os.path.join(debug_output_dir, "08b_placement_decisions.jpg"), placement_vis)
                logger.info(f"Pipeline debug: saved 08b_placement_decisions.jpg")

            # Create consolidated multi-object summary visualization
            if object_debug_results:
                summary_vis = self._create_multi_object_summary(
                    bg_image.copy(),
                    composite.copy(),
                    object_debug_results,
                    scene_analysis,
                    depth_map,
                )
                cv2.imwrite(os.path.join(debug_output_dir, "08c_multi_object_summary.jpg"), summary_vis)
                logger.info(f"Pipeline debug: saved 08c_multi_object_summary.jpg with {len(object_debug_results)} objects")

        # Apply global effects
        if EffectType.CAUSTICS in effects:
            composite = self._apply_caustics(composite, effects_config.caustics_intensity)
            # Debug: Save after caustics
            if debug_output_dir:
                cv2.imwrite(os.path.join(debug_output_dir, "09_caustics.jpg"), composite)
                logger.info(f"Pipeline debug: saved 09_caustics.jpg")

        if EffectType.UNDERWATER in effects:
            composite = self._apply_underwater_effect(
                composite,
                effects_config.underwater_intensity,
                effects_config.water_color,
            )
            # Debug: Save after underwater effect
            if debug_output_dir:
                cv2.imwrite(os.path.join(debug_output_dir, "10_underwater.jpg"), composite)
                logger.info(f"Pipeline debug: saved 10_underwater.jpg")

        # Debug: Save final result
        if debug_output_dir:
            cv2.imwrite(os.path.join(debug_output_dir, "11_final.jpg"), composite)
            logger.info(f"Pipeline debug: saved 11_final.jpg")

        # Save output
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, composite)

            if save_annotations and result.annotations:
                ann_path = output_path.rsplit('.', 1)[0] + '_annotations.json'
                annotations_dict = [ann.model_dump() for ann in result.annotations]
                with open(ann_path, 'w') as f:
                    json.dump(annotations_dict, f, indent=2)

        return result

    async def compose_batch(
        self,
        backgrounds_dir: str,
        objects_dir: str,
        output_dir: str,
        num_images: int,
        targets_per_class: Optional[Dict[str, int]] = None,
        max_objects_per_image: int = 5,
        effects: List[EffectType] = None,
        effects_config: EffectsConfig = None,
        depth_aware: bool = True,
        depth_service_url: Optional[str] = None,
        validate_quality: bool = False,
        validate_physics: bool = False,
        reject_invalid: bool = True,
        save_pipeline_debug: bool = False,
        progress_callback: Optional[Callable] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> BatchResult:
        """
        Compose multiple synthetic images in batch.

        Args:
            backgrounds_dir: Directory with background images
            objects_dir: Directory with object images by class
            output_dir: Output directory
            num_images: Number of images to generate
            targets_per_class: Optional target count per class
            max_objects_per_image: Max objects per image
            effects: Effects to apply
            effects_config: Effect configuration
            depth_aware: Use depth-aware placement
            depth_service_url: URL to depth service
            validate_quality: Run quality validation
            validate_physics: Run physics validation
            reject_invalid: Reject images that fail validation
            save_pipeline_debug: Save intermediate pipeline images for first iteration
            progress_callback: Callback for progress updates
            cancel_check: Callback that returns True if job should be cancelled

        Returns:
            BatchResult with generation statistics
        """
        effects = effects or [
            EffectType.COLOR_CORRECTION,
            EffectType.BLUR_MATCHING,
            EffectType.CAUSTICS,
        ]
        effects_config = effects_config or EffectsConfig()

        if depth_service_url:
            self.depth_service_url = depth_service_url

        # Create output directories
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Create pipeline debug directory if requested
        debug_dir = None
        if save_pipeline_debug:
            debug_dir = os.path.join(output_dir, "pipeline_debug")
            os.makedirs(debug_dir, exist_ok=True)
            logger.info(f"Pipeline debug enabled, saving to: {debug_dir}")

        # Load backgrounds
        bg_files = glob.glob(os.path.join(backgrounds_dir, "*.*"))
        bg_files = [f for f in bg_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not bg_files:
            raise FileNotFoundError(f"No backgrounds found in {backgrounds_dir}")

        # Load objects by class
        objects_by_class = self._load_objects_from_directory(objects_dir)
        if not objects_by_class:
            raise FileNotFoundError(f"No objects found in {objects_dir}")

        available_classes = list(objects_by_class.keys())
        logger.info(f"Found {len(available_classes)} classes: {available_classes}")

        # Initialize counters
        result = BatchResult()
        result.synthetic_counts = {cls: 0 for cls in available_classes}

        # Calculate required instances
        if targets_per_class:
            required = {cls: targets_per_class.get(cls, 0) for cls in available_classes}
        else:
            per_class = max(1, num_images // len(available_classes))
            required = {cls: per_class for cls in available_classes}

        total_required = sum(required.values())
        all_annotations = []

        # Generation loop
        generated = 0
        rejected = 0
        max_iterations = num_images * 10  # Safety limit

        for iteration in range(max_iterations):
            # Check for cancellation
            if cancel_check and cancel_check():
                logger.info(f"Batch composition cancelled after {generated} images")
                break

            if generated >= num_images:
                break

            # Check if any class still needs instances
            remaining = {cls: required[cls] - result.synthetic_counts[cls]
                        for cls in available_classes}
            available = [cls for cls, cnt in remaining.items() if cnt > 0 and objects_by_class.get(cls)]

            if not available:
                break

            # Select background
            bg_file = random.choice(bg_files)
            bg_image = cv2.imread(bg_file)
            if bg_image is None:
                continue

            # Get depth map if needed
            depth_map = None
            if depth_aware:
                depth_map = await self._get_depth_map(bg_image)

            # Select objects to place
            num_objects = random.randint(1, min(max_objects_per_image, len(available)))
            selected_classes = random.sample(available, k=num_objects)

            # Create object placements
            objects = []
            for cls in selected_classes:
                obj_data = random.choice(objects_by_class[cls])
                objects.append(ObjectPlacement(
                    image_path=obj_data['path'],
                    class_name=cls,
                    position=None,  # Auto-place
                    scale=None,     # Depth-aware
                    rotation=None,  # Random
                ))

            # Compose image
            output_path = os.path.join(images_dir, f"synthetic_{generated:05d}.jpg")

            # Pass debug_dir only for first iteration
            current_debug_dir = debug_dir if (save_pipeline_debug and generated == 0) else None

            try:
                compose_result = await self.compose(
                    background_path=bg_file,
                    objects=objects,
                    depth_map_path=None,
                    lighting_info=None,
                    effects=effects,
                    effects_config=effects_config,
                    output_path=output_path,
                    save_annotations=True,
                    debug_output_dir=current_debug_dir,
                )

                if compose_result.objects_placed > 0:
                    generated += 1

                    # Update class counts
                    for ann in compose_result.annotations:
                        result.synthetic_counts[ann.class_name] = \
                            result.synthetic_counts.get(ann.class_name, 0) + 1

                    # Collect annotations for COCO
                    all_annotations.append({
                        'image_name': f"synthetic_{generated-1:05d}.jpg",
                        'width': bg_image.shape[1],
                        'height': bg_image.shape[0],
                        'annotations': compose_result.annotations,
                    })

            except Exception as e:
                logger.warning(f"Failed to compose image: {e}")
                rejected += 1
                continue

            # Progress callback
            if progress_callback:
                progress_callback({
                    'generated': generated,
                    'rejected': rejected,
                    'pending': num_images - generated,
                    'counts': result.synthetic_counts,
                })

        # Generate COCO JSON
        coco_path = os.path.join(output_dir, "synthetic_dataset.json")
        self._generate_coco_json(all_annotations, available_classes, coco_path)

        result.images_generated = generated
        result.images_rejected = rejected
        result.output_coco_path = coco_path

        logger.info(f"Batch complete: {generated} generated, {rejected} rejected")
        return result

    async def estimate_lighting(
        self,
        image_path: str,
        max_light_sources: int = 3,
        intensity_threshold: float = 0.6,
        estimate_hdr: bool = False,
        apply_water_attenuation: bool = False,
        depth_category: DepthZone = DepthZone.MID,
        water_clarity: WaterClarity = WaterClarity.CLEAR,
    ) -> LightingInfo:
        """Estimate lighting conditions from an image."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        return await self._estimate_lighting(
            image,
            max_light_sources,
            intensity_threshold,
            apply_water_attenuation,
            depth_category,
            water_clarity,
        )

    # =========================================================================
    # Private Methods
    # =========================================================================

    async def _place_object(
        self,
        composite: np.ndarray,
        obj_placement: ObjectPlacement,
        depth_map: Optional[np.ndarray],
        lighting_info: Optional[LightingInfo],
        effects: List[EffectType],
        effects_config: EffectsConfig,
        placement_mask: np.ndarray,
        existing_bboxes: List[Tuple],
        debug_output_dir: Optional[str] = None,
        object_index: int = 0,
    ) -> Optional[Tuple[np.ndarray, Tuple, np.ndarray]]:
        """Place a single object on the composite image."""
        # Load object image
        obj_img = cv2.imread(obj_placement.image_path, cv2.IMREAD_UNCHANGED)
        if obj_img is None:
            logger.warning(f"Failed to load object: {obj_placement.image_path}")
            return None

        # Debug: Save original object
        if debug_output_dir:
            cv2.imwrite(os.path.join(debug_output_dir, "03_object_original.png"), obj_img)
            logger.info(f"Pipeline debug: saved 03_object_original.png")

        bg_h, bg_w = composite.shape[:2]
        bg_area = bg_h * bg_w

        for attempt in range(self.try_count):
            # Apply transformations with physics-based scaling
            transformed, pos, scale = self._transform_object(
                obj_img,
                (bg_h, bg_w),
                depth_map,
                obj_placement.position,
                obj_placement.scale,
                obj_placement.rotation,
                object_class=obj_placement.class_name,
            )

            if transformed is None:
                continue

            # Debug: Save transformed object (only once)
            if debug_output_dir and attempt == 0:
                cv2.imwrite(os.path.join(debug_output_dir, "04_object_transformed.png"), transformed)
                logger.info(f"Pipeline debug: saved 04_object_transformed.png")

            # Extract alpha channel
            if transformed.shape[2] == 4:
                alpha_channel = transformed[:, :, 3]
                obj_bgr = transformed[:, :, :3]
            else:
                alpha_channel = np.ones(transformed.shape[:2], dtype=np.uint8) * 255
                obj_bgr = transformed

            # Create mask
            _, mask_bin = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
            alpha_factor, mask_bin = refine_object_mask(alpha_channel, mask_bin)

            # Check area constraints
            object_area = cv2.countNonZero(mask_bin)
            if object_area == 0:
                continue

            object_ratio = object_area / bg_area
            if not (self.min_area_ratio <= object_ratio <= self.max_area_ratio):
                continue

            # Check overlap
            if check_mask_overlap(placement_mask, mask_bin, pos, self.overlap_threshold):
                continue

            # Calculate bounding box
            x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(mask_bin)
            bbox = (
                pos[0] + x_bb,
                pos[1] + y_bb,
                pos[0] + x_bb + w_bb,
                pos[1] + y_bb + h_bb,
            )

            # Check IoU with existing boxes
            if any(compute_iou(bbox, bb) > self.overlap_threshold for bb in existing_bboxes):
                continue

            # Calculate paste region
            p_h, p_w = transformed.shape[:2]
            x_paste, y_paste = pos

            x_start_bg = max(x_paste, 0)
            y_start_bg = max(y_paste, 0)
            x_end_bg = min(x_paste + p_w, bg_w)
            y_end_bg = min(y_paste + p_h, bg_h)
            x_start_patch = max(0, -x_paste)
            y_start_patch = max(0, -y_paste)
            eff_w = x_end_bg - x_start_bg
            eff_h = y_end_bg - y_start_bg

            if eff_w <= 0 or eff_h <= 0:
                continue

            # Extract effective regions
            obj_bgr_eff = obj_bgr[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            mask_bin_eff = mask_bin[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            alpha_eff = alpha_factor[y_start_patch:y_start_patch+eff_h, x_start_patch:x_start_patch+eff_w]
            alpha_eff_3ch = alpha_eff[:, :, np.newaxis] if len(alpha_eff.shape) == 2 else alpha_eff

            roi = composite[y_start_bg:y_end_bg, x_start_bg:x_end_bg]

            # Apply per-object effects
            obj_processed = obj_bgr_eff.copy()

            if EffectType.COLOR_CORRECTION in effects:
                obj_processed = self._apply_color_correction(
                    obj_processed, roi, mask_bin_eff, effects_config.color_intensity
                )
                # Debug: Save after color correction
                if debug_output_dir:
                    debug_img = roi.copy()
                    alpha_3ch = alpha_eff_3ch if len(alpha_eff_3ch.shape) == 3 else alpha_eff_3ch[:, :, np.newaxis]
                    debug_blend = (obj_processed.astype(np.float32) * alpha_3ch +
                                   debug_img.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)
                    cv2.imwrite(os.path.join(debug_output_dir, "05_color_correction.jpg"), debug_blend)
                    logger.info(f"Pipeline debug: saved 05_color_correction.jpg")

            if EffectType.BLUR_MATCHING in effects:
                obj_processed = self._apply_blur_matching(
                    obj_processed, roi, mask_bin_eff, effects_config.blur_strength
                )
                # Debug: Save after blur matching
                if debug_output_dir:
                    debug_img = roi.copy()
                    alpha_3ch = alpha_eff_3ch if len(alpha_eff_3ch.shape) == 3 else alpha_eff_3ch[:, :, np.newaxis]
                    debug_blend = (obj_processed.astype(np.float32) * alpha_3ch +
                                   debug_img.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)
                    cv2.imwrite(os.path.join(debug_output_dir, "06_blur_matching.jpg"), debug_blend)
                    logger.info(f"Pipeline debug: saved 06_blur_matching.jpg")

            if EffectType.LIGHTING in effects:
                obj_processed = self._apply_lighting_effect(
                    obj_processed, effects_config.lighting_type, effects_config.lighting_intensity
                )

            # Apply motion blur with probability
            if EffectType.MOTION_BLUR in effects:
                if random.random() < effects_config.motion_blur_probability:
                    obj_processed = self._apply_motion_blur(
                        obj_processed, effects_config.motion_blur_kernel
                    )

            obj_processed = np.clip(obj_processed, 0, 255).astype(np.uint8)

            # Generate depth-aware shadow
            bg_with_shadow = composite.copy()
            if EffectType.SHADOWS in effects:
                bg_with_shadow = self._apply_shadow(
                    bg_with_shadow,
                    mask_bin_eff,
                    (x_start_bg, y_start_bg, eff_w, eff_h),
                    lighting_info,
                    effects_config.shadow_opacity,
                    effects_config.shadow_blur,
                    depth_map=depth_map,
                    position=pos,
                )
                # Debug: Save after shadow application
                if debug_output_dir:
                    cv2.imwrite(os.path.join(debug_output_dir, "07_shadows.jpg"), bg_with_shadow)
                    logger.info(f"Pipeline debug: saved 07_shadows.jpg")

            # Blend object onto background
            roi_with_shadow = bg_with_shadow[y_start_bg:y_end_bg, x_start_bg:x_end_bg]

            if EffectType.POISSON_BLEND in effects:
                blended = self._apply_poisson_blend(
                    obj_processed, roi_with_shadow, mask_bin_eff
                )
            else:
                # Alpha blending
                blended = (
                    obj_processed.astype(np.float32) * alpha_eff_3ch +
                    roi_with_shadow.astype(np.float32) * (1.0 - alpha_eff_3ch)
                ).astype(np.uint8)

            # Apply edge smoothing
            if EffectType.EDGE_SMOOTHING in effects:
                blended = self._apply_edge_smoothing(
                    blended, roi_with_shadow, mask_bin_eff, effects_config.edge_feather
                )

            # Update composite
            result = bg_with_shadow.copy()
            result[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = blended

            # Create object mask in full image coordinates
            obj_mask_full = np.zeros((bg_h, bg_w), dtype=np.uint8)
            obj_mask_full[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = mask_bin_eff

            return result, bbox, obj_mask_full

        return None

    def _transform_object(
        self,
        obj_img: np.ndarray,
        bg_shape: Tuple[int, int],
        depth_map: Optional[np.ndarray],
        position: Optional[Tuple[int, int]],
        scale: Optional[float],
        rotation: Optional[float],
        object_class: str = "default",
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int], float]:
        """
        Apply geometric transformations to object with high-quality interpolation.

        Uses appropriate interpolation methods to prevent aliasing artifacts:
        - INTER_LANCZOS4 for upscaling (best quality, prevents jaggies)
        - INTER_AREA for downscaling (prevents moire patterns)
        - INTER_CUBIC for rotation (good balance of quality and speed)

        Also applies pre-filtering to prevent aliasing in low-resolution objects.
        """
        bg_h, bg_w = bg_shape
        transformed = obj_img.copy()

        # Pre-processing: Apply slight blur to very small objects to reduce aliasing
        h, w = transformed.shape[:2]
        min_dim = min(h, w)
        if min_dim < 100:
            # Small objects need anti-aliasing to prevent jaggies
            blur_size = 3 if min_dim > 50 else 5
            # Only blur the RGB channels, not alpha
            if transformed.shape[2] == 4:
                rgb = transformed[:, :, :3]
                alpha = transformed[:, :, 3:4]
                rgb = cv2.GaussianBlur(rgb, (blur_size, blur_size), 0.5)
                transformed = np.concatenate([rgb, alpha], axis=2)
            else:
                transformed = cv2.GaussianBlur(transformed, (blur_size, blur_size), 0.5)

        # Apply rotation with high-quality interpolation
        if rotation is None:
            rotation = random.uniform(-45, 45)

        if rotation != 0:
            h, w = transformed.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            # Use INTER_CUBIC for rotation - better quality than LINEAR
            transformed = cv2.warpAffine(
                transformed, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0) if transformed.shape[2] == 4 else (0, 0, 0)
            )

        p_h, p_w = transformed.shape[:2]
        if p_h == 0 or p_w == 0:
            return None, (0, 0), 1.0

        # Calculate scale using physics-based approach
        if scale is None:
            if depth_map is not None:
                scale = self._calculate_depth_aware_scale(
                    (p_h, p_w), (bg_h, bg_w), depth_map, position, object_class
                )
            else:
                max_scale_w = (bg_w * 0.7) / p_w
                max_scale_h = (bg_h * 0.7) / p_h
                max_scale = min(max_scale_w, max_scale_h, self.max_upscale_ratio)
                scale = random.uniform(0.1, max_scale) if max_scale > 0.1 else 0.5

        # Apply scale with appropriate interpolation
        if scale != 1.0:
            new_w = int(p_w * scale)
            new_h = int(p_h * scale)
            if new_w > 0 and new_h > 0:
                # Choose interpolation based on scaling direction
                if scale > 1.0:
                    # Upscaling: Use LANCZOS4 for best quality (prevents jaggies)
                    interpolation = cv2.INTER_LANCZOS4
                else:
                    # Downscaling: Use AREA for best quality (prevents moire)
                    interpolation = cv2.INTER_AREA

                transformed = cv2.resize(transformed, (new_w, new_h), interpolation=interpolation)

                # For significant upscaling, apply subtle smoothing to blend edges
                if scale > 2.0:
                    if transformed.shape[2] == 4:
                        rgb = transformed[:, :, :3]
                        alpha = transformed[:, :, 3:4]
                        rgb = cv2.GaussianBlur(rgb, (3, 3), 0.3)
                        transformed = np.concatenate([rgb, alpha], axis=2)
                    else:
                        transformed = cv2.GaussianBlur(transformed, (3, 3), 0.3)

        p_h, p_w = transformed.shape[:2]
        if p_h == 0 or p_w == 0:
            return None, (0, 0), scale

        # Minimum size check: reject objects that would be too small
        # (These would appear as blurry blobs with visible aliasing)
        if p_h < 30 or p_w < 30:
            logger.debug(f"Object too small after scaling ({p_w}x{p_h}), skipping")
            return None, (0, 0), scale

        # Calculate position
        if position is None:
            max_x = max(bg_w - p_w, 1)
            max_y = max(bg_h - p_h, 1)
            position = (random.randrange(0, max_x), random.randrange(0, max_y))

        return transformed, position, scale

    def _calculate_depth_aware_scale(
        self,
        obj_dims: Tuple[int, int],
        bg_dims: Tuple[int, int],
        depth_map: np.ndarray,
        position: Optional[Tuple[int, int]] = None,
        object_class: str = "default",
    ) -> float:
        """
        Calculate scale factor using physics-based perspective model.

        This algorithm considers:
        1. Real-world size of the object type (shark=2.5m, can=0.12m, etc.)
        2. Depth/distance in the scene (perspective scaling)
        3. Source image resolution vs object's real size
        4. Natural size constraints to avoid unrealistic compositions

        The model assumes:
        - Underwater visibility of ~15m (moderate clarity)
        - Source images captured at ~2m reference distance
        - Linear perspective scaling with distance

        Args:
            obj_dims: (height, width) of the object in pixels
            bg_dims: (height, width) of the background
            depth_map: Normalized depth map (0=near, 1=far)
            position: Optional (x, y) position for depth sampling
            object_class: Type of object for real-world size lookup

        Returns:
            Scale factor to apply to the object
        """
        obj_h, obj_w = obj_dims
        bg_h, bg_w = bg_dims

        # =================================================================
        # Step 1: Get real-world size and estimate source image scale
        # =================================================================
        real_size_meters = get_real_world_size(object_class)

        # Estimate pixels-per-meter in source image
        # Assume source image was taken at REFERENCE_CAPTURE_DISTANCE
        obj_max_dim = max(obj_h, obj_w)
        source_pixels_per_meter = obj_max_dim / real_size_meters

        # =================================================================
        # Step 2: Get depth value and convert to estimated distance
        # =================================================================
        depth_normalized = depth_map.astype(np.float32)
        if depth_normalized.max() > 1:
            depth_normalized = depth_normalized / depth_normalized.max()

        if position:
            x, y = position
            x = max(0, min(x, bg_w - 1))
            y = max(0, min(y, bg_h - 1))
            depth_value = depth_normalized[y, x]
        else:
            # Random depth selection with preference for mid-range
            depth_value = random.triangular(0.2, 0.8, 0.5)

        # Convert normalized depth (0-1) to estimated distance in meters
        # Using moderate underwater visibility (~15m)
        max_visible_distance = UNDERWATER_VISIBILITY.get('moderate', 15.0)
        min_distance = 0.5  # Objects closer than 0.5m are too close
        estimated_distance = min_distance + depth_value * (max_visible_distance - min_distance)

        # =================================================================
        # Step 3: Calculate perspective scale factor
        # =================================================================
        # Objects appear smaller as distance increases (1/d relationship)
        # Scale relative to reference capture distance
        perspective_scale = REFERENCE_CAPTURE_DISTANCE / estimated_distance

        # =================================================================
        # Step 4: Calculate target size in background image
        # =================================================================
        # Estimate background's pixels-per-meter at different depths
        # For training data, we want objects to be clearly visible
        # Using 2.5m scene width for larger, more detectable objects
        bg_scene_width_meters = 2.5  # Smaller scene = larger objects
        bg_pixels_per_meter = bg_w / bg_scene_width_meters

        # Clamp perspective scale to avoid objects being too small at distance
        # Objects beyond ~10m shouldn't shrink much more
        clamped_perspective = max(perspective_scale, 0.35)

        # Target size = real_size * bg_resolution * perspective
        # Apply boost (2.5x) to ensure objects are large enough for detection
        target_size_pixels = real_size_meters * bg_pixels_per_meter * clamped_perspective * 2.5

        # Calculate scale to achieve target size
        scale = target_size_pixels / obj_max_dim

        # =================================================================
        # Step 5: Apply relative size constraints
        # =================================================================
        obj_area = obj_h * obj_w
        bg_area = bg_h * bg_w
        relative_size = obj_area / bg_area

        # Size thresholds (relative to background area)
        SIZE_SMALL = 0.08
        SIZE_MEDIUM = 0.20
        SIZE_LARGE = 0.35

        # Determine max upscale based on current relative size
        # These are more permissive to allow physics-based sizing to work
        if relative_size < 0.01:
            max_upscale_for_size = 4.0
            size_category = "tiny"
        elif relative_size < SIZE_SMALL:
            max_upscale_for_size = 3.0
            size_category = "small"
        elif relative_size < SIZE_MEDIUM:
            max_upscale_for_size = 2.0
            size_category = "medium"
        elif relative_size < SIZE_LARGE:
            max_upscale_for_size = 1.5
            size_category = "large"
        else:
            max_upscale_for_size = 1.0
            size_category = "huge"

        # Cap upscaling based on relative size
        if scale > 1.0:
            scale = min(scale, max_upscale_for_size)

        # =================================================================
        # Step 6: Apply size moderation for large objects
        # =================================================================
        if relative_size > SIZE_SMALL and scale > 1.0:
            # Smoothly reduce upscaling as object gets larger
            moderation = 1.0 - (relative_size - SIZE_SMALL) / (1.0 - SIZE_SMALL)
            moderation = np.clip(moderation, 0.3, 1.0)
            scale = 1.0 + (scale - 1.0) * moderation

        # =================================================================
        # Step 7: Final constraints
        # =================================================================
        # Max 50% of any background dimension (more conservative for realism)
        max_scale_w = (bg_w * 0.5) / obj_w
        max_scale_h = (bg_h * 0.5) / obj_h
        scale = min(scale, max_scale_w, max_scale_h)

        # Apply global max upscale ratio
        scale = min(scale, self.max_upscale_ratio)

        # Minimum scale to keep object visible and recognizable for training
        # At least 150px for good detectability in object detection models
        min_size_pixels = 150
        min_scale = min_size_pixels / obj_max_dim
        # Minimum 15% of original size to preserve enough detail
        scale = max(scale, min_scale, 0.15)

        logger.debug(
            f"Physics scale: {object_class} ({real_size_meters:.2f}m), "
            f"obj={obj_w}x{obj_h}px, depth={depth_value:.2f} (~{estimated_distance:.1f}m), "
            f"perspective={perspective_scale:.2f}, size_cat={size_category}, "
            f"final_scale={scale:.3f}"
        )

        return scale

    async def _get_depth_map(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get depth map from depth service or local estimator."""
        try:
            # Try depth service first
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Encode image to JPEG
                _, buffer = cv2.imencode('.jpg', image)

                response = await client.post(
                    f"{self.depth_service_url}/depth/estimate",
                    files={"file": ("image.jpg", buffer.tobytes(), "image/jpeg")},
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'depth_map' in data:
                        # Decode base64 depth map
                        import base64
                        depth_bytes = base64.b64decode(data['depth_map'])
                        depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
                        h, w = data.get('height', image.shape[0]), data.get('width', image.shape[1])
                        depth_map = depth_array.reshape((h, w))
                        logger.info(f"Depth map received from service: {w}x{h}, range [{depth_map.min():.3f}, {depth_map.max():.3f}]")
                        return depth_map
                else:
                    logger.warning(f"Depth service returned status {response.status_code}: {response.text[:200]}")

        except Exception as e:
            logger.warning(f"Depth service call failed: {e}, using fallback gradient")

        # Fallback: simple gradient depth (NOT suitable for production)
        h, w = image.shape[:2]
        depth_map = np.linspace(0, 1, h)[:, np.newaxis]
        depth_map = np.repeat(depth_map, w, axis=1).astype(np.float32)
        return depth_map

    async def _estimate_lighting(
        self,
        image: np.ndarray,
        max_light_sources: int = 3,
        intensity_threshold: float = 0.6,
        apply_water_attenuation: bool = False,
        depth_category: DepthZone = DepthZone.MID,
        water_clarity: WaterClarity = WaterClarity.CLEAR,
    ) -> LightingInfo:
        """Estimate lighting from image."""
        if self.lighting_estimator:
            try:
                result = self.lighting_estimator.estimate_lighting(image)
                light_sources = []
                for ls in result.light_sources[:max_light_sources]:
                    if ls.intensity >= intensity_threshold:
                        light_sources.append(LightSourceInfo(
                            light_type=LightType(ls.light_type),
                            position=ls.position,
                            intensity=ls.intensity,
                            color=ls.color,
                            shadow_softness=ls.shadow_softness,
                        ))
                return LightingInfo(
                    light_sources=light_sources,
                    dominant_direction=result.dominant_direction,
                    color_temperature=result.color_temperature,
                    ambient_intensity=result.ambient_intensity,
                )
            except Exception as e:
                logger.warning(f"Lighting estimation failed: {e}")

        # Fallback: simple analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Find brightest region
        kernel_size = max(h, w) // 10
        blurred = cv2.GaussianBlur(gray, (kernel_size | 1, kernel_size | 1), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)

        # Calculate dominant direction
        cx, cy = w // 2, h // 2
        dx, dy = max_loc[0] - cx, max_loc[1] - cy
        azimuth = np.degrees(np.arctan2(dy, dx))
        elevation = 45.0  # Default

        # Estimate color temperature from image
        b, g, r = cv2.split(image)
        r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()

        if r_mean > b_mean:
            color_temp = 3000 + (r_mean - b_mean) * 10
        else:
            color_temp = 6500 + (b_mean - r_mean) * 20
        color_temp = np.clip(color_temp, 2000, 12000)

        return LightingInfo(
            light_sources=[
                LightSourceInfo(
                    light_type=LightType.DIRECTIONAL,
                    position=(azimuth, elevation, 1.0),
                    intensity=max_val / 255.0,
                    color=(int(r_mean), int(g_mean), int(b_mean)),
                    shadow_softness=0.5,
                )
            ],
            dominant_direction=(azimuth, elevation),
            color_temperature=float(color_temp),
            ambient_intensity=float(gray.mean() / 255.0),
        )

    def _visualize_lighting(
        self,
        image: np.ndarray,
        lighting_info: LightingInfo,
    ) -> np.ndarray:
        """
        Create a visualization of detected lighting for documentation.

        Draws:
        - Light source positions as colored circles
        - Light direction arrows
        - Ambient intensity indicator
        - Color temperature bar
        """
        try:
            vis = image.copy()
            h, w = vis.shape[:2]
            cx, cy = w // 2, h // 2

            # Draw info panel background
            panel_h = 120
            overlay = vis.copy()
            cv2.rectangle(overlay, (10, h - panel_h - 10), (350, h - 10), (0, 0, 0), -1)
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # Draw each light source
            for i, ls in enumerate(lighting_info.light_sources):
                azimuth, elevation, distance = ls.position
                intensity = ls.intensity

                # Calculate arrow endpoint from center
                arrow_len = min(w, h) // 4 * intensity
                rad_azimuth = np.radians(azimuth)
                end_x = int(cx + arrow_len * np.cos(rad_azimuth))
                end_y = int(cy + arrow_len * np.sin(rad_azimuth))

                # Light source color (warm = yellow/orange, cool = blue)
                if lighting_info.color_temperature < 4000:
                    arrow_color = (0, 165, 255)  # Orange
                elif lighting_info.color_temperature > 6500:
                    arrow_color = (255, 200, 100)  # Light blue
                else:
                    arrow_color = (255, 255, 255)  # White

                # Draw light direction arrow
                cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), arrow_color, 3, tipLength=0.3)

                # Draw light source indicator at origin point
                cv2.circle(vis, (end_x, end_y), 15, arrow_color, -1)
                cv2.circle(vis, (end_x, end_y), 15, (255, 255, 255), 2)

                # Label
                label = f"L{i+1}: {ls.light_type.value}"
                cv2.putText(vis, label, (end_x + 20, end_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw text info in panel
            y_text = h - panel_h + 10
            line_height = 22

            # Dominant direction
            dom_az, dom_el = lighting_info.dominant_direction
            cv2.putText(vis, f"Dominant Direction: {dom_az:.1f}deg az, {dom_el:.1f}deg el",
                       (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += line_height

            # Color temperature
            ct = lighting_info.color_temperature
            ct_label = "Warm" if ct < 4000 else ("Neutral" if ct < 6500 else "Cool")
            cv2.putText(vis, f"Color Temp: {ct:.0f}K ({ct_label})",
                       (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += line_height

            # Ambient intensity
            amb = lighting_info.ambient_intensity
            cv2.putText(vis, f"Ambient Intensity: {amb:.2f}",
                       (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += line_height

            # Light sources count
            cv2.putText(vis, f"Light Sources: {len(lighting_info.light_sources)}",
                       (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw color temperature bar
            bar_x, bar_y = 200, h - panel_h + 35
            bar_w, bar_h = 130, 15

            # Create gradient bar (warm to cool)
            for i in range(bar_w):
                ratio = i / bar_w
                if ratio < 0.5:
                    color = (int(255 * ratio * 2), int(165 * ratio * 2), 0)  # Black to orange
                else:
                    ratio2 = (ratio - 0.5) * 2
                    color = (int(255 - 55 * ratio2), int(165 + 90 * ratio2), int(255 * ratio2))  # Orange to light blue
                cv2.line(vis, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_h), color, 1)

            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)

            # Marker for current temperature
            ct_ratio = np.clip((ct - 2000) / 10000, 0, 1)
            marker_x = int(bar_x + ct_ratio * bar_w)
            cv2.drawMarker(vis, (marker_x, bar_y + bar_h + 8), (255, 255, 255),
                          cv2.MARKER_TRIANGLE_UP, 10, 2)

            return vis

        except Exception as e:
            logger.warning(f"Lighting visualization failed: {e}")
            return image

    def _visualize_scene_analysis(
        self,
        image: np.ndarray,
        scene_analysis,
        debug_info,
    ) -> np.ndarray:
        """
        Create visualization of scene analysis showing detected regions.

        Creates a 3-panel visualization:
        - Original image
        - Region overlay with colors
        - Region scores info panel
        """
        try:
            h, w = image.shape[:2]

            # Color map for scene regions (BGR format)
            region_colors = {
                'open_water': (255, 150, 50),     # Blue
                'seafloor': (50, 150, 200),       # Sandy brown
                'surface': (255, 255, 200),       # Light cyan
                'vegetation': (50, 200, 50),      # Green
                'rocky': (100, 100, 100),         # Gray
                'sandy': (100, 180, 220),         # Sand
                'murky': (80, 80, 60),            # Dark murky
                'unknown': (128, 128, 128),       # Neutral gray
            }

            # Create region visualization based on scores (since region_map may not be available via HTTP)
            region_viz = np.zeros((h, w, 3), dtype=np.uint8)

            # Check if region_map is available
            if hasattr(scene_analysis, 'region_map') and scene_analysis.region_map is not None:
                # Region value to name mapping
                region_value_map = {
                    1: 'open_water', 2: 'seafloor', 3: 'surface',
                    4: 'vegetation', 5: 'rocky', 6: 'sandy', 7: 'murky',
                }
                for value, region_name in region_value_map.items():
                    mask = (scene_analysis.region_map == value)
                    color = region_colors.get(region_name, (128, 128, 128))
                    region_viz[mask] = color
            else:
                # Create approximate visualization based on region scores
                # Use horizontal bands for top regions (FALLBACK - not real segmentation)
                logger.warning("Using fallback band visualization - region_map not available from SAM3")
                sorted_regions = sorted(scene_analysis.region_scores.items(), key=lambda x: -x[1])
                if sorted_regions:
                    y_pos = 0
                    for region_name, score in sorted_regions[:4]:
                        if score > 0.05:
                            band_height = int(h * score)
                            color = region_colors.get(region_name, (128, 128, 128))
                            region_viz[y_pos:y_pos + band_height, :] = color
                            y_pos += band_height
                    # Add warning text on the visualization
                    cv2.putText(region_viz, "FALLBACK VIZ (no SAM3 map)", (10, h - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Create overlay (original + colored regions at 40% opacity)
            overlay = cv2.addWeighted(image, 0.6, region_viz, 0.4, 0)

            # Create info panel
            info_panel = np.zeros((h, w, 3), dtype=np.uint8)
            info_panel[:] = (40, 40, 40)  # Dark gray background

            # Stack: Original | Overlay | Info Panel
            combined = np.hstack([image, overlay, info_panel])

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Scene Regions", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Analysis Info", (2*w + 10, 30), font, 0.7, (255, 255, 255), 2)

            # Add dominant region
            dominant = scene_analysis.dominant_region
            dominant_name = dominant.value if hasattr(dominant, 'value') else str(dominant)
            cv2.putText(combined, f"Dominant: {dominant_name}", (2*w + 10, 60), font, 0.5, (255, 255, 255), 1)

            # Add region scores legend
            y_offset = 85
            for region, score in sorted(scene_analysis.region_scores.items(), key=lambda x: -x[1])[:5]:
                if score > 0.01:
                    color = region_colors.get(region, (255, 255, 255))
                    cv2.putText(combined, f"{region}: {score:.1%}", (2*w + 10, y_offset), font, 0.4, color, 1)
                    y_offset += 20

            # Add analysis method
            method = debug_info.analysis_method if debug_info else "unknown"
            cv2.putText(combined, f"Method: {method}", (2*w + 10, y_offset + 10), font, 0.4, (200, 200, 200), 1)

            # Add water clarity and brightness info
            y_offset += 35
            cv2.putText(combined, f"Brightness: {scene_analysis.scene_brightness:.2f}", (2*w + 10, y_offset), font, 0.4, (200, 200, 200), 1)
            y_offset += 18
            cv2.putText(combined, f"Clarity: {scene_analysis.water_clarity}", (2*w + 10, y_offset), font, 0.4, (200, 200, 200), 1)
            y_offset += 18
            cv2.putText(combined, f"Temp: {scene_analysis.color_temperature}", (2*w + 10, y_offset), font, 0.4, (200, 200, 200), 1)

            # Add processing time if available
            if debug_info and hasattr(debug_info, 'processing_time_ms'):
                y_offset += 25
                cv2.putText(combined, f"Time: {debug_info.processing_time_ms:.1f}ms", (2*w + 10, y_offset), font, 0.4, (200, 200, 200), 1)

            return combined

        except Exception as e:
            logger.warning(f"Scene analysis visualization failed: {e}")
            return image

    def _visualize_placement_decisions(
        self,
        image: np.ndarray,
        scene_analysis,
        placement_decisions: list,
        bboxes: list,
    ) -> np.ndarray:
        """
        Visualize placement decisions on the composed image.

        Shows:
        - Bounding boxes with decision color coding
        - Decision labels (accepted/rejected/relocated)
        - Compatibility scores
        - Alternative positions
        """
        try:
            vis = image.copy()
            h, w = vis.shape[:2]

            # Color coding for decisions
            decision_colors = {
                'accepted': (0, 255, 0),     # Green
                'relocated': (0, 165, 255),  # Orange
                'rejected': (0, 0, 255),     # Red
            }

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw bounding boxes and labels
            for i, (decision, bbox) in enumerate(zip(placement_decisions, bboxes)):
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = decision_colors.get(decision.decision, (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                label = f"{decision.object_class}"
                score_label = f"{decision.compatibility_score:.0%}"
                decision_label = f"[{decision.decision}]"
                region_label = f"@{decision.region_at_position}"

                # Calculate label size
                (label_w, label_h), baseline = cv2.getTextSize(label, font, 0.5, 1)
                label_bg_y = max(y1 - 45, 0)

                # Draw label background
                cv2.rectangle(vis, (x1, label_bg_y), (x1 + max(label_w + 10, 120), label_bg_y + 45), (0, 0, 0), -1)
                cv2.rectangle(vis, (x1, label_bg_y), (x1 + max(label_w + 10, 120), label_bg_y + 45), color, 1)

                # Draw labels
                cv2.putText(vis, label, (x1 + 5, label_bg_y + 12), font, 0.4, (255, 255, 255), 1)
                cv2.putText(vis, f"{decision_label} {score_label}", (x1 + 5, label_bg_y + 26), font, 0.35, color, 1)
                cv2.putText(vis, region_label, (x1 + 5, label_bg_y + 40), font, 0.35, (180, 180, 180), 1)

                # Draw alternative positions (small circles) if relocated
                if decision.decision == 'relocated' and decision.alternative_positions:
                    for alt_x, alt_y, alt_score in decision.alternative_positions[:3]:
                        alt_x, alt_y = int(alt_x), int(alt_y)
                        if 0 <= alt_x < w and 0 <= alt_y < h:
                            cv2.circle(vis, (alt_x, alt_y), 8, (0, 165, 255), 2)
                            cv2.circle(vis, (alt_x, alt_y), 3, (0, 165, 255), -1)

            # Add legend panel
            panel_h = 100
            panel_y = h - panel_h - 10
            overlay = vis.copy()
            cv2.rectangle(overlay, (10, panel_y), (280, h - 10), (0, 0, 0), -1)
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            # Draw legend
            cv2.putText(vis, "PLACEMENT DECISIONS", (20, panel_y + 20), font, 0.5, (255, 255, 255), 1)
            y_leg = panel_y + 40
            for decision_type, color in decision_colors.items():
                cv2.circle(vis, (30, y_leg), 6, color, -1)
                cv2.putText(vis, decision_type.capitalize(), (45, y_leg + 5), font, 0.4, color, 1)
                y_leg += 20

            # Statistics
            accepted = sum(1 for d in placement_decisions if d.decision == 'accepted')
            relocated = sum(1 for d in placement_decisions if d.decision == 'relocated')
            total = len(placement_decisions)
            cv2.putText(vis, f"Total: {total} | OK: {accepted} | Moved: {relocated}",
                       (150, panel_y + 40), font, 0.35, (200, 200, 200), 1)

            return vis

        except Exception as e:
            logger.warning(f"Placement decisions visualization failed: {e}")
            return image

    def _load_objects_from_directory(self, objects_dir: str) -> Dict[str, List[dict]]:
        """Load objects from directory organized by class."""
        objects_by_class = defaultdict(list)

        if not os.path.isdir(objects_dir):
            return objects_by_class

        for class_name in os.listdir(objects_dir):
            class_dir = os.path.join(objects_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(class_dir, file_name)
                    objects_by_class[class_name].append({
                        'path': file_path,
                        'name': file_name,
                    })

        return objects_by_class

    def _generate_coco_json(
        self,
        annotations_data: List[dict],
        classes: List[str],
        output_path: str,
    ):
        """Generate COCO format JSON from annotations."""
        category_map = {cls: idx for idx, cls in enumerate(sorted(classes))}

        images = []
        annotations = []
        ann_id = 1

        for img_id, img_data in enumerate(annotations_data):
            images.append({
                "id": img_id,
                "file_name": img_data['image_name'],
                "width": img_data['width'],
                "height": img_data['height'],
            })

            for ann in img_data['annotations']:
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_map.get(ann.class_name, -1),
                    "bbox": [ann.x, ann.y, ann.width, ann.height],
                    "area": ann.area,
                    "segmentation": [],
                    "iscrowd": 0,
                })
                ann_id += 1

        categories = [
            {"id": cat_id, "name": name, "supercategory": "none"}
            for name, cat_id in category_map.items()
        ]

        coco_data = {
            "info": {"description": "Synthetic Dataset"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logger.info(f"COCO JSON saved to {output_path}")

    # =========================================================================
    # Effect Methods
    # =========================================================================

    def _apply_color_correction(
        self,
        obj: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
        intensity: float = 0.15,
    ) -> np.ndarray:
        """
        Transfer color statistics from background to object using Reinhard color transfer.

        Uses proper LAB color space transfer that preserves object texture/detail
        while adapting colors to match the background environment.

        Args:
            obj: Object image (BGR)
            roi: Background region of interest (BGR)
            mask: Object mask (unused but kept for interface compatibility)
            intensity: Blend intensity (0=no change, 1=full transfer).
                      Low values (0.1-0.2) recommended for synthetic data.
        """
        try:
            # Convert to LAB color space
            obj_lab = cv2.cvtColor(obj, cv2.COLOR_BGR2LAB).astype(np.float32)
            roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Calculate statistics for each channel
            obj_mean = obj_lab.mean(axis=(0, 1))
            obj_std = obj_lab.std(axis=(0, 1)) + 1e-6
            roi_mean = roi_lab.mean(axis=(0, 1))
            roi_std = roi_lab.std(axis=(0, 1)) + 1e-6

            # Reinhard color transfer: normalize then apply target statistics
            # Step 1: Normalize object (center and scale)
            normalized = (obj_lab - obj_mean) / obj_std

            # Step 2: Apply background statistics (full transfer)
            transferred = normalized * roi_std + roi_mean

            # Step 3: Blend with original based on intensity
            # Low intensity = preserve object colors, high = adapt to background
            result = obj_lab * (1 - intensity) + transferred * intensity

            # Preserve more of the original luminance (L channel) to maintain object details
            # Only transfer 50% of L channel change to keep object recognizable
            l_blend = 0.5 * intensity
            result[:, :, 0] = obj_lab[:, :, 0] * (1 - l_blend) + transferred[:, :, 0] * l_blend

            result = np.clip(result, 0, 255).astype(np.uint8)
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        except Exception as e:
            logger.warning(f"Color correction failed: {e}")
            return obj

    def _apply_blur_matching(
        self,
        obj: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
        strength: float = 1.0,
    ) -> np.ndarray:
        """
        Subtly match blur level of object to background.

        Uses conservative blur to avoid destroying object details.
        Objects need to remain recognizable for training data.
        """
        try:
            # Estimate blur from Laplacian variance (higher = sharper)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bg_sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

            gray_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
            obj_sharpness = cv2.Laplacian(gray_obj, cv2.CV_64F).var()

            # Only apply blur if object is SIGNIFICANTLY sharper than background
            # Higher threshold (3x) to preserve object details
            if obj_sharpness > bg_sharpness * 3.0 and bg_sharpness > 10:
                # Calculate blur ratio with logarithmic dampening
                # This prevents excessive blur even with large differences
                ratio = obj_sharpness / max(bg_sharpness, 100)
                dampened_ratio = np.log2(max(ratio, 1)) + 1  # Log dampening

                # Very conservative blur amount (max 5px kernel)
                blur_amount = int(dampened_ratio * strength * 0.5)
                blur_amount = max(3, min(blur_amount, 5))
                if blur_amount % 2 == 0:
                    blur_amount += 1

                # Only apply if blur would be noticeable
                if blur_amount >= 3:
                    logger.debug(f"Blur matching: obj_sharp={obj_sharpness:.0f}, bg_sharp={bg_sharpness:.0f}, blur={blur_amount}")
                    return cv2.GaussianBlur(obj, (blur_amount, blur_amount), 0)

            return obj

        except Exception as e:
            logger.warning(f"Blur matching failed: {e}")
            return obj

    def _apply_lighting_effect(
        self,
        obj: np.ndarray,
        light_type: str = "ambient",
        intensity: float = 0.5,
    ) -> np.ndarray:
        """Apply lighting effect to object."""
        try:
            h, w = obj.shape[:2]

            if light_type == "spotlight":
                Y, X = np.ogrid[:h, :w]
                cx, cy = w // 2, h // 2
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                mask = 1 - (dist / (max(h, w) * 0.7))
                mask = np.clip(mask, 0, 1)
            elif light_type == "gradient":
                mask = np.linspace(1, 0.5, h)[:, np.newaxis]
                mask = np.repeat(mask, w, axis=1)
            else:  # ambient
                mask = np.ones((h, w))

            # Apply lighting
            mask = mask[:, :, np.newaxis]
            adjusted = obj.astype(np.float32)
            adjusted = adjusted * (1 - intensity) + adjusted * mask * intensity * 1.2
            return np.clip(adjusted, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Lighting effect failed: {e}")
            return obj

    def _apply_motion_blur(
        self,
        obj: np.ndarray,
        kernel_size: int = 15,
    ) -> np.ndarray:
        """Apply motion blur to object."""
        try:
            angle = random.uniform(0, 360)
            k = np.zeros((kernel_size, kernel_size))
            k[kernel_size // 2, :] = 1
            M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
            k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
            k = k / k.sum()
            return cv2.filter2D(obj, -1, k)
        except Exception as e:
            logger.warning(f"Motion blur failed: {e}")
            return obj

    def _apply_shadow(
        self,
        bg: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        lighting_info: Optional[LightingInfo],
        opacity: float = 0.12,
        blur_size: int = 31,
        depth_map: Optional[np.ndarray] = None,
        position: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Apply subtle, realistic depth-aware shadow from object mask.

        Underwater shadows are naturally very soft and diffuse due to:
        - Light scattering in water
        - Multiple light sources (surface + ambient)
        - Reduced contrast at depth

        Args:
            bg: Background image
            mask: Object binary mask
            bbox: Bounding box (x, y, w, h)
            lighting_info: Light source information
            opacity: Base shadow opacity (kept low for realism)
            blur_size: Base blur kernel size (larger for soft shadows)
            depth_map: Optional depth map for depth-aware shadows
            position: Object position (x, y)
        """
        try:
            x, y, w, h = bbox
            bg_h, bg_w = bg.shape[:2]

            # =================================================================
            # Step 1: Calculate depth-based shadow parameters
            # =================================================================
            depth_value = 0.5  # Default mid-depth

            if depth_map is not None and position is not None:
                depth_normalized = depth_map.astype(np.float32)
                if depth_normalized.max() > 1:
                    depth_normalized = depth_normalized / depth_normalized.max()

                px, py = position
                px = max(0, min(px, bg_w - 1))
                py = max(0, min(py, bg_h - 1))
                depth_value = depth_normalized[py, px]

            # Underwater shadow physics:
            # - Near objects: slightly more defined shadows
            # - Mid/Far objects: very diffuse, almost invisible shadows
            # - All shadows are soft due to light scattering

            if depth_value < 0.3:
                # Near: slightly more visible but still soft
                depth_opacity_factor = 1.0
                depth_blur_factor = 0.8
                depth_offset_factor = 1.0
            elif depth_value < 0.6:
                # Mid: very soft shadows
                t = (depth_value - 0.3) / 0.3
                depth_opacity_factor = 1.0 - t * 0.5  # 1.0 -> 0.5
                depth_blur_factor = 0.8 + t * 0.4     # 0.8 -> 1.2
                depth_offset_factor = 1.0 - t * 0.5  # 1.0 -> 0.5
            else:
                # Far: almost no shadow (light too diffuse)
                depth_opacity_factor = 0.2
                depth_blur_factor = 1.5
                depth_offset_factor = 0.2

            # Adjust parameters based on depth
            adjusted_opacity = opacity * depth_opacity_factor
            adjusted_blur = int(blur_size * depth_blur_factor)
            if adjusted_blur % 2 == 0:
                adjusted_blur += 1
            adjusted_blur = max(5, min(adjusted_blur, 61))

            # =================================================================
            # Step 2: Determine shadow direction and offset (very subtle)
            # =================================================================
            # Base offset is small - underwater shadows are close to objects
            base_offset = 5 * depth_offset_factor

            # Check if object is near bottom (potential seafloor contact)
            relative_y = y / bg_h
            is_near_bottom = relative_y > 0.75

            if is_near_bottom:
                # Contact shadow: directly underneath, minimal offset
                offset_x = int(base_offset * 0.2)
                offset_y = int(base_offset * 0.3)
                # Contact shadows slightly more visible but still subtle
                adjusted_opacity = min(adjusted_opacity * 1.3, 0.18)
                adjusted_blur = max(adjusted_blur, 15)  # Keep soft
            elif lighting_info and lighting_info.light_sources:
                # Use lighting info for shadow direction
                ls = lighting_info.light_sources[0]
                azimuth = ls.position[0]
                offset_x = int(-np.cos(np.radians(azimuth)) * base_offset)
                offset_y = int(-np.sin(np.radians(azimuth)) * base_offset)
            else:
                # Default: light from above (typical underwater)
                offset_x = int(base_offset * 0.3)
                offset_y = int(base_offset * 0.8)

            # =================================================================
            # Step 3: Create and position shadow mask
            # =================================================================
            shadow_mask = np.zeros(bg.shape[:2], dtype=np.float32)

            sy = max(0, y + offset_y)
            sx = max(0, x + offset_x)
            ey = min(bg_h, sy + mask.shape[0])
            ex = min(bg_w, sx + mask.shape[1])

            mask_h = ey - sy
            mask_w = ex - sx

            if mask_h > 0 and mask_w > 0:
                shadow_mask[sy:ey, sx:ex] = mask[:mask_h, :mask_w].astype(np.float32) / 255.0

            # =================================================================
            # Step 4: Apply multi-pass blur for natural falloff
            # =================================================================
            # First pass: main blur
            shadow_mask = cv2.GaussianBlur(shadow_mask, (adjusted_blur, adjusted_blur), 0)

            # Second pass: additional softening for natural gradient
            extra_blur = adjusted_blur + 10
            if extra_blur % 2 == 0:
                extra_blur += 1
            shadow_mask = cv2.GaussianBlur(shadow_mask, (extra_blur, extra_blur), 0)

            # Apply opacity with soft falloff at edges
            shadow_mask = shadow_mask * adjusted_opacity

            # Feather edges for more natural look
            shadow_mask = np.power(shadow_mask, 0.8)

            # =================================================================
            # Step 5: Apply shadow as subtle darkening
            # =================================================================
            # Simple darkening without color tint (more natural underwater)
            shadow_3ch = shadow_mask[:, :, np.newaxis]

            # Apply shadow as multiplicative darkening
            result = bg.astype(np.float32) * (1.0 - shadow_3ch * 0.7)

            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Shadow application failed: {e}")
            return bg

    def _apply_caustics(
        self,
        image: np.ndarray,
        intensity: float = 0.15,
    ) -> np.ndarray:
        """Apply underwater caustics effect."""
        try:
            h, w = image.shape[:2]
            cache_key = (w, h)

            if cache_key not in self._caustics_cache:
                # Generate caustics pattern
                x = np.linspace(0, 4 * np.pi, w)
                y = np.linspace(0, 4 * np.pi, h)
                X, Y = np.meshgrid(x, y)

                caustics = np.sin(X + np.random.uniform(0, 2*np.pi))
                caustics += np.sin(Y + np.random.uniform(0, 2*np.pi))
                caustics += np.sin((X + Y) / 2 + np.random.uniform(0, 2*np.pi))
                caustics = (caustics - caustics.min()) / (caustics.max() - caustics.min())

                self._caustics_cache[cache_key] = caustics

            caustics = self._caustics_cache[cache_key]
            caustics_3ch = caustics[:, :, np.newaxis]

            result = image.astype(np.float32)
            result = result * (1 + caustics_3ch * intensity)
            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Caustics application failed: {e}")
            return image

    def _apply_underwater_effect(
        self,
        image: np.ndarray,
        intensity: float = 0.25,
        water_color: Tuple[int, int, int] = (120, 80, 20),
    ) -> np.ndarray:
        """Apply underwater color cast."""
        try:
            overlay = np.full_like(image, water_color, dtype=np.uint8)
            result = cv2.addWeighted(image, 1 - intensity, overlay, intensity, 0)
            return result
        except Exception as e:
            logger.warning(f"Underwater effect failed: {e}")
            return image

    def _apply_poisson_blend(
        self,
        obj: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Apply Poisson seamless blending."""
        try:
            if cv2.countNonZero(mask) == 0:
                return roi

            center = (roi.shape[1] // 2, roi.shape[0] // 2)
            result = cv2.seamlessClone(obj, roi, mask, center, cv2.NORMAL_CLONE)
            return result
        except Exception as e:
            logger.warning(f"Poisson blending failed: {e}")
            return roi

    def _apply_edge_smoothing(
        self,
        blended: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
        feather: int = 3,
    ) -> np.ndarray:
        """
        Apply professional edge feathering for seamless blending.

        Uses multi-pass approach:
        1. Distance-based edge detection
        2. Gradual alpha falloff at edges
        3. Color blending in transition zone

        Args:
            blended: Object already blended onto background
            roi: Original background region
            mask: Binary mask of object
            feather: Feather radius in pixels

        Returns:
            Smoothly blended result
        """
        try:
            h, w = mask.shape[:2]
            if h < 3 or w < 3:
                return blended

            # Ensure mask is binary
            mask_bin = (mask > 127).astype(np.uint8) * 255

            # Calculate distance from edge (inward)
            dist_inside = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)

            # Feather zone: pixels within 'feather' distance from edge
            feather_px = max(feather, 2)

            # Create smooth alpha gradient at edges
            # Core (far from edge): 100% object
            # Edge zone: gradual transition
            alpha = np.clip(dist_inside / feather_px, 0, 1)

            # Apply smoothing to alpha for natural falloff
            alpha = cv2.GaussianBlur(alpha, (5, 5), 1.5)

            # Ensure core stays fully opaque
            core_mask = dist_inside > feather_px * 1.5
            alpha[core_mask] = 1.0

            # Expand to 3 channels
            alpha_3ch = alpha[:, :, np.newaxis]

            # Blend: use alpha to mix between blended object and original background
            result = (
                blended.astype(np.float32) * alpha_3ch +
                roi.astype(np.float32) * (1.0 - alpha_3ch)
            )

            # Additional color harmonization at the very edge
            # This prevents color discontinuities
            edge_zone = (dist_inside > 0) & (dist_inside < feather_px)
            if np.any(edge_zone):
                # Get average colors at edge
                edge_3ch = edge_zone[:, :, np.newaxis]

                # Slight color blend at transition zone
                edge_blend = (
                    result * 0.85 +
                    cv2.GaussianBlur(result.astype(np.float32), (3, 3), 0.5) * 0.15
                )
                result = np.where(edge_3ch, edge_blend, result)

            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.warning(f"Edge smoothing failed: {e}")
            return blended

    def _create_heuristic_scene_analysis(self, image: np.ndarray):
        """
        Create a basic scene analysis using heuristics when segmentation service fails.

        This is a fallback that analyzes basic image properties to estimate scene regions.
        It's not as accurate as SAM3 but provides reasonable defaults.

        Args:
            image: BGR image array

        Returns:
            SceneAnalysis-like object with basic scene information
        """
        from dataclasses import dataclass
        from typing import Dict, Optional
        import numpy as np

        @dataclass
        class HeuristicSceneAnalysis:
            dominant_region: str
            region_scores: Dict[str, float]
            depth_zones: Dict[str, tuple]
            scene_brightness: float
            water_clarity: str
            color_temperature: str
            region_map: Optional[np.ndarray] = None

        try:
            h, w = image.shape[:2]

            # Analyze color distribution for underwater scene classification
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue, sat, val = cv2.split(hsv)

            # Calculate brightness
            brightness = val.mean() / 255.0

            # Analyze color temperature from BGR
            b, g, r = cv2.split(image)
            blue_dominance = b.mean() / max(r.mean(), 1)

            # Determine water clarity based on saturation and contrast
            contrast = val.std() / 127.0  # Normalize
            avg_saturation = sat.mean() / 255.0

            if contrast > 0.3 and avg_saturation > 0.3:
                water_clarity = "clear"
            elif contrast > 0.15:
                water_clarity = "moderate"
            else:
                water_clarity = "murky"

            # Color temperature estimation
            if blue_dominance > 1.3:
                color_temperature = "cool"
            elif blue_dominance < 0.8:
                color_temperature = "warm"
            else:
                color_temperature = "neutral"

            # Analyze vertical gradient to detect seafloor vs open water
            # Darker bottom typically indicates seafloor
            top_third = val[:h//3, :].mean()
            bottom_third = val[2*h//3:, :].mean()

            # Estimate region scores based on image characteristics
            region_scores = {
                "open_water": 0.0,
                "seafloor": 0.0,
                "surface": 0.0,
                "vegetation": 0.0,
                "rocky": 0.0,
                "sandy": 0.0,
                "murky": 0.0,
            }

            # Score based on vertical gradient (darker bottom = seafloor likely)
            if bottom_third < top_third * 0.7:
                region_scores["seafloor"] += 0.4
                region_scores["open_water"] += 0.3
            elif top_third > bottom_third * 1.2:
                region_scores["surface"] += 0.3
                region_scores["open_water"] += 0.4
            else:
                region_scores["open_water"] += 0.5

            # Blue dominance suggests open water
            if blue_dominance > 1.2:
                region_scores["open_water"] += 0.3

            # Green tint suggests vegetation
            green_ratio = g.mean() / max((r.mean() + b.mean()) / 2, 1)
            if green_ratio > 1.1:
                region_scores["vegetation"] += 0.2

            # Low saturation/contrast suggests murky water
            if water_clarity == "murky":
                region_scores["murky"] += 0.3
                region_scores["open_water"] -= 0.1

            # Brown tones suggest sandy bottom
            if r.mean() > b.mean() and g.mean() > b.mean():
                brown_score = (r.mean() + g.mean()) / (2 * max(b.mean(), 1))
                if brown_score > 1.2:
                    region_scores["sandy"] += 0.25

            # Normalize scores
            total = sum(region_scores.values())
            if total > 0:
                region_scores = {k: max(0, v/total) for k, v in region_scores.items()}
            else:
                region_scores["open_water"] = 1.0

            # Determine dominant region
            dominant_region = max(region_scores.items(), key=lambda x: x[1])[0]

            # Create simple depth zones based on image thirds
            depth_zones = {
                "near": (0.0, 0.33),
                "mid": (0.33, 0.66),
                "far": (0.66, 1.0),
            }

            logger.info(f"Heuristic scene analysis: dominant={dominant_region}, brightness={brightness:.2f}, clarity={water_clarity}")

            return HeuristicSceneAnalysis(
                dominant_region=dominant_region,
                region_scores=region_scores,
                depth_zones=depth_zones,
                scene_brightness=brightness,
                water_clarity=water_clarity,
                color_temperature=color_temperature,
                region_map=None,  # Heuristic doesn't provide pixel-level map
            )

        except Exception as e:
            logger.error(f"Heuristic scene analysis failed: {e}")
            # Ultimate fallback - return neutral defaults
            return HeuristicSceneAnalysis(
                dominant_region="open_water",
                region_scores={"open_water": 0.6, "seafloor": 0.2, "surface": 0.2},
                depth_zones={"mid": (0.3, 0.7)},
                scene_brightness=0.5,
                water_clarity="moderate",
                color_temperature="neutral",
                region_map=None,
            )

    def _create_multi_object_summary(
        self,
        background: np.ndarray,
        composite: np.ndarray,
        object_debug_results: List[dict],
        scene_analysis,
        depth_map: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Create a consolidated visualization showing all objects and their placement decisions.

        Layout (improved to prevent overlap):
        ┌─────────────────────────────────────────────┐
        │  TITLE BAR                                  │
        ├─────────────────────┬───────────────────────┤
        │                     │  OBJECT LIST          │
        │  COMPOSITE IMAGE    │  (scrollable info)    │
        │  (with bboxes)      │                       │
        │                     │                       │
        ├─────────────────────┴───────────────────────┤
        │  THUMBNAILS (horizontal strip)              │
        ├─────────────────────┬───────────────────────┤
        │  BACKGROUND         │  DEPTH MAP            │
        └─────────────────────┴───────────────────────┘
        """
        try:
            h, w = background.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_small = cv2.FONT_HERSHEY_SIMPLEX

            num_objects = len(object_debug_results)

            # Fixed dimensions for cleaner layout
            info_panel_w = 280
            thumb_h = 100
            footer_h = 150
            title_h = 35

            # Color palette for objects
            colors = [
                (0, 255, 0),    # Green
                (255, 165, 0),  # Orange
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 255, 0),  # Yellow
                (255, 0, 0),    # Blue
                (0, 0, 255),    # Red
                (128, 0, 128),  # Purple
            ]

            # =====================================================
            # 1. Create title bar
            # =====================================================
            total_w = w + info_panel_w
            title_bar = np.zeros((title_h, total_w, 3), dtype=np.uint8)
            title_bar[:] = (50, 50, 50)
            cv2.putText(title_bar, "MULTI-OBJECT PIPELINE DEBUG SUMMARY",
                       (10, 25), font, 0.7, (255, 255, 255), 2)
            cv2.putText(title_bar, f"Objects: {num_objects}",
                       (total_w - 150, 25), font, 0.5, (200, 200, 200), 1)

            # =====================================================
            # 2. Create composite panel with bounding boxes
            # =====================================================
            composite_panel = composite.copy()

            # Draw bounding boxes with smart label placement
            label_positions = []  # Track label positions to avoid overlap

            for i, obj_info in enumerate(object_debug_results):
                color = colors[i % len(colors)]
                bbox = obj_info['bbox']
                class_name = obj_info['class_name'][:15]  # Truncate long names
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Draw bounding box
                cv2.rectangle(composite_panel, (x1, y1), (x2, y2), color, 2)

                # Smart label placement (avoid overlap)
                label = f"{i}:{class_name}"
                (label_w, label_h), _ = cv2.getTextSize(label, font_small, 0.4, 1)

                # Try positions: above, below, inside-top, inside-bottom
                label_x = x1
                label_y = y1 - 8
                if label_y < label_h + 5:
                    label_y = y1 + label_h + 5  # Below top edge

                # Check for overlap with existing labels
                for (lx, ly, lw, lh) in label_positions:
                    if abs(label_x - lx) < lw and abs(label_y - ly) < lh:
                        label_y = y2 + label_h + 5  # Move to bottom
                        break

                label_positions.append((label_x, label_y, label_w + 10, label_h + 5))

                # Draw label background
                cv2.rectangle(composite_panel,
                             (label_x, label_y - label_h - 3),
                             (label_x + label_w + 6, label_y + 3),
                             color, -1)
                cv2.putText(composite_panel, label, (label_x + 3, label_y),
                           font_small, 0.4, (0, 0, 0), 1)

            # =====================================================
            # 3. Create info panel (right side)
            # =====================================================
            info_panel = np.zeros((h, info_panel_w, 3), dtype=np.uint8)
            info_panel[:] = (35, 35, 35)

            # Title
            cv2.putText(info_panel, "PLACED OBJECTS", (10, 25),
                       font, 0.5, (255, 255, 255), 1)
            cv2.line(info_panel, (10, 35), (info_panel_w - 10, 35), (80, 80, 80), 1)

            # Object list with proper spacing
            y_pos = 55
            max_objects_display = (h - 150) // 50  # Calculate how many fit

            for i, obj_info in enumerate(object_debug_results[:max_objects_display]):
                color = colors[i % len(colors)]
                class_name = obj_info['class_name'][:18]
                bbox = obj_info['bbox']
                x1, y1, x2, y2 = [int(v) for v in bbox]

                # Object name with color indicator
                cv2.circle(info_panel, (15, y_pos - 4), 5, color, -1)
                cv2.putText(info_panel, f"{i}. {class_name}",
                           (25, y_pos), font_small, 0.4, (220, 220, 220), 1)
                y_pos += 18

                # Size info
                cv2.putText(info_panel, f"   Size: {x2-x1}x{y2-y1}px",
                           (25, y_pos), font_small, 0.32, (150, 150, 150), 1)
                y_pos += 15

                # Placement decision
                placement = obj_info.get('placement_decision')
                if placement:
                    decision_color = (100, 200, 100) if placement.decision == 'accepted' else (200, 200, 100)
                    score = getattr(placement, 'compatibility_score', 0)
                    cv2.putText(info_panel, f"   {placement.decision} ({score:.0%})",
                               (25, y_pos), font_small, 0.32, decision_color, 1)
                    y_pos += 15

                y_pos += 8  # Spacing between objects

            # Show "more" indicator if truncated
            if num_objects > max_objects_display:
                cv2.putText(info_panel, f"   ... +{num_objects - max_objects_display} more",
                           (25, y_pos), font_small, 0.35, (120, 120, 120), 1)

            # Scene analysis at bottom
            if scene_analysis is not None:
                y_pos = h - 80
                cv2.line(info_panel, (10, y_pos - 10), (info_panel_w - 10, y_pos - 10), (80, 80, 80), 1)
                cv2.putText(info_panel, "SCENE INFO", (10, y_pos),
                           font_small, 0.4, (180, 180, 180), 1)
                y_pos += 20
                dominant = scene_analysis.dominant_region
                dominant_str = dominant.value if hasattr(dominant, 'value') else str(dominant)
                cv2.putText(info_panel, f"Region: {dominant_str}",
                           (15, y_pos), font_small, 0.35, (140, 140, 140), 1)
                y_pos += 16
                cv2.putText(info_panel, f"Clarity: {scene_analysis.water_clarity}",
                           (15, y_pos), font_small, 0.35, (140, 140, 140), 1)

            # =====================================================
            # 4. Combine composite + info panel
            # =====================================================
            main_panel = np.hstack([composite_panel, info_panel])

            # =====================================================
            # 5. Create thumbnail strip
            # =====================================================
            thumb_strip = np.zeros((thumb_h + 30, total_w, 3), dtype=np.uint8)
            thumb_strip[:] = (45, 45, 45)
            cv2.putText(thumb_strip, "Object Thumbnails:", (10, 18),
                       font_small, 0.4, (180, 180, 180), 1)

            x_offset = 10
            thumb_max_w = 80
            for i, obj_info in enumerate(object_debug_results):
                if x_offset + thumb_max_w + 10 > total_w - 50:
                    # Show count of remaining
                    remaining = num_objects - i
                    cv2.putText(thumb_strip, f"+{remaining}",
                               (x_offset + 5, 75), font_small, 0.4, (150, 150, 150), 1)
                    break

                color = colors[i % len(colors)]
                debug_dir = obj_info.get('debug_dir')

                thumb = None
                if debug_dir:
                    orig_path = os.path.join(debug_dir, "03_object_original.png")
                    if os.path.exists(orig_path):
                        thumb_orig = cv2.imread(orig_path, cv2.IMREAD_UNCHANGED)
                        if thumb_orig is not None:
                            if len(thumb_orig.shape) == 3 and thumb_orig.shape[2] == 4:
                                alpha = thumb_orig[:, :, 3:4].astype(np.float32) / 255.0
                                thumb = (thumb_orig[:, :, :3].astype(np.float32) * alpha +
                                        60 * (1 - alpha)).astype(np.uint8)
                            else:
                                thumb = thumb_orig[:, :, :3] if len(thumb_orig.shape) == 3 else cv2.cvtColor(thumb_orig, cv2.COLOR_GRAY2BGR)

                            # Resize maintaining aspect ratio
                            th, tw = thumb.shape[:2]
                            scale = min(thumb_max_w / tw, (thumb_h - 5) / th)
                            thumb = cv2.resize(thumb, (int(tw * scale), int(th * scale)))

                if thumb is None:
                    thumb = np.full((thumb_h - 10, thumb_max_w - 10, 3), 60, dtype=np.uint8)

                th, tw = thumb.shape[:2]
                y_thumb = 28
                thumb_strip[y_thumb:y_thumb + th, x_offset:x_offset + tw] = thumb
                cv2.rectangle(thumb_strip, (x_offset - 1, y_thumb - 1),
                             (x_offset + tw + 1, y_thumb + th + 1), color, 1)

                x_offset += tw + 8

            # =====================================================
            # 6. Create footer with background + depth
            # =====================================================
            half_w = total_w // 2
            bg_small = cv2.resize(background, (half_w, footer_h))

            if depth_map is not None:
                depth_norm = (depth_map * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                depth_small = cv2.resize(depth_color, (total_w - half_w, footer_h))
            else:
                depth_small = np.zeros((footer_h, total_w - half_w, 3), dtype=np.uint8)
                cv2.putText(depth_small, "No Depth", (10, footer_h // 2),
                           font_small, 0.5, (100, 100, 100), 1)

            footer = np.hstack([bg_small, depth_small])
            cv2.putText(footer, "Background", (5, 15), font_small, 0.4, (255, 255, 255), 1)
            cv2.putText(footer, "Depth Map", (half_w + 5, 15), font_small, 0.4, (255, 255, 255), 1)

            # =====================================================
            # 7. Stack all panels vertically
            # =====================================================
            final = np.vstack([title_bar, main_panel, thumb_strip, footer])

            return final

        except Exception as e:
            logger.warning(f"Multi-object summary visualization failed: {e}")
            return composite
