"""
Physics Validator for Synthetic Image Composites

Validates physical plausibility:
- Gravity and buoyancy checks
- Occlusion correctness based on depth
- Material property validation
- Auto-correction of implausible placements
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhysicsViolation:
    """Detected physics violation"""
    type: str              # 'gravity', 'buoyancy', 'occlusion', 'scale'
    category: str          # Object category
    severity: float        # 0-1 (higher = worse)
    message: str           # Human-readable description
    object_id: Optional[int] = None  # Annotation ID if available

    def __str__(self):
        return f"PhysicsViolation({self.type}: {self.message}, severity={self.severity:.2f})"


@dataclass
class Position:
    """2D position"""
    x: float
    y: float


@dataclass
class Object:
    """Object representation for physics checking"""
    category: str
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    mask: np.ndarray
    center_x: int
    center_y: int
    z_order: int
    area: float


class PhysicsValidator:
    """Validates physical plausibility of synthetic compositions"""

    # Material densities (kg/m³) relative to water (1000 kg/m³)
    # Values < 1.0 float, > 1.0 sink
    MATERIAL_DENSITIES = {
        # Organic materials
        'fish': 1.05,           # Neutral buoyancy (fish have swim bladders)
        'shark': 1.08,          # Slightly negative buoyancy
        'ray': 1.06,            # Neutral to slightly negative
        'jellyfish': 1.02,      # Near neutral (mostly water)
        'coral': 1.10,          # Slightly denser than water
        'kelp': 0.95,           # Slightly buoyant
        'algae': 0.98,          # Near neutral
        'seaweed': 0.97,        # Slightly buoyant
        'starfish': 1.05,       # Neutral buoyancy
        'crab': 1.15,           # Sinks
        'lobster': 1.15,        # Sinks
        'octopus': 1.03,        # Near neutral
        'squid': 1.04,          # Near neutral

        # Plastics
        'plastic': 0.90,        # Most plastics float
        'plastic_bag': 0.92,    # Floats
        'plastic_bottle': 0.95, # Floats (if sealed with air)
        'styrofoam': 0.05,      # Very buoyant
        'foam': 0.10,           # Very buoyant

        # Metals
        'metal': 7.80,          # Steel - sinks heavily
        'aluminum': 2.70,       # Sinks
        'can': 2.70,            # Aluminum can - sinks
        'bottle_cap': 7.80,     # Steel cap - sinks
        'foil': 2.70,           # Aluminum foil - sinks

        # Glass
        'glass': 2.50,          # Sinks
        'glass_bottle': 2.50,   # Sinks
        'jar': 2.50,            # Sinks

        # Other materials
        'wood': 0.60,           # Floats
        'paper': 0.70,          # Floats (if not waterlogged)
        'cardboard': 0.75,      # Floats initially
        'cloth': 1.10,          # Sinks (when wet)
        'fabric': 1.10,         # Sinks (when wet)
        'net': 1.15,            # Fishing net - sinks
        'rope': 1.05,           # Slightly sinks
        'tire': 1.20,           # Rubber tire - sinks

        # General debris
        'debris': 1.50,         # Assumed heavy debris
        'trash': 1.20,          # General trash
        'bottle': 0.95,         # Assumed plastic
        'container': 0.90,      # Assumed plastic
    }

    def __init__(self,
                 density_threshold_float: float = 0.95,
                 density_threshold_sink: float = 1.15,
                 surface_zone: float = 0.25,
                 bottom_zone: float = 0.75):
        """
        Initialize physics validator

        Args:
            density_threshold_float: Density below which objects must float (top zone)
            density_threshold_sink: Density above which objects must sink (bottom zone)
            surface_zone: Top fraction of image (0-1) considered "surface zone"
            bottom_zone: Bottom fraction of image (0-1) considered "bottom zone"
        """
        self.density_threshold_float = density_threshold_float
        self.density_threshold_sink = density_threshold_sink
        self.surface_zone = surface_zone
        self.bottom_zone = bottom_zone

    def check_gravity(self,
                     annotations: List[Dict],
                     image_height: int,
                     scene_type: str = 'underwater') -> List[PhysicsViolation]:
        """
        Detect gravity/buoyancy violations

        Args:
            annotations: List of COCO-style annotations
            image_height: Height of image in pixels
            scene_type: 'underwater', 'surface', 'ground'

        Returns:
            List of physics violations
        """
        violations = []

        for ann in annotations:
            category = ann.get('category_name', ann.get('category_id', 'unknown'))
            bbox = ann.get('bbox', [0, 0, 0, 0])

            # Get material density
            material_type = self._classify_material(category)
            density = self.MATERIAL_DENSITIES.get(material_type, 1.0)

            # Compute vertical position (0=top, 1=bottom)
            bbox_center_y = bbox[1] + bbox[3] / 2
            relative_depth = bbox_center_y / image_height

            # Check for violations based on scene type
            if scene_type == 'underwater':
                # Heavy objects floating near surface
                if density > self.density_threshold_sink and relative_depth < self.surface_zone:
                    violations.append(PhysicsViolation(
                        type='gravity',
                        category=category,
                        severity=min(1.0, (density - 1.0) / 2.0),
                        message=f"{category} (density={density:.2f}) floating near surface",
                        object_id=ann.get('id')
                    ))

                # Light objects at seafloor
                if density < self.density_threshold_float and relative_depth > self.bottom_zone:
                    violations.append(PhysicsViolation(
                        type='buoyancy',
                        category=category,
                        severity=min(1.0, (1.0 - density) * 2.0),
                        message=f"{category} (density={density:.2f}) at seafloor",
                        object_id=ann.get('id')
                    ))

                # Extremely light objects (styrofoam) not at surface
                if density < 0.20 and relative_depth > 0.15:
                    violations.append(PhysicsViolation(
                        type='buoyancy',
                        category=category,
                        severity=0.95,
                        message=f"{category} (very buoyant) not at surface",
                        object_id=ann.get('id')
                    ))

        return violations

    def check_occlusion_correctness(self,
                                   objects: List[Object],
                                   depth_map: np.ndarray) -> List[PhysicsViolation]:
        """
        Verify that objects closer to camera occlude farther objects

        Args:
            objects: List of Object instances with masks and z_order
            depth_map: Depth map (lower values = closer)

        Returns:
            List of occlusion violations
        """
        violations = []

        # Sort objects by Z-order (rendering order)
        sorted_objects = sorted(objects, key=lambda o: o.z_order)

        for i, obj1 in enumerate(sorted_objects):
            for obj2 in sorted_objects[i+1:]:
                # Check if masks overlap
                overlap = self._compute_mask_overlap(obj1.mask, obj2.mask)

                if overlap > 0.05:  # >5% overlap
                    # Get depth values at object centers
                    depth1 = depth_map[obj1.center_y, obj1.center_x]
                    depth2 = depth_map[obj2.center_y, obj2.center_x]

                    # obj1 rendered first (behind), obj2 rendered second (in front)
                    # So depth1 should be > depth2 (farther away)
                    # If depth1 < depth2, then closer object is behind = WRONG

                    if depth1 < depth2 * 0.95:  # Allow 5% tolerance
                        severity = min(1.0, overlap * 2.0)
                        violations.append(PhysicsViolation(
                            type='occlusion',
                            category=f"{obj1.category}/{obj2.category}",
                            severity=severity,
                            message=f"Occlusion error: {obj1.category} (depth={depth1:.2f}) should be in front of {obj2.category} (depth={depth2:.2f})"
                        ))

        return violations

    def check_scale_plausibility(self,
                                annotations: List[Dict],
                                image_size: Tuple[int, int]) -> List[PhysicsViolation]:
        """
        Check if object scales are physically plausible

        Args:
            annotations: List of COCO-style annotations
            image_size: (width, height) of image

        Returns:
            List of scale violations
        """
        violations = []

        img_w, img_h = image_size

        for ann in annotations:
            category = ann.get('category_name', 'unknown')
            bbox = ann.get('bbox', [0, 0, 0, 0])
            bbox_w, bbox_h = bbox[2], bbox[3]

            # Compute bbox area as fraction of image
            bbox_area = bbox_w * bbox_h
            img_area = img_w * img_h
            area_ratio = bbox_area / img_area

            # Check for implausibly large objects (>60% of image)
            if area_ratio > 0.60:
                violations.append(PhysicsViolation(
                    type='scale',
                    category=category,
                    severity=min(1.0, area_ratio - 0.6),
                    message=f"{category} occupies {area_ratio*100:.1f}% of image (too large)",
                    object_id=ann.get('id')
                ))

            # Check for implausibly small objects (<0.1% of image)
            if area_ratio < 0.001:
                violations.append(PhysicsViolation(
                    type='scale',
                    category=category,
                    severity=0.5,
                    message=f"{category} occupies {area_ratio*100:.3f}% of image (too small)",
                    object_id=ann.get('id')
                ))

        return violations

    def auto_correct_placement(self,
                              object_category: str,
                              object_bbox: Tuple[float, float, float, float],
                              image_size: Tuple[int, int],
                              depth_map: Optional[np.ndarray] = None) -> Position:
        """
        Auto-correct object position based on physics

        Args:
            object_category: Category name
            object_bbox: (x, y, w, h)
            image_size: (width, height)
            depth_map: Optional depth map for depth-aware placement

        Returns:
            Corrected Position (x, y)
        """
        img_w, img_h = image_size
        bbox_x, bbox_y, bbox_w, bbox_h = object_bbox

        # Get material density
        material_type = self._classify_material(object_category)
        density = self.MATERIAL_DENSITIES.get(material_type, 1.0)

        # Compute target vertical position based on density
        if density > 1.5:
            # Heavy objects (metal, glass) - bottom 30%
            target_y_min = img_h * 0.70
            target_y_max = img_h * 0.95
        elif density > 1.1:
            # Slightly heavy objects - bottom 50%
            target_y_min = img_h * 0.50
            target_y_max = img_h * 0.90
        elif density < 0.9:
            # Light objects (plastics, wood) - top 40%
            target_y_min = img_h * 0.05
            target_y_max = img_h * 0.40
        elif density < 0.3:
            # Very buoyant (styrofoam) - top 15%
            target_y_min = img_h * 0.00
            target_y_max = img_h * 0.15
        else:
            # Neutral buoyancy - anywhere in mid-water
            target_y_min = img_h * 0.20
            target_y_max = img_h * 0.80

        # Sample random Y within target range
        target_y = np.random.uniform(target_y_min, target_y_max)

        # Keep X position (horizontal placement less constrained)
        target_x = bbox_x

        # Ensure bbox stays within image bounds
        target_x = np.clip(target_x, 0, img_w - bbox_w)
        target_y = np.clip(target_y, 0, img_h - bbox_h)

        logger.info(
            f"Auto-corrected {object_category} (density={density:.2f}) "
            f"from y={bbox_y:.0f} to y={target_y:.0f}"
        )

        return Position(x=float(target_x), y=float(target_y))

    def _classify_material(self, category: str) -> str:
        """
        Classify material type from category name

        Args:
            category: Object category name

        Returns:
            Material type key for MATERIAL_DENSITIES
        """
        category_lower = category.lower()

        # Direct matches
        if category_lower in self.MATERIAL_DENSITIES:
            return category_lower

        # Partial matches
        if 'fish' in category_lower:
            return 'fish'
        elif 'shark' in category_lower:
            return 'shark'
        elif 'plastic' in category_lower or 'bag' in category_lower:
            return 'plastic'
        elif 'metal' in category_lower or 'can' in category_lower:
            return 'metal'
        elif 'glass' in category_lower or 'bottle' in category_lower:
            # Distinguish plastic bottles from glass
            if 'plastic' in category_lower:
                return 'plastic_bottle'
            return 'glass'
        elif 'coral' in category_lower:
            return 'coral'
        elif 'kelp' in category_lower or 'seaweed' in category_lower:
            return 'kelp'
        elif 'wood' in category_lower:
            return 'wood'
        elif 'debris' in category_lower or 'trash' in category_lower:
            return 'debris'
        elif 'net' in category_lower:
            return 'net'
        elif 'foam' in category_lower:
            return 'foam'

        # Default to neutral buoyancy
        logger.warning(f"Unknown material for category '{category}', assuming neutral buoyancy")
        return 'fish'  # Default neutral

    def _compute_mask_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute overlap ratio between two masks

        Args:
            mask1: Binary mask
            mask2: Binary mask

        Returns:
            Overlap ratio (0-1)
        """
        # Ensure same size
        if mask1.shape != mask2.shape:
            logger.warning("Masks have different sizes, cannot compute overlap")
            return 0.0

        # Compute intersection
        intersection = np.logical_and(mask1 > 0, mask2 > 0)
        intersection_area = np.sum(intersection)

        # Compute union
        union = np.logical_or(mask1 > 0, mask2 > 0)
        union_area = np.sum(union)

        if union_area == 0:
            return 0.0

        return float(intersection_area / union_area)

    def validate_all(self,
                    annotations: List[Dict],
                    image_size: Tuple[int, int],
                    depth_map: Optional[np.ndarray] = None,
                    scene_type: str = 'underwater') -> Dict[str, List[PhysicsViolation]]:
        """
        Run all physics validation checks

        Args:
            annotations: List of COCO-style annotations
            image_size: (width, height)
            depth_map: Optional depth map
            scene_type: Scene type for context

        Returns:
            Dictionary of violation types to violation lists
        """
        img_w, img_h = image_size

        violations = {
            'gravity': [],
            'buoyancy': [],
            'occlusion': [],
            'scale': []
        }

        # 1. Gravity/buoyancy checks
        gravity_violations = self.check_gravity(annotations, img_h, scene_type)
        for v in gravity_violations:
            violations[v.type].append(v)

        # 2. Scale checks
        scale_violations = self.check_scale_plausibility(annotations, image_size)
        violations['scale'].extend(scale_violations)

        # 3. Occlusion checks (if depth map available)
        if depth_map is not None:
            # Convert annotations to Object instances
            objects = []
            for i, ann in enumerate(annotations):
                if 'segmentation' in ann and ann['segmentation']:
                    # Convert segmentation to mask
                    mask = self._segmentation_to_mask(
                        ann['segmentation'],
                        image_size
                    )

                    # Compute center
                    moments = cv2.moments(mask.astype(np.uint8))
                    if moments['m00'] > 0:
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])
                    else:
                        bbox = ann.get('bbox', [0, 0, 1, 1])
                        cx = int(bbox[0] + bbox[2] / 2)
                        cy = int(bbox[1] + bbox[3] / 2)

                    objects.append(Object(
                        category=ann.get('category_name', 'unknown'),
                        bbox=tuple(ann.get('bbox', [0, 0, 0, 0])),
                        mask=mask,
                        center_x=cx,
                        center_y=cy,
                        z_order=i,
                        area=float(ann.get('area', 0))
                    ))

            if objects:
                occlusion_violations = self.check_occlusion_correctness(objects, depth_map)
                violations['occlusion'].extend(occlusion_violations)

        return violations

    def _segmentation_to_mask(self,
                             segmentation: List,
                             image_size: Tuple[int, int]) -> np.ndarray:
        """Convert COCO segmentation to binary mask"""
        from pycocotools import mask as mask_utils

        img_w, img_h = image_size

        if isinstance(segmentation, list):
            # Polygon format
            from PIL import Image, ImageDraw
            mask = Image.new('L', (img_w, img_h), 0)
            for polygon in segmentation:
                if len(polygon) >= 6:
                    poly = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
            return np.array(mask)
        elif isinstance(segmentation, dict):
            # RLE format
            if isinstance(segmentation['counts'], list):
                rle = mask_utils.frPyObjects(segmentation, img_h, img_w)
            else:
                rle = segmentation
            return mask_utils.decode(rle)
        else:
            logger.warning("Unknown segmentation format")
            return np.zeros((img_h, img_w), dtype=np.uint8)
