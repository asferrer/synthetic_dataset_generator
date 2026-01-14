"""
Mask Utilities
===============
IoU calculations and instance matching algorithms for deduplication.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class MatchingStrategy(str, Enum):
    """Strategy for matching SAM3 instances to annotations"""
    BBOX_IOU = "bbox_iou"              # Greedy matching by bbox IoU (fast, default)
    MASK_IOU = "mask_iou"              # Greedy matching by mask IoU (accurate, slower)
    CENTER_DISTANCE = "center_distance" # Match by bbox center distance


@dataclass
class SAM3Instance:
    """Single instance from SAM3 segmentation"""
    mask: np.ndarray       # Binary mask (H, W), 0 or 255
    bbox: List[float]      # [x, y, w, h]
    score: float           # Confidence score from SAM3
    assigned: bool = False # Whether assigned to annotation


@dataclass
class ExtractedMask:
    """Record of extracted mask in registry"""
    mask: np.ndarray       # Binary mask for IoU comparison
    bbox: List[float]      # [x, y, w, h] for fast pre-filtering
    annotation_id: int     # Source annotation
    category_id: int       # Category ID
    category_name: str     # Category name
    method: str            # ExtractionMethod value


# =============================================================================
# BBOX IOU CALCULATION
# =============================================================================

def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.

    Args:
        bbox1: First bbox in [x, y, width, height] format
        bbox2: Second bbox in [x, y, width, height] format

    Returns:
        IoU value between 0.0 and 1.0

    Example:
        >>> bbox1 = [10, 10, 50, 50]  # x=10, y=10, w=50, h=50
        >>> bbox2 = [10, 10, 50, 50]  # Identical
        >>> calculate_bbox_iou(bbox1, bbox2)
        1.0

        >>> bbox1 = [0, 0, 10, 10]
        >>> bbox2 = [20, 20, 10, 10]  # No overlap
        >>> calculate_bbox_iou(bbox1, bbox2)
        0.0
    """
    # Validate inputs
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0

    # Extract coordinates
    x1, y1, w1, h1 = bbox1[:4]
    x2, y2, w2, h2 = bbox2[:4]

    # Check for invalid dimensions
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    # Convert to [x1, y1, x2, y2] format
    box1_x1, box1_y1 = x1, y1
    box1_x2, box1_y2 = x1 + w1, y1 + h1

    box2_x1, box2_y1 = x2, y2
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    # Calculate intersection area
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    # Check if there is an intersection
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area <= 0:
        return 0.0

    # Calculate IoU
    iou = inter_area / union_area

    return float(iou)


# =============================================================================
# MASK IOU CALCULATION
# =============================================================================

def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU (Intersection over Union) between two binary masks.

    Args:
        mask1: First binary mask (H, W) with values 0 or 255
        mask2: Second binary mask (H, W) with values 0 or 255

    Returns:
        IoU value between 0.0 and 1.0

    Example:
        >>> mask1 = np.zeros((100, 100), dtype=np.uint8)
        >>> mask1[20:40, 20:40] = 255
        >>> mask2 = mask1.copy()
        >>> calculate_mask_iou(mask1, mask2)
        1.0

    Notes:
        - Masks are automatically resized if dimensions don't match
        - Uses fast bbox pre-check optimization
        - Binarizes masks (threshold at 127)
    """
    # Validate inputs
    if mask1 is None or mask2 is None:
        return 0.0

    if mask1.size == 0 or mask2.size == 0:
        return 0.0

    # Ensure same dimensions (resize if needed)
    if mask1.shape != mask2.shape:
        h = max(mask1.shape[0], mask2.shape[0])
        w = max(mask1.shape[1], mask2.shape[1])

        if mask1.shape != (h, w):
            import cv2
            mask1 = cv2.resize(mask1, (w, h), interpolation=cv2.INTER_NEAREST)

        if mask2.shape != (h, w):
            import cv2
            mask2 = cv2.resize(mask2, (w, h), interpolation=cv2.INTER_NEAREST)

    # Binarize masks (threshold at 127)
    mask1_bin = mask1 > 127
    mask2_bin = mask2 > 127

    # Calculate intersection
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()

    # Calculate union
    union = np.logical_or(mask1_bin, mask2_bin).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    # Calculate IoU
    iou = float(intersection) / float(union)

    return iou


# =============================================================================
# BBOX CENTER DISTANCE
# =============================================================================

def calculate_bbox_center_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Euclidean distance between centers of two bounding boxes.

    Args:
        bbox1: First bbox in [x, y, width, height] format
        bbox2: Second bbox in [x, y, width, height] format

    Returns:
        Euclidean distance between centers

    Example:
        >>> bbox1 = [0, 0, 10, 10]  # Center at (5, 5)
        >>> bbox2 = [30, 40, 10, 10]  # Center at (35, 45)
        >>> calculate_bbox_center_distance(bbox1, bbox2)
        50.0
    """
    if len(bbox1) < 4 or len(bbox2) < 4:
        return float('inf')

    x1, y1, w1, h1 = bbox1[:4]
    x2, y2, w2, h2 = bbox2[:4]

    # Calculate centers
    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2

    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2

    # Calculate Euclidean distance
    distance = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)

    return float(distance)


# =============================================================================
# INSTANCE TO ANNOTATION MATCHING
# =============================================================================

def match_instances_to_annotations(
    instances: List[SAM3Instance],
    annotations: List[Dict],
    strategy: MatchingStrategy = MatchingStrategy.BBOX_IOU
) -> Dict[int, Optional[int]]:
    """
    Match SAM3 instances to annotations using specified strategy.

    This implements a greedy one-to-one assignment algorithm:
    1. For each annotation (sorted by bbox area, descending):
       - Find the best unused instance based on matching criteria
       - If match quality exceeds threshold, assign and mark instance as used
       - Otherwise, leave annotation unmatched (will fail later)

    Args:
        instances: List of SAM3Instance objects from segmentation
        annotations: List of COCO annotation dictionaries
        strategy: Matching strategy (BBOX_IOU, MASK_IOU, CENTER_DISTANCE)

    Returns:
        Dict mapping annotation_id -> instance_index
        - annotation_id: int (from annotation["id"])
        - instance_index: Optional[int] (index in instances list, or None if unmatched)

    Example:
        >>> instances = [
        ...     SAM3Instance(mask=..., bbox=[10,10,20,20], score=0.9, assigned=False),
        ...     SAM3Instance(mask=..., bbox=[50,50,20,20], score=0.8, assigned=False),
        ... ]
        >>> annotations = [
        ...     {"id": 1, "bbox": [12, 12, 18, 18]},  # Close to instance 0
        ...     {"id": 2, "bbox": [52, 52, 18, 18]},  # Close to instance 1
        ... ]
        >>> assignment = match_instances_to_annotations(instances, annotations)
        >>> assignment
        {1: 0, 2: 1}  # Ann 1 → Instance 0, Ann 2 → Instance 1

    Notes:
        - Annotations are sorted by bbox area (larger first) to prioritize bigger objects
        - Minimum IoU threshold for matching: 0.3 (configurable)
        - Each instance can be assigned to at most ONE annotation
        - Unmatched annotations get None (will fail during extraction)
    """
    # Result: annotation_id -> instance_index
    assignment_map: Dict[int, Optional[int]] = {}

    # Handle empty cases
    if not instances or not annotations:
        for ann in annotations:
            assignment_map[ann["id"]] = None
        return assignment_map

    # Sort annotations by bbox area (descending) - prioritize larger objects
    sorted_annotations = sorted(
        annotations,
        key=lambda a: a.get("bbox", [0, 0, 0, 0])[2] * a.get("bbox", [0, 0, 0, 0])[3],
        reverse=True
    )

    # Track which instances have been assigned
    used_instances = set()

    # Minimum thresholds for matching
    MIN_IOU_THRESHOLD = 0.3
    MAX_DISTANCE_THRESHOLD = 100.0  # pixels

    # Match each annotation
    for ann in sorted_annotations:
        ann_id = ann["id"]
        ann_bbox = ann.get("bbox", [0, 0, 0, 0])

        best_instance_idx = None
        best_score = -1.0

        # Find best matching instance
        for idx, instance in enumerate(instances):
            # Skip already assigned instances
            if idx in used_instances:
                continue

            # Calculate match score based on strategy
            if strategy == MatchingStrategy.BBOX_IOU:
                match_score = calculate_bbox_iou(ann_bbox, instance.bbox)

                # Threshold check
                if match_score < MIN_IOU_THRESHOLD:
                    continue

            elif strategy == MatchingStrategy.MASK_IOU:
                # Expensive - only used if explicitly requested
                # First check bbox IoU as pre-filter
                bbox_iou = calculate_bbox_iou(ann_bbox, instance.bbox)
                if bbox_iou < 0.1:  # Too far apart
                    continue

                match_score = calculate_mask_iou(
                    _bbox_to_mask(ann_bbox, instance.mask.shape),
                    instance.mask
                )

                if match_score < MIN_IOU_THRESHOLD:
                    continue

            elif strategy == MatchingStrategy.CENTER_DISTANCE:
                distance = calculate_bbox_center_distance(ann_bbox, instance.bbox)

                # Threshold check
                if distance > MAX_DISTANCE_THRESHOLD:
                    continue

                # Convert to score (closer = higher score)
                match_score = 1.0 / (1.0 + distance / 10.0)

            else:
                logger.warning(f"Unknown matching strategy: {strategy}, using BBOX_IOU")
                match_score = calculate_bbox_iou(ann_bbox, instance.bbox)

            # Update best match
            if match_score > best_score:
                best_score = match_score
                best_instance_idx = idx

        # Assign best match (or None if no good match found)
        if best_instance_idx is not None:
            assignment_map[ann_id] = best_instance_idx
            used_instances.add(best_instance_idx)
            instances[best_instance_idx].assigned = True

            logger.debug(
                f"Matched annotation {ann_id} to instance {best_instance_idx} "
                f"(strategy={strategy.value}, score={best_score:.3f})"
            )
        else:
            assignment_map[ann_id] = None
            logger.debug(f"No match found for annotation {ann_id} (strategy={strategy.value})")

    # Log summary
    matched_count = sum(1 for v in assignment_map.values() if v is not None)
    logger.info(
        f"Matched {matched_count}/{len(annotations)} annotations to instances "
        f"(strategy={strategy.value}, {len(instances)} instances available)"
    )

    return assignment_map


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _bbox_to_mask(bbox: List[float], shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding box to binary mask.

    Args:
        bbox: Bounding box [x, y, width, height]
        shape: Mask shape (height, width)

    Returns:
        Binary mask with bbox area set to 255
    """
    mask = np.zeros(shape, dtype=np.uint8)

    x, y, w, h = [int(v) for v in bbox[:4]]
    h_img, w_img = shape

    # Clamp to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255

    return mask


def get_bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    """
    Extract bounding box from binary mask.

    Args:
        mask: Binary mask (H, W)

    Returns:
        Bbox [x, y, width, height] or None if mask is empty
    """
    # Find non-zero pixels
    coords = np.where(mask > 0)

    if len(coords[0]) == 0 or len(coords[1]) == 0:
        return None

    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    return [x_min, y_min, width, height]
